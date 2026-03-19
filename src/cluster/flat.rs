//! Flat contiguous matrix for cache-friendly row access.
//!
//! Wraps a `Vec<f32>` with row-major layout and stride `d`.
//! Eliminates `Vec<Vec<f32>>` pointer indirection in hot loops.

/// A contiguous row-major matrix of f32 values.
#[allow(dead_code)]
pub(crate) struct FlatMatrix {
    data: Vec<f32>,
    n: usize,
    d: usize,
}

#[allow(dead_code)]
impl FlatMatrix {
    /// Create from `&[Vec<f32>]` by flattening into contiguous memory.
    pub(crate) fn from_vecs(vecs: &[Vec<f32>]) -> Self {
        let n = vecs.len();
        let d = vecs.first().map_or(0, |v| v.len());
        let mut data = Vec::with_capacity(n * d);
        for v in vecs {
            data.extend_from_slice(v);
        }
        Self { data, n, d }
    }

    /// Number of rows.
    #[inline]
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Number of columns (dimension).
    #[inline]
    pub(crate) fn d(&self) -> usize {
        self.d
    }

    /// Get row i as a slice.
    #[inline]
    pub(crate) fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.d..(i + 1) * self.d]
    }

    /// Raw flat data.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Compute squared L2 norms for all rows.
    pub(crate) fn row_norms_sq(&self) -> Vec<f32> {
        (0..self.n)
            .map(|i| {
                let row = self.row(i);
                row.iter().map(|&x| x * x).sum()
            })
            .collect()
    }

    /// GEMM-based batch assignment: compute argmin_j ||x_i - c_j||^2 for all i.
    ///
    /// Uses the identity: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
    /// The x.c term is a matrix multiply: X (n x d) @ C^T (d x k) = (n x k).
    ///
    /// Returns (labels, upper_bounds) where upper_bounds[i] = distance to
    /// assigned centroid.
    pub(crate) fn gemm_assign(
        &self,
        centroids: &FlatMatrix,
        x_norms: &[f32],
        c_norms: &[f32],
    ) -> (Vec<usize>, Vec<f32>) {
        let n = self.n;
        let k = centroids.n();
        let _d = self.d;

        // Compute X @ C^T via manual tiled matrix multiply.
        // For each point i, for each centroid j:
        //   dist[i][j] = x_norms[i] + c_norms[j] - 2 * dot(x_i, c_j)
        // Then argmin over j.
        let mut labels = vec![0usize; n];
        let mut upper = vec![f32::MAX; n];

        // Process in tiles for cache efficiency.
        const TILE_N: usize = 64;
        const TILE_K: usize = 64;

        for i_start in (0..n).step_by(TILE_N) {
            let i_end = (i_start + TILE_N).min(n);
            for j_start in (0..k).step_by(TILE_K) {
                let j_end = (j_start + TILE_K).min(k);

                for i in i_start..i_end {
                    let xi = self.row(i);
                    let xn = x_norms[i];
                    for (j, &cn) in c_norms[j_start..j_end].iter().enumerate() {
                        let j = j + j_start;
                        let cj = centroids.row(j);
                        let dot: f32 = xi.iter().zip(cj.iter()).map(|(&a, &b)| a * b).sum();
                        let dist = xn + cn - 2.0 * dot;
                        if dist < upper[i] {
                            upper[i] = dist;
                            labels[i] = j;
                        }
                    }
                }
            }
        }

        (labels, upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vecs_round_trip() {
        let vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let flat = FlatMatrix::from_vecs(&vecs);
        assert_eq!(flat.n(), 3);
        assert_eq!(flat.d(), 2);
        assert_eq!(flat.row(0), &[1.0, 2.0]);
        assert_eq!(flat.row(2), &[5.0, 6.0]);
    }

    #[test]
    fn gemm_assign_correctness() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let centroids = vec![vec![0.05, 0.05], vec![10.05, 10.05]];
        let flat_data = FlatMatrix::from_vecs(&data);
        let flat_cent = FlatMatrix::from_vecs(&centroids);
        let x_norms = flat_data.row_norms_sq();
        let c_norms = flat_cent.row_norms_sq();

        let (labels, _) = flat_data.gemm_assign(&flat_cent, &x_norms, &c_norms);
        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 0);
        assert_eq!(labels[2], 1);
        assert_eq!(labels[3], 1);
    }

    #[test]
    fn row_norms() {
        let vecs = vec![vec![3.0, 4.0]];
        let flat = FlatMatrix::from_vecs(&vecs);
        let norms = flat.row_norms_sq();
        assert!((norms[0] - 25.0).abs() < 1e-6);
    }
}
