//! [`DataRef`] trait and contiguous matrix layouts.
//!
//! [`DataRef`] abstracts over input data so algorithms accept both
//! `Vec<Vec<f32>>` and [`FlatRef`] (zero-copy flat buffer).

/// Trait for read-only access to a 2D dataset of `f32` rows.
///
/// Implemented for `[Vec<f32>]` (so `&Vec<Vec<f32>>` works via auto-deref)
/// and for [`FlatRef`] (zero-copy flat buffer input).
pub trait DataRef: Sync {
    /// Number of rows (data points).
    fn n(&self) -> usize;
    /// Number of columns (dimensionality).
    fn d(&self) -> usize;
    /// Access row `i` as a contiguous `f32` slice.
    fn row(&self, i: usize) -> &[f32];
}

impl DataRef for Vec<Vec<f32>> {
    #[inline]
    fn n(&self) -> usize {
        self.len()
    }
    #[inline]
    fn d(&self) -> usize {
        self.first().map_or(0, |v| v.len())
    }
    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        &self[i]
    }
}

impl DataRef for [Vec<f32>] {
    #[inline]
    fn n(&self) -> usize {
        self.len()
    }
    #[inline]
    fn d(&self) -> usize {
        self.first().map_or(0, |v| v.len())
    }
    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        &self[i]
    }
}

impl<const N: usize> DataRef for [Vec<f32>; N] {
    #[inline]
    fn n(&self) -> usize {
        self.len()
    }
    #[inline]
    fn d(&self) -> usize {
        self.first().map_or(0, |v| v.len())
    }
    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        &self[i]
    }
}

/// Zero-copy view into a contiguous row-major `f32` buffer.
pub struct FlatRef<'a> {
    data: &'a [f32],
    n: usize,
    d: usize,
}

impl<'a> FlatRef<'a> {
    /// Create a new `FlatRef`.
    /// Panics if `data.len() != n * d`.
    pub fn new(data: &'a [f32], n: usize, d: usize) -> Self {
        assert_eq!(
            data.len(),
            n * d,
            "FlatRef: data.len() ({}) != n*d ({}*{})",
            data.len(),
            n,
            d
        );
        Self { data, n, d }
    }
}

impl DataRef for FlatRef<'_> {
    #[inline]
    fn n(&self) -> usize {
        self.n
    }
    #[inline]
    fn d(&self) -> usize {
        self.d
    }
    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.d..(i + 1) * self.d]
    }
}

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

    /// Create from a `DataRef` by copying into contiguous memory.
    pub(crate) fn from_data(data: &(impl DataRef + ?Sized)) -> Self {
        let n = data.n();
        let d = data.d();
        let mut buf = Vec::with_capacity(n * d);
        for i in 0..n {
            buf.extend_from_slice(data.row(i));
        }
        Self { data: buf, n, d }
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

    /// BLAS GEMM-based assignment using matrixmultiply crate.
    ///
    /// Computes X @ C^T via optimized SGEMM, then finds argmin per row.
    /// This is the FAISS/scikit-learn approach: 2-5x faster than per-point
    /// distance loops for large k due to micro-kernel SIMD optimization.
    #[cfg(feature = "blas")]
    #[allow(unsafe_code)]
    pub(crate) fn blas_assign(
        &self,
        centroids: &FlatMatrix,
        x_norms: &[f32],
        c_norms: &[f32],
    ) -> (Vec<usize>, Vec<f32>) {
        let n = self.n;
        let k = centroids.n();
        let d = self.d;

        // Compute X @ C^T (n x k matrix) via SGEMM.
        let mut xct = vec![0.0f32; n * k];
        unsafe {
            matrixmultiply::sgemm(
                n,   // m
                k,   // n (output cols)
                d,   // k (inner dim)
                1.0, // alpha
                self.data.as_ptr(),
                d as isize, // row stride of X
                1,          // col stride of X
                centroids.data.as_ptr(),
                d as isize, // row stride of C (each centroid is a row)
                1,          // col stride of C
                0.0,        // beta
                xct.as_mut_ptr(),
                k as isize, // row stride of output
                1,          // col stride of output
            );
        }

        // Compute ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x.c and find argmin.
        let mut labels = vec![0usize; n];
        let mut upper = vec![f32::MAX; n];

        for i in 0..n {
            let xn = x_norms[i];
            let row_offset = i * k;
            for j in 0..k {
                let dist = xn + c_norms[j] - 2.0 * xct[row_offset + j];
                if dist < upper[i] {
                    upper[i] = dist;
                    labels[i] = j;
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

    #[test]
    fn flat_ref_basic() {
        let buf = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = FlatRef::new(&buf, 3, 2);
        assert_eq!(r.n(), 3);
        assert_eq!(r.d(), 2);
        assert_eq!(r.row(0), &[1.0, 2.0]);
        assert_eq!(r.row(1), &[3.0, 4.0]);
        assert_eq!(r.row(2), &[5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "FlatRef")]
    fn flat_ref_wrong_size() {
        FlatRef::new(&[1.0, 2.0, 3.0], 2, 2);
    }

    #[test]
    fn from_data_matches_from_vecs() {
        let vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let from_vecs = FlatMatrix::from_vecs(&vecs);
        let from_data = FlatMatrix::from_data(&vecs);
        assert_eq!(from_vecs.as_slice(), from_data.as_slice());
    }

    #[test]
    fn from_data_with_flat_ref() {
        let buf = vec![1.0, 2.0, 3.0, 4.0];
        let r = FlatRef::new(&buf, 2, 2);
        let fm = FlatMatrix::from_data(&r);
        assert_eq!(fm.n(), 2);
        assert_eq!(fm.d(), 2);
        assert_eq!(fm.row(0), &[1.0, 2.0]);
        assert_eq!(fm.row(1), &[3.0, 4.0]);
    }

    /// FlatRef path: pass flat data to every algorithm to verify the DataRef
    /// trait works end-to-end.
    #[test]
    fn flat_ref_all_algorithms() {
        use super::super::*;

        // 4 points, 2 dimensions: two well-separated clusters.
        let flat = vec![0.0f32, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
        let data = FlatRef::new(&flat, 4, 2);

        // K-means
        let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);

        // K-means fit + predict + wcss
        let fit = Kmeans::new(2).with_seed(42).fit(&data).unwrap();
        let predicted = fit.predict(&data).unwrap();
        assert_eq!(predicted.len(), 4);
        let w = fit.wcss(&data);
        assert!(w >= 0.0);

        // DBSCAN
        let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
        assert_eq!(labels[0], labels[1]);

        // HDBSCAN
        let labels = Hdbscan::new()
            .with_min_samples(2)
            .with_min_cluster_size(2)
            .fit_predict(&data)
            .unwrap();
        assert_eq!(labels.len(), 4);

        // OPTICS
        let result = Optics::new(1.0, 2).fit(&data).unwrap();
        assert_eq!(result.ordering.len(), 4);

        // COP-Kmeans
        let labels = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &[])
            .unwrap();
        assert_eq!(labels.len(), 4);

        // MiniBatchKmeans
        let mut mbk = MiniBatchKmeans::new(2).with_seed(42);
        let labels = mbk.update_batch(&data).unwrap();
        assert_eq!(labels.len(), 4);
        let pred = mbk.predict(&data).unwrap();
        assert_eq!(pred.len(), 4);

        // DenStream
        let mut ds = DenStream::new(2.0, 2)
            .with_beta(0.5)
            .with_lambda(0.001)
            .with_mu(1.0);
        let labels = ds.update_batch(&data).unwrap();
        assert_eq!(labels.len(), 4);
        let pred = ds.predict_batch(&data).unwrap();
        assert_eq!(pred.len(), 4);

        // EVoC (needs higher-dim data for intermediate_dim < d requirement)
        let flat16 = vec![0.0f32; 4 * 16]; // 4 points, 16 dims -- all zeros is fine for smoke test
        let data16 = FlatRef::new(&flat16, 4, 16);
        let mut evoc = EVoC::new(EVoCParams {
            intermediate_dim: 1,
            min_cluster_size: 2,
            seed: Some(42),
            ..Default::default()
        });
        let evoc_labels = evoc.fit_predict(&data16).unwrap();
        assert_eq!(evoc_labels.len(), 4);

        // Metrics
        let labels = vec![0, 0, 1, 1];
        let centroids = vec![vec![0.05, 0.05], vec![10.05, 10.05]];
        let sil = metrics::silhouette_score(&data, &labels, &Euclidean);
        assert!(sil > 0.9);
        let ch = metrics::calinski_harabasz(&data, &labels, &centroids);
        assert!(ch > 0.0);
        let db = metrics::davies_bouldin(&data, &labels, &centroids, &Euclidean);
        assert!(db < 1.0);
        let kd = metrics::k_distance(&data, 1, &Euclidean);
        assert_eq!(kd.len(), 4);
    }
}
