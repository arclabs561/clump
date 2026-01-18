//! Dense clustering primitives.
//!
//! Current scope:
//! - **k-means** (k-means++ initialization + Lloyd iterations)
//!
//! This crate is intended to be a small, backend-agnostic primitive. It does not
//! assume tensors, GPUs, or a specific ML framework.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClumpError {
    #[error("no points provided")]
    EmptyInput,

    #[error("k must be >= 1")]
    InvalidK,

    #[error("dimension mismatch: expected dim={expected}, got dim={got}")]
    DimensionMismatch { expected: usize, got: usize },
}

#[derive(Debug, Clone)]
pub struct KMeansConfig {
    pub k: usize,
    pub max_iters: usize,
    pub tol: f32,
    pub seed: u64,
}

impl KMeansConfig {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iters: 50,
            tol: 1e-4,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub centroids: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
    pub iters: usize,
}

/// Run k-means on dense points.
///
/// Points are provided as slices; the result owns centroids and assignments.
pub fn kmeans(points: &[&[f32]], cfg: &KMeansConfig) -> Result<KMeansResult, ClumpError> {
    if points.is_empty() {
        return Err(ClumpError::EmptyInput);
    }
    if cfg.k == 0 {
        return Err(ClumpError::InvalidK);
    }

    let dim = points[0].len();
    if dim == 0 {
        return Err(ClumpError::DimensionMismatch { expected: 1, got: 0 });
    }
    for p in points.iter().skip(1) {
        if p.len() != dim {
            return Err(ClumpError::DimensionMismatch { expected: dim, got: p.len() });
        }
    }

    let n = points.len();
    let k = cfg.k.min(n);
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);

    // k-means++ init
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first_idx = rng.random_range(0..n);
    centroids.push(points[first_idx].to_vec());

    let mut min_d2: Vec<f32> = vec![0.0; n];
    for i in 0..n {
        min_d2[i] = l2_squared(points[i], &centroids[0]);
    }

    while centroids.len() < k {
        let sum: f64 = min_d2.iter().map(|&x| x.max(0.0) as f64).sum();
        // If all distances are ~0, fall back to a random point.
        let pick = if sum > 0.0 {
            let mut r = rng.random_range(0.0..sum);
            let mut idx = 0usize;
            for (i, &d) in min_d2.iter().enumerate() {
                r -= d.max(0.0) as f64;
                if r <= 0.0 {
                    idx = i;
                    break;
                }
            }
            idx
        } else {
            rng.random_range(0..n)
        };

        centroids.push(points[pick].to_vec());

        // Update min distances to nearest centroid.
        let c_last = centroids.last().unwrap();
        for i in 0..n {
            let d2 = l2_squared(points[i], c_last);
            if d2 < min_d2[i] {
                min_d2[i] = d2;
            }
        }
    }

    let mut assignments = vec![0usize; n];
    let mut prev_assignments = vec![usize::MAX; n];
    let mut iters = 0usize;

    for iter in 0..cfg.max_iters {
        iters = iter + 1;

        // Assign
        for i in 0..n {
            let mut best_k = 0usize;
            let mut best_d2 = f32::INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let d2 = l2_squared(points[i], c);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_k = j;
                }
            }
            assignments[i] = best_k;
        }

        if assignments == prev_assignments {
            break;
        }
        prev_assignments.clone_from(&assignments);

        // Recompute centroids
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (p, &a) in points.iter().zip(assignments.iter()) {
            counts[a] += 1;
            for d in 0..dim {
                sums[a][d] += p[d];
            }
        }

        let mut max_shift = 0.0f32;
        for j in 0..k {
            if counts[j] == 0 {
                // Empty cluster: re-seed to a random point.
                let idx = rng.random_range(0..n);
                centroids[j] = points[idx].to_vec();
                continue;
            }
            let inv = 1.0f32 / (counts[j] as f32);
            let mut new_c = vec![0.0f32; dim];
            for d in 0..dim {
                new_c[d] = sums[j][d] * inv;
            }
            let shift = l2_squared(&centroids[j], &new_c).sqrt();
            if shift > max_shift {
                max_shift = shift;
            }
            centroids[j] = new_c;
        }

        if max_shift <= cfg.tol {
            break;
        }
    }

    Ok(KMeansResult {
        centroids,
        assignments,
        iters,
    })
}

#[inline]
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn kmeans_two_blobs_smoke() {
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for i in 0..50 {
            pts.push(vec![i as f32 * 0.01, 0.0]);
        }
        for i in 0..50 {
            pts.push(vec![10.0 + i as f32 * 0.01, 0.0]);
        }
        let refs: Vec<&[f32]> = pts.iter().map(|v| v.as_slice()).collect();
        let cfg = KMeansConfig::new(2);
        let res = kmeans(&refs, &cfg).unwrap();
        assert_eq!(res.centroids.len(), 2);
        assert_eq!(res.assignments.len(), refs.len());
    }

    proptest! {
        #[test]
        fn prop_kmeans_assignments_in_range(
            n in 1usize..80,
            dim in 1usize..8,
            k in 1usize..8,
            seed in any::<u64>(),
        ) {
            let mut pts: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for _ in 0..n {
                let mut v = vec![0.0f32; dim];
                for d in 0..dim {
                    v[d] = rng.random_range(-1.0..1.0);
                }
                pts.push(v);
            }
            let refs: Vec<&[f32]> = pts.iter().map(|v| v.as_slice()).collect();
            let cfg = KMeansConfig { k, seed, ..KMeansConfig::new(k) };
            let res = kmeans(&refs, &cfg).unwrap();
            for &a in &res.assignments {
                prop_assert!(a < res.centroids.len());
            }
        }
    }
}

