//! K-means clustering.
//!
//! Partitions data into k clusters by minimizing **within-cluster sum of squares**
//! (WCSS). The foundational clustering algorithm, dating to 1957 (Lloyd).
//!
//! # The Objective
//!
//! K-means minimizes:
//!
//! ```text
//! WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²
//! ```
//!
//! Sum of squared distances from each point to its cluster centroid.
//!
//! # Lloyd's Algorithm
//!
//! 1. Initialize k centroids (randomly or via k-means++)
//! 2. **Assign**: Each point → nearest centroid
//! 3. **Update**: Each centroid → mean of assigned points
//! 4. Repeat until convergence
//!
//! **Why it converges**: WCSS decreases monotonically. Each step either
//! decreases WCSS or leaves it unchanged. Bounded below by 0 → must converge.
//!
//! # Failure Modes
//!
//! - **Local optima**: NP-hard problem; Lloyd finds local minimum only
//! - **Wrong k**: Must specify k in advance; use elbow method or silhouette
//! - **Non-spherical clusters**: Assumes roughly spherical, equal-sized clusters
//! - **Initialization sensitivity**: Bad initial centroids → bad results
//!
//! ## K-means++ Initialization
//!
//! Addresses initialization by spreading initial centroids:
//! 1. Choose first centroid uniformly at random
//! 2. Choose next centroid with probability proportional to D(x)²
//!    (squared distance to nearest existing centroid)
//!
//! Provides provable O(log k) approximation to optimal WCSS.
//!
//! # Connection to IVF
//!
//! K-means is the foundation of IVF (Inverted File) indexing for ANN search.
//! Partition vectors into k cells, search only nearby cells at query time.
//!
//! # Research Context
//!
//! - **Breathing K-Means** (Fritzke, 2020): Dynamically adding/removing centroids
//!   can escape local optima better than static k.
//! - **`D^alpha` Seeding** (Bamas et al., 2023): Using sharper probability weighting
//!   (`alpha > 2`) during initialization can improve final cost.
//! - **Hamerly bounds** (Hamerly, SDM 2010): Per-point upper/lower distance bounds
//!   skip assignment recomputation when the bound proves assignment cannot change.
//!   O(n) extra memory, same exact results as Lloyd's. 2-8x speedup.
//! - **Flash-KMeans** (Yang et al., 2026): IO-aware GPU k-means using online argmin
//!   (no N*K distance matrix) and sort-inverse centroid updates. The online argmin
//!   pattern transfers to CPU via cache-aware tiling.
//! - **Spherical k-means** (Dhillon & Modha, 2001): For cosine distance, centroids
//!   must be L2-normalized after each update to stay on the unit sphere.
//!
//! This implementation uses Lloyd's algorithm with **k-means++** (`alpha=2`),
//! **Hamerly bounds** for assignment pruning, and **incremental init**
//! (O(n*k) instead of O(n*k^2)).

use super::distance::{DistanceMetric, SquaredEuclidean};
use super::util;
use crate::error::{Error, Result};
use rand::prelude::*;

/// K-means clustering algorithm, generic over a distance metric.
///
/// The default metric is [`SquaredEuclidean`], which preserves backward
/// compatibility with previous versions.
///
/// ```
/// use clump::Kmeans;
///
/// let data = vec![
///     vec![0.0f32, 0.0],
///     vec![0.1, 0.1],
///     vec![10.0, 10.0],
///     vec![10.1, 10.1],
/// ];
///
/// let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
/// assert_eq!(labels[0], labels[1]);
/// assert_ne!(labels[0], labels[2]);
/// ```
#[derive(Debug, Clone)]
pub struct Kmeans<D: DistanceMetric = SquaredEuclidean> {
    /// Number of clusters.
    k: usize,
    /// Maximum iterations.
    max_iter: usize,
    /// Convergence tolerance.
    tol: f64,
    /// Random seed.
    seed: Option<u64>,
    /// Seeding alpha (exponent for k-means++). Default 2.0 (standard).
    /// α > 2 (e.g. 4.0) can improve final cost (Bamas et al. 2023).
    seeding_alpha: f32,
    /// Distance metric.
    metric: D,
    /// Optional initial centroids for warm-starting (skips k-means++ init).
    init_centroids: Option<Vec<Vec<f32>>>,
}

/// Result of fitting k-means, generic over a distance metric.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KmeansFit<D: DistanceMetric = SquaredEuclidean> {
    /// Learned centroids (k x d).
    pub centroids: Vec<Vec<f32>>,
    /// One label per training point.
    pub labels: Vec<usize>,
    /// Number of Lloyd iterations executed.
    pub iters: usize,
    /// Per-iteration inertia (WCSS) trace. Entry `i` is the WCSS after
    /// iteration `i` (0-indexed). Useful for convergence diagnostics and
    /// verifying monotone improvement.
    pub inertia_trace: Vec<f32>,
    metric: D,
}

impl<D: DistanceMetric> KmeansFit<D> {
    /// Predict cluster labels for new points using the learned centroids.
    ///
    /// ```
    /// use clump::Kmeans;
    ///
    /// let data = vec![vec![0.0f32, 0.0], vec![10.0, 10.0]];
    /// let fit = Kmeans::new(2).with_seed(42).fit(&data).unwrap();
    ///
    /// let predicted = fit.predict(&[vec![0.05, 0.05], vec![9.9, 9.9]]).unwrap();
    /// assert_ne!(predicted[0], predicted[1]);
    /// ```
    pub fn predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        if data.is_empty() {
            return Err(Error::EmptyInput);
        }
        if self.centroids.is_empty() {
            return Err(Error::InvalidParameter {
                name: "centroids",
                message: "must be non-empty",
            });
        }

        let d = self.centroids[0].len();
        let mut out = Vec::with_capacity(data.len());
        for point in data {
            if point.len() != d {
                return Err(Error::DimensionMismatch {
                    expected: d,
                    found: point.len(),
                });
            }
            out.push(util::assign_nearest(point, &self.centroids, &self.metric));
        }

        Ok(out)
    }

    /// Within-Cluster Sum of Squares (WCSS / inertia).
    ///
    /// Sum of squared distances from each point to its assigned centroid.
    /// Used for the elbow method: plot WCSS vs k, pick the "elbow."
    pub fn wcss(&self, data: &[Vec<f32>]) -> f32 {
        data.iter()
            .zip(self.labels.iter())
            .map(|(point, &label)| self.metric.distance(point, &self.centroids[label]))
            .sum()
    }
}

impl Kmeans<SquaredEuclidean> {
    /// Create a new K-means clusterer with the default squared Euclidean distance.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0`.
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be at least 1");
        Self {
            k,
            max_iter: 100,
            tol: 1e-4,
            seed: None,
            seeding_alpha: 2.0,
            metric: SquaredEuclidean,
            init_centroids: None,
        }
    }
}

impl<D: DistanceMetric> Kmeans<D> {
    /// Create a new K-means clusterer with a custom distance metric.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0`.
    pub fn with_metric(k: usize, metric: D) -> Self {
        assert!(k > 0, "k must be at least 1");
        Self {
            k,
            max_iter: 100,
            tol: 1e-4,
            seed: None,
            seeding_alpha: 2.0,
            metric,
            init_centroids: None,
        }
    }

    /// Set seeding alpha (exponent for k-means++ probability weighting).
    ///
    /// Standard k-means++ uses α=2.0 (D² weighting).
    /// Research (Bamas et al. 2023) suggests α > 2 (e.g. 4.0) can yield better final clustering.
    pub fn with_seeding_alpha(mut self, alpha: f32) -> Self {
        self.seeding_alpha = alpha;
        self
    }

    /// Provide initial centroids for warm-starting.
    ///
    /// Skips k-means++ initialization and uses these centroids directly.
    /// The number of centroids must equal k.
    pub fn with_centroids(mut self, centroids: Vec<Vec<f32>>) -> Self {
        self.init_centroids = Some(centroids);
        self
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Fit k-means and return centroids, labels, and iteration count.
    ///
    /// ```
    /// use clump::Kmeans;
    ///
    /// let data = vec![vec![0.0f32, 0.0], vec![0.1, 0.1], vec![5.0, 5.0], vec![5.1, 5.1]];
    /// let fit = Kmeans::new(2).with_seed(42).fit(&data).unwrap();
    ///
    /// assert_eq!(fit.centroids.len(), 2);
    /// assert_eq!(fit.labels.len(), 4);
    /// assert!(fit.iters > 0);
    /// ```
    pub fn fit(&self, data: &[Vec<f32>]) -> Result<KmeansFit<D>> {
        if data.is_empty() {
            return Err(Error::EmptyInput);
        }

        if self.k == 0 {
            return Err(Error::InvalidParameter {
                name: "k",
                message: "must be at least 1",
            });
        }

        let n = data.len();
        let d = data[0].len();
        if d == 0 {
            return Err(Error::InvalidParameter {
                name: "dimension",
                message: "must be at least 1",
            });
        }

        if self.k > n {
            return Err(Error::InvalidClusterCount {
                requested: self.k,
                n_items: n,
            });
        }

        // Validate uniform dimensionality.
        for point in data {
            if point.len() != d {
                return Err(Error::DimensionMismatch {
                    expected: d,
                    found: point.len(),
                });
            }
        }

        util::validate_finite(data)?;

        // Normalize tolerance by data variance so it scales with data magnitude.
        // Without this, the raw tol is compared against the sum of k*d squared
        // centroid shifts, which becomes meaninglessly tight for high-dimensional
        // or large-scale data. (Matches scikit-learn's _tolerance approach.)
        let effective_tol = (self.tol * util::mean_variance(data) * self.k as f64) as f32;

        // Initialize RNG.
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        // Initialize centroids: use provided centroids (warm-start) or k-means++.
        let mut centroids = if let Some(ref init) = self.init_centroids {
            assert_eq!(
                init.len(),
                self.k,
                "init_centroids length ({}) must equal k ({})",
                init.len(),
                self.k
            );
            init.clone()
        } else {
            util::kmeanspp_init(data, self.k, &self.metric, self.seeding_alpha, &mut rng)
        };
        let mut labels = vec![0usize; n];

        // Pre-allocate working buffers outside the iteration loop to avoid
        // per-iteration allocation overhead (AMD ROCm pattern).
        let mut new_centroids = vec![vec![0.0f32; d]; self.k];
        let mut counts = vec![0usize; self.k];

        // Hamerly bounds: per-point upper (dist to assigned centroid) and
        // lower (dist to second-nearest centroid). When upper <= lower,
        // the assignment cannot change and we skip distance computation.
        // O(n) extra memory (Hamerly, SDM 2010).
        let mut upper_bounds = vec![f32::MAX; n];
        let mut lower_bounds = vec![0.0f32; n];
        let mut centroid_shifts = vec![0.0f32; self.k];
        let mut sums_f64 = vec![vec![0.0f64; d]; self.k];
        let mut flat_buf: Vec<f32> = Vec::with_capacity(self.k * d);
        let mut inertia_trace: Vec<f32> = Vec::with_capacity(self.max_iter);

        // (Expanded squared Euclidean removed -- Hamerly bounds are used
        // for both parallel and sequential paths.)

        // GPU acceleration: initialize Metal compute pipeline for assignment
        // when using SquaredEuclidean and problem is large enough to amortize
        // GPU setup overhead. Buffers for data, labels, and params are allocated
        // once here and reused across iterations.
        #[cfg(feature = "gpu")]
        let gpu_assigner = if self.metric.supports_expanded_form() && n * self.k >= 500_000 {
            let data_flat = super::gpu::flatten(data);
            super::gpu::GpuAssigner::new(&data_flat, n, self.k, d)
        } else {
            None
        };

        // BLAS GEMM for first-iteration assignment (when blas feature enabled).
        // matrixmultiply's optimized SGEMM beats per-point distance loops
        // for large n*k due to micro-kernel SIMD and cache optimization.
        #[cfg(feature = "blas")]
        let use_blas = n * self.k >= 100_000 && self.metric.supports_expanded_form();
        #[cfg(feature = "blas")]
        let blas_data = if use_blas {
            let fd = super::flat::FlatMatrix::from_vecs(data);
            let xn = fd.row_norms_sq();
            Some((fd, xn))
        } else {
            None
        };

        let mut iters = 0usize;
        for iter in 0..self.max_iter {
            iters = iter + 1;

            // Zero the accumulators for this iteration.
            for c in &mut new_centroids {
                c.fill(0.0);
            }
            counts.fill(0);

            // Assignment step.
            // Priority: GPU > BLAS GEMM (first iter) > Hamerly bounds.
            #[cfg(feature = "blas")]
            let blas_used = if iter == 0 && use_blas {
                if let Some((ref fd, ref xn)) = blas_data {
                    let fc = super::flat::FlatMatrix::from_vecs(&centroids);
                    let cn = fc.row_norms_sq();
                    let (new_labels, new_upper) = fd.blas_assign(&fc, xn, &cn);
                    labels.copy_from_slice(&new_labels);
                    for i in 0..n {
                        upper_bounds[i] = new_upper[i];
                        lower_bounds[i] = 0.0;
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            };
            #[cfg(not(feature = "blas"))]
            let blas_used = false;

            #[cfg(feature = "gpu")]
            let gpu_used = if !blas_used {
                if let Some(ref assigner) = gpu_assigner {
                    let centroids_flat = super::gpu::flatten(&centroids);
                    let gpu_labels = assigner.assign(&centroids_flat);
                    labels.copy_from_slice(&gpu_labels);
                    true
                } else {
                    false
                }
            } else {
                false
            };
            #[cfg(not(feature = "gpu"))]
            let gpu_used = blas_used; // skip Hamerly if BLAS handled it

            if !gpu_used {
                // Hamerly bounds-based assignment: skips distance computation
                // for points whose assignment provably cannot change.
                #[cfg(feature = "parallel")]
                {
                    util::hamerly_assign_parallel(
                        data,
                        &centroids,
                        &mut labels,
                        &mut upper_bounds,
                        &mut lower_bounds,
                        &centroid_shifts,
                        &self.metric,
                        iter == 0,
                        &mut flat_buf,
                    );
                }

                #[cfg(not(feature = "parallel"))]
                if self.k <= 20 {
                    // Geometric assign: bound-free, O(k^2) centroid-pair check.
                    // Better than Hamerly for small k (no per-point bound overhead).
                    util::geometric_assign(
                        data,
                        &centroids,
                        &mut labels,
                        &centroid_shifts,
                        &self.metric,
                        iter == 0,
                    );
                } else {
                    util::hamerly_assign(
                        data,
                        &centroids,
                        &mut labels,
                        &mut upper_bounds,
                        &mut lower_bounds,
                        &centroid_shifts,
                        &self.metric,
                        iter == 0,
                        &mut flat_buf,
                    );
                }
            }

            // Update step: accumulate in f64 to avoid precision loss at
            // large n (sklearn pattern). f32 accumulation loses ~2.5 digits
            // at n=100k, causing convergence issues.
            for s in &mut sums_f64 {
                s.fill(0.0);
            }
            for i in 0..n {
                let k = labels[i];
                for j in 0..d {
                    sums_f64[k][j] += data[i][j] as f64;
                }
                counts[k] += 1;
            }

            for k in 0..self.k {
                if counts[k] > 0 {
                    let divisor = counts[k] as f64;
                    for j in 0..d {
                        new_centroids[k][j] = (sums_f64[k][j] / divisor) as f32;
                    }
                } else {
                    // Empty cluster: split the largest cluster by moving the
                    // farthest point from its centroid. More stable than random
                    // reinitialization, which can cause oscillation (Hamerly 2010).
                    let largest = counts
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, &c)| c)
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    let mut farthest_idx = 0;
                    let mut farthest_dist = -1.0f32;
                    for (i, &label) in labels.iter().enumerate() {
                        if label == largest {
                            let dist = self.metric.distance(&data[i], &new_centroids[largest]);
                            if dist > farthest_dist {
                                farthest_dist = dist;
                                farthest_idx = i;
                            }
                        }
                    }
                    new_centroids[k] = data[farthest_idx].clone();
                }
            }

            // Spherical k-means: L2-normalize centroids after update when
            // using cosine distance (Dhillon & Modha 2001).
            if self.metric.normalize_centroids() {
                for c in &mut new_centroids {
                    let norm: f32 = c.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    if norm > f32::EPSILON {
                        for val in c.iter_mut() {
                            *val /= norm;
                        }
                    }
                }
            }

            // Compute per-centroid shift distances for Hamerly bounds update
            // and total convergence shift in a single pass (avoids double
            // distance computation -- profiler finding #3).
            let mut shift = 0.0f32;
            for k in 0..self.k {
                let d = self.metric.distance(&centroids[k], &new_centroids[k]);
                centroid_shifts[k] = d;
                // Convergence uses sum of squared element-wise shifts.
                // For SquaredEuclidean, d == sum((a-b)^2) already.
                // For other metrics, d is the metric distance.
                shift += d;
            }

            std::mem::swap(&mut centroids, &mut new_centroids);

            // Record per-iteration inertia (WCSS) for convergence diagnostics.
            let wcss: f32 = labels
                .iter()
                .enumerate()
                .map(|(i, &l)| self.metric.distance(&data[i], &centroids[l]))
                .sum();
            inertia_trace.push(wcss);

            if shift < effective_tol {
                break;
            }
        }

        // Final consistency pass: reassign all points to nearest centroid
        // using the final centroids. This ensures labels and centroids are
        // mutually consistent (the correctness research identified this as
        // a common source of bugs in k-means implementations).
        for (i, label) in labels.iter_mut().enumerate() {
            *label = util::assign_nearest(&data[i], &centroids, &self.metric);
        }

        Ok(KmeansFit {
            centroids,
            labels,
            iters,
            inertia_trace,
            metric: self.metric.clone(),
        })
    }
}

impl<D: DistanceMetric> Kmeans<D> {
    /// Fit and return one cluster label per input point.
    pub fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        Ok(self.fit(data)?.labels)
    }

    /// The configured number of clusters.
    pub fn n_clusters(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::distance::Euclidean;

    #[test]
    fn test_kmeans_basic() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let kmeans = Kmeans::new(2).with_seed(42);
        let labels = kmeans.fit_predict(&data).unwrap();

        // Points 0,1 should be in same cluster, points 2,3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_kmeans_all_points_assigned() {
        // Property: every point must be assigned to exactly one cluster
        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32 * 0.1, (i % 5) as f32])
            .collect();

        let kmeans = Kmeans::new(5).with_seed(123);
        let labels = kmeans.fit_predict(&data).unwrap();

        // All points assigned
        assert_eq!(labels.len(), data.len());

        // All labels in valid range [0, k)
        for &label in &labels {
            assert!(label < 5, "label {} out of range", label);
        }
    }

    #[test]
    fn test_kmeans_k_equals_n() {
        // Edge case: k = n (each point its own cluster)
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let kmeans = Kmeans::new(3).with_seed(42);
        let labels = kmeans.fit_predict(&data).unwrap();

        // Each point in different cluster
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_kmeans_deterministic_with_seed() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let kmeans1 = Kmeans::new(2).with_seed(42);
        let kmeans2 = Kmeans::new(2).with_seed(42);

        let labels1 = kmeans1.fit_predict(&data).unwrap();
        let labels2 = kmeans2.fit_predict(&data).unwrap();

        assert_eq!(labels1, labels2, "same seed should give same result");
    }

    #[test]
    fn test_kmeans_scaling_invariant() {
        // Metamorphic: uniform scaling shouldn't change cluster assignments
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let scaled: Vec<Vec<f32>> = data
            .iter()
            .map(|v| v.iter().map(|x| x * 100.0).collect())
            .collect();

        let kmeans1 = Kmeans::new(2).with_seed(42);
        let kmeans2 = Kmeans::new(2).with_seed(42);

        let labels1 = kmeans1.fit_predict(&data).unwrap();
        let labels2 = kmeans2.fit_predict(&scaled).unwrap();

        // Same structure (labels may be permuted)
        assert_eq!(labels1[0], labels1[1]);
        assert_eq!(labels2[0], labels2[1]);
        assert_eq!(labels1[2], labels1[3]);
        assert_eq!(labels2[2], labels2[3]);
        assert_ne!(labels1[0], labels1[2]);
        assert_ne!(labels2[0], labels2[2]);
    }

    #[test]
    fn test_kmeans_empty_input_error() {
        let data: Vec<Vec<f32>> = vec![];
        let kmeans = Kmeans::new(2);
        let result = kmeans.fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_alpha_seeding() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
        ];

        // With alpha=4.0, should still work and converge
        let kmeans = Kmeans::new(2).with_seed(42).with_seeding_alpha(4.0);
        let labels = kmeans.fit_predict(&data).unwrap();

        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_kmeans_with_euclidean() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let kmeans = Kmeans::with_metric(2, Euclidean).with_seed(42);
        let labels = kmeans.fit_predict(&data).unwrap();

        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_kmeans_fit_predict_with_custom_metric() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let kmeans = Kmeans::with_metric(2, Euclidean).with_seed(42);
        let fit = kmeans.fit(&data).unwrap();

        // Predict on new data
        let new_data = vec![vec![0.05, 0.05], vec![10.05, 10.05]];
        let predicted = fit.predict(&new_data).unwrap();
        assert_ne!(predicted[0], predicted[1]);
    }

    /// NaN input should be rejected, not silently produce garbage.
    #[test]
    fn nan_input_rejected() {
        let data = vec![vec![0.0, f32::NAN], vec![1.0, 1.0]];
        let result = Kmeans::new(2).with_seed(42).fit_predict(&data);
        assert!(result.is_err());
    }

    /// Infinity input should be rejected.
    #[test]
    fn inf_input_rejected() {
        let data = vec![vec![0.0, 0.0], vec![1.0, f32::INFINITY]];
        let result = Kmeans::new(2).with_seed(42).fit_predict(&data);
        assert!(result.is_err());
    }

    /// All-identical points: k-means should converge quickly without error.
    #[test]
    fn all_identical_points() {
        let data = vec![vec![5.0, 5.0]; 10];
        let fit = Kmeans::new(2).with_seed(42).fit(&data).unwrap();
        // Should converge in very few iterations.
        assert!(
            fit.iters <= 3,
            "expected fast convergence, got {} iters",
            fit.iters
        );
    }

    /// k=1 trivial case: single centroid should equal the data mean.
    #[test]
    fn k1_centroid_equals_mean() {
        let data = vec![vec![0.0, 0.0], vec![2.0, 4.0], vec![4.0, 8.0]];
        let fit = Kmeans::new(1).with_seed(42).fit(&data).unwrap();
        let centroid = &fit.centroids[0];
        assert!(
            (centroid[0] - 2.0).abs() < 1e-4,
            "mean[0] should be 2.0, got {}",
            centroid[0]
        );
        assert!(
            (centroid[1] - 4.0).abs() < 1e-4,
            "mean[1] should be 4.0, got {}",
            centroid[1]
        );
    }

    /// Self-identity oracle: feeding centroids back as input should assign each
    /// to itself (catches distance/assignment bugs).
    #[test]
    fn self_identity_oracle() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let fit = Kmeans::new(2).with_seed(42).fit(&data).unwrap();
        let predicted = fit.predict(&fit.centroids).unwrap();
        for (k, &label) in predicted.iter().enumerate() {
            assert_eq!(label, k, "centroid {k} should map to cluster {k}");
        }
    }

    /// 1-dimensional data should work without error.
    #[test]
    fn scalar_data_d1() {
        let data = vec![vec![0.0], vec![0.1], vec![10.0], vec![10.1]];
        let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
    }

    /// Cosine k-means: centroids should be L2-normalized after fitting
    /// (spherical k-means, Dhillon & Modha 2001).
    #[test]
    fn cosine_centroids_are_normalized() {
        use crate::cluster::distance::CosineDistance;

        // Points in roughly two angular directions.
        let data = vec![
            vec![1.0, 0.1],
            vec![2.0, 0.2],
            vec![0.1, 1.0],
            vec![0.2, 2.0],
        ];
        let fit = Kmeans::with_metric(2, CosineDistance)
            .with_seed(42)
            .fit(&data)
            .unwrap();

        for (k, c) in fit.centroids.iter().enumerate() {
            let norm: f32 = c.iter().map(|&x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "centroid {k} should be unit-normalized, got norm={norm}"
            );
        }
    }

    /// Large k stress test: k=100 on 5000 points.
    #[test]
    fn large_k_stress() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<Vec<f32>> = (0..5000)
            .map(|_| (0..16).map(|_| rng.random::<f32>()).collect())
            .collect();
        let labels = Kmeans::new(100)
            .with_max_iter(5)
            .with_seed(42)
            .fit_predict(&data)
            .unwrap();
        assert_eq!(labels.len(), 5000);
        for &l in &labels {
            assert!(l < 100);
        }
    }

    /// Empty cluster reinit: when one cluster loses all points, the
    /// split-largest-cluster strategy should fire and produce k centroids.
    #[test]
    fn empty_cluster_reinit() {
        // 3 tight clusters but ask for k=4: one cluster must be reinitialized.
        let data = vec![
            vec![0.0, 0.0],
            vec![0.01, 0.01],
            vec![10.0, 0.0],
            vec![10.01, 0.01],
            vec![0.0, 10.0],
            vec![0.01, 10.01],
        ];
        let fit = Kmeans::new(4).with_seed(42).fit(&data).unwrap();
        assert_eq!(fit.centroids.len(), 4);
        // All points assigned.
        assert_eq!(fit.labels.len(), 6);
    }

    /// d >> n: high-dimensional data with few points.
    #[test]
    fn high_dim_few_points() {
        let data = vec![
            vec![0.0; 200],
            {
                let mut v = vec![0.0; 200];
                v[0] = 1.0;
                v
            },
            vec![10.0; 200],
            {
                let mut v = vec![10.0; 200];
                v[0] = 11.0;
                v
            },
        ];
        let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    /// Single point: n=1 with k=1 should work.
    #[test]
    fn single_point_k1() {
        let data = vec![vec![42.0, 7.0]];
        let fit = Kmeans::new(1).fit(&data).unwrap();
        assert_eq!(fit.centroids.len(), 1);
        assert!((fit.centroids[0][0] - 42.0).abs() < 1e-6);
        assert!((fit.centroids[0][1] - 7.0).abs() < 1e-6);
    }

    /// WCSS must be strictly better than random assignment.
    #[test]
    fn wcss_better_than_random() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<Vec<f32>> = (0..200)
            .map(|i| {
                let center = if i < 100 { 0.0 } else { 20.0 };
                vec![center + rng.random::<f32>(), center + rng.random::<f32>()]
            })
            .collect();

        let fit = Kmeans::new(2).with_seed(42).fit(&data).unwrap();
        let kmeans_wcss = fit.wcss(&data);

        // Random assignment.
        let random_wcss: f32 = data
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let label = i % 2;
                SquaredEuclidean.distance(p, &fit.centroids[label])
            })
            .sum();

        assert!(
            kmeans_wcss < random_wcss,
            "k-means WCSS ({kmeans_wcss}) should be less than random ({random_wcss})"
        );
    }

    /// Warm-start should produce equal or better WCSS than fresh init.
    #[test]
    fn warm_start_convergence() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let fit1 = Kmeans::new(2).with_seed(42).fit(&data).unwrap();
        let fit2 = Kmeans::new(2)
            .with_centroids(fit1.centroids.clone())
            .fit(&data)
            .unwrap();

        // Warm-started from optimal centroids should converge in 1 iteration.
        assert!(
            fit2.iters <= 2,
            "warm-start should converge fast, got {} iters",
            fit2.iters
        );
    }

    /// Centroids should approximate the mean of assigned points.
    /// After convergence + final reassignment, centroids may not be exact
    /// means if boundary points shifted, but should be very close.
    #[test]
    fn centroids_approximate_means() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.0],
        ];
        let fit = Kmeans::new(2).with_seed(42).fit(&data).unwrap();

        for k in 0..2 {
            let members: Vec<&Vec<f32>> = data
                .iter()
                .zip(fit.labels.iter())
                .filter(|(_, &l)| l == k)
                .map(|(p, _)| p)
                .collect();
            if members.is_empty() {
                continue;
            }
            let d = members[0].len();
            for j in 0..d {
                let mean: f32 = members.iter().map(|p| p[j]).sum::<f32>() / members.len() as f32;
                assert!(
                    (fit.centroids[k][j] - mean).abs() < 0.5,
                    "centroid[{k}][{j}] = {}, expected ~{mean}",
                    fit.centroids[k][j]
                );
            }
        }
    }

    /// Predict on training data must match fit labels (consistency).
    #[test]
    fn predict_matches_fit_labels() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<Vec<f32>> = (0..100)
            .map(|_| vec![rng.random::<f32>() * 10.0, rng.random::<f32>() * 10.0])
            .collect();
        let fit = Kmeans::new(5).with_seed(42).fit(&data).unwrap();

        // Predict on the same data should give the same labels as fit.
        let predicted = fit.predict(&data).unwrap();
        assert_eq!(
            fit.labels, predicted,
            "predict on training data must match fit labels"
        );
    }

    /// Refit from converged centroids should be a fixed point (idempotent).
    #[test]
    fn idempotent_refit() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![20.0, 20.0],
            vec![20.1, 20.1],
        ];
        let fit1 = Kmeans::new(3).with_seed(42).fit(&data).unwrap();
        let fit2 = Kmeans::new(3)
            .with_centroids(fit1.centroids.clone())
            .fit(&data)
            .unwrap();
        assert_eq!(fit1.labels, fit2.labels, "refit should produce same labels");
    }

    /// Extreme scale: data * 1e-6 should produce same cluster structure.
    #[test]
    fn extreme_scale_small() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let scaled: Vec<Vec<f32>> = data
            .iter()
            .map(|v| v.iter().map(|&x| x * 1e-6).collect())
            .collect();
        let labels1 = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
        let labels2 = Kmeans::new(2).with_seed(42).fit_predict(&scaled).unwrap();

        // Same cluster structure (labels may be permuted).
        assert_eq!(labels1[0] == labels1[1], labels2[0] == labels2[1]);
        assert_eq!(labels1[2] == labels1[3], labels2[2] == labels2[3]);
        assert_ne!(labels1[0], labels1[2]);
        assert_ne!(labels2[0], labels2[2]);
    }

    /// Centroids should be accurate to ~1e-4 even at n=10000
    /// (f64 accumulation prevents precision loss).
    #[test]
    fn precision_at_scale() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let n = 10000;
        let data: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let center = if i < n / 2 { 0.0 } else { 10.0 };
                vec![
                    center + rng.random::<f32>() * 0.1,
                    center + rng.random::<f32>() * 0.1,
                ]
            })
            .collect();

        let fit = Kmeans::new(2).with_seed(42).fit(&data).unwrap();

        // Verify centroids are close to the true means (0.05, 0.05) and (10.05, 10.05).
        for c in &fit.centroids {
            let near_zero = (c[0] - 0.05).abs() < 0.1 && (c[1] - 0.05).abs() < 0.1;
            let near_ten = (c[0] - 10.05).abs() < 0.1 && (c[1] - 10.05).abs() < 0.1;
            assert!(
                near_zero || near_ten,
                "centroid {:?} should be near (0.05, 0.05) or (10.05, 10.05)",
                c
            );
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_data(max_n: usize, d: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
        proptest::collection::vec(
            proptest::collection::vec(-100.0f32..100.0, d..=d),
            3..=max_n,
        )
    }

    proptest! {
        /// All labels must be in [0, k).
        #[test]
        fn labels_in_range(data in arb_data(50, 4)) {
            let k = 3.min(data.len());
            let labels = Kmeans::new(k).with_seed(42).with_max_iter(5)
                .fit_predict(&data).unwrap();
            prop_assert_eq!(labels.len(), data.len());
            for &l in &labels {
                prop_assert!(l < k, "label {} out of range [0, {})", l, k);
            }
        }

        /// predict on training data must match fit labels.
        #[test]
        fn predict_consistent(data in arb_data(30, 3)) {
            let k = 2.min(data.len());
            let fit = Kmeans::new(k).with_seed(42).with_max_iter(5)
                .fit(&data).unwrap();
            let predicted = fit.predict(&data).unwrap();
            prop_assert_eq!(&fit.labels, &predicted);
        }

        /// WCSS must be non-negative and finite.
        #[test]
        fn wcss_nonneg(data in arb_data(30, 3)) {
            let k = 2.min(data.len());
            let fit = Kmeans::new(k).with_seed(42).with_max_iter(5)
                .fit(&data).unwrap();
            let wcss = fit.wcss(&data);
            prop_assert!(wcss >= 0.0, "WCSS must be >= 0, got {}", wcss);
            prop_assert!(wcss.is_finite(), "WCSS must be finite");
        }

        /// Inertia trace must be monotonically non-increasing.
        /// Each iteration should produce same or lower WCSS (faiss pattern).
        #[test]
        fn inertia_monotone_decreasing(data in arb_data(30, 3)) {
            let k = 2.min(data.len());
            let fit = Kmeans::new(k).with_seed(42).with_max_iter(20)
                .fit(&data).unwrap();
            let trace = &fit.inertia_trace;
            prop_assert!(!trace.is_empty(), "inertia trace must not be empty");
            for w in trace.windows(2) {
                prop_assert!(w[1] <= w[0] + 1e-5,
                    "inertia increased: {} -> {}", w[0], w[1]);
            }
        }

        /// Inertia trace length must equal iteration count.
        #[test]
        fn inertia_trace_length(data in arb_data(20, 3)) {
            let k = 2.min(data.len());
            let fit = Kmeans::new(k).with_seed(42).with_max_iter(10)
                .fit(&data).unwrap();
            prop_assert_eq!(fit.inertia_trace.len(), fit.iters,
                "trace length {} != iters {}", fit.inertia_trace.len(), fit.iters);
        }
    }
}
