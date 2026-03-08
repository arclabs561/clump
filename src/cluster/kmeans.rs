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
//! - **$D^\alpha$ Seeding** (Bamas et al., 2023): Using sharper probability weighting
//!   ($\alpha > 2$) during initialization can improve final cost.
//!
//! This crate implements standard Lloyd's algorithm with **k-means++** ($\alpha=2$).

use super::distance::{DistanceMetric, SquaredEuclidean};
use super::util;
use crate::error::{Error, Result};
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
}

/// Result of fitting k-means, generic over a distance metric.
#[derive(Debug, Clone)]
pub struct KmeansFit<D: DistanceMetric = SquaredEuclidean> {
    /// Learned centroids (k x d).
    pub centroids: Vec<Vec<f32>>,
    /// One label per training point.
    pub labels: Vec<usize>,
    /// Number of Lloyd iterations executed.
    pub iters: usize,
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
}

impl Kmeans<SquaredEuclidean> {
    /// Create a new K-means clusterer with the default squared Euclidean distance.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 100,
            tol: 1e-4,
            seed: None,
            seeding_alpha: 2.0,
            metric: SquaredEuclidean,
        }
    }
}

impl<D: DistanceMetric> Kmeans<D> {
    /// Create a new K-means clusterer with a custom distance metric.
    pub fn with_metric(k: usize, metric: D) -> Self {
        Self {
            k,
            max_iter: 100,
            tol: 1e-4,
            seed: None,
            seeding_alpha: 2.0,
            metric,
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

        // Initialize centroids via k-means++.
        let mut centroids =
            util::kmeanspp_init(data, self.k, &self.metric, self.seeding_alpha, &mut rng);
        let mut labels = vec![0usize; n];

        let mut iters = 0usize;
        for iter in 0..self.max_iter {
            iters = iter + 1;

            // Assignment step.
            #[cfg(feature = "parallel")]
            {
                let centroids_ref = &centroids;
                let metric = &self.metric;
                labels.par_iter_mut().enumerate().for_each(|(i, label)| {
                    *label = util::assign_nearest(&data[i], centroids_ref, metric);
                });
            }

            #[cfg(not(feature = "parallel"))]
            for (i, label) in labels.iter_mut().enumerate() {
                *label = util::assign_nearest(&data[i], &centroids, &self.metric);
            }

            // Update step: recompute centroids as mean of assigned points.
            let mut new_centroids = vec![vec![0.0f32; d]; self.k];
            let mut counts = vec![0usize; self.k];

            for i in 0..n {
                let k = labels[i];
                for j in 0..d {
                    new_centroids[k][j] += data[i][j];
                }
                counts[k] += 1;
            }

            for k in 0..self.k {
                if counts[k] > 0 {
                    let divisor = counts[k] as f32;
                    for val in &mut new_centroids[k] {
                        *val /= divisor;
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

            // Check convergence: total squared shift across all centroid elements.
            let shift: f32 = centroids
                .iter()
                .zip(new_centroids.iter())
                .flat_map(|(old, new)| old.iter().zip(new.iter()).map(|(a, b)| (a - b).powi(2)))
                .sum();

            centroids = new_centroids;

            if shift < effective_tol {
                break;
            }
        }

        Ok(KmeansFit {
            centroids,
            labels,
            iters,
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
}
