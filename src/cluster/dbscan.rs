//! DBSCAN: Density-Based Spatial Clustering of Applications with Noise.
//!
//! # The Algorithm (Ester et al., 1996)
//!
//! DBSCAN is a density-based clustering algorithm that groups points based on
//! neighborhood density. Unlike k-means, it:
//!
//! - Discovers clusters of arbitrary shape
//! - Automatically determines the number of clusters
//! - Identifies noise points (outliers)
//!
//! ## Core Concepts
//!
//! - **Epsilon (ε)**: Maximum distance between two points to be neighbors.
//! - **MinPts**: Minimum neighbors within ε for a point to be "core".
//! - **Core point**: Has at least MinPts neighbors within ε.
//! - **Border point**: Within ε of a core point but not core itself.
//! - **Noise point**: Neither core nor border.
//!
//! ## Algorithm Steps
//!
//! 1. For each unvisited point P:
//!    - Find neighbors within ε
//!    - If |neighbors| < MinPts, mark as noise (may change later)
//!    - Else P is core: start new cluster, expand from neighbors
//!
//! 2. Expansion: For each core point's neighbors:
//!    - Add to cluster
//!    - If core, recursively expand
//!
//! ## Complexity
//!
//! - **Time**: O(n²) naive, O(n log n) with spatial index.
//! - **Space**: O(n) for labels.
//!
//! ## When to Use
//!
//! - Clusters have non-convex shapes
//! - Number of clusters unknown
//! - Data has outliers
//! - Clusters have similar density
//!
//! ## Limitations
//!
//! - Struggles with varying densities (consider OPTICS)
//! - ε parameter is sensitive and dataset-dependent
//!
//! ## References
//!
//! Ester et al. (1996). "A Density-Based Algorithm for Discovering Clusters
//! in Large Spatial Databases with Noise." KDD-96.
//!
//! # Research Context
//!
//! - **Parameter Sensitivity**: Epsilon is hard to set. K-distance plots help.
//! - **HDBSCAN***: A hierarchical extension that removes the epsilon parameter
//!   by building a stability-based cluster hierarchy. This is the modern
//!   default for density clustering (Campello et al., 2013).
//!   See [`Hdbscan`](super::Hdbscan).

use super::distance::{DistanceMetric, Euclidean};
use super::util;
use crate::error::{Error, Result};

/// DBSCAN clustering algorithm, generic over a distance metric.
///
/// The default metric is [`Euclidean`] (L2), matching the original behavior
/// where epsilon is compared against Euclidean distance.
///
/// ```
/// use clump::{Dbscan, NOISE};
///
/// let data = vec![
///     vec![0.0f32, 0.0], vec![0.1, 0.0], vec![0.0, 0.1], // cluster
///     vec![100.0, 100.0], // outlier
/// ];
///
/// let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
/// assert_eq!(labels[0], labels[1]); // same cluster
/// assert_eq!(labels[3], NOISE);     // outlier is noise
/// ```
#[derive(Debug, Clone)]
pub struct Dbscan<D: DistanceMetric = Euclidean> {
    /// Epsilon: maximum distance for neighborhood.
    epsilon: f32,
    /// Minimum points for core point classification.
    min_pts: usize,
    /// Distance metric.
    metric: D,
}

/// Labels from DBSCAN clustering.
pub const NOISE: usize = usize::MAX;

// Internal label encoding.
// - UNCLASSIFIED: never assigned yet
// - NOISE_LABEL: visited, but not density-reachable from any core point (may be promoted later)
const UNCLASSIFIED: i32 = -2;
const NOISE_LABEL: i32 = -1;

impl Dbscan<Euclidean> {
    /// Create a new DBSCAN clusterer with the default Euclidean distance.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Maximum distance between two points to be neighbors.
    /// * `min_pts` - Minimum number of points to form a dense region.
    ///
    /// # Typical Values
    ///
    /// - `epsilon`: Often determined by k-distance plot (k = min_pts - 1).
    /// - `min_pts`: 2 * dimension is a common heuristic. Minimum is 3.
    pub fn new(epsilon: f32, min_pts: usize) -> Self {
        Self {
            epsilon,
            min_pts,
            metric: Euclidean,
        }
    }
}

impl<D: DistanceMetric> Dbscan<D> {
    /// Create a new DBSCAN clusterer with a custom distance metric.
    pub fn with_metric(epsilon: f32, min_pts: usize, metric: D) -> Self {
        Self {
            epsilon,
            min_pts,
            metric,
        }
    }

    /// Set epsilon (neighborhood radius).
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set minimum points for core classification.
    pub fn with_min_pts(mut self, min_pts: usize) -> Self {
        self.min_pts = min_pts;
        self
    }

    /// Fit and predict, returning labels where noise is marked as `None`.
    pub fn fit_predict_with_noise(&self, data: &[Vec<f32>]) -> Result<Vec<Option<usize>>> {
        let labels = self.fit_predict(data)?;
        Ok(labels
            .into_iter()
            .map(|l| if l == NOISE { None } else { Some(l) })
            .collect())
    }

    /// Check whether a label is the DBSCAN noise sentinel.
    pub fn is_noise(label: usize) -> bool {
        label == NOISE
    }

    /// Find all neighbors within epsilon (on-demand distance computation).
    fn region_query_ondemand(&self, data: &[Vec<f32>], point_idx: usize) -> Vec<usize> {
        let point = &data[point_idx];
        (0..data.len())
            .filter(|&j| j != point_idx && self.metric.distance(point, &data[j]) <= self.epsilon)
            .collect()
    }

    /// Expand cluster from a core point (on-demand distance computation).
    #[allow(clippy::too_many_arguments)]
    fn expand_cluster_ondemand(
        &self,
        data: &[Vec<f32>],
        point_idx: usize,
        neighbors: &[usize],
        labels: &mut [i32],
        cluster_id: i32,
        visited: &mut [bool],
    ) {
        labels[point_idx] = cluster_id;
        let mut to_process: Vec<usize> = neighbors.to_vec();

        while let Some(neighbor_idx) = to_process.pop() {
            if labels[neighbor_idx] == UNCLASSIFIED || labels[neighbor_idx] == NOISE_LABEL {
                labels[neighbor_idx] = cluster_id;
            }
            if visited[neighbor_idx] {
                continue;
            }
            visited[neighbor_idx] = true;

            let neighbor_neighbors = self.region_query_ondemand(data, neighbor_idx);
            if neighbor_neighbors.len() + 1 >= self.min_pts {
                for nn in neighbor_neighbors {
                    if !visited[nn] {
                        to_process.push(nn);
                    }
                }
            }
        }
    }

    /// Find all neighbors within epsilon using precomputed distances.
    fn region_query_precomputed(&self, dists: &[f32], n: usize, point_idx: usize) -> Vec<usize> {
        let row = point_idx * n;
        (0..n)
            .filter(|&j| j != point_idx && dists[row + j] <= self.epsilon)
            .collect()
    }

    /// Expand cluster from a core point using precomputed distances.
    #[allow(clippy::too_many_arguments)]
    fn expand_cluster_precomputed(
        &self,
        dists: &[f32],
        n: usize,
        point_idx: usize,
        neighbors: &[usize],
        labels: &mut [i32],
        cluster_id: i32,
        visited: &mut [bool],
    ) {
        labels[point_idx] = cluster_id;

        let mut to_process: Vec<usize> = neighbors.to_vec();

        while let Some(neighbor_idx) = to_process.pop() {
            if labels[neighbor_idx] == UNCLASSIFIED || labels[neighbor_idx] == NOISE_LABEL {
                labels[neighbor_idx] = cluster_id;
            }

            if visited[neighbor_idx] {
                continue;
            }
            visited[neighbor_idx] = true;

            let neighbor_neighbors = self.region_query_precomputed(dists, n, neighbor_idx);

            if neighbor_neighbors.len() + 1 >= self.min_pts {
                for nn in neighbor_neighbors {
                    if !visited[nn] {
                        to_process.push(nn);
                    }
                }
            }
        }
    }
}

impl Default for Dbscan<Euclidean> {
    fn default() -> Self {
        Self::new(0.5, 5)
    }
}

impl<D: DistanceMetric> Dbscan<D> {
    /// Fit and return one cluster label per input point.
    ///
    /// Noise points are labeled with the sentinel `NOISE` (`usize::MAX`).
    pub fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        let n = data.len();
        if n == 0 {
            return Err(Error::EmptyInput);
        }

        if self.epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "epsilon",
                message: "must be positive",
            });
        }

        if self.min_pts == 0 {
            return Err(Error::InvalidParameter {
                name: "min_pts",
                message: "must be at least 1",
            });
        }

        // Validate dimensionality.
        let d = data[0].len();
        if d == 0 {
            return Err(Error::InvalidParameter {
                name: "dimension",
                message: "must be at least 1",
            });
        }
        for point in data.iter().skip(1) {
            if point.len() != d {
                return Err(Error::DimensionMismatch {
                    expected: d,
                    found: point.len(),
                });
            }
        }

        util::validate_finite(data)?;

        // Initialize: all points unclassified.
        let mut labels = vec![UNCLASSIFIED; n];
        let mut visited = vec![false; n];
        let mut cluster_id: i32 = 0;

        // Precompute pairwise distance matrix when it fits in memory
        // (n^2 * 4 bytes). Cap at ~256MB to avoid OOM on large datasets.
        // For n <= ~8000 this is always safe; beyond that, fall back to
        // on-demand distance computation.
        const MAX_MATRIX_BYTES: usize = 256 * 1024 * 1024;
        let use_precomputed = (n as u64) * (n as u64) * 4 <= MAX_MATRIX_BYTES as u64;

        if use_precomputed {
            let mut dists = vec![0.0f32; n * n];
            for i in 0..n {
                for j in (i + 1)..n {
                    let d_val = self.metric.distance(&data[i], &data[j]);
                    dists[i * n + j] = d_val;
                    dists[j * n + i] = d_val;
                }
            }

            for point_idx in 0..n {
                if visited[point_idx] {
                    continue;
                }
                visited[point_idx] = true;

                let neighbors = self.region_query_precomputed(&dists, n, point_idx);

                if neighbors.len() + 1 < self.min_pts {
                    labels[point_idx] = NOISE_LABEL;
                    continue;
                }

                self.expand_cluster_precomputed(
                    &dists,
                    n,
                    point_idx,
                    &neighbors,
                    &mut labels,
                    cluster_id,
                    &mut visited,
                );
                cluster_id += 1;
            }
        } else {
            // Large dataset: compute distances on demand.
            for point_idx in 0..n {
                if visited[point_idx] {
                    continue;
                }
                visited[point_idx] = true;

                let neighbors = self.region_query_ondemand(data, point_idx);

                if neighbors.len() + 1 < self.min_pts {
                    labels[point_idx] = NOISE_LABEL;
                    continue;
                }

                self.expand_cluster_ondemand(
                    data,
                    point_idx,
                    &neighbors,
                    &mut labels,
                    cluster_id,
                    &mut visited,
                );
                cluster_id += 1;
            }
        }

        // Convert internal labels to the public `usize` representation.
        let mut out: Vec<usize> = Vec::with_capacity(n);
        for l in labels {
            if l >= 0 {
                out.push(l as usize);
            } else {
                out.push(NOISE);
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn test_dbscan_two_clusters() {
        // Two well-separated clusters
        let data = vec![
            // Cluster 1: around (0, 0)
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            // Cluster 2: around (5, 5)
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
            vec![5.1, 5.1],
            vec![5.05, 5.05],
        ];

        let dbscan = Dbscan::new(0.3, 3);
        let labels = dbscan.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), 10);

        // First 5 should be in same cluster
        let cluster1 = labels[0];
        for label in &labels[1..5] {
            assert_eq!(*label, cluster1);
        }

        // Last 5 should be in same cluster
        let cluster2 = labels[5];
        for label in &labels[6..10] {
            assert_eq!(*label, cluster2);
        }

        // Two clusters should be different
        assert_ne!(cluster1, cluster2);
    }

    #[test]
    fn test_dbscan_with_noise() {
        // Two clusters plus an outlier
        let data = vec![
            // Cluster 1
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            // Outlier
            vec![100.0, 100.0],
            // Cluster 2
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
            vec![5.1, 5.1],
        ];

        let dbscan = Dbscan::new(0.3, 3);
        let labels = dbscan.fit_predict_with_noise(&data).unwrap();

        assert_eq!(labels.len(), 9);

        // Point 4 (outlier) should be noise
        assert!(labels[4].is_none());

        // Others should have cluster assignments
        for (i, label) in labels.iter().enumerate() {
            if i != 4 {
                assert!(label.is_some());
            }
        }
    }

    #[test]
    fn test_dbscan_fit_predict_uses_noise_sentinel() {
        // Same setup as `test_dbscan_with_noise`, but exercise `fit_predict`.
        let data = vec![
            // Cluster 1
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            // Outlier
            vec![100.0, 100.0],
            // Cluster 2
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
            vec![5.1, 5.1],
        ];

        let dbscan = Dbscan::new(0.3, 3);
        let labels = dbscan.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), 9);
        assert_eq!(labels[4], NOISE);
        assert!(Dbscan::<Euclidean>::is_noise(labels[4]));
    }

    #[test]
    fn test_dbscan_all_noise() {
        // Points too far apart
        let data = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 10.0],
        ];

        let dbscan = Dbscan::new(0.5, 3);
        let labels = dbscan.fit_predict_with_noise(&data).unwrap();

        // All should be noise
        for label in labels {
            assert!(label.is_none());
        }
    }

    #[test]
    fn test_dbscan_all_one_cluster() {
        // All points close together
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
        ];

        let dbscan = Dbscan::new(0.5, 2);
        let labels = dbscan.fit_predict(&data).unwrap();

        // All in same cluster
        let cluster = labels[0];
        for label in labels {
            assert_eq!(label, cluster);
        }
    }

    #[test]
    fn test_dbscan_empty() {
        let data: Vec<Vec<f32>> = vec![];
        let dbscan = Dbscan::new(0.5, 3);
        let result = dbscan.fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dbscan_invalid_params() {
        let data = vec![vec![0.0, 0.0]];

        // Invalid epsilon
        let dbscan = Dbscan::new(0.0, 3);
        assert!(dbscan.fit_predict(&data).is_err());

        let dbscan = Dbscan::new(-1.0, 3);
        assert!(dbscan.fit_predict(&data).is_err());

        // Invalid min_pts
        let dbscan = Dbscan::new(0.5, 0);
        assert!(dbscan.fit_predict(&data).is_err());
    }

    #[test]
    fn test_dbscan_chain() {
        // Chain of points - DBSCAN should connect them
        let data: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.3, 0.0]).collect();

        let dbscan = Dbscan::new(0.5, 2);
        let labels = dbscan.fit_predict(&data).unwrap();

        // All should be in one cluster (chain is connected)
        let cluster = labels[0];
        for label in labels {
            assert_eq!(label, cluster);
        }
    }

    #[test]
    fn nan_input_rejected() {
        let data = vec![vec![0.0, f32::NAN], vec![1.0, 1.0], vec![2.0, 2.0]];
        let result = Dbscan::new(0.5, 2).fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn inf_input_rejected() {
        let data = vec![vec![0.0, 0.0], vec![1.0, f32::INFINITY], vec![2.0, 2.0]];
        let result = Dbscan::new(0.5, 2).fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn scalar_data_d1() {
        let data = vec![
            vec![0.0],
            vec![0.1],
            vec![0.2],
            vec![10.0],
            vec![10.1],
            vec![10.2],
        ];
        let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_dbscan_with_custom_metric() {
        use crate::cluster::distance::SquaredEuclidean;

        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
            vec![5.1, 5.1],
        ];

        // With squared Euclidean, epsilon needs to be squared too
        let dbscan = Dbscan::with_metric(0.09, 3, SquaredEuclidean);
        let labels = dbscan.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), 8);
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[4]);
    }
}
