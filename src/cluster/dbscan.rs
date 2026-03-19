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
use super::projindex::ProjIndex;
use super::util;
use crate::error::{Error, Result};
use std::collections::HashMap;

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
    /// # Panics
    ///
    /// Panics if `epsilon <= 0.0` or `min_pts == 0`.
    pub fn new(epsilon: f32, min_pts: usize) -> Self {
        assert!(epsilon > 0.0, "epsilon must be positive");
        assert!(min_pts > 0, "min_pts must be at least 1");
        Self {
            epsilon,
            min_pts,
            metric: Euclidean,
        }
    }
}

impl<D: DistanceMetric> Dbscan<D> {
    /// Create a new DBSCAN clusterer with a custom distance metric.
    ///
    /// # Panics
    ///
    /// Panics if `epsilon <= 0.0` or `min_pts == 0`.
    pub fn with_metric(epsilon: f32, min_pts: usize, metric: D) -> Self {
        assert!(epsilon > 0.0, "epsilon must be positive");
        assert!(min_pts > 0, "min_pts must be at least 1");
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

    /// Build a grid-based spatial index with cell size = epsilon.
    /// Points in the same or adjacent cells are neighbor candidates.
    /// Reduces region_query from O(n) to O(neighbors) amortized for
    /// well-distributed data.
    fn build_grid(data: &[Vec<f32>], epsilon: f32) -> HashMap<Vec<i64>, Vec<usize>> {
        let cell_size = epsilon;
        let mut grid: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();
        for (i, point) in data.iter().enumerate() {
            let cell: Vec<i64> = point
                .iter()
                .map(|&x| (x / cell_size).floor() as i64)
                .collect();
            grid.entry(cell).or_default().push(i);
        }
        grid
    }

    /// Region query using the grid index. Only checks points in adjacent cells.
    fn region_query_grid(
        &self,
        data: &[Vec<f32>],
        point_idx: usize,
        grid: &HashMap<Vec<i64>, Vec<usize>>,
    ) -> Vec<usize> {
        let point = &data[point_idx];
        let d = point.len();
        let cell_size = self.epsilon;
        let center_cell: Vec<i64> = point
            .iter()
            .map(|&x| (x / cell_size).floor() as i64)
            .collect();

        // For d dimensions, iterate over all 3^d adjacent cells.
        // For high d this explodes (3^d cells), so cap at d <= 8.
        let mut neighbors = Vec::new();

        if d > 8 {
            // Fall back to brute force for high-d.
            for (j, other) in data.iter().enumerate() {
                if j != point_idx && self.metric.distance(point, other) <= self.epsilon {
                    neighbors.push(j);
                }
            }
            return neighbors;
        }

        let n_adjacent = 3i64.pow(d as u32);
        for offset_idx in 0..n_adjacent {
            let mut cell = center_cell.clone();
            let mut idx = offset_idx;
            for c in cell.iter_mut().take(d) {
                *c += (idx % 3) - 1;
                idx /= 3;
            }
            if let Some(points) = grid.get(&cell) {
                for &j in points {
                    if j != point_idx && self.metric.distance(point, &data[j]) <= self.epsilon {
                        neighbors.push(j);
                    }
                }
            }
        }
        neighbors
    }

    /// Expand cluster using grid-based neighbor lookup.
    #[allow(clippy::too_many_arguments)]
    fn expand_cluster_grid(
        &self,
        data: &[Vec<f32>],
        point_idx: usize,
        neighbors: &[usize],
        labels: &mut [i32],
        cluster_id: i32,
        visited: &mut [bool],
        grid: &HashMap<Vec<i64>, Vec<usize>>,
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

            let neighbor_neighbors = self.region_query_grid(data, neighbor_idx, grid);
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

        // Strategy selection for neighbor queries:
        // 1. Precomputed pairwise matrix: O(1) lookups, O(n^2) memory.
        //    Used when n^2 * 4 <= 512MB (~n <= 11,500).
        // 2. Grid spatial index: O(neighbors) amortized, O(n) memory.
        //    Only effective for low d (d <= 5) where 3^d <= 243 adjacent cells.
        // 3. Brute-force on-demand: O(n) per query, O(1) extra memory.
        //    Fallback for high-d large-n where neither (1) nor (2) helps.
        const MAX_MATRIX_BYTES: usize = 1024 * 1024 * 1024;
        let use_precomputed = (n as u64) * (n as u64) * 4 <= MAX_MATRIX_BYTES as u64;
        // Grid index: O(3^d) adjacent cells per query. Fast for d<=5 (243 cells).
        // VP-tree: O(log n) queries, metric-agnostic. Better for moderate d.
        let use_grid = !use_precomputed && d <= 5;

        if use_precomputed {
            let dists = util::pairwise_distance_matrix(data, &self.metric);

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
        } else if use_grid {
            // Low-d large dataset: grid spatial index (O(3^d) cells per query).
            let grid = Self::build_grid(data, self.epsilon);

            for point_idx in 0..n {
                if visited[point_idx] {
                    continue;
                }
                visited[point_idx] = true;

                let neighbors = self.region_query_grid(data, point_idx, &grid);

                if neighbors.len() + 1 < self.min_pts {
                    labels[point_idx] = NOISE_LABEL;
                    continue;
                }

                self.expand_cluster_grid(
                    data,
                    point_idx,
                    &neighbors,
                    &mut labels,
                    cluster_id,
                    &mut visited,
                    &grid,
                );
                cluster_id += 1;
            }
        } else {
            // Moderate-to-high d, large dataset: random projection index.
            // Projects points onto 12 random unit vectors with sorted arrays.
            // By Cauchy-Schwarz, each projection is a valid filter for range
            // queries. O(n * k) build, O(k * log n + candidates * d) per query.
            // More effective than VP-tree in d > 10 (avoids curse of dimensionality).
            let proj = ProjIndex::new(data, &self.metric, 12);

            for point_idx in 0..n {
                if visited[point_idx] {
                    continue;
                }
                visited[point_idx] = true;

                let mut neighbors = proj.range_query(&data[point_idx], self.epsilon);
                neighbors.retain(|&j| j != point_idx);

                if neighbors.len() + 1 < self.min_pts {
                    labels[point_idx] = NOISE_LABEL;
                    continue;
                }

                // Expand cluster using projection index for neighbor queries.
                labels[point_idx] = cluster_id;
                let mut to_process = neighbors;

                while let Some(neighbor_idx) = to_process.pop() {
                    if labels[neighbor_idx] == UNCLASSIFIED || labels[neighbor_idx] == NOISE_LABEL {
                        labels[neighbor_idx] = cluster_id;
                    }
                    if visited[neighbor_idx] {
                        continue;
                    }
                    visited[neighbor_idx] = true;

                    let mut nn = proj.range_query(&data[neighbor_idx], self.epsilon);
                    nn.retain(|&j| j != neighbor_idx);
                    if nn.len() + 1 >= self.min_pts {
                        for idx in nn {
                            if !visited[idx] {
                                to_process.push(idx);
                            }
                        }
                    }
                }
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
    #[should_panic(expected = "epsilon must be positive")]
    fn invalid_epsilon_zero() {
        Dbscan::new(0.0, 3);
    }

    #[test]
    #[should_panic(expected = "epsilon must be positive")]
    fn invalid_epsilon_negative() {
        Dbscan::new(-1.0, 3);
    }

    #[test]
    #[should_panic(expected = "min_pts must be at least 1")]
    fn invalid_min_pts_zero() {
        Dbscan::new(0.5, 0);
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

    /// Labels must be in 0..n_clusters or NOISE -- no gaps, no out-of-range.
    #[test]
    fn labels_range_property() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![10.2, 10.2],
            vec![50.0, 50.0], // noise
        ];
        let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
        let non_noise: Vec<usize> = labels.iter().copied().filter(|&l| l != NOISE).collect();
        if !non_noise.is_empty() {
            let max_label = *non_noise.iter().max().unwrap();
            // Labels should be contiguous 0..=max_label.
            for l in 0..=max_label {
                assert!(
                    non_noise.contains(&l),
                    "label {l} missing from range 0..={max_label}"
                );
            }
        }
    }

    /// Single point should be labeled as noise (not enough neighbors).
    #[test]
    fn single_point() {
        let data = vec![vec![1.0, 2.0]];
        let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], NOISE);
    }

    /// d >> n: high-dimensional data with few points.
    #[test]
    fn high_dim_few_points() {
        let data = vec![
            vec![0.0; 100],
            {
                let mut v = vec![0.0; 100];
                v[0] = 0.01;
                v
            },
            vec![10.0; 100],
        ];
        let labels = Dbscan::new(1.0, 2).fit_predict(&data).unwrap();
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0], labels[1]); // close in L2
    }

    /// eps=very_large should put everything in one cluster (if min_pts <= n).
    #[test]
    fn eps_infinity_one_cluster() {
        let data = vec![vec![0.0, 0.0], vec![100.0, 100.0], vec![200.0, 200.0]];
        let labels = Dbscan::new(1000.0, 2).fit_predict(&data).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_ne!(labels[0], NOISE);
    }

    /// min_pts=1 means every point is core: no noise.
    #[test]
    fn min_pts_1_no_noise() {
        let data = vec![vec![0.0, 0.0], vec![100.0, 100.0], vec![200.0, 200.0]];
        let labels = Dbscan::new(0.01, 1).fit_predict(&data).unwrap();
        for &l in &labels {
            assert_ne!(l, NOISE, "min_pts=1 should produce no noise");
        }
    }

    /// Points at exactly epsilon distance should be neighbors (inclusive).
    #[test]
    fn eps_boundary_inclusive() {
        // Two points exactly 1.0 apart (Euclidean).
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let labels = Dbscan::new(1.0, 2).fit_predict(&data).unwrap();
        assert_eq!(
            labels[0], labels[1],
            "points at exactly eps should be neighbors"
        );
        assert_ne!(labels[0], NOISE);
    }

    /// Duplicate points must always be in the same cluster.
    #[test]
    fn duplicate_points_same_cluster() {
        let data = vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![10.0, 10.0],
            vec![10.0, 10.0],
        ];
        let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
    }

    /// Chain of points: density-connectivity is transitive.
    #[test]
    fn chain_density_connected() {
        // Points along x-axis, each 0.3 apart. With eps=0.5, min_pts=2,
        // each is within eps of its neighbor -> one connected chain.
        let data: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.3, 0.0]).collect();
        let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
        let first = labels[0];
        for (i, &l) in labels.iter().enumerate() {
            assert_eq!(l, first, "point {i} should be in same cluster as point 0");
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_data(max_n: usize, d: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
        proptest::collection::vec(proptest::collection::vec(-10.0f32..10.0, d..=d), 2..=max_n)
    }

    proptest! {
        /// Labels must be valid: either NOISE or in [0, n_clusters).
        #[test]
        fn labels_valid(data in arb_data(30, 3)) {
            let labels = Dbscan::new(1.0, 2).fit_predict(&data).unwrap();
            prop_assert_eq!(labels.len(), data.len());
            let max_cluster = labels.iter().copied().filter(|&l| l != NOISE).max();
            if let Some(max) = max_cluster {
                for &l in &labels {
                    prop_assert!(l == NOISE || l <= max);
                }
            }
        }

        /// DBSCAN is deterministic (no randomness).
        #[test]
        fn deterministic(data in arb_data(20, 2)) {
            let l1 = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
            let l2 = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
            prop_assert_eq!(&l1, &l2, "DBSCAN must be deterministic");
        }

        /// Duplicate points must be in the same cluster.
        #[test]
        fn duplicates_same_cluster(
            base in proptest::collection::vec(-10.0f32..10.0, 3..=3),
        ) {
            let data = vec![base.clone(), base.clone(), vec![100.0, 100.0, 100.0]];
            let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
            prop_assert_eq!(labels[0], labels[1], "duplicates must be in same cluster");
        }

        /// All points must be NOISE when min_samples > n.
        #[test]
        fn all_noise_when_min_samples_gt_n(data in arb_data(10, 2)) {
            let n = data.len();
            let labels = Dbscan::new(1.0, n + 1).fit_predict(&data).unwrap();
            for (i, &l) in labels.iter().enumerate() {
                prop_assert_eq!(l, NOISE, "point {} should be NOISE when min_samples > n", i);
            }
        }

        /// Monotone eps: larger eps means no more noise than smaller eps.
        #[test]
        fn monotone_eps(data in arb_data(15, 2)) {
            let eps1 = 0.5;
            let eps2 = 2.0;
            let labels1 = Dbscan::new(eps1, 2).fit_predict(&data).unwrap();
            let labels2 = Dbscan::new(eps2, 2).fit_predict(&data).unwrap();
            let noise1 = labels1.iter().filter(|&&l| l == NOISE).count();
            let noise2 = labels2.iter().filter(|&&l| l == NOISE).count();
            prop_assert!(
                noise2 <= noise1,
                "larger eps should have <= noise: noise(eps={})={} > noise(eps={})={}",
                eps2, noise2, eps1, noise1
            );
        }

        /// Transitivity: if a and b share a cluster and b and c share a cluster,
        /// then a and c must share that cluster (label consistency).
        #[test]
        fn transitivity(data in arb_data(15, 3)) {
            let labels = Dbscan::new(1.0, 2).fit_predict(&data).unwrap();
            let n = labels.len();
            for a in 0..n {
                if labels[a] == NOISE { continue; }
                for b in (a + 1)..n {
                    if labels[b] != labels[a] { continue; }
                    for c in (b + 1)..n {
                        if labels[c] == labels[b] {
                            prop_assert_eq!(
                                labels[a], labels[c],
                                "transitivity violated: labels[{}]={}, labels[{}]={}, labels[{}]={}",
                                a, labels[a], b, labels[b], c, labels[c]
                            );
                        }
                    }
                }
            }
        }

        /// All-identical points with eps > 0 and min_samples <= n must be in one cluster.
        #[test]
        fn all_identical_points(
            point in proptest::collection::vec(-10.0f32..10.0, 2..=2),
            n in 3usize..10,
        ) {
            let data: Vec<Vec<f32>> = vec![point; n];
            let labels = Dbscan::new(0.1, n).fit_predict(&data).unwrap();
            let first = labels[0];
            prop_assert_ne!(first, NOISE, "identical points should not be noise");
            for (i, &l) in labels.iter().enumerate() {
                prop_assert_eq!(l, first, "point {} should match point 0", i);
            }
        }
    }
}
