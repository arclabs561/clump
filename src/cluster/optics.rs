//! OPTICS: Ordering Points To Identify the Clustering Structure.
//!
//! OPTICS (Ankerst et al. 1999) is a density-based algorithm that produces a
//! reachability plot -- an ordering of points where valleys correspond to clusters.
//! Unlike DBSCAN, it does not require a fixed epsilon; instead, it uses a maximum
//! epsilon and produces a hierarchical density structure.
//!
//! # How It Works
//!
//! 1. Pick an unprocessed point, find its epsilon-neighborhood
//! 2. If it's a core point (>= min_pts neighbors), compute reachability distances
//!    for all neighbors and add them to a priority queue (ordered seeds)
//! 3. Process the closest seed next, update reachability distances
//! 4. The processing order + reachability distances form the reachability plot
//!
//! Clusters can be extracted by cutting the reachability plot at a threshold
//! (DBSCAN-like) or using the Xi method for automatic detection.

use super::distance::{DistanceMetric, Euclidean};
use super::util;
use crate::error::{Error, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Sentinel for undefined reachability distance.
const UNDEFINED: f32 = f32::INFINITY;

/// Entry in the priority queue (ordered seeds).
#[derive(Clone)]
struct SeedEntry {
    index: usize,
    reachability: f32,
}

impl PartialEq for SeedEntry {
    fn eq(&self, other: &Self) -> bool {
        self.reachability == other.reachability
    }
}
impl Eq for SeedEntry {}

impl PartialOrd for SeedEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SeedEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior.
        other
            .reachability
            .partial_cmp(&self.reachability)
            .unwrap_or(Ordering::Equal)
    }
}

/// OPTICS clustering algorithm, generic over a distance metric.
///
/// Produces a reachability ordering that can be cut at any threshold
/// to extract clusters (like DBSCAN but without committing to one epsilon).
#[derive(Debug, Clone)]
pub struct Optics<D: DistanceMetric = Euclidean> {
    /// Maximum neighborhood radius.
    max_epsilon: f32,
    /// Minimum points to form a core point.
    min_pts: usize,
    /// Distance metric.
    metric: D,
}

/// Result of OPTICS fitting.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OpticsResult {
    /// Processing order (indices into original data).
    pub ordering: Vec<usize>,
    /// Reachability distance for each point in ordering.
    /// `f32::INFINITY` for the first point of each cluster seed.
    pub reachability: Vec<f32>,
    /// Core distance for each point (INFINITY if not a core point).
    pub core_distances: Vec<f32>,
}

impl Optics<Euclidean> {
    /// Create a new OPTICS clusterer with the default Euclidean distance.
    ///
    /// # Panics
    ///
    /// Panics if `max_epsilon <= 0.0` or `min_pts == 0`.
    pub fn new(max_epsilon: f32, min_pts: usize) -> Self {
        assert!(max_epsilon > 0.0, "max_epsilon must be positive");
        assert!(min_pts > 0, "min_pts must be at least 1");
        Self {
            max_epsilon,
            min_pts,
            metric: Euclidean,
        }
    }
}

impl<D: DistanceMetric> Optics<D> {
    /// Create a new OPTICS clusterer with a custom distance metric.
    pub fn with_metric(max_epsilon: f32, min_pts: usize, metric: D) -> Self {
        assert!(max_epsilon > 0.0, "max_epsilon must be positive");
        assert!(min_pts > 0, "min_pts must be at least 1");
        Self {
            max_epsilon,
            min_pts,
            metric,
        }
    }

    /// Compute the OPTICS ordering and reachability plot.
    pub fn fit(&self, data: &[Vec<f32>]) -> Result<OpticsResult> {
        let n = data.len();
        if n == 0 {
            return Err(Error::EmptyInput);
        }

        util::validate_finite(data)?;

        let mut processed = vec![false; n];
        let mut reachability = vec![UNDEFINED; n];
        let mut core_dist = vec![UNDEFINED; n];
        let mut ordering = Vec::with_capacity(n);

        // Precompute core distances. For each point, find the distance to its
        // (min_pts-1)-th nearest neighbor within max_epsilon.
        // Uses partial sort (select_nth_unstable) instead of full sort.
        let compute_core = |i: usize| -> f32 {
            let mut neighbor_dists: Vec<f32> = (0..n)
                .filter(|&j| j != i)
                .map(|j| self.metric.distance(&data[i], &data[j]))
                .filter(|&d| d <= self.max_epsilon)
                .collect();
            if neighbor_dists.len() + 1 >= self.min_pts {
                let k = self.min_pts - 2; // -2: exclude self, 0-indexed
                neighbor_dists.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
                neighbor_dists[k]
            } else {
                UNDEFINED
            }
        };

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            core_dist = (0..n).into_par_iter().map(compute_core).collect();
        }
        #[cfg(not(feature = "parallel"))]
        for (i, cd) in core_dist.iter_mut().enumerate() {
            *cd = compute_core(i);
        }

        for i in 0..n {
            if processed[i] {
                continue;
            }
            processed[i] = true;
            ordering.push(i);

            if core_dist[i] == UNDEFINED {
                continue; // Not a core point.
            }

            // Expand from this core point using ordered seeds.
            let mut seeds = BinaryHeap::new();
            self.update_seeds(
                i,
                data,
                &core_dist,
                &processed,
                &mut reachability,
                &mut seeds,
            );

            while let Some(seed) = seeds.pop() {
                if processed[seed.index] {
                    continue;
                }
                processed[seed.index] = true;
                ordering.push(seed.index);

                if core_dist[seed.index] != UNDEFINED {
                    self.update_seeds(
                        seed.index,
                        data,
                        &core_dist,
                        &processed,
                        &mut reachability,
                        &mut seeds,
                    );
                }
            }
        }

        // Build reachability in ordering order.
        let reach_ordered: Vec<f32> = ordering.iter().map(|&i| reachability[i]).collect();
        let core_ordered: Vec<f32> = ordering.iter().map(|&i| core_dist[i]).collect();

        Ok(OpticsResult {
            ordering,
            reachability: reach_ordered,
            core_distances: core_ordered,
        })
    }

    fn update_seeds(
        &self,
        point: usize,
        data: &[Vec<f32>],
        core_dist: &[f32],
        processed: &[bool],
        reachability: &mut [f32],
        seeds: &mut BinaryHeap<SeedEntry>,
    ) {
        let cd = core_dist[point];
        for j in 0..data.len() {
            if processed[j] || j == point {
                continue;
            }
            let dist = self.metric.distance(&data[point], &data[j]);
            if dist > self.max_epsilon {
                continue;
            }
            let new_reach = dist.max(cd);
            if new_reach < reachability[j] {
                reachability[j] = new_reach;
                seeds.push(SeedEntry {
                    index: j,
                    reachability: new_reach,
                });
            }
        }
    }

    /// Extract DBSCAN-like clusters from the reachability plot at a given epsilon.
    ///
    /// Points with reachability > epsilon start new clusters or become noise.
    pub fn extract_clusters(result: &OpticsResult, epsilon: f32) -> Vec<usize> {
        let n = result.ordering.len();
        let noise = crate::NOISE;
        let mut cluster_id = 0usize;
        let mut label_by_orig = vec![noise; n];

        for (pos, &orig_idx) in result.ordering.iter().enumerate() {
            if result.reachability[pos] > epsilon {
                // Check if this point is a core point that starts a new cluster.
                if result.core_distances[pos] <= epsilon {
                    label_by_orig[orig_idx] = cluster_id;
                    cluster_id += 1;
                }
                // else: noise
            } else {
                // Reachability <= epsilon: belongs to the current cluster.
                label_by_orig[orig_idx] = if cluster_id > 0 { cluster_id - 1 } else { 0 };
            }
        }

        label_by_orig
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_optics() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ];
        let result = Optics::new(1.0, 2).fit(&data).unwrap();
        assert_eq!(result.ordering.len(), 6);
        assert_eq!(result.reachability.len(), 6);

        // Extract clusters at epsilon=0.5.
        let labels = Optics::<Euclidean>::extract_clusters(&result, 0.5);
        assert_eq!(labels.len(), 6);
        // First 3 should be same cluster, last 3 another.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn optics_single_point() {
        let data = vec![vec![1.0, 2.0]];
        let result = Optics::new(1.0, 2).fit(&data).unwrap();
        assert_eq!(result.ordering.len(), 1);
    }

    #[test]
    fn optics_empty_data() {
        let data: Vec<Vec<f32>> = vec![];
        assert!(Optics::new(1.0, 2).fit(&data).is_err());
    }

    #[test]
    #[should_panic(expected = "max_epsilon must be positive")]
    fn optics_invalid_epsilon() {
        Optics::new(0.0, 2);
    }

    /// OPTICS extract_clusters at epsilon should produce the same cluster
    /// structure as DBSCAN at the same epsilon (cross-algorithm consistency).
    #[test]
    fn optics_dbscan_consistency() {
        use crate::Dbscan;

        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![10.1, 10.1],
            vec![10.05, 10.05],
        ];

        let eps = 0.3;
        let min_pts = 3;

        let dbscan_labels = Dbscan::new(eps, min_pts).fit_predict(&data).unwrap();
        let optics_result = Optics::new(eps, min_pts).fit(&data).unwrap();
        let optics_labels = Optics::<Euclidean>::extract_clusters(&optics_result, eps);

        // Both should find the same cluster structure (same/different relationships).
        for i in 0..data.len() {
            for j in (i + 1)..data.len() {
                let db_same =
                    dbscan_labels[i] == dbscan_labels[j] && dbscan_labels[i] != crate::NOISE;
                let op_same =
                    optics_labels[i] == optics_labels[j] && optics_labels[i] != crate::NOISE;
                assert_eq!(
                    db_same, op_same,
                    "DBSCAN and OPTICS disagree on points {i},{j}: \
                     DBSCAN labels=({},{}), OPTICS labels=({},{})",
                    dbscan_labels[i], dbscan_labels[j], optics_labels[i], optics_labels[j]
                );
            }
        }
    }

    /// Reachability ordering should visit all points exactly once.
    #[test]
    fn optics_ordering_complete() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<Vec<f32>> = (0..50)
            .map(|_| vec![rng.random::<f32>() * 10.0, rng.random::<f32>() * 10.0])
            .collect();
        let result = Optics::new(5.0, 3).fit(&data).unwrap();
        assert_eq!(result.ordering.len(), 50);
        let mut seen = vec![false; 50];
        for &idx in &result.ordering {
            assert!(!seen[idx], "point {idx} visited twice");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s), "not all points visited");
    }
}
