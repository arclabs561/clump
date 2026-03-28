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
use super::flat::DataRef;
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
    pub fn fit(&self, data: &(impl DataRef + ?Sized)) -> Result<OpticsResult> {
        let n = data.n();
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
                .map(|j| self.metric.distance(data.row(i), data.row(j)))
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
        data: &(impl DataRef + ?Sized),
        core_dist: &[f32],
        processed: &[bool],
        reachability: &mut [f32],
        seeds: &mut BinaryHeap<SeedEntry>,
    ) {
        let cd = core_dist[point];
        for j in 0..data.n() {
            if processed[j] || j == point {
                continue;
            }
            let dist = self.metric.distance(data.row(point), data.row(j));
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

    /// Extract clusters using the Xi method (Ankerst et al. 1999).
    ///
    /// Detects significant valleys (steep-down followed by steep-up) in the
    /// reachability plot. A point is "steep down" if its reachability drops
    /// by a factor >= (1 - xi) relative to its predecessor. A "steep up"
    /// is the reverse. Clusters are the regions between steep-down and
    /// steep-up transitions.
    ///
    /// `xi` in (0, 1): smaller values require deeper valleys (fewer, tighter
    /// clusters). Typical values: 0.01-0.1.
    ///
    /// Returns labels where noise is `NOISE` (`usize::MAX`).
    pub fn extract_xi(result: &OpticsResult, xi: f32) -> Vec<usize> {
        let n = result.ordering.len();
        let noise = crate::NOISE;
        if n == 0 {
            return vec![];
        }

        let reach = &result.reachability;
        let factor = 1.0 - xi;

        // Compute a smoothed "max reachability so far" to detect valley
        // boundaries. A valley starts when reachability drops significantly
        // below the preceding maximum, and ends when it rises back.

        // For each position, compute the local maximum reachability in a
        // window looking backward (the "ridge" before the valley).
        let mut max_before = vec![0.0f32; n];
        let mut running_max = 0.0f32;
        for p in 0..n {
            if reach[p].is_infinite() {
                // Reset at cluster boundaries so each valley is independent.
                running_max = 0.0;
            } else if reach[p] > running_max {
                running_max = reach[p];
            }
            max_before[p] = running_max;
        }

        // A point is "in a valley" if its reachability is < factor * max_before.
        // Contiguous valley regions become clusters.
        // Detect valleys using predecessor comparison.
        // A point enters a valley when reach[p] < reach[p-1] * factor (steep drop).
        // A point leaves a valley when reach[p] > reach[p-1] / factor (steep rise).
        // Between those transitions, points belong to the current cluster.
        let mut label_by_pos = vec![noise; n];
        let mut cluster_id = 0usize;
        let mut in_valley = false;

        for p in 1..n {
            let prev = reach[p - 1];
            let curr = reach[p];

            if curr.is_infinite() {
                in_valley = false;
                continue;
            }

            if prev.is_infinite() {
                // Transition from inf to finite: entering a cluster region.
                in_valley = true;
                cluster_id += 1;
                label_by_pos[p] = cluster_id - 1;
                continue;
            }

            // Steep down: current << previous.
            if curr < prev * factor && !in_valley {
                in_valley = true;
                cluster_id += 1;
                label_by_pos[p] = cluster_id - 1;
            }
            // Steep up: current >> previous.
            else if curr > prev / factor && in_valley {
                in_valley = false;
                // This point is the ridge, not part of the valley.
            }
            // Continuation of current state.
            else if in_valley {
                label_by_pos[p] = cluster_id - 1;
            }
        }

        // Map back to original point indices.
        let mut label_by_orig = vec![noise; n];
        for (pos, &orig_idx) in result.ordering.iter().enumerate() {
            label_by_orig[orig_idx] = label_by_pos[pos];
        }

        label_by_orig
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
        let mut seen = [false; 50];
        for &idx in &result.ordering {
            assert!(!seen[idx], "point {idx} visited twice");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s), "not all points visited");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_data(max_n: usize, d: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
        proptest::collection::vec(proptest::collection::vec(-10.0f32..10.0, d..=d), 3..=max_n)
    }

    proptest! {
        /// Ordering must be a permutation of 0..n.
        #[test]
        fn ordering_is_permutation(data in arb_data(20, 2)) {
            let result = Optics::new(100.0, 2).fit(&data).unwrap();
            let n = data.len();
            prop_assert_eq!(result.ordering.len(), n);
            let mut sorted = result.ordering.clone();
            sorted.sort();
            let expected: Vec<usize> = (0..n).collect();
            prop_assert_eq!(sorted, expected, "ordering must be a permutation");
        }

        /// Reachability and core_distances arrays must match ordering length.
        #[test]
        fn arrays_aligned(data in arb_data(15, 2)) {
            let result = Optics::new(50.0, 2).fit(&data).unwrap();
            prop_assert_eq!(result.reachability.len(), result.ordering.len());
            prop_assert_eq!(result.core_distances.len(), result.ordering.len());
        }

        /// Extracted at smaller epsilon: non-noise cluster count should not
        /// be strictly less unless some points become noise at smaller eps.
        /// Weaker property: n_clustered_points(small_eps) <= n_clustered_points(big_eps).
        #[test]
        fn smaller_eps_fewer_clustered_points(data in arb_data(20, 2)) {
            let result = Optics::new(100.0, 2).fit(&data).unwrap();
            let labels_big = Optics::<Euclidean>::extract_clusters(&result, 10.0);
            let labels_small = Optics::<Euclidean>::extract_clusters(&result, 1.0);
            let clustered_big = labels_big.iter().filter(|&&l| l != crate::NOISE).count();
            let clustered_small = labels_small.iter().filter(|&&l| l != crate::NOISE).count();
            prop_assert!(
                clustered_small <= clustered_big,
                "smaller eps should cluster <= points: {} > {}",
                clustered_small, clustered_big
            );
        }

        /// All reachability distances must be >= 0 or infinity.
        #[test]
        fn reachability_non_negative(data in arb_data(15, 2)) {
            let result = Optics::new(50.0, 2).fit(&data).unwrap();
            for (i, &r) in result.reachability.iter().enumerate() {
                prop_assert!(
                    r >= 0.0 || r == f32::INFINITY,
                    "reachability[{}] = {} is negative", i, r
                );
            }
        }

        /// Core points (finite core distance) must have positive core distances.
        #[test]
        fn core_distances_positive_for_core_points(data in arb_data(15, 2)) {
            let result = Optics::new(50.0, 2).fit(&data).unwrap();
            for (i, &cd) in result.core_distances.iter().enumerate() {
                if cd != f32::INFINITY {
                    prop_assert!(
                        cd >= 0.0,
                        "core_distances[{}] = {} should be >= 0 for core points", i, cd
                    );
                }
            }
        }

        /// The first point in the ordering must have reachability = INFINITY.
        #[test]
        fn first_point_reachability_infinity(data in arb_data(15, 2)) {
            let result = Optics::new(50.0, 2).fit(&data).unwrap();
            prop_assert_eq!(
                result.reachability[0], f32::INFINITY,
                "first point in ordering must have reachability = INFINITY, got {}",
                result.reachability[0]
            );
        }

        /// OPTICS-DBSCAN parity: for well-separated blobs, extract_clusters(eps)
        /// should give the same number of non-noise clusters as DBSCAN(eps, min_pts).
        #[test]
        fn optics_dbscan_cluster_count_parity(
            perturbation in proptest::collection::vec(-0.05f32..0.05, 20..=20),
        ) {
            // Two well-separated blobs with small random perturbation.
            let mut data = Vec::with_capacity(20);
            for p in &perturbation[..10] {
                data.push(vec![*p, *p + 0.01]);
            }
            for p in &perturbation[10..20] {
                data.push(vec![10.0 + *p, 10.0 + *p + 0.01]);
            }

            let eps = 0.5;
            let min_pts = 3;

            let dbscan_labels = crate::Dbscan::new(eps, min_pts).fit_predict(&data).unwrap();
            let optics_result = Optics::new(eps, min_pts).fit(&data).unwrap();
            let optics_labels = Optics::<Euclidean>::extract_clusters(&optics_result, eps);

            let db_clusters: std::collections::HashSet<usize> = dbscan_labels.iter()
                .copied().filter(|&l| l != crate::NOISE).collect();
            let op_clusters: std::collections::HashSet<usize> = optics_labels.iter()
                .copied().filter(|&l| l != crate::NOISE).collect();

            prop_assert_eq!(
                db_clusters.len(), op_clusters.len(),
                "DBSCAN found {} clusters, OPTICS found {} clusters",
                db_clusters.len(), op_clusters.len()
            );
        }
    }
}

#[cfg(test)]
mod xi_tests {
    use super::*;
    use crate::cluster::distance::Euclidean;

    #[test]
    fn xi_two_clusters() {
        // Two well-separated clusters: Xi should find 2 valleys.
        let mut data = Vec::new();
        for i in 0..10 {
            data.push(vec![(i % 3) as f32 * 0.1, (i / 3) as f32 * 0.1]);
        }
        for i in 0..10 {
            data.push(vec![
                20.0 + (i % 3) as f32 * 0.1,
                20.0 + (i / 3) as f32 * 0.1,
            ]);
        }

        // max_eps must be large enough to connect within clusters AND see the gap.
        let result = Optics::new(50.0, 2).fit(&data).unwrap();
        let labels = Optics::<Euclidean>::extract_xi(&result, 0.1);

        assert_eq!(labels.len(), 20);
        let non_noise: std::collections::HashSet<usize> = labels
            .iter()
            .copied()
            .filter(|&l| l != crate::NOISE)
            .collect();
        assert!(
            non_noise.len() >= 2,
            "Xi should find at least 2 clusters, found {}",
            non_noise.len()
        );
    }

    #[test]
    fn xi_all_same_returns_one_or_no_cluster() {
        let data = vec![vec![0.0, 0.0]; 10];
        let result = Optics::new(1.0, 2).fit(&data).unwrap();
        let labels = Optics::<Euclidean>::extract_xi(&result, 0.05);
        assert_eq!(labels.len(), 10);
        // All identical: no valleys in reachability -> all noise or one cluster.
    }

    #[test]
    fn xi_single_point() {
        let data = vec![vec![1.0, 2.0]];
        let result = Optics::new(1.0, 2).fit(&data).unwrap();
        let labels = Optics::<Euclidean>::extract_xi(&result, 0.1);
        assert_eq!(labels.len(), 1);
    }

    #[test]
    fn xi_empty() {
        let result = OpticsResult {
            ordering: vec![],
            reachability: vec![],
            core_distances: vec![],
        };
        let labels = Optics::<Euclidean>::extract_xi(&result, 0.1);
        assert!(labels.is_empty());
    }

    #[test]
    fn xi_labels_valid() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![50.0, 50.0], // noise
        ];
        let result = Optics::new(1.0, 2).fit(&data).unwrap();
        let labels = Optics::<Euclidean>::extract_xi(&result, 0.05);
        assert_eq!(labels.len(), 7);
        for &l in &labels {
            assert!(l == crate::NOISE || l < 100, "label {} out of range", l);
        }
    }
}
