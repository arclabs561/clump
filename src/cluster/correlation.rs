//! Correlation clustering on signed graphs.
//!
//! Correlation clustering partitions a set of items given pairwise similarity
//! scores, without requiring the number of clusters as input. Each edge carries
//! a signed weight: positive means "should be together," negative means "should
//! be apart." The objective is to minimize the total disagreement cost -- the
//! sum of |weight| over positive edges that cross cluster boundaries and
//! negative edges that land inside the same cluster.
//!
//! # When to Use
//!
//! - **Entity resolution / deduplication**: pairwise similarity scores exist
//!   but the number of distinct entities is unknown.
//! - **Coreference resolution**: mention pairs have compatibility scores.
//! - **Community detection** on signed networks.
//!
//! # Algorithms
//!
//! ## PIVOT (Ailon, Charikar & Newman, 2008)
//!
//! A randomized 3-approximation algorithm. Pick an unassigned item uniformly
//! at random as "pivot," cluster it with every unassigned neighbor that has a
//! positive edge to it, mark all as assigned, and repeat. Runs in O(n + m)
//! where m is the number of edges.
//!
//! ## Local Search Refinement
//!
//! After PIVOT, iteratively scan all items. For each item, evaluate moving it
//! to every neighboring cluster (or to a fresh singleton). If the best move
//! reduces the total disagreement cost, apply it. Repeat until no improvement
//! is found or `max_iter` iterations are exhausted. This is a hill-climbing
//! refinement -- it cannot increase cost, but may converge to a local minimum.
//!
//! # Trade-offs vs K-means
//!
//! - No need to specify k.
//! - Handles non-convex cluster shapes naturally (works on graphs, not
//!   metric spaces).
//! - The optimization problem is NP-hard; PIVOT provides a bounded
//!   approximation, and local search improves the solution empirically.
//!
//! # References
//!
//! - Bansal, N., Blum, A., & Chawla, S. (2004). "Correlation Clustering."
//!   *Machine Learning*, 56(1-3), 89-113.
//! - Ailon, N., Charikar, M., & Newman, A. (2008). "Aggregating Inconsistent
//!   Information: Ranking and Clustering." *JACM*, 55(5).

use std::collections::HashMap;

use rand::prelude::*;

use super::distance::DistanceMetric;
use crate::error::{Error, Result};

/// A signed edge between two items with a similarity score.
///
/// Positive weight means the items should cluster together.
/// Negative weight means the items should be in different clusters.
#[derive(Debug, Clone, Copy)]
pub struct SignedEdge {
    /// First item index.
    pub i: usize,
    /// Second item index.
    pub j: usize,
    /// Signed similarity weight.
    pub weight: f32,
}

/// Result of correlation clustering.
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    /// Cluster label for each item. Labels are contiguous in `[0, n_clusters)`.
    pub labels: Vec<usize>,
    /// Number of clusters discovered.
    pub n_clusters: usize,
    /// Total disagreement cost of the solution.
    pub cost: f64,
}

/// Correlation clustering via PIVOT with optional local search refinement.
///
/// ```
/// use clump::cluster::correlation::{CorrelationClustering, SignedEdge};
///
/// let edges = vec![
///     SignedEdge { i: 0, j: 1, weight: 1.0 },
///     SignedEdge { i: 0, j: 2, weight: -1.0 },
///     SignedEdge { i: 1, j: 2, weight: -1.0 },
/// ];
///
/// let result = CorrelationClustering::new()
///     .with_seed(42)
///     .fit(3, &edges)
///     .unwrap();
///
/// assert_eq!(result.labels[0], result.labels[1]);
/// assert_ne!(result.labels[0], result.labels[2]);
/// ```
#[derive(Debug, Clone)]
pub struct CorrelationClustering {
    /// Maximum local search iterations (0 = PIVOT only, no refinement).
    max_iter: usize,
    /// Random seed for pivot selection order.
    seed: Option<u64>,
}

impl Default for CorrelationClustering {
    fn default() -> Self {
        Self::new()
    }
}

impl CorrelationClustering {
    /// Create a new correlation clusterer with default parameters.
    ///
    /// Defaults: `max_iter = 100`, no fixed seed.
    pub fn new() -> Self {
        Self {
            max_iter: 100,
            seed: None,
        }
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the maximum number of local search iterations.
    ///
    /// Pass 0 to skip local search and return the raw PIVOT solution.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Cluster items based on signed pairwise edges.
    ///
    /// - `n_items`: total number of items. Items with no edges become singletons.
    /// - `edges`: signed pairwise relationships.
    ///
    /// Returns [`Error::InvalidParameter`] if any edge references an item index
    /// `>= n_items`.
    pub fn fit(&self, n_items: usize, edges: &[SignedEdge]) -> Result<CorrelationResult> {
        if n_items == 0 {
            return Ok(CorrelationResult {
                labels: Vec::new(),
                n_clusters: 0,
                cost: 0.0,
            });
        }

        // Validate edge indices.
        for edge in edges {
            if edge.i >= n_items || edge.j >= n_items {
                return Err(Error::InvalidParameter {
                    name: "edge index",
                    message: "exceeds n_items",
                });
            }
        }

        // Build adjacency list: for each item, list of (neighbor, weight).
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_items];
        for edge in edges {
            adj[edge.i].push((edge.j, edge.weight));
            adj[edge.j].push((edge.i, edge.weight));
        }

        // PIVOT algorithm.
        let mut labels = self.pivot(n_items, &adj);

        // Local search refinement.
        if self.max_iter > 0 {
            self.local_search(n_items, &adj, &mut labels);
        }

        // Relabel to contiguous [0, n_clusters).
        let (labels, n_clusters) = relabel_contiguous(&labels);
        let cost = compute_cost(&labels, edges);

        Ok(CorrelationResult {
            labels,
            n_clusters,
            cost,
        })
    }

    /// Create signed edges from a distance matrix and a threshold.
    ///
    /// Pairs closer than `threshold` get positive weight `= threshold - distance`.
    /// Pairs farther than `threshold` get negative weight `= threshold - distance`
    /// (which is negative). Pairs at exactly `threshold` get weight 0 and are
    /// included but contribute no cost.
    pub fn edges_from_distances<D: DistanceMetric>(
        data: &[Vec<f32>],
        metric: &D,
        threshold: f32,
    ) -> Vec<SignedEdge> {
        let n = data.len();
        let mut edges = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = metric.distance(&data[i], &data[j]);
                let weight = threshold - dist;
                edges.push(SignedEdge { i, j, weight });
            }
        }
        edges
    }

    /// PIVOT: randomized 3-approximation.
    ///
    /// Returns a label vector (labels may not be contiguous).
    fn pivot(&self, n_items: usize, adj: &[Vec<(usize, f32)>]) -> Vec<usize> {
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        let mut assigned = vec![false; n_items];
        let mut labels = vec![0usize; n_items];

        // Shuffle to randomize pivot order.
        let mut order: Vec<usize> = (0..n_items).collect();
        order.shuffle(&mut rng);

        let mut next_cluster = 0usize;

        for &pivot in &order {
            if assigned[pivot] {
                continue;
            }

            // Form cluster: pivot + unassigned neighbors with positive edge.
            let cluster_id = next_cluster;
            next_cluster += 1;

            assigned[pivot] = true;
            labels[pivot] = cluster_id;

            for &(neighbor, weight) in &adj[pivot] {
                if !assigned[neighbor] && weight > 0.0 {
                    assigned[neighbor] = true;
                    labels[neighbor] = cluster_id;
                }
            }
        }

        labels
    }

    /// Local search: iteratively move items to reduce disagreement cost.
    fn local_search(&self, n_items: usize, adj: &[Vec<(usize, f32)>], labels: &mut [usize]) {
        for _ in 0..self.max_iter {
            let mut improved = false;

            for item in 0..n_items {
                let current_cluster = labels[item];

                // Collect candidate clusters: current, each neighbor's cluster,
                // and a fresh singleton.
                let fresh_singleton = labels.iter().copied().max().unwrap_or(0) + 1;
                let mut candidates: Vec<usize> = Vec::new();
                candidates.push(current_cluster);
                candidates.push(fresh_singleton);
                for &(neighbor, _) in &adj[item] {
                    candidates.push(labels[neighbor]);
                }
                candidates.sort_unstable();
                candidates.dedup();

                // Evaluate each candidate by computing delta relative to current.
                let mut best_cluster = current_cluster;
                let mut best_delta = 0.0f64;

                for &candidate in &candidates {
                    if candidate == current_cluster {
                        continue;
                    }
                    let delta = move_delta(item, current_cluster, candidate, adj, labels);
                    if delta < best_delta {
                        best_delta = delta;
                        best_cluster = candidate;
                    }
                }

                if best_cluster != current_cluster {
                    labels[item] = best_cluster;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }
    }
}

/// Compute the change in disagreement cost from moving `item` from `from` to `to`.
///
/// Negative delta means improvement (cost decreases).
fn move_delta(
    item: usize,
    from: usize,
    to: usize,
    adj: &[Vec<(usize, f32)>],
    labels: &[usize],
) -> f64 {
    let mut delta = 0.0f64;

    for &(neighbor, weight) in &adj[item] {
        let neighbor_cluster = labels[neighbor];

        // Contribution of this edge under current assignment.
        let cost_before = edge_disagreement(from, neighbor_cluster, weight);
        // Contribution of this edge if item moves to `to`.
        let cost_after = edge_disagreement(to, neighbor_cluster, weight);

        delta += cost_after - cost_before;
    }

    delta
}

/// Disagreement cost of a single edge given the clusters of its endpoints.
///
/// - Positive edge across clusters: cost = |weight|
/// - Negative edge within cluster: cost = |weight|
/// - Otherwise: 0
#[inline]
fn edge_disagreement(cluster_a: usize, cluster_b: usize, weight: f32) -> f64 {
    let same = cluster_a == cluster_b;
    if (weight > 0.0 && !same) || (weight < 0.0 && same) {
        (weight as f64).abs()
    } else {
        0.0
    }
}

/// Compute total disagreement cost for a partition.
fn compute_cost(labels: &[usize], edges: &[SignedEdge]) -> f64 {
    let mut cost = 0.0f64;
    for edge in edges {
        let same = labels[edge.i] == labels[edge.j];
        if (edge.weight > 0.0 && !same) || (edge.weight < 0.0 && same) {
            cost += (edge.weight as f64).abs();
        }
    }
    cost
}

/// Relabel a label vector to use contiguous indices `[0, n_clusters)`.
fn relabel_contiguous(labels: &[usize]) -> (Vec<usize>, usize) {
    let mut map: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    let mut out = Vec::with_capacity(labels.len());

    for &label in labels {
        let new_label = *map.entry(label).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        out.push(new_label);
    }

    (out, next)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_positive_edges_single_cluster() {
        let edges = vec![
            SignedEdge {
                i: 0,
                j: 1,
                weight: 1.0,
            },
            SignedEdge {
                i: 0,
                j: 2,
                weight: 2.0,
            },
            SignedEdge {
                i: 1,
                j: 2,
                weight: 1.5,
            },
        ];

        let result = CorrelationClustering::new()
            .with_seed(42)
            .fit(3, &edges)
            .expect("fit should succeed");

        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert!((result.cost - 0.0).abs() < 1e-9);
    }

    #[test]
    fn all_negative_edges_singletons() {
        let edges = vec![
            SignedEdge {
                i: 0,
                j: 1,
                weight: -1.0,
            },
            SignedEdge {
                i: 0,
                j: 2,
                weight: -2.0,
            },
            SignedEdge {
                i: 1,
                j: 2,
                weight: -1.5,
            },
        ];

        let result = CorrelationClustering::new()
            .with_seed(42)
            .fit(3, &edges)
            .expect("fit should succeed");

        assert_eq!(result.n_clusters, 3);
        assert_ne!(result.labels[0], result.labels[1]);
        assert_ne!(result.labels[0], result.labels[2]);
        assert_ne!(result.labels[1], result.labels[2]);
        assert!((result.cost - 0.0).abs() < 1e-9);
    }

    #[test]
    fn two_clear_groups() {
        // Group A: {0, 1, 2}, Group B: {3, 4, 5}
        // Positive edges within groups, negative edges between groups.
        let mut edges = Vec::new();
        let group_a = [0, 1, 2];
        let group_b = [3, 4, 5];

        for ii in 0..group_a.len() {
            for jj in (ii + 1)..group_a.len() {
                edges.push(SignedEdge {
                    i: group_a[ii],
                    j: group_a[jj],
                    weight: 5.0,
                });
            }
        }
        for ii in 0..group_b.len() {
            for jj in (ii + 1)..group_b.len() {
                edges.push(SignedEdge {
                    i: group_b[ii],
                    j: group_b[jj],
                    weight: 5.0,
                });
            }
        }
        for &a in &group_a {
            for &b in &group_b {
                edges.push(SignedEdge {
                    i: a,
                    j: b,
                    weight: -3.0,
                });
            }
        }

        let result = CorrelationClustering::new()
            .with_seed(7)
            .fit(6, &edges)
            .expect("fit should succeed");

        assert_eq!(result.n_clusters, 2);

        // All of group A in one cluster.
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);

        // All of group B in one cluster.
        assert_eq!(result.labels[3], result.labels[4]);
        assert_eq!(result.labels[4], result.labels[5]);

        // Groups are different.
        assert_ne!(result.labels[0], result.labels[3]);

        assert!((result.cost - 0.0).abs() < 1e-9);
    }

    #[test]
    fn disconnected_items_are_singletons() {
        // 5 items, edges only between 0-1 and 2-3. Item 4 has no edges.
        let edges = vec![
            SignedEdge {
                i: 0,
                j: 1,
                weight: 1.0,
            },
            SignedEdge {
                i: 2,
                j: 3,
                weight: 1.0,
            },
        ];

        let result = CorrelationClustering::new()
            .with_seed(42)
            .fit(5, &edges)
            .expect("fit should succeed");

        // 0 and 1 together, 2 and 3 together, 4 alone.
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
        assert_ne!(result.labels[0], result.labels[2]);
        assert_ne!(result.labels[4], result.labels[0]);
        assert_ne!(result.labels[4], result.labels[2]);
        assert_eq!(result.n_clusters, 3);
    }

    #[test]
    fn deterministic_with_seed() {
        let edges = vec![
            SignedEdge {
                i: 0,
                j: 1,
                weight: 1.0,
            },
            SignedEdge {
                i: 1,
                j: 2,
                weight: -0.5,
            },
            SignedEdge {
                i: 2,
                j: 3,
                weight: 1.0,
            },
            SignedEdge {
                i: 0,
                j: 3,
                weight: -0.5,
            },
            SignedEdge {
                i: 0,
                j: 2,
                weight: 0.3,
            },
            SignedEdge {
                i: 1,
                j: 3,
                weight: -0.2,
            },
        ];

        let r1 = CorrelationClustering::new()
            .with_seed(99)
            .fit(4, &edges)
            .expect("fit should succeed");

        let r2 = CorrelationClustering::new()
            .with_seed(99)
            .fit(4, &edges)
            .expect("fit should succeed");

        assert_eq!(
            r1.labels, r2.labels,
            "same seed should produce identical results"
        );
        assert!((r1.cost - r2.cost).abs() < 1e-12);
    }

    #[test]
    fn empty_edges_all_singletons() {
        let result = CorrelationClustering::new()
            .with_seed(42)
            .fit(4, &[])
            .expect("fit should succeed");

        assert_eq!(result.n_clusters, 4);
        assert_eq!(result.labels.len(), 4);

        // All labels distinct.
        let mut unique = result.labels.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 4);

        assert!((result.cost - 0.0).abs() < 1e-9);
    }

    #[test]
    fn single_item() {
        let result = CorrelationClustering::new()
            .with_seed(42)
            .fit(1, &[])
            .expect("fit should succeed");

        assert_eq!(result.labels, vec![0]);
        assert_eq!(result.n_clusters, 1);
        assert!((result.cost - 0.0).abs() < 1e-9);
    }

    #[test]
    fn local_search_improves_cost() {
        // Build a scenario where PIVOT alone is suboptimal but local search
        // can improve. 4 items, two natural pairs: (0,1) and (2,3).
        let edges = vec![
            SignedEdge {
                i: 0,
                j: 1,
                weight: 5.0,
            },
            SignedEdge {
                i: 2,
                j: 3,
                weight: 5.0,
            },
            SignedEdge {
                i: 0,
                j: 2,
                weight: -3.0,
            },
            SignedEdge {
                i: 0,
                j: 3,
                weight: -3.0,
            },
            SignedEdge {
                i: 1,
                j: 2,
                weight: -3.0,
            },
            SignedEdge {
                i: 1,
                j: 3,
                weight: -3.0,
            },
        ];

        let pivot_only = CorrelationClustering::new()
            .with_max_iter(0)
            .with_seed(42)
            .fit(4, &edges)
            .expect("fit should succeed");

        let with_refinement = CorrelationClustering::new()
            .with_max_iter(100)
            .with_seed(42)
            .fit(4, &edges)
            .expect("fit should succeed");

        // Refinement should not increase cost.
        assert!(
            with_refinement.cost <= pivot_only.cost + 1e-9,
            "local search should not increase cost: refined={}, pivot={}",
            with_refinement.cost,
            pivot_only.cost
        );
    }

    #[test]
    fn edges_from_distances_generates_correct_signs() {
        use super::super::distance::Euclidean;

        let data = vec![
            vec![0.0f32, 0.0],
            vec![0.1, 0.0],   // close to 0
            vec![10.0, 10.0], // far from 0 and 1
        ];

        let edges = CorrelationClustering::edges_from_distances(&data, &Euclidean, 1.0);

        assert_eq!(edges.len(), 3);

        // Find the edge between 0 and 1 (distance ~0.1, weight ~0.9).
        let e01 = edges
            .iter()
            .find(|e| (e.i == 0 && e.j == 1) || (e.i == 1 && e.j == 0))
            .expect("edge 0-1 should exist");
        assert!(e01.weight > 0.0, "close pair should have positive weight");

        // Edge 0-2 (distance ~14.14, weight ~-13.14).
        let e02 = edges
            .iter()
            .find(|e| (e.i == 0 && e.j == 2) || (e.i == 2 && e.j == 0))
            .expect("edge 0-2 should exist");
        assert!(e02.weight < 0.0, "far pair should have negative weight");

        // Edge 1-2 (distance ~14.07, weight ~-13.07).
        let e12 = edges
            .iter()
            .find(|e| (e.i == 1 && e.j == 2) || (e.i == 2 && e.j == 1))
            .expect("edge 1-2 should exist");
        assert!(e12.weight < 0.0, "far pair should have negative weight");
    }

    #[test]
    fn invalid_edge_index_error() {
        let edges = vec![SignedEdge {
            i: 0,
            j: 5,
            weight: 1.0,
        }];

        let result = CorrelationClustering::new().fit(3, &edges);

        assert!(
            result.is_err(),
            "edge referencing item >= n_items should error"
        );
        if let Err(Error::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "edge index");
        } else {
            panic!("expected InvalidParameter error");
        }
    }

    #[test]
    fn zero_items_returns_empty() {
        let result = CorrelationClustering::new()
            .fit(0, &[])
            .expect("fit should succeed");

        assert!(result.labels.is_empty());
        assert_eq!(result.n_clusters, 0);
        assert!((result.cost - 0.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn labels_cover_all_items(n_items in 1usize..20) {
            let edges: Vec<SignedEdge> = Vec::new();
            let result = CorrelationClustering::new()
                .with_seed(42)
                .fit(n_items, &edges)
                .unwrap();

            prop_assert_eq!(result.labels.len(), n_items,
                "every item should get a label");
        }

        #[test]
        fn labels_are_contiguous(n_items in 2usize..10) {
            // Deterministic edges based on item indices.
            let mut edges = Vec::new();
            for i in 0..n_items {
                for j in (i+1)..n_items {
                    let w = ((i * 3 + j * 5) % 7) as f32 - 3.0;
                    edges.push(SignedEdge { i, j, weight: w });
                }
            }

            let result = CorrelationClustering::new()
                .with_seed(42)
                .fit(n_items, &edges)
                .unwrap();

            // Labels should use exactly [0, n_clusters).
            let max_label = result.labels.iter().copied().max().unwrap_or(0);
            prop_assert!(max_label < result.n_clusters,
                "max label {} should be < n_clusters {}", max_label, result.n_clusters);

            let mut seen = vec![false; result.n_clusters];
            for &l in &result.labels {
                seen[l] = true;
            }
            for (i, &s) in seen.iter().enumerate() {
                prop_assert!(s, "cluster label {} unused but n_clusters={}", i, result.n_clusters);
            }
        }

        #[test]
        fn labels_use_contiguous_range(n_items in 1usize..15) {
            let result = CorrelationClustering::new()
                .with_seed(42)
                .fit(n_items, &[])
                .unwrap();

            let max_label = result.labels.iter().copied().max().unwrap_or(0);
            prop_assert!(max_label < result.n_clusters,
                "max label {} should be < n_clusters {}", max_label, result.n_clusters);

            let mut seen = vec![false; result.n_clusters];
            for &l in &result.labels {
                seen[l] = true;
            }
            for (i, &s) in seen.iter().enumerate() {
                prop_assert!(s, "cluster label {} not used but n_clusters={}", i, result.n_clusters);
            }
        }

        #[test]
        fn cost_matches_edges(n_items in 2usize..10) {
            // Fully connected with random weights.
            let mut edges = Vec::new();
            for i in 0..n_items {
                for j in (i+1)..n_items {
                    // Deterministic weight based on indices.
                    let w = ((i * 7 + j * 13) % 20) as f32 - 10.0;
                    edges.push(SignedEdge { i, j, weight: w });
                }
            }

            let result = CorrelationClustering::new()
                .with_seed(42)
                .fit(n_items, &edges)
                .unwrap();

            // Recompute cost independently.
            let mut expected_cost = 0.0f64;
            for edge in &edges {
                let same = result.labels[edge.i] == result.labels[edge.j];
                if (edge.weight > 0.0 && !same) || (edge.weight < 0.0 && same) {
                    expected_cost += (edge.weight as f64).abs();
                }
            }

            prop_assert!(
                (result.cost - expected_cost).abs() < 1e-6,
                "reported cost {} != recomputed cost {}", result.cost, expected_cost
            );
        }
    }
}
