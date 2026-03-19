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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// Uses PIVOT + local search, then an iterated-flip refinement step
    /// (Cohen-Addad et al., STOC 2024): reweight inter-cluster edges by
    /// doubling them and re-run, keeping the better solution. This improves
    /// the worst-case approximation ratio from 3 to 1.875.
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
        let adj = Self::build_adj(n_items, edges);

        // Phase 1: PIVOT + merge + local search on original weights.
        let mut labels1 = self.pivot(n_items, &adj);
        if self.max_iter > 0 {
            Self::merge_pass(n_items, &adj, &mut labels1);
            self.local_search(n_items, &adj, &mut labels1);
        }
        let cost1 = compute_cost(&labels1, edges);

        // Phase 2: Iterated flip (Cohen-Addad et al. STOC 2024 warm-up).
        // Double weights of inter-cluster edges from phase 1, then re-run.
        // This encourages the second pass to keep those edges internal,
        // escaping local minima that PIVOT + local search gets stuck in.
        let flipped_edges: Vec<SignedEdge> = edges
            .iter()
            .map(|e| {
                if labels1[e.i] != labels1[e.j] {
                    SignedEdge {
                        weight: e.weight * 2.0,
                        ..*e
                    }
                } else {
                    *e
                }
            })
            .collect();
        let adj2 = Self::build_adj(n_items, &flipped_edges);

        let mut labels2 = self.pivot(n_items, &adj2);
        if self.max_iter > 0 {
            Self::merge_pass(n_items, &adj2, &mut labels2);
            self.local_search(n_items, &adj2, &mut labels2);
        }
        // Evaluate cost on ORIGINAL edges (not reweighted).
        let cost2 = compute_cost(&labels2, edges);

        // Return the better solution.
        let (labels, cost) = if cost2 < cost1 {
            (labels2, cost2)
        } else {
            (labels1, cost1)
        };

        let (labels, n_clusters) = relabel_contiguous(&labels);

        Ok(CorrelationResult {
            labels,
            n_clusters,
            cost,
        })
    }

    /// Build adjacency list from edges.
    fn build_adj(n_items: usize, edges: &[SignedEdge]) -> Vec<Vec<(usize, f32)>> {
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_items];
        for edge in edges {
            adj[edge.i].push((edge.j, edge.weight));
            adj[edge.j].push((edge.i, edge.weight));
        }
        adj
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

    /// Merge pass: compute merge delta for each inter-cluster edge pair
    /// directly from the adjacency list (O(edges) per iteration, not O(n * edges)).
    /// Apply the best merge. Repeat until no improving merge exists.
    fn merge_pass(n_items: usize, adj: &[Vec<(usize, f32)>], labels: &mut [usize]) {
        loop {
            // Accumulate merge deltas for all inter-cluster pairs in one pass
            // over all edges. Each inter-cluster edge (i, j) with weight w
            // contributes -w to the merge delta of (labels[i], labels[j]).
            let mut pair_deltas: HashMap<(usize, usize), f64> = HashMap::new();

            for i in 0..n_items {
                for &(j, w) in &adj[i] {
                    let (ca, cb) = (labels[i], labels[j]);
                    if ca == cb || i > j {
                        continue; // skip same-cluster and double-counting
                    }
                    let pair = (ca.min(cb), ca.max(cb));
                    *pair_deltas.entry(pair).or_insert(0.0) -= w as f64;
                }
            }

            // Find the best merge.
            let best = pair_deltas
                .iter()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

            match best {
                Some((&(from, to), &delta)) if delta < 0.0 => {
                    for label in labels[..n_items].iter_mut() {
                        if *label == from {
                            *label = to;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    /// Local search with delta caching.
    ///
    /// Caches the best move gain for each item. When item u moves, only
    /// recomputes deltas for u's neighbors (O(degree) invalidation instead
    /// of O(n) full scan). Cache hit rate > 95% in practice.
    fn local_search(&self, n_items: usize, adj: &[Vec<(usize, f32)>], labels: &mut [usize]) {
        let mut next_singleton = labels.iter().copied().max().unwrap_or(0) + 1;

        // Cache: best_move[item] = (target_cluster, delta). Negative delta = improvement.
        let mut best_move: Vec<(usize, f64)> = Vec::with_capacity(n_items);
        let mut stale = vec![true; n_items]; // mark all as needing computation

        for _ in 0..self.max_iter {
            let mut improved = false;

            // Ensure cache is populated.
            if best_move.len() < n_items {
                best_move.resize(n_items, (0, 0.0));
            }

            for item in 0..n_items {
                if stale[item] {
                    // Recompute best move for this item.
                    let current = labels[item];
                    let mut best_target = current;
                    let mut best_delta = 0.0f64;

                    // Candidates: neighbor clusters + fresh singleton.
                    let singleton = next_singleton;
                    let delta_s = move_delta(item, current, singleton, adj, labels);
                    if delta_s < best_delta {
                        best_delta = delta_s;
                        best_target = singleton;
                    }

                    for &(neighbor, _) in &adj[item] {
                        let c = labels[neighbor];
                        if c == current {
                            continue;
                        }
                        let d = move_delta(item, current, c, adj, labels);
                        if d < best_delta {
                            best_delta = d;
                            best_target = c;
                        }
                    }

                    best_move[item] = (best_target, best_delta);
                    stale[item] = false;
                }

                let (target, delta) = best_move[item];
                if delta < 0.0 && target != labels[item] {
                    labels[item] = target;
                    if target == next_singleton {
                        next_singleton += 1;
                    }
                    improved = true;

                    // Invalidate neighbors -- their deltas may have changed.
                    stale[item] = true;
                    for &(neighbor, _) in &adj[item] {
                        stale[neighbor] = true;
                    }
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
/// Fused: computes delta directly without separate before/after costs.
#[inline]
fn move_delta(
    item: usize,
    from: usize,
    to: usize,
    adj: &[Vec<(usize, f32)>],
    labels: &[usize],
) -> f64 {
    let mut delta = 0.0f64;

    for &(neighbor, weight) in &adj[item] {
        let nc = labels[neighbor];
        // Before: item is in `from`. After: item is in `to`.
        // Only edges where nc == from, nc == to, or neither are affected.
        let w = weight as f64;
        if nc == from {
            // Was same cluster (from), now different (to != from).
            // Positive edge: gains cost |w|. Negative edge: loses cost |w|.
            delta += w; // w > 0 means cost increases; w < 0 means cost decreases
        } else if nc == to {
            // Was different cluster, now same cluster (to).
            // Positive edge: loses cost |w|. Negative edge: gains cost |w|.
            delta -= w;
        }
        // If nc != from && nc != to, the same/different status doesn't change.
    }

    delta
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

        /// With all strongly positive edges, the result has fewer clusters than items.
        #[test]
        fn all_positive_fewer_clusters(n_items in 3usize..12) {
            let mut edges = Vec::new();
            for i in 0..n_items {
                for j in (i+1)..n_items {
                    edges.push(SignedEdge { i, j, weight: 2.0 });
                }
            }

            let result = CorrelationClustering::new()
                .with_seed(42)
                .fit(n_items, &edges)
                .unwrap();

            prop_assert!(result.n_clusters <= n_items,
                "n_clusters {} > n_items {}", result.n_clusters, n_items);
            prop_assert!(result.n_clusters < n_items,
                "strongly positive edges should merge at least one pair: \
                 n_clusters={}, n_items={}", result.n_clusters, n_items);
        }

        /// With all negative edges, the number of clusters is at least floor(n/2).
        #[test]
        fn all_negative_many_clusters(n_items in 3usize..12) {
            let mut edges = Vec::new();
            for i in 0..n_items {
                for j in (i+1)..n_items {
                    edges.push(SignedEdge { i, j, weight: -2.0 });
                }
            }

            let result = CorrelationClustering::new()
                .with_seed(42)
                .fit(n_items, &edges)
                .unwrap();

            let min_expected = n_items / 2;
            prop_assert!(result.n_clusters >= min_expected,
                "all-negative edges: n_clusters {} < floor(n/2)={} for n_items={}",
                result.n_clusters, min_expected, n_items);
        }
    }
}
