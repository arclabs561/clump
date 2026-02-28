//! HDBSCAN: Hierarchical Density-Based Spatial Clustering of Applications with Noise.
//!
//! HDBSCAN (Campello, Moulavi, Sander 2013) extends DBSCAN by removing the global
//! epsilon parameter and instead building a hierarchy of density-based clusters.
//! It selects the most stable clusters from the hierarchy automatically.
//!
//! # Algorithm Outline
//!
//! 1. **Core distance**: For each point, compute the distance to its k-th nearest
//!    neighbor (where k = `min_samples`). This estimates local density.
//!
//! 2. **Mutual reachability distance**: For each pair (i, j):
//!    `mrd(i, j) = max(core_dist[i], core_dist[j], dist(i, j))`.
//!    This smooths out density spikes so sparse regions don't create spurious links.
//!
//! 3. **MST on mutual reachability graph**: Build a minimum spanning tree over the
//!    mutual reachability distances using Prim's algorithm (O(n^2)).
//!
//! 4. **Condensed cluster tree**: Walk MST edges in ascending distance order, merging
//!    components. When a merge produces a component below `min_cluster_size`, those
//!    points "fall out" as noise rather than forming a cluster split.
//!
//! 5. **Stability-based cluster extraction**: Each cluster in the condensed tree has a
//!    stability score = sum over member points of (lambda_p - lambda_birth).
//!    Select the set of non-overlapping clusters that maximizes total stability.
//!
//! 6. **Noise labeling**: Points not in any selected cluster are labeled as noise.
//!
//! # When to Use
//!
//! - **HDBSCAN vs DBSCAN**: HDBSCAN handles varying-density clusters without requiring
//!   a global epsilon. Prefer HDBSCAN when cluster densities differ significantly.
//! - **HDBSCAN vs k-means**: HDBSCAN finds non-convex clusters and identifies noise.
//!   Prefer k-means when clusters are roughly spherical and you know k.
//! - **Complexity**: O(n^2) time and space for the dense pairwise distance computation.
//!   Suitable for datasets up to ~10k-50k points depending on dimensionality.
//!
//! # References
//!
//! Campello, R. J. G. B., Moulavi, D., Sander, J. (2013). "Density-Based Clustering
//! Based on Hierarchical Density Estimates." PAKDD 2013.

use super::traits::Clustering;
use super::util::{self, UnionFind};
use crate::error::{Error, Result};

use super::dbscan::NOISE;

/// HDBSCAN clustering algorithm.
#[derive(Debug, Clone)]
pub struct Hdbscan {
    min_samples: usize,
    min_cluster_size: usize,
}

impl Hdbscan {
    /// Create a new HDBSCAN clusterer with default parameters.
    ///
    /// Defaults: `min_samples = 5`, `min_cluster_size = 5`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `min_samples` (k for core distance computation).
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Set `min_cluster_size` (minimum points for a cluster to persist).
    pub fn with_min_cluster_size(mut self, min_cluster_size: usize) -> Self {
        self.min_cluster_size = min_cluster_size;
        self
    }

    /// Fit and predict, returning `None` for noise points.
    pub fn fit_predict_with_noise(&self, data: &[Vec<f32>]) -> Result<Vec<Option<usize>>> {
        let labels = self.fit_predict(data)?;
        Ok(labels
            .into_iter()
            .map(|l| if l == NOISE { None } else { Some(l) })
            .collect())
    }
}

impl Default for Hdbscan {
    fn default() -> Self {
        Self {
            min_samples: 5,
            min_cluster_size: 5,
        }
    }
}

impl Clustering for Hdbscan {
    fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        let n = data.len();
        if n == 0 {
            return Err(Error::EmptyInput);
        }

        if self.min_samples == 0 {
            return Err(Error::InvalidParameter {
                name: "min_samples",
                message: "must be at least 1",
            });
        }

        if self.min_cluster_size < 2 {
            return Err(Error::InvalidParameter {
                name: "min_cluster_size",
                message: "must be at least 2",
            });
        }

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

        let dists = pairwise_distances(data);
        let core_dists = core_distances(&dists, n, self.min_samples);

        let mut mst = util::prim_mst(n, |i, j| {
            mutual_reachability(dists[i * n + j], core_dists[i], core_dists[j])
        });
        mst.sort_by(|a, b| a.2.total_cmp(&b.2));

        Ok(extract_clusters(&mst, n, self.min_cluster_size))
    }

    fn n_clusters(&self) -> usize {
        0
    }
}

fn pairwise_distances(data: &[Vec<f32>]) -> Vec<f32> {
    let n = data.len();
    let mut dists = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = util::squared_euclidean(&data[i], &data[j]).sqrt();
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    dists
}

fn core_distances(dists: &[f32], n: usize, min_samples: usize) -> Vec<f32> {
    let k = min_samples.min(n - 1).max(1);
    let mut core = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .map(|j| dists[i * n + j])
            .collect();
        row.sort_by(|a, b| a.total_cmp(b));
        core.push(row[k - 1]);
    }
    core
}

#[inline]
fn mutual_reachability(dist: f32, core_i: f32, core_j: f32) -> f32 {
    dist.max(core_i).max(core_j)
}

// ---------------------------------------------------------------------------
// Condensed cluster tree
// ---------------------------------------------------------------------------

/// An entry in the condensed cluster tree stored as a flat table.
///
/// Each row represents either:
/// - A point falling out of a cluster (child is a point index, child_size = 1)
/// - A cluster splitting into a child cluster (child is a cluster id, child_size > 1)
struct CondensedEdge {
    parent: usize,   // cluster id
    child: usize,    // point index or cluster id
    lambda: f64,     // 1/distance at which this happened
    child_size: usize,
}

fn extract_clusters(
    mst: &[(usize, usize, f32)],
    n: usize,
    min_cluster_size: usize,
) -> Vec<usize> {
    if n == 1 {
        return vec![NOISE];
    }

    // Cluster ids start at n (point ids are 0..n-1).
    let mut next_cluster_id = n;
    let mut uf = UnionFind::new(n);
    // UF root -> current cluster id (None if no cluster formed yet).
    let mut comp_cluster: Vec<Option<usize>> = vec![None; n];
    let mut condensed: Vec<CondensedEdge> = Vec::new();

    for &(u, v, dist) in mst {
        let ru = uf.find(u);
        let rv = uf.find(v);
        if ru == rv {
            continue;
        }

        let lambda = if dist > 0.0 { 1.0 / dist as f64 } else { f64::INFINITY };
        let ru_size = uf.size[ru];
        let rv_size = uf.size[rv];

        let left_big = ru_size >= min_cluster_size;
        let right_big = rv_size >= min_cluster_size;

        if left_big && right_big {
            // Genuine split: both sides are large. Create a new parent cluster.
            let new_cluster = next_cluster_id;
            next_cluster_id += 1;

            // Left child: if it has a cluster, use it; otherwise create one.
            let left_child = comp_cluster[ru].unwrap_or_else(|| {
                let id = next_cluster_id;
                next_cluster_id += 1;
                id
            });
            let right_child = comp_cluster[rv].unwrap_or_else(|| {
                let id = next_cluster_id;
                next_cluster_id += 1;
                id
            });

            condensed.push(CondensedEdge {
                parent: new_cluster,
                child: left_child,
                lambda,
                child_size: ru_size,
            });
            condensed.push(CondensedEdge {
                parent: new_cluster,
                child: right_child,
                lambda,
                child_size: rv_size,
            });

            // Also record individual point fallouts for the children if they
            // had no prior cluster (all their points are "born" into the child).
            if comp_cluster[ru].is_none() {
                add_point_fallouts(&mut condensed, &uf, ru, left_child, lambda, n);
            }
            if comp_cluster[rv].is_none() {
                add_point_fallouts(&mut condensed, &uf, rv, right_child, lambda, n);
            }

            let new_root = uf.union_roots(ru, rv);
            comp_cluster[new_root] = Some(new_cluster);
        } else if left_big || right_big {
            let (big, small) = if left_big { (ru, rv) } else { (rv, ru) };

            // Ensure big side has a cluster.
            let cluster = comp_cluster[big].unwrap_or_else(|| {
                let id = next_cluster_id;
                next_cluster_id += 1;
                // Record all existing big-component points as born into this cluster.
                add_point_fallouts(&mut condensed, &uf, big, id, lambda, n);
                id
            });

            // Small side's points fall out.
            add_point_fallouts(&mut condensed, &uf, small, cluster, lambda, n);

            let new_root = uf.union_roots(big, small);
            comp_cluster[new_root] = Some(cluster);
        } else {
            // Neither is large. Just merge; no cluster event.
            let existing = comp_cluster[ru].or(comp_cluster[rv]);
            let new_root = uf.union_roots(ru, rv);
            comp_cluster[new_root] = existing;
        }
    }

    let num_clusters = next_cluster_id - n;
    if num_clusters == 0 {
        return vec![NOISE; n];
    }

    // Compute lambda_birth for each cluster.
    // A cluster is "born" when it first appears as a child in the condensed tree.
    // The root cluster (never appears as a child) is born at lambda=0.
    let mut lambda_birth = vec![0.0f64; num_clusters];

    for edge in &condensed {
        if edge.child_size > 1 && edge.child >= n {
            let child_idx = edge.child - n;
            lambda_birth[child_idx] = edge.lambda;
        }
    }

    // Compute stability for each cluster.
    // stability(c) = sum over points p that fell out of c: (lambda_p - lambda_birth(c))
    //              + sum over child clusters of c: child_size * (lambda_split - lambda_birth(c))
    //
    // But a simpler formulation: for each condensed edge with parent=c,
    // contribution = child_size * (lambda - lambda_birth(c))
    let mut stability = vec![0.0f64; num_clusters];
    for edge in &condensed {
        if edge.parent < n {
            continue;
        }
        let cluster_idx = edge.parent - n;
        let birth = lambda_birth[cluster_idx];
        stability[cluster_idx] += edge.child_size as f64 * (edge.lambda - birth);
    }

    // Identify which clusters are leaves (no cluster children).
    let mut has_cluster_child = vec![false; num_clusters];
    for edge in &condensed {
        if edge.parent < n {
            continue;
        }
        if edge.child_size > 1 && edge.child >= n {
            let parent_idx = edge.parent - n;
            has_cluster_child[parent_idx] = true;
        }
    }

    // Build children map: parent cluster -> list of child clusters.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];
    for edge in &condensed {
        if edge.parent < n || edge.child < n {
            continue;
        }
        if edge.child_size > 1 {
            let parent_idx = edge.parent - n;
            let child_idx = edge.child - n;
            children[parent_idx].push(child_idx);
        }
    }

    // Select clusters: bottom-up stability selection.
    let mut selected = vec![false; num_clusters];
    let mut subtree_stab = stability.clone();

    // Process bottom-up. Since cluster ids increase as we process edges (parents
    // always have higher ids than children), reverse order is bottom-up.
    for i in (0..num_clusters).rev() {
        if !has_cluster_child[i] {
            // Leaf cluster: select it.
            selected[i] = true;
        } else {
            let child_sum: f64 = children[i].iter().map(|&c| subtree_stab[c]).sum();
            if stability[i] > child_sum {
                selected[i] = true;
                deselect_descendants(&children, i, &mut selected);
                subtree_stab[i] = stability[i];
            } else {
                subtree_stab[i] = child_sum;
            }
        }
    }

    // Assign labels by walking selected clusters and labeling all their points
    // (direct fallouts + non-selected descendant subtrees).
    let mut labels = vec![NOISE; n];
    let mut label_map = vec![usize::MAX; num_clusters];
    let mut next_label = 0usize;
    for (i, &sel) in selected.iter().enumerate() {
        if sel {
            label_map[i] = next_label;
            next_label += 1;
        }
    }

    // For each selected cluster, label all its points (direct + descendants).
    for i in 0..num_clusters {
        if !selected[i] {
            continue;
        }
        let label = label_map[i];
        label_all_points(&condensed, &selected, n, i, label, &mut labels);
    }

    labels
}

/// Add individual point fallouts for all points in the component rooted at `comp_root`.
fn add_point_fallouts(
    condensed: &mut Vec<CondensedEdge>,
    uf: &UnionFind,
    comp_root: usize,
    parent_cluster: usize,
    lambda: f64,
    n: usize,
) {
    // Find all points in this component.
    // Since UnionFind doesn't track members, scan all points.
    for p in 0..n {
        // We can't call find() because uf is not mutable, but we can check
        // if a point's root matches comp_root by walking the parent chain.
        // Actually, UnionFind::find needs &mut self due to path compression.
        // Use the parent array directly for a non-mutating check.
        if find_root_readonly(&uf.parent, p) == comp_root {
            condensed.push(CondensedEdge {
                parent: parent_cluster,
                child: p,
                lambda,
                child_size: 1,
            });
        }
    }
}

fn find_root_readonly(parent: &[usize], mut x: usize) -> usize {
    while parent[x] != x {
        x = parent[x];
    }
    x
}

/// Label all points belonging to cluster `cluster_idx` (and non-selected descendants).
fn label_all_points(
    condensed: &[CondensedEdge],
    selected: &[bool],
    n: usize,
    cluster_idx: usize,
    label: usize,
    labels: &mut [usize],
) {
    let cluster_id = cluster_idx + n;

    for edge in condensed {
        if edge.parent != cluster_id {
            continue;
        }
        if edge.child_size == 1 && edge.child < n {
            // Direct point fallout.
            labels[edge.child] = label;
        } else if edge.child_size > 1 && edge.child >= n {
            // Child cluster.
            let child_idx = edge.child - n;
            if selected[child_idx] {
                // Child is independently selected; don't override.
                continue;
            }
            // Recursively label all points in this non-selected child.
            label_all_points(condensed, selected, n, child_idx, label, labels);
        }
    }
}

fn deselect_descendants(children: &[Vec<usize>], node: usize, selected: &mut [bool]) {
    for &child in &children[node] {
        selected[child] = false;
        deselect_descendants(children, child, selected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cluster(center: &[f32], n: usize, spread: f32) -> Vec<Vec<f32>> {
        let dim = center.len();
        let mut points = Vec::with_capacity(n);
        for i in 0..n {
            let mut p = Vec::with_capacity(dim);
            for (d, &c) in center.iter().enumerate() {
                let offset = spread * ((i * 7 + d * 13) % 11) as f32 / 11.0 - spread / 2.0;
                p.push(c + offset);
            }
            points.push(p);
        }
        points
    }

    #[test]
    fn two_well_separated_clusters() {
        let mut data = make_cluster(&[0.0, 0.0], 20, 0.5);
        data.extend(make_cluster(&[20.0, 20.0], 20, 0.5));

        let hdbscan = Hdbscan::new()
            .with_min_samples(3)
            .with_min_cluster_size(10);
        let labels = hdbscan.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), 40);

        // All points in the first spatial group should share one label.
        let l0 = labels[0];
        assert_ne!(l0, NOISE);
        for &l in &labels[1..20] {
            assert_eq!(l, l0);
        }

        // All points in the second spatial group should share one label.
        let l20 = labels[20];
        assert_ne!(l20, NOISE);
        for &l in &labels[21..40] {
            assert_eq!(l, l20);
        }

        // The two groups should have different labels.
        assert_ne!(l0, l20);
    }

    #[test]
    fn clusters_with_different_densities() {
        let mut data = make_cluster(&[0.0, 0.0], 30, 0.3);
        data.extend(make_cluster(&[50.0, 50.0], 30, 3.0));

        let hdbscan = Hdbscan::new()
            .with_min_samples(3)
            .with_min_cluster_size(5);
        let labels = hdbscan.fit_predict(&data).unwrap();

        let dense_non_noise = labels[..30].iter().filter(|&&l| l != NOISE).count();
        let sparse_non_noise = labels[30..].iter().filter(|&&l| l != NOISE).count();

        assert!(
            dense_non_noise >= 20,
            "dense cluster should have most points assigned, got {dense_non_noise}"
        );
        assert!(
            sparse_non_noise >= 15,
            "sparse cluster should have many points assigned, got {sparse_non_noise}"
        );
    }

    #[test]
    fn noise_points_between_clusters() {
        let mut data = make_cluster(&[0.0, 0.0], 15, 0.3);
        data.extend(make_cluster(&[20.0, 20.0], 15, 0.3));
        data.push(vec![10.0, 10.0]);
        data.push(vec![8.0, 12.0]);
        data.push(vec![12.0, 8.0]);

        let hdbscan = Hdbscan::new()
            .with_min_samples(3)
            .with_min_cluster_size(5);
        let labels = hdbscan.fit_predict(&data).unwrap();

        // In HDBSCAN, noise points that are absorbed into a cluster before the
        // cross-cluster merge get labeled with that cluster. Points truly between
        // clusters may or may not be noise depending on their mutual reachability
        // distances. We check that at least 2 distinct clusters are found.
        let non_noise: std::collections::HashSet<usize> =
            labels.iter().copied().filter(|&l| l != NOISE).collect();
        assert!(non_noise.len() >= 2, "should find at least 2 clusters");
    }

    #[test]
    fn single_cluster() {
        let data = make_cluster(&[0.0, 0.0], 20, 0.5);

        // Use min_cluster_size > n/2 to prevent internal splits from creating
        // two sub-clusters within the single spatial group.
        let hdbscan = Hdbscan::new()
            .with_min_samples(3)
            .with_min_cluster_size(15);
        let labels = hdbscan.fit_predict(&data).unwrap();

        let non_noise: Vec<usize> = labels.iter().copied().filter(|&l| l != NOISE).collect();
        assert!(!non_noise.is_empty(), "should find at least one cluster");

        let first = non_noise[0];
        for &l in &non_noise[1..] {
            assert_eq!(l, first);
        }
    }

    #[test]
    fn all_noise_high_min_cluster_size() {
        let data = vec![
            vec![0.0, 0.0],
            vec![10.0, 10.0],
            vec![20.0, 20.0],
        ];

        let hdbscan = Hdbscan::new()
            .with_min_samples(2)
            .with_min_cluster_size(100);
        let labels = hdbscan.fit_predict(&data).unwrap();

        for &l in &labels {
            assert_eq!(l, NOISE);
        }
    }

    #[test]
    fn empty_input() {
        let data: Vec<Vec<f32>> = vec![];
        let hdbscan = Hdbscan::new();
        let result = hdbscan.fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_min_samples_zero() {
        let data = vec![vec![0.0, 0.0]];
        let hdbscan = Hdbscan::new().with_min_samples(0);
        let result = hdbscan.fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_min_cluster_size_one() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let hdbscan = Hdbscan::new().with_min_cluster_size(1);
        let result = hdbscan.fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn large_min_samples_relative_to_data() {
        let data = make_cluster(&[0.0, 0.0], 10, 0.5);

        let hdbscan = Hdbscan::new()
            .with_min_samples(100)
            .with_min_cluster_size(3);
        let labels = hdbscan.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), 10);
    }

    #[test]
    fn fit_predict_with_noise_api() {
        let mut data = make_cluster(&[0.0, 0.0], 15, 0.3);
        data.push(vec![100.0, 100.0]);

        let hdbscan = Hdbscan::new()
            .with_min_samples(3)
            .with_min_cluster_size(5);
        let labels = hdbscan.fit_predict_with_noise(&data).unwrap();

        assert_eq!(labels.len(), 16);
        // The distant outlier may or may not be noise depending on when it merges.
        // But the cluster points should be labeled.
        let cluster_labels: Vec<_> = labels[..15].iter().filter_map(|l| *l).collect();
        assert!(!cluster_labels.is_empty(), "cluster should have labeled points");
    }

    #[test]
    fn property_non_noise_labels_meet_min_cluster_size() {
        let mut data = make_cluster(&[0.0, 0.0], 25, 0.5);
        data.extend(make_cluster(&[30.0, 30.0], 25, 0.5));
        data.push(vec![15.0, 15.0]);

        let min_cluster_size = 5;
        let hdbscan = Hdbscan::new()
            .with_min_samples(3)
            .with_min_cluster_size(min_cluster_size);
        let labels = hdbscan.fit_predict(&data).unwrap();

        let mut counts = std::collections::HashMap::new();
        for &l in &labels {
            if l != NOISE {
                *counts.entry(l).or_insert(0usize) += 1;
            }
        }

        for (&label, &count) in &counts {
            assert!(
                count >= min_cluster_size,
                "label {label} has {count} points, expected at least {min_cluster_size}"
            );
        }
    }

    #[test]
    fn dimension_mismatch() {
        let data = vec![vec![0.0, 0.0], vec![1.0]];
        let hdbscan = Hdbscan::new();
        let result = hdbscan.fit_predict(&data);
        assert!(result.is_err());
    }
}
