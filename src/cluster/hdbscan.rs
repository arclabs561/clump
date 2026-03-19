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

use super::distance::{DistanceMetric, Euclidean};
use super::flat::DataRef;
use super::util::{self, UnionFind};
use crate::error::{Error, Result};

use super::dbscan::NOISE;

/// HDBSCAN clustering algorithm, generic over a distance metric.
///
/// The default metric is [`Euclidean`] (L2), matching the original behavior.
///
/// ```
/// use clump::{Hdbscan, NOISE};
///
/// // Two tight clusters, well-separated
/// let mut data: Vec<Vec<f32>> = (0..15).map(|i| {
///     vec![(i % 3) as f32 * 0.1, (i / 3) as f32 * 0.1]
/// }).collect();
/// data.extend((0..15).map(|i| {
///     vec![50.0 + (i % 3) as f32 * 0.1, 50.0 + (i / 3) as f32 * 0.1]
/// }));
///
/// let labels = Hdbscan::new()
///     .with_min_samples(3)
///     .with_min_cluster_size(5)
///     .fit_predict(&data)
///     .unwrap();
///
/// assert_eq!(labels.len(), 30);
/// // At least 2 distinct non-noise clusters found
/// let clusters: std::collections::HashSet<_> =
///     labels.iter().copied().filter(|&l| l != NOISE).collect();
/// assert!(clusters.len() >= 2);
/// ```
#[derive(Debug, Clone)]
pub struct Hdbscan<D: DistanceMetric = Euclidean> {
    min_samples: usize,
    min_cluster_size: usize,
    metric: D,
}

impl Hdbscan<Euclidean> {
    /// Create a new HDBSCAN clusterer with default parameters.
    ///
    /// Defaults: `min_samples = 5`, `min_cluster_size = 5`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<D: DistanceMetric> Hdbscan<D> {
    /// Create a new HDBSCAN clusterer with a custom distance metric.
    pub fn with_metric(metric: D) -> Self {
        Self {
            min_samples: 5,
            min_cluster_size: 5,
            metric,
        }
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
    pub fn fit_predict_with_noise(
        &self,
        data: &(impl DataRef + ?Sized),
    ) -> Result<Vec<Option<usize>>> {
        let labels = self.fit_predict(data)?;
        Ok(labels
            .into_iter()
            .map(|l| if l == NOISE { None } else { Some(l) })
            .collect())
    }
}

impl Default for Hdbscan<Euclidean> {
    fn default() -> Self {
        Self {
            min_samples: 5,
            min_cluster_size: 5,
            metric: Euclidean,
        }
    }
}

/// Validate inputs and build the sorted MST. Shared by `fit_predict` and `fit`.
fn build_mst<D: DistanceMetric>(
    data: &(impl DataRef + ?Sized),
    metric: &D,
    min_samples: usize,
    min_cluster_size: usize,
) -> Result<Vec<(usize, usize, f32)>> {
    let n = data.n();
    if n == 0 {
        return Err(Error::EmptyInput);
    }
    if min_samples == 0 {
        return Err(Error::InvalidParameter {
            name: "min_samples",
            message: "must be at least 1",
        });
    }
    if min_cluster_size < 2 {
        return Err(Error::InvalidParameter {
            name: "min_cluster_size",
            message: "must be at least 2",
        });
    }
    let d = data.d();
    if d == 0 {
        return Err(Error::InvalidParameter {
            name: "dimension",
            message: "must be at least 1",
        });
    }
    for i in 1..n {
        if data.row(i).len() != d {
            return Err(Error::DimensionMismatch {
                expected: d,
                found: data.row(i).len(),
            });
        }
    }
    util::validate_finite(data)?;

    const MAX_MATRIX_BYTES: usize = 1024 * 1024 * 1024;
    let use_matrix = (n as u64) * (n as u64) * 4 <= MAX_MATRIX_BYTES as u64;

    let (_core_dists, mst) = if use_matrix {
        let dists = util::pairwise_distance_matrix(data, metric);
        let cd = core_distances(&dists, n, min_samples);
        let mst = util::prim_mst(n, |i, j| {
            mutual_reachability(dists[i * n + j], cd[i], cd[j])
        });
        (cd, mst)
    } else {
        let vecs: Vec<Vec<f32>> = (0..n).map(|i| data.row(i).to_vec()).collect();
        let cd = core_distances_vptree(&vecs, metric, min_samples);
        let mst = util::prim_mst(n, |i, j| {
            let d = metric.distance(data.row(i), data.row(j));
            mutual_reachability(d, cd[i], cd[j])
        });
        (cd, mst)
    };

    let mut mst = mst;
    mst.sort_by(|a, b| a.2.total_cmp(&b.2));
    Ok(mst)
}

impl<D: DistanceMetric> Hdbscan<D> {
    /// Fit and return one cluster label per input point.
    ///
    /// Noise points are labeled with `NOISE` (`usize::MAX`).
    pub fn fit_predict(&self, data: &(impl DataRef + ?Sized)) -> Result<Vec<usize>> {
        let mst = build_mst(data, &self.metric, self.min_samples, self.min_cluster_size)?;
        let (labels, _) = extract_clusters(&mst, data.n(), self.min_cluster_size);
        Ok(labels)
    }

    /// Fit and return labels with outlier scores.
    ///
    /// Outlier scores use GLOSH (Global-Local Outlier Score from Hierarchies,
    /// Campello et al. 2015). For each point p in selected cluster c:
    ///
    /// `GLOSH(p) = 1 - (lambda_p / lambda_max(c))`
    ///
    /// where `lambda_p` is the 1/distance at which p fell out of the condensed
    /// tree and `lambda_max(c)` is the maximum such value in cluster c.
    /// Noise points get score 1.0. Score in [0, 1]; higher = more outlier-like.
    pub fn fit(&self, data: &(impl DataRef + ?Sized)) -> Result<HdbscanResult> {
        let mst = build_mst(data, &self.metric, self.min_samples, self.min_cluster_size)?;
        let (labels, outlier_scores) = extract_clusters(&mst, data.n(), self.min_cluster_size);
        Ok(HdbscanResult {
            labels,
            outlier_scores,
        })
    }
}

/// Result of HDBSCAN fitting, including labels and outlier scores.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HdbscanResult {
    /// Cluster label for each point. Noise points have label `NOISE` (`usize::MAX`).
    pub labels: Vec<usize>,
    /// GLOSH outlier score for each point in [0, 1]. Higher = more outlier-like.
    /// Noise points have score 1.0.
    pub outlier_scores: Vec<f32>,
}

/// NaN/Inf input validation tests (hdbscan-specific).
#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn nan_input_rejected() {
        let data = vec![vec![0.0, f32::NAN], vec![1.0, 1.0], vec![2.0, 2.0]];
        let result = Hdbscan::new().fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn inf_input_rejected() {
        let data = vec![vec![0.0, 0.0], vec![f32::INFINITY, 1.0], vec![2.0, 2.0]];
        let result = Hdbscan::new().fit_predict(&data);
        assert!(result.is_err());
    }
}

/// Compute core distances using VP-tree kNN queries.
/// O(n log n) build + O(n * k * log n) queries vs O(n^2) brute force.
fn core_distances_vptree(
    data: &[Vec<f32>],
    metric: &impl super::distance::DistanceMetric,
    min_samples: usize,
) -> Vec<f32> {
    let n = data.len();
    let k = min_samples.min(n - 1).max(1);
    let tree = super::vptree::VpTree::new(data, metric);

    // k+1 because knn includes the query point itself.
    (0..n)
        .map(|i| {
            let neighbors = tree.knn(&data[i], k + 1);
            // The k-th nearest neighbor (excluding self) is at index k.
            if neighbors.len() > k {
                neighbors[k].1
            } else {
                neighbors.last().map_or(0.0, |&(_, d)| d)
            }
        })
        .collect()
}

fn core_distances(dists: &[f32], n: usize, min_samples: usize) -> Vec<f32> {
    let k = min_samples.min(n - 1).max(1);

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row: Vec<f32> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| dists[i * n + j])
                    .collect();
                row.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
                row[k - 1]
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut core = Vec::with_capacity(n);
        let mut row = Vec::with_capacity(n.saturating_sub(1));
        for i in 0..n {
            row.clear();
            row.extend((0..n).filter(|&j| j != i).map(|j| dists[i * n + j]));
            row.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
            core.push(row[k - 1]);
        }
        core
    }
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
    parent: usize, // cluster id
    child: usize,  // point index or cluster id
    lambda: f64,   // 1/distance at which this happened
    child_size: usize,
}

/// Extract clusters from the MST and compute GLOSH outlier scores.
///
/// Returns `(labels, outlier_scores)` where labels use `NOISE` for unassigned
/// points and outlier_scores are in [0, 1] (GLOSH; noise = 1.0).
fn extract_clusters(
    mst: &[(usize, usize, f32)],
    n: usize,
    min_cluster_size: usize,
) -> (Vec<usize>, Vec<f32>) {
    if n == 1 {
        return (vec![NOISE], vec![1.0]);
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

        let lambda = if dist > 0.0 {
            1.0 / dist as f64
        } else {
            f64::INFINITY
        };
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
        return (vec![NOISE; n], vec![1.0; n]);
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

    // Enforce min_cluster_size: relabel undersized clusters as NOISE.
    if min_cluster_size > 1 {
        let mut counts = vec![0usize; next_label];
        for &l in &labels {
            if l != NOISE && l < next_label {
                counts[l] += 1;
            }
        }
        for l in labels.iter_mut() {
            if *l != NOISE && *l < next_label && counts[*l] < min_cluster_size {
                *l = NOISE;
            }
        }
        let mut remap = vec![NOISE; next_label];
        let mut new_id = 0;
        for (old, &count) in counts.iter().enumerate() {
            if count >= min_cluster_size {
                remap[old] = new_id;
                new_id += 1;
            }
        }
        for l in &mut labels {
            if *l != NOISE {
                *l = remap[*l];
            }
        }
    }

    // -------------------------------------------------------------------
    // GLOSH: Global-Local Outlier Score from Hierarchies (Campello et al. 2015)
    //
    // For each point p assigned to selected cluster c:
    //   lambda_p      = lambda at which p fell out of the condensed tree
    //   lambda_max(c) = max lambda of any point fallout belonging to c
    //   GLOSH(p)      = 1 - lambda_p / lambda_max(c)
    // Noise points get GLOSH = 1.0.
    // -------------------------------------------------------------------

    // Map each condensed-tree cluster index to its owning selected cluster.
    let mut cluster_owner = vec![usize::MAX; num_clusters];
    for i in 0..num_clusters {
        if selected[i] {
            assign_owner(&children, &selected, i, i, &mut cluster_owner);
        }
    }

    // Build reverse map: final_label -> selected cluster index.
    let final_label_count = labels
        .iter()
        .filter(|&&l| l != NOISE)
        .copied()
        .max()
        .map_or(0, |m| m + 1);
    let mut label_to_selected = vec![usize::MAX; final_label_count];
    for (i, &lm) in label_map.iter().enumerate() {
        if selected[i] && lm != usize::MAX && lm < final_label_count {
            label_to_selected[lm] = i;
        }
    }

    // For each point, find the maximum lambda from condensed tree edges
    // whose parent cluster maps to the same selected cluster as the point's label.
    let mut point_lambda = vec![0.0f64; n];
    for edge in &condensed {
        if edge.child_size != 1 || edge.child >= n {
            continue;
        }
        let p = edge.child;
        if labels[p] == NOISE {
            continue;
        }
        let parent_idx = match edge.parent.checked_sub(n) {
            Some(idx) if idx < num_clusters => idx,
            _ => continue,
        };
        let owner = cluster_owner[parent_idx];
        if owner >= num_clusters {
            continue;
        }
        let owner_label = label_map[owner];
        if owner_label == labels[p] && edge.lambda > point_lambda[p] {
            point_lambda[p] = edge.lambda;
        }
    }

    // Compute lambda_max per selected cluster.
    let mut cluster_lambda_max = vec![0.0f64; num_clusters];
    for (p, &l) in labels.iter().enumerate() {
        if l == NOISE || l >= final_label_count {
            continue;
        }
        let sel_idx = label_to_selected[l];
        if sel_idx < num_clusters && point_lambda[p] > cluster_lambda_max[sel_idx] {
            cluster_lambda_max[sel_idx] = point_lambda[p];
        }
    }

    // Compute GLOSH scores.
    let mut outlier_scores = vec![1.0f32; n];
    for (p, &l) in labels.iter().enumerate() {
        if l == NOISE || l >= final_label_count {
            continue;
        }
        let sel_idx = label_to_selected[l];
        if sel_idx >= num_clusters {
            continue;
        }
        let lmax = cluster_lambda_max[sel_idx];
        if lmax > 0.0 {
            outlier_scores[p] = ((1.0 - point_lambda[p] / lmax) as f32).clamp(0.0, 1.0);
        } else {
            outlier_scores[p] = 0.0;
        }
    }

    (labels, outlier_scores)
}

/// Recursively assign `owner` as the owning selected cluster for `node` and
/// all non-selected descendants.
fn assign_owner(
    children: &[Vec<usize>],
    selected: &[bool],
    node: usize,
    owner: usize,
    cluster_owner: &mut [usize],
) {
    cluster_owner[node] = owner;
    for &child in &children[node] {
        if selected[child] {
            continue;
        }
        assign_owner(children, selected, child, owner, cluster_owner);
    }
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
    for p in 0..n {
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
            labels[edge.child] = label;
        } else if edge.child_size > 1 && edge.child >= n {
            let child_idx = edge.child - n;
            if selected[child_idx] {
                continue;
            }
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

        let hdbscan = Hdbscan::new().with_min_samples(3).with_min_cluster_size(10);
        let labels = hdbscan.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), 40);

        let l0 = labels[0];
        assert_ne!(l0, NOISE);
        for &l in &labels[1..20] {
            assert_eq!(l, l0);
        }

        let l20 = labels[20];
        assert_ne!(l20, NOISE);
        for &l in &labels[21..40] {
            assert_eq!(l, l20);
        }

        assert_ne!(l0, l20);
    }

    #[test]
    fn clusters_with_different_densities() {
        let mut data = make_cluster(&[0.0, 0.0], 30, 0.3);
        data.extend(make_cluster(&[50.0, 50.0], 30, 3.0));

        let hdbscan = Hdbscan::new().with_min_samples(3).with_min_cluster_size(5);
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

        let hdbscan = Hdbscan::new().with_min_samples(3).with_min_cluster_size(5);
        let labels = hdbscan.fit_predict(&data).unwrap();

        let non_noise: std::collections::HashSet<usize> =
            labels.iter().copied().filter(|&l| l != NOISE).collect();
        assert!(non_noise.len() >= 2, "should find at least 2 clusters");
    }

    #[test]
    fn single_cluster() {
        let data = make_cluster(&[0.0, 0.0], 20, 0.5);

        let hdbscan = Hdbscan::new().with_min_samples(3).with_min_cluster_size(15);
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
        let data = vec![vec![0.0, 0.0], vec![10.0, 10.0], vec![20.0, 20.0]];

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

        let hdbscan = Hdbscan::new().with_min_samples(3).with_min_cluster_size(5);
        let labels = hdbscan.fit_predict_with_noise(&data).unwrap();

        assert_eq!(labels.len(), 16);
        let cluster_labels: Vec<_> = labels[..15].iter().filter_map(|l| *l).collect();
        assert!(
            !cluster_labels.is_empty(),
            "cluster should have labeled points"
        );
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

    #[test]
    fn with_custom_metric() {
        use crate::cluster::distance::SquaredEuclidean;

        let mut data = make_cluster(&[0.0, 0.0], 20, 0.5);
        data.extend(make_cluster(&[20.0, 20.0], 20, 0.5));

        let hdbscan = Hdbscan::with_metric(SquaredEuclidean)
            .with_min_samples(3)
            .with_min_cluster_size(10);
        let labels = hdbscan.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), 40);
        let non_noise: std::collections::HashSet<usize> =
            labels.iter().copied().filter(|&l| l != NOISE).collect();
        assert!(non_noise.len() >= 2, "should find at least 2 clusters");
    }

    /// Outlier scores should be 1.0 for noise, [0, 1] for cluster members.
    #[test]
    fn outlier_scores_range() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![100.0, 100.0], // outlier
        ];
        let result = Hdbscan::new().with_min_samples(2).fit(&data).unwrap();
        for (i, &score) in result.outlier_scores.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&score),
                "outlier score for point {i} should be in [0,1], got {score}"
            );
        }
        // The isolated point should have highest outlier score.
        let max_score_idx = result
            .outlier_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            max_score_idx, 4,
            "isolated point should have highest outlier score"
        );
    }

    /// Duplicate points should be in the same cluster.
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
        let labels = Hdbscan::new()
            .with_min_samples(2)
            .fit_predict(&data)
            .unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
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
        /// Labels must be NOISE or in [0, n_clusters).
        #[test]
        fn labels_valid(data in arb_data(20, 3)) {
            let labels = Hdbscan::new().fit_predict(&data).unwrap();
            prop_assert_eq!(labels.len(), data.len());
            for &l in &labels {
                prop_assert!(l == NOISE || l < data.len());
            }
        }

        /// Outlier scores must be in [0, 1].
        #[test]
        fn outlier_scores_valid(data in arb_data(15, 2)) {
            let result = Hdbscan::new().fit(&data).unwrap();
            for (i, &s) in result.outlier_scores.iter().enumerate() {
                prop_assert!(
                    (0.0..=1.0).contains(&s),
                    "outlier_scores[{}] = {} not in [0,1]", i, s
                );
            }
        }

        /// Non-noise clusters must have >= min_cluster_size points.
        #[test]
        fn min_cluster_size_respected(data in arb_data(30, 2)) {
            let min_cs = 3;
            let labels = Hdbscan::new()
                .with_min_cluster_size(min_cs)
                .fit_predict(&data).unwrap();
            let mut counts = std::collections::HashMap::new();
            for &l in &labels {
                if l != NOISE {
                    *counts.entry(l).or_insert(0usize) += 1;
                }
            }
            for (&cluster, &count) in &counts {
                prop_assert!(
                    count >= min_cs,
                    "cluster {} has {} points < min_cluster_size {}",
                    cluster, count, min_cs
                );
            }
        }

        /// All points must be NOISE when min_cluster_size > n.
        #[test]
        fn all_noise_when_min_cluster_size_gt_n(data in arb_data(15, 2)) {
            let n = data.len();
            let labels = Hdbscan::new()
                .with_min_samples(2)
                .with_min_cluster_size(n + 1)
                .fit_predict(&data).unwrap();
            for (i, &l) in labels.iter().enumerate() {
                prop_assert_eq!(l, NOISE, "point {} should be NOISE when min_cluster_size > n", i);
            }
        }

        /// HDBSCAN is deterministic: same data gives same labels.
        #[test]
        fn deterministic(data in arb_data(15, 2)) {
            let hdb = Hdbscan::new().with_min_samples(2).with_min_cluster_size(3);
            let l1 = hdb.fit_predict(&data).unwrap();
            let l2 = hdb.fit_predict(&data).unwrap();
            prop_assert_eq!(&l1, &l2, "HDBSCAN must be deterministic");
        }

        /// All-identical points must not crash and must produce valid labels.
        #[test]
        fn all_identical_no_crash(
            point in proptest::collection::vec(-10.0f32..10.0, 2..=2),
            n in 3usize..10,
        ) {
            let data: Vec<Vec<f32>> = vec![point; n];
            let labels = Hdbscan::new()
                .with_min_samples(2)
                .with_min_cluster_size(2)
                .fit_predict(&data).unwrap();
            prop_assert_eq!(labels.len(), n);
            for &l in &labels {
                prop_assert!(l == NOISE || l < n, "label out of range: {}", l);
            }
        }

        /// Outlier scores from fit() must be in [0.0, 1.0] and finite.
        #[test]
        fn outlier_scores_bounded(data in arb_data(15, 2)) {
            let result = Hdbscan::new()
                .with_min_samples(2)
                .with_min_cluster_size(3)
                .fit(&data).unwrap();
            for (i, &s) in result.outlier_scores.iter().enumerate() {
                prop_assert!(s.is_finite(), "outlier_scores[{}] is not finite: {}", i, s);
                prop_assert!(
                    (0.0..=1.0).contains(&s),
                    "outlier_scores[{}] = {} not in [0,1]", i, s
                );
            }
        }
    }
}
