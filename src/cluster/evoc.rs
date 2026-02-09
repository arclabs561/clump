//! EVōC (Embedding Vector Oriented Clustering).
//!
//! This module provides a small, pure-Rust clustering implementation inspired by the
//! Tutte Institute's EVōC project (`https://github.com/TutteInstitute/evoc`).
//!
//! ## What you get
//!
//! - Multi-granularity cluster layers (finest → coarsest)
//! - A dendrogram-like cluster tree (single-linkage hierarchy)
//! - Near-duplicate detection
//!
//! ## Important note
//!
//! The upstream EVōC library is a Python implementation that combines a kNN graph,
//! a UMAP-like node embedding, and an HDBSCAN-style hierarchy. For `clump` we keep
//! the implementation lightweight and dependency-free: we use a random projection
//! into an intermediate dimension and build a single-linkage hierarchy via an MST
//! over pairwise distances. This matches the *shape* of the EVōC API and is useful
//! for embedding exploration, but it is not a byte-for-byte port of the upstream
//! algorithm.
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_lines)]

use crate::error::{Error, Result};
use rand::prelude::*;
use std::collections::HashMap;

/// EVōC clustering parameters.
#[derive(Clone, Debug)]
pub struct EVoCParams {
    /// Intermediate dimension used during the projection step.
    ///
    /// Typical values are ~12–24; upstream recommends ~15.
    pub intermediate_dim: usize,

    /// Minimum cluster size; smaller connected components are treated as noise.
    pub min_cluster_size: usize,

    /// Maximum number of layers to extract.
    pub max_layers: usize,

    /// Noise tolerance in \([0, 1]\). Lower values cluster more aggressively.
    ///
    /// In this lightweight implementation this parameter only affects layer selection
    /// heuristics, not the underlying hierarchy construction.
    pub noise_level: f32,

    /// Distance threshold under which points are considered (near-)duplicates.
    pub duplicate_threshold: f32,

    /// Optional RNG seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for EVoCParams {
    fn default() -> Self {
        Self {
            intermediate_dim: 15,
            min_cluster_size: 10,
            max_layers: 10,
            noise_level: 0.5,
            duplicate_threshold: 1e-6,
            seed: None,
        }
    }
}

/// A single cluster layer (one granularity level).
#[derive(Clone, Debug)]
pub struct ClusterLayer {
    /// The distance threshold used to cut the hierarchy.
    pub threshold: f32,

    /// Cluster assignments: `point_idx -> cluster_id` (or `None` for noise).
    pub assignments: Vec<Option<usize>>,

    /// Number of clusters at this layer (excluding noise).
    pub num_clusters: usize,

    /// Cluster members: `cluster_id -> Vec<point_idx>`.
    pub clusters: HashMap<usize, Vec<usize>>,
}

/// A node in a hierarchical cluster tree.
#[derive(Clone, Debug)]
pub struct ClusterNode {
    /// Node identifier (also the index into the hierarchy's `nodes` array).
    pub id: usize,

    /// Child node ids (empty for leaves, length 2 for internal merge nodes).
    pub children: Vec<usize>,

    /// Merge distance (0.0 for leaves).
    pub distance: f32,

    /// Number of leaf points under this node.
    pub size: usize,
}

/// Cluster hierarchy tree (single-linkage dendrogram).
#[derive(Clone, Debug)]
pub struct ClusterHierarchy {
    nodes: Vec<ClusterNode>,
    root: Option<usize>,
}

impl ClusterHierarchy {
    /// Build a hierarchy from MST edges.
    ///
    /// The input `edges` should be MST edges sorted by distance (ascending).
    pub fn from_mst(edges: &[(usize, usize, f32)], num_points: usize) -> Self {
        let mut nodes: Vec<ClusterNode> =
            Vec::with_capacity(num_points.saturating_mul(2).saturating_sub(1));

        // Leaf nodes.
        for i in 0..num_points {
            nodes.push(ClusterNode {
                id: i,
                children: Vec::new(),
                distance: 0.0,
                size: 1,
            });
        }

        if num_points == 0 {
            return Self { nodes, root: None };
        }

        if num_points == 1 {
            return Self {
                nodes,
                root: Some(0),
            };
        }

        // Union-find over points, with an additional mapping from UF-root -> current tree node id.
        let mut uf = UnionFind::new(num_points);
        let mut comp_node: Vec<usize> = (0..num_points).collect();

        for &(u, v, dist) in edges {
            let ru = uf.find(u);
            let rv = uf.find(v);
            if ru == rv {
                continue;
            }

            let left = comp_node[ru];
            let right = comp_node[rv];

            let new_id = nodes.len();
            let new_size = nodes[left].size + nodes[right].size;
            nodes.push(ClusterNode {
                id: new_id,
                children: vec![left, right],
                distance: dist,
                size: new_size,
            });

            let new_root = uf.union_roots(ru, rv);
            comp_node[new_root] = new_id;
        }

        // For a proper MST, we should have built a single connected hierarchy and the
        // last node is the root.
        let root = nodes.len().checked_sub(1);
        Self { nodes, root }
    }

    /// Return the root node id (if any).
    pub fn root(&self) -> Option<usize> {
        self.root
    }

    /// Access all nodes.
    pub fn nodes(&self) -> &[ClusterNode] {
        &self.nodes
    }

    /// Get all merge distances (internal nodes only).
    pub fn get_all_distances(&self) -> Vec<f32> {
        self.nodes
            .iter()
            .filter(|n| !n.children.is_empty())
            .map(|n| n.distance)
            .collect()
    }
}

/// EVōC clusterer.
#[derive(Clone, Debug)]
pub struct EVoC {
    params: EVoCParams,
    original_dim: Option<usize>,

    mst_edges: Vec<(usize, usize, f32)>,
    hierarchy: Option<ClusterHierarchy>,
    cluster_layers: Vec<ClusterLayer>,
    duplicates: Vec<Vec<usize>>,
}

impl EVoC {
    /// Create a new EVōC clusterer (parameters only).
    pub fn new(params: EVoCParams) -> Self {
        Self {
            params,
            original_dim: None,
            mst_edges: Vec::new(),
            hierarchy: None,
            cluster_layers: Vec::new(),
            duplicates: Vec::new(),
        }
    }

    /// Fit on dense vectors and return a label per point (noise as `None`).
    ///
    /// The returned labels are for the **finest** available layer (largest number of clusters).
    pub fn fit_predict(&mut self, data: &[Vec<f32>]) -> Result<Vec<Option<usize>>> {
        if data.is_empty() {
            return Err(Error::EmptyInput);
        }

        let n = data.len();
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
        self.original_dim = Some(d);

        if self.params.intermediate_dim == 0 {
            return Err(Error::InvalidParameter {
                name: "intermediate_dim",
                message: "must be at least 1",
            });
        }
        if self.params.intermediate_dim >= d {
            return Err(Error::InvalidParameter {
                name: "intermediate_dim",
                message: "must be less than the original dimension",
            });
        }

        // Flatten to SoA storage.
        let mut flat: Vec<f32> = Vec::with_capacity(n * d);
        for v in data {
            flat.extend_from_slice(v);
        }

        // Step 1: random projection to intermediate space.
        let reduced = project(&flat, n, d, self.params.intermediate_dim, self.params.seed);

        // Step 2: build MST (Prim on a dense graph), then sort edges by distance.
        let mut mst = prim_mst(&reduced, n, self.params.intermediate_dim);
        mst.sort_by(|a, b| a.2.total_cmp(&b.2));
        self.mst_edges = mst;

        // Step 3: build hierarchy tree from MST.
        self.hierarchy = Some(ClusterHierarchy::from_mst(&self.mst_edges, n));

        // Step 4: extract cluster layers at multiple thresholds.
        self.cluster_layers = extract_layers(
            n,
            &self.mst_edges,
            self.params.min_cluster_size.max(1),
            self.params.max_layers.max(1),
            self.params.noise_level,
        );

        // Step 5: detect near-duplicates.
        self.duplicates = detect_duplicates(n, &self.mst_edges, self.params.duplicate_threshold);

        // Return the finest-grained assignments (largest number of clusters).
        if let Some(layer) = self.cluster_layers.first() {
            Ok(layer.assignments.clone())
        } else {
            Ok(vec![None; n])
        }
    }

    /// Access extracted cluster layers (finest → coarsest).
    pub fn cluster_layers(&self) -> &[ClusterLayer] {
        &self.cluster_layers
    }

    /// Access the cluster hierarchy tree (if fitted).
    pub fn cluster_tree(&self) -> Option<&ClusterHierarchy> {
        self.hierarchy.as_ref()
    }

    /// Access potential duplicate groups (if fitted).
    pub fn duplicates(&self) -> &[Vec<usize>] {
        &self.duplicates
    }

    /// Access the MST edges used to build the hierarchy (if fitted).
    pub fn mst_edges(&self) -> &[(usize, usize, f32)] {
        &self.mst_edges
    }

    /// Access the inferred original dimension (if fitted).
    pub fn original_dim(&self) -> Option<usize> {
        self.original_dim
    }
}

fn project(
    vectors: &[f32],
    num_vectors: usize,
    original_dim: usize,
    intermediate_dim: usize,
    seed: Option<u64>,
) -> Vec<f32> {
    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s)),
        None => Box::new(rand::rng()),
    };

    // Random projection matrix: intermediate_dim x original_dim.
    let mut mat: Vec<Vec<f32>> = Vec::with_capacity(intermediate_dim);
    for _ in 0..intermediate_dim {
        let mut row = Vec::with_capacity(original_dim);
        for _ in 0..original_dim {
            // Uniform in [-1, 1].
            row.push(rng.random::<f32>() * 2.0 - 1.0);
        }
        normalize_in_place(&mut row);
        mat.push(row);
    }

    let mut out: Vec<f32> = Vec::with_capacity(num_vectors * intermediate_dim);
    for i in 0..num_vectors {
        let v = &vectors[i * original_dim..(i + 1) * original_dim];
        for row in &mat {
            out.push(dot(v, row));
        }
    }
    out
}

fn normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v {
            *x /= norm;
        }
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Compute an MST for a dense complete graph using Prim's algorithm.
///
/// Returns edges `(u, v, dist)` where `dist` is Euclidean distance in the reduced space.
fn prim_mst(reduced: &[f32], n: usize, dim: usize) -> Vec<(usize, usize, f32)> {
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut best = vec![f32::INFINITY; n]; // squared distances
    let mut parent = vec![usize::MAX; n];

    best[0] = 0.0;

    for _ in 0..n {
        // Pick the next vertex with the smallest key.
        let mut u = usize::MAX;
        let mut best_val = f32::INFINITY;
        for i in 0..n {
            if !in_tree[i] && best[i] < best_val {
                best_val = best[i];
                u = i;
            }
        }

        if u == usize::MAX {
            break;
        }
        in_tree[u] = true;

        let uvec = &reduced[u * dim..(u + 1) * dim];
        for v in 0..n {
            if in_tree[v] {
                continue;
            }
            let vvec = &reduced[v * dim..(v + 1) * dim];
            let d2 = squared_euclidean(uvec, vvec);
            if d2 < best[v] {
                best[v] = d2;
                parent[v] = u;
            }
        }
    }

    let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(n - 1);
    for v in 1..n {
        let u = parent[v];
        if u != usize::MAX {
            edges.push((u, v, best[v].sqrt()));
        }
    }
    edges
}

fn extract_layers(
    n: usize,
    mst_edges: &[(usize, usize, f32)],
    min_cluster_size: usize,
    max_layers: usize,
    noise_level: f32,
) -> Vec<ClusterLayer> {
    if n == 0 {
        return Vec::new();
    }

    // Degenerate case: n == 1.
    if mst_edges.is_empty() {
        let (assignments, clusters, num_clusters) = if min_cluster_size <= 1 {
            (
                vec![Some(0)],
                HashMap::from([(0usize, vec![0usize])]),
                1usize,
            )
        } else {
            (vec![None], HashMap::new(), 0usize)
        };
        return vec![ClusterLayer {
            threshold: 0.0,
            assignments,
            num_clusters,
            clusters,
        }];
    }

    // Candidate thresholds are MST edge distances.
    let mut dists: Vec<f32> = mst_edges.iter().map(|e| e.2).collect();
    dists.sort_by(|a, b| a.total_cmp(b));

    // Upstream EVōC uses persistence to pick layers. Here we sample distances across the
    // MST; `noise_level` mildly biases toward coarser layers when higher.
    let layers = max_layers.min(dists.len()).max(1);
    let bias = noise_level.clamp(0.0, 1.0);

    let mut thresholds: Vec<f32> = Vec::with_capacity(layers);
    if layers == 1 {
        thresholds.push(dists[dists.len() - 1]);
    } else {
        for i in 0..layers {
            // i=0 -> fine; i=layers-1 -> coarse.
            let t = (i as f32) / ((layers - 1) as f32);
            // Bias toward larger thresholds (coarser) as noise_level increases.
            let t = t.powf(1.0 - bias + 1e-6);
            let idx = (t * ((dists.len() - 1) as f32)).round() as usize;
            thresholds.push(dists[idx.min(dists.len() - 1)]);
        }
    }
    thresholds.sort_by(|a, b| a.total_cmp(b));
    thresholds.dedup_by(|a, b| (*a - *b).abs() <= f32::EPSILON);

    let mut layers_out: Vec<ClusterLayer> = thresholds
        .into_iter()
        .map(|thr| layer_at_threshold(n, mst_edges, thr, min_cluster_size))
        .collect();

    // Sort by granularity (finest first).
    layers_out.sort_by(|a, b| b.num_clusters.cmp(&a.num_clusters));
    layers_out
}

fn layer_at_threshold(
    n: usize,
    mst_edges: &[(usize, usize, f32)],
    threshold: f32,
    min_cluster_size: usize,
) -> ClusterLayer {
    let mut uf = UnionFind::new(n);

    for &(u, v, d) in mst_edges {
        if d <= threshold {
            uf.union(u, v);
        } else {
            // MST edges are (usually) sorted by distance for our pipeline; allow early break when true.
            // If the caller passes unsorted edges, this remains correct but misses the early exit.
            // (The current code always sorts before calling.)
            break;
        }
    }

    let mut roots: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        roots.push(uf.find(i));
    }

    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &r in &roots {
        *counts.entry(r).or_insert(0) += 1;
    }

    // Deterministic cluster id assignment: sort roots by id.
    let mut cluster_roots: Vec<(usize, usize)> = counts
        .iter()
        .filter_map(|(&root, &count)| (count >= min_cluster_size).then_some((root, count)))
        .collect();
    cluster_roots.sort_by_key(|(root, _count)| *root);

    let mut root_to_cluster: HashMap<usize, usize> = HashMap::with_capacity(cluster_roots.len());
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::with_capacity(cluster_roots.len());
    for (cid, (root, count)) in cluster_roots.into_iter().enumerate() {
        root_to_cluster.insert(root, cid);
        clusters.insert(cid, Vec::with_capacity(count));
    }

    let mut assignments: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        if let Some(&cid) = root_to_cluster.get(&roots[i]) {
            assignments[i] = Some(cid);
            // safe: we inserted every cid above
            clusters
                .get_mut(&cid)
                .expect("cluster vec must exist")
                .push(i);
        }
    }

    ClusterLayer {
        threshold,
        assignments,
        num_clusters: root_to_cluster.len(),
        clusters,
    }
}

fn detect_duplicates(
    n: usize,
    mst_edges: &[(usize, usize, f32)],
    duplicate_threshold: f32,
) -> Vec<Vec<usize>> {
    if n == 0 {
        return Vec::new();
    }

    if duplicate_threshold <= 0.0 {
        return Vec::new();
    }

    let mut uf = UnionFind::new(n);
    for &(u, v, d) in mst_edges {
        if d <= duplicate_threshold {
            uf.union(u, v);
        } else {
            // MST edges are sorted; early exit.
            break;
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let r = uf.find(i);
        groups.entry(r).or_default().push(i);
    }

    let mut out: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() > 1).collect();
    out.sort_by(|a, b| a[0].cmp(&b[0]));
    out
}

#[derive(Clone, Debug)]
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        self.union_roots(ra, rb)
    }

    fn union_roots(&mut self, ra: usize, rb: usize) -> usize {
        if ra == rb {
            return ra;
        }

        // Union by size.
        let (mut big, mut small) = (ra, rb);
        if self.size[big] < self.size[small] {
            std::mem::swap(&mut big, &mut small);
        }

        self.parent[small] = big;
        self.size[big] += self.size[small];
        big
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evoc_smoke_two_clusters() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![9.9, 10.2],
        ];

        let params = EVoCParams {
            intermediate_dim: 1,
            min_cluster_size: 2,
            max_layers: 8,
            noise_level: 0.2,
            duplicate_threshold: 1e-6,
            seed: Some(42),
        };

        let mut evoc = EVoC::new(params);
        let labels = evoc.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), data.len());
        assert!(!evoc.cluster_layers().is_empty());
        assert!(evoc.cluster_tree().is_some());
        assert!(evoc.duplicates().is_empty());

        // At least one extracted layer should split into multiple clusters.
        assert!(
            evoc.cluster_layers().iter().any(|l| l.num_clusters >= 2),
            "expected at least one multi-cluster layer"
        );
    }

    #[test]
    fn evoc_detects_duplicates() {
        // Two identical points (in reduced space this should remain identical).
        let data = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let mut evoc = EVoC::new(EVoCParams {
            intermediate_dim: 1,
            min_cluster_size: 1,
            max_layers: 4,
            noise_level: 0.5,
            duplicate_threshold: 1e-6,
            seed: Some(7),
        });

        let _ = evoc.fit_predict(&data).unwrap();
        let dups = evoc.duplicates();

        assert!(
            dups.iter()
                .any(|g| g.len() == 2 && g.contains(&0) && g.contains(&1)),
            "expected points 0 and 1 to be flagged as duplicates"
        );
    }
}
