//! Vantage-point tree for fast nearest-neighbor and range queries.
//!
//! A VP-tree partitions points by their distance to a randomly chosen
//! "vantage point." Points closer than the median distance go left;
//! farther points go right. This gives O(log n) average-case queries
//! for well-distributed data, with O(n log n) construction.
//!
//! Used internally by DBSCAN (range queries) and HDBSCAN (k-nearest
//! neighbor queries) to avoid O(n^2) pairwise distance computation.

use super::distance::DistanceMetric;
use rand::prelude::*;

/// A node in the VP-tree.
struct VpNode {
    /// Index of the vantage point in the original data.
    point_idx: usize,
    /// Median distance to the vantage point (split threshold).
    threshold: f32,
    /// Left subtree: points with distance <= threshold.
    left: Option<Box<VpNode>>,
    /// Right subtree: points with distance > threshold.
    right: Option<Box<VpNode>>,
}

/// Vantage-point tree for O(log n) distance-based queries.
pub(crate) struct VpTree<'a, D: DistanceMetric> {
    root: Option<Box<VpNode>>,
    data: &'a [Vec<f32>],
    metric: &'a D,
}

#[allow(dead_code)]
impl<'a, D: DistanceMetric> VpTree<'a, D> {
    /// Build a VP-tree from the given data. O(n log n) average.
    pub(crate) fn new(data: &'a [Vec<f32>], metric: &'a D) -> Self {
        let n = data.len();
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = StdRng::seed_from_u64(42);
        let root = Self::build(&mut indices, data, metric, &mut rng);
        Self { root, data, metric }
    }

    fn build(
        indices: &mut [usize],
        data: &[Vec<f32>],
        metric: &D,
        rng: &mut StdRng,
    ) -> Option<Box<VpNode>> {
        if indices.is_empty() {
            return None;
        }
        if indices.len() == 1 {
            return Some(Box::new(VpNode {
                point_idx: indices[0],
                threshold: 0.0,
                left: None,
                right: None,
            }));
        }

        // Choose a random vantage point (swap to front).
        let vp_pos = rng.random_range(0..indices.len());
        indices.swap(0, vp_pos);
        let vp_idx = indices[0];

        // Compute distances from vantage point to all other points.
        let rest = &mut indices[1..];
        let mut dists: Vec<(usize, f32)> = rest
            .iter()
            .map(|&i| (i, metric.distance(&data[vp_idx], &data[i])))
            .collect();

        // Find median distance.
        let mid = dists.len() / 2;
        dists.select_nth_unstable_by(mid, |a, b| a.1.total_cmp(&b.1));
        let threshold = dists[mid].1;

        // Partition into left (<= median) and right (> median).
        let mut left_indices: Vec<usize> = Vec::with_capacity(mid + 1);
        let mut right_indices: Vec<usize> = Vec::with_capacity(dists.len() - mid);
        for &(idx, dist) in &dists {
            if dist <= threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        let left = Self::build(&mut left_indices, data, metric, rng);
        let right = Self::build(&mut right_indices, data, metric, rng);

        Some(Box::new(VpNode {
            point_idx: vp_idx,
            threshold,
            left,
            right,
        }))
    }

    /// Find all points within `radius` of `query`. O(log n) average for
    /// well-distributed data, O(n) worst case.
    pub(crate) fn range_query(&self, query: &[f32], radius: f32) -> Vec<usize> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.range_search(root, query, radius, &mut results);
        }
        results
    }

    fn range_search(&self, node: &VpNode, query: &[f32], radius: f32, results: &mut Vec<usize>) {
        let dist = self.metric.distance(query, &self.data[node.point_idx]);

        if dist <= radius {
            results.push(node.point_idx);
        }

        // Prune: if the query ball doesn't intersect the left subtree's
        // region (dist - radius > threshold), skip it.
        if dist - radius <= node.threshold {
            if let Some(ref left) = node.left {
                self.range_search(left, query, radius, results);
            }
        }

        // Prune: if the query ball doesn't intersect the right subtree's
        // region (dist + radius <= threshold), skip it.
        if dist + radius > node.threshold {
            if let Some(ref right) = node.right {
                self.range_search(right, query, radius, results);
            }
        }
    }

    /// Find the k nearest neighbors of `query`. Returns (index, distance)
    /// pairs sorted by distance.
    pub(crate) fn knn(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut heap: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
        let mut tau = f32::MAX; // current k-th nearest distance
        if let Some(ref root) = self.root {
            self.knn_search(root, query, k, &mut heap, &mut tau);
        }
        heap.sort_by(|a, b| a.1.total_cmp(&b.1));
        heap
    }

    fn knn_search(
        &self,
        node: &VpNode,
        query: &[f32],
        k: usize,
        heap: &mut Vec<(usize, f32)>,
        tau: &mut f32,
    ) {
        let dist = self.metric.distance(query, &self.data[node.point_idx]);

        if heap.len() < k {
            heap.push((node.point_idx, dist));
            if heap.len() == k {
                // Find the current k-th nearest.
                *tau = heap.iter().map(|(_, d)| *d).fold(0.0f32, f32::max);
            }
        } else if dist < *tau {
            // Replace the farthest point in the heap.
            let max_pos = heap
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
                .unwrap();
            heap[max_pos] = (node.point_idx, dist);
            *tau = heap.iter().map(|(_, d)| *d).fold(0.0f32, f32::max);
        }

        // Search the subtree that contains the query point first (closer).
        let (first, second) = if dist <= node.threshold {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref child) = first {
            // Always search the near subtree.
            if dist <= node.threshold {
                // Query is in the left region; search left if it could contain closer points.
                if dist - *tau <= node.threshold {
                    self.knn_search(child, query, k, heap, tau);
                }
            } else {
                // Query is in the right region.
                if dist + *tau > node.threshold {
                    self.knn_search(child, query, k, heap, tau);
                }
            }
        }

        if let Some(ref child) = second {
            if dist <= node.threshold {
                // Query in left; search right if ball crosses threshold.
                if dist + *tau > node.threshold {
                    self.knn_search(child, query, k, heap, tau);
                }
            } else {
                // Query in right; search left if ball crosses threshold.
                if dist - *tau <= node.threshold {
                    self.knn_search(child, query, k, heap, tau);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::distance::Euclidean;

    #[test]
    fn range_query_finds_neighbors() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let tree = VpTree::new(&data, &Euclidean);
        let mut results = tree.range_query(&[0.0, 0.0], 0.5);
        results.sort();
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
    }

    #[test]
    fn knn_finds_nearest() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
        ];
        let tree = VpTree::new(&data, &Euclidean);
        let results = tree.knn(&[0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // self
        assert_eq!(results[1].0, 1); // nearest
    }

    #[test]
    fn empty_data() {
        let data: Vec<Vec<f32>> = vec![];
        let tree = VpTree::new(&data, &Euclidean);
        assert!(tree.range_query(&[0.0], 1.0).is_empty());
        assert!(tree.knn(&[0.0], 1).is_empty());
    }

    #[test]
    fn single_point() {
        let data = vec![vec![5.0, 5.0]];
        let tree = VpTree::new(&data, &Euclidean);
        let results = tree.range_query(&[5.0, 5.0], 0.1);
        assert_eq!(results, vec![0]);
    }
}
