//! Random projection index for fast approximate range queries in moderate-to-high d.
//!
//! Projects all points onto k random unit vectors and maintains sorted arrays.
//! Range queries use binary search on each projection to find candidates, then
//! intersect across projections. By Cauchy-Schwarz, if `||p - q|| < eps` then
//! `|dot(p-q, r)| < eps` for any unit vector r, so each projection is a valid
//! filter.
//!
//! Effective for d=10-50 where grid hashing (3^d cells) and VP-trees degrade.
//! O(n * k) build, O(k * log n + |candidates| * d) per query.

use super::distance::DistanceMetric;
use super::flat::DataRef;
use rand::prelude::*;

/// A random projection index with k projection axes.
pub(crate) struct ProjIndex<'a, T: DataRef + ?Sized, D: DistanceMetric> {
    /// Original data reference.
    data: &'a T,
    metric: &'a D,
    /// k random unit vectors, each of dimension d.
    axes: Vec<Vec<f32>>,
    /// For each axis: sorted (projected_value, original_index) pairs.
    sorted_projs: Vec<Vec<(f32, usize)>>,
}

impl<'a, T: DataRef + ?Sized, D: DistanceMetric> ProjIndex<'a, T, D> {
    /// Build a projection index with `num_axes` random projections.
    /// More axes = better pruning but more build/query time. 8-16 is typical.
    pub(crate) fn new(data: &'a T, metric: &'a D, num_axes: usize) -> Self {
        let n = data.n();
        if n == 0 {
            return Self {
                data,
                metric,
                axes: Vec::new(),
                sorted_projs: Vec::new(),
            };
        }
        let d = data.d();
        let mut rng = StdRng::seed_from_u64(12345);

        let axes: Vec<Vec<f32>> = (0..num_axes)
            .map(|_| {
                let mut v: Vec<f32> = (0..d).map(|_| rng.random::<f32>() - 0.5).collect();
                let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm > f32::EPSILON {
                    for x in &mut v {
                        *x /= norm;
                    }
                }
                v
            })
            .collect();

        let sorted_projs: Vec<Vec<(f32, usize)>> = axes
            .iter()
            .map(|axis| {
                let mut projs: Vec<(f32, usize)> = (0..n)
                    .map(|i| {
                        let proj: f32 = data
                            .row(i)
                            .iter()
                            .zip(axis.iter())
                            .map(|(&x, &a)| x * a)
                            .sum();
                        (proj, i)
                    })
                    .collect();
                projs.sort_by(|a, b| a.0.total_cmp(&b.0));
                projs
            })
            .collect();

        Self {
            data,
            metric,
            axes,
            sorted_projs,
        }
    }

    /// Find all points within `radius` of `query`.
    pub(crate) fn range_query(&self, query: &[f32], radius: f32) -> Vec<usize> {
        if self.sorted_projs.is_empty() {
            return Vec::new();
        }

        let n = self.data.n();
        let mut candidate_counts = vec![0u16; n];
        let num_axes = self.axes.len();

        for (axis_idx, axis) in self.axes.iter().enumerate() {
            let proj_q: f32 = query.iter().zip(axis.iter()).map(|(&x, &a)| x * a).sum();
            let lo = proj_q - radius;
            let hi = proj_q + radius;
            let sorted = &self.sorted_projs[axis_idx];
            let start = sorted.partition_point(|&(v, _)| v < lo);
            let end = sorted.partition_point(|&(v, _)| v <= hi);
            for &(_, idx) in &sorted[start..end] {
                candidate_counts[idx] += 1;
            }
        }

        let mut results = Vec::new();
        for (i, &count) in candidate_counts.iter().enumerate() {
            if count == num_axes as u16 && self.metric.distance(query, self.data.row(i)) <= radius {
                results.push(i);
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::distance::Euclidean;

    #[test]
    fn range_query_correctness() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let index = ProjIndex::new(data.as_slice(), &Euclidean, 8);
        let mut results = index.range_query(&[0.0, 0.0], 0.5);
        results.sort();
        assert!(results.contains(&0));
        assert!(results.contains(&1));
        assert!(!results.contains(&2));
    }

    #[test]
    fn empty_data() {
        let data: Vec<Vec<f32>> = vec![];
        let index = ProjIndex::new(data.as_slice(), &Euclidean, 8);
        assert!(index.range_query(&[0.0], 1.0).is_empty());
    }

    #[test]
    fn high_dimensional() {
        let mut rng = StdRng::seed_from_u64(42);
        let data: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..20).map(|_| rng.random::<f32>()).collect())
            .collect();
        let index = ProjIndex::new(data.as_slice(), &Euclidean, 12);

        let query = &data[0];
        let radius = 1.0;
        let mut brute: Vec<usize> = (0..100)
            .filter(|&i| Euclidean.distance(query, &data[i]) <= radius)
            .collect();
        brute.sort();

        let mut indexed = index.range_query(query, radius);
        indexed.sort();

        for &b in &brute {
            assert!(indexed.contains(&b), "projection index missed neighbor {b}");
        }
    }
}
