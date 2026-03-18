//! Constrained clustering algorithms.
//!
//! Semi-supervised clustering where pairwise constraints guide the assignment.
//! Constraints come from domain knowledge or limited labeled data: "these two
//! items must be together" or "these two items must be apart."
//!
//! # COP-Kmeans (Wagstaff et al., 2001)
//!
//! COP-Kmeans modifies the assignment step of standard k-means to respect
//! pairwise constraints. During each iteration:
//!
//! 1. **Assign**: For each point, find the nearest centroid whose assignment
//!    would not violate any cannot-link constraint. If a must-link partner
//!    has already been assigned, the point must join the same cluster.
//! 2. **Update**: Recompute centroids as the mean of assigned points (identical
//!    to standard k-means).
//!
//! If no valid assignment exists for a point (all candidate clusters violate
//! some constraint), the algorithm returns [`Error::ConstraintViolation`].
//!
//! ## Constraint Types
//!
//! - **Must-link**: Two points must belong to the same cluster. Transitive:
//!   if (a, b) and (b, c) are must-linked, then a, b, c are all co-clustered.
//! - **Cannot-link**: Two points must belong to different clusters.
//!
//! ## Limitations
//!
//! - Constraint feasibility is NP-complete in general. COP-Kmeans uses a
//!   greedy assignment order, so it may fail even when a feasible solution
//!   exists under a different ordering.
//! - Like standard k-means, it assumes roughly spherical clusters and
//!   requires k to be specified.
//!
//! ## References
//!
//! Wagstaff, K. et al. (2001). "Constrained K-means Clustering with
//! Background Knowledge." ICML 2001.

use super::distance::{DistanceMetric, SquaredEuclidean};
use super::util;
use crate::error::{Error, Result};
use rand::prelude::*;

/// A pairwise constraint for semi-supervised clustering.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Constraint {
    /// These two points must be in the same cluster.
    MustLink(usize, usize),
    /// These two points must be in different clusters.
    CannotLink(usize, usize),
}

/// COP-Kmeans: constrained k-means clustering (Wagstaff et al., 2001).
///
/// Standard k-means with pairwise must-link and cannot-link constraints
/// enforced during the assignment step.
///
/// ```
/// use clump::cluster::constrained::{CopKmeans, Constraint};
///
/// let data = vec![
///     vec![0.0f32, 0.0],
///     vec![0.1, 0.1],
///     vec![10.0, 10.0],
///     vec![10.1, 10.1],
/// ];
///
/// // Force points 0 and 1 together, points 0 and 2 apart.
/// let constraints = vec![
///     Constraint::MustLink(0, 1),
///     Constraint::CannotLink(0, 2),
/// ];
///
/// let labels = CopKmeans::new(2)
///     .with_seed(42)
///     .fit_predict_constrained(&data, &constraints)
///     .unwrap();
///
/// assert_eq!(labels[0], labels[1]);
/// assert_ne!(labels[0], labels[2]);
/// ```
#[derive(Debug, Clone)]
pub struct CopKmeans<D: DistanceMetric = SquaredEuclidean> {
    /// Number of clusters.
    k: usize,
    /// Maximum iterations.
    max_iter: usize,
    /// Convergence tolerance.
    tol: f64,
    /// Random seed.
    seed: Option<u64>,
    /// Distance metric.
    metric: D,
}

impl CopKmeans<SquaredEuclidean> {
    /// Create a new COP-Kmeans clusterer with default squared Euclidean distance.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 100,
            tol: 1e-4,
            seed: None,
            metric: SquaredEuclidean,
        }
    }
}

impl<D: DistanceMetric> CopKmeans<D> {
    /// Create a new COP-Kmeans clusterer with a custom distance metric.
    pub fn with_metric(k: usize, metric: D) -> Self {
        Self {
            k,
            max_iter: 100,
            tol: 1e-4,
            seed: None,
            metric,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Check whether assigning `point_idx` to `cluster` violates any cannot-link constraint.
    fn violates_cannot_link(
        point_idx: usize,
        cluster: usize,
        labels: &[Option<usize>],
        cannot_links: &[Vec<usize>],
    ) -> bool {
        for &other in &cannot_links[point_idx] {
            if labels[other] == Some(cluster) {
                return true;
            }
        }
        false
    }

    /// Find the cluster that a must-link partner has already been assigned to, if any.
    fn must_link_cluster(
        point_idx: usize,
        labels: &[Option<usize>],
        must_links: &[Vec<usize>],
    ) -> Option<usize> {
        for &other in &must_links[point_idx] {
            if let Some(c) = labels[other] {
                return Some(c);
            }
        }
        None
    }

    /// Compute transitive closure of must-link constraints via DFS.
    ///
    /// Returns groups of mutually must-linked indices.
    fn transitive_closure(adjacency: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
        let mut visited = vec![false; n];
        let mut groups = Vec::new();

        for start in 0..n {
            if visited[start] || adjacency[start].is_empty() {
                continue;
            }

            let mut group = Vec::new();
            let mut stack = vec![start];

            while let Some(node) = stack.pop() {
                if visited[node] {
                    continue;
                }
                visited[node] = true;
                group.push(node);

                for &neighbor in &adjacency[node] {
                    if !visited[neighbor] {
                        stack.push(neighbor);
                    }
                }
            }

            groups.push(group);
        }

        groups
    }

    /// Assign all points under must-link / cannot-link constraints.
    ///
    /// Processes points in `order`, assigning each to the nearest valid cluster.
    /// Returns `labels[i] = Some(cluster_id)` for every point, or an error if
    /// no feasible assignment exists.
    fn constrained_assign(
        &self,
        data: &[Vec<f32>],
        centroids: &[Vec<f32>],
        must_links: &[Vec<usize>],
        cannot_links: &[Vec<usize>],
        order: &[usize],
    ) -> Result<Vec<Option<usize>>> {
        let n = data.len();
        let mut labels: Vec<Option<usize>> = vec![None; n];
        // Pre-allocate candidate buffer outside the per-point loop.
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(self.k);

        for &i in order {
            // If a must-link partner is already assigned, force that cluster.
            if let Some(forced) = Self::must_link_cluster(i, &labels, must_links) {
                if Self::violates_cannot_link(i, forced, &labels, cannot_links) {
                    return Err(Error::ConstraintViolation(format!(
                        "point {i}: must-link forces cluster {forced} but cannot-link forbids it"
                    )));
                }
                labels[i] = Some(forced);
                continue;
            }

            // Sort candidate clusters by distance (nearest first).
            candidates.clear();
            candidates
                .extend((0..self.k).map(|k| (k, self.metric.distance(&data[i], &centroids[k]))));
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Pick nearest valid cluster.
            let mut assigned = false;
            for (k, _) in &candidates {
                if !Self::violates_cannot_link(i, *k, &labels, cannot_links) {
                    labels[i] = Some(*k);
                    assigned = true;
                    break;
                }
            }

            if !assigned {
                return Err(Error::ConstraintViolation(format!(
                    "point {i}: no valid cluster assignment exists"
                )));
            }
        }

        Ok(labels)
    }
}

impl<D: DistanceMetric> CopKmeans<D> {
    /// Fit the model with pairwise constraints.
    pub fn fit_predict_constrained(
        &self,
        data: &[Vec<f32>],
        constraints: &[Constraint],
    ) -> Result<Vec<usize>> {
        if data.is_empty() {
            return Err(Error::EmptyInput);
        }

        if self.k == 0 {
            return Err(Error::InvalidParameter {
                name: "k",
                message: "must be at least 1",
            });
        }

        let n = data.len();
        let d = data[0].len();

        if d == 0 {
            return Err(Error::InvalidParameter {
                name: "dimension",
                message: "must be at least 1",
            });
        }

        if self.k > n {
            return Err(Error::InvalidClusterCount {
                requested: self.k,
                n_items: n,
            });
        }

        // Validate dimensions.
        for p in data {
            if p.len() != d {
                return Err(Error::DimensionMismatch {
                    expected: d,
                    found: p.len(),
                });
            }
        }

        util::validate_finite(data)?;

        // Validate constraint indices.
        for c in constraints {
            let (a, b) = match c {
                Constraint::MustLink(a, b) | Constraint::CannotLink(a, b) => (*a, *b),
            };
            if a >= n || b >= n {
                return Err(Error::InvalidParameter {
                    name: "constraint index",
                    message: "exceeds dataset size",
                });
            }
        }

        // Build adjacency lists for fast lookup.
        let mut must_links: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut cannot_links: Vec<Vec<usize>> = vec![Vec::new(); n];

        for c in constraints {
            match c {
                Constraint::MustLink(a, b) => {
                    must_links[*a].push(*b);
                    must_links[*b].push(*a);
                }
                Constraint::CannotLink(a, b) => {
                    cannot_links[*a].push(*b);
                    cannot_links[*b].push(*a);
                }
            }
        }

        // Compute transitive closure of must-link constraints via DFS.
        // After this, must_links[i] contains all points transitively must-linked
        // to i (not just direct neighbors). This makes the assignment step O(1)
        // for must-link lookups instead of requiring repeated traversal.
        let must_link_groups = Self::transitive_closure(&must_links, n);
        let mut must_links_closed: Vec<Vec<usize>> = vec![Vec::new(); n];
        for group in &must_link_groups {
            for &member in group {
                for &other in group {
                    if other != member {
                        must_links_closed[member].push(other);
                    }
                }
            }
        }
        let must_links = must_links_closed;

        // Check for contradictions: a pair that is both must-link and cannot-link.
        for c in constraints {
            if let Constraint::CannotLink(a, b) = c {
                if must_links[*a].contains(b) {
                    return Err(Error::ConstraintViolation(format!(
                        "points {} and {} are both must-linked (transitively) and cannot-linked",
                        a, b
                    )));
                }
            }
        }

        // Initialize RNG.
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        // Initialize centroids via k-means++.
        let mut centroids = util::kmeanspp_init(data, self.k, &self.metric, 2.0, &mut rng);

        // Pre-allocate working buffers outside iteration loop.
        let mut new_centroids = vec![vec![0.0f32; d]; self.k];
        let mut counts = vec![0usize; self.k];
        let mut order: Vec<usize> = (0..n).collect();
        let effective_tol = (self.tol * util::mean_variance(data) * self.k as f64) as f32;

        for _ in 0..self.max_iter {
            // Assignment step (constrained).
            order.iter_mut().enumerate().for_each(|(i, v)| *v = i);
            order.shuffle(&mut rng);
            let labels =
                self.constrained_assign(data, &centroids, &must_links, &cannot_links, &order)?;

            // Update step: recompute centroids.
            for c in &mut new_centroids {
                c.fill(0.0);
            }
            counts.fill(0);

            for (i, label) in labels.iter().enumerate() {
                let k = label.unwrap(); // All points are assigned at this stage.
                for j in 0..d {
                    new_centroids[k][j] += data[i][j];
                }
                counts[k] += 1;
            }

            for k in 0..self.k {
                if counts[k] > 0 {
                    let divisor = counts[k] as f32;
                    for val in &mut new_centroids[k] {
                        *val /= divisor;
                    }
                } else {
                    // Empty cluster: reinitialize randomly.
                    let idx = rng.random_range(0..n);
                    new_centroids[k] = data[idx].clone();
                }
            }

            // Spherical k-means: L2-normalize centroids for cosine distance.
            if self.metric.normalize_centroids() {
                for c in &mut new_centroids {
                    let norm: f32 = c.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    if norm > f32::EPSILON {
                        for val in c.iter_mut() {
                            *val /= norm;
                        }
                    }
                }
            }

            // Convergence check.
            let shift: f32 = centroids
                .iter()
                .zip(new_centroids.iter())
                .flat_map(|(old, new)| old.iter().zip(new.iter()).map(|(a, b)| (a - b).powi(2)))
                .sum();

            std::mem::swap(&mut centroids, &mut new_centroids);

            if shift < effective_tol {
                break;
            }
        }

        // Final assignment with constraints.
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut rng);
        let labels =
            self.constrained_assign(data, &centroids, &must_links, &cannot_links, &order)?;

        Ok(labels.into_iter().map(|l| l.unwrap()).collect())
    }
}

#[cfg(test)]
mod autotrait_tests {
    use super::*;

    fn assert_autotraits<T: Send + Sync + Sized + Unpin>() {}

    #[test]
    fn cop_kmeans_is_send_sync() {
        assert_autotraits::<CopKmeans<SquaredEuclidean>>();
        assert_autotraits::<CopKmeans<super::super::distance::Euclidean>>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn must_link_forces_same_cluster() {
        let data = vec![
            vec![0.0f32, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let constraints = vec![Constraint::MustLink(0, 1)];

        let labels = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &constraints)
            .unwrap();

        assert_eq!(
            labels[0], labels[1],
            "must-linked points should share a cluster"
        );
    }

    #[test]
    fn cannot_link_forces_different_clusters() {
        let data = vec![
            vec![0.0f32, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let constraints = vec![Constraint::CannotLink(0, 1)];

        let labels = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &constraints)
            .unwrap();

        assert_ne!(
            labels[0], labels[1],
            "cannot-linked points should be in different clusters"
        );
    }

    #[test]
    fn must_link_transitive() {
        // If (0,1) must-link and (1,2) must-link, then 0,1,2 should all be co-clustered.
        let data = vec![
            vec![0.0f32, 0.0],
            vec![5.0, 5.0], // Midpoint -- would normally go to either cluster.
            vec![0.1, 0.1],
            vec![10.0, 10.0],
        ];

        let constraints = vec![Constraint::MustLink(0, 1), Constraint::MustLink(1, 2)];

        let labels = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &constraints)
            .unwrap();

        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn infeasible_constraints_return_error() {
        // Three points, k=2, all pairwise cannot-link: impossible.
        let data = vec![vec![0.0f32, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];

        let constraints = vec![
            Constraint::CannotLink(0, 1),
            Constraint::CannotLink(0, 2),
            Constraint::CannotLink(1, 2),
        ];

        let result = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &constraints);

        assert!(
            result.is_err(),
            "infeasible constraints should return error"
        );
        if let Err(Error::ConstraintViolation(msg)) = result {
            assert!(
                msg.contains("no valid cluster"),
                "error message should mention no valid cluster: {msg}"
            );
        }
    }

    #[test]
    fn conflicting_must_and_cannot_link_error() {
        // (0,1) must-link AND (0,1) cannot-link: contradictory.
        let data = vec![
            vec![0.0f32, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let constraints = vec![Constraint::MustLink(0, 1), Constraint::CannotLink(0, 1)];

        let result = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &constraints);

        assert!(result.is_err(), "contradictory constraints should fail");
    }

    #[test]
    fn no_constraints_matches_kmeans_structure() {
        let data = vec![
            vec![0.0f32, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let labels = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &[])
            .unwrap();

        // Well-separated data should still cluster correctly without constraints.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn empty_input_error() {
        let result = CopKmeans::new(2)
            .with_seed(1)
            .fit_predict_constrained(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_constraint_index_error() {
        let data = vec![vec![0.0f32, 0.0], vec![1.0, 1.0]];
        let constraints = vec![Constraint::MustLink(0, 5)]; // Index 5 out of bounds.

        let result = CopKmeans::new(2)
            .with_seed(1)
            .fit_predict_constrained(&data, &constraints);

        assert!(result.is_err());
    }

    #[test]
    fn with_custom_metric() {
        use crate::cluster::distance::Euclidean;

        let data = vec![
            vec![0.0f32, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let constraints = vec![Constraint::MustLink(0, 1), Constraint::CannotLink(0, 2)];

        let labels = CopKmeans::with_metric(2, Euclidean)
            .with_seed(42)
            .fit_predict_constrained(&data, &constraints)
            .unwrap();

        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
    }

    /// Early contradiction detection: contradictory must-link + cannot-link
    /// should be caught before any iteration, not as a runtime failure.
    #[test]
    fn transitive_contradiction_detected_early() {
        // (0,1) must-link, (1,2) must-link => 0,1,2 transitively linked.
        // (0,2) cannot-link => contradicts the transitive must-link.
        let data = vec![
            vec![0.0f32, 0.0],
            vec![0.1, 0.1],
            vec![0.2, 0.2],
            vec![10.0, 10.0],
        ];

        let constraints = vec![
            Constraint::MustLink(0, 1),
            Constraint::MustLink(1, 2),
            Constraint::CannotLink(0, 2),
        ];

        let result = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &constraints);

        assert!(result.is_err(), "transitive contradiction should be caught");
        if let Err(Error::ConstraintViolation(msg)) = result {
            assert!(
                msg.contains("must-linked") && msg.contains("cannot-linked"),
                "error should mention both constraint types: {msg}"
            );
        }
    }

    /// Monotonicity: adding a must-link between two points already in the
    /// same cluster should not change the partition.
    #[test]
    fn monotonicity_same_cluster_must_link() {
        let data = vec![
            vec![0.0f32, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        // Without constraints.
        let labels_none = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &[])
            .unwrap();

        // Points 0 and 1 should already be co-clustered.
        assert_eq!(labels_none[0], labels_none[1]);

        // Adding a must-link between already-co-clustered points.
        let labels_with = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &[Constraint::MustLink(0, 1)])
            .unwrap();

        // Partition structure should be identical (labels may differ in absolute value).
        assert_eq!(
            labels_none[0] == labels_none[2],
            labels_with[0] == labels_with[2]
        );
        assert_eq!(
            labels_none[0] == labels_none[3],
            labels_with[0] == labels_with[3]
        );
    }

    #[test]
    fn deterministic_with_seed() {
        let data = vec![
            vec![0.0f32, 0.0],
            vec![0.1, 0.1],
            vec![5.0, 5.0],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let constraints = vec![Constraint::MustLink(0, 1), Constraint::CannotLink(0, 3)];

        let labels1 = CopKmeans::new(2)
            .with_seed(99)
            .fit_predict_constrained(&data, &constraints)
            .unwrap();

        let labels2 = CopKmeans::new(2)
            .with_seed(99)
            .fit_predict_constrained(&data, &constraints)
            .unwrap();

        assert_eq!(
            labels1, labels2,
            "same seed should produce identical results"
        );
    }

    #[test]
    fn nan_input_rejected() {
        let data = vec![vec![0.0, f32::NAN], vec![1.0, 1.0]];
        let result = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn inf_input_rejected() {
        let data = vec![vec![0.0, 0.0], vec![f32::INFINITY, 1.0]];
        let result = CopKmeans::new(2)
            .with_seed(42)
            .fit_predict_constrained(&data, &[]);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Generate well-separated cluster data where constraints are satisfiable.
    fn separated_data_and_constraints() -> impl Strategy<Value = (Vec<Vec<f32>>, Vec<Constraint>)> {
        // 4 points: 2 near origin, 2 near (100, 100).
        // Must-link within each pair, cannot-link across.
        Just((
            vec![
                vec![0.0, 0.0],
                vec![0.1, 0.1],
                vec![100.0, 100.0],
                vec![100.1, 100.1],
            ],
            vec![
                Constraint::MustLink(0, 1),
                Constraint::MustLink(2, 3),
                Constraint::CannotLink(0, 2),
            ],
        ))
    }

    proptest! {
        #[test]
        fn must_links_satisfied((data, constraints) in separated_data_and_constraints()) {
            let labels = CopKmeans::new(2)
                .with_seed(42)
                .fit_predict_constrained(&data, &constraints)
                .unwrap();

            for c in &constraints {
                if let Constraint::MustLink(a, b) = c {
                    prop_assert_eq!(
                        labels[*a], labels[*b],
                        "must-link ({}, {}) violated", a, b
                    );
                }
            }
        }

        #[test]
        fn cannot_links_satisfied((data, constraints) in separated_data_and_constraints()) {
            let labels = CopKmeans::new(2)
                .with_seed(42)
                .fit_predict_constrained(&data, &constraints)
                .unwrap();

            for c in &constraints {
                if let Constraint::CannotLink(a, b) = c {
                    prop_assert_ne!(
                        labels[*a], labels[*b],
                        "cannot-link ({}, {}) violated", a, b
                    );
                }
            }
        }

        #[test]
        fn labels_in_range((data, constraints) in separated_data_and_constraints()) {
            let labels = CopKmeans::new(2)
                .with_seed(42)
                .fit_predict_constrained(&data, &constraints)
                .unwrap();

            for (i, &l) in labels.iter().enumerate() {
                prop_assert!(l < 2, "point {i}: label {l} >= k=2");
            }
        }
    }
}
