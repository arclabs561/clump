//! Streaming (online) clustering algorithms.
//!
//! Unlike batch clustering (e.g. [`Kmeans`](super::Kmeans)) which requires
//! all data upfront, streaming algorithms process data incrementally. This is
//! useful when data arrives in a continuous stream, when the full dataset does
//! not fit in memory, or when you need to update clusters as new observations
//! arrive.
//!
//! # Mini-Batch K-means (Sculley, 2010)
//!
//! A stochastic variant of k-means that processes small random batches instead
//! of the full dataset on each iteration. The key insight: using a decaying
//! learning rate `eta = 1 / count[k]` for each centroid converges to the same
//! objective as batch k-means, with much lower per-iteration cost.
//!
//! ## Update Rule
//!
//! For a point `x` assigned to cluster `k`:
//! ```text
//! count[k] += 1
//! eta = 1 / count[k]
//! centroid[k] = (1 - eta) * centroid[k] + eta * x
//! ```
//!
//! This is an online running average that gives equal weight to every point
//! ever assigned to the cluster.
//!
//! ## Initialization
//!
//! The first batch seeds centroids via k-means++ (Arthur & Vassilvitskii, 2007),
//! the same initialization used by the batch [`Kmeans`](super::Kmeans). Subsequent
//! batches update centroids incrementally.
//!
//! ## Trade-offs vs Batch K-means
//!
//! - Lower per-iteration cost: O(batch_size * k * d) vs O(n * k * d)
//! - Slightly worse final objective on small datasets
//! - Can handle streaming / out-of-core data
//! - More sensitive to batch ordering early on
//!
//! ## References
//!
//! Sculley, D. (2010). "Web-Scale K-Means Clustering." WWW 2010.

use super::distance::{DistanceMetric, SquaredEuclidean};
use super::flat::DataRef;
use super::util;
use crate::error::{Error, Result};
use rand::prelude::*;

/// Mini-Batch K-means clustering (Sculley, 2010).
///
/// Maintains k centroids and updates them incrementally as new data arrives.
/// The first batch initializes centroids via k-means++; subsequent batches
/// use a decaying learning rate per centroid.
///
/// ```
/// use clump::MiniBatchKmeans;
///
/// let mut mbk = MiniBatchKmeans::new(2).with_seed(42);
///
/// // First batch seeds centroids via k-means++
/// let batch1 = vec![
///     vec![0.0f32, 0.0], vec![0.1, 0.1],
///     vec![10.0, 10.0], vec![10.1, 10.1],
/// ];
/// let labels = mbk.update_batch(&batch1).unwrap();
/// assert_eq!(labels.len(), 4);
///
/// // Subsequent updates refine centroids
/// let label = mbk.update(&[0.05, 0.05]).unwrap();
/// assert_eq!(mbk.n_clusters(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct MiniBatchKmeans<D: DistanceMetric = SquaredEuclidean> {
    /// Number of clusters.
    k: usize,
    /// Random seed.
    seed: Option<u64>,
    /// Distance metric.
    metric: D,
    /// Current centroids (k x d). Empty before first batch.
    centroids_vec: Vec<Vec<f32>>,
    /// Per-centroid assignment count (for learning rate decay).
    counts: Vec<usize>,
    /// Whether centroids have been initialized.
    initialized: bool,
    /// RNG state.
    rng: StdRng,
}

impl MiniBatchKmeans<SquaredEuclidean> {
    /// Create a new Mini-Batch K-means clusterer with default squared Euclidean distance.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            seed: None,
            metric: SquaredEuclidean,
            centroids_vec: Vec::new(),
            counts: Vec::new(),
            initialized: false,
            rng: StdRng::from_os_rng(),
        }
    }
}

impl<D: DistanceMetric> MiniBatchKmeans<D> {
    /// Create a new Mini-Batch K-means clusterer with a custom distance metric.
    pub fn with_metric(k: usize, metric: D) -> Self {
        Self {
            k,
            seed: None,
            metric,
            centroids_vec: Vec::new(),
            counts: Vec::new(),
            initialized: false,
            rng: StdRng::from_os_rng(),
        }
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Assign a point to the nearest centroid. Requires initialized centroids.
    fn assign(&self, point: &[f32]) -> usize {
        util::assign_nearest(point, &self.centroids_vec, &self.metric)
    }

    /// Update a single centroid with a new point using the decaying learning rate.
    fn update_centroid(&mut self, cluster: usize, point: &[f32]) {
        self.counts[cluster] += 1;
        let eta = 1.0 / self.counts[cluster] as f32;
        let centroid = &mut self.centroids_vec[cluster];
        for (c, &p) in centroid.iter_mut().zip(point.iter()) {
            *c = (1.0 - eta) * *c + eta * p;
        }
        // Spherical k-means: re-normalize after update for cosine distance.
        if self.metric.normalize_centroids() {
            let norm: f32 = centroid.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                for val in centroid.iter_mut() {
                    *val /= norm;
                }
            }
        }
    }

    /// Initialize centroids from a batch using k-means++ seeding.
    fn init_centroids(&mut self, points: &(impl DataRef + ?Sized)) -> Result<()> {
        let n = points.n();
        let d = points.d();

        if d == 0 {
            return Err(Error::InvalidParameter {
                name: "dimension",
                message: "must be at least 1",
            });
        }

        if n < self.k {
            return Err(Error::InvalidClusterCount {
                requested: self.k,
                n_items: n,
            });
        }

        self.centroids_vec = util::kmeanspp_init(points, self.k, &self.metric, 2.0, &mut self.rng);
        self.counts = vec![0; self.k];
        self.initialized = true;
        Ok(())
    }

    fn validate_dimension(&self, point: &[f32]) -> Result<()> {
        if self.centroids_vec.is_empty() {
            return Ok(());
        }
        let expected = self.centroids_vec[0].len();
        if point.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                found: point.len(),
            });
        }
        Ok(())
    }
}

impl<D: DistanceMetric> MiniBatchKmeans<D> {
    /// Update the model with a single new point, returning its cluster assignment.
    pub fn update(&mut self, point: &[f32]) -> Result<usize> {
        if point.is_empty() {
            return Err(Error::InvalidParameter {
                name: "point",
                message: "must be non-empty",
            });
        }

        for &val in point {
            if !val.is_finite() {
                return Err(Error::InvalidParameter {
                    name: "data",
                    message: "contains NaN or infinity",
                });
            }
        }

        if !self.initialized {
            // Cannot initialize from a single point when k > 1.
            // Seed with a trivial batch of this one point repeated k times.
            if self.k <= 1 {
                self.centroids_vec = vec![point.to_vec()];
                self.counts = vec![1];
                self.initialized = true;
                return Ok(0);
            }
            return Err(Error::InvalidParameter {
                name: "state",
                message: "must call update_batch first to initialize centroids when k > 1",
            });
        }

        self.validate_dimension(point)?;
        let cluster = self.assign(point);
        self.update_centroid(cluster, point);
        Ok(cluster)
    }

    /// Update the model with a mini-batch of points.
    pub fn update_batch(&mut self, points: &(impl DataRef + ?Sized)) -> Result<Vec<usize>> {
        if points.n() == 0 {
            return Err(Error::EmptyInput);
        }

        let d = points.d();
        if d == 0 {
            return Err(Error::InvalidParameter {
                name: "dimension",
                message: "must be at least 1",
            });
        }

        // Validate uniform dimensionality.
        for i in 0..points.n() {
            if points.row(i).len() != d {
                return Err(Error::DimensionMismatch {
                    expected: d,
                    found: points.row(i).len(),
                });
            }
        }

        util::validate_finite(points)?;

        if !self.initialized {
            self.init_centroids(points)?;
        } else {
            self.validate_dimension(points.row(0))?;
        }

        // Assign and update.
        let mut labels = Vec::with_capacity(points.n());
        for i in 0..points.n() {
            let point = points.row(i);
            let cluster = self.assign(point);
            self.update_centroid(cluster, point);
            labels.push(cluster);
        }

        Ok(labels)
    }

    /// Predict cluster labels for new points without modifying centroids.
    ///
    /// Read-only inference: assigns each point to its nearest centroid
    /// but does not update centroid positions or counts.
    pub fn predict(&self, data: &(impl DataRef + ?Sized)) -> Result<Vec<usize>> {
        if !self.initialized {
            return Err(Error::InvalidParameter {
                name: "state",
                message: "must call update_batch first to initialize centroids",
            });
        }
        if data.n() == 0 {
            return Err(Error::EmptyInput);
        }
        self.validate_dimension(data.row(0))?;
        Ok((0..data.n()).map(|i| self.assign(data.row(i))).collect())
    }

    /// Get current cluster centroids.
    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids_vec
    }

    /// Get the per-centroid assignment count.
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }

    /// Get the current number of clusters.
    pub fn n_clusters(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod autotrait_tests {
    use super::*;

    /// Compile-time assertion that MiniBatchKmeans is Send + Sync + Sized + Unpin.
    /// Catches accidental introduction of Rc, RefCell, or raw pointers.
    /// (Pattern from linfa.)
    fn assert_autotraits<T: Send + Sync + Sized + Unpin>() {}

    #[test]
    fn minibatch_kmeans_is_send_sync() {
        assert_autotraits::<MiniBatchKmeans<SquaredEuclidean>>();
        assert_autotraits::<MiniBatchKmeans<super::super::distance::Euclidean>>();
        assert_autotraits::<MiniBatchKmeans<super::super::distance::CosineDistance>>();
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    /// Well-separated 2D data: cluster A around (0,0), cluster B around (100,100).
    fn well_separated_data() -> Vec<Vec<f32>> {
        let mut data = Vec::new();
        for i in 0..20 {
            let offset = i as f32 * 0.1;
            data.push(vec![offset, offset]);
        }
        for i in 0..20 {
            let offset = 100.0 + i as f32 * 0.1;
            data.push(vec![offset, offset]);
        }
        data
    }

    #[test]
    fn converges_to_same_structure_as_batch_kmeans() {
        use crate::cluster::kmeans::Kmeans;

        let data = well_separated_data();

        // Batch k-means reference.
        let batch_labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();

        // Mini-batch: feed everything in one batch.
        let mut mbk = MiniBatchKmeans::new(2).with_seed(42);
        let mb_labels = mbk.update_batch(&data).unwrap();

        // Same cluster structure (labels may be permuted).
        // Points 0..20 should be co-clustered, points 20..40 should be co-clustered.
        assert_eq!(batch_labels[0], batch_labels[1]);
        assert_eq!(mb_labels[0], mb_labels[1]);
        assert_ne!(batch_labels[0], batch_labels[20]);
        assert_ne!(mb_labels[0], mb_labels[20]);

        for i in 0..20 {
            assert_eq!(
                mb_labels[i], mb_labels[0],
                "point {i} should be in cluster A"
            );
        }
        for i in 20..40 {
            assert_eq!(
                mb_labels[i], mb_labels[20],
                "point {i} should be in cluster B"
            );
        }
    }

    #[test]
    fn streaming_one_at_a_time() {
        let data = well_separated_data();

        // Initialize with full first pass so centroids are reasonable.
        let mut mbk = MiniBatchKmeans::new(2).with_seed(42);
        let _ = mbk.update_batch(&data).unwrap();

        // Stream all points again one at a time (second pass).
        // With well-trained centroids, each point should land correctly.
        let mut labels = Vec::new();
        for point in &data {
            let label = mbk.update(point).unwrap();
            labels.push(label);
        }

        // Check cluster structure.
        let cluster_a = labels[0];
        let cluster_b = labels[20];
        assert_ne!(cluster_a, cluster_b);

        for i in 0..20 {
            assert_eq!(labels[i], cluster_a, "point {i} misassigned");
        }
        for i in 20..40 {
            assert_eq!(labels[i], cluster_b, "point {i} misassigned");
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let data = well_separated_data();

        let mut mbk1 = MiniBatchKmeans::new(2).with_seed(99);
        let labels1 = mbk1.update_batch(&data).unwrap();

        let mut mbk2 = MiniBatchKmeans::new(2).with_seed(99);
        let labels2 = mbk2.update_batch(&data).unwrap();

        assert_eq!(labels1, labels2, "same seed should give identical labels");

        // Centroids should also match exactly.
        for (c1, c2) in mbk1.centroids().iter().zip(mbk2.centroids()) {
            assert_eq!(c1, c2, "centroids should match with same seed");
        }
    }

    #[test]
    fn centroid_count_matches_k() {
        let mut mbk = MiniBatchKmeans::new(3).with_seed(7);
        let data: Vec<Vec<f32>> = (0..30)
            .map(|i| vec![(i % 3) as f32 * 50.0, (i / 3) as f32 * 0.1])
            .collect();
        let _ = mbk.update_batch(&data).unwrap();

        assert_eq!(mbk.n_clusters(), 3);
        assert_eq!(mbk.centroids().len(), 3);
    }

    #[test]
    fn empty_input_error() {
        let mut mbk = MiniBatchKmeans::new(2).with_seed(1);
        let result = mbk.update_batch(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn dimension_mismatch_error() {
        let mut mbk = MiniBatchKmeans::new(2).with_seed(1);
        let init = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let _ = mbk.update_batch(&init).unwrap();

        // 3D point into a 2D model.
        let result = mbk.update(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn update_before_init_k1() {
        // k=1: single-point initialization is valid.
        let mut mbk = MiniBatchKmeans::new(1).with_seed(1);
        let label = mbk.update(&[5.0, 5.0]).unwrap();
        assert_eq!(label, 0);
        assert_eq!(mbk.centroids().len(), 1);
    }

    #[test]
    fn update_before_init_k_gt_1_errors() {
        let mut mbk = MiniBatchKmeans::new(3).with_seed(1);
        let result = mbk.update(&[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn incremental_batches_refine_centroids() {
        let mut mbk = MiniBatchKmeans::new(2).with_seed(42);

        // Batch 1: seed.
        let batch1 = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![100.0, 100.0],
            vec![100.1, 100.1],
        ];
        let _ = mbk.update_batch(&batch1).unwrap();

        let centroids_after_1 = mbk.centroids().to_vec();

        // Batch 2: more points near cluster A.
        let batch2: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32 * 0.01, 0.0]).collect();
        let _ = mbk.update_batch(&batch2).unwrap();

        // Cluster A's centroid should have moved toward the new points.
        // Find which centroid is cluster A (closer to origin).
        let ca = if mbk.centroids()[0][0] < mbk.centroids()[1][0] {
            0
        } else {
            1
        };
        let old_ca = if centroids_after_1[0][0] < centroids_after_1[1][0] {
            0
        } else {
            1
        };

        // The centroid shouldn't be identical (it should have shifted).
        assert_ne!(
            mbk.centroids()[ca],
            centroids_after_1[old_ca],
            "centroid should shift after new batch"
        );
    }

    #[test]
    fn nan_input_rejected_batch() {
        let mut mbk = MiniBatchKmeans::new(2).with_seed(42);
        let data = vec![vec![0.0, f32::NAN], vec![1.0, 1.0]];
        let result = mbk.update_batch(&data);
        assert!(result.is_err());
    }

    #[test]
    fn nan_input_rejected_single() {
        let mut mbk = MiniBatchKmeans::new(1).with_seed(42);
        let _ = mbk.update(&[1.0, 1.0]).unwrap();
        let result = mbk.update(&[1.0, f32::NAN]);
        assert!(result.is_err());
    }

    #[test]
    fn inf_input_rejected() {
        let mut mbk = MiniBatchKmeans::new(2).with_seed(42);
        let data = vec![vec![0.0, 0.0], vec![f32::INFINITY, 1.0]];
        let result = mbk.update_batch(&data);
        assert!(result.is_err());
    }

    #[test]
    fn with_custom_metric() {
        use crate::cluster::distance::Euclidean;

        let mut mbk = MiniBatchKmeans::with_metric(2, Euclidean).with_seed(42);
        let data = well_separated_data();
        let labels = mbk.update_batch(&data).unwrap();

        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[20]);
    }

    /// Self-identity oracle (pattern from linfa): feed centroids back as input.
    /// Each centroid should be assigned to itself -- catches distance/assignment bugs.
    #[test]
    fn self_identity_oracle() {
        let data = well_separated_data();
        let mut mbk = MiniBatchKmeans::new(2).with_seed(42);
        let _ = mbk.update_batch(&data).unwrap();

        let centroids: Vec<Vec<f32>> = mbk.centroids().to_vec();
        for (k, centroid) in centroids.iter().enumerate() {
            let label = mbk.assign(centroid);
            assert_eq!(
                label, k,
                "centroid {k} should be assigned to cluster {k}, got {label}"
            );
        }
    }

    /// Counts should track how many points were assigned to each centroid.
    #[test]
    fn counts_track_assignments() {
        let data = well_separated_data(); // 20 near origin + 20 near (100,100)
        let mut mbk = MiniBatchKmeans::new(2).with_seed(42);
        let _ = mbk.update_batch(&data).unwrap();

        let counts = mbk.counts();
        assert_eq!(counts.len(), 2);
        // Each cluster should have ~20 points.
        let total: usize = counts.iter().sum();
        assert_eq!(total, 40, "total counts should equal number of points");
        for &c in counts {
            assert!(c > 0, "no empty clusters expected with well-separated data");
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_data(n: usize, d: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
        proptest::collection::vec(proptest::collection::vec(-100.0f32..100.0, d), n)
    }

    proptest! {
        #[test]
        fn labels_in_range(data in arb_data(20, 3)) {
            let mut mbk = MiniBatchKmeans::new(3).with_seed(1);
            let labels = mbk.update_batch(&data).unwrap();
            for &l in &labels {
                prop_assert!(l < 3, "label {l} >= k=3");
            }
        }

        #[test]
        fn centroid_dimension_matches_input(data in arb_data(10, 5)) {
            let mut mbk = MiniBatchKmeans::new(2).with_seed(1);
            let _ = mbk.update_batch(&data).unwrap();
            for c in mbk.centroids() {
                prop_assert_eq!(c.len(), 5, "centroid dim should match input dim");
            }
        }

        #[test]
        fn streaming_labels_in_range(data in arb_data(20, 4)) {
            let mut mbk = MiniBatchKmeans::new(2).with_seed(1);
            let _ = mbk.update_batch(&data).unwrap();
            // Stream additional points.
            for point in &data {
                let label = mbk.update(point).unwrap();
                prop_assert!(label < 2, "streaming label {label} >= k=2");
            }
        }
    }
}
