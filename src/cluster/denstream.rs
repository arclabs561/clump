//! DenStream: Density-Based Clustering over an Evolving Data Stream with Noise.
//!
//! # The Algorithm (Cao et al., 2006)
//!
//! DenStream is a streaming density-based clustering algorithm with two phases:
//!
//! - **Online phase**: maintains micro-clusters that summarize the stream. Each
//!   incoming point is absorbed into the nearest micro-cluster, or spawns a new
//!   one. Micro-clusters are split into *potential* (high weight) and *outlier*
//!   (low weight). Outlier micro-clusters that accumulate enough weight are
//!   promoted to potential. Periodic pruning removes stale micro-clusters.
//!
//! - **Offline phase** (on demand): runs DBSCAN on the centroids of potential
//!   micro-clusters to produce macro-clusters. Each data point inherits the
//!   label of its nearest potential micro-cluster.
//!
//! ## Micro-Cluster Summary
//!
//! Each micro-cluster is a CF-like (Clustering Feature) structure that tracks:
//! - Linear sum and squared sum of absorbed points (for centroid and radius)
//! - Decayed weight (recency-aware)
//! - Creation and last-update timestamps
//!
//! The centroid is `ls / n` and the radius is `sqrt(ss/n - (ls/n)^2)`.
//!
//! ## Parameters
//!
//! - `epsilon`: radius threshold for micro-cluster absorption
//! - `macro_epsilon`: epsilon for the offline DBSCAN pass
//! - `min_pts`: minimum points for DBSCAN core classification
//! - `beta`: weight factor -- a micro-cluster needs `weight >= beta * mu` to be potential
//! - `lambda`: decay factor (higher = faster forgetting)
//! - `mu`: base weight for new points
//! - `t_p`: pruning period (prune every `t_p` updates)
//!
//! ## Trade-offs
//!
//! - vs DBSCAN: handles streaming data without storing all points; adapts to
//!   concept drift via decay. But the micro-cluster approximation loses detail.
//! - vs MiniBatchKmeans: discovers clusters of arbitrary shape and number;
//!   identifies noise. But more parameters to tune and higher per-point cost.
//!
//! ## References
//!
//! Cao, F., Ester, M., Qian, W., & Zhou, A. (2006). "Density-Based Clustering
//! over an Evolving Data Stream with Noise." SDM 2006.

use super::dbscan::{Dbscan, NOISE};
use super::distance::{DistanceMetric, SquaredEuclidean};
use crate::error::{Error, Result};

/// A micro-cluster summary (CF-like structure).
///
/// Tracks the linear sum, squared sum, weight, and timestamps of absorbed
/// points. The centroid and radius are derived quantities.
#[derive(Debug, Clone)]
struct MicroCluster {
    /// Number of absorbed points.
    n: usize,
    /// Linear sum of points (centroid = ls / n).
    ls: Vec<f32>,
    /// Squared sum of points (for radius computation).
    ss: Vec<f32>,
    /// Decayed weight (accounts for recency).
    weight: f64,
    /// Timestamp when the micro-cluster was created.
    creation_time: u64,
    /// Timestamp of the most recent absorbed point.
    last_update: u64,
}

impl MicroCluster {
    /// Create a new micro-cluster from a single point.
    fn new(point: &[f32], timestamp: u64) -> Self {
        let ls = point.to_vec();
        let ss: Vec<f32> = point.iter().map(|&x| x * x).collect();
        Self {
            n: 1,
            ls,
            ss,
            weight: 1.0,
            creation_time: timestamp,
            last_update: timestamp,
        }
    }

    /// Compute the centroid (decay-weighted mean of absorbed points).
    fn centroid(&self) -> Vec<f32> {
        let w = self.weight as f32;
        if w <= 0.0 {
            return self.ls.clone(); // degenerate: return raw sums
        }
        self.ls.iter().map(|&x| x / w).collect()
    }

    /// Compute the radius: `sqrt(ss/w - (ls/w)^2)`, clamped to 0.
    #[allow(dead_code)]
    fn radius(&self) -> f32 {
        self.radius_from(self.weight as f32, &self.ls, &self.ss)
    }

    /// Compute the radius that would result from absorbing an additional point,
    /// without actually modifying the micro-cluster.
    fn radius_if_absorbed(&self, point: &[f32]) -> f32 {
        let new_w = self.weight as f32 + 1.0;
        let new_ls: Vec<f32> = self.ls.iter().zip(point).map(|(&l, &p)| l + p).collect();
        let new_ss: Vec<f32> = self
            .ss
            .iter()
            .zip(point)
            .map(|(&s, &p)| s + p * p)
            .collect();
        self.radius_from(new_w, &new_ls, &new_ss)
    }

    /// Shared radius computation from arbitrary CF sums.
    fn radius_from(&self, w: f32, ls: &[f32], ss: &[f32]) -> f32 {
        if w <= 0.0 {
            return 0.0;
        }
        let mut sum = 0.0f32;
        for (&l, &s) in ls.iter().zip(ss) {
            let mean = l / w;
            let var = s / w - mean * mean;
            sum += var;
        }
        sum.max(0.0).sqrt()
    }

    /// Apply time decay to all CF components.
    fn apply_decay(&mut self, decay_factor: f64, timestamp: u64) {
        let elapsed = timestamp.saturating_sub(self.last_update);
        if elapsed > 0 {
            let decay = (-decay_factor * elapsed as f64).exp() as f32;
            self.weight *= decay as f64;
            for l in &mut self.ls {
                *l *= decay;
            }
            for s in &mut self.ss {
                *s *= decay;
            }
            self.last_update = timestamp;
        }
    }

    /// Absorb a point into this micro-cluster, applying time decay.
    fn absorb(&mut self, point: &[f32], decay_factor: f64, timestamp: u64) {
        // Decay existing sums before adding the new point.
        self.apply_decay(decay_factor, timestamp);

        // Add the point.
        self.n += 1;
        self.weight += 1.0;
        for (&p, (l, s)) in point.iter().zip(self.ls.iter_mut().zip(self.ss.iter_mut())) {
            *l += p;
            *s += p * p;
        }
        self.last_update = timestamp;
    }

    /// Apply time decay without absorbing a point.
    fn decay(&mut self, decay_factor: f64, timestamp: u64) {
        self.apply_decay(decay_factor, timestamp);
    }
}

/// DenStream: streaming density-based clustering.
///
/// Maintains online micro-clusters that summarize the stream, then runs DBSCAN
/// on their centroids to produce macro-clusters on demand.
///
/// ```
/// use clump::DenStream;
///
/// let mut ds = DenStream::new(1.0, 3)
///     .with_beta(0.5)
///     .with_lambda(0.01)
///     .with_mu(1.0);
///
/// // Feed points from two clusters.
/// for i in 0..20 {
///     let offset = i as f32 * 0.1;
///     ds.update(&[offset, offset]).unwrap();
/// }
/// for i in 0..20 {
///     let offset = 50.0 + i as f32 * 0.1;
///     ds.update(&[offset, offset]).unwrap();
/// }
///
/// assert!(ds.n_clusters() >= 2);
/// ```
#[derive(Debug, Clone)]
pub struct DenStream<D: DistanceMetric = SquaredEuclidean> {
    /// Micro-cluster radius threshold.
    epsilon: f32,
    /// DBSCAN epsilon for macro-clustering.
    macro_epsilon: f32,
    /// Minimum points for DBSCAN core in macro-clustering.
    min_pts: usize,
    /// Weight threshold factor. A micro-cluster needs weight >= beta * mu to be "potential".
    beta: f64,
    /// Decay factor lambda. Higher = faster forgetting of old data.
    lambda: f64,
    /// Base weight for new points.
    mu: f64,
    /// Pruning period (prune every t_p updates).
    t_p: usize,
    /// Distance metric.
    metric: D,
    /// Potential micro-clusters (high weight, form the basis of macro-clusters).
    p_micro_clusters: Vec<MicroCluster>,
    /// Outlier micro-clusters (low weight, may be promoted or pruned).
    o_micro_clusters: Vec<MicroCluster>,
    /// Current logical timestamp (incremented on each update).
    timestamp: u64,
    /// Counter for triggering periodic pruning.
    updates_since_prune: usize,
    /// Dimensionality of the first point seen (for validation).
    dim: Option<usize>,
}

impl DenStream<SquaredEuclidean> {
    /// Create a new DenStream with the default squared Euclidean distance.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Maximum radius for micro-cluster absorption.
    /// * `min_pts` - Minimum points for DBSCAN core in macro-clustering.
    pub fn new(epsilon: f32, min_pts: usize) -> Self {
        Self::with_metric(epsilon, min_pts, SquaredEuclidean)
    }
}

impl<D: DistanceMetric> DenStream<D> {
    /// Create a new DenStream with a custom distance metric.
    pub fn with_metric(epsilon: f32, min_pts: usize, metric: D) -> Self {
        Self {
            epsilon,
            macro_epsilon: epsilon * 2.0,
            min_pts,
            beta: 0.5,
            lambda: 0.001,
            mu: 1.0,
            t_p: 100,
            metric,
            p_micro_clusters: Vec::new(),
            o_micro_clusters: Vec::new(),
            timestamp: 0,
            updates_since_prune: 0,
            dim: None,
        }
    }

    /// Set the weight threshold factor beta.
    ///
    /// A micro-cluster needs `weight >= beta * mu` to be classified as potential.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Set the decay factor lambda.
    ///
    /// Higher values cause faster forgetting of old data.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set the base weight mu.
    pub fn with_mu(mut self, mu: f64) -> Self {
        self.mu = mu;
        self
    }

    /// Set the DBSCAN epsilon for macro-clustering.
    ///
    /// Defaults to `2 * epsilon` if not set.
    pub fn with_macro_epsilon(mut self, eps: f32) -> Self {
        self.macro_epsilon = eps;
        self
    }

    /// Set the pruning period.
    ///
    /// Stale micro-clusters are pruned every `t_p` updates.
    pub fn with_pruning_period(mut self, t_p: usize) -> Self {
        self.t_p = t_p;
        self
    }

    /// Run DBSCAN on potential micro-cluster centroids to produce macro-clusters.
    ///
    /// Returns one label per potential micro-cluster. Labels are cluster indices
    /// or `NOISE` (`usize::MAX`) for micro-clusters that DBSCAN considers noise.
    pub fn macro_cluster(&self) -> Result<Vec<usize>> {
        if self.p_micro_clusters.is_empty() {
            return Err(Error::EmptyInput);
        }

        let centroids: Vec<Vec<f32>> = self
            .p_micro_clusters
            .iter()
            .map(|mc| mc.centroid())
            .collect();
        let dbscan = Dbscan::with_metric(self.macro_epsilon, self.min_pts, self.metric.clone());
        dbscan.fit_predict(&centroids)
    }

    /// Validate a point's dimensionality against previously seen points.
    fn validate_point(&self, point: &[f32]) -> Result<()> {
        if point.is_empty() {
            return Err(Error::InvalidParameter {
                name: "point",
                message: "must be non-empty",
            });
        }
        if let Some(expected) = self.dim {
            if point.len() != expected {
                return Err(Error::DimensionMismatch {
                    expected,
                    found: point.len(),
                });
            }
        }
        Ok(())
    }

    /// Find the nearest micro-cluster to a point, returning (index, distance).
    fn nearest_micro_cluster(
        &self,
        point: &[f32],
        clusters: &[MicroCluster],
    ) -> Option<(usize, f32)> {
        let mut best_idx = None;
        let mut best_dist = f32::MAX;

        // Reuse a scratch buffer for centroid computation to avoid
        // allocating a Vec per micro-cluster per call.
        let d = point.len();
        let mut centroid_buf = vec![0.0f32; d];

        for (i, mc) in clusters.iter().enumerate() {
            let w = mc.weight as f32;
            if w > 0.0 {
                for (j, &x) in mc.ls.iter().enumerate() {
                    centroid_buf[j] = x / w;
                }
            } else {
                centroid_buf[..d].copy_from_slice(&mc.ls[..d]);
            }
            let dist = self.metric.distance(point, &centroid_buf);
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(i);
            }
        }

        best_idx.map(|idx| (idx, best_dist))
    }

    /// Prune stale micro-clusters.
    ///
    /// - Remove potential micro-clusters whose decayed weight < beta * mu.
    /// - Remove outlier micro-clusters whose weight is below the time-based threshold.
    fn prune(&mut self) {
        let threshold = self.beta * self.mu;
        let ts = self.timestamp;
        let lambda = self.lambda;

        // Decay all micro-clusters to current time, then prune.
        for mc in &mut self.p_micro_clusters {
            mc.decay(lambda, ts);
        }
        self.p_micro_clusters.retain(|mc| mc.weight >= threshold);

        for mc in &mut self.o_micro_clusters {
            mc.decay(lambda, ts);
        }
        // Outlier threshold: based on creation recency.
        // From the paper: xi(t_c, t) = (2^(-lambda*(t - t_c + t_p)) - 1) / (2^(-lambda * t_p) - 1)
        // Outliers that haven't accumulated enough weight relative to their age are removed.
        let current_ts = self.timestamp;
        let t_p = self.t_p as u64;
        let lam = self.lambda;
        self.o_micro_clusters.retain(|mc| {
            let age = current_ts.saturating_sub(mc.creation_time);
            mc.weight >= outlier_weight_threshold(lam, t_p, age, threshold)
        });
    }
}

/// Compute the weight threshold for an outlier micro-cluster given its age.
///
/// From the paper: xi(t_c, t) = (2^{-lambda*(t - t_c + t_p)} - 1) / (2^{-lambda*t_p} - 1).
/// An outlier whose weight falls below `xi * beta * mu` is pruned.
fn outlier_weight_threshold(lambda: f64, t_p: u64, age: u64, potential_threshold: f64) -> f64 {
    let denom = 2.0_f64.powf(-lambda * t_p as f64) - 1.0;
    if denom.abs() < f64::EPSILON {
        // lambda * t_p is near zero: xi approaches 1. Use potential threshold.
        return potential_threshold;
    }
    let numer = 2.0_f64.powf(-lambda * (age + t_p) as f64) - 1.0;
    let xi = numer / denom;
    xi * potential_threshold
}

impl<D: DistanceMetric> DenStream<D> {
    /// Absorb a single point, returning the index of the nearest potential
    /// micro-cluster, or `NOISE` if the point was placed in an outlier cluster.
    pub fn update(&mut self, point: &[f32]) -> Result<usize> {
        self.validate_point(point)?;

        // Validate finite values.
        for &val in point {
            if !val.is_finite() {
                return Err(Error::InvalidParameter {
                    name: "data",
                    message: "contains NaN or infinity",
                });
            }
        }

        // Set dimensionality on first point.
        if self.dim.is_none() {
            self.dim = Some(point.len());
        }

        self.timestamp += 1;
        let ts = self.timestamp;

        let potential_threshold = self.beta * self.mu;
        let mut assigned_p_idx = None;

        // Step 1: Try to absorb into nearest potential micro-cluster.
        if let Some((idx, dist)) = self.nearest_micro_cluster(point, &self.p_micro_clusters) {
            if dist <= self.epsilon {
                // Check if absorbing would keep radius within epsilon.
                let new_radius = self.p_micro_clusters[idx].radius_if_absorbed(point);
                if new_radius <= self.epsilon {
                    self.p_micro_clusters[idx].absorb(point, self.lambda, ts);

                    assigned_p_idx = Some(idx);
                }
            }
        }

        // Step 2: If not absorbed into a p-cluster, try outlier micro-clusters.
        if assigned_p_idx.is_none() {
            let mut absorbed_into_outlier = false;
            if let Some((idx, dist)) = self.nearest_micro_cluster(point, &self.o_micro_clusters) {
                if dist <= self.epsilon {
                    let new_radius = self.o_micro_clusters[idx].radius_if_absorbed(point);
                    if new_radius <= self.epsilon {
                        self.o_micro_clusters[idx].absorb(point, self.lambda, ts);
                        absorbed_into_outlier = true;

                        // Check if this outlier should be promoted to potential.
                        if self.o_micro_clusters[idx].weight >= potential_threshold {
                            let promoted = self.o_micro_clusters.remove(idx);
                            self.p_micro_clusters.push(promoted);

                            // The promoted cluster is now the last p-cluster.
                            assigned_p_idx = Some(self.p_micro_clusters.len() - 1);
                        }
                    }
                }
            }

            // Step 3: If not absorbed anywhere, create new outlier micro-cluster.
            if !absorbed_into_outlier && assigned_p_idx.is_none() {
                let mc = MicroCluster::new(point, ts);
                // If a single-point cluster already meets the potential threshold, add as potential.
                if mc.weight >= potential_threshold {
                    self.p_micro_clusters.push(mc);

                    assigned_p_idx = Some(self.p_micro_clusters.len() - 1);
                } else {
                    self.o_micro_clusters.push(mc);
                }
            }
        }

        // Periodic pruning.
        self.updates_since_prune += 1;
        if self.updates_since_prune >= self.t_p {
            self.prune();
            self.updates_since_prune = 0;
        }

        // Return the p-micro-cluster index, or NOISE if only in outlier.
        Ok(assigned_p_idx.unwrap_or(NOISE))
    }

    /// Update the model with a mini-batch of points.
    pub fn update_batch(&mut self, points: &[Vec<f32>]) -> Result<Vec<usize>> {
        if points.is_empty() {
            return Err(Error::EmptyInput);
        }

        let mut labels = Vec::with_capacity(points.len());
        for point in points {
            labels.push(self.update(point)?);
        }
        Ok(labels)
    }

    /// Get current cluster centroids (one per potential micro-cluster).
    pub fn centroids(&self) -> Vec<Vec<f32>> {
        self.p_micro_clusters
            .iter()
            .map(|mc| mc.centroid())
            .collect()
    }

    /// Get the per-centroid point count.
    pub fn counts(&self) -> Vec<usize> {
        self.p_micro_clusters.iter().map(|mc| mc.n).collect()
    }

    /// Get the current number of potential micro-clusters.
    pub fn n_clusters(&self) -> usize {
        self.p_micro_clusters.len()
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::cluster::dbscan::NOISE;

    /// Helper: create a DenStream configured for testing with tight clusters.
    fn test_denstream() -> DenStream<SquaredEuclidean> {
        DenStream::new(2.0, 2)
            .with_beta(0.5)
            .with_lambda(0.001)
            .with_mu(1.0)
            .with_macro_epsilon(4.0)
            .with_pruning_period(1000)
    }

    #[test]
    fn absorbs_nearby_points() {
        let mut ds = test_denstream();

        // First point creates a micro-cluster.
        ds.update(&[0.0, 0.0]).ok();

        // Second point within epsilon should join the same micro-cluster.
        ds.update(&[0.1, 0.1]).ok();

        // With beta=0.5, mu=1.0, threshold=0.5. A single point has weight 1.0,
        // so even a single-point cluster is potential when mu >= threshold.
        // After two absorptions, we should still have a compact set.
        assert!(
            ds.p_micro_clusters.len() + ds.o_micro_clusters.len() <= 2,
            "nearby points should merge"
        );

        // The total point count across all clusters should be 2.
        let total: usize = ds
            .p_micro_clusters
            .iter()
            .chain(ds.o_micro_clusters.iter())
            .map(|mc| mc.n)
            .sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn creates_new_micro_cluster_for_distant_points() {
        let mut ds = test_denstream();

        ds.update(&[0.0, 0.0]).ok();
        ds.update(&[100.0, 100.0]).ok();

        let total_clusters = ds.p_micro_clusters.len() + ds.o_micro_clusters.len();
        assert_eq!(
            total_clusters, 2,
            "distant points should create separate micro-clusters"
        );
    }

    #[test]
    fn pruning_removes_stale_clusters() {
        // Use aggressive decay and short pruning period.
        let mut ds = DenStream::new(2.0, 2)
            .with_beta(0.5)
            .with_lambda(1.0) // very aggressive decay
            .with_mu(1.0)
            .with_pruning_period(5);

        // Create a cluster far from subsequent activity.
        ds.update(&[100.0, 100.0]).ok();

        // Feed many points elsewhere to advance time and trigger pruning.
        for i in 0..20 {
            ds.update(&[0.0 + i as f32 * 0.01, 0.0]).ok();
        }

        // The distant cluster should have been pruned due to weight decay.
        let has_distant = ds
            .p_micro_clusters
            .iter()
            .chain(ds.o_micro_clusters.iter())
            .any(|mc| {
                let c = mc.centroid();
                c[0] > 50.0
            });
        assert!(
            !has_distant,
            "stale distant cluster should have been pruned"
        );
    }

    #[test]
    fn macro_clustering_finds_groups() {
        let mut ds = DenStream::new(1.0, 2)
            .with_beta(0.2)
            .with_lambda(0.0001)
            .with_mu(1.0)
            .with_macro_epsilon(3.0)
            .with_pruning_period(10_000);

        // Feed two well-separated clusters.
        for i in 0..30 {
            let offset = i as f32 * 0.05;
            ds.update(&[offset, offset]).ok();
        }
        for i in 0..30 {
            let offset = 50.0 + i as f32 * 0.05;
            ds.update(&[offset, offset]).ok();
        }

        let macro_labels = ds.macro_cluster();
        assert!(macro_labels.is_ok(), "macro_cluster should succeed");

        let labels = macro_labels.expect("checked above");

        // Collect distinct non-noise labels.
        let distinct: std::collections::HashSet<usize> =
            labels.iter().copied().filter(|&l| l != NOISE).collect();
        assert!(
            distinct.len() >= 2,
            "should find at least 2 macro-clusters, found {}",
            distinct.len()
        );
    }

    #[test]
    fn with_custom_metric() {
        use crate::cluster::distance::Euclidean;

        let mut ds = DenStream::with_metric(2.0, 2, Euclidean)
            .with_beta(0.5)
            .with_lambda(0.001)
            .with_mu(1.0);

        ds.update(&[0.0, 0.0]).ok();
        ds.update(&[0.5, 0.5]).ok();
        ds.update(&[100.0, 100.0]).ok();

        let total = ds.p_micro_clusters.len() + ds.o_micro_clusters.len();
        assert!(
            total >= 2,
            "should have at least 2 micro-clusters with Euclidean"
        );
    }

    #[test]
    fn empty_update_error() {
        let mut ds = test_denstream();
        let result = ds.update(&[]);
        assert!(result.is_err(), "empty point should error");
    }

    #[test]
    fn dimension_mismatch_error() {
        let mut ds = test_denstream();
        ds.update(&[1.0, 2.0]).ok();

        let result = ds.update(&[1.0, 2.0, 3.0]);
        assert!(result.is_err(), "dimension mismatch should error");
    }

    #[test]
    fn streaming_trait_consistency() {
        let mut ds = test_denstream();

        for i in 0..10 {
            ds.update(&[i as f32, i as f32]).ok();
        }

        assert_eq!(
            ds.n_clusters(),
            ds.centroids().len(),
            "n_clusters should match centroids().len()"
        );
    }

    #[test]
    fn update_batch_processes_all_points() {
        let mut ds = test_denstream();

        let points: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 10.0, 0.0]).collect();
        let labels = ds.update_batch(&points);
        assert!(labels.is_ok());
        assert_eq!(labels.expect("checked above").len(), 10);
    }

    #[test]
    fn update_batch_empty_errors() {
        let mut ds = test_denstream();
        let result = ds.update_batch(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn micro_cluster_radius_single_point_is_zero() {
        let mc = MicroCluster::new(&[1.0, 2.0, 3.0], 0);
        assert!(mc.radius().abs() < 1e-6, "single-point radius should be 0");
    }

    #[test]
    fn micro_cluster_centroid_matches_single_point() {
        let mc = MicroCluster::new(&[3.0, 4.0], 0);
        let c = mc.centroid();
        assert!((c[0] - 3.0).abs() < 1e-6);
        assert!((c[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn macro_cluster_on_empty_errors() {
        let ds = test_denstream();
        let result = ds.macro_cluster();
        assert!(result.is_err(), "macro_cluster on empty should error");
    }

    #[test]
    fn noise_sentinel_value() {
        // Verify our NOISE constant matches what callers expect.
        assert_eq!(NOISE, usize::MAX);
    }

    #[test]
    fn nan_input_rejected() {
        let mut ds = test_denstream();
        let result = ds.update(&[1.0, f32::NAN]);
        assert!(result.is_err());
    }

    #[test]
    fn inf_input_rejected() {
        let mut ds = test_denstream();
        let result = ds.update(&[f32::INFINITY, 0.0]);
        assert!(result.is_err());
    }

    /// DenStream centroid drift under decay: after feeding many points at a new
    /// location with time gaps, centroids should track recent data.
    #[test]
    fn centroid_drift_under_decay() {
        let mut ds = DenStream::new(2.0, 2)
            .with_beta(0.2)
            .with_lambda(0.1) // moderate decay
            .with_mu(1.0)
            .with_pruning_period(10_000);

        // Phase 1: 50 points at origin.
        for _ in 0..50 {
            ds.update(&[0.0, 0.0]).ok();
        }

        // Phase 2: 200 points at (10, 10) to overwhelm decayed origin cluster.
        for _ in 0..200 {
            ds.update(&[10.0, 10.0]).ok();
        }

        // After heavy decay, the centroid closest to (10, 10) should dominate.
        let centroids = ds.centroids();
        assert!(!centroids.is_empty());
        let has_near_10 = centroids.iter().any(|c| c[0] > 5.0 && c[1] > 5.0);
        assert!(has_near_10, "centroid should track recent (10,10) points");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::cluster::dbscan::NOISE;
    use proptest::prelude::*;

    fn arb_point(d: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0f32..100.0, d)
    }

    fn arb_points(n: usize, d: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
        proptest::collection::vec(arb_point(d), n)
    }

    proptest! {
        #[test]
        fn labels_in_valid_range(points in arb_points(30, 3)) {
            let mut ds = DenStream::new(5.0, 2)
                .with_beta(0.5)
                .with_lambda(0.001)
                .with_mu(1.0)
                .with_pruning_period(1000);

            for point in &points {
                let label = ds.update(point).expect("update should succeed");
                // Label is either a valid p-micro-cluster index or NOISE.
                prop_assert!(
                    label == NOISE || label < ds.p_micro_clusters.len(),
                    "label {} out of range (n_p={})",
                    label,
                    ds.p_micro_clusters.len()
                );
            }
        }

        #[test]
        fn centroid_dimension_matches_input(points in arb_points(10, 5)) {
            let mut ds = DenStream::new(5.0, 2)
                .with_beta(0.5)
                .with_lambda(0.001)
                .with_mu(1.0);

            for point in &points {
                ds.update(point).expect("update should succeed");
            }


            for c in ds.centroids() {
                prop_assert_eq!(c.len(), 5, "centroid dim should match input dim");
            }
        }
    }

    /// Centroid should drift toward recent points under decay.
    #[test]
    fn centroid_drift_under_decay() {
        let mut ds = DenStream::new(2.0, 2)
            .with_beta(0.5)
            .with_lambda(0.1)
            .with_mu(1.0);

        // Feed points near origin.
        for _ in 0..20 {
            ds.update(&[0.0, 0.0]).unwrap();
        }
        // Feed points near [10, 10] with time gaps.
        for _ in 0..20 {
            ds.update(&[10.0, 10.0]).unwrap();
        }

        let centroids = ds.centroids();
        // At least one centroid should be near [10, 10].
        let near_target = centroids.iter().any(|c| {
            let dist_sq = c
                .iter()
                .zip([10.0, 10.0].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>();
            dist_sq < 25.0
        });
        assert!(near_target, "centroid should drift toward recent points");
    }
}
