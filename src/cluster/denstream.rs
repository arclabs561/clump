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
//! - **Initialization**: buffers the first 1,000 points without decay, then
//!   applies DBSCAN with `epsilon` and `beta * mu` as the density threshold.
//! - **Offline phase** (on demand): connects dense potential micro-clusters
//!   whose centers and radii overlap. Border micro-clusters join a neighboring
//!   core but do not expand it.
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
//! - `macro_epsilon`: maximum center distance in the weighted offline phase
//! - `min_pts`: minimum points used only by the legacy unweighted helper
//! - `beta`: weight factor -- an outlier is promoted when `weight > beta * mu`
//! - `lambda`: decay factor (higher = faster forgetting)
//! - `mu`: base weight for new points
//! - `t_p`: pruning period, derived from `beta`, `mu`, and `lambda` unless
//!   explicitly overridden
//! - initialization buffer size: 1,000 points by default; zero starts online
//!   processing immediately
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
use super::distance::{DistanceMetric, Euclidean, SquaredEuclidean};
use super::flat::DataRef;
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
    fn from_points(points: &[Vec<f32>], timestamp: u64) -> Self {
        let mut cluster = Self::new(&points[0], timestamp);
        for point in &points[1..] {
            cluster.n += 1;
            cluster.weight += 1.0;
            for (&value, (ls, ss)) in point
                .iter()
                .zip(cluster.ls.iter_mut().zip(cluster.ss.iter_mut()))
            {
                *ls += value;
                *ss += value * value;
            }
        }
        cluster
    }
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
    fn radius_if_absorbed(&self, point: &[f32], decay_factor: f64, timestamp: u64) -> f32 {
        let mut decayed = self.clone();
        decayed.apply_decay(decay_factor, timestamp);
        let new_w = decayed.weight as f32 + 1.0;
        let new_ls: Vec<f32> = decayed.ls.iter().zip(point).map(|(&l, &p)| l + p).collect();
        let new_ss: Vec<f32> = decayed
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
            // Base-2 decay to match the outlier threshold formula (Cao et al. 2006).
            let decay = 2.0_f64.powf(-decay_factor * elapsed as f64) as f32;
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
/// Maintains online micro-clusters that summarize the stream, then applies the
/// weighted DenStream offline phase on demand.
///
/// ```
/// use clump::DenStream;
///
/// let mut ds = DenStream::new(1.0, 3)
///     .with_initial_buffer_size(0)
///     .with_lambda(0.01)
///     .with_mu(2.0);
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
pub struct DenStream<D: DistanceMetric = Euclidean> {
    /// Micro-cluster radius threshold.
    epsilon: f32,
    /// DBSCAN epsilon for macro-clustering.
    macro_epsilon: f32,
    /// Minimum points for the legacy unweighted macro-clustering helper.
    min_pts: usize,
    /// Weight threshold factor. An outlier is promoted when its weight exceeds `beta * mu`.
    beta: f64,
    /// Decay factor lambda. Higher = faster forgetting of old data.
    lambda: f64,
    /// Base weight for new points.
    mu: f64,
    /// Explicit pruning-period override. `None` uses the paper's derived period.
    pruning_period_override: Option<usize>,
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
    /// Number of raw points used for the paper's initialization phase.
    initial_buffer_size: usize,
    /// Undecayed points waiting for initialization.
    initial_buffer: Vec<Vec<f32>>,
    /// Whether the initialization phase has completed.
    initialized: bool,
}

impl DenStream<Euclidean> {
    /// Create a new DenStream with the default Euclidean distance.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Maximum radius for micro-cluster absorption.
    /// * `min_pts` - Minimum points for [`Self::macro_cluster_unweighted`].
    pub fn new(epsilon: f32, min_pts: usize) -> Self {
        Self::with_metric(epsilon, min_pts, Euclidean)
    }
}

impl DenStream<SquaredEuclidean> {
    /// Create a DenStream with the pre-0.6 squared-Euclidean online metric.
    ///
    /// This is retained for source migrations. Its online `predict` threshold
    /// is expressed in squared-distance units; new code should use [`DenStream::new`].
    pub fn new_squared_euclidean_legacy(epsilon: f32, min_pts: usize) -> Self {
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
            beta: 0.75,
            lambda: 0.25,
            mu: 2.0,
            pruning_period_override: None,
            metric,
            p_micro_clusters: Vec::new(),
            o_micro_clusters: Vec::new(),
            timestamp: 0,
            updates_since_prune: 0,
            dim: None,
            initial_buffer_size: 1000,
            initial_buffer: Vec::new(),
            initialized: false,
        }
    }

    /// Set the weight threshold factor beta.
    ///
    /// An outlier is promoted when its weight exceeds `beta * mu`.
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

    /// Override the pruning period.
    ///
    /// By default, the period is derived from `beta`, `mu`, and `lambda` using
    /// the equation from Cao et al. This override is retained for applications
    /// that need an externally controlled update cadence.
    pub fn with_pruning_period(mut self, t_p: usize) -> Self {
        self.pruning_period_override = Some(t_p);
        self
    }

    /// Set the number of points used by the initialization phase.
    ///
    /// The default is 1,000. Setting this to zero starts the online phase
    /// immediately, which is useful for low-latency streams and preserves the
    /// pre-0.6 behavior.
    pub fn with_initial_buffer_size(mut self, size: usize) -> Self {
        self.initial_buffer_size = size;
        self.initialized = size == 0;
        self
    }

    /// Return whether the initialization phase has completed.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Validate the coupled DenStream parameters.
    fn validate_parameters(&self) -> Result<()> {
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "epsilon",
                message: "must be finite and greater than zero",
            });
        }
        if !self.macro_epsilon.is_finite() || self.macro_epsilon <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "macro_epsilon",
                message: "must be finite and greater than zero",
            });
        }
        if self.min_pts == 0 {
            return Err(Error::InvalidParameter {
                name: "min_pts",
                message: "must be greater than zero",
            });
        }
        if !self.beta.is_finite() || !(0.0..=1.0).contains(&self.beta) || self.beta == 0.0 {
            return Err(Error::InvalidParameter {
                name: "beta",
                message: "must be finite and in (0, 1]",
            });
        }
        if !self.mu.is_finite() || self.mu <= 0.0 || self.beta * self.mu <= 1.0 {
            return Err(Error::InvalidParameter {
                name: "mu",
                message: "must be finite, positive, and satisfy beta * mu > 1",
            });
        }
        if !self.lambda.is_finite() || self.lambda <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "lambda",
                message: "must be finite and greater than zero",
            });
        }
        if self.pruning_period_override == Some(0) {
            return Err(Error::InvalidParameter {
                name: "t_p",
                message: "must be greater than zero",
            });
        }
        Ok(())
    }

    /// Effective pruning period from Eq. 4 of Cao et al., rounded up to an
    /// integral stream timestamp.
    fn pruning_period(&self) -> usize {
        self.pruning_period_override.unwrap_or_else(|| {
            let threshold = self.beta * self.mu;
            ((threshold / (threshold - 1.0)).log2() / self.lambda).ceil() as usize
        })
    }

    /// Apply DenStream's weighted offline phase to potential micro-clusters.
    ///
    /// Returns one label per potential micro-cluster. Labels are cluster indices
    /// or `NOISE` (`usize::MAX`). A core micro-cluster has its own weight at
    /// least `mu`. Two micro-clusters are neighbors only when their Euclidean
    /// center distance is at most `macro_epsilon` and their radii overlap.
    /// The configured generic metric is used by the online nearest-cluster
    /// search only; this paper-defined phase is Euclidean.
    pub fn macro_cluster(&self) -> Result<Vec<usize>> {
        self.validate_parameters()?;
        if self.p_micro_clusters.is_empty() {
            return Err(Error::EmptyInput);
        }

        let mut clusters = self.p_micro_clusters.clone();
        for cluster in &mut clusters {
            cluster.decay(self.lambda, self.timestamp);
        }
        let centroids: Vec<Vec<f32>> = clusters.iter().map(MicroCluster::centroid).collect();
        let radii: Vec<f32> = clusters.iter().map(MicroCluster::radius).collect();
        let core: Vec<bool> = clusters.iter().map(|mc| mc.weight >= self.mu).collect();
        let euclidean = Euclidean;
        let neighbors = |a: usize, b: usize| {
            let distance = euclidean.distance(&centroids[a], &centroids[b]);
            distance <= self.macro_epsilon && distance <= radii[a] + radii[b]
        };

        let mut labels = vec![NOISE; centroids.len()];
        let mut next_label = 0;
        for seed in 0..centroids.len() {
            if !core[seed] || labels[seed] != NOISE {
                continue;
            }
            labels[seed] = next_label;
            let mut queue = vec![seed];
            let mut cursor = 0;
            while cursor < queue.len() {
                let current = queue[cursor];
                cursor += 1;
                for candidate in 0..centroids.len() {
                    if labels[candidate] == NOISE && neighbors(current, candidate) {
                        labels[candidate] = next_label;
                        if core[candidate] {
                            queue.push(candidate);
                        }
                    }
                }
            }
            next_label += 1;
        }
        Ok(labels)
    }

    /// Run the pre-0.6 unweighted centroid DBSCAN helper.
    pub fn macro_cluster_unweighted(&self) -> Result<Vec<usize>> {
        self.validate_parameters()?;
        if self.p_micro_clusters.is_empty() {
            return Err(Error::EmptyInput);
        }
        let centroids: Vec<Vec<f32>> = self
            .p_micro_clusters
            .iter()
            .map(MicroCluster::centroid)
            .collect();
        Dbscan::with_metric(self.macro_epsilon, self.min_pts, self.metric.clone())
            .fit_predict(&centroids)
    }

    fn initialize_from_buffer(&mut self) -> Result<()> {
        let min_weight = (self.beta * self.mu).ceil() as usize;
        let labels = Dbscan::new(self.epsilon, min_weight).fit_predict(&self.initial_buffer)?;
        let n_clusters = labels
            .iter()
            .copied()
            .filter(|&label| label != NOISE)
            .max()
            .map_or(0, |label| label + 1);
        for label in 0..n_clusters {
            let points: Vec<Vec<f32>> = self
                .initial_buffer
                .iter()
                .zip(&labels)
                .filter(|(_, &point_label)| point_label == label)
                .map(|(point, _)| point.clone())
                .collect();
            if points.len() as f64 >= self.beta * self.mu {
                self.p_micro_clusters
                    .push(MicroCluster::from_points(&points, self.timestamp));
            }
        }
        self.initial_buffer.clear();
        self.initialized = true;
        self.updates_since_prune = 0;
        Ok(())
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
        let t_p = self.pruning_period() as u64;
        let lam = self.lambda;
        self.o_micro_clusters.retain(|mc| {
            let age = current_ts.saturating_sub(mc.creation_time);
            mc.weight >= outlier_weight_threshold(lam, t_p, age)
        });
    }
}

/// Compute the weight threshold for an outlier micro-cluster given its age.
///
/// From the paper: xi(t_c, t) = (2^{-lambda*(t - t_c + t_p)} - 1) / (2^{-lambda*t_p} - 1).
/// An outlier whose weight falls below this raw `xi` value is pruned.
fn outlier_weight_threshold(lambda: f64, t_p: u64, age: u64) -> f64 {
    // exp_m1 retains the ratio's precision when lambda is very small.
    let denom = (-lambda * t_p as f64 * std::f64::consts::LN_2).exp_m1();
    let numer = (-lambda * (age + t_p) as f64 * std::f64::consts::LN_2).exp_m1();
    numer / denom
}

impl<D: DistanceMetric> DenStream<D> {
    /// Absorb a single point, returning the index of the nearest potential
    /// micro-cluster, or `NOISE` if the point was placed in an outlier cluster.
    /// Updates made during initialization also return `NOISE`; inspect
    /// [`Self::is_initialized`] before consuming online assignments.
    pub fn update(&mut self, point: &[f32]) -> Result<usize> {
        self.validate_parameters()?;
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

        if !self.initialized {
            self.initial_buffer.push(point.to_vec());
            if self.initial_buffer.len() >= self.initial_buffer_size {
                self.initialize_from_buffer()?;
            }
            return Ok(NOISE);
        }

        let potential_threshold = self.beta * self.mu;
        let mut assigned_p_idx = None;

        // Step 1: Try to absorb into nearest potential micro-cluster.
        if let Some((idx, _)) = self.nearest_micro_cluster(point, &self.p_micro_clusters) {
            // The paper's merge condition is the projected micro-cluster
            // radius, not the point-to-center distance.
            let new_radius = self.p_micro_clusters[idx].radius_if_absorbed(point, self.lambda, ts);
            if new_radius <= self.epsilon {
                self.p_micro_clusters[idx].absorb(point, self.lambda, ts);

                assigned_p_idx = Some(idx);
            }
        }

        // Step 2: If not absorbed into a p-cluster, try outlier micro-clusters.
        if assigned_p_idx.is_none() {
            let mut absorbed_into_outlier = false;
            if let Some((idx, _)) = self.nearest_micro_cluster(point, &self.o_micro_clusters) {
                let new_radius =
                    self.o_micro_clusters[idx].radius_if_absorbed(point, self.lambda, ts);
                if new_radius <= self.epsilon {
                    self.o_micro_clusters[idx].absorb(point, self.lambda, ts);
                    absorbed_into_outlier = true;

                    // Promotion is strict in the paper: w > beta * mu.
                    if self.o_micro_clusters[idx].weight > potential_threshold {
                        let promoted = self.o_micro_clusters.remove(idx);
                        self.p_micro_clusters.push(promoted);

                        // The promoted cluster is now the last p-cluster.
                        assigned_p_idx = Some(self.p_micro_clusters.len() - 1);
                    }
                }
            }

            // Step 3: If not absorbed anywhere, create new outlier micro-cluster.
            if !absorbed_into_outlier && assigned_p_idx.is_none() {
                // Online-start mode: without the paper's initialization buffer,
                // every new singleton begins as an outlier micro-cluster.
                self.o_micro_clusters.push(MicroCluster::new(point, ts));
            }
        }

        // Periodic pruning.
        self.updates_since_prune += 1;
        if self.updates_since_prune >= self.pruning_period() {
            self.prune();
            self.updates_since_prune = 0;
        }

        // Return the p-micro-cluster index, or NOISE if only in outlier.
        Ok(assigned_p_idx.unwrap_or(NOISE))
    }

    /// Update the model with a mini-batch of points.
    pub fn update_batch(&mut self, points: &(impl DataRef + ?Sized)) -> Result<Vec<usize>> {
        if points.n() == 0 {
            return Err(Error::EmptyInput);
        }

        let mut labels = Vec::with_capacity(points.n());
        for i in 0..points.n() {
            labels.push(self.update(points.row(i))?);
        }
        Ok(labels)
    }

    /// Predict the nearest potential micro-cluster for a point without
    /// modifying the model. Returns the cluster index or `NOISE` if no
    /// potential micro-cluster is within epsilon.
    pub fn predict(&self, point: &[f32]) -> Result<usize> {
        if self.p_micro_clusters.is_empty() {
            return Err(Error::InvalidParameter {
                name: "state",
                message: "no potential micro-clusters exist yet",
            });
        }
        if let Some(dim) = self.dim {
            if point.len() != dim {
                return Err(Error::DimensionMismatch {
                    expected: dim,
                    found: point.len(),
                });
            }
        }
        match self.nearest_micro_cluster(point, &self.p_micro_clusters) {
            Some((idx, dist)) if dist <= self.epsilon => Ok(idx),
            _ => Ok(super::dbscan::NOISE),
        }
    }

    /// Predict labels for multiple points without modifying the model.
    pub fn predict_batch(&self, points: &(impl DataRef + ?Sized)) -> Result<Vec<usize>> {
        (0..points.n())
            .map(|i| self.predict(points.row(i)))
            .collect()
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
    fn test_denstream() -> DenStream<Euclidean> {
        DenStream::new(2.0, 2)
            .with_initial_buffer_size(0)
            .with_beta(0.5)
            .with_lambda(0.001)
            .with_mu(3.0)
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

        // A singleton begins as an outlier. The second nearby point is absorbed
        // into the same summary.
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
            .with_initial_buffer_size(0)
            .with_beta(0.5)
            .with_lambda(1.0) // very aggressive decay
            .with_mu(3.0)
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
        let mut ds = DenStream::new(1.0, 1)
            .with_initial_buffer_size(0)
            .with_beta(0.2)
            .with_lambda(0.0001)
            .with_mu(6.0)
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
            .with_initial_buffer_size(0)
            .with_beta(0.5)
            .with_lambda(0.001)
            .with_mu(3.0);

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
            .with_initial_buffer_size(0)
            .with_beta(0.2)
            .with_lambda(0.1) // moderate decay
            .with_mu(6.0)
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

    /// Verify that MicroCluster weight after t decay steps with factor lambda
    /// matches the formula 2^(-lambda * t) within floating-point tolerance.
    ///
    /// This test accesses the private `weight` field directly; it lives in the
    /// module's test block rather than in the integration test suite because
    /// MicroCluster is not pub(crate).
    #[test]
    fn denstream_weight_decay_invariant() {
        let lambda = 0.5_f64;
        let t_steps = [1u64, 2, 5, 10];

        for &t in &t_steps {
            let mut mc = MicroCluster::new(&[0.0, 0.0], 0);
            // Weight starts at 1.0 at timestamp 0.
            assert!(
                (mc.weight - 1.0).abs() < 1e-12,
                "initial weight should be 1.0"
            );

            // Advance time by t without absorbing any points.
            mc.decay(lambda, t);

            let expected = 2.0_f64.powf(-lambda * t as f64);
            // Tolerance accounts for the f32 intermediate in apply_decay
            // (`decay` is computed as f32 before being cast back to f64).
            assert!(
                (mc.weight - expected).abs() < 1e-6,
                "weight after {t} steps with lambda={lambda}: got {}, expected {}",
                mc.weight,
                expected
            );
        }
    }

    #[test]
    fn defaults_derive_paper_pruning_period() {
        let ds = DenStream::new(1.0, 2);
        // ceil(log2(1.5 / 0.5) / 0.25) = ceil(log2(3) * 4) = 7.
        assert_eq!(ds.pruning_period(), 7);
    }

    #[test]
    fn outlier_threshold_matches_paper_equation() {
        let lambda = 0.25;
        let t_p = 7;
        let age = 11;
        let expected = (2.0_f64.powf(-lambda * 18.0) - 1.0) / (2.0_f64.powf(-lambda * 7.0) - 1.0);
        assert!((outlier_weight_threshold(lambda, t_p, age) - expected).abs() < 1e-12);

        // As lambda approaches zero, xi approaches (age + t_p) / t_p.
        let tiny_lambda = 1e-20;
        let expected_limit = (age + t_p) as f64 / t_p as f64;
        assert!((outlier_weight_threshold(tiny_lambda, t_p, age) - expected_limit).abs() < 1e-12);
    }

    #[test]
    fn singleton_is_outlier_and_promotion_is_strict() {
        let mut ds = DenStream::new(1.0, 1)
            .with_initial_buffer_size(0)
            .with_beta(0.75)
            .with_mu(2.0)
            .with_lambda(1.0)
            .with_pruning_period(100);

        assert_eq!(ds.update(&[0.0]).unwrap(), NOISE);
        assert_eq!(ds.p_micro_clusters.len(), 0);
        assert_eq!(ds.o_micro_clusters.len(), 1);

        // After one step: 1 * 2^-1 + 1 = beta * mu = 1.5, so equality
        // must not promote.
        assert_eq!(ds.update(&[0.0]).unwrap(), NOISE);
        assert_eq!(ds.p_micro_clusters.len(), 0);

        assert_ne!(ds.update(&[0.0]).unwrap(), NOISE);
        assert_eq!(ds.p_micro_clusters.len(), 1);
    }

    #[test]
    fn merge_uses_projected_radius_not_center_distance() {
        let mut ds = DenStream::new(1.0, 1)
            .with_initial_buffer_size(0)
            .with_beta(0.75)
            .with_mu(2.0)
            .with_lambda(0.000_001)
            .with_pruning_period(100);

        ds.update(&[0.0]).unwrap();
        // Squared center distance is 2.25 > epsilon, while the projected
        // two-point radius is 0.75 <= epsilon.
        ds.update(&[1.5]).unwrap();
        assert_eq!(ds.o_micro_clusters.len() + ds.p_micro_clusters.len(), 1);
    }

    #[test]
    fn projected_radius_applies_decay_before_testing_merge() {
        let cluster = MicroCluster::from_points(&[vec![-1.0], vec![1.0]], 0);
        assert!(cluster.radius_if_absorbed(&[0.0], 0.0, 10) > 0.8);
        assert!(cluster.radius_if_absorbed(&[0.0], 10.0, 10) < 0.01);
    }

    #[test]
    fn default_predict_uses_euclidean_epsilon_units() {
        let mut ds = DenStream::new(2.0, 1)
            .with_initial_buffer_size(0)
            .with_beta(0.5)
            .with_mu(3.0)
            .with_lambda(0.001);
        ds.p_micro_clusters.push(summary(0.0, 0.0, 3));
        ds.dim = Some(1);
        assert_eq!(ds.predict(&[1.5]).unwrap(), 0);
    }

    #[test]
    fn invalid_coupled_parameters_are_rejected() {
        let cases = [
            DenStream::new(0.0, 1),
            DenStream::new(1.0, 0),
            DenStream::new(1.0, 1).with_beta(0.0),
            DenStream::new(1.0, 1).with_beta(0.5).with_mu(2.0),
            DenStream::new(1.0, 1).with_lambda(0.0),
            DenStream::new(1.0, 1).with_pruning_period(0),
        ];
        for mut ds in cases {
            assert!(ds.update(&[0.0]).is_err());
        }
    }

    #[test]
    fn initialization_uses_raw_closed_epsilon_neighborhoods() {
        let mut ds = DenStream::new(2.0, 1)
            .with_initial_buffer_size(2)
            .with_beta(0.5)
            .with_mu(4.0)
            .with_lambda(10.0);

        assert!(!ds.is_initialized());
        ds.update(&[0.0]).unwrap();
        assert!(!ds.is_initialized());
        ds.update(&[2.0]).unwrap();

        assert!(ds.is_initialized());
        assert_eq!(ds.n_clusters(), 1);
        assert_eq!(ds.counts(), vec![2]);
        assert!((ds.centroids()[0][0] - 1.0).abs() < 1e-6);
        assert!((ds.p_micro_clusters[0].radius() - 1.0).abs() < 1e-6);
        assert!((ds.p_micro_clusters[0].weight - 2.0).abs() < 1e-12);
    }

    #[test]
    fn initialization_discards_noise_and_respects_beta_mu_threshold() {
        let mut ds = DenStream::new(0.25, 1)
            .with_initial_buffer_size(4)
            .with_beta(0.75)
            .with_mu(4.0)
            .with_lambda(0.1);
        for point in [[0.0], [0.1], [0.2], [10.0]] {
            ds.update(&point).unwrap();
        }
        assert_eq!(ds.n_clusters(), 1);
        assert_eq!(ds.counts(), vec![3]);
        assert!(ds.o_micro_clusters.is_empty());
    }

    #[test]
    fn zero_buffer_selects_immediate_lifecycle() {
        let mut ds = DenStream::new(1.0, 1)
            .with_initial_buffer_size(0)
            .with_beta(0.75)
            .with_mu(2.0)
            .with_lambda(0.1);
        assert!(ds.is_initialized());
        ds.update(&[0.0]).unwrap();
        assert_eq!(ds.o_micro_clusters.len(), 1);
    }

    fn summary(center: f32, radius: f32, weight: usize) -> MicroCluster {
        let spread = radius;
        let points: Vec<Vec<f32>> = if weight == 1 {
            vec![vec![center]]
        } else {
            (0..weight)
                .map(|index| {
                    if index % 2 == 0 {
                        vec![center - spread]
                    } else {
                        vec![center + spread]
                    }
                })
                .collect()
        };
        MicroCluster::from_points(&points, 0)
    }

    fn offline_model(clusters: Vec<MicroCluster>) -> DenStream<Euclidean> {
        let mut ds = DenStream::new(1.0, 1)
            .with_initial_buffer_size(0)
            .with_beta(0.5)
            .with_mu(3.0)
            .with_lambda(0.1)
            .with_macro_epsilon(2.0);
        ds.p_micro_clusters = clusters;
        ds
    }

    #[test]
    fn offline_density_is_each_micro_clusters_own_weight() {
        let ds = offline_model(vec![
            summary(0.0, 1.0, 2),
            summary(0.5, 1.0, 2),
            summary(1.0, 1.0, 2),
        ]);
        assert_eq!(ds.macro_cluster().unwrap(), vec![NOISE; 3]);
    }

    #[test]
    fn offline_core_weight_is_decayed_to_current_timestamp() {
        let mut ds = offline_model(vec![summary(0.0, 1.0, 4)]);
        ds.lambda = 1.0;
        ds.timestamp = 2;
        assert_eq!(ds.macro_cluster().unwrap(), vec![NOISE]);
    }

    #[test]
    fn offline_requires_radius_overlap() {
        let ds = offline_model(vec![summary(0.0, 0.0, 3), summary(0.5, 0.0, 3)]);
        assert_eq!(ds.macro_cluster().unwrap(), vec![0, 1]);
    }

    #[test]
    fn offline_border_micro_cluster_does_not_expand_chain() {
        let ds = offline_model(vec![
            summary(0.0, 1.0, 4),
            summary(1.5, 1.0, 2),
            summary(3.0, 1.0, 4),
        ]);
        assert_eq!(ds.macro_cluster().unwrap(), vec![0, 0, 1]);
    }

    #[test]
    fn offline_partition_is_permutation_invariant() {
        let first = offline_model(vec![
            summary(0.0, 1.0, 4),
            summary(1.0, 1.0, 4),
            summary(10.0, 1.0, 4),
        ]);
        let second = offline_model(vec![
            summary(10.0, 1.0, 4),
            summary(1.0, 1.0, 4),
            summary(0.0, 1.0, 4),
        ]);
        let a = first.macro_cluster().unwrap();
        let b = second.macro_cluster().unwrap();
        assert_eq!(a[0] == a[1], b[1] == b[2]);
        assert_eq!(a[0] == a[2], b[2] == b[0]);
    }

    #[test]
    fn unweighted_helper_matches_centroid_dbscan() {
        let mut ds = offline_model(vec![
            summary(0.0, 0.2, 4),
            summary(1.0, 0.2, 4),
            summary(10.0, 0.2, 4),
        ]);
        ds.min_pts = 2;
        assert_eq!(ds.macro_cluster_unweighted().unwrap(), vec![0, 0, NOISE]);
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
                .with_initial_buffer_size(0)
                .with_beta(0.5)
                .with_lambda(0.001)
                .with_mu(3.0)
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
                .with_initial_buffer_size(0)
                .with_beta(0.5)
                .with_lambda(0.001)
                .with_mu(3.0);

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
            .with_initial_buffer_size(0)
            .with_beta(0.5)
            .with_lambda(0.1)
            .with_mu(3.0);

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
