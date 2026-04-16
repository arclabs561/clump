//! CLAM: Clustering with Associative Memory helpers.
//!
//! Based on Saha et al. (2023), "End-to-End Differentiable Clustering with AM".
//!
//! Requires the `rkhs` feature: `clump = { version = "...", features = ["rkhs"] }`.
//!
//! ## Overview
//!
//! CLAM uses Dense Associative Memory (AM) energy functions to define a
//! differentiable clustering objective. Points are "contracted" toward cluster
//! centroids via AM energy descent, and the clustering loss measures how much
//! each point moves during contraction.
//!
//! - [`am_assign`]: hard (nearest-centroid) cluster assignment
//! - [`am_soft_assign`]: soft (probabilistic) cluster assignment
//! - [`am_contract`]: contract a point toward centroids via AM energy descent
//! - [`clam_loss`]: differentiable clustering loss

/// Hard assignment of points to cluster centroids (nearest centroid by Euclidean distance).
///
/// # Example
///
/// ```rust
/// use clump::clam::am_assign;
///
/// let data = vec![vec![0.1f64, 0.1], vec![9.9, 9.9]];
/// let centroids = vec![vec![0.0f64, 0.0], vec![10.0, 10.0]];
/// let labels = am_assign(&data, &centroids);
/// assert_eq!(labels, vec![0, 1]);
/// ```
pub fn am_assign(data: &[Vec<f64>], centroids: &[Vec<f64>]) -> Vec<usize> {
    data.iter()
        .map(|point| {
            let mut min_dist_sq = f64::MAX;
            let mut best_label = 0;
            for (i, centroid) in centroids.iter().enumerate() {
                let dist_sq: f64 = point
                    .iter()
                    .zip(centroid.iter())
                    .map(|(p, c)| (p - c).powi(2))
                    .sum();
                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                    best_label = i;
                }
            }
            best_label
        })
        .collect()
}

/// Soft assignment (probabilities) of points to cluster centroids.
///
/// `P(i) = exp(-beta/2 ||x - c_i||^2) / Σ_j exp(-beta/2 ||x - c_j||^2)`
///
/// # Example
///
/// ```rust
/// use clump::clam::am_soft_assign;
///
/// let data = vec![vec![0.0f64, 0.0]];
/// let centroids = vec![vec![0.0f64, 0.0], vec![10.0, 10.0]];
/// let probs = am_soft_assign(&data, &centroids, 1.0);
/// // Point at centroid 0 should assign mostly to centroid 0
/// assert!(probs[0][0] > probs[0][1]);
/// ```
pub fn am_soft_assign(data: &[Vec<f64>], centroids: &[Vec<f64>], beta: f64) -> Vec<Vec<f64>> {
    data.iter()
        .map(|point| {
            let neg_half_beta = -0.5 * beta;
            let log_weights: Vec<f64> = centroids
                .iter()
                .map(|c| {
                    let sq_dist: f64 = point
                        .iter()
                        .zip(c.iter())
                        .map(|(pi, ci)| (pi - ci).powi(2))
                        .sum();
                    neg_half_beta * sq_dist
                })
                .collect();

            let max_log = log_weights
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_weights: Vec<f64> = log_weights.iter().map(|&w| (w - max_log).exp()).collect();
            let sum_exp: f64 = exp_weights.iter().sum();

            exp_weights.iter().map(|&w| w / sum_exp).collect()
        })
        .collect()
}

/// Contract a point toward centroids using AM energy descent.
///
/// Uses the Log-Sum-Exp (LSE) energy from [`rkhs`]: gradient descent on
/// `E_β(v; Ξ) = -log Σ_μ exp(-β/2 ||v - ξ^μ||²)` converges toward the
/// nearest centroid attractor.
///
/// # Arguments
///
/// * `point` - Starting position
/// * `centroids` - Stored patterns (cluster centers)
/// * `beta` - Inverse temperature (larger = sharper contraction)
/// * `steps` - Maximum gradient descent iterations
/// * `lr` - Learning rate
///
/// # Example
///
/// ```rust
/// use clump::clam::am_contract;
///
/// let centroids = vec![vec![0.0f64, 0.0], vec![10.0, 10.0]];
/// let point = vec![0.5f64, 0.5];
/// let contracted = am_contract(&point, &centroids, 2.0, 50, 0.1);
/// // Should move toward centroid 0
/// assert!(contracted[0] < point[0] + 0.01);
/// ```
pub fn am_contract(
    point: &[f64],
    centroids: &[Vec<f64>],
    beta: f64,
    steps: usize,
    lr: f64,
) -> Vec<f64> {
    let (contracted, _) = rkhs::retrieve_memory(
        point.to_vec(),
        centroids,
        |v, m| rkhs::energy_lse_grad(v, m, beta),
        lr,
        steps,
        1e-10,
    );
    contracted
}

/// Differentiable clustering loss (CLAM loss).
///
/// `L = Σ_i ||x_i - contract(x_i, centroids)||^2`
///
/// Measures how much each point moves during AM energy descent toward the
/// centroids. Lower loss means points are already near centroids (better
/// clustering). Can be used as a gradient-free objective for centroid search.
///
/// # Arguments
///
/// * `data` - Input data points
/// * `centroids` - Current cluster centroids
/// * `beta` - Inverse temperature for AM energy
/// * `steps` - Contraction steps per point
/// * `lr` - Learning rate for contraction
pub fn clam_loss(
    data: &[Vec<f64>],
    centroids: &[Vec<f64>],
    beta: f64,
    steps: usize,
    lr: f64,
) -> f64 {
    data.iter()
        .map(|point| {
            let contracted = am_contract(point, centroids, beta, steps, lr);
            point
                .iter()
                .zip(contracted.iter())
                .map(|(p, c)| (p - c).powi(2))
                .sum::<f64>()
        })
        .sum()
}
