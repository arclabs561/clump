//! Euclidean e-means (Bessa et al., [2026](https://arxiv.org/abs/2611.09501)).
//! The automatic batch and step sizes follow the [Faiss CPU recipe](https://github.com/facebookresearch/faiss/blob/108b92a248790b0f065f57deeb072039f75efca5/contrib/e_means/base.py).

use super::{util, DataRef, SquaredEuclidean};
use crate::{Error, Result};
use rand::prelude::*;

/// Fits centroids to a fixed dataset with exponentially weighted mini-batch updates.
#[derive(Debug, Clone)]
pub struct Emeans {
    k: usize,
    epochs: f64,
    seed: Option<u64>,
}

/// Finalized centroids and their nearest-centroid assignments.
#[derive(Debug, Clone)]
pub struct EmeansFit {
    /// Unweighted means of the moving assignments from the closing pass.
    /// A center with no assignments in that pass retains its last position.
    pub centroids: Vec<Vec<f32>>,
    /// Nearest finalized centroid for every training row, in input order.
    pub labels: Vec<usize>,
}

impl EmeansFit {
    /// Assign new rows to the finalized centroids.
    pub fn predict(&self, data: &(impl DataRef + ?Sized)) -> Result<Vec<usize>> {
        let d = self.centroids.first().ok_or(Error::EmptyInput)?.len();
        validate_rows(data, d)?;
        let mut labels = vec![0; data.n()];
        util::assign_all(data, &self.centroids, &mut labels, &SquaredEuclidean);
        Ok(labels)
    }
}

fn validate_rows(data: &(impl DataRef + ?Sized), d: usize) -> Result<()> {
    if data.n() == 0 {
        return Err(Error::EmptyInput);
    }
    if d == 0 {
        return Err(Error::InvalidParameter {
            name: "dimension",
            message: "must be positive",
        });
    }
    for i in 0..data.n() {
        let found = data.row(i).len();
        if found != d {
            return Err(Error::DimensionMismatch { expected: d, found });
        }
    }
    util::validate_finite(data)
}

// Faiss selects the quality law through 16 dimensions and throughput above it.
fn recipe(n: usize, d: usize, k: usize, epochs: f64) -> (usize, f64) {
    let power = |v: f64| 2.0_f64.powf(v.max(1.0).log2().round()) as usize;
    let (coefficient, cap) = if d <= 16 {
        (0.00439175, n / 2)
    } else {
        (0.035134, power(n as f64 / 2.0))
    };
    let raw =
        coefficient * (k as f64).powf(0.134749) * (n as f64).powf(0.876473) * epochs.powf(0.813344);
    let batch = power(raw).min(cap).clamp(1, n);
    let alpha = 0.8244
        * (k as f64).powf(-0.4617)
        * (batch as f64).powf(0.9003)
        * (n as f64).powf(-0.4498)
        * epochs.powf(-0.1057);
    (batch, alpha.clamp(1e-4, 1.0))
}

fn step_size(alpha: f64, step: usize, total: usize) -> f64 {
    let phase = std::f64::consts::PI * step as f64 / total as f64;
    alpha * (0.1 + 0.45 * (1.0 + phase.cos()))
}

fn revive(centers: &mut [Vec<f32>], mass: &mut [f64], rng: &mut StdRng) {
    let threshold = 0.1 * mass.iter().sum::<f64>() / mass.len() as f64;
    let dead: Vec<_> = (0..mass.len()).filter(|&j| mass[j] < threshold).collect();
    if dead.is_empty() {
        return;
    }
    // Exponential races sample distinct donors in proportion to their EMA mass.
    let mut donors: Vec<_> = (0..mass.len())
        .filter(|&j| mass[j] >= threshold && mass[j] > 0.0)
        .map(|j| (-(1.0 - rng.random::<f64>()).ln() / mass[j], j))
        .collect();
    donors.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    for (child, (_, donor)) in dead.into_iter().zip(donors) {
        mass[donor] *= 0.5;
        mass[child] = mass[donor];
        for axis in 0..centers[donor].len() {
            let value = f64::from(centers[donor][axis]);
            let delta = if axis % 2 == 0 { 1e-4 } else { -1e-4 };
            let finite = |x: f64| x.clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32;
            centers[donor][axis] = finite(value * (1.0 - delta));
            centers[child][axis] = finite(value * (1.0 + delta));
        }
        if centers[child] == centers[donor] {
            let x = centers[child][0];
            centers[child][0] = if x == f32::MAX {
                x.next_down()
            } else {
                x.next_up()
            };
        }
    }
}

// Store the roundoff from each addition as a smaller, nonoverlapping partial.
// Ordinary f32 data sums exactly in f64 for many rows, leaving partials empty.
#[derive(Clone, Copy, Default)]
struct ClosingSum {
    sum: f64,
    partials: Option<std::num::NonZeroUsize>,
}

impl ClosingSum {
    // FastTwoSum with its operands ordered by magnitude.
    fn split_add(a: f64, b: f64) -> (f64, f64) {
        let (large, small) = if a.abs() >= b.abs() { (a, b) } else { (b, a) };
        let sum = large + small;
        (sum, small - (sum - large))
    }

    fn add(&mut self, value: f32, storage: &mut Vec<Vec<f64>>) {
        let mut value = f64::from(value);
        if let Some(index) = self.partials {
            storage[index.get() - 1].retain_mut(|partial| {
                let (sum, error) = Self::split_add(value, *partial);
                value = sum;
                *partial = error;
                error != 0.0
            });
        }
        let (sum, error) = Self::split_add(value, self.sum);
        self.sum = sum;
        if error != 0.0 {
            if let Some(index) = self.partials {
                storage[index.get() - 1].push(error);
            } else {
                storage.push(vec![error]);
                self.partials = std::num::NonZeroUsize::new(storage.len());
            }
        }
    }

    fn mean(&self, count: usize, storage: &[Vec<f64>]) -> f32 {
        let correction = self
            .partials
            .map_or(0.0, |i| storage[i.get() - 1].iter().sum::<f64>());
        ((correction + self.sum) / count as f64) as f32
    }
}

struct Totals {
    sums: Vec<Vec<f64>>,
    counts: Vec<usize>,
}

impl Totals {
    fn new(k: usize, d: usize) -> Self {
        Self {
            sums: vec![vec![0.; d]; k],
            counts: vec![0; k],
        }
    }

    fn add(&mut self, cluster: usize, point: &[f32]) {
        self.counts[cluster] += 1;
        for (sum, &x) in self.sums[cluster].iter_mut().zip(point) {
            *sum += f64::from(x);
        }
    }

    fn update(&mut self, centers: &mut [Vec<f32>], mass: &mut [f64], alpha: f64) {
        for (j, center) in centers.iter_mut().enumerate() {
            let count = self.counts[j] as f64;
            // At a unit step Faiss retains an empty cluster's prior mass.
            if alpha != 1.0 || count > 0.0 {
                mass[j] *= 1.0 - alpha;
            }
            if count > 0.0 {
                let added = alpha * count;
                mass[j] += added;
                let weight = added / mass[j];
                for (x, sum) in center.iter_mut().zip(&mut self.sums[j]) {
                    let old = f64::from(*x);
                    *x = (old + weight * (*sum / count - old)) as f32;
                    *sum = 0.0;
                }
                self.counts[j] = 0;
            }
        }
    }
}

impl Emeans {
    /// Create a squared-Euclidean clusterer with a budget of five passes.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            epochs: 5.0,
            seed: None,
        }
    }

    /// Set a positive training budget in passes. A fractional pass precedes
    /// the final full pass when the budget is at least one.
    pub fn with_epochs(mut self, epochs: f64) -> Self {
        self.epochs = epochs;
        self
    }

    /// Fix the initialization, batch order and donor sampling for repeated fits.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Fit e-means, then assign all rows to their nearest finalized centroids.
    pub fn fit(&self, data: &(impl DataRef + ?Sized)) -> Result<EmeansFit> {
        let (n, d) = (data.n(), data.d());
        validate_rows(data, d)?;
        if self.k == 0 || self.k > n {
            return Err(Error::InvalidClusterCount {
                requested: self.k,
                n_items: n,
            });
        }
        let bad_epochs = || Error::InvalidParameter {
            name: "epochs",
            message: "must give a finite, positive, representable number of updates",
        };
        if !self.epochs.is_finite() || self.epochs <= 0.0 || self.epochs >= usize::MAX as f64 {
            return Err(bad_epochs());
        }
        let whole = self.epochs.floor() as usize;
        let partial = (self.epochs.fract() * n as f64).round_ties_even() as usize;
        let (batch_size, alpha) = recipe(n, d, self.k, self.epochs);
        let total_steps = whole
            .checked_mul(n.div_ceil(batch_size))
            .and_then(|steps| steps.checked_add(partial.div_ceil(batch_size)))
            .filter(|&steps| steps > 0)
            .ok_or_else(bad_epochs)?;
        let passes = std::iter::repeat_n(n, whole.saturating_sub(1))
            .chain((partial > 0).then_some(partial))
            .chain((whole > 0).then_some(n));

        let mut rng = self
            .seed
            .map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64);
        let mut order: Vec<_> = (0..n).collect();
        if self.k < n {
            order.shuffle(&mut rng);
        }
        let mut centers: Vec<_> = order[..self.k]
            .iter()
            .map(|&i| data.row(i).to_vec())
            .collect();
        let mut mass = vec![1.0 / 65536.0; self.k];
        let mut batch = Totals::new(self.k, d);
        let mut closing = vec![vec![ClosingSum::default(); d]; self.k];
        let mut closing_counts = vec![0; self.k];
        let mut partials = Vec::new();
        let mut step = 0;
        let mut passes = passes.peekable();
        while let Some(rows) = passes.next() {
            let last = passes.peek().is_none();
            order.shuffle(&mut rng);
            for indices in order[..rows].chunks(batch_size) {
                for &i in indices {
                    let point = data.row(i);
                    let cluster = util::assign_nearest(point, &centers, &SquaredEuclidean);
                    batch.add(cluster, point);
                    if last {
                        closing_counts[cluster] += 1;
                        for (sum, &value) in closing[cluster].iter_mut().zip(point) {
                            sum.add(value, &mut partials);
                        }
                    }
                }
                batch.update(&mut centers, &mut mass, step_size(alpha, step, total_steps));
                step += 1;
                if d > 2 && step % 5 == 0 {
                    revive(&mut centers, &mut mass, &mut rng);
                }
            }
        }
        for (j, center) in centers.iter_mut().enumerate() {
            if closing_counts[j] > 0 {
                for (x, sum) in center.iter_mut().zip(&closing[j]) {
                    *x = sum.mean(closing_counts[j], &partials);
                }
            }
        }
        let mut labels = vec![0; n];
        util::assign_all(data, &centers, &mut labels, &SquaredEuclidean);
        Ok(EmeansFit {
            centroids: centers,
            labels,
        })
    }

    /// Return one nearest-centroid label per input row.
    pub fn fit_predict(&self, data: &(impl DataRef + ?Sized)) -> Result<Vec<usize>> {
        Ok(self.fit(data)?.labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FlatRef;

    #[test]
    fn matches_faiss_cpu_with_the_same_initial_rows_and_batch_order() {
        // NumpyEmeans at 108b92a, using begin/accumulate/update/end_epoch/finalize.
        // Feed it the same row permutations (Rust and NumPy use different RNGs).
        // The partial pass precedes closing; 257 rows leave a one-row tail.
        let data: Vec<_> = (0..257)
            .flat_map(|i| [((17 * i) % 47) as f32, ((i * i + 3 * i) % 37) as f32])
            .collect();
        let data = FlatRef::new(&data, 257, 2);
        let (batch, alpha) = recipe(257, 2, 5, 3.5);
        assert_eq!(batch, 2);
        assert!((alpha - 0.05283782425706363).abs() < 1e-12);
        assert_eq!(recipe(100000, 32, 128, 5.0), (8192, 1.0));
        let model = Emeans::new(5).with_epochs(3.5).with_seed(42);
        let fit = model.fit(&data).unwrap();
        let reference = [
            [23.282608, 25.652174],
            [12.637681, 7.6811595],
            [39.17647, 28.441177],
            [35.969696, 8.621212],
            [6.0, 25.880953],
        ];
        for (center, want) in fit.centroids.iter().zip(reference) {
            for (x, y) in center.iter().zip(want) {
                assert!((x - y).abs() < 1e-5);
            }
        }
        assert_eq!(fit.labels, fit.predict(&data).unwrap());
        assert_eq!(model.fit_predict(&data).unwrap(), fit.labels);
    }

    #[test]
    fn revival_uses_distinct_donors_and_halves_their_mass() {
        let mut centers = vec![vec![1., 2., 3.], vec![3., 4., 5.], vec![0.; 3], vec![0.; 3]];
        let mut mass = [4., 8., 0., 0.];
        revive(&mut centers, &mut mass, &mut StdRng::seed_from_u64(42));
        assert_eq!(mass[..2], [2., 4.]);
        mass[2..].sort_by(f64::total_cmp);
        assert_eq!(mass[2..], [2., 4.]);
        assert!(centers[0][0] < 1. && centers[0][1] > 2.);
        assert!(centers[2][0] > 1. && centers[3][0] > 1.);
    }

    #[test]
    fn one_cluster_closes_on_the_mean_and_sub_epoch_labels_every_row() {
        let data = [vec![0.], vec![3.], vec![6.]];
        assert_eq!(Emeans::new(1).fit(&data).unwrap().centroids, [vec![3.]]);
        let fit = Emeans::new(1).with_epochs(0.34).fit(&data).unwrap();
        assert!(data.contains(&fit.centroids[0]));
        assert_eq!(fit.labels, [0; 3]);
    }

    #[test]
    fn closing_mean_retains_small_residuals_across_seeded_orders() {
        // The large values cancel exactly on each axis.
        let large = 2_f32.powi(80);
        let data = [
            vec![large, -large],
            vec![1., -2.],
            vec![1., -2.],
            vec![-large, large],
        ];
        for seed in 0..32 {
            let fit = Emeans::new(1).with_seed(seed).fit(&data).unwrap();
            assert_eq!(fit.centroids, [vec![0.5, -1.]], "seed {seed}");
            assert_eq!(fit.labels, [0; 4]);
        }
    }

    #[test]
    fn closing_sum_handles_cancellation_at_multiple_scales() {
        let mut sum = ClosingSum::default();
        let mut partials = Vec::new();
        for x in [
            2_f32.powi(107),
            2_f32.powi(90),
            -2_f32.powi(54),
            -2_f32.powi(107),
            2_f32.powi(54),
            -12.,
            -4.,
            -2_f32.powi(90),
            13.,
        ] {
            sum.add(x, &mut partials);
        }
        assert_eq!(sum.mean(9, &partials), -1. / 3.);
    }

    #[test]
    fn rejects_invalid_training_inputs() {
        let valid = [vec![0.]];
        for epochs in [0., -1., f64::NAN, f64::INFINITY, 0.1] {
            assert!(Emeans::new(1).with_epochs(epochs).fit(&valid).is_err());
        }
        for k in [0, 2] {
            assert!(Emeans::new(k).fit(&valid).is_err());
        }
        let model = Emeans::new(1);
        assert!(model.fit(&[]).is_err());
        assert!(model.fit(&[vec![]]).is_err());
        assert!(model.fit(&[vec![f32::NAN]]).is_err());
        assert!(model.fit(&[vec![0.], vec![0., 1.]]).is_err());
    }
}
