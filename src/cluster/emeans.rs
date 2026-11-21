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
    speed: Option<f64>,
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

fn revive(
    centers: &mut [Vec<f32>],
    mass: &mut [f64],
    rng: &mut StdRng,
    mut split: impl FnMut(usize, usize),
) {
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
        split_pair(centers, mass, child, donor);
        split(child, donor);
    }
}

fn split_pair(centers: &mut [Vec<f32>], mass: &mut [f64], child: usize, donor: usize) {
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

// An expansion of nonoverlapping f64 terms, used only when the plain sum
// cannot establish the rounded f32 mean.
#[derive(Default)]
struct AccurateSum(Vec<f64>);

impl AccurateSum {
    fn add(&mut self, mut value: f64) {
        self.0.retain_mut(|partial| {
            let (large, small) = if value.abs() >= partial.abs() {
                (value, *partial)
            } else {
                (*partial, value)
            };
            value = large + small;
            *partial = small - (value - large);
            *partial != 0.0
        });
        if value != 0.0 {
            self.0.push(value);
        }
    }

    fn mean(mut self, mut count: usize) -> f32 {
        let approx = self.0.iter().sum::<f64>() / count as f64;
        let rounded = approx as f32;
        let center = f64::from(rounded);
        if approx == center {
            return rounded;
        }
        let next = if approx > center {
            rounded.next_up()
        } else {
            rounded.next_down()
        };
        if !next.is_finite() {
            return rounded;
        }

        // Compare the exact sum with count times the midpoint. The midpoint
        // and its powers-of-two multiples are exact in f64; their product
        // with the full count need not be. The largest surviving term gives
        // the sign even when the f64 approximation landed on the midpoint.
        let mut term = -(center + f64::from(next)) * 0.5;
        while count != 0 {
            if count & 1 != 0 {
                self.add(term);
            }
            term *= 2.0;
            count >>= 1;
        }
        let residual = self.0.last().copied().unwrap_or(0.0);
        if residual == 0.0 {
            if rounded.to_bits() & 1 == 0 {
                rounded
            } else {
                next
            }
        } else if (residual > 0.0) == (next > rounded) {
            next
        } else {
            rounded
        }
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

    fn update<const CLEAR: bool>(
        &mut self,
        centers: &mut [Vec<f32>],
        mass: &mut [f64],
        alpha: f64,
    ) {
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
                    if CLEAR {
                        *sum = 0.0;
                    }
                }
                if CLEAR {
                    self.counts[j] = 0;
                }
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
            speed: None,
        }
    }

    /// Use an independent constant-step EMA for the final readout. The
    /// multiplier is relative to the training schedule’s floor.
    pub fn with_shadow_speed(mut self, speed: f64) -> Self {
        self.speed = Some(speed);
        self
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
        let mut shadow = if let Some(speed) = self.speed {
            let rate = speed * 0.1 * alpha;
            if !rate.is_finite() || rate <= 0.0 || rate > 1.0 {
                return Err(Error::InvalidParameter {
                    name: "speed",
                    message: "must give a constant step in (0, 1]",
                });
            }
            Some((centers.clone(), mass.clone(), rate))
        } else {
            None
        };
        let mut batch = Totals::new(self.k, d);
        let mut closing = Totals::new(self.k, d);
        let mut closing_rows = 0;
        let mut largest = vec![0_u32; self.k];
        let mut labels = vec![0; n];
        let mut step = 0;
        let mut passes = passes.peekable();
        while let Some(rows) = passes.next() {
            let last = passes.peek().is_none();
            if last {
                closing_rows = rows;
            }
            order.shuffle(&mut rng);
            for indices in order[..rows].chunks(batch_size) {
                for &i in indices {
                    let point = data.row(i);
                    let cluster = util::assign_nearest(point, &centers, &SquaredEuclidean);
                    batch.add(cluster, point);
                    if last {
                        closing.add(cluster, point);
                        labels[i] = cluster;
                        // Positive float bits sort by magnitude; integer max vectorizes.
                        let max_bits = point.iter().map(|x| x.abs().to_bits()).max().unwrap();
                        largest[cluster] = largest[cluster].max(max_bits);
                    }
                }
                if let Some((centers, mass, rate)) = &mut shadow {
                    batch.update::<false>(centers, mass, *rate);
                }
                batch.update::<true>(&mut centers, &mut mass, step_size(alpha, step, total_steps));
                step += 1;
                if d > 2 && step % 5 == 0 {
                    revive(&mut centers, &mut mass, &mut rng, |child, donor| {
                        if let Some((centers, mass, _)) = &mut shadow {
                            split_pair(centers, mass, child, donor);
                        }
                    });
                }
            }
        }
        let mut retry: Vec<Vec<(usize, AccurateSum)>> = (0..self.k).map(|_| Vec::new()).collect();
        for (j, center) in centers.iter_mut().enumerate() {
            let count = closing.counts[j] as f64;
            if count == 0.0 {
                continue;
            }
            // Summing n terms in f64 incurs at most gamma(n) * sum(abs(x))
            // error. Bound sum(abs(x)) by n * max and allow for rounding.
            let error = if count * f64::EPSILON < 1.0 {
                2.0 * count * count * f64::from(f32::from_bits(largest[j])) * f64::EPSILON
            } else {
                f64::INFINITY
            };
            for (axis, (x, &sum)) in center.iter_mut().zip(&closing.sums[j]).enumerate() {
                *x = (sum / count) as f32;
                if ((sum - error) / count) as f32 != ((sum + error) / count) as f32 {
                    retry[j].push((axis, AccurateSum::default()));
                }
            }
        }
        if retry.iter().any(|axes| !axes.is_empty()) {
            for &i in &order[..closing_rows] {
                let point = data.row(i);
                for (axis, acc) in &mut retry[labels[i]] {
                    acc.add(f64::from(point[*axis]));
                }
            }
            for (j, axes) in retry.into_iter().enumerate() {
                for (axis, acc) in axes {
                    centers[j][axis] = acc.mean(closing.counts[j]);
                }
            }
        }
        if let Some((shadow_centers, _, _)) = shadow {
            centers = shadow_centers;
        }
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
        revive(
            &mut centers,
            &mut mass,
            &mut StdRng::seed_from_u64(42),
            |_, _| {},
        );
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
    fn fractional_closing_retry_only_uses_sampled_rows() {
        let large = 2_f32.powi(80);
        let rows = [
            vec![large, 0.],
            vec![1., 3.],
            vec![1., 6.],
            vec![-large, 9.],
        ];
        let mut seen = [false; 4];
        for seed in 0..64 {
            let fit = Emeans::new(1)
                .with_seed(seed)
                .with_epochs(0.75)
                .fit(&rows)
                .unwrap();
            // The second coordinate identifies the omitted row.
            let center = &fit.centroids[0];
            let omitted = [6., 5., 4., 3.]
                .iter()
                .position(|&y| y == center[1])
                .unwrap();
            seen[omitted] = true;
            let want = [-large / 3., 1. / 3., 1. / 3., large / 3.];
            assert_eq!(center[0], want[omitted]);
        }
        assert_eq!(seen, [true; 4]);
    }

    #[test]
    fn closing_mean_rounds_on_the_correct_side_of_a_midpoint() {
        let tiny = 2_f32.powi(-100);
        for low in [0.5_f32, 0.5_f32.next_up()] {
            let high = low.next_up();
            let even = if low.to_bits() & 1 == 0 { low } else { high };
            for (delta, expected) in [(-tiny, low), (0., even), (tiny, high)] {
                for sign in [1., -1.] {
                    let data = [2. * low, 2. * high, delta, 0.].map(|value| vec![value * sign]);
                    for seed in [0, 1, 7] {
                        let fit = Emeans::new(1).with_seed(seed).fit(&data).unwrap();
                        assert_eq!(fit.centroids[0][0].to_bits(), (expected * sign).to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn closing_sum_handles_cancellation_at_multiple_scales() {
        let mut sum = AccurateSum::default();
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
            sum.add(f64::from(x));
        }
        assert_eq!(sum.mean(9), -1. / 3.);
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
