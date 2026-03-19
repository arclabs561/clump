#[derive(Clone, Debug)]
pub(crate) struct UnionFind {
    pub(crate) parent: Vec<usize>,
    pub(crate) size: Vec<usize>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    pub(crate) fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    pub(crate) fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        self.union_roots(ra, rb)
    }

    pub(crate) fn union_roots(&mut self, ra: usize, rb: usize) -> usize {
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

use super::distance::DistanceMetric;
use crate::error::{Error, Result};
use rand::prelude::*;

/// Validate that all values in the dataset are finite (no NaN or infinity).
///
/// Every `fit_predict` entry point should call this before doing any computation.
/// A single non-finite value poisons all distance calculations silently.
pub(crate) fn validate_finite(data: &[Vec<f32>]) -> Result<()> {
    for (i, point) in data.iter().enumerate() {
        for (j, &val) in point.iter().enumerate() {
            if !val.is_finite() {
                // Leak-free: use a static message since Error::InvalidParameter
                // takes &'static str. The index info goes through Other.
                return Err(Error::Other(format!(
                    "data[{i}][{j}] is not finite (NaN or infinity)"
                )));
            }
        }
    }
    Ok(())
}

/// Compute mean per-dimension variance of the dataset.
///
/// Used to normalize convergence tolerance so it scales with data magnitude.
pub(crate) fn mean_variance(data: &[Vec<f32>]) -> f64 {
    let n = data.len() as f64;
    if n < 1.0 {
        return 1.0;
    }
    let d = data[0].len();
    if d == 0 {
        return 1.0;
    }
    let mut total_var = 0.0f64;
    for j in 0..d {
        let mean = data.iter().map(|p| p[j] as f64).sum::<f64>() / n;
        let var = data
            .iter()
            .map(|p| {
                let diff = p[j] as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;
        total_var += var;
    }
    let mv = total_var / d as f64;
    if mv < f64::EPSILON {
        1.0
    } else {
        mv
    }
}

/// Update per-point minimum distances against a new centroid.
/// Parallelized when n >= 20000 and the `parallel` feature is enabled.
fn update_min_dists_for_centroid<D: DistanceMetric>(
    data: &[Vec<f32>],
    centroid: &[f32],
    min_dists: &mut [f32],
    metric: &D,
) {
    #[cfg(feature = "parallel")]
    if data.len() >= 20_000 {
        use rayon::prelude::*;
        min_dists.par_iter_mut().enumerate().for_each(|(i, md)| {
            let d = metric.distance(&data[i], centroid).max(0.0);
            if d < *md {
                *md = d;
            }
        });
        return;
    }

    for (i, md) in min_dists.iter_mut().enumerate() {
        let d = metric.distance(&data[i], centroid).max(0.0);
        if d < *md {
            *md = d;
        }
    }
}

/// Initialize centroids using k-means++ (Arthur & Vassilvitskii, 2007).
///
/// `data` is `&[Vec<f32>]`, `k` is the number of centroids to select.
/// `alpha` controls the distance weighting exponent: standard k-means++ uses 2.0
/// (D^2 weighting). Negative distances are clamped to 0 to handle non-metric
/// distances safely.
pub(crate) fn kmeanspp_init<D: DistanceMetric>(
    data: &[Vec<f32>],
    k: usize,
    metric: &D,
    alpha: f32,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let n = data.len();
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

    // Maintain per-point minimum distance to any selected centroid.
    // Avoids recomputing distances to all centroids each round --
    // only the new centroid needs checking (Rodriguez Corominas et al. 2024).
    let mut min_dists = vec![f32::MAX; n];

    // First centroid: random point.
    let first = rng.random_range(0..n);
    centroids.push(data[first].clone());

    // Set initial min_dists (all points vs first centroid).
    // Uses MAX as initial value, so the helper's `if d < *md` always fires.
    update_min_dists_for_centroid(data, &centroids[0], &mut min_dists, metric);

    // Precompute the exponent once. For the common case alpha=2.0,
    // the exponent is 1.0 and powf is a no-op -- use distances directly.
    let exp = alpha / 2.0;
    let identity_exp = (exp - 1.0).abs() < f32::EPSILON;

    // Remaining centroids: k-means++ selection.
    for _ in 1..k {
        let total: f32 = if identity_exp {
            min_dists.iter().sum()
        } else {
            min_dists.iter().map(|&d| d.powf(exp)).sum()
        };

        if total == 0.0 || !total.is_finite() {
            let idx = rng.random_range(0..n);
            centroids.push(data[idx].clone());
            let new_c = centroids.last().unwrap();
            update_min_dists_for_centroid(data, new_c, &mut min_dists, metric);
            continue;
        }

        let threshold = rng.random::<f32>() * total;
        let mut cumsum = 0.0f32;
        let mut selected = 0;
        for (j, &d) in min_dists.iter().enumerate() {
            let w = if identity_exp { d } else { d.powf(exp) };
            cumsum += w;
            if cumsum >= threshold {
                selected = j;
                break;
            }
        }

        centroids.push(data[selected].clone());

        // Update min_dists: only check the newly added centroid.
        let new_c = centroids.last().unwrap();
        update_min_dists_for_centroid(data, new_c, &mut min_dists, metric);
    }

    centroids
}

/// k-means|| initialization (Bahmani et al. 2012).
///
/// Oversamples O(k * log(n)) candidate centroids in O(log(n)) rounds,
/// then reduces to exactly k centroids via weighted k-means++.
/// Each round is O(n) and embarrassingly parallel.
///
/// Faster than k-means++ when k is large (k >= 50) because it does
/// O(log(k)) rounds of O(n) work instead of k rounds of O(n) work.
#[allow(dead_code)]
pub(crate) fn kmeanspar_init<D: DistanceMetric + Sync>(
    data: &[Vec<f32>],
    k: usize,
    metric: &D,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let n = data.len();
    if k >= n {
        return data.to_vec();
    }

    // Oversampling factor.
    let l = (2.0 * k as f64).ceil() as usize;

    // Start with one random center.
    let first = rng.random_range(0..n);
    let mut candidates: Vec<Vec<f32>> = vec![data[first].clone()];
    let mut min_dists = vec![f32::MAX; n];
    update_min_dists_for_centroid(data, &candidates[0], &mut min_dists, metric);

    // O(log(n)) rounds of oversampling.
    let rounds = (n as f64).ln().ceil() as usize;
    for _ in 0..rounds.max(1) {
        let cost: f64 = min_dists.iter().map(|&d| d as f64).sum();
        if cost <= 0.0 {
            break;
        }

        // Sample each point with probability proportional to its distance.
        let mut new_candidates = Vec::new();
        for i in 0..n {
            let prob = l as f64 * min_dists[i] as f64 / cost;
            if rng.random::<f64>() < prob {
                new_candidates.push(data[i].clone());
            }
        }

        // Update min_dists for all new candidates.
        for c in &new_candidates {
            update_min_dists_for_centroid(data, c, &mut min_dists, metric);
        }
        candidates.extend(new_candidates);

        if candidates.len() >= k * 5 {
            break; // Enough candidates.
        }
    }

    if candidates.len() <= k {
        // Not enough candidates -- pad with k-means++.
        while candidates.len() < k {
            let idx = rng.random_range(0..n);
            candidates.push(data[idx].clone());
        }
        return candidates;
    }

    // Reduce candidates to exactly k using weighted k-means++.
    // Weight each candidate by the number of data points closest to it.
    let mut weights = vec![0usize; candidates.len()];
    for point in data {
        let nearest = assign_nearest(point, &candidates, metric);
        weights[nearest] += 1;
    }

    // Weighted k-means++ selection from candidates.
    let mut final_centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut candidate_min_dists = vec![f32::MAX; candidates.len()];

    let first_idx = {
        let total_weight: f64 = weights.iter().map(|&w| w as f64).sum();
        let threshold = rng.random::<f64>() * total_weight;
        let mut cumsum = 0.0f64;
        let mut selected = 0;
        for (j, &w) in weights.iter().enumerate() {
            cumsum += w as f64;
            if cumsum >= threshold {
                selected = j;
                break;
            }
        }
        selected
    };
    final_centroids.push(candidates[first_idx].clone());

    for _ in 1..k {
        // Update min_dists for candidates.
        let last = final_centroids.last().unwrap();
        for (j, c) in candidates.iter().enumerate() {
            let d = metric.distance(c, last).max(0.0);
            if d < candidate_min_dists[j] {
                candidate_min_dists[j] = d;
            }
        }

        let total: f64 = candidate_min_dists
            .iter()
            .zip(weights.iter())
            .map(|(&d, &w)| d as f64 * w as f64)
            .sum();

        if total <= 0.0 || !total.is_finite() {
            let idx = rng.random_range(0..candidates.len());
            final_centroids.push(candidates[idx].clone());
            continue;
        }

        let threshold = rng.random::<f64>() * total;
        let mut cumsum = 0.0f64;
        let mut selected = 0;
        for (j, (&d, &w)) in candidate_min_dists.iter().zip(weights.iter()).enumerate() {
            cumsum += d as f64 * w as f64;
            if cumsum >= threshold {
                selected = j;
                break;
            }
        }
        final_centroids.push(candidates[selected].clone());
    }

    final_centroids
}

/// Assign a point to the nearest centroid. Returns the cluster index.
pub(crate) fn assign_nearest<D: DistanceMetric>(
    point: &[f32],
    centroids: &[Vec<f32>],
    metric: &D,
) -> usize {
    let mut best_cluster = 0;
    let mut best_dist = f32::MAX;
    for (k, centroid) in centroids.iter().enumerate() {
        let dist = metric.distance(point, centroid);
        if dist < best_dist {
            best_dist = dist;
            best_cluster = k;
        }
    }
    best_cluster
}

/// Geometric k-means assignment (inspired by Sharma et al. 2025).
///
/// Bound-free: uses inter-centroid distances to skip points whose
/// assignment provably cannot change. No per-point bound arrays.
/// O(k^2) centroid-pair precomputation, O(n) scan with early skip.
pub(crate) fn geometric_assign<D: DistanceMetric>(
    data: &[Vec<f32>],
    centroids: &[Vec<f32>],
    labels: &mut [usize],
    centroid_shifts: &[f32],
    metric: &D,
    first_iter: bool,
) {
    let n = data.len();
    let k = centroids.len();

    if first_iter || k <= 1 {
        for i in 0..n {
            let mut best = f32::MAX;
            let mut best_k = 0;
            for (j, c) in centroids.iter().enumerate() {
                let d = metric.distance(&data[i], c);
                if d < best {
                    best = d;
                    best_k = j;
                }
            }
            labels[i] = best_k;
        }
        return;
    }

    // Precompute half inter-centroid distances.
    let mut half_inter = vec![vec![0.0f32; k]; k];
    for j1 in 0..k {
        for j2 in (j1 + 1)..k {
            let d = metric.distance(&centroids[j1], &centroids[j2]) * 0.5;
            half_inter[j1][j2] = d;
            half_inter[j2][j1] = d;
        }
    }

    // For each point: skip if inter-centroid gap exceeds displacement sum.
    for i in 0..n {
        let assigned = labels[i];
        let my_shift = centroid_shifts[assigned];

        let mut can_skip = true;
        for j in 0..k {
            if j == assigned {
                continue;
            }
            // If any centroid could have gotten close enough to steal this point,
            // we must recompute.
            if half_inter[assigned][j] <= my_shift + centroid_shifts[j] {
                can_skip = false;
                break;
            }
        }

        if can_skip {
            continue;
        }

        let mut best = f32::MAX;
        let mut best_k = assigned;
        for (j, c) in centroids.iter().enumerate() {
            let d = metric.distance(&data[i], c);
            if d < best {
                best = d;
                best_k = j;
            }
        }
        labels[i] = best_k;
    }
}

/// Parallel Hamerly assignment using rayon.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn hamerly_assign_parallel<D: DistanceMetric>(
    data: &[Vec<f32>],
    centroids: &[Vec<f32>],
    labels: &mut [usize],
    upper: &mut [f32],
    lower: &mut [f32],
    centroid_shifts: &[f32],
    metric: &D,
    first_iter: bool,
    flat_buf: &mut Vec<f32>,
) {
    use rayon::prelude::*;
    let n = data.len();
    let k = centroids.len();
    let d = if k > 0 { centroids[0].len() } else { 0 };

    // Reuse caller-provided buffer for cache-friendly parallel access.
    flat_buf.clear();
    flat_buf.extend(centroids.iter().flat_map(|c| c.iter().copied()));
    let flat_centroids = &*flat_buf;

    if first_iter || k <= 1 {
        let results: Vec<(usize, f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut best = f32::MAX;
                let mut second = f32::MAX;
                let mut best_k = 0;
                for j in 0..k {
                    let c = &flat_centroids[j * d..(j + 1) * d];
                    let d = metric.distance(&data[i], c);
                    if d < best {
                        second = best;
                        best = d;
                        best_k = j;
                    } else if d < second {
                        second = d;
                    }
                }
                (best_k, best, second)
            })
            .collect();
        for (i, (lbl, u, l)) in results.into_iter().enumerate() {
            labels[i] = lbl;
            upper[i] = u;
            lower[i] = l;
        }
        return;
    }

    // Compute max and second-max centroid shifts for bounds update.
    let max_shift = centroid_shifts.iter().copied().fold(0.0f32, f32::max);
    let mut max_shift_idx = 0;
    for (j, &s) in centroid_shifts.iter().enumerate() {
        if s >= max_shift {
            max_shift_idx = j;
        }
    }
    let mut second_max_shift = 0.0f32;
    for (j, &s) in centroid_shifts.iter().enumerate() {
        if j != max_shift_idx && s > second_max_shift {
            second_max_shift = s;
        }
    }

    // Parallel bounds check and recompute.
    // Each element is (label, upper, lower) -- independent per point.
    let updates: Vec<(usize, f32, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let old_label = labels[i];
            let mut u = upper[i] + centroid_shifts[old_label];
            let relevant_max = if old_label == max_shift_idx {
                second_max_shift
            } else {
                max_shift
            };
            let l = lower[i] - relevant_max;

            if u <= l {
                return (old_label, u, l);
            }

            // Tighten upper bound using flat centroids.
            let old_c = &flat_centroids[old_label * d..(old_label + 1) * d];
            u = metric.distance(&data[i], old_c);

            if u <= l {
                return (old_label, u, l);
            }

            // Full recompute using flat centroids.
            let mut best = u;
            let mut second = f32::MAX;
            let mut best_k = old_label;
            for j in 0..k {
                if j == old_label {
                    if best < second {
                        second = best;
                    }
                    continue;
                }
                let c = &flat_centroids[j * d..(j + 1) * d];
                let dist = metric.distance(&data[i], c);
                if dist < best {
                    second = best;
                    best = dist;
                    best_k = j;
                } else if dist < second {
                    second = dist;
                }
            }
            (best_k, best, second)
        })
        .collect();

    for (i, (lbl, u, l)) in updates.into_iter().enumerate() {
        labels[i] = lbl;
        upper[i] = u;
        lower[i] = l;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hamerly_assign<D: DistanceMetric>(
    data: &[Vec<f32>],
    centroids: &[Vec<f32>],
    labels: &mut [usize],
    upper: &mut [f32],
    lower: &mut [f32],
    centroid_shifts: &[f32],
    metric: &D,
    first_iter: bool,
    flat_buf: &mut Vec<f32>,
) -> usize {
    let n = data.len();
    let k = centroids.len();
    let d = if k > 0 { centroids[0].len() } else { 0 };
    let mut recomputed = 0;

    // Reuse caller-provided buffer for cache-friendly access.
    // Only worth the copy overhead when k is large enough that cache misses
    // from Vec<Vec<f32>> pointer chasing dominate.
    let use_flat = k >= 16;
    flat_buf.clear();
    if use_flat {
        flat_buf.extend(centroids.iter().flat_map(|c| c.iter().copied()));
    }
    let flat_centroids = &*flat_buf;

    let batch_dist = |point: &[f32], centroid_idx: usize| -> f32 {
        if use_flat {
            let c = &flat_centroids[centroid_idx * d..(centroid_idx + 1) * d];
            metric.distance(point, c)
        } else {
            metric.distance(point, &centroids[centroid_idx])
        }
    };

    if first_iter || k <= 1 {
        for i in 0..n {
            let mut best = f32::MAX;
            let mut second = f32::MAX;
            let mut best_k = 0;
            for j in 0..k {
                let dist = batch_dist(&data[i], j);
                if dist < best {
                    second = best;
                    best = dist;
                    best_k = j;
                } else if dist < second {
                    second = dist;
                }
            }
            labels[i] = best_k;
            upper[i] = best;
            lower[i] = second;
        }
        return n;
    }

    // Update bounds based on centroid movement.
    // upper[i] += shift of assigned centroid (could have gotten farther)
    // lower[i] -= max shift of any other centroid (could have gotten closer)
    let max_shift = centroid_shifts.iter().copied().fold(0.0f32, f32::max);
    let mut second_max_shift = 0.0f32;
    let mut max_shift_idx = 0;
    for (j, &s) in centroid_shifts.iter().enumerate() {
        if s >= max_shift {
            max_shift_idx = j;
        }
    }
    for (j, &s) in centroid_shifts.iter().enumerate() {
        if j != max_shift_idx && s > second_max_shift {
            second_max_shift = s;
        }
    }

    for i in 0..n {
        upper[i] += centroid_shifts[labels[i]];
        // Lower bound decreases by the maximum shift of any centroid
        // other than the assigned one.
        let relevant_max = if labels[i] == max_shift_idx {
            second_max_shift
        } else {
            max_shift
        };
        lower[i] -= relevant_max;

        // Hamerly test: if upper bound <= lower bound, assignment is unchanged.
        if upper[i] <= lower[i] {
            continue;
        }

        // Tighten upper bound by recomputing distance to assigned centroid.
        upper[i] = batch_dist(&data[i], labels[i]);

        if upper[i] <= lower[i] {
            continue;
        }

        // Must recompute all distances using flat centroids.
        recomputed += 1;
        let mut best = upper[i];
        let mut second = f32::MAX;
        let mut best_k = labels[i];
        for j in 0..k {
            if j == labels[i] {
                if best < second {
                    second = best;
                }
                continue;
            }
            let dist = batch_dist(&data[i], j);
            if dist < best {
                second = best;
                best = dist;
                best_k = j;
            } else if dist < second {
                second = dist;
            }
        }
        labels[i] = best_k;
        upper[i] = best;
        lower[i] = second;
    }

    recomputed
}

/// Compute pairwise distance matrix for n points. Returns flat n*n row-major vec.
///
/// When the `parallel` feature is enabled, rows are computed in parallel
/// using rayon (O(n^2/p) with p cores).
pub(crate) fn pairwise_distance_matrix<D: DistanceMetric>(
    data: &[Vec<f32>],
    metric: &D,
) -> Vec<f32> {
    let n = data.len();
    let mut dists = vec![0.0f32; n * n];

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        // Compute each row's upper triangle in parallel.
        let rows: Vec<Vec<(usize, f32)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                ((i + 1)..n)
                    .map(|j| (j, metric.distance(&data[i], &data[j])))
                    .collect()
            })
            .collect();
        for (i, row) in rows.into_iter().enumerate() {
            for (j, d) in row {
                dists[i * n + j] = d;
                dists[j * n + i] = d;
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    for i in 0..n {
        for j in (i + 1)..n {
            let d = metric.distance(&data[i], &data[j]);
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }

    dists
}

/// Compute an MST for a dense complete graph using Prim's algorithm.
///
/// `dist_fn(i, j)` returns the edge weight between points `i` and `j`.
/// Returns edges `(u, v, dist)`.
///
/// When the `parallel` feature is enabled and n >= 5000, the inner update
/// loop is parallelized using rayon (O(n/p) per iteration with p cores).
pub(crate) fn prim_mst(
    n: usize,
    dist_fn: impl Fn(usize, usize) -> f32 + Sync,
) -> Vec<(usize, usize, f32)> {
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut best = vec![f32::INFINITY; n];
    let mut parent = vec![usize::MAX; n];

    best[0] = 0.0;
    let mut next_u = 0usize;

    for _ in 0..n {
        let u = next_u;
        if best[u] == f32::INFINITY && u != 0 {
            break;
        }
        in_tree[u] = true;

        // Parallel inner loop for large n.
        #[cfg(feature = "parallel")]
        if n >= 5000 {
            use rayon::prelude::*;
            // Parallel update: compute new distances and find min.
            // Each chunk returns its local (best_v, best_val, updates).
            let chunk_size = (n / rayon::current_num_threads().max(1)).max(256);
            let results: Vec<(usize, f32, Vec<(usize, f32, usize)>)> = (0..n)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local_best_v = usize::MAX;
                    let mut local_best_val = f32::INFINITY;
                    let mut updates = Vec::new();
                    for &v in chunk {
                        if in_tree[v] {
                            continue;
                        }
                        let d = dist_fn(u, v);
                        if d < best[v] {
                            updates.push((v, d, u));
                        }
                        // Use the potentially-updated value for min tracking.
                        let current = if d < best[v] { d } else { best[v] };
                        if current < local_best_val {
                            local_best_val = current;
                            local_best_v = v;
                        }
                    }
                    (local_best_v, local_best_val, updates)
                })
                .collect();

            // Apply updates sequentially.
            for (_, _, updates) in &results {
                for &(v, d, p) in updates {
                    if d < best[v] {
                        best[v] = d;
                        parent[v] = p;
                    }
                }
            }

            // Find global min.
            next_u = usize::MAX;
            let mut next_best = f32::INFINITY;
            for &(v, _val, _) in &results {
                if v != usize::MAX && best[v] < next_best {
                    next_best = best[v];
                    next_u = v;
                }
            }
            // Also re-scan to find true min after updates.
            for v in 0..n {
                if !in_tree[v] && best[v] < next_best {
                    next_best = best[v];
                    next_u = v;
                }
            }

            if next_u == usize::MAX {
                break;
            }
            continue;
        }

        // Serial inner loop.
        let mut next_best = f32::INFINITY;
        next_u = usize::MAX;
        for v in 0..n {
            if in_tree[v] {
                continue;
            }
            let d = dist_fn(u, v);
            if d < best[v] {
                best[v] = d;
                parent[v] = u;
            }
            if best[v] < next_best {
                next_best = best[v];
                next_u = v;
            }
        }

        if next_u == usize::MAX {
            break;
        }
    }

    let mut edges: Vec<(usize, usize, f32)> = Vec::with_capacity(n - 1);
    for v in 1..n {
        let u = parent[v];
        if u != usize::MAX {
            edges.push((u, v, best[v]));
        }
    }
    edges
}
