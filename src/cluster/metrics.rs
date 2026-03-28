//! Cluster evaluation metrics.
//!
//! Functions for evaluating clustering quality without ground-truth labels.
//!
//! - [`silhouette_score`]: how well each point fits its cluster vs neighbors (-1 to 1)
//! - [`calinski_harabasz`]: ratio of between-cluster to within-cluster variance (higher = better)
//! - [`davies_bouldin`]: average worst-case cluster similarity (lower = better)
//! - [`disco_score`]: density-connectivity silhouette with noise evaluation (Beer et al. 2025)

use super::distance::DistanceMetric;
use super::flat::DataRef;

/// Debug-mode finiteness check for metrics input.
fn debug_validate_finite(data: &(impl DataRef + ?Sized)) {
    if cfg!(debug_assertions) {
        for i in 0..data.n() {
            let row = data.row(i);
            for (j, &val) in row.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "data[{i}][{j}] is not finite (NaN or infinity)"
                );
            }
        }
    }
}

/// Silhouette score: mean silhouette coefficient across all points.
///
/// For each point i:
/// - a(i) = mean distance to other points in the same cluster
/// - b(i) = mean distance to points in the nearest other cluster
/// - s(i) = (b(i) - a(i)) / max(a(i), b(i))
///
/// Returns the mean s(i) over all points. Range: [-1, 1].
/// Higher = better clustering. Score near 0 = overlapping clusters.
///
/// Complexity: O(n^2 * d) for exact computation. For large n, consider
/// sampling a subset of points.
pub fn silhouette_score<D: DistanceMetric>(
    data: &(impl DataRef + ?Sized),
    labels: &[usize],
    metric: &D,
) -> f32 {
    debug_validate_finite(data);
    let n = data.n();
    if n <= 1 {
        return 0.0;
    }

    // Find the number of clusters (exclude NOISE sentinel = usize::MAX).
    let noise = super::dbscan::NOISE;
    let k = labels
        .iter()
        .copied()
        .filter(|&l| l != noise)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    if k <= 1 {
        return 0.0;
    }

    // For each point, accumulate distance sums to each cluster.
    // cluster_dist_sum[i][c] = sum of distances from point i to all points in cluster c.
    // cluster_sizes[c] = number of points in cluster c.
    let mut cluster_sizes = vec![0usize; k];
    for &l in labels {
        if l < k {
            cluster_sizes[l] += 1;
        }
    }

    let mut total_silhouette = 0.0f64;

    let mut counted = 0usize;
    for i in 0..n {
        let ci = labels[i];
        if ci == noise || ci >= k {
            continue;
        }
        if cluster_sizes[ci] <= 1 {
            // Singleton cluster: silhouette undefined, treat as 0.
            continue;
        }

        // Accumulate distances from point i to each cluster.
        let mut cluster_dist_sum = vec![0.0f64; k];
        for j in 0..n {
            if i == j || labels[j] == noise || labels[j] >= k {
                continue;
            }
            let d = metric.distance(data.row(i), data.row(j)) as f64;
            cluster_dist_sum[labels[j]] += d;
        }

        // a(i) = mean intra-cluster distance.
        let a = cluster_dist_sum[ci] / (cluster_sizes[ci] - 1) as f64;

        // b(i) = min mean inter-cluster distance.
        let mut b = f64::MAX;
        for c in 0..k {
            if c == ci || cluster_sizes[c] == 0 {
                continue;
            }
            let mean_dist = cluster_dist_sum[c] / cluster_sizes[c] as f64;
            if mean_dist < b {
                b = mean_dist;
            }
        }

        let s = if a.max(b) > 0.0 {
            (b - a) / a.max(b)
        } else {
            0.0
        };
        total_silhouette += s;
        counted += 1;
    }

    if counted == 0 {
        return 0.0;
    }
    (total_silhouette / counted as f64) as f32
}

/// Calinski-Harabasz index (variance ratio criterion).
///
/// Ratio of between-cluster dispersion to within-cluster dispersion,
/// adjusted by degrees of freedom. Higher = better.
///
/// Complexity: O(n * k * d). Only needs centroids + one data pass.
pub fn calinski_harabasz(
    data: &(impl DataRef + ?Sized),
    labels: &[usize],
    centroids: &[Vec<f32>],
) -> f32 {
    debug_validate_finite(data);
    let n = data.n();
    let k = centroids.len();
    if k <= 1 || n <= k {
        return 0.0;
    }

    let d = data.d();

    // Global centroid.
    let mut global = vec![0.0f64; d];
    for i in 0..n {
        for (j, &x) in data.row(i).iter().enumerate() {
            global[j] += x as f64;
        }
    }
    for x in &mut global {
        *x /= n as f64;
    }

    // Cluster sizes (skip noise points).
    let noise = super::dbscan::NOISE;
    let mut sizes = vec![0usize; k];
    for &l in labels {
        if l != noise && l < k {
            sizes[l] += 1;
        }
    }

    // Between-cluster dispersion: sum of n_k * ||c_k - c_global||^2.
    let mut between = 0.0f64;
    for (ki, centroid) in centroids.iter().enumerate() {
        let sq_dist: f64 = centroid
            .iter()
            .zip(global.iter())
            .map(|(&c, &g)| {
                let d = c as f64 - g;
                d * d
            })
            .sum();
        between += sizes[ki] as f64 * sq_dist;
    }

    // Within-cluster dispersion: sum of ||x_i - c_{label(i)}||^2.
    // Skip noise points to avoid index-out-of-bounds.
    let mut within = 0.0f64;
    #[allow(clippy::needless_range_loop)] // i indexes both labels and data.row
    for i in 0..n {
        let l = labels[i];
        if l == noise || l >= k {
            continue;
        }
        let centroid = &centroids[l];
        let sq_dist: f64 = data
            .row(i)
            .iter()
            .zip(centroid.iter())
            .map(|(&x, &c)| {
                let d = x as f64 - c as f64;
                d * d
            })
            .sum();
        within += sq_dist;
    }

    if within < f64::EPSILON {
        return f32::MAX;
    }

    let ch = (between / (k - 1) as f64) / (within / (n - k) as f64);
    ch as f32
}

/// Davies-Bouldin index.
///
/// Average of the worst-case cluster similarity ratio. Lower = better.
/// DB = (1/k) * sum_i max_{j != i} (S_i + S_j) / d(c_i, c_j)
/// where S_i = mean intra-cluster distance for cluster i.
///
/// Complexity: O(n * d + k^2 * d).
pub fn davies_bouldin<D: DistanceMetric>(
    data: &(impl DataRef + ?Sized),
    labels: &[usize],
    centroids: &[Vec<f32>],
    metric: &D,
) -> f32 {
    debug_validate_finite(data);
    let k = centroids.len();
    if k <= 1 {
        return 0.0;
    }

    // Mean intra-cluster distance per cluster (skip noise points).
    let noise = super::dbscan::NOISE;
    let mut intra_sum = vec![0.0f64; k];
    let mut sizes = vec![0usize; k];
    #[allow(clippy::needless_range_loop)] // i indexes both labels and data.row
    for i in 0..data.n() {
        let ci = labels[i];
        if ci == noise || ci >= k {
            continue;
        }
        let d = metric.distance(data.row(i), &centroids[ci]) as f64;
        intra_sum[ci] += d;
        sizes[ci] += 1;
    }

    let scatter: Vec<f64> = (0..k)
        .map(|i| {
            if sizes[i] > 0 {
                intra_sum[i] / sizes[i] as f64
            } else {
                0.0
            }
        })
        .collect();

    // For each cluster i, find max_j (S_i + S_j) / d(c_i, c_j).
    let mut db_sum = 0.0f64;
    for i in 0..k {
        let mut max_ratio = 0.0f64;
        for j in 0..k {
            if i == j {
                continue;
            }
            let d = metric.distance(&centroids[i], &centroids[j]) as f64;
            if d > f64::EPSILON {
                let ratio = (scatter[i] + scatter[j]) / d;
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
        }
        db_sum += max_ratio;
    }

    (db_sum / k as f64) as f32
}

/// Sampled silhouette score for large datasets.
///
/// Computes the silhouette score on a random sample of `sample_size` points.
/// Error scales as O(1/sqrt(sample_size)). For n > 10k, sampling 2000-5000
/// points gives a good estimate with O(sample^2) instead of O(n^2) cost.
pub fn silhouette_score_sampled<D: DistanceMetric>(
    data: &(impl DataRef + ?Sized),
    labels: &[usize],
    metric: &D,
    sample_size: usize,
    seed: u64,
) -> f32 {
    debug_validate_finite(data);
    let n = data.n();
    if n <= sample_size {
        return silhouette_score(data, labels, metric);
    }

    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    indices.truncate(sample_size);

    let sampled_data: Vec<Vec<f32>> = indices.iter().map(|&i| data.row(i).to_vec()).collect();
    let sampled_labels: Vec<usize> = indices.iter().map(|&i| labels[i]).collect();

    silhouette_score(&sampled_data, &sampled_labels, metric)
}

/// Noise-aware silhouette score for density-based clustering.
///
/// Like standard silhouette, but noise points (labeled `NOISE = usize::MAX`)
/// are excluded from the computation rather than being treated as misclassified.
/// This gives more meaningful scores for DBSCAN/HDBSCAN results where noise
/// is an intentional output, not a failure.
///
/// Returns 0.0 if all points are noise or only one cluster exists.
pub fn silhouette_score_noise_aware<D: DistanceMetric>(
    data: &(impl DataRef + ?Sized),
    labels: &[usize],
    metric: &D,
) -> f32 {
    debug_validate_finite(data);
    let noise = super::dbscan::NOISE;
    let n = data.n();

    // Filter to non-noise points.
    let non_noise_indices: Vec<usize> = (0..n).filter(|&i| labels[i] != noise).collect();
    if non_noise_indices.len() <= 1 {
        return 0.0;
    }

    let max_label = non_noise_indices
        .iter()
        .map(|&i| labels[i])
        .max()
        .unwrap_or(0);
    let k = max_label + 1;
    if k <= 1 {
        return 0.0;
    }

    let mut cluster_sizes = vec![0usize; k];
    for &i in &non_noise_indices {
        cluster_sizes[labels[i]] += 1;
    }

    let mut total = 0.0f64;

    for &i in &non_noise_indices {
        let ci = labels[i];
        if cluster_sizes[ci] <= 1 {
            continue;
        }

        let mut cluster_dist_sum = vec![0.0f64; k];
        for &j in &non_noise_indices {
            if i == j {
                continue;
            }
            let d = metric.distance(data.row(i), data.row(j)) as f64;
            cluster_dist_sum[labels[j]] += d;
        }

        let a = cluster_dist_sum[ci] / (cluster_sizes[ci] - 1) as f64;

        let mut b = f64::MAX;
        for c in 0..k {
            if c == ci || cluster_sizes[c] == 0 {
                continue;
            }
            let mean_dist = cluster_dist_sum[c] / cluster_sizes[c] as f64;
            if mean_dist < b {
                b = mean_dist;
            }
        }

        let s = if a.max(b) > 0.0 {
            (b - a) / a.max(b)
        } else {
            0.0
        };
        total += s;
    }

    (total / non_noise_indices.len() as f64) as f32
}

/// Adjusted Rand Index (ARI) for comparing two clusterings.
///
/// Measures agreement between two label vectors, adjusted for chance.
/// Range: [-1, 1]. ARI = 1 means perfect agreement, ARI = 0 means
/// random agreement, ARI < 0 means worse than random.
///
/// Useful for comparing a clustering result against ground truth, or
/// for comparing two different algorithms on the same data.
///
/// Noise labels (`usize::MAX`) are treated as a special cluster.
pub fn adjusted_rand_index(labels_a: &[usize], labels_b: &[usize]) -> f64 {
    let n = labels_a.len();
    assert_eq!(n, labels_b.len());
    if n <= 1 {
        return 1.0;
    }

    // Build contingency table using HashMaps (sparse).
    use std::collections::HashMap;
    let mut contingency: HashMap<(usize, usize), usize> = HashMap::new();
    let mut row_sums: HashMap<usize, usize> = HashMap::new();
    let mut col_sums: HashMap<usize, usize> = HashMap::new();

    for i in 0..n {
        *contingency.entry((labels_a[i], labels_b[i])).or_insert(0) += 1;
        *row_sums.entry(labels_a[i]).or_insert(0) += 1;
        *col_sums.entry(labels_b[i]).or_insert(0) += 1;
    }

    // ARI = (index - expected) / (max_index - expected)
    let comb2 = |x: usize| -> f64 { (x as f64) * (x as f64 - 1.0) / 2.0 };

    let sum_comb_nij: f64 = contingency.values().map(|&v| comb2(v)).sum();
    let sum_comb_ai: f64 = row_sums.values().map(|&v| comb2(v)).sum();
    let sum_comb_bi: f64 = col_sums.values().map(|&v| comb2(v)).sum();
    let comb_n = comb2(n);

    let expected = sum_comb_ai * sum_comb_bi / comb_n;
    let max_index = (sum_comb_ai + sum_comb_bi) / 2.0;

    if (max_index - expected).abs() < f64::EPSILON {
        return if (sum_comb_nij - expected).abs() < f64::EPSILON {
            1.0
        } else {
            0.0
        };
    }

    (sum_comb_nij - expected) / (max_index - expected)
}

/// K-distance curve for DBSCAN epsilon selection.
///
/// Computes the distance to each point's k-th nearest neighbor, sorted in
/// ascending order. The "elbow" (sharpest increase) suggests a good epsilon
/// value for DBSCAN.
///
/// `k` should typically be `min_pts - 1` (same parameter you'd use for DBSCAN).
///
/// Returns a sorted vector of k-th nearest neighbor distances.
pub fn k_distance<D: DistanceMetric>(
    data: &(impl DataRef + ?Sized),
    k: usize,
    metric: &D,
) -> Vec<f32> {
    debug_validate_finite(data);
    let n = data.n();
    let k = k.min(n.saturating_sub(1)).max(1);
    let mut k_dists = Vec::with_capacity(n);
    let mut dists = Vec::with_capacity(n.saturating_sub(1));

    for i in 0..n {
        dists.clear();
        dists.extend(
            (0..n)
                .filter(|&j| j != i)
                .map(|j| metric.distance(data.row(i), data.row(j))),
        );
        dists.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
        k_dists.push(dists[k - 1]);
    }

    k_dists.sort_by(|a, b| a.total_cmp(b));
    k_dists
}

/// DISCO: density-connectivity-based cluster validation with noise evaluation.
///
/// Evaluates both cluster quality AND noise assignment quality, unlike standard
/// silhouette which ignores noise. Uses mutual reachability distance as the
/// density-connectivity measure (Beer et al. 2025, ICLR 2026).
///
/// For each clustered point, computes a silhouette-like score using mutual
/// reachability distance (MRD) instead of raw distance. For each noise point,
/// evaluates whether the noise label is justified: if the point's mean MRD to
/// its nearest cluster exceeds that cluster's internal mean MRD, noise is
/// correct (score +1); otherwise the point should have been clustered (score -1).
///
/// `min_pts` controls the density estimate (same parameter as DBSCAN/HDBSCAN).
/// Range: \[-1, 1\]. Higher = better clustering with correct noise assignments.
pub fn disco_score<D: DistanceMetric>(
    data: &(impl DataRef + ?Sized),
    labels: &[usize],
    metric: &D,
    min_pts: usize,
) -> f32 {
    debug_validate_finite(data);
    let n = data.n();
    let noise = super::dbscan::NOISE;

    if n <= 1 {
        return 0.0;
    }

    // All noise or single cluster -> 0.0.
    let k = labels
        .iter()
        .copied()
        .filter(|&l| l != noise)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    if k == 0 {
        return 0.0;
    }
    let has_multiple_clusters = k >= 2;

    // Step 1: Compute pairwise raw distances.
    let mut raw_dist = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = metric.distance(data.row(i), data.row(j));
            raw_dist[i * n + j] = d;
            raw_dist[j * n + i] = d;
        }
    }

    // Step 2: Core distances (distance to min_pts-th nearest neighbor).
    let k_nn = min_pts.min(n - 1).max(1);
    let mut core_dist = vec![0.0f32; n];
    let mut neighbor_dists = Vec::with_capacity(n);
    for i in 0..n {
        neighbor_dists.clear();
        for j in 0..n {
            if i != j {
                neighbor_dists.push(raw_dist[i * n + j]);
            }
        }
        neighbor_dists.select_nth_unstable_by(k_nn - 1, |a, b| a.total_cmp(b));
        core_dist[i] = neighbor_dists[k_nn - 1];
    }

    // Step 3: Mutual reachability distance matrix.
    // mrd(i,j) = max(core_dist[i], core_dist[j], raw_dist(i,j))
    let mut mrd = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = raw_dist[i * n + j].max(core_dist[i]).max(core_dist[j]);
            mrd[i * n + j] = d;
            mrd[j * n + i] = d;
        }
    }

    // Cluster sizes.
    let mut cluster_sizes = vec![0usize; k];
    for &l in labels {
        if l < k {
            cluster_sizes[l] += 1;
        }
    }

    // Single cluster with no noise -> 0.0 (no inter-cluster comparison possible).
    if !has_multiple_clusters {
        // Check if there are noise points to score.
        let has_noise = labels.contains(&noise);
        if !has_noise {
            return 0.0;
        }
    }

    // Precompute mean intra-cluster MRD for each cluster (used for noise evaluation).
    let mut intra_mrd_sum = vec![0.0f64; k];
    let mut intra_mrd_count = vec![0usize; k];
    for i in 0..n {
        let ci = labels[i];
        if ci >= k {
            continue;
        }
        for j in (i + 1)..n {
            if labels[j] == ci {
                let d = mrd[i * n + j] as f64;
                intra_mrd_sum[ci] += d;
                intra_mrd_count[ci] += 1;
            }
        }
    }
    let intra_mrd_mean: Vec<f64> = (0..k)
        .map(|c| {
            if intra_mrd_count[c] > 0 {
                intra_mrd_sum[c] / intra_mrd_count[c] as f64
            } else {
                0.0
            }
        })
        .collect();

    let mut total_score = 0.0f64;
    let mut scored = 0usize;

    // Step 4: Silhouette-like score for non-noise points.
    if has_multiple_clusters {
        for i in 0..n {
            let ci = labels[i];
            if ci == noise || ci >= k {
                continue;
            }
            if cluster_sizes[ci] <= 1 {
                // Singleton: score 0 (Rousseeuw convention).
                scored += 1;
                continue;
            }

            // a(i) = mean MRD to same-cluster points.
            let mut a_sum = 0.0f64;
            let mut a_count = 0usize;
            let mut cluster_mrd_sum = vec![0.0f64; k];
            let mut cluster_mrd_count = vec![0usize; k];

            for j in 0..n {
                if i == j {
                    continue;
                }
                let lj = labels[j];
                if lj == noise || lj >= k {
                    continue;
                }
                let d = mrd[i * n + j] as f64;
                cluster_mrd_sum[lj] += d;
                cluster_mrd_count[lj] += 1;
                if lj == ci {
                    a_sum += d;
                    a_count += 1;
                }
            }

            let a = if a_count > 0 {
                a_sum / a_count as f64
            } else {
                0.0
            };

            // b(i) = min mean MRD to another cluster.
            let mut b = f64::MAX;
            for c in 0..k {
                if c == ci || cluster_mrd_count[c] == 0 {
                    continue;
                }
                let mean_d = cluster_mrd_sum[c] / cluster_mrd_count[c] as f64;
                if mean_d < b {
                    b = mean_d;
                }
            }

            let s = if b == f64::MAX {
                0.0
            } else if a.max(b) > 0.0 {
                (b - a) / a.max(b)
            } else {
                0.0
            };
            total_score += s;
            scored += 1;
        }
    }

    // Step 5: Noise point evaluation.
    for i in 0..n {
        if labels[i] != noise {
            continue;
        }
        // Find nearest cluster by mean MRD.
        let mut best_cluster = 0usize;
        let mut best_mean = f64::MAX;
        #[allow(clippy::needless_range_loop)]
        // c indexes cluster_sizes and iterates cluster members
        for c in 0..k {
            if cluster_sizes[c] == 0 {
                continue;
            }
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            for j in 0..n {
                if labels[j] == c {
                    sum += mrd[i * n + j] as f64;
                    cnt += 1;
                }
            }
            if cnt > 0 {
                let mean = sum / cnt as f64;
                if mean < best_mean {
                    best_mean = mean;
                    best_cluster = c;
                }
            }
        }

        // If mean MRD to nearest cluster > cluster's internal mean MRD,
        // the noise label is justified -> +1. Otherwise -> -1.
        let s = if best_mean > intra_mrd_mean[best_cluster] {
            1.0
        } else {
            -1.0
        };
        total_score += s;
        scored += 1;
    }

    if scored == 0 {
        return 0.0;
    }
    (total_score / scored as f64) as f32
}

/// Noise ratio: fraction of points labeled as noise.
///
/// Useful as a companion metric for density-based clustering.
/// Range: [0, 1]. Zero = no noise, 1 = all noise.
pub fn noise_ratio(labels: &[usize]) -> f32 {
    let noise = super::dbscan::NOISE;
    let noise_count = labels.iter().filter(|&&l| l == noise).count();
    noise_count as f32 / labels.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::distance::Euclidean;

    #[test]
    fn silhouette_perfect_clusters() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data, &labels, &Euclidean);
        assert!(
            score > 0.9,
            "perfect clusters should have silhouette > 0.9, got {score}"
        );
    }

    #[test]
    fn silhouette_single_cluster() {
        let data = vec![vec![0.0], vec![1.0], vec![2.0]];
        let labels = vec![0, 0, 0];
        let score = silhouette_score(&data, &labels, &Euclidean);
        assert!(
            score.abs() < 0.01,
            "single cluster should have silhouette ~0"
        );
    }

    #[test]
    fn calinski_harabasz_well_separated() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let labels = vec![0, 0, 1, 1];
        let centroids = vec![vec![0.05, 0.05], vec![10.05, 10.05]];
        let ch = calinski_harabasz(&data, &labels, &centroids);
        assert!(
            ch > 100.0,
            "well-separated clusters should have high CH, got {ch}"
        );
    }

    #[test]
    fn davies_bouldin_well_separated() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let labels = vec![0, 0, 1, 1];
        let centroids = vec![vec![0.05, 0.05], vec![10.05, 10.05]];
        let db = davies_bouldin(&data, &labels, &centroids, &Euclidean);
        assert!(
            db < 0.1,
            "well-separated clusters should have low DB, got {db}"
        );
    }

    #[test]
    fn calinski_harabasz_with_noise_labels() {
        let noise = crate::NOISE;
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![50.0, 50.0], // noise point
        ];
        let labels = vec![0, 0, 1, 1, noise];
        let centroids = vec![vec![0.05, 0.05], vec![10.05, 10.05]];
        // Must not panic on NOISE labels.
        let ch = calinski_harabasz(&data, &labels, &centroids);
        assert!(ch > 0.0, "CH should be positive, got {ch}");
    }

    #[test]
    fn davies_bouldin_with_noise_labels() {
        let noise = crate::NOISE;
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
            vec![50.0, 50.0], // noise point
        ];
        let labels = vec![0, 0, 1, 1, noise];
        let centroids = vec![vec![0.05, 0.05], vec![10.05, 10.05]];
        // Must not panic on NOISE labels.
        let db = davies_bouldin(&data, &labels, &centroids, &Euclidean);
        assert!(db < 1.0, "DB should be low for well-separated, got {db}");
    }

    #[test]
    fn silhouette_noise_aware_excludes_noise() {
        let noise = crate::NOISE;
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
            vec![50.0, 50.0], // noise point
        ];
        let labels = vec![0, 0, 1, 1, noise];

        let aware = silhouette_score_noise_aware(&data, &labels, &Euclidean);

        // Noise-aware on well-separated clusters should give a high score.
        assert!(
            aware > 0.8,
            "noise-aware should be high for well-separated clusters, got {aware}"
        );
    }

    #[test]
    fn silhouette_noise_aware_all_noise() {
        let noise = crate::NOISE;
        let data = vec![vec![0.0], vec![1.0], vec![2.0]];
        let labels = vec![noise, noise, noise];
        let score = silhouette_score_noise_aware(&data, &labels, &Euclidean);
        assert!(score.abs() < 0.01, "all noise should give ~0");
    }

    #[test]
    fn ari_perfect_agreement() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let b = vec![0, 0, 1, 1, 2, 2];
        let ari = adjusted_rand_index(&a, &b);
        assert!(
            (ari - 1.0).abs() < 0.01,
            "perfect agreement should give ARI=1, got {ari}"
        );
    }

    #[test]
    fn ari_permuted_labels() {
        // Same structure, different label numbering.
        let a = vec![0, 0, 1, 1, 2, 2];
        let b = vec![2, 2, 0, 0, 1, 1];
        let ari = adjusted_rand_index(&a, &b);
        assert!(
            (ari - 1.0).abs() < 0.01,
            "permuted labels should give ARI=1, got {ari}"
        );
    }

    #[test]
    fn ari_random_is_near_zero() {
        // All same vs all different: low agreement.
        let a = vec![0, 0, 0, 0, 0, 0];
        let b = vec![0, 1, 2, 3, 4, 5];
        let ari = adjusted_rand_index(&a, &b);
        assert!(
            ari.abs() < 0.5,
            "random-ish should give ARI near 0, got {ari}"
        );
    }

    #[test]
    fn k_distance_sorted() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.2, 0.0],
            vec![10.0, 0.0],
        ];
        let kd = k_distance(&data, 1, &Euclidean);
        assert_eq!(kd.len(), 4);
        // Must be sorted ascending.
        for w in kd.windows(2) {
            assert!(w[0] <= w[1], "k-distance must be sorted");
        }
        // First value should be small (nearest neighbor of closest pair).
        assert!(kd[0] < 0.2);
        // Last value should be large (the outlier at 10.0).
        assert!(kd[3] > 5.0);
    }

    #[test]
    fn noise_ratio_basic() {
        let noise = crate::NOISE;
        assert!((noise_ratio(&[0, 1, noise]) - 1.0 / 3.0).abs() < 0.01);
        assert!((noise_ratio(&[0, 0, 0]) - 0.0).abs() < 0.01);
        assert!((noise_ratio(&[noise, noise]) - 1.0).abs() < 0.01);
    }

    #[test]
    fn sampled_silhouette_close_to_exact() {
        // Two well-separated clusters: sampled score should approximate exact.
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for i in 0..50 {
            data.push(vec![i as f32 * 0.1, 0.0]);
            labels.push(0);
        }
        for i in 0..50 {
            data.push(vec![100.0 + i as f32 * 0.1, 0.0]);
            labels.push(1);
        }
        let exact = silhouette_score(&data, &labels, &Euclidean);
        let sampled = silhouette_score_sampled(&data, &labels, &Euclidean, 40, 42);
        assert!(
            (exact - sampled).abs() < 0.15,
            "sampled {sampled} too far from exact {exact}"
        );
        assert!(
            sampled > 0.8,
            "well-separated clusters should score high, got {sampled}"
        );
    }

    #[test]
    fn sampled_silhouette_fallback_when_small() {
        // n <= sample_size: should return exact score.
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let exact = silhouette_score(&data, &labels, &Euclidean);
        let sampled = silhouette_score_sampled(&data, &labels, &Euclidean, 100, 42);
        assert!(
            (exact - sampled).abs() < 1e-6,
            "fallback should be exact: {sampled} vs {exact}"
        );
    }

    #[test]
    fn disco_perfect_clusters() {
        // Two well-separated clusters, no noise -> high score.
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.2, 0.0],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
            vec![10.2, 0.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let score = disco_score(&data, &labels, &Euclidean, 2);
        assert!(
            score > 0.8,
            "perfect clusters should have DISCO > 0.8, got {score}"
        );
    }

    #[test]
    fn disco_with_correct_noise() {
        // Two clusters + distant outlier labeled NOISE -> high score.
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.2, 0.0],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
            vec![10.2, 0.0],
            vec![50.0, 50.0], // distant outlier
        ];
        let noise = crate::NOISE;
        let labels = vec![0, 0, 0, 1, 1, 1, noise];
        let score = disco_score(&data, &labels, &Euclidean, 2);
        assert!(
            score > 0.5,
            "correct noise should yield high DISCO, got {score}"
        );
    }

    #[test]
    fn disco_with_wrong_noise() {
        // Two clusters, but a core cluster member is labeled NOISE -> lower score.
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.2, 0.0],
            vec![10.0, 0.0],
            vec![10.1, 0.0],
            vec![10.2, 0.0],
        ];
        let noise = crate::NOISE;
        // Correct labels for reference.
        let correct_labels = vec![0, 0, 0, 1, 1, 1];
        let correct_score = disco_score(&data, &correct_labels, &Euclidean, 2);
        // Mislabel a cluster member as noise.
        let wrong_labels = vec![0, 0, noise, 1, 1, 1];
        let wrong_score = disco_score(&data, &wrong_labels, &Euclidean, 2);
        assert!(
            wrong_score < correct_score,
            "wrong noise ({wrong_score}) should score lower than correct ({correct_score})"
        );
    }

    #[test]
    fn disco_all_noise() {
        let noise = crate::NOISE;
        let data = vec![vec![0.0], vec![1.0], vec![2.0]];
        let labels = vec![noise, noise, noise];
        let score = disco_score(&data, &labels, &Euclidean, 2);
        assert!(
            score.abs() < 0.01,
            "all noise should give DISCO ~0, got {score}"
        );
    }

    #[test]
    fn disco_single_cluster() {
        let data = vec![vec![0.0], vec![1.0], vec![2.0]];
        let labels = vec![0, 0, 0];
        let score = disco_score(&data, &labels, &Euclidean, 2);
        assert!(
            score.abs() < 0.01,
            "single cluster should give DISCO ~0, got {score}"
        );
    }
}
