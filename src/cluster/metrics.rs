//! Cluster evaluation metrics.
//!
//! Functions for evaluating clustering quality without ground-truth labels.
//!
//! - [`silhouette_score`]: how well each point fits its cluster vs neighbors (-1 to 1)
//! - [`calinski_harabasz`]: ratio of between-cluster to within-cluster variance (higher = better)
//! - [`davies_bouldin`]: average worst-case cluster similarity (lower = better)

use super::distance::DistanceMetric;

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
pub fn silhouette_score<D: DistanceMetric>(data: &[Vec<f32>], labels: &[usize], metric: &D) -> f32 {
    let n = data.len();
    if n <= 1 {
        return 0.0;
    }

    // Find the number of clusters.
    let k = labels.iter().copied().max().unwrap_or(0) + 1;
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

    for i in 0..n {
        let ci = labels[i];
        if cluster_sizes[ci] <= 1 {
            // Singleton cluster: silhouette undefined, treat as 0.
            continue;
        }

        // Accumulate distances from point i to each cluster.
        let mut cluster_dist_sum = vec![0.0f64; k];
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = metric.distance(&data[i], &data[j]) as f64;
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
    }

    (total_silhouette / n as f64) as f32
}

/// Calinski-Harabasz index (variance ratio criterion).
///
/// Ratio of between-cluster dispersion to within-cluster dispersion,
/// adjusted by degrees of freedom. Higher = better.
///
/// Complexity: O(n * k * d). Only needs centroids + one data pass.
pub fn calinski_harabasz(data: &[Vec<f32>], labels: &[usize], centroids: &[Vec<f32>]) -> f32 {
    let n = data.len();
    let k = centroids.len();
    if k <= 1 || n <= k {
        return 0.0;
    }

    let d = data[0].len();

    // Global centroid.
    let mut global = vec![0.0f64; d];
    for point in data {
        for (j, &x) in point.iter().enumerate() {
            global[j] += x as f64;
        }
    }
    for x in &mut global {
        *x /= n as f64;
    }

    // Cluster sizes.
    let mut sizes = vec![0usize; k];
    for &l in labels {
        sizes[l] += 1;
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
    let mut within = 0.0f64;
    for (i, point) in data.iter().enumerate() {
        let centroid = &centroids[labels[i]];
        let sq_dist: f64 = point
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
    data: &[Vec<f32>],
    labels: &[usize],
    centroids: &[Vec<f32>],
    metric: &D,
) -> f32 {
    let k = centroids.len();
    if k <= 1 {
        return 0.0;
    }

    // Mean intra-cluster distance per cluster.
    let mut intra_sum = vec![0.0f64; k];
    let mut sizes = vec![0usize; k];
    for (i, point) in data.iter().enumerate() {
        let ci = labels[i];
        let d = metric.distance(point, &centroids[ci]) as f64;
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
}
