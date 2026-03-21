//! Python bindings for clump -- dense clustering primitives.
//!
//! Exposes constrained clustering (COP-Kmeans), correlation clustering,
//! k-means, DBSCAN, HDBSCAN, and evaluation metrics as a flat Python API.

use pyo3::prelude::*;

use clump::{
    CorrelationClustering, CorrelationResult, CopKmeans, Constraint, SignedEdge,
    Dbscan, Hdbscan, Kmeans, NOISE,
};
use clump::cluster::distance::Euclidean;
use clump::cluster::metrics;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert Python list[list[float]] -> Vec<Vec<f32>>.
fn to_vecs(data: Vec<Vec<f64>>) -> Vec<Vec<f32>> {
    data.into_iter()
        .map(|row| row.into_iter().map(|v| v as f32).collect())
        .collect()
}

/// Map clump NOISE sentinel (usize::MAX) to -1 for Python convention.
fn map_noise(labels: Vec<usize>) -> Vec<i64> {
    labels
        .into_iter()
        .map(|l| if l == NOISE { -1 } else { l as i64 })
        .collect()
}

// ---------------------------------------------------------------------------
// Correlation clustering (the gap -- no Python package does this well)
// ---------------------------------------------------------------------------

/// Cluster items from signed pairwise edges.
///
/// Each edge is `(i, j, weight)` where positive weight means "should be
/// together" and negative means "should be apart." The number of clusters
/// is determined automatically.
///
/// Args:
///     edges: List of (i, j, weight) tuples.
///     n_nodes: Total number of nodes.
///
/// Returns:
///     List of cluster labels (one per node).
#[pyfunction]
#[pyo3(signature = (edges, n_nodes))]
fn correlation_clustering(
    edges: Vec<(usize, usize, f64)>,
    n_nodes: usize,
) -> PyResult<Vec<usize>> {
    let signed_edges: Vec<SignedEdge> = edges
        .into_iter()
        .map(|(i, j, w)| SignedEdge {
            i,
            j,
            weight: w as f32,
        })
        .collect();

    let result: CorrelationResult = CorrelationClustering::new()
        .with_seed(42)
        .fit(n_nodes, &signed_edges)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.labels)
}

// ---------------------------------------------------------------------------
// COP-Kmeans (the gap -- constrained clustering)
// ---------------------------------------------------------------------------

/// Constrained k-means clustering (Wagstaff et al., 2001).
///
/// Standard k-means with must-link and cannot-link pairwise constraints.
///
/// Args:
///     data: List of points, each a list of floats.
///     k: Number of clusters.
///     must_link: List of (i, j) pairs that must be in the same cluster.
///     cannot_link: List of (i, j) pairs that must be in different clusters.
///     max_iter: Maximum iterations (default 300).
///     seed: Random seed for reproducibility (default None).
///
/// Returns:
///     List of cluster labels.
#[pyfunction]
#[pyo3(signature = (data, k, must_link, cannot_link, max_iter = 300, seed = None))]
fn cop_kmeans(
    data: Vec<Vec<f64>>,
    k: usize,
    must_link: Vec<(usize, usize)>,
    cannot_link: Vec<(usize, usize)>,
    max_iter: usize,
    seed: Option<u64>,
) -> PyResult<Vec<usize>> {
    let vecs = to_vecs(data);

    let mut constraints: Vec<Constraint> = Vec::with_capacity(must_link.len() + cannot_link.len());
    for (a, b) in must_link {
        constraints.push(Constraint::MustLink(a, b));
    }
    for (a, b) in cannot_link {
        constraints.push(Constraint::CannotLink(a, b));
    }

    let mut builder = CopKmeans::new(k).with_max_iter(max_iter);
    if let Some(s) = seed {
        builder = builder.with_seed(s);
    }

    let labels = builder
        .fit_predict_constrained(&vecs, &constraints)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(labels)
}

// ---------------------------------------------------------------------------
// K-means
// ---------------------------------------------------------------------------

/// K-means clustering.
///
/// Args:
///     data: List of points, each a list of floats.
///     k: Number of clusters.
///     max_iter: Maximum iterations (default 300).
///     seed: Random seed for reproducibility (default None).
///
/// Returns:
///     Tuple of (labels, centroids).
#[pyfunction]
#[pyo3(signature = (data, k, max_iter = 300, seed = None))]
fn kmeans(
    data: Vec<Vec<f64>>,
    k: usize,
    max_iter: usize,
    seed: Option<u64>,
) -> PyResult<(Vec<usize>, Vec<Vec<f64>>)> {
    let vecs = to_vecs(data);

    let mut builder = Kmeans::new(k).with_max_iter(max_iter);
    if let Some(s) = seed {
        builder = builder.with_seed(s);
    }

    let fit = builder
        .fit(&vecs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let centroids: Vec<Vec<f64>> = fit
        .centroids
        .iter()
        .map(|c| c.iter().map(|&v| v as f64).collect())
        .collect();

    Ok((fit.labels, centroids))
}

// ---------------------------------------------------------------------------
// DBSCAN
// ---------------------------------------------------------------------------

/// DBSCAN density-based clustering.
///
/// Args:
///     data: List of points, each a list of floats.
///     eps: Maximum distance for neighborhood.
///     min_points: Minimum neighbors to form a core point.
///
/// Returns:
///     List of cluster labels. Noise points are labeled -1.
#[pyfunction]
#[pyo3(signature = (data, eps, min_points))]
fn dbscan(data: Vec<Vec<f64>>, eps: f64, min_points: usize) -> PyResult<Vec<i64>> {
    let vecs = to_vecs(data);

    let labels = Dbscan::new(eps as f32, min_points)
        .fit_predict(&vecs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(map_noise(labels))
}

// ---------------------------------------------------------------------------
// HDBSCAN
// ---------------------------------------------------------------------------

/// HDBSCAN hierarchical density-based clustering.
///
/// Args:
///     data: List of points, each a list of floats.
///     min_cluster_size: Minimum points for a cluster to persist.
///     min_points: k for core distance computation (default: min_cluster_size).
///
/// Returns:
///     List of cluster labels. Noise points are labeled -1.
#[pyfunction]
#[pyo3(signature = (data, min_cluster_size, min_points = None))]
fn hdbscan(
    data: Vec<Vec<f64>>,
    min_cluster_size: usize,
    min_points: Option<usize>,
) -> PyResult<Vec<i64>> {
    let vecs = to_vecs(data);

    let min_samples = min_points.unwrap_or(min_cluster_size);

    let labels = Hdbscan::new()
        .with_min_cluster_size(min_cluster_size)
        .with_min_samples(min_samples)
        .fit_predict(&vecs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(map_noise(labels))
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Silhouette score: mean silhouette coefficient across all points.
///
/// Range: [-1, 1]. Higher is better.
///
/// Args:
///     data: List of points, each a list of floats.
///     labels: Cluster labels (non-negative integers).
///
/// Returns:
///     Mean silhouette score.
#[pyfunction]
#[pyo3(signature = (data, labels))]
fn silhouette_score(data: Vec<Vec<f64>>, labels: Vec<usize>) -> PyResult<f64> {
    let vecs = to_vecs(data);
    let score = metrics::silhouette_score(&vecs, &labels, &Euclidean);
    Ok(score as f64)
}

/// Calinski-Harabasz index (variance ratio criterion). Higher is better.
///
/// Args:
///     data: List of points, each a list of floats.
///     labels: Cluster labels (non-negative integers).
///
/// Returns:
///     Calinski-Harabasz score.
#[pyfunction]
#[pyo3(signature = (data, labels))]
fn calinski_harabasz_score(data: Vec<Vec<f64>>, labels: Vec<usize>) -> PyResult<f64> {
    let vecs = to_vecs(data);

    // Compute centroids from labels.
    let k = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let d = vecs.first().map(|r| r.len()).unwrap_or(0);
    let mut sums = vec![vec![0.0f64; d]; k];
    let mut counts = vec![0usize; k];
    for (i, &l) in labels.iter().enumerate() {
        if l < k {
            counts[l] += 1;
            for (j, &v) in vecs[i].iter().enumerate() {
                sums[l][j] += v as f64;
            }
        }
    }
    let centroids: Vec<Vec<f32>> = sums
        .iter()
        .zip(counts.iter())
        .map(|(s, &c)| {
            if c > 0 {
                s.iter().map(|&v| (v / c as f64) as f32).collect()
            } else {
                vec![0.0f32; d]
            }
        })
        .collect();

    let score = metrics::calinski_harabasz(&vecs, &labels, &centroids);
    Ok(score as f64)
}

/// Davies-Bouldin index. Lower is better.
///
/// Args:
///     data: List of points, each a list of floats.
///     labels: Cluster labels (non-negative integers).
///
/// Returns:
///     Davies-Bouldin score.
#[pyfunction]
#[pyo3(signature = (data, labels))]
fn davies_bouldin_score(data: Vec<Vec<f64>>, labels: Vec<usize>) -> PyResult<f64> {
    let vecs = to_vecs(data);

    // Compute centroids from labels.
    let k = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let d = vecs.first().map(|r| r.len()).unwrap_or(0);
    let mut sums = vec![vec![0.0f64; d]; k];
    let mut counts = vec![0usize; k];
    for (i, &l) in labels.iter().enumerate() {
        if l < k {
            counts[l] += 1;
            for (j, &v) in vecs[i].iter().enumerate() {
                sums[l][j] += v as f64;
            }
        }
    }
    let centroids: Vec<Vec<f32>> = sums
        .iter()
        .zip(counts.iter())
        .map(|(s, &c)| {
            if c > 0 {
                s.iter().map(|&v| (v / c as f64) as f32).collect()
            } else {
                vec![0.0f32; d]
            }
        })
        .collect();

    let score = metrics::davies_bouldin(&vecs, &labels, &centroids, &Euclidean);
    Ok(score as f64)
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn clumppy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Unique algorithms (the gaps)
    m.add_function(wrap_pyfunction!(correlation_clustering, m)?)?;
    m.add_function(wrap_pyfunction!(cop_kmeans, m)?)?;

    // Standard algorithms
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(dbscan, m)?)?;
    m.add_function(wrap_pyfunction!(hdbscan, m)?)?;

    // Evaluation metrics
    m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(calinski_harabasz_score, m)?)?;
    m.add_function(wrap_pyfunction!(davies_bouldin_score, m)?)?;

    Ok(())
}
