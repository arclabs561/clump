//! Python bindings for clump -- dense clustering primitives.
//!
//! Exposes constrained clustering (COP-Kmeans), correlation clustering,
//! k-means, DBSCAN, HDBSCAN, and evaluation metrics as a flat Python API.
//! All point-data functions accept either a 2D numpy array or a list of lists.
//! Clustering labels are returned as numpy int64 arrays (noise = -1).

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use clump::cluster::distance::Euclidean;
use clump::cluster::metrics;
use clump::{
    Constraint, CopKmeans, CorrelationClustering, CorrelationResult, Dbscan, Hdbscan, Kmeans,
    KmeansFit, MiniBatchKmeans, Optics, OpticsResult, SignedEdge, NOISE,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Accept either a 2D numpy array (f64 or f32) or list[list[float]].
/// Returns Vec<Vec<f32>> for internal use.
fn extract_data(data: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    // Try numpy f64 array first (most common from Python).
    if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
        let view = arr.as_array();
        return Ok(view
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect());
    }
    // Try numpy f32 array (zero-copy friendly).
    if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
        let view = arr.as_array();
        return Ok(view
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect());
    }
    // Fall back to list[list[float]].
    let lists: Vec<Vec<f64>> = data.extract()?;
    Ok(lists
        .into_iter()
        .map(|row| row.into_iter().map(|v| v as f32).collect())
        .collect())
}

/// Accept either a 1D numpy int64 array or list[int] for label input.
/// Converts -1 back to NOISE sentinel.
fn extract_labels(labels: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(arr) = labels.extract::<PyReadonlyArray1<i64>>() {
        let view = arr.as_array();
        return Ok(view
            .iter()
            .map(|&v| if v < 0 { NOISE } else { v as usize })
            .collect());
    }
    // Fall back to list[int].
    let list: Vec<i64> = labels.extract()?;
    Ok(list
        .into_iter()
        .map(|v| if v < 0 { NOISE } else { v as usize })
        .collect())
}

/// Convert label Vec<usize> to a numpy int64 array, mapping NOISE -> -1.
fn labels_to_numpy<'py>(py: Python<'py>, labels: Vec<usize>) -> Bound<'py, PyArray1<i64>> {
    let mapped: Vec<i64> = labels
        .into_iter()
        .map(|l| if l == NOISE { -1 } else { l as i64 })
        .collect();
    mapped.into_pyarray(py)
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
///     numpy.ndarray of cluster labels (int64, one per node).
#[pyfunction]
#[pyo3(signature = (edges, n_nodes, seed=None))]
fn correlation_clustering<'py>(
    py: Python<'py>,
    edges: Vec<(usize, usize, f64)>,
    n_nodes: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let signed_edges: Vec<SignedEdge> = edges
        .into_iter()
        .map(|(i, j, w)| SignedEdge {
            i,
            j,
            weight: w as f32,
        })
        .collect();

    let result: CorrelationResult = CorrelationClustering::new()
        .with_seed(seed.unwrap_or(42))
        .fit(n_nodes, &signed_edges)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(labels_to_numpy(py, result.labels))
}

// ---------------------------------------------------------------------------
// COP-Kmeans (the gap -- constrained clustering)
// ---------------------------------------------------------------------------

/// Constrained k-means clustering (Wagstaff et al., 2001).
///
/// Standard k-means with must-link and cannot-link pairwise constraints.
///
/// Args:
///     data: Points as (n, d) numpy array or list of lists.
///     k: Number of clusters.
///     must_link: List of (i, j) pairs that must be in the same cluster.
///     cannot_link: List of (i, j) pairs that must be in different clusters.
///     max_iter: Maximum iterations (default 300).
///     seed: Random seed for reproducibility (default None).
///
/// Returns:
///     numpy.ndarray of cluster labels (int64).
#[pyfunction]
#[pyo3(signature = (data, k, must_link, cannot_link, max_iter = 300, seed = None))]
fn cop_kmeans<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    k: usize,
    must_link: Vec<(usize, usize)>,
    cannot_link: Vec<(usize, usize)>,
    max_iter: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let vecs = extract_data(data)?;

    let mut constraints: Vec<Constraint> =
        Vec::with_capacity(must_link.len() + cannot_link.len());
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

    Ok(labels_to_numpy(py, labels))
}

// ---------------------------------------------------------------------------
// K-means
// ---------------------------------------------------------------------------

/// K-means clustering.
///
/// Args:
///     data: Points as (n, d) numpy array or list of lists.
///     k: Number of clusters.
///     max_iter: Maximum iterations (default 300).
///     seed: Random seed for reproducibility (default None).
///
/// Returns:
///     Tuple of (labels, centroids) as numpy arrays.
///     labels: int64 array of shape (n,).
///     centroids: float64 array of shape (k, d).
#[pyfunction]
#[pyo3(signature = (data, k, max_iter = 300, seed = None))]
fn kmeans<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    k: usize,
    max_iter: usize,
    seed: Option<u64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>)> {
    let vecs = extract_data(data)?;

    let mut builder = Kmeans::new(k).with_max_iter(max_iter);
    if let Some(s) = seed {
        builder = builder.with_seed(s);
    }

    let fit = builder
        .fit(&vecs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let centroids = centroids_to_numpy(py, &fit.centroids)?;
    Ok((labels_to_numpy(py, fit.labels), centroids))
}

// ---------------------------------------------------------------------------
// DBSCAN
// ---------------------------------------------------------------------------

/// DBSCAN density-based clustering.
///
/// Args:
///     data: Points as (n, d) numpy array or list of lists.
///     eps: Maximum distance for neighborhood.
///     min_points: Minimum neighbors to form a core point.
///
/// Returns:
///     numpy.ndarray of cluster labels (int64). Noise points are labeled -1.
#[pyfunction]
#[pyo3(signature = (data, eps, min_points))]
fn dbscan<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    eps: f64,
    min_points: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let vecs = extract_data(data)?;

    let labels = Dbscan::new(eps as f32, min_points)
        .fit_predict(&vecs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(labels_to_numpy(py, labels))
}

// ---------------------------------------------------------------------------
// HDBSCAN
// ---------------------------------------------------------------------------

/// HDBSCAN hierarchical density-based clustering.
///
/// Args:
///     data: Points as (n, d) numpy array or list of lists.
///     min_cluster_size: Minimum points for a cluster to persist.
///     min_points: k for core distance computation (default: min_cluster_size).
///
/// Returns:
///     numpy.ndarray of cluster labels (int64). Noise points are labeled -1.
#[pyfunction]
#[pyo3(signature = (data, min_cluster_size, min_points = None))]
fn hdbscan<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    min_cluster_size: usize,
    min_points: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let vecs = extract_data(data)?;

    let min_samples = min_points.unwrap_or(min_cluster_size);

    let labels = Hdbscan::new()
        .with_min_cluster_size(min_cluster_size)
        .with_min_samples(min_samples)
        .fit_predict(&vecs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(labels_to_numpy(py, labels))
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Silhouette score: mean silhouette coefficient across all points.
///
/// Range: [-1, 1]. Higher is better.
///
/// Args:
///     data: Points as (n, d) numpy array or list of lists.
///     labels: Cluster labels as numpy int64 array or list of ints.
///
/// Returns:
///     Mean silhouette score.
#[pyfunction]
#[pyo3(signature = (data, labels))]
fn silhouette_score(data: &Bound<'_, PyAny>, labels: &Bound<'_, PyAny>) -> PyResult<f64> {
    let vecs = extract_data(data)?;
    let labs = extract_labels(labels)?;
    let score = metrics::silhouette_score(&vecs, &labs, &Euclidean);
    Ok(score as f64)
}

/// Calinski-Harabasz index (variance ratio criterion). Higher is better.
///
/// Args:
///     data: Points as (n, d) numpy array or list of lists.
///     labels: Cluster labels as numpy int64 array or list of ints.
///
/// Returns:
///     Calinski-Harabasz score.
#[pyfunction]
#[pyo3(signature = (data, labels))]
fn calinski_harabasz_score(
    data: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let vecs = extract_data(data)?;
    let labs = extract_labels(labels)?;

    // Compute centroids from labels (filter NOISE = usize::MAX).
    let k = labs.iter().copied()
        .filter(|&l| l != NOISE)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let d = vecs.first().map(|r| r.len()).unwrap_or(0);
    let mut sums = vec![vec![0.0f64; d]; k];
    let mut counts = vec![0usize; k];
    for (i, &l) in labs.iter().enumerate() {
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

    let score = metrics::calinski_harabasz(&vecs, &labs, &centroids);
    Ok(score as f64)
}

/// Davies-Bouldin index. Lower is better.
///
/// Args:
///     data: Points as (n, d) numpy array or list of lists.
///     labels: Cluster labels as numpy int64 array or list of ints.
///
/// Returns:
///     Davies-Bouldin score.
#[pyfunction]
#[pyo3(signature = (data, labels))]
fn davies_bouldin_score(data: &Bound<'_, PyAny>, labels: &Bound<'_, PyAny>) -> PyResult<f64> {
    let vecs = extract_data(data)?;
    let labs = extract_labels(labels)?;

    // Compute centroids from labels (filter NOISE = usize::MAX).
    let k = labs.iter().copied()
        .filter(|&l| l != NOISE)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let d = vecs.first().map(|r| r.len()).unwrap_or(0);
    let mut sums = vec![vec![0.0f64; d]; k];
    let mut counts = vec![0usize; k];
    for (i, &l) in labs.iter().enumerate() {
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

    let score = metrics::davies_bouldin(&vecs, &labs, &centroids, &Euclidean);
    Ok(score as f64)
}

// ---------------------------------------------------------------------------
// KmeansModel -- fitted k-means with predict
// ---------------------------------------------------------------------------

/// Fitted k-means model. Wraps the Rust KmeansFit struct and exposes
/// labels, centroids, inertia, and prediction on new data.
#[pyclass]
struct KmeansModel {
    fit: KmeansFit,
    /// Original training data (needed for wcss computation).
    training_data: Vec<Vec<f32>>,
}

#[pymethods]
impl KmeansModel {
    /// Cluster labels for the training data (int64 numpy array).
    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        labels_to_numpy(py, self.fit.labels.clone())
    }

    /// Learned centroids as (k, d) float64 numpy array.
    #[getter]
    fn centroids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        centroids_to_numpy(py, &self.fit.centroids)
    }

    /// Final WCSS (within-cluster sum of squares / inertia).
    #[getter]
    fn inertia(&self) -> f64 {
        self.fit
            .inertia_trace
            .last()
            .copied()
            .unwrap_or(self.fit.wcss(&self.training_data)) as f64
    }

    /// Number of Lloyd iterations executed.
    #[getter]
    fn n_iter(&self) -> usize {
        self.fit.iters
    }

    /// Assign new points to the nearest learned centroid.
    ///
    /// Args:
    ///     data: Points as (n, d) numpy array or list of lists.
    ///
    /// Returns:
    ///     numpy.ndarray of cluster labels (int64).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let vecs = extract_data(data)?;
        let labels = self
            .fit
            .predict(&vecs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(labels_to_numpy(py, labels))
    }
}

/// Fit k-means and return a model object with predict capability.
///
/// Args:
///     data: Points as (n, d) numpy array or list of lists.
///     k: Number of clusters.
///     max_iter: Maximum iterations (default 300).
///     seed: Random seed for reproducibility (default None).
///
/// Returns:
///     KmeansModel with .labels, .centroids, .inertia, .predict(data).
#[pyfunction]
#[pyo3(signature = (data, k, max_iter = 300, seed = None))]
fn kmeans_fit<'py>(
    _py: Python<'py>,
    data: &Bound<'py, PyAny>,
    k: usize,
    max_iter: usize,
    seed: Option<u64>,
) -> PyResult<KmeansModel> {
    let vecs = extract_data(data)?;

    let mut builder = Kmeans::new(k).with_max_iter(max_iter);
    if let Some(s) = seed {
        builder = builder.with_seed(s);
    }

    let fit = builder
        .fit(&vecs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(KmeansModel {
        fit,
        training_data: vecs,
    })
}

// ---------------------------------------------------------------------------
// MiniBatchKmeans -- streaming clustering
// ---------------------------------------------------------------------------

/// Mini-Batch K-means for streaming / incremental clustering.
///
/// Construct with `MiniBatchKmeans(k, seed=None)`, then call
/// `partial_fit(batch)` to update incrementally.
#[pyclass(name = "MiniBatchKmeans")]
struct PyMiniBatchKmeans {
    inner: MiniBatchKmeans,
}

#[pymethods]
impl PyMiniBatchKmeans {
    /// Create a new Mini-Batch K-means clusterer.
    ///
    /// Args:
    ///     k: Number of clusters.
    ///     seed: Random seed for reproducibility (default None).
    #[new]
    #[pyo3(signature = (k, seed = None))]
    fn new(k: usize, seed: Option<u64>) -> Self {
        let mut mbk = MiniBatchKmeans::new(k);
        if let Some(s) = seed {
            mbk = mbk.with_seed(s);
        }
        Self { inner: mbk }
    }

    /// Update the model with a batch of points.
    ///
    /// The first call initializes centroids via k-means++.
    /// Subsequent calls refine centroids incrementally.
    ///
    /// Args:
    ///     data: Points as (n, d) numpy array or list of lists.
    ///
    /// Returns:
    ///     numpy.ndarray of cluster labels for this batch (int64).
    fn partial_fit<'py>(
        &mut self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let vecs = extract_data(data)?;
        let labels = self
            .inner
            .update_batch(&vecs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(labels_to_numpy(py, labels))
    }

    /// Assign new points to the nearest centroid without updating the model.
    ///
    /// Args:
    ///     data: Points as (n, d) numpy array or list of lists.
    ///
    /// Returns:
    ///     numpy.ndarray of cluster labels (int64).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let vecs = extract_data(data)?;
        let labels = self
            .inner
            .predict(&vecs)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(labels_to_numpy(py, labels))
    }

    /// Current centroids as (k, d) float64 numpy array.
    #[getter]
    fn centroids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        centroids_to_numpy(py, self.inner.centroids())
    }

    /// Number of clusters.
    #[getter]
    fn n_clusters(&self) -> usize {
        self.inner.n_clusters()
    }
}

// ---------------------------------------------------------------------------
// OPTICS
// ---------------------------------------------------------------------------

/// Result of OPTICS clustering. Holds the reachability ordering and
/// supports cluster extraction at different thresholds.
#[pyclass(name = "OpticsResult")]
struct PyOpticsResult {
    inner: OpticsResult,
}

#[pymethods]
impl PyOpticsResult {
    /// Processing order (indices into original data) as int64 numpy array.
    #[getter]
    fn ordering<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let v: Vec<i64> = self.inner.ordering.iter().map(|&i| i as i64).collect();
        v.into_pyarray(py)
    }

    /// Reachability distances in ordering order as float64 numpy array.
    #[getter]
    fn reachability<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let v: Vec<f64> = self.inner.reachability.iter().map(|&r| r as f64).collect();
        v.into_pyarray(py)
    }

    /// Core distances in ordering order as float64 numpy array.
    #[getter]
    fn core_distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let v: Vec<f64> = self.inner.core_distances.iter().map(|&d| d as f64).collect();
        v.into_pyarray(py)
    }

    /// Extract DBSCAN-like clusters at a given epsilon threshold.
    ///
    /// Args:
    ///     epsilon: Reachability threshold for cluster extraction.
    ///
    /// Returns:
    ///     numpy.ndarray of cluster labels (int64). Noise = -1.
    #[pyo3(signature = (epsilon,))]
    fn extract_clusters<'py>(
        &self,
        py: Python<'py>,
        epsilon: f64,
    ) -> Bound<'py, PyArray1<i64>> {
        let labels = Optics::<Euclidean>::extract_clusters(&self.inner, epsilon as f32);
        labels_to_numpy(py, labels)
    }

    /// Extract clusters using the Xi method (automatic valley detection).
    ///
    /// Args:
    ///     xi: Steepness parameter in (0, 1). Smaller = fewer, tighter clusters.
    ///
    /// Returns:
    ///     numpy.ndarray of cluster labels (int64). Noise = -1.
    #[pyo3(signature = (xi,))]
    fn extract_xi<'py>(
        &self,
        py: Python<'py>,
        xi: f64,
    ) -> Bound<'py, PyArray1<i64>> {
        let labels = Optics::<Euclidean>::extract_xi(&self.inner, xi as f32);
        labels_to_numpy(py, labels)
    }
}

/// OPTICS: Ordering Points To Identify the Clustering Structure.
///
/// Produces a reachability ordering that can be cut at any threshold to
/// extract clusters. Unlike DBSCAN, does not require a fixed epsilon.
///
/// Args:
///     data: Points as (n, d) numpy array or list of lists.
///     max_epsilon: Maximum neighborhood radius.
///     min_points: Minimum neighbors to form a core point.
///
/// Returns:
///     OpticsResult with .ordering, .reachability, .core_distances,
///     .extract_clusters(eps), .extract_xi(xi).
#[pyfunction]
#[pyo3(signature = (data, max_epsilon, min_points))]
fn optics<'py>(
    _py: Python<'py>,
    data: &Bound<'py, PyAny>,
    max_epsilon: f64,
    min_points: usize,
) -> PyResult<PyOpticsResult> {
    let vecs = extract_data(data)?;

    let result = Optics::new(max_epsilon as f32, min_points)
        .fit(&vecs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(PyOpticsResult { inner: result })
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Convert centroids (Vec<Vec<f32>>) to a (k, d) float64 numpy array.
fn centroids_to_numpy<'py>(
    py: Python<'py>,
    centroids: &[Vec<f32>],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n_centroids = centroids.len();
    let d = centroids.first().map(|c| c.len()).unwrap_or(0);
    let flat: Vec<f64> = centroids
        .iter()
        .flat_map(|c| c.iter().map(|&v| v as f64))
        .collect();
    numpy::PyArray1::from_vec(py, flat).reshape([n_centroids, d])
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn clumppy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Unique algorithms (the gaps)
    m.add_function(wrap_pyfunction!(correlation_clustering, m)?)?;
    m.add_function(wrap_pyfunction!(cop_kmeans, m)?)?;

    // Standard algorithms
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans_fit, m)?)?;
    m.add_function(wrap_pyfunction!(dbscan, m)?)?;
    m.add_function(wrap_pyfunction!(hdbscan, m)?)?;
    m.add_function(wrap_pyfunction!(optics, m)?)?;

    // Classes
    m.add_class::<KmeansModel>()?;
    m.add_class::<PyMiniBatchKmeans>()?;
    m.add_class::<PyOpticsResult>()?;

    // Evaluation metrics
    m.add_function(wrap_pyfunction!(silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(calinski_harabasz_score, m)?)?;
    m.add_function(wrap_pyfunction!(davies_bouldin_score, m)?)?;

    Ok(())
}
