"""Type stubs for clumppy -- Python bindings to the clump Rust crate.

Note: NaN values in numeric data inputs will raise a ValueError.
The Rust backend validates inputs and rejects NaN before processing.
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

__version__: str

def correlation_clustering(
    edges: list[tuple[int, int, float]],
    n_nodes: int,
    seed: Optional[int] = None,
) -> NDArray[np.int64]:
    """Cluster items from signed pairwise edges.

    Each edge is (i, j, weight) where positive weight means "should be
    together" and negative means "should be apart." The number of clusters
    is determined automatically.

    Args:
        edges: List of (i, j, weight) tuples.
        n_nodes: Total number of nodes.

    Returns:
        Cluster labels as int64 array (one per node).
    """
    ...

def cop_kmeans(
    data: Union[NDArray[np.floating], list[list[float]]],
    k: int,
    must_link: list[tuple[int, int]],
    cannot_link: list[tuple[int, int]],
    max_iter: int = 300,
    seed: Optional[int] = None,
) -> NDArray[np.int64]:
    """Constrained k-means clustering (Wagstaff et al., 2001).

    Standard k-means with must-link and cannot-link pairwise constraints.

    Args:
        data: Points as (n, d) array or list of lists.
        k: Number of clusters.
        must_link: List of (i, j) pairs that must be in the same cluster.
        cannot_link: List of (i, j) pairs that must be in different clusters.
        max_iter: Maximum iterations.
        seed: Random seed for reproducibility.

    Returns:
        Cluster labels as int64 array.
    """
    ...

def kmeans(
    data: Union[NDArray[np.floating], list[list[float]]],
    k: int,
    max_iter: int = 300,
    seed: Optional[int] = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """K-means clustering.

    Args:
        data: Points as (n, d) array or list of lists.
        k: Number of clusters.
        max_iter: Maximum iterations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (labels, centroids) as numpy arrays.
        labels: int64 array of shape (n,).
        centroids: float64 array of shape (k, d).
    """
    ...

def kmeans_fit(
    data: Union[NDArray[np.floating], list[list[float]]],
    k: int,
    max_iter: int = 300,
    seed: Optional[int] = None,
) -> "KmeansModel":
    """Fit k-means and return a model object with predict capability.

    Args:
        data: Points as (n, d) array or list of lists.
        k: Number of clusters.
        max_iter: Maximum iterations.
        seed: Random seed for reproducibility.

    Returns:
        KmeansModel with .labels, .centroids, .inertia, .predict(data).
    """
    ...

class KmeansModel:
    """Fitted k-means model with prediction on new data."""

    @property
    def labels(self) -> NDArray[np.int64]:
        """Cluster labels for the training data."""
        ...

    @property
    def centroids(self) -> NDArray[np.float64]:
        """Learned centroids as (k, d) array."""
        ...

    @property
    def inertia(self) -> float:
        """Final WCSS (within-cluster sum of squares)."""
        ...

    @property
    def n_iter(self) -> int:
        """Number of Lloyd iterations executed."""
        ...

    def predict(
        self,
        data: Union[NDArray[np.floating], list[list[float]]],
    ) -> NDArray[np.int64]:
        """Assign new points to the nearest learned centroid.

        Args:
            data: Points as (n, d) array or list of lists.

        Returns:
            Cluster labels as int64 array.
        """
        ...

class MiniBatchKmeans:
    """Mini-Batch K-means for streaming / incremental clustering."""

    def __init__(self, k: int, seed: Optional[int] = None) -> None:
        """Create a new Mini-Batch K-means clusterer.

        Args:
            k: Number of clusters.
            seed: Random seed for reproducibility.
        """
        ...

    def partial_fit(
        self,
        data: Union[NDArray[np.floating], list[list[float]]],
    ) -> NDArray[np.int64]:
        """Update the model with a batch of points.

        The first call initializes centroids via k-means++.
        Subsequent calls refine centroids incrementally.

        Args:
            data: Points as (n, d) array or list of lists.

        Returns:
            Cluster labels for this batch as int64 array.
        """
        ...

    def predict(
        self,
        data: Union[NDArray[np.floating], list[list[float]]],
    ) -> NDArray[np.int64]:
        """Assign points to nearest centroid without updating the model.

        Args:
            data: Points as (n, d) array or list of lists.

        Returns:
            Cluster labels as int64 array.
        """
        ...

    @property
    def centroids(self) -> NDArray[np.float64]:
        """Current centroids as (k, d) array."""
        ...

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        ...

def dbscan(
    data: Union[NDArray[np.floating], list[list[float]]],
    eps: float,
    min_points: int,
) -> NDArray[np.int64]:
    """DBSCAN density-based clustering.

    Args:
        data: Points as (n, d) array or list of lists.
        eps: Maximum distance for neighborhood.
        min_points: Minimum neighbors to form a core point.

    Returns:
        Cluster labels as int64 array. Noise points are labeled -1.
    """
    ...

def hdbscan(
    data: Union[NDArray[np.floating], list[list[float]]],
    min_cluster_size: int,
    min_points: Optional[int] = None,
) -> NDArray[np.int64]:
    """HDBSCAN hierarchical density-based clustering.

    Args:
        data: Points as (n, d) array or list of lists.
        min_cluster_size: Minimum points for a cluster to persist.
        min_points: k for core distance computation (default: min_cluster_size).

    Returns:
        Cluster labels as int64 array. Noise points are labeled -1.
    """
    ...

def optics(
    data: Union[NDArray[np.floating], list[list[float]]],
    max_epsilon: float,
    min_points: int,
) -> "OpticsResult":
    """OPTICS: Ordering Points To Identify the Clustering Structure.

    Produces a reachability ordering that can be cut at any threshold
    to extract clusters. Unlike DBSCAN, does not require a fixed epsilon.

    Args:
        data: Points as (n, d) array or list of lists.
        max_epsilon: Maximum neighborhood radius.
        min_points: Minimum neighbors to form a core point.

    Returns:
        OpticsResult with .ordering, .reachability, .core_distances,
        .extract_clusters(eps), .extract_xi(xi).
    """
    ...

class OpticsResult:
    """Result of OPTICS clustering with reachability ordering."""

    @property
    def ordering(self) -> NDArray[np.int64]:
        """Processing order (indices into original data)."""
        ...

    @property
    def reachability(self) -> NDArray[np.float64]:
        """Reachability distances in ordering order."""
        ...

    @property
    def core_distances(self) -> NDArray[np.float64]:
        """Core distances in ordering order."""
        ...

    def extract_clusters(self, epsilon: float) -> NDArray[np.int64]:
        """Extract DBSCAN-like clusters at a given epsilon threshold.

        Args:
            epsilon: Reachability threshold for cluster extraction.

        Returns:
            Cluster labels as int64 array. Noise = -1.
        """
        ...

    def extract_xi(self, xi: float) -> NDArray[np.int64]:
        """Extract clusters using the Xi method (automatic valley detection).

        Args:
            xi: Steepness parameter in (0, 1). Smaller = fewer, tighter clusters.

        Returns:
            Cluster labels as int64 array. Noise = -1.
        """
        ...

def silhouette_score(
    data: Union[NDArray[np.floating], list[list[float]]],
    labels: Union[NDArray[np.integer], list[int]],
) -> float:
    """Silhouette score: mean silhouette coefficient across all points.

    Range: [-1, 1]. Higher is better.

    Args:
        data: Points as (n, d) array or list of lists.
        labels: Cluster labels as numpy array or list of ints.

    Returns:
        Mean silhouette score.
    """
    ...

def calinski_harabasz_score(
    data: Union[NDArray[np.floating], list[list[float]]],
    labels: Union[NDArray[np.integer], list[int]],
) -> float:
    """Calinski-Harabasz index (variance ratio criterion). Higher is better.

    Args:
        data: Points as (n, d) array or list of lists.
        labels: Cluster labels as numpy array or list of ints.

    Returns:
        Calinski-Harabasz score.
    """
    ...

def davies_bouldin_score(
    data: Union[NDArray[np.floating], list[list[float]]],
    labels: Union[NDArray[np.integer], list[int]],
) -> float:
    """Davies-Bouldin index. Lower is better.

    Args:
        data: Points as (n, d) array or list of lists.
        labels: Cluster labels as numpy array or list of ints.

    Returns:
        Davies-Bouldin score.
    """
    ...
