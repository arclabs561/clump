"""Type stubs for clumppy -- Python bindings to the clump Rust crate."""

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
