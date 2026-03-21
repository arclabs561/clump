"""Type stubs for clumppy -- Python bindings to the clump Rust crate."""

def correlation_clustering(
    edges: list[tuple[int, int, float]],
    n_nodes: int,
) -> list[int]:
    """Cluster items from signed pairwise edges.

    Each edge is (i, j, weight) where positive weight means "should be
    together" and negative means "should be apart." The number of clusters
    is determined automatically.

    Args:
        edges: List of (i, j, weight) tuples.
        n_nodes: Total number of nodes.

    Returns:
        List of cluster labels (one per node).
    """
    ...

def cop_kmeans(
    data: list[list[float]],
    k: int,
    must_link: list[tuple[int, int]],
    cannot_link: list[tuple[int, int]],
    max_iter: int = 300,
    seed: int | None = None,
) -> list[int]:
    """Constrained k-means clustering (Wagstaff et al., 2001).

    Standard k-means with must-link and cannot-link pairwise constraints.

    Args:
        data: List of points, each a list of floats.
        k: Number of clusters.
        must_link: List of (i, j) pairs that must be in the same cluster.
        cannot_link: List of (i, j) pairs that must be in different clusters.
        max_iter: Maximum iterations.
        seed: Random seed for reproducibility.

    Returns:
        List of cluster labels.
    """
    ...

def kmeans(
    data: list[list[float]],
    k: int,
    max_iter: int = 300,
    seed: int | None = None,
) -> tuple[list[int], list[list[float]]]:
    """K-means clustering.

    Args:
        data: List of points, each a list of floats.
        k: Number of clusters.
        max_iter: Maximum iterations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (labels, centroids).
    """
    ...

def dbscan(
    data: list[list[float]],
    eps: float,
    min_points: int,
) -> list[int]:
    """DBSCAN density-based clustering.

    Args:
        data: List of points, each a list of floats.
        eps: Maximum distance for neighborhood.
        min_points: Minimum neighbors to form a core point.

    Returns:
        List of cluster labels. Noise points are labeled -1.
    """
    ...

def hdbscan(
    data: list[list[float]],
    min_cluster_size: int,
    min_points: int | None = None,
) -> list[int]:
    """HDBSCAN hierarchical density-based clustering.

    Args:
        data: List of points, each a list of floats.
        min_cluster_size: Minimum points for a cluster to persist.
        min_points: k for core distance computation (default: min_cluster_size).

    Returns:
        List of cluster labels. Noise points are labeled -1.
    """
    ...

def silhouette_score(
    data: list[list[float]],
    labels: list[int],
) -> float:
    """Silhouette score: mean silhouette coefficient across all points.

    Range: [-1, 1]. Higher is better.

    Args:
        data: List of points, each a list of floats.
        labels: Cluster labels (non-negative integers).

    Returns:
        Mean silhouette score.
    """
    ...

def calinski_harabasz_score(
    data: list[list[float]],
    labels: list[int],
) -> float:
    """Calinski-Harabasz index (variance ratio criterion). Higher is better.

    Args:
        data: List of points, each a list of floats.
        labels: Cluster labels (non-negative integers).

    Returns:
        Calinski-Harabasz score.
    """
    ...

def davies_bouldin_score(
    data: list[list[float]],
    labels: list[int],
) -> float:
    """Davies-Bouldin index. Lower is better.

    Args:
        data: List of points, each a list of floats.
        labels: Cluster labels (non-negative integers).

    Returns:
        Davies-Bouldin score.
    """
    ...
