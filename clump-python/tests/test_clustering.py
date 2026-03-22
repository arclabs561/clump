"""Tests for clumppy Python bindings."""

import numpy as np

import clumppy


def _three_clusters():
    """Generate 3 well-separated clusters of 5 points each."""
    cluster_a = [[0.0 + i * 0.1, 0.0 + i * 0.1] for i in range(5)]
    cluster_b = [[50.0 + i * 0.1, 50.0 + i * 0.1] for i in range(5)]
    cluster_c = [[100.0 + i * 0.1, 100.0 + i * 0.1] for i in range(5)]
    return cluster_a + cluster_b + cluster_c


# ---------------------------------------------------------------------------
# Existing list-based tests (must keep working)
# ---------------------------------------------------------------------------


def test_kmeans_basic():
    data = _three_clusters()
    labels, centroids = clumppy.kmeans(data, k=3, seed=42)

    assert isinstance(labels, np.ndarray)
    assert isinstance(centroids, np.ndarray)
    assert len(labels) == 15
    assert centroids.shape == (3, 2)

    # All points within each group of 5 should share a label.
    assert len(set(labels[0:5].tolist())) == 1
    assert len(set(labels[5:10].tolist())) == 1
    assert len(set(labels[10:15].tolist())) == 1

    # Three distinct labels.
    assert len(set(labels.tolist())) == 3


def test_dbscan_basic():
    # Dense cluster + one outlier.
    data = [
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [0.1, 0.1],
        [100.0, 100.0],  # outlier
    ]
    labels = clumppy.dbscan(data, eps=0.5, min_points=3)

    assert isinstance(labels, np.ndarray)
    assert len(labels) == 5
    # First 4 points should be in the same cluster.
    assert labels[0] == labels[1] == labels[2] == labels[3]
    # Outlier should be noise.
    assert labels[4] == -1


def test_hdbscan_basic():
    # Two tight clusters, well-separated.
    cluster_a = [[i * 0.1, j * 0.1] for i in range(4) for j in range(4)]
    cluster_b = [[50.0 + i * 0.1, 50.0 + j * 0.1] for i in range(4) for j in range(4)]
    data = cluster_a + cluster_b

    labels = clumppy.hdbscan(data, min_cluster_size=5, min_points=3)

    assert isinstance(labels, np.ndarray)
    assert len(labels) == 32
    # Should find at least 2 non-noise clusters.
    non_noise = set(l for l in labels.tolist() if l != -1)
    assert len(non_noise) >= 2


def test_cop_kmeans_must_link():
    data = [
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 10.0],
        [10.1, 10.1],
    ]
    labels = clumppy.cop_kmeans(
        data, k=2, must_link=[(0, 1)], cannot_link=[], seed=42
    )

    assert isinstance(labels, np.ndarray)
    assert labels[0] == labels[1]


def test_cop_kmeans_cannot_link():
    data = [
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 10.0],
        [10.1, 10.1],
    ]
    labels = clumppy.cop_kmeans(
        data, k=2, must_link=[], cannot_link=[(0, 1)], seed=42
    )

    assert isinstance(labels, np.ndarray)
    assert labels[0] != labels[1]


def test_correlation_clustering():
    # Triangle: 0-1 positive, 0-2 positive, 1-2 negative.
    # Optimal: {0, 1} and {2}.
    edges = [
        (0, 1, 1.0),
        (0, 2, -1.0),
        (1, 2, -1.0),
    ]
    labels = clumppy.correlation_clustering(edges, n_nodes=3)

    assert isinstance(labels, np.ndarray)
    assert len(labels) == 3
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_silhouette_score():
    # Two well-separated clusters.
    data = [
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1],
    ]
    labels = [0, 0, 0, 1, 1, 1]
    score = clumppy.silhouette_score(data, labels)
    # Well-separated clusters should have high silhouette score.
    assert score > 0.5


def test_noise_labels_are_negative_one():
    """DBSCAN and HDBSCAN should map noise to -1, not usize::MAX."""
    # Single point + outlier: too few for a cluster.
    data = [[0.0, 0.0], [100.0, 100.0]]
    labels = clumppy.dbscan(data, eps=0.5, min_points=3)
    # Both points should be noise with these parameters.
    for l in labels.tolist():
        assert l == -1 or l >= 0, f"unexpected label: {l}"
    # At least one should be noise.
    assert -1 in labels.tolist()


# ---------------------------------------------------------------------------
# Numpy array input tests
# ---------------------------------------------------------------------------


def test_kmeans_numpy():
    data = np.array(_three_clusters(), dtype=np.float64)
    labels, centroids = clumppy.kmeans(data, k=3, seed=42)

    assert isinstance(labels, np.ndarray)
    assert isinstance(centroids, np.ndarray)
    assert labels.dtype == np.int64
    assert centroids.dtype == np.float64
    assert len(labels) == 15
    assert centroids.shape == (3, 2)
    assert len(set(labels.tolist())) == 3


def test_kmeans_numpy_f32():
    data = np.array(_three_clusters(), dtype=np.float32)
    labels, centroids = clumppy.kmeans(data, k=3, seed=42)

    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int64
    assert len(labels) == 15


def test_dbscan_numpy():
    rng = np.random.default_rng(42)
    cluster = rng.normal(loc=0.0, scale=0.1, size=(20, 3))
    outlier = np.array([[100.0, 100.0, 100.0]])
    data = np.vstack([cluster, outlier])

    labels = clumppy.dbscan(data, eps=0.5, min_points=3)

    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int64
    assert len(labels) == 21
    # Outlier should be noise.
    assert labels[-1] == -1


def test_hdbscan_numpy():
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=0.0, scale=0.1, size=(20, 2))
    c2 = rng.normal(loc=10.0, scale=0.1, size=(20, 2))
    data = np.vstack([c1, c2])

    labels = clumppy.hdbscan(data, min_cluster_size=5, min_points=3)

    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int64
    assert len(labels) == 40


def test_cop_kmeans_numpy():
    data = np.array(
        [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]], dtype=np.float64
    )
    labels = clumppy.cop_kmeans(
        data, k=2, must_link=[(0, 1)], cannot_link=[], seed=42
    )

    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int64
    assert labels[0] == labels[1]


def test_silhouette_score_numpy():
    data = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    score = clumppy.silhouette_score(data, labels)
    assert score > 0.5


def test_calinski_harabasz_numpy():
    data = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    score = clumppy.calinski_harabasz_score(data, labels)
    assert score > 0.0


def test_davies_bouldin_numpy():
    data = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    score = clumppy.davies_bouldin_score(data, labels)
    # Well-separated clusters should have low DB index.
    assert score < 1.0


def test_correlation_clustering_returns_numpy():
    edges = [(0, 1, 1.0), (0, 2, -1.0), (1, 2, -1.0)]
    labels = clumppy.correlation_clustering(edges, n_nodes=3)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int64


def test_correlation_clustering_seed():
    edges = [(0, 1, 1.0), (0, 2, -1.0), (1, 2, -1.0)]
    labels_a = clumppy.correlation_clustering(edges, n_nodes=3, seed=123)
    labels_b = clumppy.correlation_clustering(edges, n_nodes=3, seed=123)
    np.testing.assert_array_equal(labels_a, labels_b)
