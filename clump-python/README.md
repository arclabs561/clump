# clumppy

Clustering algorithms (k-means, DBSCAN, HDBSCAN, COP-Kmeans, correlation clustering) with evaluation metrics, backed by a Rust implementation.

Python bindings for the [clump](https://crates.io/crates/clump) Rust crate.

## Install

    pip install clumppy

## Quick start

```python
import clumppy

# Correlation clustering -- determines k automatically from signed edges
edges = [
    (0, 1, 1.0),   # positive = should be together
    (0, 2, -1.0),  # negative = should be apart
    (1, 2, -1.0),
]
labels = clumppy.correlation_clustering(edges, n_nodes=3)

# Constrained k-means with must-link / cannot-link
data = [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]]
labels = clumppy.cop_kmeans(
    data, k=2,
    must_link=[(0, 1)],
    cannot_link=[(0, 2)],
    seed=42,
)

# Standard algorithms
labels, centroids = clumppy.kmeans(data, k=2, seed=42)
labels = clumppy.dbscan(data, eps=0.5, min_points=5)    # noise = -1
labels = clumppy.hdbscan(data, min_cluster_size=5)       # noise = -1

# Evaluation
score = clumppy.silhouette_score(data, labels)
```

## API

| Name | Description |
|------|-------------|
| `correlation_clustering` | Signed-edge graph partitioning, auto-determines k |
| `cop_kmeans` | K-means with must-link and cannot-link constraints |
| `kmeans` | Standard k-means, returns (labels, centroids) |
| `dbscan` | Density-based clustering, noise labeled -1 |
| `hdbscan` | Hierarchical DBSCAN, noise labeled -1 |
| `silhouette_score` | Mean silhouette coefficient, range [-1, 1] |
| `calinski_harabasz_score` | Variance ratio criterion (higher = better) |
| `davies_bouldin_score` | Cluster similarity index (lower = better) |

## numpy support

All functions accept numpy arrays or Python lists. Labels are returned as numpy int64 arrays; centroids as numpy float64 arrays.

## License

MIT OR Apache-2.0
