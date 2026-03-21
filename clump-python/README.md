# clumppy

Python bindings for [clump](https://docs.rs/clump), a Rust library for dense clustering.

Provides COP-Kmeans (constrained k-means with must-link/cannot-link constraints),
correlation clustering (signed-edge graph partitioning without specifying k),
k-means, DBSCAN, HDBSCAN, and cluster evaluation metrics.

## Install

```
pip install clumppy
```

## Usage

### Constrained clustering (COP-Kmeans)

```python
import clumppy

data = [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]]
labels = clumppy.cop_kmeans(
    data, k=2,
    must_link=[(0, 1)],
    cannot_link=[(0, 2)],
    seed=42,
)
# labels[0] == labels[1], labels[0] != labels[2]
```

### Correlation clustering

```python
edges = [
    (0, 1, 1.0),   # positive = should be together
    (0, 2, -1.0),  # negative = should be apart
    (1, 2, -1.0),
]
labels = clumppy.correlation_clustering(edges, n_nodes=3)
# Automatically determines number of clusters
```

### K-means, DBSCAN, HDBSCAN

```python
labels, centroids = clumppy.kmeans(data, k=3, seed=42)
labels = clumppy.dbscan(data, eps=0.5, min_points=5)    # noise = -1
labels = clumppy.hdbscan(data, min_cluster_size=10)      # noise = -1
```

### Evaluation metrics

```python
score = clumppy.silhouette_score(data, labels)
score = clumppy.calinski_harabasz_score(data, labels)
score = clumppy.davies_bouldin_score(data, labels)
```

## Functions

| Function | Description |
|----------|-------------|
| `cop_kmeans` | K-means with must-link and cannot-link constraints |
| `correlation_clustering` | Signed-edge graph partitioning (auto k) |
| `kmeans` | Standard k-means, returns (labels, centroids) |
| `dbscan` | Density-based clustering, noise = -1 |
| `hdbscan` | Hierarchical DBSCAN, noise = -1 |
| `silhouette_score` | Mean silhouette coefficient [-1, 1] |
| `calinski_harabasz_score` | Variance ratio criterion (higher = better) |
| `davies_bouldin_score` | Cluster similarity index (lower = better) |

## Rust documentation

Full algorithm details, complexity analysis, and benchmarks: <https://docs.rs/clump>
