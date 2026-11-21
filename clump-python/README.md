# clumppy

Cluster vectors and partition graphs using signed edges.

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
labels, centroids = clumppy.emeans(data, k=2, epochs=3.5, seed=42)
db_labels = clumppy.dbscan(data, eps=0.5, min_points=2)
hdb_labels = clumppy.hdbscan(data, min_cluster_size=2, min_points=1)

# Evaluation
score = clumppy.silhouette_score(data, labels)
```

## Assign new vectors with e-means

A codebook is useful only if its centroids represent future inputs. Fit on
the training rows and measure the squared distance from each new row to its
nearest centroid. Here the two future rows lie beyond the ends of two groups.

```python
import clumppy
import numpy as np

train = np.array([[0.], [1.], [2.], [10.], [11.], [12.]])
future = np.array([[-1.], [13.]])
_, centers = clumppy.emeans(train, k=2, epochs=3, seed=7)
d2 = ((future[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
nearest = d2.argmin(axis=1)
print(centers[nearest].ravel())  # [ 1. 11.]
print(d2.min(axis=1).mean())    # 4.0 squared distance per future row
```

The [held-out codebook experiment](../examples/README.md#e-means-finalizers)
uses the same score on digit images and word vectors.

## API

| Name | Description |
|------|-------------|
| `correlation_clustering` | Signed-edge graph partitioning, auto-determines k |
| `cop_kmeans` | K-means with must-link and cannot-link constraints |
| `kmeans` | Standard k-means, returns (labels, centroids) |
| `emeans` | CPU e-means with fractional epochs, returns (labels, centroids) |
| `dbscan` | Density-based clustering, noise labeled -1 |
| `hdbscan` | Hierarchical DBSCAN, noise labeled -1 |
| `silhouette_score` | Mean silhouette coefficient, range [-1, 1] |
| `calinski_harabasz_score` | Variance ratio criterion (higher = better) |
| `davies_bouldin_score` | Cluster similarity index (lower = better) |

## NumPy support

Functions that take point data accept two-dimensional NumPy arrays or lists of
lists. Labels are returned as NumPy `int64` arrays; centroids as NumPy `float64`
arrays. Correlation clustering takes a list of `(i, j, weight)` edge tuples.

## License

MIT OR Apache-2.0
