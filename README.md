# clump

[![crates.io](https://img.shields.io/crates/v/clump.svg)](https://crates.io/crates/clump)
[![Documentation](https://docs.rs/clump/badge.svg)](https://docs.rs/clump)
[![CI](https://github.com/arclabs561/clump/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/clump/actions/workflows/ci.yml)

Clustering algorithms for dense `f32` vectors in Rust. No unsafe code.

## Algorithms

| Algorithm | Kind | Discovers k | Noise handling | Input |
|-----------|------|-------------|----------------|-------|
| K-means | Centroid | No (k required) | None | `&[Vec<f32>]` |
| Mini-Batch K-means | Centroid (streaming) | No (k required) | None | `StreamingClustering` trait |
| DBSCAN | Density | Yes | Labels noise (`NOISE` sentinel) | `&[Vec<f32>]` |
| HDBSCAN | Density (hierarchical) | Yes | Labels noise | `&[Vec<f32>]` |
| DenStream | Density (streaming) | Yes | Decays outliers | `StreamingClustering` trait |
| EVoC | Hierarchical | Yes | Near-duplicate detection | `&[Vec<f32>]` |
| COP-Kmeans | Constrained centroid | No (k required) | None | `&[Vec<f32>]` + constraints |
| Correlation Clustering | Graph-based | Yes | None | `SignedEdge` list |

## Quickstart

```toml
[dependencies]
clump = "0.3.0"
```

```rust
use clump::{Dbscan, Kmeans};

let data = vec![
    vec![0.0, 0.0],
    vec![0.1, 0.1],
    vec![10.0, 10.0],
    vec![11.0, 11.0],
];

// K-means: returns labels (default: squared Euclidean)
let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
assert_eq!(labels[0], labels[1]);
assert_ne!(labels[0], labels[2]);

// DBSCAN: discovers clusters from density (default: Euclidean)
let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
```

`Kmeans::fit` returns `KmeansFit` with centroids, which supports `predict` on new points. `Dbscan::fit_predict` assigns noise points to `clump::NOISE`; use `fit_predict_with_noise` for `Option` labels.

## Streaming clustering

```rust
use clump::{MiniBatchKmeans, DenStream, StreamingClustering};

let mut mbk = MiniBatchKmeans::new(3).with_seed(42);
mbk.partial_fit(&batch1).unwrap();
mbk.partial_fit(&batch2).unwrap();
let labels = mbk.predict(&data).unwrap();

let mut ds = DenStream::new(0.5, 3);
ds.partial_fit(&batch1).unwrap();
let labels = ds.predict(&data).unwrap();
```

## Constrained clustering

```rust
use clump::{CopKmeans, Constraint};

let constraints = vec![
    Constraint::MustLink(0, 1),
    Constraint::CannotLink(0, 2),
];
let labels = CopKmeans::new(2)
    .with_constraints(&constraints)
    .with_seed(42)
    .fit_predict(&data)
    .unwrap();
```

## Correlation clustering

```rust
use clump::{CorrelationClustering, SignedEdge};

let edges = vec![
    SignedEdge { i: 0, j: 1, weight: 1.0 },   // similar
    SignedEdge { i: 0, j: 2, weight: -1.0 },   // dissimilar
];
let labels = CorrelationClustering::new().fit_predict_from_edges(&edges, 3).unwrap();
```

Also see `edges_from_distances` to build signed edges from a distance matrix.

## Distance metrics

All algorithms are generic over `DistanceMetric`. Built-in metrics:

| Metric | Formula |
|--------|---------|
| `SquaredEuclidean` | `sum((a_i - b_i)^2)` |
| `Euclidean` | `sqrt(sum((a_i - b_i)^2))` |
| `CosineDistance` | `1 - cos_sim(a, b)` |
| `InnerProductDistance` | `-dot(a, b)` |
| `CompositeDistance` | Weighted sum of metrics |

Use `with_metric` on any algorithm to swap the metric:

```rust
use clump::{Kmeans, CosineDistance, DistanceMetric};

let labels = Kmeans::with_metric(2, CosineDistance)
    .with_seed(42)
    .fit_predict(&data)
    .unwrap();
```

Custom metrics: implement `DistanceMetric` (one method: `fn distance(&self, a: &[f32], b: &[f32]) -> f32`).

## Features

| Feature | Default | Effect |
|---------|---------|--------|
| `parallel` | off | Enables Rayon parallelism for k-means and batch operations |

## License

MIT OR Apache-2.0
