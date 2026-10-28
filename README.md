# clump

[![crates.io](https://img.shields.io/crates/v/clump.svg)](https://crates.io/crates/clump)
[![Documentation](https://docs.rs/clump/badge.svg)](https://docs.rs/clump)

Clustering algorithms.

## Overview

Group vectors into clusters or partition a graph from signed edges. The
quickstart compares k-means, where you choose the number of clusters, with
DBSCAN, which finds clusters from density and labels isolated points as noise.
See the [examples](examples/README.md) for streaming and constrained clustering.

## Quickstart

```toml
[dependencies]
clump = "0.6"
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
assert_eq!(labels[0], labels[1]);
assert_eq!(&labels[2..], &[clump::NOISE, clump::NOISE]);
```

`Kmeans::fit` returns a `KmeansFit` containing the centroids; call its `predict`
method to label new points. `Dbscan::fit_predict` assigns noise points to
`clump::NOISE`. Use `fit_predict_with_noise` to get `Option` labels instead.

## Algorithms

| Algorithm | Kind | Discovers k | Noise handling | Batch input |
|-----------|------|-------------|----------------|-------------|
| K-means | Centroid | No (k required) | None | `DataRef` |
| Mini-Batch K-means | Centroid (streaming) | No (k required) | None | `DataRef` |
| DBSCAN | Density | Yes | `NOISE` sentinel | `DataRef` |
| HDBSCAN | Density (hierarchical) | Yes | `NOISE` sentinel | `DataRef` |
| DenStream | Density (streaming) | Yes | Decaying outlier micro-clusters | `DataRef` |
| EVoC | Hierarchical | Yes | `None` outside retained components | `DataRef` |
| COP-Kmeans | Constrained centroid | No (k required) | None | `DataRef` + constraints |
| OPTICS | Density (reachability) | Yes | `NOISE` when extracting clusters | `DataRef` |
| Correlation Clustering | Graph-based | Yes | None | `SignedEdge` list |

## Zero-copy flat input

The vector APIs in the table accept `DataRef` batches. Pass `Vec<Vec<f32>>` or
use `FlatRef` for a zero-copy view over a flat buffer:

```rust
use clump::{FlatRef, Kmeans};

let flat = vec![0.0f32, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
let data = FlatRef::new(&flat, 4, 2);
let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
```

## Streaming clustering

```rust
use clump::MiniBatchKmeans;

let batch1 = vec![vec![0.0, 0.0], vec![5.0, 5.0], vec![10.0, 10.0]];
let batch2 = vec![vec![0.1, 0.1], vec![5.1, 5.1], vec![10.1, 10.1]];

let mut mbk = MiniBatchKmeans::new(3).with_seed(42);
mbk.update_batch(&batch1).unwrap();
mbk.update_batch(&batch2).unwrap();
assert_eq!(mbk.centroids().len(), 3);
```

## Constrained clustering

```rust
use clump::{CopKmeans, Constraint};

let data = vec![
    vec![0.0, 0.0],
    vec![0.1, 0.1],
    vec![10.0, 10.0],
    vec![10.1, 10.1],
];
let constraints = vec![
    Constraint::MustLink(0, 1),
    Constraint::CannotLink(0, 2),
];
let labels = CopKmeans::new(2)
    .with_seed(42)
    .fit_predict_constrained(&data, &constraints)
    .unwrap();
```

## Correlation clustering

```rust
use clump::{CorrelationClustering, SignedEdge};

let edges = vec![
    SignedEdge { i: 0, j: 1, weight: 1.0 },   // similar
    SignedEdge { i: 0, j: 2, weight: -1.0 },   // dissimilar
];
let result = CorrelationClustering::new().fit(3, &edges).unwrap();
let labels = result.labels;
```

Also see `edges_from_distances` to build signed edges from a distance matrix.

## Distance metrics

For vector algorithms that expose `with_metric`, use it to choose a
`DistanceMetric`. Correlation clustering uses weighted edges instead.
Built-in metrics are `SquaredEuclidean`, `Euclidean`, `CosineDistance`,
`InnerProductDistance`, and `CompositeDistance`. Custom metrics implement
`DistanceMetric` (`fn distance(&self, a: &[f32], b: &[f32]) -> f32`).

## Features

No optional features are enabled by default.

| Feature | Enables |
|---|---|
| `parallel` | Parallel computation with Rayon |
| `gpu` | K-means assignment with Metal on macOS |
| `serde` | Serialization support |
| `ndarray` | Conversions from `Array2` |
| `simd` | Hardware-accelerated distance calculations through `innr` |
| `blas` | SGEMM through ndarray |
| `hopfield` | Clustering with associative memory (CLAM) helpers |

## Examples

See [`examples/README.md`](examples/README.md) for run commands and sample
output.

| Example | What it shows |
|---------|---------------|
| `quickstart` | K-means and DBSCAN on synthetic data |
| `clustering` | Multiple algorithms on the same dataset, label comparison |
| `streaming` | Mini-Batch K-means and DenStream on streaming data |
| `evaluation` | Silhouette score, cluster quality metrics |
| `flat_input` | Zero-copy `FlatRef` input from raw `&[f32]` |

```sh
cargo run --example quickstart
```

## Benchmarks

`benches/comparison.rs` measures clump and linfa-clustering on identical
synthetic inputs for k-means and DBSCAN. No comparative results are published
here.

## License

MIT OR Apache-2.0
