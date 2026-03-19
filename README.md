# clump

[![crates.io](https://img.shields.io/crates/v/clump.svg)](https://crates.io/crates/clump)
[![Documentation](https://docs.rs/clump/badge.svg)](https://docs.rs/clump)
[![CI](https://github.com/arclabs561/clump/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/clump/actions/workflows/ci.yml)

Clustering algorithms for dense `f32` vectors in Rust. 9 algorithms, SIMD-accelerated, with optional GPU and parallel support.

## Algorithms

| Algorithm | Kind | Discovers k | Noise handling | Input |
|-----------|------|-------------|----------------|-------|
| K-means | Centroid | No (k required) | None | `&impl DataRef` |
| Mini-Batch K-means | Centroid (streaming) | No (k required) | None | `&impl DataRef` |
| DBSCAN | Density | Yes | Labels noise (`NOISE` sentinel) | `&impl DataRef` |
| HDBSCAN | Density (hierarchical) | Yes | Labels noise | `&impl DataRef` |
| DenStream | Density (streaming) | Yes | Decays outliers | `&impl DataRef` |
| EVoC | Hierarchical | Yes | Near-duplicate detection | `&impl DataRef` |
| COP-Kmeans | Constrained centroid | No (k required) | None | `&impl DataRef` + constraints |
| OPTICS | Density (reachability) | Yes | Reachability plot | `&impl DataRef` |
| Correlation Clustering | Graph-based | Yes | None | `SignedEdge` list |

## Quickstart

```toml
[dependencies]
clump = "0.5.1"
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

## Zero-copy flat input

All algorithms accept `&impl DataRef`. Pass `Vec<Vec<f32>>` or use `FlatRef` for zero-copy flat buffers:

```rust
use clump::{FlatRef, Kmeans};

let flat = vec![0.0f32, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
let data = FlatRef::new(&flat, 4, 2);
let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
```

## Streaming clustering

```rust
use clump::MiniBatchKmeans;

let mut mbk = MiniBatchKmeans::new(3).with_seed(42);
mbk.update_batch(&batch1).unwrap();
mbk.update_batch(&batch2).unwrap();
// Centroids available via mbk.centroids()
```

## Constrained clustering

```rust
use clump::{CopKmeans, Constraint};

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
| `gpu` | off | Metal GPU acceleration for k-means assignment (macOS only) |
| `serde` | off | Serialize/deserialize for `KmeansFit`, `SignedEdge`, `Constraint`, etc. |
| `ndarray` | off | Conversion helpers between `Array2<f32>` and clump input format |
| `simd` | off | SIMD-accelerated distance via [innr](https://crates.io/crates/innr) (NEON/AVX2/AVX-512) |

## License

MIT OR Apache-2.0
