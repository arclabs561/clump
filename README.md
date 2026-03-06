# clump

[![crates.io](https://img.shields.io/crates/v/clump.svg)](https://crates.io/crates/clump)
[![Documentation](https://docs.rs/clump/badge.svg)](https://docs.rs/clump)
[![CI](https://github.com/arclabs561/clump/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/clump/actions/workflows/ci.yml)

Dense clustering primitives (k-means, DBSCAN, HDBSCAN, EVoC).

Dual-licensed under MIT or Apache-2.0.

## Quickstart

```toml
[dependencies]
clump = "0.2"
```

```rust
use clump::{Clustering, Dbscan, Kmeans};

let data = vec![
    vec![0.0, 0.0],
    vec![0.1, 0.1],
    vec![1.0, 1.0],
    vec![10.0, 10.0],
    vec![11.0, 11.0],
];

// Hard clustering with k-means (default: squared Euclidean distance)
let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();

assert_eq!(labels.len(), data.len());
assert_eq!(labels[0], labels[1]); // near each other
assert_ne!(labels[0], labels[2]); // far apart

// Density clustering with DBSCAN (default: Euclidean distance)
let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
assert_eq!(labels.len(), data.len());
```

## Distance metrics

All clustering algorithms are generic over the `DistanceMetric` trait. The
default metric differs per algorithm:

| Algorithm | Default metric | Constructor |
|-----------|----------------|-------------|
| `Kmeans`  | `SquaredEuclidean` | `Kmeans::new(k)` |
| `Dbscan`  | `Euclidean` | `Dbscan::new(eps, min_pts)` |
| `Hdbscan` | `Euclidean` | `Hdbscan::new()` |
| `EVoC`    | `SquaredEuclidean` | `EVoC::new(params)` |

### Built-in metrics

| Metric | Formula |
|--------|---------|
| `SquaredEuclidean` | `sum((a_i - b_i)^2)` |
| `Euclidean` | `sqrt(sum((a_i - b_i)^2))` |
| `CosineDistance` | `1 - cos_sim(a, b)`, range `[0, 2]` |
| `InnerProductDistance` | `-dot(a, b)` |

### Using a different built-in metric

Each algorithm provides a `with_metric` constructor:

```rust
use clump::{Kmeans, CosineDistance, Clustering};

let data = vec![
    vec![1.0, 0.0],
    vec![0.9, 0.1],
    vec![0.0, 1.0],
    vec![0.1, 0.9],
];

let labels = Kmeans::with_metric(2, CosineDistance)
    .with_seed(42)
    .fit_predict(&data)
    .unwrap();

assert_eq!(labels[0], labels[1]);
assert_ne!(labels[0], labels[2]);
```

The same pattern works for the other algorithms:

```rust
use clump::{Dbscan, Hdbscan, CosineDistance, SquaredEuclidean, Clustering};

# let data = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0], vec![0.1, 0.9]];
// DBSCAN with cosine distance (epsilon is in cosine distance units)
let _labels = Dbscan::with_metric(0.1, 2, CosineDistance)
    .fit_predict(&data)
    .unwrap();

// HDBSCAN with squared Euclidean distance
let _labels = Hdbscan::with_metric(SquaredEuclidean)
    .with_min_samples(2)
    .with_min_cluster_size(2)
    .fit_predict(&data)
    .unwrap();
```

### Implementing a custom metric

Implement the `DistanceMetric` trait:

```rust
use clump::{DistanceMetric, Kmeans, Clustering};

/// Manhattan (L1) distance.
#[derive(Clone)]
struct Manhattan;

impl DistanceMetric for Manhattan {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
    }
}

let data = vec![
    vec![0.0, 0.0],
    vec![0.1, 0.1],
    vec![10.0, 10.0],
    vec![10.1, 10.1],
];

let labels = Kmeans::with_metric(2, Manhattan)
    .with_seed(42)
    .fit_predict(&data)
    .unwrap();

assert_eq!(labels[0], labels[1]);
assert_ne!(labels[0], labels[2]);
```

## EVoC

EVoC (Embedding Vector Oriented Clustering) produces hierarchical clusters
with multiple granularity layers and near-duplicate detection:

```rust
use clump::{EVoC, EVoCParams, Clustering};

let data = vec![
    vec![0.0, 0.0],
    vec![0.1, 0.1],
    vec![10.0, 10.0],
    vec![11.0, 11.0],
];

let mut evoc = EVoC::new(EVoCParams {
    intermediate_dim: 1,
    min_cluster_size: 2,
    seed: Some(42),
    ..Default::default()
});
let labels = evoc.fit_predict(&data).unwrap();
assert_eq!(labels.len(), data.len());
assert!(!evoc.cluster_layers().is_empty());
```

## Notes

- `Dbscan::fit_predict` returns a label for every point; noise points are assigned to a special
  cluster (`clump::NOISE`). If you want `Option` labels, use `Dbscan::fit_predict_with_noise`
  (or import the `DbscanExt` trait).
- `Kmeans::fit` returns centroids + labels (`KmeansFit`), which you can reuse to `predict` labels
  for new points.
- `docs.rs/clump` currently documents the latest crates.io release. If you depend on git main,
  prefer local rustdoc (`cargo doc --open`) for up-to-date docs.

## License

MIT OR Apache-2.0
