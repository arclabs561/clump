# clump

[![Documentation](https://docs.rs/clump/badge.svg)](https://docs.rs/clump)

Dense clustering primitives (k-means, DBSCAN, EVōC).

Dual-licensed under MIT or Apache-2.0.

## Quickstart

```toml
[dependencies]
# Git dependency (main). Pin `rev` for reproducibility.
clump = { git = "https://github.com/arclabs561/clump" }
```

```rust
use clump::cluster::{Clustering, Dbscan, Kmeans};
use clump::cluster::{EVoC, EVoCParams};

let data = vec![
    vec![0.0, 0.0],
    vec![0.1, 0.1],
    vec![1.0, 1.0],
    vec![10.0, 10.0],
    vec![11.0, 11.0],
];

// Hard clustering with k-means
let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();

assert_eq!(labels.len(), data.len());
assert_eq!(labels[0], labels[1]); // near each other
assert_ne!(labels[0], labels[2]); // far apart

// Density clustering with DBSCAN
let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
assert_eq!(labels.len(), data.len());

// Hierarchical clustering with EVōC (noise as `None`)
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
  cluster (`clump::NOISE`). If you want `Option` labels, use `DbscanExt::fit_predict_with_noise`.
- `Kmeans::fit` returns centroids + labels (`KmeansFit`), which you can reuse to `predict` labels
  for new points.
- `docs.rs/clump` currently documents the latest crates.io release. If you depend on git main,
  prefer local rustdoc (`cargo doc --open`) for up-to-date docs.

## License

MIT OR Apache-2.0
