# clump

[![Documentation](https://docs.rs/clump/badge.svg)](https://docs.rs/clump)

Dense clustering primitives (k-means and related helpers).

Dual-licensed under MIT or Apache-2.0.

## Quickstart

```toml
[dependencies]
clump = "0.1.0"
```

```rust
use clump::{kmeans, KMeansConfig};

let data: Vec<Vec<f32>> = vec![
    vec![0.0, 0.0],
    vec![1.0, 1.0],
    vec![10.0, 10.0],
    vec![11.0, 11.0],
];
let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();

let cfg = KMeansConfig::new(2);
let result = kmeans(&refs, &cfg).unwrap();

assert_eq!(result.centroids.len(), 2);
assert_eq!(result.assignments.len(), 4);
```

## License

MIT OR Apache-2.0
