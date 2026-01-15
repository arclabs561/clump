# clump

Dense clustering primitives: k-means, DBSCAN.

Dual-licensed under MIT or Apache-2.0.

```rust
use clump::cluster::{Clustering, Kmeans};

let data = vec![vec![1.0, 2.0], vec![1.1, 2.1], vec![5.0, 5.0]];
let model = Kmeans::new(2).with_seed(42);
let labels = model.fit_predict(&data).unwrap();
assert_eq!(labels.len(), data.len());
```

