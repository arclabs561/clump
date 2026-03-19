# CLAUDE.md -- clump

Dense clustering primitives for f32 vectors in Rust.

## Build & test

```bash
cargo test                        # default (no features)
cargo test --features parallel    # with rayon
cargo test --features simd        # with innr SIMD
cargo test --features gpu         # with Metal (macOS)
cargo test --features serde       # with serde derives
cargo test --features ndarray     # with ndarray adapters
cargo bench                       # criterion benchmarks
cargo bench --bench comparison    # vs linfa-clustering
```

## Architecture

```
src/cluster/
  mod.rs          -- module root, re-exports
  kmeans.rs       -- Lloyd + Hamerly bounds + warm-start
  dbscan.rs       -- tiered: matrix > grid > projection > VP-tree
  hdbscan.rs      -- MST + condensed tree + stability selection
  optics.rs       -- reachability ordering + DBSCAN-like extraction
  constrained.rs  -- COP-Kmeans with must/cannot-link
  correlation.rs  -- PIVOT + best-merge + delta-cached local search
  streaming.rs    -- MiniBatchKmeans (online learning rate)
  denstream.rs    -- streaming density with micro-cluster decay
  evoc.rs         -- MST on random projections, multi-layer hierarchy
  distance.rs     -- DistanceMetric trait + 5 implementations
  metrics.rs      -- silhouette, Calinski-Harabasz, Davies-Bouldin
  util.rs         -- Hamerly assign, k-means++ init, Prim's MST, pairwise distances
  flat.rs         -- FlatMatrix contiguous row-major layout
  vptree.rs       -- vantage-point tree for range/kNN queries
  projindex.rs    -- random projection index for high-d range queries
  gpu.rs          -- Metal compute shader for k-means assignment
  adapt.rs        -- ndarray/flat-slice input conversion helpers
```

## Performance invariants

- K-means: Hamerly bounds skip unchanged assignments. First iteration is brute-force O(n*k).
- K-means++: incremental min-distance cache, O(n*k) not O(n*k^2). Parallel for n >= 20000.
- DBSCAN: precomputed matrix for n^2*4 <= 1GB; grid index for d <= 5; projection index for d > 5.
- HDBSCAN: partial sort for core distances (select_nth_unstable). Parallel pairwise distances + core distances.
- Prim's MST: fused update + find-min. Parallel inner loop for n >= 5000.
- Distance: innr SIMD dispatch at d >= 32 (L2) or d >= 16 (cosine/dot) when `simd` feature enabled.
- Final reassignment pass after k-means convergence ensures labels match centroids.

## Feature flags

| Feature | Dep | Effect |
|---------|-----|--------|
| `parallel` | rayon | Parallel Hamerly assign, pairwise distances, init, core distances, MST |
| `simd` | innr | NEON/AVX2/AVX-512 distance functions via runtime dispatch |
| `gpu` | metal | Metal compute shader for k-means assignment (n*k >= 500k) |
| `serde` | serde | Serialize/Deserialize on KmeansFit, SignedEdge, etc. |
| `ndarray` | ndarray | array2_to_vecs, flat_to_vecs conversion helpers |

## Testing

- 198 tests: unit, proptest, cross-algorithm consistency
- Proptests on: k-means, DBSCAN, HDBSCAN, OPTICS, correlation, constrained, streaming, denstream
- Cross-algorithm: OPTICS extract_clusters(eps) must match DBSCAN(eps) cluster structure
- Key invariants tested: labels in range, predict consistency, WCSS non-negative, min_cluster_size enforced, ordering is permutation

## Known issues

- OPTICS Xi extraction not yet implemented (DBSCAN-like extraction only)
- k-means|| (parallel init) benchmarked and rejected: 2.3x slower than k-means++ at k=100

## Benchmarking

Run `cargo bench` for the full suite. Key configs:
- `n100000_d16_k100`: large-scale k-means
- `n200000_d128_k50`: massive high-dim (use `--features simd,parallel`)
- `--bench comparison`: head-to-head vs linfa-clustering

When optimizing: always benchmark before AND after. Reject changes that don't prove a measurable improvement. Document rejected approaches in commit messages.
