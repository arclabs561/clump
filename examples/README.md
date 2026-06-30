# clump examples

Each example answers one question and is runnable as-is. Examples that need a
dataset are **data-gated**: they exit 0 with a message telling you which fetch
script to run, so they are safe to compile/run in CI.

All outputs below are real, captured from a run.

## Getting started

### `quickstart` — what does clustering with clump look like end to end?

Generates a mixed 2D dataset (two blobs plus an outlier), runs k-means and
DBSCAN, and predicts the cluster of new points.

```bash
cargo run --release --example quickstart
```
```text
Generated 101 points: 50 near origin, 50 near (20,20), 1 outlier

--- K-means (k=2) ---
Iterations: 2
WCSS (inertia): 12208.66
Cluster sizes: 50 and 51
Predicted: (0.5,0.5) -> cluster 0, (20.5,20.5) -> cluster 1

--- DBSCAN (eps=2.0, min_pts=3) ---
Clusters found: 2
Noise points: 1
Outlier label: NOISE
```

### `clustering` — how do k-means and DBSCAN label the same points?

Prints the per-point cluster assignment for k-means, DBSCAN, and HDBSCAN on a
small three-blob dataset, so you can see where the algorithms agree and differ.

```bash
cargo run --release --example clustering
```
```text
=== K-means (k=3) ===
  point  0 (  0.0,   0.0) => cluster 0
  point  4 (  5.0,   5.0) => cluster 2
  point  8 ( 10.0,   0.0) => cluster 1
  ...

=== DBSCAN (eps=1.0, min_pts=2) ===
  point  0 (  0.0,   0.0) => cluster 0
  point  4 (  5.0,   5.0) => cluster 1
  ...
```

### `flat_input` — what input layouts does clump accept?

The same k-means call over a flat `&[f32]` slice, a `Vec<Vec<f32>>`, and (with
`--features ndarray`) an `ndarray` matrix, showing all paths produce identical
labels.

```bash
cargo run --release --example flat_input
```
```text
--- FlatRef path ---
Labels: [0, 0, 0, 1, 1, 1]
WCSS:   0.0800

--- Vec<Vec<f32>> path ---
Labels: [0, 0, 0, 1, 1, 1]
WCSS:   0.0800

Both paths produce identical labels and WCSS. PASSED

ndarray path skipped (enable with --features ndarray)
```

## Choosing and evaluating

### `evaluation` — how do I choose k and judge cluster quality?

Sweeps `k` and reports WCSS, silhouette, Calinski-Harabasz, and Davies-Bouldin
so the trade-offs are visible, then scores a DBSCAN run with DISCO (which also
grades noise assignment).

```bash
cargo run --release --example evaluation
```
```text
Data: 121 points (3 clusters of 40 + 1 outlier)

  k        WCSS   Silhouette  Calinski-Hab   Davies-Bould
--------------------------------------------------------
  2     29750.2       0.6863         104.8         0.6814
  3     12265.2       0.9280         223.2         0.1989
  4        80.0       0.9663       28575.8         0.0385
  5        60.9       0.8421       28008.2         0.2259
  6        41.5       0.7182       32609.5         0.3460

--- DBSCAN + DISCO ---
DBSCAN (eps=3.0, min_pts=3): 3 clusters, 1 noise points
DISCO score: 0.9665
```

## Streaming

### `streaming` — can I cluster a stream without holding all the data?

Feeds points in batches to MiniBatchKmeans and DenStream, showing centroids and
micro-clusters update online.

```bash
cargo run --release --example streaming
```
```text
--- MiniBatchKmeans (k=2) ---
After batch 1 (20 pts): centroids at ["(0.50, 0.50)", "(1.50, 1.50)"]
After batch 2 (20 pts): centroids at ["(0.50, 0.50)", "(35.60, 35.60)"]
After batch 3 (30 pts): centroids at ["(0.66, 0.66)", "(35.60, 35.60)"]
Predict: (0.5,0.5)->cluster 0, (50.5,50.5)->cluster 1

--- DenStream ---
After 30 pts near origin: 2 potential micro-clusters
After 30 more pts near (50,50): 4 potential micro-clusters
Macro-clusters: 2, noise micro-clusters: 0
Predict: (0.5,0.5)->mc 0, (50.5,50.5)->mc 2
```

## Real data

### `mnist_kmeans_ari` — does k-means recover real structure on MNIST?

Runs k-means on the 10000-image MNIST test split and scores the clusters against
the true digit labels (ARI / NMI / purity), plus an internal silhouette.

Data-gated: needs the MNIST test split (`data/` is gitignored). Fetch it first:

```bash
./scripts/fetch_mnist.sh
cargo run --release --example mnist_kmeans_ari
```
```text
images: 10000  dims: 784  classes: 10

external (vs true digit labels):
  ARI     = 0.2619
  NMI     = 0.4030
  purity  = 0.4652
internal (cluster geometry):
  silhouette = 0.0415
```

Plain k-means on raw pixels recovers moderate digit structure (ARI 0.26); the
gap to 1.0 is the well-known limit of Euclidean k-means on unprocessed images.

## Datasets

`data/` is not tracked. The data-gated examples no-op with a fetch message when
it is absent; `scripts/fetch_mnist.sh` downloads the MNIST test split.
