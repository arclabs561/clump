# Changelog

## [Unreleased]

### Changed

- DenStream now buffers 1,000 raw points for its initialization phase and uses
  the weighted, radius-aware offline phase described by Cao et al. Set
  `with_initial_buffer_size(0)` for immediate online processing. The former
  centroid-DBSCAN helper remains available as `macro_cluster_unweighted`.
- `DenStream::new` now uses Euclidean distance so `epsilon`, micro-cluster
  radius, offline reachability, and prediction share units. Explicit legacy
  squared-distance behavior is available through
  `new_squared_euclidean_legacy`.

## [0.5.8] - 2026-07-03

### Fixed

- HDBSCAN excess-of-mass selection over-split stable clusters: the selection
  loop walked clusters in reverse-id order assuming ids are topological, but
  a merge allocates the parent id before fresh child ids, so leaf clusters
  born at early merges can carry lower ids than later ancestors. Such leaves
  were visited after their ancestor had already selected itself and
  deselected the subtree, and the unconditional leaf-select re-selected them
  (three well-separated blobs came back as six clusters). Selection now runs
  in explicit post-order, which also guarantees child subtree stabilities are
  fully propagated before any parent compares against them. Pinned by a new
  rosetta fixture (`tests/rosetta_clump_hdbscan.json`) generated from the
  McInnes `hdbscan` reference implementation on a separated-blobs control and
  a three-level nested-density hierarchy; both now match the reference
  partition exactly. The suspected core-distance off-by-one was empirically
  refuted with the same fixture (shifting the convention in either direction
  breaks agreement; the shipped convention matches the reference).

## [0.5.7] - 2026-06-10

### Fixed

- GPU module gated to macOS targets so `--all-features` builds cleanly on Linux.
