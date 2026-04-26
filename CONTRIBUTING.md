# Contributing to clump

Thanks for your interest. clump is a clustering library covering K-means, DBSCAN, HDBSCAN, OPTICS, EVoC, DenStream, COP-Kmeans, and correlation clustering.

## Before you start

For non-trivial work (new algorithms, API changes, large refactors), open an issue first to align on scope. Drive-by bug fixes and doc patches don't need an issue.

## Setup

- Rust toolchain: stable. Use `rustup` to manage.
- Optional: `cargo-nextest` for faster test runs (`cargo install cargo-nextest`).

```
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

## Style

- Direct, lowercase prose in commits. No marketing words ("powerful", "robust", "elegant"). No em-dashes in prose.
- Commit messages: `clump: short lowercase description`. One commit per logical change. Don't mix renames with behavioral changes.
- `cargo fmt` and `cargo clippy --all-targets --all-features -- -D warnings` must pass before `git add`.
- Test names should describe the property under test, not the function under test.

## Testing

- `cargo test --all-features` for the full matrix.
- Algorithm-specific tests live next to the implementation under `src/cluster/<algo>/`.
- For new algorithms, include at least one convergence-on-known-data test plus property tests for invariants (label count, noise handling).

## Feature flags

`Cargo.toml` defines `gpu` (Metal-only, macOS), `parallel` (rayon), `simd` (innr), `blas` (matrixmultiply), `ndarray`, `serde`, `hopfield` (CLAM module). The `gpu` feature is gated to `target_os = "macos"` so `--all-features` works on Linux CI.

## Pull requests

- Keep PRs scoped to one concern.
- Show before/after for behavior changes.
- Link the related issue.
- CI must be green before requesting review.

## License

Dual-licensed under MIT or Apache-2.0 at your option. By contributing you agree your contributions are licensed under both.
