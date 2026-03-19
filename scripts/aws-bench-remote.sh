#!/usr/bin/env bash
# aws-bench-remote.sh -- runs on each EC2 instance.
#
# Installs Rust, clones the repo at a given ref, runs tests (fail-fast),
# then runs criterion benchmarks.
#
# Usage: aws-bench-remote.sh <repo_url> <git_sha>

set -euo pipefail

REPO_URL="$1"
GIT_SHA="$2"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# -- install system dependencies ---------------------------------------------

log "Installing build dependencies..."
sudo dnf install -y gcc gcc-c++ make git openssl-devel pkgconfig \
    >/dev/null 2>&1

# -- install Rust stable -----------------------------------------------------

log "Installing Rust stable toolchain..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
source "${HOME}/.cargo/env"

rustc --version
cargo --version

# -- clone repo at specified ref ---------------------------------------------

log "Cloning ${REPO_URL} at ${GIT_SHA}..."
git clone "${REPO_URL}" clump
cd clump
git checkout "${GIT_SHA}"

log "Checked out $(git rev-parse --short HEAD)"

# -- run tests (fail-fast) ---------------------------------------------------

log "Running tests (--all-features)..."
if ! cargo test --all-features 2>&1; then
    log "Tests FAILED. Skipping benchmarks."
    exit 1
fi
log "Tests passed."

# -- run benchmarks ----------------------------------------------------------

log "Running bench: clustering..."
cargo bench --bench clustering 2>&1

log "Running bench: comparison..."
cargo bench --bench comparison 2>&1

log "Benchmarks complete. Criterion output in target/criterion/."
