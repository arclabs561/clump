#!/usr/bin/env bash
# Run the MNIST comparison in fresh processes and retain raw JSON and RSS data.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPEATS="${1:-3}"
FEATURES="${CLUMP_FEATURES:-parallel}"
case "$REPEATS" in
  ''|*[!0-9]*|0) echo "usage: $0 [positive-repeat-count]" >&2; exit 2 ;;
esac

if [ ! -f "$ROOT/data/mnist/t10k-images-idx3-ubyte" ]; then
  echo "MNIST data not found; run ./scripts/fetch_mnist.sh" >&2
  exit 2
fi

mkdir -p "$ROOT/target/mnist-comparison"
RUN_DIR="$(mktemp -d "$ROOT/target/mnist-comparison/run.XXXXXX")"
BUILD_LOG="$RUN_DIR/build.jsonl"
PLATFORM="$(uname -s)"
case "$PLATFORM" in
  Darwin) RSS_COLUMN="peak_rss_bytes" ;;
  Linux) RSS_COLUMN="peak_rss_kib" ;;
  *) echo "unsupported platform for peak RSS: $PLATFORM" >&2; exit 2 ;;
esac

cd "$ROOT"
cargo build --release --features "$FEATURES" --bench mnist_comparison --message-format=json >"$BUILD_LOG"
BENCH_BIN="$(jq -r 'select(.reason == "compiler-artifact" and .target.name == "mnist_comparison") | .executable // empty' "$BUILD_LOG")"
if [ -z "$BENCH_BIN" ] || [ ! -x "$BENCH_BIN" ]; then
  echo "could not resolve the mnist_comparison executable" >&2
  exit 1
fi

measure() {
  local implementation="$1"
  local repeat="$2"
  local stem="$RUN_DIR/${implementation}-r${repeat}"
  case "$PLATFORM" in
    Darwin) /usr/bin/time -l -o "$stem.time" "$BENCH_BIN" "$implementation" >"$stem.json" ;;
    Linux) /usr/bin/time -v -o "$stem.time" "$BENCH_BIN" "$implementation" >"$stem.json" ;;
  esac
}

for ((repeat = 0; repeat < REPEATS; repeat++)); do
  if ((repeat % 2 == 0)); then
    measure clump "$repeat"
    measure linfa "$repeat"
  else
    measure linfa "$repeat"
    measure clump "$repeat"
  fi
done

"$BENCH_BIN" diagnose >"$RUN_DIR/diagnose.json"

printf 'implementation\trepeat\telapsed_seconds\t%s\n' "$RSS_COLUMN"
for implementation in clump linfa; do
  for ((repeat = 0; repeat < REPEATS; repeat++)); do
    stem="$RUN_DIR/${implementation}-r${repeat}"
    elapsed="$(jq -r '.result.elapsed_seconds' "$stem.json")"
    peak_rss="$(awk '/maximum resident set size/ {print $1} /Maximum resident set size/ {print $NF}' "$stem.time")"
    printf '%s\t%s\t%s\t%s\n' "$implementation" "$repeat" "$elapsed" "$peak_rss"
  done
done
printf 'clump_features\t%s\n' "$FEATURES"
printf 'diagnostic\t%s\n' "$(jq -c '{cross_implementation_ari, results: [.results[] | {implementation, ari, nmi, purity, wcss}]}' "$RUN_DIR/diagnose.json")"
printf 'raw_results\t%s\n' "$RUN_DIR"
