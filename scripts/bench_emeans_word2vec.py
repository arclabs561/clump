#!/usr/bin/env python3
"""Compare clump trainers at matched training distortion on Word2Vec 10K.

Build and run from the repository root:
  cargo build --release --features simd --example emeans_word2vec
  python3 scripts/bench_emeans_word2vec.py target/release/examples/emeans_word2vec

Uses the public trainers (including their default initializers and final labeling),
k=32, unnormalized embeddings, three timing repeats, and five independent seeds.
Comparisons use the smallest tested budget reaching a common loss threshold for
all five seeds. We report the median over seeds of within-seed median fit times.
The result measures time for a preselected budget, not online stopping time.
"""

import argparse
import csv
import hashlib
import random
import statistics
import subprocess
import urllib.request
from pathlib import Path


URL = "https://storage.googleapis.com/embedding-projector/data/word2vec_10000_200d_tensors.bytes"
SHA256 = "deaedca3920530dcf2e6eb4084fd6a6af747bdfada8919886d1ed8b9438c297a"
BUDGETS = {"kmeans": [1, 2, 4, 8, 16, 32], "emeans": [0.5, 1, 2, 4, 8, 16, 32]}
SEEDS = [7, 13, 23, 37, 53]


def load_data(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=90) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("downloaded Word2Vec file failed checksum")
        path.write_bytes(data)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("cached Word2Vec file failed checksum")


def fit(binary, data, method, budget, seed):
    result = subprocess.check_output([binary, data, method, str(budget), str(seed)], text=True)
    seconds, loss = map(float, result.strip().split(","))
    return seconds, loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--data", type=Path, default=Path("target/word2vec-10k.f32"))
    parser.add_argument("--output", type=Path, default=Path("target/word2vec-matched.csv"))
    args = parser.parse_args()
    load_data(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    values = {}
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["phase", "method", "budget", "seed", "repeat", "fit_seconds", "loss_per_row"])
        # Three disjoint seeds establish the Lloyd-reference loss before any
        # measured comparisons. Both methods have a separate untimed warm-up.
        for method in BUDGETS:
            fit(args.binary, args.data, method, 1, 99)
        reference = []
        for seed in [101, 102, 103]:
            secs, loss = fit(args.binary, args.data, "kmeans", 100, seed)
            reference.append(loss)
            writer.writerow(["reference", "kmeans", 100, seed, 0, secs, loss])
        ref = statistics.median(reference)
        schedule = [(method, b, seed) for method, budgets in BUDGETS.items()
                    for b in budgets for seed in SEEDS]
        rng = random.Random(4123)
        for rep in range(3):
            rng.shuffle(schedule)
            for method, b, seed in schedule:
                secs, loss = fit(args.binary, args.data, method, b, seed)
                writer.writerow(["grid", method, b, seed, rep, secs, loss])
                output.flush()
                values.setdefault((method, b, seed), []).append((secs, loss))

    print(f"Reference loss {ref:.8f}; raw results {args.output}")
    for factor in [1.01, 1.00]:
        target = ref * factor
        print(f"Target {target:.8f} ({factor:.2f} times reference)")
        for method, budgets in BUDGETS.items():
            for budget in budgets:
                runs = [values[method, budget, seed] for seed in SEEDS]
                losses = [max(loss for _, loss in repetitions) for repetitions in runs]
                if max(losses) <= target:
                    times = [statistics.median(t for t, _ in r) for r in runs]
                    print(f"  {method}: budget {budget}, 5/5, {statistics.median(times):.6f}s, "
                          f"median loss {statistics.median(losses):.8f}")
                    break
            else:
                print(f"  {method}: no tested budget met target on every seed")


if __name__ == "__main__":
    main()
