"""Repeat the Appendix I readout trial with Clump's own assignment and split stream."""
import argparse
import csv
from pathlib import Path

import clumppy
import numpy as np
from sklearn.datasets import load_digits

from bench_emeans_finalizers import SPEEDS, SEEDS, mse
from bench_emeans_word2vec import load_data


def compare(dataset, rows, n, k, writer):
    order = np.random.default_rng(20261121).permutation(len(rows))
    train, val, test = np.split(rows[order], [n, (len(rows) + n) // 2])
    trials = {}
    for seed in SEEDS:
        # Only the returned codebook changes. The fixed seed gives identical
        # initial centers, row order, assignments and donor/child splits.
        trials[seed] = {}
        for speed in [None, *SPEEDS]:
            _, centers = clumppy.emeans(train, k, epochs=3, seed=seed, speed=speed)
            trials[seed][speed] = centers
    speed = min(SPEEDS, key=lambda s: np.median([mse(val, run[s]) for run in trials.values()]))
    deltas = []
    for seed, run in trials.items():
        closing, slow = run[None], run[speed]
        baseline, candidate = mse(test, closing), mse(test, slow)
        deltas.append(100 * (candidate / baseline - 1))
        writer.writerow([dataset, k, seed, speed, mse(train, closing), mse(train, slow),
                         baseline, candidate])
    print(f"{dataset} k={k} speed={speed:g}: {np.median(deltas):+.3f}% paired test error, "
          f"{sum(d < 0 for d in deltas)}/5 wins")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("target/word2vec-10k.f32"))
    parser.add_argument("--output", type=Path, default=Path("target/emeans-native-readout.csv"))
    args = parser.parse_args()
    load_data(args.data)
    datasets = [("digits", load_digits().data.astype(np.float32), 1024),
                ("word2vec", np.fromfile(args.data, dtype="<f4").reshape(10_000, 200), 4096)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "k", "seed", "speed", "train_closing", "train_slow",
                         "test_closing", "test_slow"])
        for name, rows, n in datasets:
            for k in (n // 128, n // 16):
                compare(name, rows, n, k, writer)


if __name__ == "__main__":
    main()
