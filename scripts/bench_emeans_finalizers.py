"""Compare Appendix I readouts on the same training assignments.

Run with the clumppy wheel built from this checkout, faiss-cpu==1.16.0,
scikit-learn and numpy. The Faiss version fixes the two internal hooks used
below. They replay the actual batch statistics and donor/child splits.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/bench_emeans_finalizers.py
"""
import argparse
import csv
from pathlib import Path
from time import perf_counter

import clumppy
import numpy as np
from faiss.contrib.e_means.e_means import NumpyEmeans
from sklearn.datasets import load_digits

from bench_emeans_word2vec import load_data

SPEEDS = [1 / 16, 1 / 4, 1., 4.]
SEEDS = [7, 13, 23, 37, 53]


class Readouts(NumpyEmeans):
    def begin(self, global_n, init_centroids):
        super().begin(global_n, init_centroids)
        self.shadows = {}
        for speed in SPEEDS:
            run = NumpyEmeans(self.d, self.k, nepochs=self.nepochs, seed=self.seed)
            run.begin(global_n, init_centroids)
            self.shadows[speed] = run

    def _apply_statistics(self, ids, sums, counts, objective, alpha, decay, replace_usage=False):
        super()._apply_statistics(ids, sums, counts, objective, alpha, decay, replace_usage)
        floor = self.recipe["alpha0"] * self.recipe["alpha_min_factor"]
        for speed, run in self.shadows.items():
            step = speed * floor
            run._apply_statistics(ids, sums, counts, 0., step, 1. - step)

    def _apply_splits(self, source, target):
        for run in self.shadows.values():
            run._apply_splits(source, target)
        return super()._apply_splits(source, target)

    def finalize(self):
        self.main_ema = self.current_centroids().copy()
        return super().finalize()


def mse(rows, centers):
    # Exact nearest-centroid reassignment, squared Euclidean error per row.
    total = 0.
    for chunk in np.array_split(rows, (len(rows) + 63) // 64):
        diff = chunk.astype(np.float64)[:, None] - centers[None]
        total += np.einsum("nkd,nkd->nk", diff, diff).min(axis=1).sum()
    return total / len(rows)


def compare(name, rows, ntrain, k, writer):
    # Separate construction, model selection and final evaluation. All five
    # seeds use the same split. k changes while the data stay fixed.
    order = np.random.default_rng(20261121).permutation(len(rows))
    train, val, test = np.split(rows[order], [ntrain, (len(rows) + ntrain) // 2])
    trials = []
    for seed in SEEDS:
        trainer = Readouts(rows.shape[1], k, nepochs=3, seed=seed, capture_assignments=False)
        closing, _ = trainer.train(train)
        if seed == SEEDS[0]:
            # Reading a batch or a split must never feed back into training.
            plain = NumpyEmeans(rows.shape[1], k, nepochs=3, seed=seed, capture_assignments=False)
            plain_closing, _ = plain.train(train)
            np.testing.assert_array_equal(closing, plain_closing)
            assert trainer.n_splits == plain.n_splits
            assert all(run.n_splits == trainer.n_splits for run in trainer.shadows.values())
        choices = {s: t.current_centroids().copy() for s, t in trainer.shadows.items()}
        trials.append((seed, trainer.main_ema, closing, choices))
    speed = min(SPEEDS, key=lambda s: np.median([mse(val, t[3][s]) for t in trials]))
    paired = []
    for seed, main, closing, variations in trials:
        started = perf_counter()
        _, native = clumppy.emeans(train, k, epochs=3., seed=seed)
        elapsed = perf_counter() - started
        results = {}
        for method, centers in [("clump_closing", native), ("faiss_main", main),
                                ("faiss_closing", closing), ("faiss_slow", variations[speed])]:
            loss = mse(test, centers)
            results[method] = loss
            writer.writerow([name, ntrain, k, seed, method, speed if method == "faiss_slow" else "",
                             mse(train, centers), mse(val, centers), loss,
                             elapsed if method == "clump_closing" else ""])
        paired.append(100 * (results["faiss_slow"] / results["faiss_closing"] - 1))
    print(f"{name} n/k={ntrain/k:g} speed={speed:g} median paired test delta="
          f"{np.median(paired):+.3f}% slow wins={sum(d < 0 for d in paired)}/{len(paired)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("target/word2vec-10k.f32"))
    parser.add_argument("--output", type=Path, default=Path("target/emeans-finalizers.csv"))
    args = parser.parse_args()
    load_data(args.data)
    words = np.fromfile(args.data, dtype="<f4").reshape(10_000, 200)
    digits = load_digits().data.astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["data", "ntrain", "k", "seed", "method", "selected_speed",
                         "train_mse", "validation_mse", "test_mse", "clump_fit_seconds"])
        for k in (8, 64):
            compare("digits", digits, 1024, k, writer)
        for k in (32, 256):
            compare("word2vec", words, 4096, k, writer)


if __name__ == "__main__":
    main()
