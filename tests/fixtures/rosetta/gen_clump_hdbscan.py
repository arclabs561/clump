# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["hdbscan", "numpy<2", "scikit-learn"]
# ///
"""Reference HDBSCAN labels from the canonical McInnes implementation.

Two datasets:
- control: three well-separated blobs (shallow hierarchy) — any convention
  disagreement (e.g. core-distance off-by-one) shows here as label drift.
- nested: a >=3-level density hierarchy engineered so excess-of-mass
  selection must propagate subtree stabilities through internal clusters
  (two tight blobs inside a loose region, that loose region paired with a
  third moderate blob, plus a distant fourth cluster and sparse noise).
  This is the trigger regime for traversal-order bugs.
"""
import json

import numpy as np
import hdbscan

rng = np.random.default_rng(42)


def blob(center, scale, n):
    return rng.normal(loc=center, scale=scale, size=(n, 2))


datasets = {}

# Control: three separated blobs.
control = np.vstack(
    [
        blob([0.0, 0.0], 0.3, 40),
        blob([10.0, 0.0], 0.3, 40),
        blob([5.0, 9.0], 0.3, 40),
    ]
)
datasets["control"] = (control, 5, 5)

# Nested: tight A1, A2 inside loose region A; moderate B nearby; far C; noise.
nested = np.vstack(
    [
        blob([0.0, 0.0], 0.15, 30),   # A1 tight
        blob([1.6, 0.0], 0.15, 30),   # A2 tight (close to A1: joint loose A)
        blob([0.8, 0.9], 0.55, 25),   # A halo (binds A1+A2 into loose A)
        blob([6.0, 0.5], 0.35, 30),   # B moderate
        blob([20.0, 20.0], 0.3, 25),  # C far
        rng.uniform(low=-3, high=24, size=(15, 2)),  # sparse noise
    ]
)
datasets["nested"] = (nested, 8, 5)

out = {"source": "mcinnes hdbscan (pip 'hdbscan'), EOM selection", "cases": []}
for name, (X, mcs, ms) in datasets.items():
    X = np.round(X.astype(np.float64), 6)
    cl = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
    ).fit(X)
    labels = cl.labels_.tolist()
    n_clusters = len(set(l for l in labels if l >= 0))
    print(f"{name}: n={len(X)} clusters={n_clusters} noise={sum(1 for l in labels if l < 0)}")
    out["cases"].append(
        {
            "name": name,
            "min_cluster_size": mcs,
            "min_samples": ms,
            "points": X.tolist(),
            "labels": labels,
        }
    )

import pathlib
with open(pathlib.Path(__file__).parent / "clump_hdbscan.json", "w") as f:
    json.dump(out, f)
print("wrote clump_hdbscan.json")
