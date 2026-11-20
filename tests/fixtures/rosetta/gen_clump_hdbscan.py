# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["hdbscan", "numpy<2", "scikit-learn"]
# ///
"""Reference HDBSCAN labels from the canonical McInnes implementation.

Datasets:
- control: three well-separated blobs (shallow hierarchy) — any convention
  disagreement (e.g. core-distance off-by-one) shows here as label drift.
- uneven / single_blob: cases where the root of the hierarchy would be
  selected if it were a candidate (see the comments at their definitions).
- nested: a >=3-level density hierarchy engineered so excess-of-mass
  selection must propagate subtree stabilities through internal clusters
  (two tight blobs inside a loose region, that loose region paired with a
  third moderate blob, plus a distant fourth cluster and sparse noise).
  This is the trigger regime for traversal-order bugs.
"""
import json

import numpy as np
import hdbscan
from sklearn.datasets import make_blobs

rng = np.random.default_rng(42)


def blob(center, scale, n):
    return rng.normal(loc=center, scale=scale, size=(n, 2))


datasets = {}
# Cases generated with algorithm="generic" (exact mutual-reachability MST).
# The library's default picks a Boruvka variant, and its labels differ from
# the exact path on some inputs: with hdbscan 0.8.44, "nested" below comes
# back as 4 clusters / 17 noise by default and 5 clusters / 26 noise with
# "generic". The committed labels for "control" and "nested" predate this
# note and were kept as they were; new cases use the exact path.
EXACT = set()

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

# Uneven: three overlapping blobs of different spread. The root's stability
# exceeds the sum of its children's here, so an implementation that lets the
# root compete in excess-of-mass selection returns one cluster and no noise.
# Built with make_blobs, which has its own random state, so the draws for
# the cases above do not change.
uneven, _ = make_blobs(
    n_samples=100, centers=3, cluster_std=[0.5, 0.9, 1.3], random_state=0
)
datasets["uneven"] = (uneven, 8, 5)
EXACT.add("uneven")

# Single blob: with allow_single_cluster=False the reference labels every
# point as noise, because the only candidate cluster is the root.
single = blob([0.0, 0.0], 0.5, 40)
datasets["single_blob"] = (single, 10, 5)
EXACT.add("single_blob")

def ambiguous_points(X, labels, min_samples):
    """Points whose cheapest mutual-reachability edge is tied between
    neighbours that the reference puts in different clusters.

    The minimum spanning tree is not unique at such a point: which neighbour
    it attaches to depends on tie-breaking in the MST construction, and so
    does its label. Mutual reachability makes these ties exact, not merely
    close, because several edges equal the point's own core distance. The
    comparison skips these points.
    """
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    core = np.sort(D, axis=1)[:, min_samples]  # k-th neighbour, self excluded
    M = np.maximum(np.maximum(core[:, None], core[None]), D)
    np.fill_diagonal(M, np.inf)
    out = []
    for p in range(len(X)):
        tied = np.flatnonzero(M[p] <= M[p].min() + 1e-9)
        if len({labels[j] for j in tied if labels[j] >= 0}) > 1:
            out.append(p)
    return out


out = {"source": "mcinnes hdbscan (pip 'hdbscan'), EOM selection", "cases": []}
for name, (X, mcs, ms) in datasets.items():
    X = np.round(X.astype(np.float64), 6)
    cl = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
        **({"algorithm": "generic"} if name in EXACT else {}),
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
            **(
                {"algorithm": "generic", "ambiguous": ambiguous_points(X, labels, ms)}
                if name in EXACT
                else {}
            ),
        }
    )

import pathlib
with open(pathlib.Path(__file__).parent / "clump_hdbscan.json", "w") as f:
    json.dump(out, f)
print("wrote clump_hdbscan.json")
