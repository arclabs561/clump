# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn", "numpy"]
# ///
"""Rosetta fixture generator for clump OPTICS.

Provenance for clump_optics.json.

Core distances depend only on the data and min_samples, so they are compared
point by point. scikit-learn's `min_samples` counts the point itself, which is
the same convention as clump's `min_pts`.

The processing order is not compared: many reachability values tie exactly
(a neighbor's reachability is often the expanding point's core distance), and
the two implementations break those ties differently. The DBSCAN-like
extraction at an eps inside the gap between blobs is order-independent on this
data, so its partition and noise set are compared instead.

Regenerate: uv run tests/fixtures/rosetta/gen_clump_optics.py
"""

import json
import platform
from pathlib import Path

import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_dbscan

SEED = 0
rng = np.random.default_rng(SEED)

# Three blobs of different density plus isolated noise points.
points = np.vstack(
    [
        rng.normal((0.0, 0.0), 0.2, size=(30, 2)),
        rng.normal((8.0, 0.0), 0.4, size=(30, 2)),
        rng.normal((0.0, 8.0), 0.6, size=(30, 2)),
        np.array([[4.0, 4.0], [12.0, 9.0], [-6.0, -5.0]]),
    ]
)

min_samples = 5
eps = 2.0

optics = OPTICS(min_samples=min_samples, max_eps=np.inf, metric="euclidean").fit(points)
labels = cluster_optics_dbscan(
    reachability=optics.reachability_,
    core_distances=optics.core_distances_,
    ordering=optics.ordering_,
    eps=eps,
)

fixture = {
    "provenance": {
        "generator": "gen_clump_optics.py",
        "library": "scikit-learn",
        "sklearn_version": __import__("sklearn").__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "note": "core distances per point; extraction partition at eps",
    },
    "min_samples": min_samples,
    "eps": eps,
    "points": points.tolist(),
    "expected": {
        "core_distances": optics.core_distances_.tolist(),
        "dbscan_labels": labels.tolist(),  # -1 = noise
    },
}

out = Path(__file__).parent / "clump_optics.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
n_clusters = len({x for x in labels.tolist() if x != -1})
print(f"optics extraction: {n_clusters} clusters, {labels.tolist().count(-1)} noise")
print(f"wrote {out}")
