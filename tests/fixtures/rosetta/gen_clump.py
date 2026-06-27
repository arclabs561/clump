# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn", "numpy"]
# ///
"""Rosetta fixture generator for clump DBSCAN and k-means (QUALITY class).

Provenance for clump_clustering.json.

DBSCAN is deterministic given (eps, min_samples), so clump's labels must match
scikit-learn's exactly up to a label permutation (equivalently ARI = 1.0). The
data is three well-separated blobs plus isolated noise points, with eps chosen so
intra-blob distances are far below eps and inter-blob / noise distances far above
it. That margin makes the result independent of boundary conventions (whether a
point counts itself toward min_samples, whether the eps test is < or <=), so the
two implementations agree on the exact partition AND the noise set.

k-means uses different k-means++ RNG in clump (f32) vs sklearn (f64), so the
centroids are not identical. On separable blobs both reach the same optimal
clustering, so inertia (WCSS) is compared within a small percentage, not exactly.

Regenerate: uv run tests/fixtures/rosetta/gen_clump.py
"""

import json
import platform
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

SEED = 0
rng = np.random.default_rng(SEED)

centers = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
blobs = np.vstack([c + rng.normal(0.0, 0.1, size=(10, 2)) for c in centers])  # 30 x 2

# Isolated noise points, each far (> eps) from every blob and from each other.
noise = np.array([[5.0, 5.0], [20.0, 20.0], [-10.0, 5.0]])
dbscan_points = np.vstack([blobs, noise])  # 33 x 2

eps = 1.0
min_samples = 3
k = 3

db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit(dbscan_points).labels_.tolist()
km_inertia = float(KMeans(n_clusters=k, n_init=10, random_state=0).fit(blobs).inertia_)

fixture = {
    "provenance": {
        "generator": "gen_clump.py",
        "library": "scikit-learn",
        "sklearn_version": __import__("sklearn").__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "note": "DBSCAN: exact partition match (ARI=1). k-means: inertia within %.",
    },
    "eps": eps,
    "min_samples": min_samples,
    "k": k,
    "blobs": blobs.tolist(),
    "dbscan_points": dbscan_points.tolist(),
    "expected": {
        "dbscan_labels": db_labels,  # -1 = noise
        "kmeans_inertia": km_inertia,
    },
}

out = Path(__file__).parent / "clump_clustering.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
n_clusters = len({x for x in db_labels if x != -1})
print(f"dbscan: {n_clusters} clusters, {db_labels.count(-1)} noise points")
print(f"kmeans inertia {km_inertia:.6f}")
print(f"wrote {out}")
