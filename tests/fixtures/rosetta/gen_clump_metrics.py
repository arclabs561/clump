# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn", "numpy"]
# ///
"""Rosetta fixture generator for clump cluster-evaluation metrics.

Provenance for clump_metrics.json.

Grounds four deterministic metrics against scikit-learn:
  - silhouette_score        (sklearn.metrics.silhouette_score, euclidean)
  - calinski_harabasz       (sklearn.metrics.calinski_harabasz_score)
  - davies_bouldin          (sklearn.metrics.davies_bouldin_score)
  - adjusted_rand_index     (sklearn.metrics.adjusted_rand_score)

Two tolerance classes:

EXACT class (adjusted_rand_index). ARI is pure integer combinatorics over the
contingency table, evaluated in f64 on both sides with no floating-point input
data, so agreement is at f64-noise level (1e-9).

TIGHT class (silhouette / Calinski-Harabasz / Davies-Bouldin). clump computes
pairwise / point-to-centroid distances in f32 (its public API is f32-only) and
accumulates in f64; scikit-learn works in f64 throughout. The fixture stores X
in f64 and the Rust side casts to f32, so the gap is bounded by f32 rounding of
the coordinates and distances, not by any formula difference. The comparison is
therefore relative-1e-4, tight enough that a genuine formula divergence (which
would be O(1) relative) fails loudly, loose enough to absorb f32 noise.

clump's calinski_harabasz and davies_bouldin take the cluster centroids as an
argument, whereas the sklearn functions compute them internally as the
arithmetic mean of each cluster. To make the comparison test the FORMULA (not a
centroid-computation difference), the fixture stores centroids computed by numpy
as those same cluster means; sklearn's internal centroids are identical to them,
and the Rust side passes the f32 cast of the stored centroids.

Datasets (fixed seeds, frozen labels -- the test never re-runs any estimator):
  well_separated: 3 tight blobs, perfect labels_pred (ARI = 1).
  overlapping:    3 overlapping blobs, labels_pred = KMeans result (ARI < 1,
                  mid-regime silhouette). The KMeans labels are frozen into the
                  JSON, so cross-version KMeans drift cannot affect the test.
  small_cluster:  two size-15 blobs + one size-3 blob, perfect labels_pred
                  (imbalanced sizes stress the size-weighting in CH / DB). The
                  small cluster is size 3, not 1, so it stays clear of the
                  singleton convention gap (clump drops singleton clusters from
                  the silhouette mean; sklearn scores them 0 and keeps them).

Regenerate: uv run tests/fixtures/rosetta/gen_clump_metrics.py
"""

import json
import platform
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def blobs(centers, std, n_per, seed):
    rng = np.random.default_rng(seed)
    parts = [c + rng.normal(0.0, std, size=(n, 2)) for c, n in zip(centers, n_per)]
    labels = np.concatenate([np.full(n, i) for i, n in enumerate(n_per)])
    return np.vstack(parts), labels


def centroids_of(x, labels, k):
    return np.vstack([x[labels == c].mean(axis=0) for c in range(k)])


def dataset(name, x, labels_true, labels_pred):
    k = int(labels_pred.max()) + 1
    cents = centroids_of(x, labels_pred, k)
    return {
        "name": name,
        "x": x.tolist(),
        "labels_true": labels_true.astype(int).tolist(),
        "labels_pred": labels_pred.astype(int).tolist(),
        "centroids": cents.tolist(),
        "expected": {
            "silhouette": float(silhouette_score(x, labels_pred, metric="euclidean")),
            "calinski_harabasz": float(calinski_harabasz_score(x, labels_pred)),
            "davies_bouldin": float(davies_bouldin_score(x, labels_pred)),
            "adjusted_rand": float(adjusted_rand_score(labels_true, labels_pred)),
        },
    }


datasets = []

# well_separated: perfect labels -> ARI = 1, high silhouette, huge CH, tiny DB.
x_a, y_a = blobs([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)], 0.35, [12, 12, 12], seed=0)
datasets.append(dataset("well_separated", x_a, y_a, y_a.copy()))

# overlapping: labels_pred from KMeans -> ARI < 1, mid-regime metrics.
x_b, y_b = blobs([(0.0, 0.0), (2.5, 0.0), (1.25, 2.2)], 1.0, [20, 20, 20], seed=1)
pred_b = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(x_b)
datasets.append(dataset("overlapping", x_b, y_b, pred_b))

# small_cluster: imbalanced sizes (15, 15, 3), perfect labels.
x_c, y_c = blobs([(0.0, 0.0), (8.0, 0.0), (4.0, 6.0)], 0.4, [15, 15, 3], seed=2)
datasets.append(dataset("small_cluster", x_c, y_c, y_c.copy()))

fixture = {
    "provenance": {
        "generator": "gen_clump_metrics.py",
        "library": "scikit-learn",
        "sklearn_version": __import__("sklearn").__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "note": (
            "ARI: EXACT (1e-9, integer combinatorics). "
            "silhouette/CH/DB: TIGHT (rel 1e-4, f32 distances vs f64). "
            "centroids are numpy cluster means == sklearn's internal centroids."
        ),
    },
    "datasets": datasets,
}

out = Path(__file__).parent / "clump_metrics.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")

for d in datasets:
    e = d["expected"]
    sizes = [int((np.array(d["labels_pred"]) == c).sum()) for c in range(int(np.array(d["labels_pred"]).max()) + 1)]
    print(
        f"{d['name']:>14}: sizes={sizes} "
        f"sil={e['silhouette']:.6f} ch={e['calinski_harabasz']:.4f} "
        f"db={e['davies_bouldin']:.6f} ari={e['adjusted_rand']:.6f}"
    )
print(f"wrote {out}")
