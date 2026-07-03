//! Rosetta correctness fixtures: clump cluster-evaluation metrics asserted
//! against scikit-learn reference values.
//!
//! Reference values in `fixtures/rosetta/clump_metrics.json` come from
//! `gen_clump_metrics.py` (their provenance). Four metrics are grounded:
//! `silhouette_score`, `calinski_harabasz`, `davies_bouldin`, and
//! `adjusted_rand_index`.
//!
//! Two tolerance classes:
//!
//! - `adjusted_rand_index` is EXACT: pure integer combinatorics over the
//!   contingency table in f64, no floating-point input data, so it agrees with
//!   sklearn at f64-noise level (1e-9). Grounding it also hardens the sibling
//!   `rosetta_clump.rs` DBSCAN test, which uses ARI = 1 as its comparator
//!   without itself checking ARI against a reference.
//! - `silhouette_score` / `calinski_harabasz` / `davies_bouldin` are TIGHT:
//!   clump computes distances in f32 (its API is f32-only) and accumulates in
//!   f64, while sklearn works in f64 throughout. The fixture stores X in f64
//!   and this test casts to f32, so the gap is bounded by f32 rounding of the
//!   coordinates and distances, not by any formula difference. Relative 1e-4 is
//!   tight enough that a real formula divergence (O(1) relative) fails loudly.
//!
//! clump's `calinski_harabasz` and `davies_bouldin` take the cluster centroids
//! as an argument; the sklearn functions compute them internally as each
//! cluster's arithmetic mean. The fixture stores centroids computed as those
//! same means (identical to sklearn's internal ones), and this test passes the
//! f32 cast of them, so the comparison tests the metric formula rather than a
//! centroid-computation difference.
//!
//! Datasets: well_separated (perfect labels, ARI = 1), overlapping (labels from
//! KMeans, ARI < 1, mid-regime silhouette), small_cluster (sizes 15/15/3,
//! perfect labels, imbalanced sizes stressing the size-weighting in CH / DB).
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_clump_metrics.py`.

use clump::cluster::metrics::{
    adjusted_rand_index, calinski_harabasz, davies_bouldin, silhouette_score,
};
use clump::Euclidean;
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/rosetta/clump_metrics.json");

#[derive(Deserialize)]
struct Fixture {
    datasets: Vec<Dataset>,
}

#[derive(Deserialize)]
struct Dataset {
    name: String,
    x: Vec<Vec<f64>>,
    labels_true: Vec<usize>,
    labels_pred: Vec<usize>,
    centroids: Vec<Vec<f64>>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    silhouette: f64,
    calinski_harabasz: f64,
    davies_bouldin: f64,
    adjusted_rand: f64,
}

fn to_f32_rows(rows: &[Vec<f64>]) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|r| r.iter().map(|&x| x as f32).collect())
        .collect()
}

/// EXACT-class tolerance: ARI is deterministic integer combinatorics, so
/// agreement with sklearn should be at f64-noise level. A loose tolerance would
/// hide a real divergence, which is the whole point.
fn close_exact(got: f64, want: f64, label: &str) {
    let tol = 1e-9 * (1.0 + want.abs());
    let diff = (got - want).abs();
    assert!(
        diff <= tol,
        "{label}: clump={got} sklearn={want} diff={diff} tol={tol}"
    );
}

/// TIGHT-class tolerance: f32 distance arithmetic vs sklearn's f64. Bounded by
/// f32 rounding of coordinates and distances, not by any formula difference, so
/// relative 1e-4 still fails loudly on a genuine (O(1) relative) divergence.
fn close_tight(got: f64, want: f64, label: &str) {
    let tol = 1e-4 * (1.0 + want.abs());
    let diff = (got - want).abs();
    assert!(
        diff <= tol,
        "{label}: clump={got} sklearn={want} diff={diff} tol={tol}"
    );
}

#[test]
fn rosetta_metrics_match_sklearn() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    assert!(!fx.datasets.is_empty(), "fixture has no datasets");

    for ds in &fx.datasets {
        let x = to_f32_rows(&ds.x);
        let centroids = to_f32_rows(&ds.centroids);
        let e = &ds.expected;
        let name = &ds.name;

        let sil = silhouette_score(&x, &ds.labels_pred, &Euclidean) as f64;
        close_tight(sil, e.silhouette, &format!("{name}/silhouette"));

        let ch = calinski_harabasz(&x, &ds.labels_pred, &centroids) as f64;
        close_tight(
            ch,
            e.calinski_harabasz,
            &format!("{name}/calinski_harabasz"),
        );

        let db = davies_bouldin(&x, &ds.labels_pred, &centroids, &Euclidean) as f64;
        close_tight(db, e.davies_bouldin, &format!("{name}/davies_bouldin"));

        let ari = adjusted_rand_index(&ds.labels_true, &ds.labels_pred);
        close_exact(ari, e.adjusted_rand, &format!("{name}/adjusted_rand"));
    }
}
