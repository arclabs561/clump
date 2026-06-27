//! Rosetta correctness fixtures: clump DBSCAN and k-means asserted against
//! scikit-learn (the QUALITY tolerance class).
//!
//! Reference values in `fixtures/rosetta/clump_clustering.json` come from
//! `gen_clump.py` (their provenance).
//!
//! DBSCAN is deterministic given (eps, min_samples), so clump's labels must
//! match sklearn's exactly up to a label permutation (equivalently ARI = 1.0).
//! This test does not compare label values; it checks that the two labelings
//! induce the SAME partition: identical noise set, and for every pair of points
//! the co-cluster relation agrees. The data is three well-separated blobs plus
//! isolated noise, so the margin makes the result independent of boundary
//! conventions (self-counting in min_samples, < vs <= on eps).
//!
//! k-means uses different k-means++ RNG in clump (f32) vs sklearn (f64), so the
//! centroids are not identical and the inertia is compared within a small
//! percentage (2%), not exactly. On separable blobs both reach the same optimal
//! clustering, so the gap is small.
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_clump.py`.

use clump::{Dbscan, Kmeans, NOISE};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/rosetta/clump_clustering.json");

#[derive(Deserialize)]
struct Fixture {
    eps: f64,
    min_samples: usize,
    k: usize,
    blobs: Vec<Vec<f64>>,
    dbscan_points: Vec<Vec<f64>>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    dbscan_labels: Vec<i64>, // -1 = noise (sklearn convention)
    kmeans_inertia: f64,
}

fn to_f32_rows(rows: &[Vec<f64>]) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|r| r.iter().map(|&x| x as f32).collect())
        .collect()
}

#[test]
fn rosetta_dbscan_partition_matches_sklearn() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let pts = to_f32_rows(&fx.dbscan_points);
    let labels = Dbscan::new(fx.eps as f32, fx.min_samples)
        .fit_predict(&pts)
        .expect("dbscan");
    let sk = &fx.expected.dbscan_labels;
    let n = labels.len();
    assert_eq!(n, sk.len(), "label count");

    // Identical noise set.
    for i in 0..n {
        assert_eq!(
            labels[i] == NOISE,
            sk[i] == -1,
            "noise disagreement at point {i}: clump={} sklearn={}",
            labels[i],
            sk[i]
        );
    }

    // Identical partition: for every pair, the co-cluster relation agrees.
    // Noise points are co-clustered with nobody (including other noise).
    for i in 0..n {
        for j in (i + 1)..n {
            let clump_same = labels[i] != NOISE && labels[i] == labels[j];
            let sk_same = sk[i] != -1 && sk[i] == sk[j];
            assert_eq!(
                clump_same, sk_same,
                "co-cluster disagreement for pair ({i},{j})"
            );
        }
    }
}

#[test]
fn rosetta_kmeans_inertia_matches_sklearn() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let blobs = to_f32_rows(&fx.blobs);
    let fit = Kmeans::new(fx.k).with_seed(0).fit(&blobs).expect("kmeans");
    let inertia = fit.wcss(&blobs) as f64;
    let want = fx.expected.kmeans_inertia;

    let rel = (inertia - want).abs() / want;
    assert!(
        rel < 0.02,
        "kmeans inertia: clump={inertia} sklearn={want} relative_diff={rel}"
    );
}
