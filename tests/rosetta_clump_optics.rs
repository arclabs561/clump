//! Rosetta correctness fixture: clump OPTICS asserted against scikit-learn.
//!
//! Reference values in `fixtures/rosetta/clump_optics.json` come from
//! `gen_clump_optics.py` (their provenance). Core distances are compared per
//! point; the DBSCAN-like extraction at `eps` is compared as a partition with
//! an identical noise set. The processing order is not compared because tied
//! reachability values are broken differently (see the generator docstring).
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_clump_optics.py`.

use clump::{Euclidean, Optics, NOISE};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/rosetta/clump_optics.json");

#[derive(Deserialize)]
struct Fixture {
    min_samples: usize,
    eps: f64,
    points: Vec<Vec<f64>>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    core_distances: Vec<f64>,
    dbscan_labels: Vec<i64>, // -1 = noise (sklearn convention)
}

fn fit() -> (Fixture, clump::OpticsResult) {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let pts: Vec<Vec<f32>> = fx
        .points
        .iter()
        .map(|r| r.iter().map(|&x| x as f32).collect())
        .collect();
    let result = Optics::new(1.0e6, fx.min_samples)
        .fit(&pts)
        .expect("optics");
    (fx, result)
}

#[test]
fn rosetta_optics_core_distances_match_sklearn() {
    let (fx, result) = fit();
    let expected = &fx.expected.core_distances;
    assert_eq!(result.ordering.len(), expected.len());
    for (pos, &i) in result.ordering.iter().enumerate() {
        let got = f64::from(result.core_distances[pos]);
        let want = expected[i];
        assert!(
            (got - want).abs() <= 1e-5 * want.max(1.0),
            "core distance of point {i}: clump={got} sklearn={want}"
        );
    }
}

#[test]
fn rosetta_optics_extraction_partition_matches_sklearn() {
    let (fx, result) = fit();
    let labels = Optics::<Euclidean>::extract_clusters(&result, fx.eps as f32);
    let sk = &fx.expected.dbscan_labels;
    let n = labels.len();
    assert_eq!(n, sk.len(), "label count");

    for i in 0..n {
        assert_eq!(
            labels[i] == NOISE,
            sk[i] == -1,
            "noise disagreement at point {i}: clump={} sklearn={}",
            labels[i],
            sk[i]
        );
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let clump_same = labels[i] != NOISE && labels[i] == labels[j];
            let sk_same = sk[i] != -1 && sk[i] == sk[j];
            assert_eq!(clump_same, sk_same, "co-cluster disagreement for ({i},{j})");
        }
    }
}
