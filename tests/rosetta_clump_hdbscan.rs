//! Rosetta correctness fixtures: clump HDBSCAN asserted against the McInnes
//! `hdbscan` reference implementation (the QUALITY tolerance class).
//!
//! Reference values in `fixtures/rosetta/clump_hdbscan.json` come from
//! `gen_clump_hdbscan.py` (their provenance): excess-of-mass selection,
//! euclidean metric, `allow_single_cluster=False`.
//!
//! Two datasets pin two failure classes this suite previously could not see:
//! `control` (three well-separated blobs) caught the reverse-id selection
//! traversal re-selecting deselected leaf clusters (three blobs came back as
//! six); `nested` (a three-level density hierarchy: two tight blobs inside a
//! loose region, a sibling blob, a distant cluster, sparse noise) exercises
//! subtree-stability propagation through internal clusters, the regime where
//! a non-topological processing order under-counts child stability.
//!
//! Labels are compared as partitions (identical noise set + pairwise
//! co-cluster agreement), not by value, matching `rosetta_clump.rs`.
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_clump_hdbscan.py`.

use clump::Hdbscan;

const NOISE: usize = usize::MAX;

fn load_cases() -> serde_json::Value {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/rosetta/clump_hdbscan.json"
    ))
    .expect("fixture readable");
    serde_json::from_str(&raw).expect("fixture parses")
}

fn assert_same_partition(name: &str, got: &[usize], reference: &[i64]) {
    assert_eq!(got.len(), reference.len(), "{name}: length mismatch");
    let n = got.len();
    for i in 0..n {
        let got_noise = got[i] == NOISE;
        let ref_noise = reference[i] < 0;
        assert_eq!(
            got_noise, ref_noise,
            "{name}: noise disagreement at point {i} (got {:?}, ref {})",
            got[i], reference[i]
        );
    }
    for i in 0..n {
        if got[i] == NOISE {
            continue;
        }
        for j in (i + 1)..n {
            if got[j] == NOISE {
                continue;
            }
            let same_got = got[i] == got[j];
            let same_ref = reference[i] == reference[j];
            assert_eq!(
                same_got, same_ref,
                "{name}: co-cluster disagreement for points {i},{j} \
                 (got {} vs {}, ref {} vs {})",
                got[i], got[j], reference[i], reference[j]
            );
        }
    }
}

#[test]
fn hdbscan_matches_reference_implementation() {
    let v = load_cases();
    let cases = v["cases"].as_array().expect("cases");
    assert!(cases.len() >= 2, "fixture unexpectedly small");
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let mcs = case["min_cluster_size"].as_u64().expect("mcs") as usize;
        let ms = case["min_samples"].as_u64().expect("ms") as usize;
        let points: Vec<Vec<f32>> = case["points"]
            .as_array()
            .expect("points")
            .iter()
            .map(|p| {
                p.as_array()
                    .expect("point")
                    .iter()
                    .map(|x| x.as_f64().expect("coord") as f32)
                    .collect()
            })
            .collect();
        let reference: Vec<i64> = case["labels"]
            .as_array()
            .expect("labels")
            .iter()
            .map(|x| x.as_i64().expect("label"))
            .collect();

        let got = Hdbscan::new()
            .with_min_cluster_size(mcs)
            .with_min_samples(ms)
            .fit_predict(&points)
            .expect("fit_predict");

        assert_same_partition(name, &got, &reference);
    }
}
