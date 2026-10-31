//! k-means from fixed initial centres must match scikit-learn's Lloyd run
//! with the same tolerance (tests/fixtures/sklearn_kmeans/fixture.rs).
include!("fixtures/sklearn_kmeans/fixture.rs");

use clump::cluster::metrics::adjusted_rand_index;
use clump::Kmeans;

fn x() -> Vec<Vec<f32>> {
    fx::X.iter().map(|r| r.to_vec()).collect()
}

fn sk(l: &[i64]) -> Vec<usize> {
    l.iter().map(|&v| v as usize).collect()
}

#[test]
fn kmeans_fixed_init_matches_sklearn_lloyd() {
    let init: Vec<Vec<f32>> = fx::KM_INIT.iter().map(|r| r.to_vec()).collect();
    let fit = Kmeans::new(init.len())
        .with_centroids(init)
        .with_seed(0)
        .fit(&x())
        .unwrap();
    let data = x();
    let inertia: f64 = data
        .iter()
        .zip(&fit.labels)
        .map(|(p, &l)| {
            p.iter()
                .zip(&fit.centroids[l])
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
        })
        .sum();
    let ari = adjusted_rand_index(&fit.labels, &sk(fx::KM_LABELS));
    assert!(
        (inertia - fx::KM_INERTIA).abs() <= 1e-4 * fx::KM_INERTIA && ari > 0.9999,
        "clump: iters {} inertia {inertia}; sklearn (tol=1e-4): iters {} inertia {}; ARI {ari}",
        fit.iters,
        fx::KM_ITER,
        fx::KM_INERTIA
    );
}
