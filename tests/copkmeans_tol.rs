//! Without constraints, COP-Kmeans assigns every point to its nearest centre,
//! so from the same k-means++ seed it runs Lloyd's iterations and must stop
//! where `Kmeans` stops: both scale `tol` by the mean data variance only.
include!("fixtures/sklearn_kmeans/fixture.rs");

use clump::cluster::metrics::adjusted_rand_index;
use clump::{CopKmeans, Kmeans};

#[test]
fn unconstrained_copkmeans_matches_kmeans_from_the_same_seed() {
    let data: Vec<Vec<f32>> = fx::X.iter().map(|r| r.to_vec()).collect();
    let k = fx::KM_INIT.len();
    // A loose tol makes the stopping threshold matter: a k-times larger
    // threshold stops COP-Kmeans several iterations before Kmeans.
    let tol = 1e-2;
    for seed in 0..5 {
        let km = Kmeans::new(k)
            .with_seed(seed)
            .with_tol(tol)
            .fit(&data)
            .unwrap();
        let cop = CopKmeans::new(k)
            .with_seed(seed)
            .with_tol(tol)
            .fit_predict_constrained(&data, &[])
            .unwrap();
        let ari = adjusted_rand_index(&km.labels, &cop);
        assert!(
            ari > 0.9999,
            "seed {seed}: ARI {ari} between Kmeans ({} iters) and unconstrained COP-Kmeans",
            km.iters
        );
    }
}
