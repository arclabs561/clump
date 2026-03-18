//! Head-to-head comparison of clump vs linfa-clustering
//! for k-means and DBSCAN on identical synthetic data.

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::prelude::*;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Shared synthetic data generators
// ---------------------------------------------------------------------------

/// Generate data as Vec<Vec<f32>> for clump (which takes `&[Vec<f32>]`).
fn synth_vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.random::<f32>()).collect())
        .collect()
}

/// Generate data as ndarray Array2<f64> for linfa.
fn synth_array2(n: usize, d: usize, seed: u64) -> Array2<f64> {
    use ndarray_rand::rand::SeedableRng;
    let rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(seed);
    Array2::random_using((n, d), Uniform::new(0.0, 1.0), &mut rng.clone())
}

// ---------------------------------------------------------------------------
// K-Means comparison
// ---------------------------------------------------------------------------

fn bench_kmeans_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_comparison");

    let n = 1000;
    let d = 16;
    let k = 10;
    let seed = 42u64;
    let max_iter = 10;

    let clump_data = synth_vecs(n, d, seed);
    let linfa_data = synth_array2(n, d, seed);

    // -- clump --
    group.bench_function("clump/n1000_d16_k10", |b| {
        b.iter(|| {
            clump::Kmeans::new(k)
                .with_max_iter(max_iter)
                .with_seed(seed)
                .fit_predict(black_box(&clump_data))
                .unwrap()
        })
    });

    // -- linfa --
    group.bench_function("linfa/n1000_d16_k10", |b| {
        use linfa::prelude::*;
        use linfa::DatasetBase;
        use linfa_clustering::KMeans;

        b.iter(|| {
            let dataset = DatasetBase::from(black_box(linfa_data.clone()));
            KMeans::params(k)
                .max_n_iterations(max_iter as u64)
                .n_runs(1)
                .tolerance(1e-4)
                .fit(&dataset)
                .unwrap()
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// DBSCAN comparison
// ---------------------------------------------------------------------------

fn bench_dbscan_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbscan_comparison");

    let n = 1000;
    let d = 16;
    let seed = 42u64;
    let min_pts = 5;

    let clump_data = synth_vecs(n, d, seed);
    let linfa_data = synth_array2(n, d, seed);

    // -- clump --
    group.bench_function("clump/n1000_d16", |b| {
        b.iter(|| {
            clump::Dbscan::new(0.5_f32, min_pts)
                .fit_predict(black_box(&clump_data))
                .unwrap()
        })
    });

    // -- linfa --
    group.bench_function("linfa/n1000_d16", |b| {
        use linfa::prelude::*;
        use linfa_clustering::Dbscan;

        b.iter(|| {
            Dbscan::params::<f64>(min_pts)
                .tolerance(0.5_f64)
                .transform(black_box(&linfa_data))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_kmeans_comparison, bench_dbscan_comparison);
criterion_main!(benches);
