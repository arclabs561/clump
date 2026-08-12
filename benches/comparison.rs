//! Head-to-head comparison of clump vs linfa-clustering
//! for k-means and DBSCAN on identical synthetic data.

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use rand::prelude::*;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Shared synthetic data generators
// ---------------------------------------------------------------------------

struct SharedData {
    vecs: Vec<Vec<f32>>,
    array: Array2<f32>,
}

/// Generate one f32 dataset, then expose it in each library's input shape.
fn synth_data(n: usize, d: usize, seed: u64) -> SharedData {
    let mut rng = StdRng::seed_from_u64(seed);
    let vecs: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..d).map(|_| rng.random::<f32>()).collect())
        .collect();
    let array = Array2::from_shape_fn((n, d), |(row, col)| vecs[row][col]);

    SharedData { vecs, array }
}

/// Choose identical initial centroids for both implementations.
fn initial_centroids(data: &SharedData, k: usize) -> (Vec<Vec<f32>>, Array2<f32>) {
    let vecs = data.vecs.iter().take(k).cloned().collect::<Vec<_>>();
    let d = data.array.ncols();
    let array = Array2::from_shape_fn((k, d), |(row, col)| vecs[row][col]);
    (vecs, array)
}

fn assert_identical_inputs(data: &SharedData) {
    assert_eq!(data.vecs.len(), data.array.nrows());
    assert!(data
        .vecs
        .iter()
        .flatten()
        .copied()
        .eq(data.array.iter().copied()));
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

    let data = synth_data(n, d, seed);
    let (clump_centroids, linfa_centroids) = initial_centroids(&data, k);
    assert_identical_inputs(&data);

    // Keep the timing comparison tied to a quality oracle. Both fits start
    // from the same centroids; their final mean squared distances should agree.
    {
        use linfa::prelude::*;
        use linfa::DatasetBase;
        use linfa_clustering::{KMeans, KMeansInit};

        let clump_fit = clump::Kmeans::new(k)
            .with_max_iter(max_iter)
            .with_centroids(clump_centroids.clone())
            .fit(&data.vecs)
            .unwrap();
        let linfa_fit = KMeans::params(k)
            .max_n_iterations(max_iter as u64)
            .n_runs(1)
            .tolerance(1e-4)
            .init_method(KMeansInit::Precomputed(linfa_centroids.clone()))
            .fit(&DatasetBase::from(data.array.clone()))
            .unwrap();
        let clump_inertia = clump_fit.inertia_trace.last().unwrap() / n as f32;
        // The implementations use different convergence/update details, so
        // require comparable rather than bit-identical ten-iteration output.
        let allowed_delta = linfa_fit.inertia().abs().max(1e-5) * 0.05;
        assert!(
            (clump_inertia - linfa_fit.inertia()).abs() <= allowed_delta,
            "k-means mean inertia differs: clump={clump_inertia}, linfa={}",
            linfa_fit.inertia()
        );
    }

    // -- clump --
    group.bench_function("clump/n1000_d16_k10", |b| {
        b.iter(|| {
            clump::Kmeans::new(k)
                .with_max_iter(max_iter)
                .with_centroids(clump_centroids.clone())
                .fit(black_box(&data.vecs))
                .unwrap()
        })
    });

    // -- linfa --
    group.bench_function("linfa/n1000_d16_k10", |b| {
        use linfa::prelude::*;
        use linfa::DatasetBase;
        use linfa_clustering::{KMeans, KMeansInit};

        let dataset = DatasetBase::from(data.array.view());
        b.iter(|| {
            KMeans::params(k)
                .max_n_iterations(max_iter as u64)
                .n_runs(1)
                .tolerance(1e-4)
                .init_method(KMeansInit::Precomputed(linfa_centroids.clone()))
                .fit(black_box(&dataset))
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

    let data = synth_data(n, d, seed);
    assert_identical_inputs(&data);

    // -- clump --
    group.bench_function("clump/n1000_d16", |b| {
        b.iter(|| {
            clump::Dbscan::new(0.5_f32, min_pts)
                .fit_predict(black_box(&data.vecs))
                .unwrap()
        })
    });

    // -- linfa --
    group.bench_function("linfa/n1000_d16", |b| {
        use linfa::prelude::*;
        use linfa_clustering::Dbscan;

        b.iter(|| {
            Dbscan::params::<f32>(min_pts)
                .tolerance(0.5_f32)
                .transform(black_box(&data.array))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_kmeans_comparison, bench_dbscan_comparison);
criterion_main!(benches);
