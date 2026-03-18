use clump::{Constraint, CopKmeans, Dbscan, Hdbscan, Kmeans, MiniBatchKmeans};
use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use std::hint::black_box;

fn synth_data(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.random::<f32>()).collect())
        .collect()
}

fn bench_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans");

    // Original benchmark
    let data = synth_data(1000, 16, 42);
    group.bench_function("n1000_d16_k10", |b| {
        b.iter(|| {
            Kmeans::new(10)
                .with_max_iter(10)
                .with_seed(42)
                .fit_predict(black_box(&data))
                .unwrap()
        })
    });

    // Larger N
    let data_5k = synth_data(5000, 16, 42);
    group.bench_function("n5000_d16_k10", |b| {
        b.iter(|| {
            Kmeans::new(10)
                .with_max_iter(10)
                .with_seed(42)
                .fit_predict(black_box(&data_5k))
                .unwrap()
        })
    });

    // Large k
    let data_10k = synth_data(10000, 16, 42);
    group.bench_function("n10000_d16_k100", |b| {
        b.iter(|| {
            Kmeans::new(100)
                .with_max_iter(10)
                .with_seed(42)
                .fit_predict(black_box(&data_10k))
                .unwrap()
        })
    });

    // High dimension
    let data_hd = synth_data(1000, 128, 42);
    group.bench_function("n1000_d128_k10", |b| {
        b.iter(|| {
            Kmeans::new(10)
                .with_max_iter(10)
                .with_seed(42)
                .fit_predict(black_box(&data_hd))
                .unwrap()
        })
    });

    // Large scale
    let data_50k = synth_data(50000, 16, 42);
    group.bench_function("n50000_d16_k100", |b| {
        b.iter(|| {
            Kmeans::new(100)
                .with_max_iter(10)
                .with_seed(42)
                .fit_predict(black_box(&data_50k))
                .unwrap()
        })
    });

    group.finish();
}

fn bench_dbscan(c: &mut Criterion) {
    let mut group = c.benchmark_group("dbscan");

    let data = synth_data(1000, 16, 42);
    group.bench_function("n1000_d16", |b| {
        b.iter(|| Dbscan::new(0.5, 5).fit_predict(black_box(&data)).unwrap())
    });

    let data_2k = synth_data(2000, 16, 42);
    group.bench_function("n2000_d16", |b| {
        b.iter(|| {
            Dbscan::new(0.5, 5)
                .fit_predict(black_box(&data_2k))
                .unwrap()
        })
    });

    let data_10k = synth_data(10000, 16, 42);
    group.bench_function("n10000_d16", |b| {
        b.iter(|| {
            Dbscan::new(0.5, 5)
                .fit_predict(black_box(&data_10k))
                .unwrap()
        })
    });

    // Low-d large-n: grid index.
    let data_50k_d3 = synth_data(50000, 3, 42);
    group.bench_function("n50000_d3", |b| {
        b.iter(|| {
            Dbscan::new(0.1, 5)
                .fit_predict(black_box(&data_50k_d3))
                .unwrap()
        })
    });

    // High-d beyond matrix limit: VP-tree path.
    let data_15k_d16 = synth_data(15000, 16, 42);
    group.bench_function("n15000_d16", |b| {
        b.iter(|| {
            Dbscan::new(0.5, 5)
                .fit_predict(black_box(&data_15k_d16))
                .unwrap()
        })
    });

    group.finish();
}

fn bench_hdbscan(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdbscan");

    let data = synth_data(500, 16, 42);
    group.bench_function("n500_d16", |b| {
        b.iter(|| Hdbscan::new().fit_predict(black_box(&data)).unwrap())
    });

    let data_1k = synth_data(1000, 16, 42);
    group.bench_function("n1000_d16", |b| {
        b.iter(|| Hdbscan::new().fit_predict(black_box(&data_1k)).unwrap())
    });

    group.finish();
}

fn bench_minibatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("minibatch_kmeans");

    let batches: Vec<Vec<Vec<f32>>> = (0..5).map(|i| synth_data(200, 16, 42 + i)).collect();

    group.bench_function("5x200_d16_k10", |b| {
        b.iter(|| {
            let mut mbk = MiniBatchKmeans::new(10).with_seed(42);
            for batch in &batches {
                mbk.update_batch(black_box(batch)).unwrap();
            }
        })
    });

    group.finish();
}

fn bench_cop_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("cop_kmeans");

    let data = synth_data(500, 16, 42);
    let constraints = vec![
        Constraint::MustLink(0, 1),
        Constraint::MustLink(2, 3),
        Constraint::CannotLink(0, 4),
    ];

    group.bench_function("n500_d16_k5", |b| {
        b.iter(|| {
            CopKmeans::new(5)
                .with_max_iter(10)
                .with_seed(42)
                .fit_predict_constrained(black_box(&data), black_box(&constraints))
                .unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_kmeans,
    bench_dbscan,
    bench_hdbscan,
    bench_minibatch,
    bench_cop_kmeans,
);
criterion_main!(benches);
