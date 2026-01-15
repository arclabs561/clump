use clump::cluster::{Clustering, Kmeans};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;

fn bench_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans");

    // Generate synthetic data
    let mut rng = StdRng::seed_from_u64(42);
    let n = 1000;
    let d = 16;
    let k = 10;

    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..d).map(|_| rng.random::<f32>()).collect())
        .collect();

    group.bench_function("fit_predict_n1000_d16_k10", |b| {
        b.iter(|| {
            let model = Kmeans::new(k).with_max_iter(10).with_seed(42);
            model.fit_predict(black_box(&data)).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_kmeans);
criterion_main!(benches);
