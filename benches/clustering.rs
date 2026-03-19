use clump::{
    Constraint, CopKmeans, CorrelationClustering, Dbscan, EVoC, EVoCParams, Hdbscan, Kmeans,
    MiniBatchKmeans, SignedEdge,
};
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
    group.bench_function("n50000_d16_k10", |b| {
        b.iter(|| {
            Kmeans::new(10)
                .with_max_iter(10)
                .with_seed(42)
                .fit_predict(black_box(&data_50k))
                .unwrap()
        })
    });

    group.bench_function("n50000_d16_k100", |b| {
        b.iter(|| {
            Kmeans::new(100)
                .with_max_iter(10)
                .with_seed(42)
                .fit_predict(black_box(&data_50k))
                .unwrap()
        })
    });

    let data_100k = synth_data(100000, 16, 42);
    group.bench_function("n100000_d16_k100", |b| {
        b.iter(|| {
            Kmeans::new(100)
                .with_max_iter(10)
                .with_seed(42)
                .fit_predict(black_box(&data_100k))
                .unwrap()
        })
    });

    // High magnitude data (simulates un-normalized embeddings).
    let data_hm: Vec<Vec<f32>> = synth_data(5000, 128, 42)
        .into_iter()
        .map(|v| v.into_iter().map(|x| x * 1000.0).collect())
        .collect();
    group.bench_function("n5000_d128_k10_highmag", |b| {
        b.iter(|| {
            Kmeans::new(10)
                .with_max_iter(10)
                .with_seed(42)
                .fit_predict(black_box(&data_hm))
                .unwrap()
        })
    });

    let data_200k = synth_data(200000, 128, 42);
    group.bench_function("n200000_d128_k50", |b| {
        b.iter(|| {
            Kmeans::new(50)
                .with_max_iter(5)
                .with_seed(42)
                .fit_predict(black_box(&data_200k))
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

    let data_2k = synth_data(2000, 16, 42);
    group.bench_function("n2000_d16", |b| {
        b.iter(|| Hdbscan::new().fit_predict(black_box(&data_2k)).unwrap())
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

fn bench_optics(c: &mut Criterion) {
    let mut group = c.benchmark_group("optics");

    let data = synth_data(1000, 16, 42);
    group.bench_function("n1000_d16", |b| {
        b.iter(|| clump::Optics::new(1.0, 5).fit(black_box(&data)).unwrap())
    });

    let data_2k = synth_data(2000, 16, 42);
    group.bench_function("n2000_d16", |b| {
        b.iter(|| clump::Optics::new(1.0, 5).fit(black_box(&data_2k)).unwrap())
    });

    // Extract clusters from OPTICS result.
    let data_500 = synth_data(500, 16, 42);
    group.bench_function("n500_d16_extract", |b| {
        let result = clump::Optics::new(1.0, 5).fit(&data_500).unwrap();
        b.iter(|| clump::Optics::<clump::Euclidean>::extract_clusters(black_box(&result), 0.5))
    });

    group.finish();
}

fn bench_metrics(c: &mut Criterion) {
    use clump::cluster::metrics;
    use clump::Euclidean;

    let mut group = c.benchmark_group("metrics");

    let data = synth_data(500, 16, 42);
    let labels = Kmeans::new(5)
        .with_seed(42)
        .with_max_iter(10)
        .fit_predict(&data)
        .unwrap();
    let fit = Kmeans::new(5)
        .with_seed(42)
        .with_max_iter(10)
        .fit(&data)
        .unwrap();

    group.bench_function("silhouette_n500_d16_k5", |b| {
        b.iter(|| metrics::silhouette_score(black_box(&data), black_box(&labels), &Euclidean))
    });

    group.bench_function("calinski_harabasz_n500_d16_k5", |b| {
        b.iter(|| {
            metrics::calinski_harabasz(
                black_box(&data),
                black_box(&labels),
                black_box(&fit.centroids),
            )
        })
    });

    group.bench_function("davies_bouldin_n500_d16_k5", |b| {
        b.iter(|| {
            metrics::davies_bouldin(
                black_box(&data),
                black_box(&labels),
                black_box(&fit.centroids),
                &Euclidean,
            )
        })
    });

    group.bench_function("k_distance_n500_d16_k4", |b| {
        b.iter(|| metrics::k_distance(black_box(&data), 4, &Euclidean))
    });

    group.finish();
}

fn bench_evoc(c: &mut Criterion) {
    let mut group = c.benchmark_group("evoc");

    let data = synth_data(1000, 16, 42);
    group.bench_function("n1000_d16", |b| {
        b.iter(|| {
            let mut evoc = EVoC::new(EVoCParams {
                intermediate_dim: 4,
                min_cluster_size: 5,
                seed: Some(42),
                ..Default::default()
            });
            evoc.fit_predict(black_box(&data)).unwrap()
        })
    });

    group.finish();
}

fn bench_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation");

    // Generate random signed edges.
    let mut rng = StdRng::seed_from_u64(42);
    let n_items = 500;
    let edges: Vec<SignedEdge> = (0..2000)
        .map(|_| SignedEdge {
            i: rng.random_range(0..n_items),
            j: rng.random_range(0..n_items),
            weight: rng.random::<f32>() * 2.0 - 1.0,
        })
        .filter(|e| e.i != e.j)
        .collect();

    group.bench_function("500items_2000edges", |b| {
        b.iter(|| {
            CorrelationClustering::new()
                .with_seed(42)
                .fit(n_items, black_box(&edges))
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
    bench_optics,
    bench_evoc,
    bench_correlation,
    bench_metrics,
);
criterion_main!(benches);
