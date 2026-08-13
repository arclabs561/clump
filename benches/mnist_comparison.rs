//! Reproducible real-data K-means comparison on the MNIST test split.
//!
//! Each timed invocation runs one implementation in a fresh process:
//!
//! ```sh
//! ./scripts/fetch_mnist.sh
//! cargo bench --bench mnist_comparison -- clump
//! cargo bench --bench mnist_comparison -- linfa
//! cargo bench --bench mnist_comparison -- diagnose
//! ```
//!
//! Both implementations receive the same `f32` pixels, initial centroids,
//! iteration limit, and tolerance. `diagnose` runs both without reporting a
//! timing comparison and measures their label agreement. Wall time covers fit
//! only. Peak RSS is deliberately omitted: there is no portable in-process
//! measurement, and wrapping each command with the platform's `time` utility
//! is clearer than pretending those interfaces are equivalent.

use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_clustering::{KMeans, KMeansInit};
use ndarray::Array2;
use rand::prelude::*;
use serde_json::{json, Value};

const K: usize = 10;
const MAX_ITER: usize = 100;
const TOLERANCE: f32 = 1e-12;
const SEED: u64 = 42;

struct Mnist {
    vecs: Vec<Vec<f32>>,
    array: Array2<f32>,
    truth: Vec<usize>,
}

struct FitResult {
    labels: Vec<usize>,
    centroids: Vec<Vec<f32>>,
    elapsed: Duration,
    iterations: Option<usize>,
}

fn be_u32(bytes: &[u8], offset: usize) -> Result<usize, String> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| format!("truncated IDX header at byte {offset}"))?;
    Ok(u32::from_be_bytes(raw.try_into().unwrap()) as usize)
}

fn load_mnist(dir: &Path) -> Result<Mnist, String> {
    let image_path = dir.join("t10k-images-idx3-ubyte");
    let label_path = dir.join("t10k-labels-idx1-ubyte");
    let images = std::fs::read(&image_path)
        .map_err(|error| format!("failed to read {}: {error}", image_path.display()))?;
    let labels = std::fs::read(&label_path)
        .map_err(|error| format!("failed to read {}: {error}", label_path.display()))?;

    if be_u32(&images, 0)? != 2051 || be_u32(&labels, 0)? != 2049 {
        return Err("unexpected MNIST IDX magic number".to_owned());
    }
    let n = be_u32(&images, 4)?;
    let label_count = be_u32(&labels, 4)?;
    let rows = be_u32(&images, 8)?;
    let cols = be_u32(&images, 12)?;
    let d = rows * cols;
    if label_count != n || images.len() != 16 + n * d || labels.len() != 8 + n {
        return Err("MNIST IDX lengths do not match their headers".to_owned());
    }

    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
    for pixels in images[16..].chunks_exact(d) {
        vecs.push(pixels.iter().map(|&pixel| pixel as f32 / 255.0).collect());
    }
    let array = Array2::from_shape_fn((n, d), |(row, col)| vecs[row][col]);
    let truth = labels[8..].iter().map(|&label| label as usize).collect();
    Ok(Mnist { vecs, array, truth })
}

fn initial_centroids(data: &Mnist) -> (Vec<Vec<f32>>, Array2<f32>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut indices = rand::seq::index::sample(&mut rng, data.vecs.len(), K).into_vec();
    indices.sort_unstable();
    let vecs = indices
        .iter()
        .map(|&index| data.vecs[index].clone())
        .collect::<Vec<_>>();
    let d = data.array.ncols();
    let array = Array2::from_shape_fn((K, d), |(row, col)| vecs[row][col]);
    (vecs, array, indices)
}

fn fit_clump(data: &Mnist, centroids: Vec<Vec<f32>>) -> Result<FitResult, String> {
    let started = Instant::now();
    let fit = clump::Kmeans::new(K)
        .with_max_iter(MAX_ITER)
        .with_tol(TOLERANCE as f64)
        .with_centroids(centroids)
        .fit(&data.vecs)
        .map_err(|error| error.to_string())?;
    let elapsed = started.elapsed();
    Ok(FitResult {
        labels: fit.labels,
        centroids: fit.centroids,
        elapsed,
        iterations: Some(fit.iters),
    })
}

fn fit_linfa(data: &Mnist, centroids: Array2<f32>) -> Result<FitResult, String> {
    let dataset = DatasetBase::from(data.array.view());
    let started = Instant::now();
    let fit = KMeans::params(K)
        .max_n_iterations(MAX_ITER as u64)
        .n_runs(1)
        .tolerance(TOLERANCE)
        .init_method(KMeansInit::Precomputed(centroids))
        .fit(&dataset)
        .map_err(|error| error.to_string())?;
    let labels = fit.predict(data.array.view()).targets.to_vec();
    let elapsed = started.elapsed();
    let centroids = fit
        .centroids()
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    Ok(FitResult {
        labels,
        centroids,
        elapsed,
        iterations: None,
    })
}

fn choose2(count: usize) -> f64 {
    (count.saturating_sub(1) * count / 2) as f64
}

fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    let mut joint = vec![0usize; K * K];
    let mut count_a = [0usize; K];
    let mut count_b = [0usize; K];
    for (&x, &y) in a.iter().zip(b) {
        joint[x * K + y] += 1;
        count_a[x] += 1;
        count_b[y] += 1;
    }
    let pairs = choose2(a.len());
    let sum_joint: f64 = joint.into_iter().map(choose2).sum();
    let sum_a: f64 = count_a.into_iter().map(choose2).sum();
    let sum_b: f64 = count_b.into_iter().map(choose2).sum();
    let expected = sum_a * sum_b / pairs;
    let maximum = (sum_a + sum_b) / 2.0;
    if (maximum - expected).abs() <= f64::EPSILON {
        1.0
    } else {
        (sum_joint - expected) / (maximum - expected)
    }
}

fn nmi(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len() as f64;
    let mut joint = vec![0.0f64; K * K];
    let mut count_a = [0.0f64; K];
    let mut count_b = [0.0f64; K];
    for (&x, &y) in a.iter().zip(b) {
        joint[x * K + y] += 1.0;
        count_a[x] += 1.0;
        count_b[y] += 1.0;
    }
    let entropy = |counts: &[f64]| {
        counts
            .iter()
            .filter(|&&count| count > 0.0)
            .map(|&count| -(count / n) * (count / n).ln())
            .sum::<f64>()
    };
    let mut mutual_information = 0.0;
    for x in 0..K {
        for y in 0..K {
            let count = joint[x * K + y];
            if count > 0.0 {
                mutual_information += (count / n) * ((count * n) / (count_a[x] * count_b[y])).ln();
            }
        }
    }
    let scale = (entropy(&count_a) * entropy(&count_b)).sqrt();
    if scale == 0.0 {
        0.0
    } else {
        mutual_information / scale
    }
}

fn purity(predicted: &[usize], truth: &[usize]) -> f64 {
    let mut counts = vec![0usize; K * K];
    for (&cluster, &class) in predicted.iter().zip(truth) {
        counts[cluster * K + class] += 1;
    }
    let majority: usize = (0..K)
        .map(|cluster| {
            (0..K)
                .map(|class| counts[cluster * K + class])
                .max()
                .unwrap()
        })
        .sum();
    majority as f64 / predicted.len() as f64
}

fn wcss(data: &Mnist, fit: &FitResult) -> f64 {
    data.vecs
        .iter()
        .zip(&fit.labels)
        .map(|(point, &label)| {
            point
                .iter()
                .zip(&fit.centroids[label])
                .map(|(&value, &center)| {
                    let delta = f64::from(value) - f64::from(center);
                    delta * delta
                })
                .sum::<f64>()
        })
        .sum()
}

fn metrics(name: &str, data: &Mnist, fit: &FitResult) -> Value {
    json!({
        "implementation": name,
        "elapsed_seconds": fit.elapsed.as_secs_f64(),
        "iterations": fit.iterations,
        "ari": adjusted_rand_index(&fit.labels, &data.truth),
        "nmi": nmi(&fit.labels, &data.truth),
        "purity": purity(&fit.labels, &data.truth),
        "wcss": wcss(data, fit),
    })
}

fn run(mode: &str) -> Result<Value, String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mnist");
    let data = load_mnist(&dir).map_err(|error| {
        format!("{error}\nfetch the dataset first with ./scripts/fetch_mnist.sh")
    })?;
    let (clump_centroids, linfa_centroids, centroid_indices) = initial_centroids(&data);
    let common = json!({
        "dataset": "MNIST test",
        "samples": data.vecs.len(),
        "dimensions": data.array.ncols(),
        "clusters": K,
        "max_iterations": MAX_ITER,
        "tolerance": TOLERANCE,
        "initial_centroid_indices": centroid_indices,
    });

    match mode {
        "clump" => {
            let fit = fit_clump(&data, clump_centroids)?;
            Ok(json!({"config": common, "result": metrics("clump", &data, &fit)}))
        }
        "linfa" => {
            let fit = fit_linfa(&data, linfa_centroids)?;
            Ok(json!({"config": common, "result": metrics("linfa", &data, &fit)}))
        }
        "diagnose" => {
            let clump = fit_clump(&data, clump_centroids)?;
            let linfa = fit_linfa(&data, linfa_centroids)?;
            Ok(json!({
                "config": common,
                "results": [metrics("clump", &data, &clump), metrics("linfa", &data, &linfa)],
                "cross_implementation_ari": adjusted_rand_index(&clump.labels, &linfa.labels),
                "timing_boundary": "fit through training-label assignment",
            }))
        }
        _ => {
            Err("usage: cargo bench --bench mnist_comparison -- <clump|linfa|diagnose>".to_owned())
        }
    }
}

fn main() -> ExitCode {
    let mode = std::env::args().nth(1).unwrap_or_default();
    match run(&mode) {
        Ok(output) => {
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
