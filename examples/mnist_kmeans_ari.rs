//! k-means on the MNIST test split, scored against true digit labels.
//!
//! ARI, NMI, and purity measure recovery of the 10 digit classes; silhouette
//! measures cluster geometry.
//!
//! ```sh
//! ./scripts/fetch_mnist.sh
//! cargo run --release --example mnist_kmeans_ari
//! ```
//!
//! Raw-pixel k-means recovers digit structure only partially (ARI around 0.26),
//! as expected without a learned representation.

use std::path::Path;
use std::process::ExitCode;

use clump::cluster::metrics::{adjusted_rand_index, silhouette_score};
use clump::{Euclidean, Kmeans};

const K: usize = 10;

/// Read a big-endian u32 at offset `o`.
fn be_u32(b: &[u8], o: usize) -> usize {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

/// Parse an IDX image file into `n` flattened, [0,1]-scaled feature vectors.
fn load_images(path: &Path) -> std::io::Result<Vec<Vec<f32>>> {
    let b = std::fs::read(path)?;
    let n = be_u32(&b, 4);
    let rows = be_u32(&b, 8);
    let cols = be_u32(&b, 12);
    let d = rows * cols;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = 16 + i * d;
        out.push(b[start..start + d].iter().map(|&p| p as f32 / 255.0).collect());
    }
    Ok(out)
}

/// Parse an IDX label file into `n` class ids.
fn load_labels(path: &Path) -> std::io::Result<Vec<usize>> {
    let b = std::fs::read(path)?;
    let n = be_u32(&b, 4);
    Ok(b[8..8 + n].iter().map(|&l| l as usize).collect())
}

/// Normalized mutual information (sqrt normalization) between two labelings.
fn nmi(a: &[usize], b: &[usize], n_a: usize, n_b: usize) -> f64 {
    let n = a.len() as f64;
    let mut joint = vec![0.0f64; n_a * n_b];
    let mut ca = vec![0.0f64; n_a];
    let mut cb = vec![0.0f64; n_b];
    for (&x, &y) in a.iter().zip(b) {
        joint[x * n_b + y] += 1.0;
        ca[x] += 1.0;
        cb[y] += 1.0;
    }
    let entropy = |counts: &[f64]| -> f64 {
        counts
            .iter()
            .filter(|&&c| c > 0.0)
            .map(|&c| -(c / n) * (c / n).ln())
            .sum()
    };
    let (h_a, h_b) = (entropy(&ca), entropy(&cb));
    let mut mi = 0.0;
    for x in 0..n_a {
        for y in 0..n_b {
            let j = joint[x * n_b + y];
            if j > 0.0 {
                mi += (j / n) * ((j * n) / (ca[x] * cb[y])).ln();
            }
        }
    }
    if h_a <= 0.0 || h_b <= 0.0 {
        return 0.0;
    }
    mi / (h_a * h_b).sqrt()
}

/// Purity: weighted fraction of each cluster covered by its majority true class.
fn purity(pred: &[usize], truth: &[usize], k: usize, n_classes: usize) -> f64 {
    let mut counts = vec![0usize; k * n_classes];
    for (&p, &t) in pred.iter().zip(truth) {
        counts[p * n_classes + t] += 1;
    }
    let majority: usize = (0..k)
        .map(|c| (0..n_classes).map(|t| counts[c * n_classes + t]).max().unwrap_or(0))
        .sum();
    majority as f64 / pred.len() as f64
}

fn main() -> ExitCode {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/mnist");
    let images = dir.join("t10k-images-idx3-ubyte");
    let labels = dir.join("t10k-labels-idx1-ubyte");
    if !images.exists() {
        eprintln!("dataset not found at {}\nrun: ./scripts/fetch_mnist.sh", dir.display());
        return ExitCode::FAILURE;
    }

    let data = load_images(&images).unwrap();
    let truth = load_labels(&labels).unwrap();
    println!("images: {}  dims: {}  classes: {K}", data.len(), data[0].len());

    let fit = Kmeans::new(K)
        .with_max_iter(100)
        .with_seed(42)
        .fit(&data)
        .unwrap();
    let pred = fit.labels;

    let ari = adjusted_rand_index(&pred, &truth);
    let nmi_score = nmi(&pred, &truth, K, K);
    let pur = purity(&pred, &truth, K, K);
    let sil = silhouette_score(&data, &pred, &Euclidean);

    println!("\nexternal (vs true digit labels):");
    println!("  ARI     = {ari:.4}");
    println!("  NMI     = {nmi_score:.4}");
    println!("  purity  = {pur:.4}");
    println!("internal (cluster geometry):");
    println!("  silhouette = {sil:.4}");

    ExitCode::SUCCESS
}
