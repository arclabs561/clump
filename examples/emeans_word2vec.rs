//! One timed fit for scripts/bench_emeans_word2vec.py.
//! The little-endian f32 input is TensorFlow Embedding Projector's Word2Vec 10K.
use clump::{Emeans, FlatRef, Kmeans};
use std::{env, error::Error, fs, io, time::Instant};

const N: usize = 10_000;
const D: usize = 200;
const K: usize = 32;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let [path, method, budget, seed] = args.as_slice() else {
        return Err("usage: emeans_word2vec FILE kmeans|emeans BUDGET SEED".into());
    };
    let seed: u64 = seed.parse()?;
    let bytes = fs::read(path)?;
    if bytes.len() != N * D * 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "expected 10,000 by 200 f32 vectors",
        )
        .into());
    }
    let points: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect();
    let data = FlatRef::new(&points, N, D);
    let started = Instant::now();
    let centers = match method.as_str() {
        "kmeans" => {
            Kmeans::new(K)
                .with_seed(seed)
                .with_max_iter(budget.parse()?)
                .fit(&data)?
                .centroids
        }
        "emeans" => {
            Emeans::new(K)
                .with_seed(seed)
                .with_epochs(budget.parse()?)
                .fit(&data)?
                .centroids
        }
        _ => return Err("unknown method".into()),
    };
    let elapsed = started.elapsed().as_secs_f64();

    // Score both trainers in f64 using an independent nearest-center scan.
    // Input, scoring, process startup and cargo compilation are outside the timer.
    let loss = points
        .as_chunks::<D>()
        .0
        .iter()
        .map(|p| {
            centers
                .iter()
                .map(|c| {
                    p.iter()
                        .zip(c)
                        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                        .sum::<f64>()
                })
                .fold(f64::INFINITY, f64::min)
        })
        .sum::<f64>()
        / N as f64;
    println!("{elapsed:.9},{loss:.12}");
    Ok(())
}
