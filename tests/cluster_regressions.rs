//! Regression tests for DBSCAN with non-Euclidean metrics at large n,
//! weighted correlation clustering, DenStream, CLAM, and the README's default
//! k-means settings.

use clump::cluster::correlation::{CorrelationClustering, SignedEdge};
use clump::{CosineDistance, Dbscan, DenStream, Kmeans, NOISE};

// ------------------------------------------------------------------ DBSCAN

/// Above the precomputed-matrix threshold DBSCAN used a grid or
/// random-projection index that assumes Euclidean geometry. Points on one ray
/// are at cosine distance ~0 but far apart in space, so every point became
/// noise.
#[test]
fn dbscan_cosine_large_n_finds_same_neighbors_as_small_n() {
    let dirs = [(1.0f32, 0.0f32), (0.0, 1.0), (-0.6, -0.8)];
    let n = 16_385;
    let data: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let (dx, dy) = dirs[i % 3];
            let r = 1.0 + (i / 3) as f32;
            vec![dx * r, dy * r]
        })
        .collect();
    let labels = Dbscan::with_metric(0.01, 5, CosineDistance)
        .fit_predict(&data)
        .unwrap();
    let noise = labels.iter().filter(|&&l| l == NOISE).count();
    let mut ids: Vec<usize> = labels.iter().copied().filter(|&l| l != NOISE).collect();
    ids.sort_unstable();
    ids.dedup();
    assert!(
        noise == 0 && ids.len() == 3,
        "noise {noise}, clusters {}",
        ids.len()
    );

    let small: Vec<Vec<f32>> = data[..300].to_vec();
    let sl = Dbscan::with_metric(0.01, 5, CosineDistance)
        .fit_predict(&small)
        .unwrap();
    assert!(sl.iter().all(|&l| l != NOISE));
}

// ------------------------------------------------------------- correlation

fn e(i: usize, j: usize, weight: f32) -> SignedEdge {
    SignedEdge { i, j, weight }
}

fn cost(labels: &[usize], edges: &[SignedEdge]) -> f64 {
    edges
        .iter()
        .map(|x| {
            let same = labels[x.i] == labels[x.j];
            if (x.weight > 0.0 && !same) || (x.weight < 0.0 && same) {
                (x.weight as f64).abs()
            } else {
                0.0
            }
        })
        .sum()
}

/// Brute-force optimum over all set partitions (restricted growth strings).
fn brute_opt(n: usize, edges: &[SignedEdge]) -> f64 {
    fn rec(
        i: usize,
        n: usize,
        lab: &mut Vec<usize>,
        maxl: usize,
        edges: &[SignedEdge],
        best: &mut f64,
    ) {
        if i == n {
            *best = best.min(cost(lab, edges));
            return;
        }
        for l in 0..=maxl + 1 {
            if i == 0 && l > 0 {
                break;
            }
            lab[i] = l;
            rec(i + 1, n, lab, maxl.max(l), edges, best);
        }
    }
    let mut lab = vec![0; n];
    let mut best = f64::INFINITY;
    rec(0, n, &mut lab, 0, edges, &mut best);
    best
}

/// Preclustering by unweighted neighbourhood overlap glued vertices 0 and 1
/// over a +0.01 edge, although 0 belongs with 2 (+100) and 1 with 3 (+100).
/// The optimum {0,2},{1,3} costs 2.01; the forced merge costs 101.
#[test]
fn correlation_precluster_does_not_force_costly_merge_on_weighted_graph() {
    let edges = vec![
        e(0, 1, 0.01),
        e(0, 2, 100.0),
        e(1, 3, 100.0),
        e(0, 3, 1.0),
        e(1, 2, 1.0),
        e(2, 3, -1000.0),
    ];
    let opt = brute_opt(4, &edges);
    assert!((opt - 2.01).abs() < 1e-4, "opt {opt}");
    for seed in 0..10 {
        let r = CorrelationClustering::new()
            .with_seed(seed)
            .fit(4, &edges)
            .unwrap();
        assert!(
            r.cost <= 3.0 * opt + 1e-6,
            "seed {seed}: labels {:?} cost {} vs optimum {opt}",
            r.labels,
            r.cost
        );
    }
}

// ---------------------------------------------------------------- DenStream

/// Cao et al. (2006) section 4.1 initialises micro-clusters whose radii stay
/// within epsilon. A 50-unit segment sampled every 0.05 must not collapse
/// into one wide micro-cluster that calls in-distribution points noise.
#[test]
fn denstream_init_micro_clusters_respect_epsilon() {
    let mut ds = DenStream::new(1.0, 3);
    for i in 0..1000 {
        ds.update(&[i as f32 * 0.05, 0.0]).unwrap();
    }
    assert!(ds.is_initialized());
    let n = ds.n_clusters();
    let label = ds.predict(&[5.0, 0.0]).unwrap();
    assert_ne!(
        label,
        NOISE,
        "{n} p-micro-cluster(s), centroids {:?}",
        ds.centroids()
    );
}

/// The same stream shifted by +1e4 must give the same model.
#[test]
fn denstream_is_translation_invariant() {
    let run = |off: f32| {
        let mut ds = DenStream::new(0.5, 3)
            .with_initial_buffer_size(0)
            .with_lambda(0.01);
        let mut s: u64 = 7;
        for _ in 0..400 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let a = ((s >> 33) as f32 / (1u64 << 31) as f32) * 0.3;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = ((s >> 33) as f32 / (1u64 << 31) as f32) * 0.3;
            ds.update(&[off + a, off + b]).unwrap();
        }
        ds.n_clusters()
    };
    let (a, b) = (run(0.0), run(1e4));
    assert_eq!(a, b, "p-micro-clusters at origin {a}, shifted by 1e4 {b}");
}

// --------------------------------------------------------------------- CLAM

/// The documented energy E = -(1/beta) log sum exp(-beta/2 ||v - xi||^2) has
/// gradient sum_mu softmax_mu (v - xi_mu), with no factor beta (Saha et al.
/// 2023). With a beta-scaled step, beta 25 and lr 0.1 overshoot and diverge.
#[cfg(feature = "hopfield")]
#[test]
fn clam_contract_converges_for_large_beta() {
    let centroids = vec![vec![0.0f64, 0.0], vec![10.0, 10.0]];
    let p = vec![0.5f64, 0.5];
    let out = clump::clam::am_contract(&p, &centroids, 25.0, 50, 0.1);
    let d = (out[0].powi(2) + out[1].powi(2)).sqrt();
    assert!(d < 0.5, "contracted to {out:?}");
}

#[cfg(feature = "hopfield")]
#[test]
fn clam_loss_small_for_points_at_centroids_large_beta() {
    let centroids = vec![vec![0.0f64, 0.0], vec![10.0, 10.0]];
    let data = vec![vec![0.1f64, 0.0], vec![10.0, 9.9]];
    let l = clump::clam::clam_loss(&data, &centroids, 30.0, 20, 0.1);
    assert!(l < 0.1, "loss {l}");
}

// ------------------------------------------------------------------ k-means

fn blobs(seed: u64, k: usize, per: usize) -> Vec<Vec<f32>> {
    let mut s = seed;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 53) as f64) as f32
    };
    let mut out = vec![];
    for c in 0..k {
        let (cx, cy) = ((c % 4) as f32 * 3.0, (c / 4) as f32 * 3.0);
        for _ in 0..per {
            let (a, b) = (u().max(1e-7), u());
            let r = (-2.0 * a.ln()).sqrt();
            let t = std::f32::consts::TAU * b;
            out.push(vec![cx + r * t.cos(), cy + r * t.sin()]);
        }
    }
    out
}

/// The README quickstart pattern `Kmeans::new(k).with_seed(..)` with the
/// default tolerance must end within 1% of the inertia a much tighter run
/// reaches from the same k-means++ initialisation.
#[test]
fn readme_default_kmeans_reaches_lloyd_fixed_point() {
    let k = 12;
    let data = blobs(7, k, 150);
    let mut worse = 0;
    let mut worst = 0.0f64;
    for seed in 0..10u64 {
        let d = Kmeans::new(k).with_seed(seed).fit(&data).unwrap();
        let t = Kmeans::new(k)
            .with_seed(seed)
            .with_tol(1e-4 / k as f64)
            .with_max_iter(1000)
            .fit(&data)
            .unwrap();
        let di = *d.inertia_trace.last().unwrap() as f64;
        let ti = *t.inertia_trace.last().unwrap() as f64;
        let rel = (di - ti) / ti;
        if rel > 0.01 {
            worse += 1;
        }
        worst = worst.max(rel);
    }
    assert_eq!(
        worse, 0,
        "default tolerance left >1% inertia in {worse}/10 seeds (worst {worst:.3})"
    );
}
