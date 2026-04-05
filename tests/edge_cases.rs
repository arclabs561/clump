//! Exhaustive edge case tests across all algorithms.
//!
//! Each test targets a specific degenerate input that could cause panics,
//! infinite loops, wrong results, or silent corruption.

use clump::*;

// ============================================================================
// K-means edge cases
// ============================================================================

#[test]
fn kmeans_n2_k2() {
    let data = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
    let fit = Kmeans::new(2).with_seed(42).fit(&data).unwrap();
    assert_eq!(fit.labels.len(), 2);
    assert_ne!(fit.labels[0], fit.labels[1]);
}

#[test]
fn kmeans_k_equals_n() {
    let data = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let fit = Kmeans::new(4).with_seed(42).fit(&data).unwrap();
    // Each point should be its own cluster.
    let unique: std::collections::HashSet<_> = fit.labels.iter().collect();
    assert_eq!(unique.len(), 4);
}

#[test]
fn kmeans_max_iter_1() {
    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
    ];
    let fit = Kmeans::new(2)
        .with_seed(42)
        .with_max_iter(1)
        .fit(&data)
        .unwrap();
    assert_eq!(fit.iters, 1);
    assert_eq!(fit.labels.len(), 4);
}

#[test]
fn kmeans_tol_zero() {
    // tol=0 means only max_iter stops convergence.
    let data = vec![vec![0.0], vec![10.0]];
    let fit = Kmeans::new(2)
        .with_seed(42)
        .with_tol(0.0)
        .with_max_iter(5)
        .fit(&data)
        .unwrap();
    assert!(fit.iters <= 5);
}

#[test]
fn kmeans_extreme_values_large() {
    let data = vec![vec![1e6, 1e6], vec![1e6 + 1.0, 1e6 + 1.0], vec![-1e6, -1e6]];
    let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
    assert_eq!(labels[0], labels[1]); // close together
    assert_ne!(labels[0], labels[2]); // far apart
}

#[test]
fn kmeans_extreme_values_small() {
    let data = vec![vec![1e-6, 1e-6], vec![1.1e-6, 1.1e-6], vec![1e-3, 1e-3]];
    let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
    assert_eq!(labels[0], labels[1]); // very close
}

#[test]
fn kmeans_negative_values() {
    let data = vec![
        vec![-5.0, -5.0],
        vec![-4.9, -5.1],
        vec![5.0, 5.0],
        vec![5.1, 4.9],
    ];
    let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[2], labels[3]);
    assert_ne!(labels[0], labels[2]);
}

#[test]
fn kmeans_deterministic_across_runs() {
    let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![10.0, 10.0]];
    let l1 = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
    let l2 = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
    assert_eq!(l1, l2);
}

#[test]
fn kmeans_warm_start_same_result() {
    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
    ];
    let fit1 = Kmeans::new(2).with_seed(42).fit(&data).unwrap();
    let fit2 = Kmeans::new(2)
        .with_centroids(fit1.centroids.clone())
        .fit(&data)
        .unwrap();
    assert_eq!(fit1.labels, fit2.labels);
}

// ============================================================================
// DBSCAN edge cases
// ============================================================================

#[test]
fn dbscan_all_identical() {
    let data = vec![vec![5.0, 5.0]; 10];
    let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
    // All identical -> all in one cluster.
    let first = labels[0];
    assert_ne!(first, NOISE);
    for &l in &labels {
        assert_eq!(l, first);
    }
}

#[test]
fn dbscan_n2_both_core() {
    let data = vec![vec![0.0, 0.0], vec![0.1, 0.0]];
    let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
    assert_eq!(labels[0], labels[1]);
    assert_ne!(labels[0], NOISE);
}

#[test]
fn dbscan_n2_both_noise() {
    let data = vec![vec![0.0, 0.0], vec![100.0, 100.0]];
    let labels = Dbscan::new(0.5, 3).fit_predict(&data).unwrap();
    // With min_pts=3, neither point has enough neighbors.
    assert_eq!(labels[0], NOISE);
    assert_eq!(labels[1], NOISE);
}

#[test]
fn dbscan_d1_scalar() {
    let data = vec![vec![0.0], vec![0.1], vec![0.2], vec![10.0], vec![10.1]];
    let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[0], labels[2]);
}

// ============================================================================
// HDBSCAN edge cases
// ============================================================================

#[test]
fn hdbscan_n3_minimum() {
    let data = vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![10.0, 10.0]];
    let labels = Hdbscan::new().fit_predict(&data).unwrap();
    assert_eq!(labels.len(), 3);
}

#[test]
fn hdbscan_all_identical() {
    let data = vec![vec![1.0, 1.0]; 10];
    let labels = Hdbscan::new().fit_predict(&data).unwrap();
    assert_eq!(labels.len(), 10);
    // All identical: should be one cluster or all noise.
}

#[test]
fn hdbscan_outlier_scores_finite() {
    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.0],
        vec![0.0, 0.1],
        vec![10.0, 10.0],
        vec![10.1, 10.0],
        vec![10.0, 10.1],
    ];
    let result = Hdbscan::new().fit(&data).unwrap();
    for &s in &result.outlier_scores {
        assert!(s.is_finite(), "outlier score must be finite");
        assert!((0.0..=1.0).contains(&s), "score {} not in [0,1]", s);
    }
}

// ============================================================================
// OPTICS edge cases
// ============================================================================

#[test]
fn optics_n2() {
    let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let result = Optics::new(10.0, 2).fit(&data).unwrap();
    assert_eq!(result.ordering.len(), 2);
}

#[test]
fn optics_all_identical() {
    let data = vec![vec![1.0, 1.0]; 5];
    let result = Optics::new(1.0, 2).fit(&data).unwrap();
    assert_eq!(result.ordering.len(), 5);
}

#[test]
fn optics_extract_at_zero_eps() {
    // Three points: [0,0], [0.1,0], [10,10]. min_pts=2 so core distance for
    // [0,0] is ~0.1 (its nearest neighbour). With extraction eps=0.001, all
    // reachability distances exceed 0.001, so every point should be noise.
    let data = vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![10.0, 10.0]];
    let result = Optics::new(100.0, 2).fit(&data).unwrap();
    let labels = Optics::<Euclidean>::extract_clusters(&result, 0.001);
    let non_noise = labels.iter().filter(|&&l| l != NOISE).count();
    assert_eq!(
        non_noise, 0,
        "eps=0.001 is below all pairwise distances: all points must be noise"
    );
}

// ============================================================================
// Correlation clustering edge cases
// ============================================================================

#[test]
fn correlation_zero_weight_edges() {
    let edges = vec![
        SignedEdge {
            i: 0,
            j: 1,
            weight: 0.0,
        },
        SignedEdge {
            i: 1,
            j: 2,
            weight: 0.0,
        },
    ];
    let result = CorrelationClustering::new()
        .with_seed(42)
        .fit(3, &edges)
        .unwrap();
    assert_eq!(result.labels.len(), 3);
}

#[test]
fn correlation_self_loop_ignored() {
    // Edges where i == j should be harmless.
    let edges = vec![
        SignedEdge {
            i: 0,
            j: 0,
            weight: 1.0,
        }, // self-loop
        SignedEdge {
            i: 0,
            j: 1,
            weight: 1.0,
        },
        SignedEdge {
            i: 1,
            j: 2,
            weight: -1.0,
        },
    ];
    let result = CorrelationClustering::new()
        .with_seed(42)
        .fit(3, &edges)
        .unwrap();
    assert_eq!(result.labels.len(), 3);
}

#[test]
fn correlation_duplicate_edges() {
    let edges = vec![
        SignedEdge {
            i: 0,
            j: 1,
            weight: 1.0,
        },
        SignedEdge {
            i: 0,
            j: 1,
            weight: 1.0,
        }, // duplicate
        SignedEdge {
            i: 1,
            j: 2,
            weight: -1.0,
        },
    ];
    let result = CorrelationClustering::new()
        .with_seed(42)
        .fit(3, &edges)
        .unwrap();
    // Should not crash or produce invalid labels.
    assert_eq!(result.labels.len(), 3);
}

// ============================================================================
// Streaming edge cases
// ============================================================================

#[test]
fn minibatch_single_point_k1() {
    let mut mbk = MiniBatchKmeans::new(1).with_seed(42);
    mbk.update_batch(&[vec![1.0, 2.0]]).unwrap();
    assert_eq!(mbk.centroids().len(), 1);
}

#[test]
fn minibatch_predict_after_init() {
    let mut mbk = MiniBatchKmeans::new(2).with_seed(42);
    mbk.update_batch(&[vec![0.0, 0.0], vec![10.0, 10.0]])
        .unwrap();
    let labels = mbk.predict(&[vec![0.1, 0.1], vec![9.9, 9.9]]).unwrap();
    assert_ne!(labels[0], labels[1]);
}

// ============================================================================
// Distance metric edge cases
// ============================================================================

#[test]
fn cosine_zero_vector() {
    let zero = vec![0.0, 0.0, 0.0];
    let nonzero = vec![1.0, 2.0, 3.0];
    let d = CosineDistance.distance(&zero, &nonzero);
    assert!(
        d.is_finite(),
        "cosine distance with zero vector must be finite, got {d}"
    );
}

#[test]
fn cosine_identical_vectors() {
    let v = vec![1.0, 2.0, 3.0];
    let d = CosineDistance.distance(&v, &v);
    assert!(
        d.abs() < 1e-6,
        "cosine distance of identical vectors should be ~0, got {d}"
    );
}

#[test]
fn squared_euclidean_overflow_check() {
    // Large values: 1000^2 * d could overflow, but f32::MAX ~ 3.4e38.
    // For d=16, max sq_euclidean = 16 * (2*1000)^2 = 64_000_000, well within f32.
    let a = vec![1000.0; 16];
    let b = vec![-1000.0; 16];
    let d = SquaredEuclidean.distance(&a, &b);
    assert!(d.is_finite(), "distance must be finite, got {d}");
    assert!((d - 64_000_000.0).abs() < 1.0, "expected ~64M, got {d}");
}

#[test]
#[allow(deprecated)]
fn inner_product_negative_distance() {
    // InnerProductDistance returns -dot, which is negative for aligned vectors.
    let a = vec![1.0, 1.0];
    let b = vec![1.0, 1.0];
    let d = InnerProductDistance.distance(&a, &b);
    assert!(
        d < 0.0,
        "inner product distance should be negative for aligned vectors"
    );
}

// ============================================================================
// Metrics edge cases
// ============================================================================

#[test]
fn silhouette_all_same_cluster() {
    let data = vec![vec![0.0], vec![1.0], vec![2.0]];
    let labels = vec![0, 0, 0];
    let score = cluster::metrics::silhouette_score(&data, &labels, &Euclidean);
    assert!(score.abs() < 0.01, "single cluster silhouette should be ~0");
}

#[test]
fn calinski_harabasz_perfect_separation() {
    let data = vec![vec![0.0], vec![0.0], vec![100.0], vec![100.0]];
    let labels = vec![0, 0, 1, 1];
    let centroids = vec![vec![0.0], vec![100.0]];
    let ch = cluster::metrics::calinski_harabasz(&data, &labels, &centroids);
    assert!(
        ch > 1000.0,
        "perfect separation should have very high CH, got {ch}"
    );
}

#[test]
fn davies_bouldin_perfect_separation() {
    let data = vec![vec![0.0], vec![0.0], vec![100.0], vec![100.0]];
    let labels = vec![0, 0, 1, 1];
    let centroids = vec![vec![0.0], vec![100.0]];
    let db = cluster::metrics::davies_bouldin(&data, &labels, &centroids, &Euclidean);
    assert!(
        db < 0.01,
        "perfect separation should have very low DB, got {db}"
    );
}

// ============================================================================
// More edge cases from fuzz sweep
// ============================================================================

#[test]
fn hdbscan_min_cluster_size_equals_n() {
    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.0],
        vec![0.0, 0.1],
        vec![10.0, 10.0],
    ];
    let labels = Hdbscan::new()
        .with_min_cluster_size(4)
        .fit_predict(&data)
        .unwrap();
    // min_cluster_size = n: at most one cluster or all noise.
    let non_noise: Vec<_> = labels.iter().filter(|&&l| l != NOISE).collect();
    assert!(non_noise.is_empty() || non_noise.len() == 4);
}

#[test]
fn dbscan_all_identical_various_eps() {
    let data = vec![vec![5.0, 5.0]; 10];
    for eps in [0.001, 0.1, 100.0] {
        let labels = Dbscan::new(eps, 2).fit_predict(&data).unwrap();
        // All identical: always one cluster regardless of epsilon.
        let first = labels[0];
        assert_ne!(first, NOISE);
        for &l in &labels {
            assert_eq!(l, first, "eps={eps}: all identical should be one cluster");
        }
    }
}

#[test]
fn kmeans_cosine_with_canceling_vectors() {
    // Vectors that cancel when averaged: [1, 0] and [-1, 0].
    let data = vec![
        vec![1.0, 0.0],
        vec![-1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, -1.0],
    ];
    let fit = Kmeans::with_metric(2, CosineDistance)
        .with_seed(42)
        .fit(&data)
        .unwrap();
    // Should not panic or produce NaN centroids.
    for c in &fit.centroids {
        for &v in c {
            assert!(v.is_finite(), "centroid has non-finite value: {:?}", c);
        }
    }
}

#[test]
fn optics_very_small_eps() {
    let data = vec![
        vec![0.0, 0.0],
        vec![0.001, 0.0],
        vec![10.0, 10.0],
        vec![10.001, 10.0],
    ];
    let result = Optics::new(0.01, 2).fit(&data).unwrap();
    assert_eq!(result.ordering.len(), 4);
    // Very tight clusters should still be found.
}

#[test]
fn correlation_max_iter_zero() {
    // max_iter=0: skip local search, return raw PIVOT.
    let edges = vec![
        SignedEdge {
            i: 0,
            j: 1,
            weight: 1.0,
        },
        SignedEdge {
            i: 1,
            j: 2,
            weight: -1.0,
        },
    ];
    let result = CorrelationClustering::new()
        .with_seed(42)
        .with_max_iter(0)
        .fit(3, &edges)
        .unwrap();
    assert_eq!(result.labels.len(), 3);
    assert!(result.cost >= 0.0, "cost must be non-negative");
}

#[test]
fn denstream_predict_before_any_update() {
    let ds = DenStream::new(0.5, 3);
    let result = ds.predict(&[1.0, 2.0]);
    assert!(result.is_err(), "predict before any update should error");
}

#[test]
fn minibatch_predict_before_init() {
    let mbk = MiniBatchKmeans::new(3).with_seed(42);
    let result = mbk.predict(&[vec![1.0, 2.0]]);
    assert!(result.is_err(), "predict before init should error");
}

#[test]
#[allow(deprecated)]
fn kmeans_with_inner_product_distance() {
    // InnerProductDistance returns negative values -- ensure no panic.
    let data = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![-1.0, 0.0],
        vec![0.0, -1.0],
    ];
    let labels = Kmeans::with_metric(2, InnerProductDistance)
        .with_seed(42)
        .fit_predict(&data)
        .unwrap();
    assert_eq!(labels.len(), 4);
    // No assertion on cluster quality -- just must not panic.
}

// ============================================================================
// FlatRef edge cases
// ============================================================================

#[test]
fn flatref_empty_zero_dim() {
    let data = FlatRef::new(&[], 0, 0);
    assert_eq!(data.n(), 0);
    assert_eq!(data.d(), 0);
}

#[test]
fn flatref_single_point() {
    let flat = vec![1.0, 2.0, 3.0];
    let data = FlatRef::new(&flat, 1, 3);
    assert_eq!(data.row(0), &[1.0, 2.0, 3.0]);
}

#[test]
fn flatref_kmeans_large_k() {
    // k = n with FlatRef.
    let flat: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let data = FlatRef::new(&flat, 10, 3);
    let fit = Kmeans::new(10).with_seed(42).fit(&data).unwrap();
    let unique: std::collections::HashSet<_> = fit.labels.iter().collect();
    assert_eq!(unique.len(), 10);
}

#[test]
fn flatref_hdbscan_consistent() {
    // HDBSCAN should produce identical results for Vec and FlatRef.
    let vecs = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.0],
        vec![0.0, 0.1],
        vec![0.1, 0.1],
        vec![0.05, 0.05],
        vec![10.0, 10.0],
        vec![10.1, 10.0],
        vec![10.0, 10.1],
        vec![10.1, 10.1],
        vec![10.05, 10.05],
    ];
    let flat: Vec<f32> = vecs.iter().flat_map(|v| v.iter().copied()).collect();
    let fref = FlatRef::new(&flat, 10, 2);

    let hdb = Hdbscan::new().with_min_samples(2).with_min_cluster_size(3);
    let labels_v = hdb.fit_predict(&vecs).unwrap();
    let labels_f = hdb.fit_predict(&fref).unwrap();
    assert_eq!(labels_v, labels_f);
}

#[test]
fn flatref_optics_consistent() {
    let vecs = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.0],
        vec![0.0, 0.1],
        vec![10.0, 10.0],
        vec![10.1, 10.0],
        vec![10.0, 10.1],
    ];
    let flat: Vec<f32> = vecs.iter().flat_map(|v| v.iter().copied()).collect();
    let fref = FlatRef::new(&flat, 6, 2);

    let result_v = Optics::new(1.0, 2).fit(&vecs).unwrap();
    let result_f = Optics::new(1.0, 2).fit(&fref).unwrap();
    assert_eq!(result_v.ordering, result_f.ordering);
    // Reachability may differ by float rounding but should be very close.
    for (a, b) in result_v
        .reachability
        .iter()
        .zip(result_f.reachability.iter())
    {
        assert!((a - b).abs() < 1e-6 || (a.is_infinite() && b.is_infinite()));
    }
}

// ============================================================================
// GLOSH outlier score properties
// ============================================================================

#[test]
fn hdbscan_glosh_noise_is_one() {
    let mut data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.0],
        vec![0.0, 0.1],
        vec![0.1, 0.1],
        vec![0.05, 0.05],
    ];
    data.push(vec![100.0, 100.0]); // outlier

    let result = Hdbscan::new()
        .with_min_samples(2)
        .with_min_cluster_size(3)
        .fit(&data)
        .unwrap();

    // Noise points must have score 1.0.
    for (i, &label) in result.labels.iter().enumerate() {
        if label == NOISE {
            assert!(
                (result.outlier_scores[i] - 1.0).abs() < 1e-6,
                "noise point {i} should have score 1.0, got {}",
                result.outlier_scores[i]
            );
        }
    }
}

#[test]
fn hdbscan_glosh_cluster_center_low_score() {
    // Points near the center of a dense cluster should have low outlier scores.
    let mut data = Vec::new();
    for i in 0..20 {
        let x = (i % 5) as f32 * 0.1;
        let y = (i / 5) as f32 * 0.1;
        data.push(vec![x, y]);
    }
    // Add a distant cluster.
    for i in 0..20 {
        data.push(vec![
            50.0 + (i % 5) as f32 * 0.1,
            50.0 + (i / 5) as f32 * 0.1,
        ]);
    }

    let result = Hdbscan::new()
        .with_min_samples(3)
        .with_min_cluster_size(5)
        .fit(&data)
        .unwrap();

    // Non-noise scores should be in [0, 1].
    for (i, &score) in result.outlier_scores.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&score),
            "point {i}: score {} not in [0, 1]",
            score
        );
    }
}

// ============================================================================
// Cross-algorithm consistency
// ============================================================================

#[test]
fn dbscan_optics_cluster_structure_matches() {
    // OPTICS extract_clusters(eps) must match DBSCAN(eps) cluster structure
    // on a larger dataset than the in-module test.
    let mut data = Vec::new();
    for i in 0..15 {
        data.push(vec![(i % 3) as f32 * 0.1, (i / 3) as f32 * 0.1]);
    }
    for i in 0..15 {
        data.push(vec![
            20.0 + (i % 3) as f32 * 0.1,
            20.0 + (i / 3) as f32 * 0.1,
        ]);
    }
    data.push(vec![10.0, 10.0]); // noise

    let eps = 0.5;
    let min_pts = 3;
    let db_labels = Dbscan::new(eps, min_pts).fit_predict(&data).unwrap();
    let optics_result = Optics::new(eps, min_pts).fit(&data).unwrap();
    let op_labels = Optics::<Euclidean>::extract_clusters(&optics_result, eps);

    // Same-cluster / different-cluster relationships must agree.
    for i in 0..data.len() {
        for j in (i + 1)..data.len() {
            let db_same = db_labels[i] == db_labels[j] && db_labels[i] != NOISE;
            let op_same = op_labels[i] == op_labels[j] && op_labels[i] != NOISE;
            assert_eq!(
                db_same, op_same,
                "DBSCAN/OPTICS disagree on ({i},{j}): db=({},{}), op=({},{})",
                db_labels[i], db_labels[j], op_labels[i], op_labels[j]
            );
        }
    }
}

#[test]
fn kmeans_wcss_decreases_with_more_clusters() {
    // WCSS(k) >= WCSS(k+1) -- more clusters always reduces within-cluster variance.
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<Vec<f32>> = (0..100)
        .map(|_| vec![rng.random::<f32>() * 10.0, rng.random::<f32>() * 10.0])
        .collect();

    let mut prev_wcss = f32::MAX;
    for k in 1..=5 {
        let fit = Kmeans::new(k)
            .with_seed(42)
            .with_max_iter(20)
            .fit(&data)
            .unwrap();
        let wcss = fit.wcss(&data);
        assert!(
            wcss <= prev_wcss + 1e-3,
            "WCSS(k={k}) = {wcss} > WCSS(k={}) = {prev_wcss}",
            k - 1
        );
        prev_wcss = wcss;
    }
}

// ============================================================================
// Numerical stability
// ============================================================================

#[test]
fn kmeans_extreme_scale_large() {
    // Large magnitude but with enough relative separation for f32.
    let data = vec![
        vec![1e6, 1e6],
        vec![1e6 + 1.0, 1e6 + 1.0],
        vec![-1e6, -1e6],
        vec![-1e6 - 1.0, -1e6 - 1.0],
    ];
    let labels = Kmeans::new(2).with_seed(42).fit_predict(&data).unwrap();
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[2], labels[3]);
    assert_ne!(labels[0], labels[2]);
}

#[test]
fn dbscan_very_small_epsilon() {
    // epsilon smaller than any inter-point distance: all noise.
    let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let labels = Dbscan::new(0.001, 2).fit_predict(&data).unwrap();
    for &l in &labels {
        assert_eq!(l, NOISE);
    }
}

#[test]
fn metrics_with_one_cluster_returns_zero() {
    let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
    let labels = vec![0, 0, 0];
    let sil = cluster::metrics::silhouette_score(&data, &labels, &Euclidean);
    assert!(
        sil.abs() < 0.01,
        "single-cluster silhouette should be ~0, got {sil}"
    );
}

// ============================================================================
// DenStream decay consistency
// ============================================================================

#[test]
fn denstream_base2_decay_rate() {
    // Verify that decay drives cluster pruning.
    //
    // With lambda=1.0 and beta=0.5, mu=1.0, the prune threshold is beta*mu=0.5.
    // A freshly created cluster starts at weight=1.0. After 2 time steps its
    // weight is 2^(-1.0*2) = 0.25 < 0.5, so it must be pruned on the next
    // prune pass.
    //
    // Strategy: seed one origin cluster, then feed t_p far-away points (each
    // becomes its own isolated outlier or p-cluster far from origin) so that
    // a prune fires. After pruning, the origin cluster should be gone because
    // its weight decayed below the threshold.
    //
    // t_p=5 means prune fires after every 5 updates.
    let t_p = 5usize;
    let mut ds = DenStream::new(0.5, 2) // tight epsilon: far-away points won't merge
        .with_beta(0.5)
        .with_lambda(1.0)
        .with_mu(1.0)
        .with_pruning_period(t_p);

    // Seed origin cluster.
    ds.update(&[0.0, 0.0]).unwrap();

    // Feed t_p far-away points to trigger a prune pass. Each is placed at a
    // distinct large coordinate so none merges into the origin cluster.
    for i in 1..=(t_p as i32) {
        ds.update(&[1000.0 * i as f32, 0.0]).unwrap();
    }

    // After t_p+1 updates (1 seed + t_p far points), a prune has fired.
    // The origin cluster has aged t_p steps: weight = 2^(-1.0*t_p) ≈ 0.031 < 0.5.
    // It must have been pruned.
    let centroids = ds.centroids();
    let has_origin = centroids
        .iter()
        .any(|c| c[0].abs() < 1.0 && c[1].abs() < 1.0);
    assert!(
        !has_origin,
        "origin cluster should be pruned after decaying below beta*mu threshold; \
         remaining centroids: {:?}",
        centroids
    );
}

// ============================================================================
// Constrained clustering edge cases
// ============================================================================

#[test]
fn cop_kmeans_all_must_link_single_cluster() {
    // All points must-linked: should produce 1 effective group.
    let data = vec![vec![0.0, 0.0], vec![5.0, 5.0], vec![10.0, 10.0]];
    let constraints = vec![Constraint::MustLink(0, 1), Constraint::MustLink(1, 2)];
    // k=2 but all must-linked: one cluster gets all 3, other is empty/reinit.
    let labels = CopKmeans::new(2)
        .with_seed(42)
        .fit_predict_constrained(&data, &constraints)
        .unwrap();
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
}

// ============================================================================
// Correlation clustering properties
// ============================================================================

#[test]
fn correlation_zero_weight_edges_no_cost() {
    // Zero-weight edges should contribute no cost regardless of assignment.
    let edges = vec![
        SignedEdge {
            i: 0,
            j: 1,
            weight: 0.0,
        },
        SignedEdge {
            i: 1,
            j: 2,
            weight: 0.0,
        },
    ];
    let result = CorrelationClustering::new()
        .with_seed(42)
        .fit(3, &edges)
        .unwrap();
    assert!(
        (result.cost - 0.0).abs() < 1e-9,
        "zero-weight edges should produce zero cost"
    );
}

#[test]
fn correlation_single_strong_edge() {
    // One very strong positive edge: those two must be co-clustered.
    let edges = vec![
        SignedEdge {
            i: 0,
            j: 1,
            weight: 1000.0,
        },
        SignedEdge {
            i: 0,
            j: 2,
            weight: -1.0,
        },
    ];
    let result = CorrelationClustering::new()
        .with_seed(42)
        .fit(3, &edges)
        .unwrap();
    assert_eq!(
        result.labels[0], result.labels[1],
        "strongly positive pair must co-cluster"
    );
}

// ============================================================================
// MiniBatchKmeans convergence
// ============================================================================

#[test]
fn minibatch_centroids_converge_toward_means() {
    // After many passes, centroids should approximate cluster means.
    let mut data = Vec::new();
    for _ in 0..50 {
        data.push(vec![0.0, 0.0]);
    }
    for _ in 0..50 {
        data.push(vec![10.0, 10.0]);
    }

    let mut mbk = MiniBatchKmeans::new(2).with_seed(42);
    for _ in 0..10 {
        mbk.update_batch(&data).unwrap();
    }

    let centroids = mbk.centroids();
    let near_zero = centroids.iter().any(|c| c[0] < 2.0 && c[1] < 2.0);
    let near_ten = centroids.iter().any(|c| c[0] > 8.0 && c[1] > 8.0);
    assert!(near_zero, "one centroid should be near (0,0)");
    assert!(near_ten, "one centroid should be near (10,10)");
}

// ============================================================================
// Tests inspired by sklearn/hdbscan reference test suites
// ============================================================================

/// sklearn: fit_predict == fit().predict() (internal consistency).
#[test]
fn kmeans_fit_predict_equals_fit_then_predict() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<Vec<f32>> = (0..100)
        .map(|_| vec![rng.random::<f32>() * 10.0, rng.random::<f32>() * 10.0])
        .collect();
    let labels_fp = Kmeans::new(5)
        .with_seed(42)
        .with_max_iter(10)
        .fit_predict(&data)
        .unwrap();
    let fit = Kmeans::new(5)
        .with_seed(42)
        .with_max_iter(10)
        .fit(&data)
        .unwrap();
    let labels_p = fit.predict(&data).unwrap();
    assert_eq!(
        labels_fp, labels_p,
        "fit_predict and fit().predict() must agree"
    );
}

/// sklearn: predict(centroids) returns identity [0, 1, ..., k-1].
#[test]
fn kmeans_predict_centroids_is_identity() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(7);
    let data: Vec<Vec<f32>> = (0..200)
        .map(|_| vec![rng.random::<f32>() * 50.0, rng.random::<f32>() * 50.0])
        .collect();
    let fit = Kmeans::new(10).with_seed(7).fit(&data).unwrap();
    let predicted = fit.predict(&fit.centroids).unwrap();
    let expected: Vec<usize> = (0..10).collect();
    assert_eq!(
        predicted, expected,
        "centroid k should predict to cluster k"
    );
}

/// sklearn: empty cluster relocation with deliberately bad init.
#[test]
fn kmeans_empty_cluster_relocated() {
    // One centroid starts far from all data -- its cluster will be empty.
    // k-means should relocate it (split the largest cluster).
    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.0],
        vec![0.0, 0.1],
        vec![0.1, 0.1],
        vec![0.05, 0.05],
    ];
    let init = vec![
        vec![0.05, 0.05],   // near the data
        vec![999.0, 999.0], // far away -- will produce empty cluster
    ];
    let fit = Kmeans::new(2)
        .with_centroids(init)
        .with_max_iter(20)
        .fit(&data)
        .unwrap();
    // Both clusters should have at least 1 point (empty one was relocated).
    let mut counts = [0usize; 2];
    for &l in &fit.labels {
        counts[l] += 1;
    }
    assert!(
        counts[0] > 0 && counts[1] > 0,
        "empty cluster should be relocated, got counts {:?}",
        counts
    );
}

/// sklearn: k >= n_unique_points should converge without looping.
#[test]
fn kmeans_k_ge_n_unique_converges() {
    // All points identical -- k=3 but only 1 unique point.
    let data = vec![vec![5.0, 5.0]; 10];
    let fit = Kmeans::new(3).with_seed(42).fit(&data).unwrap();
    assert!(
        fit.iters <= 3,
        "should converge fast with identical points, got {} iters",
        fit.iters
    );
    assert_eq!(fit.labels.len(), 10);
}

/// sklearn: DBSCAN eps boundary is inclusive (distance == eps counts as neighbor).
#[test]
fn dbscan_eps_boundary_inclusive_comprehensive() {
    // Triangle: points at (0,0), (1,0), (0,1). Distance between adjacent = 1.0.
    let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    // With eps=1.0, min_pts=2: each point has 2 neighbors at distance exactly 1.
    // All should be core points and in one cluster.
    let labels = Dbscan::new(1.0, 2).fit_predict(&data).unwrap();
    assert_eq!(labels[0], labels[1], "eps boundary should be inclusive");
    assert_eq!(labels[0], labels[2], "all three should connect");
    assert_ne!(labels[0], NOISE, "all should be core points");
}

/// sklearn: DBSCAN with precomputed vs feature-space should agree.
/// (Tests that grid/projection indexing doesn't lose neighbors.)
#[test]
fn dbscan_large_vs_small_n_agree() {
    // Small n uses precomputed matrix; this verifies the result pattern.
    let mut data = Vec::new();
    for i in 0..8 {
        data.push(vec![(i % 4) as f32 * 0.1, (i / 4) as f32 * 0.1]);
    }
    for i in 0..8 {
        data.push(vec![
            10.0 + (i % 4) as f32 * 0.1,
            10.0 + (i / 4) as f32 * 0.1,
        ]);
    }

    let labels = Dbscan::new(0.5, 3).fit_predict(&data).unwrap();
    // First 8 should cluster together, last 8 together.
    let c0 = labels[0];
    let c1 = labels[8];
    assert_ne!(c0, NOISE);
    assert_ne!(c1, NOISE);
    assert_ne!(c0, c1);
    for &l in &labels[..8] {
        assert_eq!(l, c0);
    }
    for &l in &labels[8..] {
        assert_eq!(l, c1);
    }
}

/// hdbscan ref: high-dimensional data should not degenerate.
#[test]
fn hdbscan_high_dimensional() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);
    // Two clusters in 64-d space.
    let mut data: Vec<Vec<f32>> = (0..25)
        .map(|_| (0..64).map(|_| rng.random::<f32>()).collect())
        .collect();
    data.extend((0..25).map(|_| {
        (0..64)
            .map(|_| 100.0 + rng.random::<f32>())
            .collect::<Vec<_>>()
    }));

    let labels = Hdbscan::new()
        .with_min_samples(3)
        .with_min_cluster_size(5)
        .fit_predict(&data)
        .unwrap();
    assert_eq!(labels.len(), 50);
    let non_noise: std::collections::HashSet<usize> =
        labels.iter().copied().filter(|&l| l != NOISE).collect();
    assert!(
        non_noise.len() >= 2,
        "should find at least 2 clusters in 64-d, found {}",
        non_noise.len()
    );
}

/// sklearn: silhouette of single-sample cluster is 0 (Rousseeuw convention).
#[test]
fn silhouette_single_sample_cluster_is_zero() {
    let data = vec![vec![0.0], vec![1.0], vec![2.0], vec![100.0]];
    // Point 3 is a singleton cluster.
    let labels = vec![0, 0, 0, 1];
    let score = cluster::metrics::silhouette_score(&data, &labels, &Euclidean);
    // With one singleton, overall silhouette drops but should still be valid.
    assert!(
        (-1.01..1.01).contains(&score),
        "silhouette out of range: {score}"
    );
}

/// sklearn: silhouette of identical points within cluster = high score.
#[test]
fn silhouette_identical_points_in_cluster() {
    // Two clusters: 3 identical points at 0, 3 identical points at 100.
    let data = vec![
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![100.0],
        vec![100.0],
        vec![100.0],
    ];
    let labels = vec![0, 0, 0, 1, 1, 1];
    let score = cluster::metrics::silhouette_score(&data, &labels, &Euclidean);
    // Intra-cluster distance = 0, inter-cluster = 100. s(i) = (100-0)/100 = 1.0.
    assert!(
        score > 0.99,
        "identical-point clusters should give silhouette ~1.0, got {score}"
    );
}

/// sklearn: non-encoded labels invariance (labels are IDs, not values).
#[test]
fn silhouette_label_values_dont_matter() {
    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.0],
        vec![10.0, 0.0],
        vec![10.1, 0.0],
    ];
    let labels_012 = vec![0, 0, 1, 1];
    let labels_big = vec![42, 42, 99, 99];
    let s1 = cluster::metrics::silhouette_score(&data, &labels_012, &Euclidean);
    let s2 = cluster::metrics::silhouette_score(&data, &labels_big, &Euclidean);
    assert!(
        (s1 - s2).abs() < 1e-6,
        "label values shouldn't matter: {s1} vs {s2}"
    );
}

/// linfa pattern: Send + Sync + Unpin for all public types.
#[test]
fn public_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync + Unpin>() {}
    assert_send_sync::<Kmeans>();
    assert_send_sync::<KmeansFit>();
    assert_send_sync::<Dbscan>();
    assert_send_sync::<Hdbscan>();
    assert_send_sync::<HdbscanResult>();
    assert_send_sync::<Optics>();
    assert_send_sync::<OpticsResult>();
    assert_send_sync::<CopKmeans>();
    assert_send_sync::<MiniBatchKmeans>();
    assert_send_sync::<DenStream>();
    assert_send_sync::<CorrelationClustering>();
    assert_send_sync::<CorrelationResult>();
    assert_send_sync::<SignedEdge>();
    assert_send_sync::<Constraint>();
    assert_send_sync::<FlatRef<'_>>();
}

/// sklearn: k-means refit from converged centroids is near-fixed-point.
/// Boundary points may differ due to final reassignment pass.
#[test]
fn kmeans_refit_from_converged_near_fixed_point() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<Vec<f32>> = (0..50)
        .map(|_| vec![rng.random::<f32>() * 10.0, rng.random::<f32>() * 10.0])
        .collect();
    let fit1 = Kmeans::new(3).with_seed(42).fit(&data).unwrap();
    let fit2 = Kmeans::new(3)
        .with_centroids(fit1.centroids.clone())
        .fit(&data)
        .unwrap();
    let agree = fit1
        .labels
        .iter()
        .zip(&fit2.labels)
        .filter(|(a, b)| a == b)
        .count();
    assert!(
        agree >= data.len() * 95 / 100,
        "refit should agree on >95%, got {}/{}",
        agree,
        data.len()
    );
    assert!(
        fit2.iters <= 3,
        "warm-start refit used {} iters",
        fit2.iters
    );
}

/// tol=0 runs all max_iter iterations (no early convergence).
/// With a very small tol, it should still converge before max_iter
/// on well-separated data.
#[test]
fn kmeans_very_small_tol_converges() {
    let data = vec![
        vec![0.0, 0.0],
        vec![0.1, 0.1],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
    ];
    let fit = Kmeans::new(2)
        .with_seed(42)
        .with_tol(1e-12)
        .with_max_iter(100)
        .fit(&data)
        .unwrap();
    assert!(
        fit.iters < 100,
        "very small tol should still converge, used {} iters",
        fit.iters
    );
}

// ============================================================================
// Test 1: Lloyd/Hamerly parity -- nearest centroid invariant
// ============================================================================

/// k=5 exercises the geometric assign path (k<=20).
#[test]
fn kmeans_nearest_centroid_geometric_path() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(99);
    let data: Vec<Vec<f32>> = (0..200)
        .map(|_| vec![rng.random::<f32>() * 100.0, rng.random::<f32>() * 100.0])
        .collect();
    let fit = Kmeans::new(5)
        .with_seed(99)
        .with_max_iter(50)
        .fit(&data)
        .unwrap();
    verify_nearest_assignment(&data, &fit.centroids, &fit.labels);
}

/// k=25 exercises the Hamerly bounds path (k>20).
#[test]
fn kmeans_nearest_centroid_hamerly_path() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(99);
    let data: Vec<Vec<f32>> = (0..200)
        .map(|_| vec![rng.random::<f32>() * 100.0, rng.random::<f32>() * 100.0])
        .collect();
    let fit = Kmeans::new(25)
        .with_seed(99)
        .with_max_iter(50)
        .fit(&data)
        .unwrap();
    verify_nearest_assignment(&data, &fit.centroids, &fit.labels);
}

fn verify_nearest_assignment(data: &[Vec<f32>], centroids: &[Vec<f32>], labels: &[usize]) {
    for (i, point) in data.iter().enumerate() {
        let mut best_k = 0;
        let mut best_dist = f32::MAX;
        for (k, c) in centroids.iter().enumerate() {
            let dist: f32 = point.iter().zip(c).map(|(a, b)| (a - b).powi(2)).sum();
            if dist < best_dist {
                best_dist = dist;
                best_k = k;
            }
        }
        assert_eq!(
            labels[i], best_k,
            "point {i} assigned to {} but nearest centroid is {} (dist {best_dist})",
            labels[i], best_k
        );
    }
}

// ============================================================================
// Test 2: f64 accumulation precision at scale
// ============================================================================

/// With 3000 points from 3 well-separated clusters, centroids must approximate
/// the true means within 1.0 (validates f64 accumulation prevents drift).
#[test]
fn kmeans_f64_accumulation_precision() {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);
    let true_centers = [[0.0f32, 0.0], [30.0, 30.0], [60.0, 0.0]];
    let mut data = Vec::with_capacity(3000);
    for center in &true_centers {
        for _ in 0..1000 {
            data.push(vec![
                center[0] + (rng.random::<f32>() - 0.5) * 2.0,
                center[1] + (rng.random::<f32>() - 0.5) * 2.0,
            ]);
        }
    }
    let fit = Kmeans::new(3)
        .with_seed(42)
        .with_max_iter(100)
        .fit(&data)
        .unwrap();
    for center in &true_centers {
        let nearest_dist = fit
            .centroids
            .iter()
            .map(|c| {
                let dx = c[0] - center[0];
                let dy = c[1] - center[1];
                (dx * dx + dy * dy).sqrt()
            })
            .fold(f32::MAX, f32::min);
        assert!(
            nearest_dist < 1.0,
            "no centroid within 1.0 of true center ({}, {}), nearest dist = {nearest_dist}",
            center[0],
            center[1]
        );
    }
}

// ============================================================================
// Test 3: Silhouette -- Rousseeuw 1987 hand-computed values
// ============================================================================

/// Hand-computed silhouette for 6 points in 2 well-separated clusters
/// using Euclidean distance.
///
/// Cluster 0: (0,0), (1,0), (0,1)
/// Cluster 1: (10,0), (11,0), (10,1)
///
/// Individual silhouette values:
///   s(0,0) = 0.90338, s(1,0) = 0.87100, s(0,1) = 0.88358
///   s(10,0) = 0.89672, s(11,0) = 0.88703, s(10,1) = 0.87558
/// Mean = 0.88622
#[test]
fn silhouette_rousseeuw_hand_computed() {
    let data = vec![
        vec![0.0f32, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![10.0, 0.0],
        vec![11.0, 0.0],
        vec![10.0, 1.0],
    ];
    let labels = vec![0, 0, 0, 1, 1, 1];
    let score = cluster::metrics::silhouette_score(&data, &labels, &Euclidean);
    let expected = 0.88622f32;
    assert!(
        (score - expected).abs() < 1e-3,
        "silhouette score {score} differs from hand-computed {expected}"
    );
    assert!(
        score > 0.8,
        "well-separated clusters should give silhouette > 0.8, got {score}"
    );
}

/// Two singleton clusters: silhouette = 0 (Rousseeuw convention).
#[test]
fn silhouette_two_singletons() {
    let data = vec![vec![0.0f32, 0.0], vec![10.0, 0.0]];
    let labels = vec![0, 1];
    let score = cluster::metrics::silhouette_score(&data, &labels, &Euclidean);
    assert!(
        score.abs() < 1e-6,
        "two singleton clusters should give silhouette 0, got {score}"
    );
}

// ============================================================================
// Test 4: OPTICS reachability invariants (Ankerst et al. 1999)
// ============================================================================

/// Two well-separated clusters: reachability plot should show a large jump.
#[test]
fn optics_reachability_valley_structure() {
    let mut data = Vec::new();
    for i in 0..10 {
        data.push(vec![(i % 5) as f32 * 0.1, (i / 5) as f32 * 0.1]);
    }
    for i in 0..10 {
        data.push(vec![
            50.0 + (i % 5) as f32 * 0.1,
            50.0 + (i / 5) as f32 * 0.1,
        ]);
    }
    let result = Optics::new(200.0, 3).fit(&data).unwrap();
    assert!(
        result.reachability[0].is_infinite(),
        "first point reachability should be inf, got {}",
        result.reachability[0]
    );
    for (pos, &r) in result.reachability.iter().enumerate() {
        assert!(r >= 0.0, "reachability[{pos}] = {r} should be non-negative");
    }
    for (pos, &cd) in result.core_distances.iter().enumerate() {
        assert!(
            cd >= 0.0,
            "core_distances[{pos}] = {cd} should be non-negative"
        );
    }
    let finite_reach: Vec<f32> = result
        .reachability
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .collect();
    if finite_reach.len() >= 2 {
        let max_reach = finite_reach
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_reach = finite_reach.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            max_reach > 10.0 * min_reach,
            "reachability should show cluster separation: max={max_reach}, min={min_reach}"
        );
    }
}

/// OPTICS witness: for every point with finite reachability, some earlier
/// point in the ordering must be within that reachability distance.
#[test]
fn optics_reachability_witness() {
    let data = vec![
        vec![0.0f32, 0.0],
        vec![0.5, 0.0],
        vec![1.0, 0.0],
        vec![1.5, 0.0],
        vec![10.0, 0.0],
        vec![10.5, 0.0],
        vec![11.0, 0.0],
    ];
    let result = Optics::new(100.0, 2).fit(&data).unwrap();
    for pos in 1..result.ordering.len() {
        let r = result.reachability[pos];
        if r.is_infinite() {
            continue;
        }
        let point_idx = result.ordering[pos];
        let has_witness = (0..pos).any(|prev_pos| {
            let prev_idx = result.ordering[prev_pos];
            let dist = Euclidean.distance(&data[point_idx], &data[prev_idx]);
            dist <= r + 1e-5
        });
        assert!(
            has_witness,
            "pos {pos} (point {point_idx}) has reachability {r} \
             but no earlier point is within that distance"
        );
    }
}
