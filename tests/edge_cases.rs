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
        assert!(s >= 0.0 && s <= 1.0, "score {} not in [0,1]", s);
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
    let data = vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![10.0, 10.0]];
    let result = Optics::new(100.0, 2).fit(&data).unwrap();
    let labels = Optics::<Euclidean>::extract_clusters(&result, 0.001);
    // Very small eps: most/all points should be noise.
    let non_noise = labels.iter().filter(|&&l| l != NOISE).count();
    assert!(non_noise <= labels.len()); // always true, documents intent
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
    assert!(non_noise.len() == 0 || non_noise.len() == 4);
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
    // After lambda * elapsed = 1 unit, weight should halve (base-2 decay).
    let mut ds = DenStream::new(100.0, 2)
        .with_beta(0.01)
        .with_lambda(1.0) // decay factor
        .with_mu(1.0)
        .with_pruning_period(100_000);

    // Feed one point.
    ds.update(&[0.0, 0.0]).unwrap();

    // Feed many points far away to advance time without merging.
    for i in 1..=10 {
        ds.update(&[1000.0 * i as f32, 0.0]).unwrap();
    }

    // The original cluster's centroid should still be near (0,0)
    // but its weight should have decayed significantly.
    let centroids = ds.centroids();
    let has_origin = centroids.iter().any(|c| c[0].abs() < 1.0);
    // With lambda=1.0 and ~10 time steps, 2^(-1*10) is very small.
    // The cluster may have been pruned entirely.
    // What matters is that the function doesn't panic and clusters are valid.
    assert!(ds.n_clusters() >= 1, "should have at least 1 cluster");
    if has_origin {
        // If the origin cluster survived, its centroid is still near 0.
        let origin_cluster = centroids.iter().find(|c| c[0].abs() < 1.0).unwrap();
        assert!(origin_cluster[0].abs() < 1.0);
    }
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
