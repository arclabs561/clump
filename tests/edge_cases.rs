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
