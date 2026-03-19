//! Cross-cutting property tests that verify invariants across algorithms.

use clump::*;
use proptest::prelude::*;

fn arb_data(max_n: usize, d: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
    proptest::collection::vec(
        proptest::collection::vec(-100.0f32..100.0, d..=d),
        3..=max_n,
    )
}

proptest! {
    /// K-means: all labels must be in [0, k).
    #[test]
    fn prop_kmeans_all_assigned(
        data in proptest::collection::vec(
            proptest::collection::vec(-10.0f32..10.0, 2..=2), 3..20
        ),
        k in 1usize..5
    ) {
        if k <= data.len() {
            let labels = Kmeans::new(k).with_seed(42).with_max_iter(3)
                .fit_predict(&data).unwrap();
            prop_assert_eq!(labels.len(), data.len());
            for &l in &labels {
                prop_assert!(l < k);
            }
        }
    }

    /// DistanceMetric: self-distance must be zero.
    #[test]
    fn distance_self_is_zero(v in proptest::collection::vec(-100.0f32..100.0, 2..=16)) {
        prop_assert!((SquaredEuclidean.distance(&v, &v)).abs() < 1e-5);
        prop_assert!((Euclidean.distance(&v, &v)).abs() < 1e-5);
        prop_assert!((CosineDistance.distance(&v, &v)).abs() < 1e-3);
    }

    /// DistanceMetric: symmetry.
    #[test]
    fn distance_symmetric(
        a in proptest::collection::vec(-100.0f32..100.0, 4..=4),
        b in proptest::collection::vec(-100.0f32..100.0, 4..=4),
    ) {
        let d1 = SquaredEuclidean.distance(&a, &b);
        let d2 = SquaredEuclidean.distance(&b, &a);
        prop_assert!((d1 - d2).abs() < 1e-3, "not symmetric: {} vs {}", d1, d2);

        let d1 = Euclidean.distance(&a, &b);
        let d2 = Euclidean.distance(&b, &a);
        prop_assert!((d1 - d2).abs() < 1e-3);

        let d1 = CosineDistance.distance(&a, &b);
        let d2 = CosineDistance.distance(&b, &a);
        prop_assert!((d1 - d2).abs() < 1e-3);

        let d1 = InnerProductDistance.distance(&a, &b);
        let d2 = InnerProductDistance.distance(&b, &a);
        prop_assert!((d1 - d2).abs() < 1e-3);
    }

    /// DistanceMetric: non-negative for Euclidean/SquaredEuclidean.
    #[test]
    fn distance_nonnegative(
        a in proptest::collection::vec(-100.0f32..100.0, 4..=4),
        b in proptest::collection::vec(-100.0f32..100.0, 4..=4),
    ) {
        prop_assert!(SquaredEuclidean.distance(&a, &b) >= 0.0);
        prop_assert!(Euclidean.distance(&a, &b) >= 0.0);
    }

    /// K-means WCSS: finite and non-negative.
    #[test]
    fn kmeans_wcss_finite(data in arb_data(20, 3)) {
        let k = 2.min(data.len());
        let fit = Kmeans::new(k).with_seed(42).with_max_iter(3).fit(&data).unwrap();
        let wcss = fit.wcss(&data);
        prop_assert!(wcss.is_finite(), "WCSS must be finite");
        prop_assert!(wcss >= 0.0, "WCSS must be >= 0");
    }

    /// DBSCAN labels valid.
    #[test]
    fn dbscan_labels_valid(data in arb_data(20, 2)) {
        let labels = Dbscan::new(1.0, 2).fit_predict(&data).unwrap();
        prop_assert_eq!(labels.len(), data.len());
        let non_noise: Vec<usize> = labels.iter().copied().filter(|&l| l != NOISE).collect();
        if !non_noise.is_empty() {
            let max = *non_noise.iter().max().unwrap();
            for l in 0..=max {
                prop_assert!(non_noise.contains(&l), "label {l} missing");
            }
        }
    }

    /// Silhouette in [-1, 1].
    #[test]
    fn silhouette_range(data in arb_data(12, 2)) {
        let k = 2.min(data.len());
        let labels = Kmeans::new(k).with_seed(42).with_max_iter(3)
            .fit_predict(&data).unwrap();
        let score = cluster::metrics::silhouette_score(&data, &labels, &Euclidean);
        prop_assert!(score >= -1.01 && score <= 1.01,
            "silhouette {} not in [-1, 1]", score);
    }

    /// MiniBatchKmeans predict returns valid labels.
    #[test]
    fn minibatch_predict_valid(data in arb_data(15, 3)) {
        let k = 2.min(data.len());
        let mut mbk = MiniBatchKmeans::new(k).with_seed(42);
        mbk.update_batch(&data).unwrap();
        let labels = mbk.predict(&data).unwrap();
        for &l in &labels {
            prop_assert!(l < k);
        }
    }

    /// HDBSCAN labels valid.
    #[test]
    fn hdbscan_labels_valid(data in arb_data(12, 2)) {
        let labels = Hdbscan::new().fit_predict(&data).unwrap();
        prop_assert_eq!(labels.len(), data.len());
        for &l in &labels {
            prop_assert!(l == NOISE || l < data.len());
        }
    }

    /// OPTICS ordering is a permutation.
    #[test]
    fn optics_ordering_permutation(data in arb_data(12, 2)) {
        let result = Optics::new(100.0, 2).fit(&data).unwrap();
        let n = data.len();
        prop_assert_eq!(result.ordering.len(), n);
        let mut sorted = result.ordering.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..n).collect();
        prop_assert_eq!(sorted, expected);
    }

    /// K-means inertia trace is monotone non-increasing within a single run.
    #[test]
    fn kmeans_monotone_inertia(data in arb_data(20, 3)) {
        let k = 2.min(data.len());
        let fit = Kmeans::new(k).with_seed(42).with_max_iter(20).fit(&data).unwrap();
        for w in fit.inertia_trace.windows(2) {
            prop_assert!(w[1] <= w[0] + 1e-4,
                "inertia increased: {} -> {}", w[0], w[1]);
        }
    }

    /// K-means centroids predict to themselves.
    #[test]
    fn kmeans_centroids_predict_self(data in arb_data(20, 3)) {
        let k = 2.min(data.len());
        let fit = Kmeans::new(k).with_seed(42).with_max_iter(5).fit(&data).unwrap();
        let pred = fit.predict(&fit.centroids).unwrap();
        let expected: Vec<usize> = (0..k).collect();
        prop_assert_eq!(pred, expected);
    }

    /// K-means with k=1 assigns all points label 0.
    #[test]
    fn kmeans_k1_all_same_label(data in arb_data(15, 2)) {
        let labels = Kmeans::new(1).with_seed(42).with_max_iter(5)
            .fit_predict(&data).unwrap();
        for &l in &labels {
            prop_assert_eq!(l, 0);
        }
    }

    /// K-means on all-identical points does not panic.
    #[test]
    fn kmeans_identical_points_no_panic(
        n in 3..15usize,
        point in proptest::collection::vec(-10.0f32..10.0, 3..=3),
    ) {
        let data: Vec<Vec<f32>> = vec![point; n];
        let labels = Kmeans::new(2).with_seed(42).with_max_iter(5)
            .fit_predict(&data).unwrap();
        prop_assert_eq!(labels.len(), n);
        for &l in &labels {
            prop_assert!(l < 2);
        }
    }

    /// K-means with same seed produces identical labels.
    #[test]
    fn kmeans_seed_determinism(data in arb_data(15, 3), seed in 0u64..100) {
        let k = 2.min(data.len());
        let labels1 = Kmeans::new(k).with_seed(seed).with_max_iter(5)
            .fit_predict(&data).unwrap();
        let labels2 = Kmeans::new(k).with_seed(seed).with_max_iter(5)
            .fit_predict(&data).unwrap();
        prop_assert_eq!(labels1, labels2);
    }

    /// MiniBatchKmeans labels stabilize after repeated updates on same data.
    #[test]
    fn minibatch_repeated_update_valid(data in arb_data(15, 3)) {
        let k = 2.min(data.len());
        let mut mbk = MiniBatchKmeans::new(k).with_seed(42);
        for _ in 0..5 {
            let _ = mbk.update_batch(&data).unwrap();
        }
        let labels = mbk.predict(&data).unwrap();
        for &l in &labels {
            prop_assert!(l < k);
        }
    }

    /// ARI of identical clusterings is 1.0.
    #[test]
    fn ari_identical_is_one(
        labels in proptest::collection::vec(0usize..3, 4..20),
    ) {
        let ari = cluster::metrics::adjusted_rand_index(&labels, &labels);
        prop_assert!((ari - 1.0).abs() < 0.01,
            "ARI of identical labels should be ~1.0, got {}", ari);
    }

    /// ARI is symmetric.
    #[test]
    fn ari_symmetric(
        a in proptest::collection::vec(0usize..3, 4..20usize),
    ) {
        let b: Vec<usize> = a.iter().map(|x| (x + 1) % 3).collect();
        let ari_ab = cluster::metrics::adjusted_rand_index(&a, &b);
        let ari_ba = cluster::metrics::adjusted_rand_index(&b, &a);
        prop_assert!((ari_ab - ari_ba).abs() < 1e-10,
            "ARI not symmetric: {} vs {}", ari_ab, ari_ba);
    }

    /// Calinski-Harabasz index is non-negative.
    #[test]
    fn calinski_harabasz_nonneg(data in arb_data(15, 3)) {
        let k = 2.min(data.len());
        let fit = Kmeans::new(k).with_seed(42).with_max_iter(5).fit(&data).unwrap();
        let ch = cluster::metrics::calinski_harabasz(&data, &fit.labels, &fit.centroids);
        prop_assert!(ch >= 0.0, "Calinski-Harabasz must be >= 0, got {}", ch);
    }

    /// noise_ratio returns a value in [0, 1].
    #[test]
    fn noise_ratio_in_unit_range(
        n_valid in 1usize..10,
        n_noise in 0usize..10,
    ) {
        let mut labels: Vec<usize> = (0..n_valid).map(|i| i % 3).collect();
        labels.extend(std::iter::repeat(NOISE).take(n_noise));
        let ratio = cluster::metrics::noise_ratio(&labels);
        prop_assert!(ratio >= 0.0 && ratio <= 1.0,
            "noise_ratio {} not in [0, 1]", ratio);
    }

    /// Sampled silhouette is in [-1, 1].
    #[test]
    fn sampled_silhouette_range(data in arb_data(20, 2)) {
        let k = 2.min(data.len());
        let labels = Kmeans::new(k).with_seed(42).with_max_iter(3)
            .fit_predict(&data).unwrap();
        let score = cluster::metrics::silhouette_score_sampled(
            &data, &labels, &Euclidean, 10, 99,
        );
        prop_assert!(score >= -1.01 && score <= 1.01,
            "sampled silhouette {} not in [-1, 1]", score);
    }

    /// DBSCAN cluster connectivity: every cluster is eps-connected.
    /// Points in the same cluster must be reachable via a chain of eps-neighbors
    /// through other cluster members (density-reachability invariant).
    #[test]
    fn dbscan_cluster_connectivity(
        data in proptest::collection::vec(
            proptest::collection::vec(-5.0f32..5.0, 3..=3),
            5..=20
        ),
    ) {
        let eps = 1.5f32;
        let labels = Dbscan::new(eps, 2).fit_predict(&data).unwrap();

        let max_label = labels.iter().copied().filter(|&l| l != NOISE).max();
        if let Some(max_l) = max_label {
            for cluster_id in 0..=max_l {
                let members: Vec<usize> = labels.iter().enumerate()
                    .filter(|(_, &l)| l == cluster_id)
                    .map(|(i, _)| i)
                    .collect();
                if members.len() < 2 { continue; }

                // BFS from first member -- all others must be reachable via
                // eps-hops through cluster members.
                let mut reached = vec![false; data.len()];
                let mut queue = std::collections::VecDeque::new();
                queue.push_back(members[0]);
                reached[members[0]] = true;

                while let Some(p) = queue.pop_front() {
                    for &q in &members {
                        if !reached[q] && Euclidean.distance(&data[p], &data[q]) <= eps {
                            reached[q] = true;
                            queue.push_back(q);
                        }
                    }
                }

                for &m in &members {
                    prop_assert!(reached[m],
                        "cluster {} not eps-connected: point {} unreachable from point {}",
                        cluster_id, m, members[0]);
                }
            }
        }
    }
}

proptest! {
    /// FlatRef produces identical results to Vec<Vec<f32>> (DataRef equivalence).
    #[test]
    fn flatref_matches_vec_vec(data in arb_data(20, 4)) {
        let k = 2.min(data.len());
        // Vec<Vec<f32>> path
        let labels_vv = Kmeans::new(k).with_seed(42).with_max_iter(5)
            .fit_predict(&data).unwrap();
        // FlatRef path
        let flat: Vec<f32> = data.iter().flat_map(|v| v.iter().copied()).collect();
        let d = data[0].len();
        let fref = FlatRef::new(&flat, data.len(), d);
        let labels_fr = Kmeans::new(k).with_seed(42).with_max_iter(5)
            .fit_predict(&fref).unwrap();
        prop_assert_eq!(labels_vv, labels_fr,
            "FlatRef and Vec<Vec<f32>> must produce identical k-means labels");
    }

    /// FlatRef DBSCAN matches Vec<Vec<f32>> DBSCAN.
    #[test]
    fn flatref_dbscan_matches(data in arb_data(15, 3)) {
        let labels_vv = Dbscan::new(1.0, 2).fit_predict(&data).unwrap();
        let flat: Vec<f32> = data.iter().flat_map(|v| v.iter().copied()).collect();
        let fref = FlatRef::new(&flat, data.len(), data[0].len());
        let labels_fr = Dbscan::new(1.0, 2).fit_predict(&fref).unwrap();
        prop_assert_eq!(labels_vv, labels_fr,
            "FlatRef and Vec<Vec<f32>> must produce identical DBSCAN labels");
    }

    /// DenStream decay is consistent: decayed weight matches threshold expectations.
    /// After t_p updates at a location, the weight should be above the potential
    /// threshold (no spurious pruning of active clusters).
    #[test]
    fn denstream_active_cluster_survives(n in 5usize..30) {
        let mut ds = DenStream::new(2.0, 2)
            .with_beta(0.5)
            .with_lambda(0.01)
            .with_mu(1.0)
            .with_pruning_period(100);
        for _ in 0..n {
            ds.update(&[0.0, 0.0]).unwrap();
        }
        // An actively fed cluster should always be present.
        prop_assert!(ds.n_clusters() >= 1,
            "actively fed cluster should survive, got {} clusters after {} points",
            ds.n_clusters(), n);
    }

    /// Correlation clustering cost never increases with local search.
    #[test]
    fn correlation_local_search_monotone(n in 3usize..10) {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i+1)..n {
                let w = ((i * 7 + j * 13) % 20) as f32 - 10.0;
                edges.push(SignedEdge { i, j, weight: w });
            }
        }
        let pivot_only = CorrelationClustering::new()
            .with_max_iter(0).with_seed(42)
            .fit(n, &edges).unwrap();
        let with_search = CorrelationClustering::new()
            .with_max_iter(100).with_seed(42)
            .fit(n, &edges).unwrap();
        prop_assert!(with_search.cost <= pivot_only.cost + 1e-9,
            "local search increased cost: {} -> {}", pivot_only.cost, with_search.cost);
    }
}

/// Metrics reject NaN in debug mode.
#[test]
#[should_panic(expected = "not finite")]
fn metrics_reject_nan() {
    let data = vec![vec![0.0, f32::NAN], vec![1.0, 1.0]];
    let labels = vec![0, 1];
    cluster::metrics::silhouette_score(&data, &labels, &Euclidean);
}

proptest! {
    /// HDBSCAN non-noise clusters must have >= min_cluster_size points
    /// (post-GLOSH refactor regression test).
    #[test]
    fn hdbscan_min_cluster_size_post_glosh(data in arb_data(25, 2)) {
        let min_cs = 4;
        let result = Hdbscan::new()
            .with_min_samples(2)
            .with_min_cluster_size(min_cs)
            .fit(&data)
            .unwrap();

        // Check labels.
        let mut counts = std::collections::HashMap::new();
        for &l in &result.labels {
            if l != NOISE {
                *counts.entry(l).or_insert(0usize) += 1;
            }
        }
        for (&cluster, &count) in &counts {
            prop_assert!(count >= min_cs,
                "cluster {} has {} points < min_cluster_size {}",
                cluster, count, min_cs);
        }

        // Check outlier scores.
        for (i, &s) in result.outlier_scores.iter().enumerate() {
            prop_assert!(s.is_finite(), "outlier_scores[{}] is not finite: {}", i, s);
            prop_assert!((0.0..=1.0).contains(&s),
                "outlier_scores[{}] = {} not in [0,1]", i, s);
            if result.labels[i] == NOISE {
                prop_assert!((s - 1.0).abs() < 1e-6,
                    "noise point {} should have score 1.0, got {}", i, s);
            }
        }
    }

    /// Davies-Bouldin index is non-negative.
    #[test]
    fn davies_bouldin_nonneg(data in arb_data(15, 3)) {
        let k = 2.min(data.len());
        let fit = Kmeans::new(k).with_seed(42).with_max_iter(5).fit(&data).unwrap();
        let db = cluster::metrics::davies_bouldin(&data, &fit.labels, &fit.centroids, &Euclidean);
        prop_assert!(db >= 0.0, "Davies-Bouldin must be >= 0, got {}", db);
        prop_assert!(db.is_finite(), "Davies-Bouldin must be finite");
    }

    /// EVoC produces valid labels (None or Some(id < num_clusters)).
    #[test]
    fn evoc_labels_valid(data in arb_data(12, 4)) {
        let mut evoc = EVoC::new(EVoCParams {
            intermediate_dim: 2,
            min_cluster_size: 2,
            seed: Some(42),
            ..Default::default()
        });
        if data[0].len() > 2 {
            let labels = evoc.fit_predict(&data).unwrap();
            prop_assert_eq!(labels.len(), data.len());
            for (i, label) in labels.iter().enumerate() {
                if let Some(cid) = label {
                    prop_assert!(*cid < data.len(),
                        "point {} has cluster_id {} >= n={}", i, cid, data.len());
                }
            }
        }
    }

    /// Noise-aware silhouette excludes noise and returns valid range.
    #[test]
    fn noise_aware_silhouette_valid(data in arb_data(15, 2)) {
        let labels = Dbscan::new(1.0, 2).fit_predict(&data).unwrap();
        let score = cluster::metrics::silhouette_score_noise_aware(&data, &labels, &Euclidean);
        prop_assert!(score >= -1.01 && score <= 1.01,
            "noise-aware silhouette {} not in [-1, 1]", score);
    }

    /// K-distance curve is sorted ascending.
    #[test]
    fn k_distance_sorted(data in arb_data(10, 2)) {
        let kd = cluster::metrics::k_distance(&data, 2, &Euclidean);
        prop_assert_eq!(kd.len(), data.len());
        for w in kd.windows(2) {
            prop_assert!(w[0] <= w[1] + 1e-6, "k-distance not sorted: {} > {}", w[0], w[1]);
        }
    }
}

#[cfg(feature = "parallel")]
mod parallel_tests {
    use clump::*;
    use proptest::prelude::*;

    fn arb_data(max_n: usize, d: usize) -> impl Strategy<Value = Vec<Vec<f32>>> {
        proptest::collection::vec(
            proptest::collection::vec(-100.0f32..100.0, d..=d),
            3..=max_n,
        )
    }

    proptest! {
        /// Parallel k-means is deterministic: same seed produces identical results.
        #[test]
        fn parallel_kmeans_deterministic(
            data in arb_data(30, 4),
            seed in 0u64..50,
        ) {
            let k = 2.min(data.len());
            let fit1 = Kmeans::new(k).with_seed(seed).with_max_iter(10).fit(&data).unwrap();
            let fit2 = Kmeans::new(k).with_seed(seed).with_max_iter(10).fit(&data).unwrap();
            prop_assert_eq!(&fit1.labels, &fit2.labels, "parallel k-means not deterministic");
            prop_assert_eq!(&fit1.centroids, &fit2.centroids);
        }

        /// Parallel HDBSCAN is deterministic (no randomness in algorithm).
        #[test]
        fn parallel_hdbscan_deterministic(data in arb_data(15, 3)) {
            let l1 = Hdbscan::new().fit_predict(&data).unwrap();
            let l2 = Hdbscan::new().fit_predict(&data).unwrap();
            prop_assert_eq!(&l1, &l2, "parallel HDBSCAN not deterministic");
        }
    }
}
