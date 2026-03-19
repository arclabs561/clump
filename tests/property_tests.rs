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

    /// K-means WCSS is monotone non-increasing with more iterations.
    #[test]
    fn kmeans_monotone_inertia(data in arb_data(20, 3)) {
        let k = 2.min(data.len());
        let fit1 = Kmeans::new(k).with_seed(42).with_max_iter(1).fit(&data).unwrap();
        let fit10 = Kmeans::new(k).with_seed(42).with_max_iter(10).fit(&data).unwrap();
        let wcss1 = fit1.wcss(&data);
        let wcss10 = fit10.wcss(&data);
        prop_assert!(wcss10 <= wcss1 + 1e-4,
            "WCSS(10 iter)={} > WCSS(1 iter)={}", wcss10, wcss1);
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
