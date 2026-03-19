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
}
