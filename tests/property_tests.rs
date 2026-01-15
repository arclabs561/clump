use clump::cluster::{Clustering, Kmeans};
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_kmeans_all_assigned(
        data in prop::collection::vec(prop::collection::vec(-10.0f32..10.0, 2), 1..20),
        k in 1usize..5
    ) {
        // Skip if k > n
        if k <= data.len() {
            let model = Kmeans::new(k).with_seed(42);
            let labels = model.fit_predict(&data).unwrap();

            prop_assert_eq!(labels.len(), data.len());
            for &l in &labels {
                prop_assert!(l < k);
            }
        }
    }
}
