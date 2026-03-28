/// Basic k-means and DBSCAN on synthetic 2D data.
///
/// Generates two well-separated clusters, fits k-means (with WCSS) and
/// DBSCAN (with noise detection), then prints results.
fn main() {
    use clump::{Dbscan, Kmeans, NOISE};

    // Two clusters: one near the origin, one near (20, 20).
    let mut data = Vec::new();
    for i in 0..50 {
        let offset = i as f32 * 0.1;
        data.push(vec![offset, offset]);
    }
    for i in 0..50 {
        let offset = 20.0 + i as f32 * 0.1;
        data.push(vec![offset, offset]);
    }
    // One outlier far from both clusters.
    data.push(vec![100.0, 100.0]);

    let n = data.len();
    println!("Generated {n} points: 50 near origin, 50 near (20,20), 1 outlier");

    // -- K-means --
    let fit = Kmeans::new(2).with_seed(42).fit(&data).unwrap();
    println!("\n--- K-means (k=2) ---");
    println!("Iterations: {}", fit.iters);
    println!("WCSS (inertia): {:.2}", fit.wcss(&data));

    // Count points per cluster.
    let mut counts = [0usize; 2];
    for &l in &fit.labels {
        counts[l] += 1;
    }
    println!("Cluster sizes: {} and {}", counts[0], counts[1]);

    // Predict a new point.
    let new_points = vec![vec![0.5, 0.5], vec![20.5, 20.5]];
    let predicted = fit.predict(&new_points).unwrap();
    println!(
        "Predicted: (0.5,0.5) -> cluster {}, (20.5,20.5) -> cluster {}",
        predicted[0], predicted[1]
    );

    // -- DBSCAN --
    let labels = Dbscan::new(2.0, 3).fit_predict(&data).unwrap();
    println!("\n--- DBSCAN (eps=2.0, min_pts=3) ---");

    let n_noise = labels.iter().filter(|&&l| l == NOISE).count();
    let max_label = labels.iter().filter(|&&l| l != NOISE).copied().max();
    let n_clusters = max_label.map_or(0, |m| m + 1);
    println!("Clusters found: {n_clusters}");
    println!("Noise points: {n_noise}");

    // The outlier at (100,100) should be noise.
    let outlier_label = labels[n - 1];
    println!(
        "Outlier label: {}",
        if outlier_label == NOISE {
            "NOISE".to_string()
        } else {
            format!("{outlier_label}")
        }
    );
}
