//! K-means, DBSCAN, and HDBSCAN on a simple 2D dataset.

use clump::{Clustering, Dbscan, Hdbscan, Kmeans, NOISE};

fn main() {
    // Three well-separated clusters in 2D.
    let data: Vec<Vec<f32>> = vec![
        // Cluster A (near origin)
        vec![0.0, 0.0],
        vec![0.1, 0.2],
        vec![0.2, 0.1],
        vec![-0.1, 0.1],
        // Cluster B (near (5, 5))
        vec![5.0, 5.0],
        vec![5.1, 4.9],
        vec![4.9, 5.1],
        vec![5.2, 5.2],
        // Cluster C (near (10, 0))
        vec![10.0, 0.0],
        vec![10.1, 0.1],
        vec![9.9, -0.1],
        vec![10.2, 0.2],
    ];

    // --- K-means (k=3) ---
    let kmeans = Kmeans::new(3).with_seed(42);
    let labels = kmeans.fit_predict(&data).unwrap();
    println!("=== K-means (k=3) ===");
    for (i, label) in labels.iter().enumerate() {
        println!("  point {:2} ({:5.1}, {:5.1}) => cluster {}", i, data[i][0], data[i][1], label);
    }

    // --- DBSCAN (eps=1.0, min_pts=2) ---
    let dbscan = Dbscan::new(1.0, 2);
    let labels = dbscan.fit_predict(&data).unwrap();
    println!("\n=== DBSCAN (eps=1.0, min_pts=2) ===");
    for (i, label) in labels.iter().enumerate() {
        let tag = if *label == NOISE {
            "NOISE".to_string()
        } else {
            format!("cluster {}", label)
        };
        println!("  point {:2} ({:5.1}, {:5.1}) => {}", i, data[i][0], data[i][1], tag);
    }

    // --- HDBSCAN ---
    let hdbscan = Hdbscan::new().with_min_samples(2).with_min_cluster_size(2);
    let labels = hdbscan.fit_predict(&data).unwrap();
    println!("\n=== HDBSCAN (min_samples=2, min_cluster_size=2) ===");
    for (i, label) in labels.iter().enumerate() {
        let tag = if *label == NOISE {
            "NOISE".to_string()
        } else {
            format!("cluster {}", label)
        };
        println!("  point {:2} ({:5.1}, {:5.1}) => {}", i, data[i][0], data[i][1], tag);
    }
}
