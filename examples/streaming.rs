/// Streaming clustering with MiniBatchKmeans and DenStream.
///
/// Feeds data in batches and shows how centroids and micro-clusters
/// evolve over time.
fn main() {
    use clump::{DenStream, MiniBatchKmeans, NOISE};

    // --- MiniBatchKmeans ---
    println!("--- MiniBatchKmeans (k=2) ---");
    let mut mbk = MiniBatchKmeans::new(2).with_seed(42);

    // First batch: seeds centroids via k-means++.
    let batch1: Vec<Vec<f32>> = (0..20)
        .map(|i| vec![i as f32 * 0.1, i as f32 * 0.1])
        .collect();
    let labels1 = mbk.update_batch(&batch1).unwrap();
    println!(
        "After batch 1 ({} pts): centroids at {:?}",
        batch1.len(),
        format_centroids(mbk.centroids()),
    );
    assert_eq!(labels1.len(), batch1.len());

    // Second batch: points from a different region.
    let batch2: Vec<Vec<f32>> = (0..20)
        .map(|i| vec![50.0 + i as f32 * 0.1, 50.0 + i as f32 * 0.1])
        .collect();
    let labels2 = mbk.update_batch(&batch2).unwrap();
    println!(
        "After batch 2 ({} pts): centroids at {:?}",
        batch2.len(),
        format_centroids(mbk.centroids()),
    );
    assert_eq!(labels2.len(), batch2.len());

    // Third batch: more points near the first cluster.
    let batch3: Vec<Vec<f32>> = (0..30)
        .map(|i| vec![i as f32 * 0.05, i as f32 * 0.05])
        .collect();
    let labels3 = mbk.update_batch(&batch3).unwrap();
    println!(
        "After batch 3 ({} pts): centroids at {:?}",
        batch3.len(),
        format_centroids(mbk.centroids()),
    );
    assert_eq!(labels3.len(), batch3.len());

    // Predict on new data.
    let test = vec![vec![0.5, 0.5], vec![50.5, 50.5]];
    let pred = mbk.predict(&test).unwrap();
    println!(
        "Predict: (0.5,0.5)->cluster {}, (50.5,50.5)->cluster {}",
        pred[0], pred[1]
    );

    // --- DenStream ---
    println!("\n--- DenStream ---");
    let mut ds = DenStream::new(2.0, 2)
        .with_beta(0.5)
        .with_lambda(0.001)
        .with_mu(1.0)
        .with_macro_epsilon(10.0);

    // Feed cluster A: points near (0, 0).
    for i in 0..30 {
        let v = i as f32 * 0.1;
        ds.update(&[v, v]).unwrap();
    }
    println!(
        "After 30 pts near origin: {} potential micro-clusters",
        ds.n_clusters(),
    );

    // Feed cluster B: points near (50, 50).
    for i in 0..30 {
        let v = 50.0 + i as f32 * 0.1;
        ds.update(&[v, v]).unwrap();
    }
    println!(
        "After 30 more pts near (50,50): {} potential micro-clusters",
        ds.n_clusters(),
    );

    // Run macro-clustering (DBSCAN on micro-cluster centroids).
    let macro_labels = ds.macro_cluster().unwrap();
    let n_macro = macro_labels
        .iter()
        .filter(|&&l| l != NOISE)
        .copied()
        .max()
        .map_or(0, |m| m + 1);
    let n_macro_noise = macro_labels.iter().filter(|&&l| l == NOISE).count();
    println!("Macro-clusters: {n_macro}, noise micro-clusters: {n_macro_noise}");

    // Predict batch: each point maps to its nearest potential micro-cluster.
    let test_pts = vec![vec![0.5, 0.5], vec![50.5, 50.5]];
    let ds_pred = ds.predict_batch(&test_pts).unwrap();
    println!(
        "Predict: (0.5,0.5)->mc {}, (50.5,50.5)->mc {}",
        ds_pred[0], ds_pred[1],
    );
}

/// Format centroid coordinates for display (2 decimal places).
fn format_centroids(centroids: &[Vec<f32>]) -> Vec<String> {
    centroids
        .iter()
        .map(|c| {
            let coords: Vec<String> = c.iter().map(|v| format!("{v:.2}")).collect();
            format!("({})", coords.join(", "))
        })
        .collect()
}
