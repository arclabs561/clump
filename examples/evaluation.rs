/// Cluster evaluation metrics: silhouette, Calinski-Harabasz, Davies-Bouldin,
/// and DISCO for density-based results with noise.
///
/// Fits k-means for k=2..6 and prints a comparison table, then evaluates
/// a DBSCAN result with DISCO.
fn main() {
    use clump::cluster::metrics::{
        calinski_harabasz, davies_bouldin, disco_score, silhouette_score,
    };
    use clump::{Dbscan, Euclidean, Kmeans, NOISE};

    // Three well-separated clusters in 2D with an outlier.
    let mut data = Vec::new();
    // Cluster A: near (0, 0)
    for i in 0..40 {
        let v = i as f32 * 0.05;
        data.push(vec![v, v]);
    }
    // Cluster B: near (20, 20)
    for i in 0..40 {
        let v = 20.0 + i as f32 * 0.05;
        data.push(vec![v, v]);
    }
    // Cluster C: near (40, 0)
    for i in 0..40 {
        let v = 40.0 + i as f32 * 0.05;
        data.push(vec![v, -v + 40.0]);
    }
    // Outlier
    data.push(vec![100.0, 100.0]);

    let n = data.len();
    println!("Data: {n} points (3 clusters of 40 + 1 outlier)\n");

    // -- Elbow / metric comparison for k=2..6 --
    println!(
        "{:>3}  {:>10}  {:>11}  {:>12}  {:>13}",
        "k", "WCSS", "Silhouette", "Calinski-Hab", "Davies-Bould"
    );
    println!("{}", "-".repeat(56));

    for k in 2..=6 {
        let fit = Kmeans::new(k).with_seed(42).fit(&data).unwrap();
        let wcss = fit.wcss(&data);
        let sil = silhouette_score(&data, &fit.labels, &Euclidean);
        let ch = calinski_harabasz(&data, &fit.labels, &fit.centroids);
        let db = davies_bouldin(&data, &fit.labels, &fit.centroids, &Euclidean);
        println!("{k:>3}  {wcss:>10.1}  {sil:>11.4}  {ch:>12.1}  {db:>13.4}");
    }

    println!("\nBest k: highest silhouette, highest Calinski-Harabasz, lowest Davies-Bouldin.");
    println!("WCSS decreases monotonically; the elbow is where the rate of decrease slows.\n");

    // -- DISCO on a DBSCAN result --
    println!("--- DBSCAN + DISCO ---");
    let labels = Dbscan::new(3.0, 3).fit_predict(&data).unwrap();
    let n_noise = labels.iter().filter(|&&l| l == NOISE).count();
    let n_clusters = labels
        .iter()
        .filter(|&&l| l != NOISE)
        .copied()
        .max()
        .map_or(0, |m| m + 1);
    println!("DBSCAN (eps=3.0, min_pts=3): {n_clusters} clusters, {n_noise} noise points");

    let disco = disco_score(&data, &labels, &Euclidean, 3);
    println!("DISCO score: {disco:.4}");
    println!("DISCO evaluates both cluster quality and noise assignment correctness.");
}
