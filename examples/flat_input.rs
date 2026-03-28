/// Zero-copy input via FlatRef and the DataRef trait.
///
/// Shows how to pass a contiguous f32 buffer directly to clustering
/// algorithms without converting to Vec<Vec<f32>>. Also demonstrates
/// the ndarray adapter (feature-gated).
fn main() {
    use clump::{FlatRef, Kmeans};

    // 6 points, 2 dimensions, stored as a contiguous row-major buffer.
    let flat_buf: Vec<f32> = vec![
        0.0, 0.0, // point 0
        0.1, 0.1, // point 1
        0.2, 0.2, // point 2
        10.0, 10.0, // point 3
        10.1, 10.1, // point 4
        10.2, 10.2, // point 5
    ];

    let n = 6;
    let d = 2;

    // --- FlatRef path (zero-copy) ---
    let flat = FlatRef::new(&flat_buf, n, d);
    let fit_flat = Kmeans::new(2).with_seed(42).fit(&flat).unwrap();
    println!("--- FlatRef path ---");
    println!("Labels: {:?}", fit_flat.labels);
    println!("WCSS:   {:.4}", fit_flat.wcss(&flat));

    // --- Vec<Vec<f32>> path (allocates) ---
    let vecs: Vec<Vec<f32>> = (0..n)
        .map(|i| flat_buf[i * d..(i + 1) * d].to_vec())
        .collect();
    let fit_vecs = Kmeans::new(2).with_seed(42).fit(&vecs).unwrap();
    println!("\n--- Vec<Vec<f32>> path ---");
    println!("Labels: {:?}", fit_vecs.labels);
    println!("WCSS:   {:.4}", fit_vecs.wcss(&vecs));

    // --- Verify identical results ---
    assert_eq!(fit_flat.labels, fit_vecs.labels, "labels must match");
    let wcss_diff = (fit_flat.wcss(&flat) - fit_vecs.wcss(&vecs)).abs();
    assert!(wcss_diff < 1e-6, "WCSS must match (diff={wcss_diff})");
    println!("\nBoth paths produce identical labels and WCSS. PASSED");

    // --- ndarray path (requires `--features ndarray`) ---
    #[cfg(feature = "ndarray")]
    {
        use clump::cluster::adapt::array2_to_vecs;
        use ndarray::Array2;

        let arr = Array2::from_shape_vec((n, d), flat_buf.clone()).expect("valid shape");
        let arr_vecs = array2_to_vecs(&arr);
        let fit_nd = Kmeans::new(2).with_seed(42).fit(&arr_vecs).unwrap();
        println!("\n--- ndarray path ---");
        println!("Labels: {:?}", fit_nd.labels);
        assert_eq!(fit_nd.labels, fit_flat.labels, "ndarray labels must match");
        println!("ndarray labels match FlatRef labels. PASSED");
    }
    #[cfg(not(feature = "ndarray"))]
    {
        println!("\nndarray path skipped (enable with --features ndarray)");
    }
}
