//! Adapters for converting between clump's `&[Vec<f32>]` input format
//! and other Rust data structures.

/// Convert an ndarray `Array2<f32>` to clump's input format.
///
/// Each row becomes one data point.
///
/// ```ignore
/// use ndarray::Array2;
/// use clump::cluster::adapt::array2_to_vecs;
///
/// let arr = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let vecs = array2_to_vecs(&arr);
/// assert_eq!(vecs.len(), 3);
/// assert_eq!(vecs[0], vec![1.0, 2.0]);
/// ```
#[cfg(feature = "ndarray")]
pub fn array2_to_vecs(arr: &ndarray::Array2<f32>) -> Vec<Vec<f32>> {
    arr.rows().into_iter().map(|row| row.to_vec()).collect()
}

/// Convert an ndarray `ArrayView2<f32>` to clump's input format.
#[cfg(feature = "ndarray")]
pub fn array_view_to_vecs(arr: ndarray::ArrayView2<f32>) -> Vec<Vec<f32>> {
    arr.rows().into_iter().map(|row| row.to_vec()).collect()
}

/// Convert a flat slice + dimensions to clump's input format.
///
/// ```
/// use clump::cluster::adapt::flat_to_vecs;
///
/// let flat = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let vecs = flat_to_vecs(&flat, 3, 2);
/// assert_eq!(vecs.len(), 3);
/// assert_eq!(vecs[1], vec![3.0, 4.0]);
/// ```
pub fn flat_to_vecs(flat: &[f32], n: usize, d: usize) -> Vec<Vec<f32>> {
    assert_eq!(flat.len(), n * d, "flat.len() must equal n * d");
    (0..n).map(|i| flat[i * d..(i + 1) * d].to_vec()).collect()
}

/// Convert clump labels to an ndarray `Array1<usize>`.
#[cfg(feature = "ndarray")]
pub fn labels_to_array1(labels: &[usize]) -> ndarray::Array1<usize> {
    ndarray::Array1::from(labels.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_round_trip() {
        let flat = vec![1.0f32, 2.0, 3.0, 4.0];
        let vecs = flat_to_vecs(&flat, 2, 2);
        assert_eq!(vecs, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    #[should_panic(expected = "flat.len() must equal n * d")]
    fn flat_wrong_size() {
        flat_to_vecs(&[1.0, 2.0, 3.0], 2, 2);
    }
}
