//! Distance metrics for clustering algorithms.
//!
//! All clustering algorithms in this crate are generic over `DistanceMetric`.
//! The default is `SquaredEuclidean`, which preserves backward compatibility.

/// A distance function between two equal-length `f32` slices.
///
/// Implementations must satisfy:
/// - `distance(a, a) == 0`
/// - `distance(a, b) >= 0`
/// - `distance(a, b) == distance(b, a)`
pub trait DistanceMetric: Clone + Send + Sync {
    /// Compute the distance between two vectors of equal length.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Squared Euclidean distance: `sum((a_i - b_i)^2)`.
///
/// This is the default metric. It preserves the same behavior as the
/// original hardcoded distance functions.
#[derive(Clone, Copy, Debug, Default)]
pub struct SquaredEuclidean;

impl DistanceMetric for SquaredEuclidean {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }
}

/// Euclidean (L2) distance: `sqrt(sum((a_i - b_i)^2))`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Euclidean;

impl DistanceMetric for Euclidean {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        SquaredEuclidean.distance(a, b).sqrt()
    }
}

/// Cosine distance: `1 - cos_sim(a, b)`.
///
/// Returns a value in `[0, 2]`. Zero means identical direction.
#[derive(Clone, Copy, Debug, Default)]
pub struct CosineDistance;

impl DistanceMetric for CosineDistance {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom < f32::EPSILON {
            return 0.0;
        }
        1.0 - (dot / denom)
    }
}

/// Inner product distance: `-dot(a, b)`.
///
/// Useful when vectors are normalized and you want to maximize similarity.
/// Note: this can be negative (for vectors pointing in the same direction).
#[derive(Clone, Copy, Debug, Default)]
pub struct InnerProductDistance;

impl DistanceMetric for InnerProductDistance {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        -dot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_euclidean_basic() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!((SquaredEuclidean.distance(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_basic() {
        let a = [3.0, 0.0];
        let b = [0.0, 4.0];
        assert!((Euclidean.distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical_vectors() {
        let a = [1.0, 2.0, 3.0];
        assert!(CosineDistance.distance(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!((CosineDistance.distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn inner_product_basic() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        // dot = 3 + 8 = 11, distance = -11
        assert!((InnerProductDistance.distance(&a, &b) - (-11.0)).abs() < 1e-6);
    }

    #[test]
    fn self_distance_is_zero() {
        let v = [1.0, 2.0, 3.0];
        assert!(SquaredEuclidean.distance(&v, &v).abs() < 1e-6);
        assert!(Euclidean.distance(&v, &v).abs() < 1e-6);
        assert!(CosineDistance.distance(&v, &v).abs() < 1e-6);
    }
}
