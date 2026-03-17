//! Distance metrics for clustering algorithms.
//!
//! All clustering algorithms in this crate are generic over `DistanceMetric`.
//! The default metric varies by algorithm: `Kmeans` and `EVoC` default to
//! `SquaredEuclidean`; `Dbscan` and `Hdbscan` default to `Euclidean`.

/// A distance function between two equal-length `f32` slices.
///
/// Implementations must satisfy:
/// - `distance(a, a) == 0`
/// - `distance(a, b) >= 0`
/// - `distance(a, b) == distance(b, a)`
pub trait DistanceMetric: Clone + Send + Sync {
    /// Compute the distance between two vectors of equal length.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// Whether this metric supports the expanded squared Euclidean identity
    /// `||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c` for faster assignment.
    fn supports_expanded_form(&self) -> bool {
        false
    }

    /// Whether centroids should be L2-normalized after each update step.
    ///
    /// For cosine-based k-means, the correct centroid is the L2-normalized
    /// mean of assigned points (Dhillon & Modha 2001). Without normalization,
    /// centroids drift off the unit sphere and convergence degrades.
    fn normalize_centroids(&self) -> bool {
        false
    }
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

    fn supports_expanded_form(&self) -> bool {
        true
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
    fn normalize_centroids(&self) -> bool {
        true
    }

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

/// Weighted combination of two distance metrics.
///
/// Computes `weight_a * A::distance(a, b) + weight_b * B::distance(a, b)`.
///
/// This lets callers express composite similarity notions such as
/// "0.6 * string_similarity + 0.4 * embedding_distance" as a single metric
/// that plugs into any clustering algorithm.
///
/// ```
/// use clump::cluster::distance::{CompositeDistance, SquaredEuclidean, Euclidean, DistanceMetric};
///
/// let metric = CompositeDistance::new(SquaredEuclidean, Euclidean, 0.5, 0.5);
/// let d = metric.distance(&[0.0, 0.0], &[3.0, 4.0]);
/// // 0.5 * 25.0 + 0.5 * 5.0 = 15.0
/// assert!((d - 15.0).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
pub struct CompositeDistance<A: DistanceMetric, B: DistanceMetric> {
    a: A,
    b: B,
    weight_a: f32,
    weight_b: f32,
}

impl<A: DistanceMetric, B: DistanceMetric> CompositeDistance<A, B> {
    /// Create a new composite distance from two metrics and their weights.
    ///
    /// Weights are not required to sum to 1 -- they are used as-is.
    pub fn new(a: A, b: B, weight_a: f32, weight_b: f32) -> Self {
        Self {
            a,
            b,
            weight_a,
            weight_b,
        }
    }
}

impl<A: DistanceMetric, B: DistanceMetric> DistanceMetric for CompositeDistance<A, B> {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.weight_a * self.a.distance(a, b) + self.weight_b * self.b.distance(a, b)
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

    #[test]
    fn composite_weighted_combination() {
        // SquaredEuclidean([0,0],[3,4]) = 25, Euclidean([0,0],[3,4]) = 5
        // 0.5 * 25 + 0.5 * 5 = 15.0
        let metric = CompositeDistance::new(SquaredEuclidean, Euclidean, 0.5, 0.5);
        let d = metric.distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 15.0).abs() < 1e-6);
    }

    #[test]
    fn composite_weight_one_zero_degenerates_to_a() {
        let metric = CompositeDistance::new(SquaredEuclidean, Euclidean, 1.0, 0.0);
        let a = [1.0, 2.0];
        let b = [4.0, 6.0];
        let expected = SquaredEuclidean.distance(&a, &b);
        assert!((metric.distance(&a, &b) - expected).abs() < 1e-6);
    }

    #[test]
    fn composite_weight_zero_one_degenerates_to_b() {
        let metric = CompositeDistance::new(SquaredEuclidean, Euclidean, 0.0, 1.0);
        let a = [1.0, 2.0];
        let b = [4.0, 6.0];
        let expected = Euclidean.distance(&a, &b);
        assert!((metric.distance(&a, &b) - expected).abs() < 1e-6);
    }

    #[test]
    fn composite_self_distance_zero() {
        let metric = CompositeDistance::new(SquaredEuclidean, Euclidean, 0.7, 0.3);
        let v = [1.0, 2.0, 3.0];
        assert!(metric.distance(&v, &v).abs() < 1e-6);
    }

    #[test]
    fn composite_symmetry() {
        let metric = CompositeDistance::new(SquaredEuclidean, CosineDistance, 0.6, 0.4);
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let d_ab = metric.distance(&a, &b);
        let d_ba = metric.distance(&b, &a);
        assert!((d_ab - d_ba).abs() < 1e-6);
    }

    #[test]
    fn composite_equal_weights() {
        // Equal weights: should be average of the two metrics.
        let metric = CompositeDistance::new(SquaredEuclidean, Euclidean, 1.0, 1.0);
        let a = [0.0f32];
        let b = [2.0f32];
        // SquaredEuclidean = 4.0, Euclidean = 2.0, sum = 6.0
        let d = metric.distance(&a, &b);
        assert!((d - 6.0).abs() < 1e-6);
    }
}
