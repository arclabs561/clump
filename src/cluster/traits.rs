use crate::error::Result;

/// Common interface for hard clustering algorithms (one label per point).
pub trait Clustering {
    /// Fit the model (if needed) and return one cluster label per input point.
    fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>>;

    /// The configured number of clusters (if applicable).
    ///
    /// For algorithms that discover the number of clusters dynamically (e.g. DBSCAN),
    /// this returns 0.
    fn n_clusters(&self) -> usize;
}
