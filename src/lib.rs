//! Dense clustering primitives.
//!
//! 9 clustering algorithms for dense `f32` vectors, generic over pluggable
//! distance metrics. SIMD-accelerated (innr), with optional GPU (Metal) and
//! parallel (rayon) support.
//!
//! **Batch**: [`Kmeans`], [`Dbscan`], [`Hdbscan`], [`EVoC`], [`CopKmeans`],
//! [`CorrelationClustering`].
//!
//! **Streaming**: [`MiniBatchKmeans`], [`DenStream`].
//!
//! **Evaluation**: [`cluster::metrics`] -- silhouette score, Calinski-Harabasz,
//! Davies-Bouldin index.
//!
//! Noise points from DBSCAN/HDBSCAN are labeled with the sentinel
//! [`NOISE`] (`usize::MAX`).

#![cfg_attr(not(any(feature = "gpu", feature = "blas")), forbid(unsafe_code))]
#![cfg_attr(any(feature = "gpu", feature = "blas"), deny(unsafe_code))]
#![warn(missing_docs)]

pub mod cluster;
/// Error types for clustering operations.
pub mod error;

pub use cluster::{
    ClusterHierarchy, ClusterLayer, ClusterNode, CompositeDistance, Constraint, CopKmeans,
    CorrelationClustering, CorrelationResult, CosineDistance, Dbscan, DenStream, DistanceMetric,
    EVoC, EVoCParams, Euclidean, Hdbscan, HdbscanResult, InnerProductDistance, Kmeans, KmeansFit,
    MiniBatchKmeans, Optics, OpticsResult, SignedEdge, SquaredEuclidean, NOISE,
};
pub use error::{Error, Result};
