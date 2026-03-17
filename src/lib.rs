//! Dense clustering primitives.
//!
//! `clump` provides clustering algorithms for dense `f32` vectors, generic over
//! a pluggable distance metric.
//!
//! **Batch**: [`Kmeans`], [`Dbscan`], [`Hdbscan`], [`EVoC`], [`CopKmeans`],
//! [`CorrelationClustering`].
//!
//! **Streaming**: [`MiniBatchKmeans`], [`DenStream`].

#![cfg_attr(not(feature = "gpu"), forbid(unsafe_code))]
#![cfg_attr(feature = "gpu", deny(unsafe_code))]
#![warn(missing_docs)]

pub mod cluster;
/// Error types for clustering operations.
pub mod error;

pub use cluster::{
    ClusterHierarchy, ClusterLayer, ClusterNode, CompositeDistance, Constraint, CopKmeans,
    CorrelationClustering, CorrelationResult, CosineDistance, Dbscan, DenStream, DistanceMetric,
    EVoC, EVoCParams, Euclidean, Hdbscan, InnerProductDistance, Kmeans, KmeansFit, MiniBatchKmeans,
    SignedEdge, SquaredEuclidean, NOISE,
};
pub use error::{Error, Result};
