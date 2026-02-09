//! Dense clustering primitives.
//!
//! `clump` is a small, backend-agnostic library of clustering algorithms for dense vectors.
//!
//! The primary public API is under [`cluster`], which provides:
//! - k-means (k-means++ seeding, Lloyd iterations)
//! - DBSCAN (density clustering, with optional noise labeling)

#![forbid(unsafe_code)]

pub mod cluster;
pub mod error;

pub use cluster::{
    ClusterHierarchy, ClusterLayer, ClusterNode, Clustering, Dbscan, DbscanExt, EVoC, EVoCParams,
    Kmeans, KmeansFit, NOISE,
};
pub use error::{Error, Result};
