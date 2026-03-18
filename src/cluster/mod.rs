//! Clustering algorithms for grouping similar items.
//!
//! This module provides clustering algorithms for dense vectors.
//!
//! ## Hard vs Soft Clustering
//!
//! **Hard clustering** assigns each item to exactly one cluster. Simple, but
//! loses information when items genuinely span multiple groups.
//!
//! **Soft clustering** gives each item a probability distribution over clusters.
//! A text chunk might be 60% about "machine learning", 30% about "statistics",
//! 10% about "software". This reflects reality better than forcing a choice.
//!
//! Soft clustering (e.g., GMM) is not implemented in this crate yet.
//!
//! ## Algorithms
//!
//! **Batch**
//! - [`Kmeans`]: Lloyd's algorithm with k-means++ seeding, generic over distance metrics.
//! - [`Dbscan`]: density-based clustering with noise detection (Ester et al. 1996).
//! - [`Hdbscan`]: hierarchical density clustering without a global epsilon (Campello et al. 2013).
//! - [`EVoC`]: multi-granularity hierarchy via MST on random projections.
//! - [`CopKmeans`]: constrained k-means with must-link / cannot-link (Wagstaff et al. 2001).
//! - [`CorrelationClustering`]: PIVOT + local search on signed graphs (Bansal et al. 2004).
//!
//! **Streaming**
//! - [`MiniBatchKmeans`]: online k-means with decaying learning rate (Sculley 2010).
//! - [`DenStream`]: streaming density-based clustering with decay (Cao et al. 2006).
//!
//! ## Usage
//!
//! ```rust
//! use clump::cluster::{Dbscan, EVoC, EVoCParams, Kmeans};
//!
//! let data = vec![
//!     vec![0.0, 0.0],
//!     vec![0.1, 0.1],
//!     vec![10.0, 10.0],
//!     vec![10.1, 10.1],
//! ];
//!
//! // Hard clustering with K-means
//! let labels = Kmeans::new(2).fit_predict(&data).unwrap();
//! assert_eq!(labels[0], labels[1]);  // First two together
//! assert_ne!(labels[0], labels[2]);  // Separate from last two
//!
//! // Density-based clustering with DBSCAN
//! let labels = Dbscan::new(0.5, 2).fit_predict(&data).unwrap();
//! assert_eq!(labels.len(), data.len());
//!
//! // Hierarchical clustering with EVōC (noise as `None`)
//! let mut evoc = EVoC::new(EVoCParams {
//!     intermediate_dim: 1,
//!     min_cluster_size: 2,
//!     seed: Some(42),
//!     ..Default::default()
//! });
//! let labels = evoc.fit_predict(&data).unwrap();
//! assert_eq!(labels.len(), data.len());
//! ```

pub mod constrained;
pub mod correlation;
mod dbscan;
pub mod denstream;
pub mod distance;
mod evoc;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
mod hdbscan;
mod kmeans;
pub mod streaming;
mod util;
mod vptree;

pub use constrained::{Constraint, CopKmeans};
pub use correlation::{CorrelationClustering, CorrelationResult, SignedEdge};
pub use dbscan::{Dbscan, NOISE};
pub use denstream::DenStream;
pub use distance::{
    CompositeDistance, CosineDistance, DistanceMetric, Euclidean, InnerProductDistance,
    SquaredEuclidean,
};
pub use evoc::{ClusterHierarchy, ClusterLayer, ClusterNode, EVoC, EVoCParams};
pub use hdbscan::Hdbscan;
pub use kmeans::{Kmeans, KmeansFit};
pub use streaming::MiniBatchKmeans;
