//! # clump
//!
//! Dense clustering primitives: K-Means, DBSCAN.
//!
//! (clump: a compacted mass or group)
//!
//! ## Modules
//!
//! - `kmeans`: K-Means clustering (Lloyd's and Elkan's algorithms)
//! - `dbscan`: Density-based spatial clustering
//! - (planned) `gmm`: Gaussian Mixture Models
//!
//! ## Quick Start
//!
//! ```rust
//! use clump::cluster::{Clustering, Kmeans};
//!
//! let data = vec![vec![1.0, 2.0], vec![1.1, 2.1], vec![5.0, 5.0]];
//! let kmeans = Kmeans::new(2).with_seed(42);
//! let labels = kmeans.fit_predict(&data).unwrap();
//! assert_eq!(labels.len(), data.len());
//! ```
#![warn(missing_docs)]

/// Clustering algorithms.
pub mod cluster;
/// Error types for this crate.
pub mod error;

/// Convenience re-export of [`error::Error`].
pub use error::{Error, Result};
