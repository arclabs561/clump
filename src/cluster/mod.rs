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
//! ## Algorithms (implemented)
//!
//! ### K-means
//!
//! The classic algorithm: assign each point to the nearest centroid, then
//! update centroids to the mean of their points. Repeat.
//!
//! **Objective**: Minimize within-cluster sum of squares:
//!
//! ```text
//! J = Σ_k Σ_{x ∈ C_k} ||x - μ_k||²
//! ```
//!
//! **Assumptions**:
//! - Clusters are roughly spherical
//! - Clusters have similar sizes
//! - You know k in advance
//!
//! **When to use**: Fast initial exploration, or when you need hard assignments
//! and can accept the spherical assumption.
//!
//! ### DBSCAN
//!
//! Density-based clustering that can discover non-convex clusters and identify
//! outliers (noise points). DBSCAN does not require specifying the number of
//! clusters in advance.
//!
//! ## Usage
//!
//! ```rust
//! use clump::cluster::{Clustering, Dbscan, Kmeans};
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
//! ```

mod dbscan;
mod kmeans;
mod traits;

pub use dbscan::{Dbscan, DbscanExt, NOISE};
pub use kmeans::Kmeans;
pub use traits::Clustering;
