use thiserror::Error;

/// Errors returned by clustering algorithms in this crate.
#[derive(Debug, Error)]
pub enum Error {
    /// Input slice is empty.
    #[error("empty input")]
    EmptyInput,

    /// Invalid parameter value.
    #[error("invalid parameter {name}: {message}")]
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// Human-readable explanation.
        message: &'static str,
    },

    /// Requested cluster count is incompatible with the dataset.
    #[error("invalid cluster count: requested {requested}, but dataset has {n_items} items")]
    InvalidClusterCount {
        /// Requested number of clusters.
        requested: usize,
        /// Number of items in the dataset.
        n_items: usize,
    },

    /// Points in a dataset have inconsistent dimensionality.
    #[error("dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch {
        /// Expected dimensionality.
        expected: usize,
        /// Found dimensionality.
        found: usize,
    },

    /// Other error.
    #[error("{0}")]
    Other(String),
}

/// Result type used by this crate.
pub type Result<T> = std::result::Result<T, Error>;
