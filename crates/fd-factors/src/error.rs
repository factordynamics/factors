//! Error types for factor computations.

use thiserror::Error;

/// Result type for factor operations.
pub type Result<T> = std::result::Result<T, FactorError>;

/// Errors that can occur during factor computation.
#[derive(Debug, Error)]
pub enum FactorError {
    /// Missing required column in input data
    #[error("Missing required column: {0}")]
    MissingColumn(String),

    /// Insufficient data for lookback period
    #[error("Insufficient data: need {required} periods, got {available}")]
    InsufficientData {
        /// Required number of periods
        required: usize,
        /// Available number of periods
        available: usize,
    },

    /// Invalid date range
    #[error("Invalid date range: start {start} is after end {end}")]
    InvalidDateRange {
        /// Start date of the range
        start: String,
        /// End date of the range
        end: String,
    },

    /// Polars DataFrame error
    #[error("DataFrame error: {0}")]
    Polars(#[from] polars::error::PolarsError),

    /// Factor not found in registry
    #[error("Factor not found: {0}")]
    NotFound(String),

    /// Computation error
    #[error("Computation error: {0}")]
    Computation(String),
}
