//! Time series error types
//!
//! Defines the standardized error type for all algorithm operations.

use thiserror::Error;

/// Result type alias for algorithm operations
pub type Result<T> = std::result::Result<T, TsError>;

/// Errors that can occur during time series operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TsError {
    /// Insufficient data points for the operation
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// Invalid parameter value
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Model has not been fitted yet
    #[error("Model must be fitted before prediction")]
    NotFitted,

    /// Convergence failure during optimization
    #[error("Optimization failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },

    /// Numerical computation error
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Invalid time series data
    #[error("Invalid data: {0}")]
    InvalidData(String),
}
