//! Forecast error types

use thiserror::Error;

/// Errors that can occur during forecasting operations
#[derive(Error, Debug)]
pub enum ForecastError {
    /// Insufficient data points for the operation
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// Invalid parameter value
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Pipeline step has not been fitted
    #[error("Pipeline step must be fitted before transformation")]
    NotFitted,

    /// Numerical computation error
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Invalid period for seasonality
    #[error("Invalid period: {0}")]
    InvalidPeriod(String),
}
