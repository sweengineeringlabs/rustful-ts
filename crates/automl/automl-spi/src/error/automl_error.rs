//! AutoML error types.

use thiserror::Error;

/// Errors that can occur during AutoML operations.
#[derive(Error, Debug)]
pub enum AutoMLError {
    /// Insufficient data points for the operation.
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// Invalid parameter value.
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Model fitting failed.
    #[error("Model fitting failed: {0}")]
    FitError(String),

    /// Optimization failed to converge.
    #[error("Optimization failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },

    /// No valid models could be fitted.
    #[error("No models could be fitted to the data: {0}")]
    NoValidModels(String),

    /// Numerical computation error.
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Prediction error.
    #[error("Prediction failed: {0}")]
    PredictionError(String),
}
