//! Anomaly detection error types.

use thiserror::Error;

/// Anomaly detection errors.
#[derive(Debug, Error)]
pub enum AnomalyError {
    #[error("Insufficient data: required {required}, got {got}")]
    InsufficientData { required: usize, got: usize },

    #[error("Detector not fitted: call fit() before detect()")]
    NotFitted,

    #[error("Invalid parameter: {name} - {reason}")]
    InvalidParameter { name: String, reason: String },

    #[error("Detection error: {0}")]
    DetectionError(String),
}

/// Result type for anomaly detection operations.
pub type Result<T> = std::result::Result<T, AnomalyError>;
