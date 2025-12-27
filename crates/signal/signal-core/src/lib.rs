//! Signal Core
//!
//! Core types, error handling, and utilities for trading signals.

pub use signal_spi::{Signal, SignalGenerator};
use thiserror::Error;

/// Result type for signal operations
pub type Result<T> = std::result::Result<T, SignalError>;

/// Errors that can occur during signal generation
#[derive(Error, Debug)]
pub enum SignalError {
    /// Insufficient data points for the operation
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// Invalid parameter value
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },
}
