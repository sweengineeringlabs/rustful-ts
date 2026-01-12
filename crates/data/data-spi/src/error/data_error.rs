//! Data error types.

use thiserror::Error;

/// Data source errors.
#[derive(Debug, Error)]
pub enum DataError {
    /// HTTP request failed
    #[error("Request failed: {0}")]
    RequestFailed(String),

    /// Failed to parse response
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Invalid date format
    #[error("Invalid date: {0}")]
    InvalidDate(String),

    /// No data returned
    #[error("No data returned")]
    NoData,

    /// API error from data provider
    #[error("API error [{code}]: {description}")]
    ApiError { code: String, description: String },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Result type for data operations.
pub type Result<T> = std::result::Result<T, DataError>;
