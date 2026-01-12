//! Financial analytics error types.

use thiserror::Error;

/// Financial analytics errors.
#[derive(Debug, Error)]
pub enum FinancialError {
    #[error("Insufficient data: required {required}, got {got}")]
    InsufficientData { required: usize, got: usize },

    #[error("Invalid parameter: {name} - {reason}")]
    InvalidParameter { name: String, reason: String },

    #[error("Portfolio error: {0}")]
    PortfolioError(String),

    #[error("Backtest error: {0}")]
    BacktestError(String),

    #[error("Risk calculation error: {0}")]
    RiskError(String),
}

/// Result type alias for financial operations.
pub type Result<T> = std::result::Result<T, FinancialError>;
