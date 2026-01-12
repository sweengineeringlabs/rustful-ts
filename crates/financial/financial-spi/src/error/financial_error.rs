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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_data_error_display() {
        let error = FinancialError::InsufficientData {
            required: 100,
            got: 50,
        };
        assert_eq!(
            error.to_string(),
            "Insufficient data: required 100, got 50"
        );
    }

    #[test]
    fn test_insufficient_data_error_fields() {
        let error = FinancialError::InsufficientData {
            required: 200,
            got: 10,
        };
        match error {
            FinancialError::InsufficientData { required, got } => {
                assert_eq!(required, 200);
                assert_eq!(got, 10);
            }
            _ => panic!("Expected InsufficientData variant"),
        }
    }

    #[test]
    fn test_invalid_parameter_error_display() {
        let error = FinancialError::InvalidParameter {
            name: "window_size".to_string(),
            reason: "must be positive".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid parameter: window_size - must be positive"
        );
    }

    #[test]
    fn test_invalid_parameter_error_fields() {
        let error = FinancialError::InvalidParameter {
            name: "confidence".to_string(),
            reason: "must be between 0 and 1".to_string(),
        };
        match error {
            FinancialError::InvalidParameter { name, reason } => {
                assert_eq!(name, "confidence");
                assert_eq!(reason, "must be between 0 and 1");
            }
            _ => panic!("Expected InvalidParameter variant"),
        }
    }

    #[test]
    fn test_portfolio_error_display() {
        let error = FinancialError::PortfolioError("insufficient funds".to_string());
        assert_eq!(error.to_string(), "Portfolio error: insufficient funds");
    }

    #[test]
    fn test_portfolio_error_with_various_messages() {
        let messages = vec![
            "position not found",
            "duplicate symbol",
            "invalid quantity",
        ];
        for msg in messages {
            let error = FinancialError::PortfolioError(msg.to_string());
            assert_eq!(error.to_string(), format!("Portfolio error: {}", msg));
        }
    }

    #[test]
    fn test_backtest_error_display() {
        let error = FinancialError::BacktestError("no signals provided".to_string());
        assert_eq!(error.to_string(), "Backtest error: no signals provided");
    }

    #[test]
    fn test_backtest_error_with_various_messages() {
        let messages = vec![
            "price data mismatch",
            "empty equity curve",
            "invalid date range",
        ];
        for msg in messages {
            let error = FinancialError::BacktestError(msg.to_string());
            assert_eq!(error.to_string(), format!("Backtest error: {}", msg));
        }
    }

    #[test]
    fn test_risk_error_display() {
        let error = FinancialError::RiskError("VaR calculation failed".to_string());
        assert_eq!(
            error.to_string(),
            "Risk calculation error: VaR calculation failed"
        );
    }

    #[test]
    fn test_risk_error_with_various_messages() {
        let messages = vec![
            "insufficient return data",
            "invalid confidence level",
            "negative variance",
        ];
        for msg in messages {
            let error = FinancialError::RiskError(msg.to_string());
            assert_eq!(error.to_string(), format!("Risk calculation error: {}", msg));
        }
    }

    #[test]
    fn test_error_is_debug() {
        let error = FinancialError::InsufficientData {
            required: 10,
            got: 5,
        };
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("InsufficientData"));
        assert!(debug_str.contains("10"));
        assert!(debug_str.contains("5"));
    }

    #[test]
    fn test_result_type_alias_ok() {
        fn returns_ok() -> Result<i32> {
            Ok(42)
        }
        assert_eq!(returns_ok().unwrap(), 42);
    }

    #[test]
    fn test_result_type_alias_err() {
        fn returns_err() -> Result<i32> {
            Err(FinancialError::RiskError("test error".to_string()))
        }
        assert!(returns_err().is_err());
    }

    #[test]
    fn test_error_implements_std_error() {
        let error: Box<dyn std::error::Error> =
            Box::new(FinancialError::PortfolioError("test".to_string()));
        assert!(error.to_string().contains("Portfolio error"));
    }
}
