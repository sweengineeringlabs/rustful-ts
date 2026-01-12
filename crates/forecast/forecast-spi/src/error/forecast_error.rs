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

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_insufficient_data_error_message() {
        let error = ForecastError::InsufficientData {
            required: 100,
            actual: 10,
        };
        assert_eq!(
            error.to_string(),
            "Insufficient data: need at least 100 points, got 10"
        );
    }

    #[test]
    fn test_insufficient_data_error_fields() {
        let error = ForecastError::InsufficientData {
            required: 50,
            actual: 25,
        };
        if let ForecastError::InsufficientData { required, actual } = error {
            assert_eq!(required, 50);
            assert_eq!(actual, 25);
        } else {
            panic!("Expected InsufficientData variant");
        }
    }

    #[test]
    fn test_invalid_parameter_error_message() {
        let error = ForecastError::InvalidParameter {
            name: "window_size".to_string(),
            reason: "must be positive".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid parameter 'window_size': must be positive"
        );
    }

    #[test]
    fn test_invalid_parameter_error_fields() {
        let error = ForecastError::InvalidParameter {
            name: "alpha".to_string(),
            reason: "must be between 0 and 1".to_string(),
        };
        if let ForecastError::InvalidParameter { name, reason } = error {
            assert_eq!(name, "alpha");
            assert_eq!(reason, "must be between 0 and 1");
        } else {
            panic!("Expected InvalidParameter variant");
        }
    }

    #[test]
    fn test_not_fitted_error_message() {
        let error = ForecastError::NotFitted;
        assert_eq!(
            error.to_string(),
            "Pipeline step must be fitted before transformation"
        );
    }

    #[test]
    fn test_numerical_error_message() {
        let error = ForecastError::NumericalError("division by zero".to_string());
        assert_eq!(error.to_string(), "Numerical error: division by zero");
    }

    #[test]
    fn test_numerical_error_with_various_messages() {
        let messages = vec![
            "overflow detected",
            "NaN encountered",
            "matrix is singular",
            "convergence failed after 1000 iterations",
        ];

        for msg in messages {
            let error = ForecastError::NumericalError(msg.to_string());
            assert_eq!(error.to_string(), format!("Numerical error: {}", msg));
        }
    }

    #[test]
    fn test_invalid_period_error_message() {
        let error = ForecastError::InvalidPeriod("period must be at least 2".to_string());
        assert_eq!(
            error.to_string(),
            "Invalid period: period must be at least 2"
        );
    }

    #[test]
    fn test_invalid_period_with_various_messages() {
        let messages = vec![
            "period cannot be zero",
            "period exceeds data length",
            "negative period not allowed",
        ];

        for msg in messages {
            let error = ForecastError::InvalidPeriod(msg.to_string());
            assert_eq!(error.to_string(), format!("Invalid period: {}", msg));
        }
    }

    #[test]
    fn test_error_implements_std_error() {
        let error: Box<dyn Error> = Box::new(ForecastError::NotFitted);
        assert!(error.source().is_none());
    }

    #[test]
    fn test_error_debug_impl() {
        let error = ForecastError::InsufficientData {
            required: 10,
            actual: 5,
        };
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("InsufficientData"));
        assert!(debug_str.contains("10"));
        assert!(debug_str.contains("5"));
    }

    #[test]
    fn test_all_variants_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ForecastError>();
    }

    #[test]
    fn test_all_variants_are_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ForecastError>();
    }

    #[test]
    fn test_error_can_be_boxed() {
        let error = ForecastError::NotFitted;
        let boxed: Box<dyn Error + Send + Sync> = Box::new(error);
        assert_eq!(
            boxed.to_string(),
            "Pipeline step must be fitted before transformation"
        );
    }

    #[test]
    fn test_error_downcast() {
        let error: Box<dyn Error> = Box::new(ForecastError::NotFitted);
        let downcasted = error.downcast_ref::<ForecastError>();
        assert!(downcasted.is_some());
        assert!(matches!(downcasted.unwrap(), ForecastError::NotFitted));
    }

    #[test]
    fn test_insufficient_data_edge_cases() {
        // Zero values
        let error = ForecastError::InsufficientData {
            required: 0,
            actual: 0,
        };
        assert_eq!(
            error.to_string(),
            "Insufficient data: need at least 0 points, got 0"
        );

        // Large values
        let error = ForecastError::InsufficientData {
            required: usize::MAX,
            actual: 0,
        };
        assert!(error.to_string().contains("got 0"));
    }

    #[test]
    fn test_invalid_parameter_empty_strings() {
        let error = ForecastError::InvalidParameter {
            name: String::new(),
            reason: String::new(),
        };
        assert_eq!(error.to_string(), "Invalid parameter '': ");
    }

    #[test]
    fn test_invalid_parameter_special_characters() {
        let error = ForecastError::InvalidParameter {
            name: "param\nwith\nnewlines".to_string(),
            reason: "contains 'quotes' and \"double quotes\"".to_string(),
        };
        let msg = error.to_string();
        assert!(msg.contains("param\nwith\nnewlines"));
        assert!(msg.contains("contains 'quotes' and \"double quotes\""));
    }
}
