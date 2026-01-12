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

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // TsError Variant Creation Tests
    // ==========================================================================

    #[test]
    fn test_insufficient_data_error_creation() {
        let error = TsError::InsufficientData {
            required: 10,
            actual: 5,
        };

        match error {
            TsError::InsufficientData { required, actual } => {
                assert_eq!(required, 10);
                assert_eq!(actual, 5);
            }
            _ => panic!("Expected InsufficientData variant"),
        }
    }

    #[test]
    fn test_invalid_parameter_error_creation() {
        let error = TsError::InvalidParameter {
            name: "alpha".to_string(),
            reason: "must be between 0 and 1".to_string(),
        };

        match error {
            TsError::InvalidParameter { name, reason } => {
                assert_eq!(name, "alpha");
                assert_eq!(reason, "must be between 0 and 1");
            }
            _ => panic!("Expected InvalidParameter variant"),
        }
    }

    #[test]
    fn test_not_fitted_error_creation() {
        let error = TsError::NotFitted;
        assert_eq!(error, TsError::NotFitted);
    }

    #[test]
    fn test_convergence_failure_error_creation() {
        let error = TsError::ConvergenceFailure { iterations: 1000 };

        match error {
            TsError::ConvergenceFailure { iterations } => {
                assert_eq!(iterations, 1000);
            }
            _ => panic!("Expected ConvergenceFailure variant"),
        }
    }

    #[test]
    fn test_numerical_error_creation() {
        let error = TsError::NumericalError("division by zero".to_string());

        match error {
            TsError::NumericalError(msg) => {
                assert_eq!(msg, "division by zero");
            }
            _ => panic!("Expected NumericalError variant"),
        }
    }

    #[test]
    fn test_invalid_data_error_creation() {
        let error = TsError::InvalidData("contains NaN values".to_string());

        match error {
            TsError::InvalidData(msg) => {
                assert_eq!(msg, "contains NaN values");
            }
            _ => panic!("Expected InvalidData variant"),
        }
    }

    // ==========================================================================
    // TsError Display Tests
    // ==========================================================================

    #[test]
    fn test_insufficient_data_display() {
        let error = TsError::InsufficientData {
            required: 10,
            actual: 5,
        };
        let display = format!("{}", error);
        assert_eq!(display, "Insufficient data: need at least 10 points, got 5");
    }

    #[test]
    fn test_invalid_parameter_display() {
        let error = TsError::InvalidParameter {
            name: "alpha".to_string(),
            reason: "must be between 0 and 1".to_string(),
        };
        let display = format!("{}", error);
        assert_eq!(
            display,
            "Invalid parameter 'alpha': must be between 0 and 1"
        );
    }

    #[test]
    fn test_not_fitted_display() {
        let error = TsError::NotFitted;
        let display = format!("{}", error);
        assert_eq!(display, "Model must be fitted before prediction");
    }

    #[test]
    fn test_convergence_failure_display() {
        let error = TsError::ConvergenceFailure { iterations: 1000 };
        let display = format!("{}", error);
        assert_eq!(
            display,
            "Optimization failed to converge after 1000 iterations"
        );
    }

    #[test]
    fn test_numerical_error_display() {
        let error = TsError::NumericalError("division by zero".to_string());
        let display = format!("{}", error);
        assert_eq!(display, "Numerical error: division by zero");
    }

    #[test]
    fn test_invalid_data_display() {
        let error = TsError::InvalidData("contains NaN values".to_string());
        let display = format!("{}", error);
        assert_eq!(display, "Invalid data: contains NaN values");
    }

    // ==========================================================================
    // TsError Trait Implementation Tests
    // ==========================================================================

    #[test]
    fn test_error_is_debug() {
        let error = TsError::NotFitted;
        let debug = format!("{:?}", error);
        assert!(debug.contains("NotFitted"));
    }

    #[test]
    fn test_error_is_clone() {
        let error = TsError::InsufficientData {
            required: 10,
            actual: 5,
        };
        let cloned = error.clone();
        assert_eq!(error, cloned);
    }

    #[test]
    fn test_error_is_partial_eq() {
        let error1 = TsError::NotFitted;
        let error2 = TsError::NotFitted;
        let error3 = TsError::NumericalError("test".to_string());

        assert_eq!(error1, error2);
        assert_ne!(error1, error3);
    }

    #[test]
    fn test_error_implements_std_error() {
        let error: &dyn std::error::Error = &TsError::NotFitted;
        // Verify it implements std::error::Error by calling the trait method
        let _ = error.to_string();
    }

    // ==========================================================================
    // Result Type Alias Tests
    // ==========================================================================

    #[test]
    fn test_result_ok_variant() {
        let result: Result<i32> = Ok(42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_result_err_variant() {
        let result: Result<i32> = Err(TsError::NotFitted);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TsError::NotFitted);
    }

    #[test]
    fn test_result_with_different_types() {
        // Test with Vec<f64>
        let vec_result: Result<Vec<f64>> = Ok(vec![1.0, 2.0, 3.0]);
        assert!(vec_result.is_ok());

        // Test with ()
        let unit_result: Result<()> = Ok(());
        assert!(unit_result.is_ok());

        // Test with String
        let string_result: Result<String> = Ok("test".to_string());
        assert!(string_result.is_ok());
    }

    #[test]
    fn test_result_error_propagation() {
        fn inner_function() -> Result<i32> {
            Err(TsError::NotFitted)
        }

        fn outer_function() -> Result<i32> {
            inner_function()?;
            Ok(42)
        }

        let result = outer_function();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TsError::NotFitted);
    }

    #[test]
    fn test_result_map_operations() {
        let ok_result: Result<i32> = Ok(10);
        let mapped = ok_result.map(|x| x * 2);
        assert_eq!(mapped.unwrap(), 20);

        let err_result: Result<i32> = Err(TsError::NotFitted);
        let mapped_err = err_result.map(|x| x * 2);
        assert!(mapped_err.is_err());
    }

    // ==========================================================================
    // Edge Case Tests
    // ==========================================================================

    #[test]
    fn test_insufficient_data_with_zero_values() {
        let error = TsError::InsufficientData {
            required: 0,
            actual: 0,
        };
        let display = format!("{}", error);
        assert_eq!(display, "Insufficient data: need at least 0 points, got 0");
    }

    #[test]
    fn test_invalid_parameter_with_empty_strings() {
        let error = TsError::InvalidParameter {
            name: String::new(),
            reason: String::new(),
        };
        let display = format!("{}", error);
        assert_eq!(display, "Invalid parameter '': ");
    }

    #[test]
    fn test_convergence_failure_with_zero_iterations() {
        let error = TsError::ConvergenceFailure { iterations: 0 };
        let display = format!("{}", error);
        assert_eq!(display, "Optimization failed to converge after 0 iterations");
    }

    #[test]
    fn test_numerical_error_with_empty_message() {
        let error = TsError::NumericalError(String::new());
        let display = format!("{}", error);
        assert_eq!(display, "Numerical error: ");
    }

    #[test]
    fn test_invalid_data_with_empty_message() {
        let error = TsError::InvalidData(String::new());
        let display = format!("{}", error);
        assert_eq!(display, "Invalid data: ");
    }

    #[test]
    fn test_error_with_special_characters() {
        let error = TsError::InvalidParameter {
            name: "param\nwith\nnewlines".to_string(),
            reason: "contains \"quotes\" and 'apostrophes'".to_string(),
        };
        let display = format!("{}", error);
        assert!(display.contains("param\nwith\nnewlines"));
        assert!(display.contains("\"quotes\""));
    }

    #[test]
    fn test_error_with_unicode() {
        let error = TsError::InvalidData("contains unicode: \u{03B1} \u{03B2} \u{03B3}".to_string());
        let display = format!("{}", error);
        assert!(display.contains("\u{03B1}"));
    }

    #[test]
    fn test_large_values() {
        let error = TsError::InsufficientData {
            required: usize::MAX,
            actual: usize::MAX - 1,
        };
        // Should not panic
        let _ = format!("{}", error);
    }
}
