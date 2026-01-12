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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_data_error() {
        let error = AutoMLError::InsufficientData {
            required: 100,
            actual: 10,
        };
        assert_eq!(
            error.to_string(),
            "Insufficient data: need at least 100 points, got 10"
        );
    }

    #[test]
    fn test_insufficient_data_edge_cases() {
        // Zero actual data
        let error = AutoMLError::InsufficientData {
            required: 10,
            actual: 0,
        };
        assert_eq!(
            error.to_string(),
            "Insufficient data: need at least 10 points, got 0"
        );

        // Large numbers
        let error = AutoMLError::InsufficientData {
            required: 1_000_000,
            actual: 999_999,
        };
        assert!(error.to_string().contains("1000000"));
        assert!(error.to_string().contains("999999"));
    }

    #[test]
    fn test_invalid_parameter_error() {
        let error = AutoMLError::InvalidParameter {
            name: "alpha".to_string(),
            reason: "must be between 0 and 1".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid parameter 'alpha': must be between 0 and 1"
        );
    }

    #[test]
    fn test_invalid_parameter_with_special_chars() {
        let error = AutoMLError::InvalidParameter {
            name: "learning_rate".to_string(),
            reason: "value -0.5 is negative".to_string(),
        };
        assert!(error.to_string().contains("learning_rate"));
        assert!(error.to_string().contains("-0.5"));
    }

    #[test]
    fn test_fit_error() {
        let error = AutoMLError::FitError("matrix is singular".to_string());
        assert_eq!(error.to_string(), "Model fitting failed: matrix is singular");
    }

    #[test]
    fn test_fit_error_empty_message() {
        let error = AutoMLError::FitError(String::new());
        assert_eq!(error.to_string(), "Model fitting failed: ");
    }

    #[test]
    fn test_convergence_failure_error() {
        let error = AutoMLError::ConvergenceFailure { iterations: 1000 };
        assert_eq!(
            error.to_string(),
            "Optimization failed to converge after 1000 iterations"
        );
    }

    #[test]
    fn test_convergence_failure_zero_iterations() {
        let error = AutoMLError::ConvergenceFailure { iterations: 0 };
        assert_eq!(
            error.to_string(),
            "Optimization failed to converge after 0 iterations"
        );
    }

    #[test]
    fn test_no_valid_models_error() {
        let error = AutoMLError::NoValidModels("all models produced NaN".to_string());
        assert_eq!(
            error.to_string(),
            "No models could be fitted to the data: all models produced NaN"
        );
    }

    #[test]
    fn test_numerical_error() {
        let error = AutoMLError::NumericalError("division by zero".to_string());
        assert_eq!(error.to_string(), "Numerical error: division by zero");
    }

    #[test]
    fn test_numerical_error_with_values() {
        let error = AutoMLError::NumericalError("overflow at value 1e308".to_string());
        assert!(error.to_string().contains("1e308"));
    }

    #[test]
    fn test_prediction_error() {
        let error = AutoMLError::PredictionError("model not fitted".to_string());
        assert_eq!(error.to_string(), "Prediction failed: model not fitted");
    }

    #[test]
    fn test_error_is_debug() {
        let error = AutoMLError::FitError("test".to_string());
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("FitError"));
    }

    #[test]
    fn test_error_implements_std_error() {
        fn assert_std_error<E: std::error::Error>() {}
        assert_std_error::<AutoMLError>();
    }

    #[test]
    fn test_all_variants_can_be_constructed() {
        let errors: Vec<AutoMLError> = vec![
            AutoMLError::InsufficientData {
                required: 10,
                actual: 5,
            },
            AutoMLError::InvalidParameter {
                name: "test".to_string(),
                reason: "test".to_string(),
            },
            AutoMLError::FitError("test".to_string()),
            AutoMLError::ConvergenceFailure { iterations: 100 },
            AutoMLError::NoValidModels("test".to_string()),
            AutoMLError::NumericalError("test".to_string()),
            AutoMLError::PredictionError("test".to_string()),
        ];

        // Verify all variants produce non-empty error messages
        for error in errors {
            assert!(!error.to_string().is_empty());
        }
    }
}
