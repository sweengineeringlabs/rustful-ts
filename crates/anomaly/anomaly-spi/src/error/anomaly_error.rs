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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_data_display() {
        let error = AnomalyError::InsufficientData {
            required: 100,
            got: 10,
        };
        assert_eq!(
            error.to_string(),
            "Insufficient data: required 100, got 10"
        );
    }

    #[test]
    fn test_insufficient_data_zero_got() {
        let error = AnomalyError::InsufficientData {
            required: 50,
            got: 0,
        };
        assert_eq!(error.to_string(), "Insufficient data: required 50, got 0");
    }

    #[test]
    fn test_not_fitted_display() {
        let error = AnomalyError::NotFitted;
        assert_eq!(
            error.to_string(),
            "Detector not fitted: call fit() before detect()"
        );
    }

    #[test]
    fn test_invalid_parameter_display() {
        let error = AnomalyError::InvalidParameter {
            name: "threshold".to_string(),
            reason: "must be positive".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid parameter: threshold - must be positive"
        );
    }

    #[test]
    fn test_invalid_parameter_empty_name() {
        let error = AnomalyError::InvalidParameter {
            name: String::new(),
            reason: "value required".to_string(),
        };
        assert_eq!(error.to_string(), "Invalid parameter:  - value required");
    }

    #[test]
    fn test_invalid_parameter_special_characters() {
        let error = AnomalyError::InvalidParameter {
            name: "window_size".to_string(),
            reason: "must be in range [1, 1000]".to_string(),
        };
        assert_eq!(
            error.to_string(),
            "Invalid parameter: window_size - must be in range [1, 1000]"
        );
    }

    #[test]
    fn test_detection_error_display() {
        let error = AnomalyError::DetectionError("numerical overflow".to_string());
        assert_eq!(error.to_string(), "Detection error: numerical overflow");
    }

    #[test]
    fn test_detection_error_empty_message() {
        let error = AnomalyError::DetectionError(String::new());
        assert_eq!(error.to_string(), "Detection error: ");
    }

    #[test]
    fn test_detection_error_long_message() {
        let long_msg = "a".repeat(1000);
        let error = AnomalyError::DetectionError(long_msg.clone());
        assert_eq!(error.to_string(), format!("Detection error: {}", long_msg));
    }

    #[test]
    fn test_error_is_debug() {
        let error = AnomalyError::NotFitted;
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("NotFitted"));
    }

    #[test]
    fn test_insufficient_data_debug() {
        let error = AnomalyError::InsufficientData {
            required: 10,
            got: 5,
        };
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("InsufficientData"));
        assert!(debug_str.contains("10"));
        assert!(debug_str.contains("5"));
    }

    #[test]
    fn test_result_type_ok() {
        let result: Result<i32> = Ok(42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_result_type_err() {
        let result: Result<i32> = Err(AnomalyError::NotFitted);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AnomalyError::NotFitted));
    }

    #[test]
    fn test_error_implements_std_error() {
        let error: Box<dyn std::error::Error> =
            Box::new(AnomalyError::DetectionError("test".to_string()));
        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn test_all_error_variants_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        // This will fail to compile if AnomalyError is not Send + Sync
        assert_send_sync::<AnomalyError>();
    }
}
