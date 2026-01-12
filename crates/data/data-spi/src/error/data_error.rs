//! Data error types.

use thiserror::Error;

/// Data source errors.
#[derive(Debug, Clone, Error)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_failed_error() {
        let error = DataError::RequestFailed("Connection timeout".to_string());
        assert_eq!(error.to_string(), "Request failed: Connection timeout");
    }

    #[test]
    fn test_parse_error() {
        let error = DataError::ParseError("Invalid JSON".to_string());
        assert_eq!(error.to_string(), "Parse error: Invalid JSON");
    }

    #[test]
    fn test_invalid_date_error() {
        let error = DataError::InvalidDate("2024-13-45".to_string());
        assert_eq!(error.to_string(), "Invalid date: 2024-13-45");
    }

    #[test]
    fn test_no_data_error() {
        let error = DataError::NoData;
        assert_eq!(error.to_string(), "No data returned");
    }

    #[test]
    fn test_api_error() {
        let error = DataError::ApiError {
            code: "404".to_string(),
            description: "Symbol not found".to_string(),
        };
        assert_eq!(error.to_string(), "API error [404]: Symbol not found");
    }

    #[test]
    fn test_config_error() {
        let error = DataError::ConfigError("Missing API key".to_string());
        assert_eq!(error.to_string(), "Configuration error: Missing API key");
    }

    #[test]
    fn test_error_debug_format() {
        let error = DataError::RequestFailed("test".to_string());
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("RequestFailed"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_result_type_ok() {
        let result: Result<i32> = Ok(42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_result_type_err() {
        let result: Result<i32> = Err(DataError::NoData);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DataError::NoData));
    }

    #[test]
    fn test_api_error_with_empty_code() {
        let error = DataError::ApiError {
            code: "".to_string(),
            description: "Unknown error".to_string(),
        };
        assert_eq!(error.to_string(), "API error []: Unknown error");
    }

    #[test]
    fn test_error_is_std_error() {
        let error: Box<dyn std::error::Error> =
            Box::new(DataError::RequestFailed("test".to_string()));
        assert_eq!(error.to_string(), "Request failed: test");
    }
}
