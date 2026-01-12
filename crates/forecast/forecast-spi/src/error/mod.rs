//! Error module containing error types and result aliases

mod forecast_error;

pub use forecast_error::ForecastError;

use std::error::Error;

/// Result type for forecast operations
pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
