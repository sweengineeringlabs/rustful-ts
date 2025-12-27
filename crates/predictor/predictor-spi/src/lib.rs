//! Predictor Service Provider Interface
//!
//! Defines traits for time series prediction algorithms.

use std::error::Error;

/// Result type for predictor operations
pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Common trait for all time series predictors
pub trait Predictor {
    /// Fit the model to historical data
    fn fit(&mut self, data: &[f64]) -> Result<()>;

    /// Predict future values
    fn predict(&self, steps: usize) -> Result<Vec<f64>>;

    /// Check if the model has been fitted
    fn is_fitted(&self) -> bool;
}

/// Trait for models that support incremental updates
pub trait IncrementalPredictor: Predictor {
    /// Update the model with new data point(s)
    fn update(&mut self, data: &[f64]) -> Result<()>;
}
