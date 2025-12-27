//! Time series prediction algorithms
//!
//! This module contains implementations of various time series forecasting algorithms,
//! ranging from classical statistical methods to machine learning approaches.

pub mod arima;
pub mod exponential_smoothing;
pub mod moving_average;
pub mod linear_regression;
pub mod knn;

/// Common trait for all time series predictors
pub trait Predictor {
    /// Fit the model to historical data
    fn fit(&mut self, data: &[f64]) -> crate::Result<()>;

    /// Predict future values
    fn predict(&self, steps: usize) -> crate::Result<Vec<f64>>;

    /// Check if the model has been fitted
    fn is_fitted(&self) -> bool;
}

/// Trait for models that support incremental updates
pub trait IncrementalPredictor: Predictor {
    /// Update the model with new data point(s)
    fn update(&mut self, data: &[f64]) -> crate::Result<()>;
}
