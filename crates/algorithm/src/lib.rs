//! Time series prediction algorithms
//!
//! This crate provides implementations of various time series forecasting algorithms,
//! organized by category:
//!
//! - [`smoothing`]: Exponential smoothing, moving averages
//! - [`regression`]: Linear regression, ARIMA
//! - [`ml`]: Machine learning approaches (KNN)
//! - [`utils`]: Metrics, preprocessing, validation
//!
//! ## Example
//!
//! ```rust
//! use algorithm::prelude::*;
//!
//! let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
//! let mut model = Arima::new(1, 1, 0).unwrap();
//! model.fit(&data).unwrap();
//! let forecast = model.predict(3).unwrap();
//! ```

mod error;
pub mod ml;
pub mod regression;
pub mod smoothing;
pub mod utils;

pub use error::{Result, TsError};

// Re-export for convenience
pub use ml::*;
pub use regression::*;
pub use smoothing::*;

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

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::Predictor;
    // Smoothing
    pub use crate::smoothing::{
        DoubleExponentialSmoothing, HoltWinters, SeasonalType, SimpleExponentialSmoothing,
        SimpleMovingAverage, WeightedMovingAverage,
    };
    // Regression
    pub use crate::regression::{Arima, LinearRegression, SeasonalLinearRegression};
    // ML
    pub use crate::ml::{DistanceMetric, TimeSeriesKNN};
    // Error types
    pub use crate::{Result, TsError};
}
