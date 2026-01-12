//! Algorithm Core Implementations
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
//! use algorithm_core::prelude::*;
//!
//! let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
//! let mut model = Arima::new(1, 1, 0).unwrap();
//! model.fit(&data).unwrap();
//! let forecast = model.predict(3).unwrap();
//! ```

pub mod ml;
pub mod regression;
pub mod smoothing;
pub mod utils;

// Re-export from SPI
pub use algorithm_spi::{IncrementalPredictor, Predictor, Result, TsError};

// Re-export implementations for convenience
pub use ml::*;
pub use regression::*;
pub use smoothing::*;

/// Prelude module for convenient imports
pub mod prelude {
    pub use algorithm_spi::Predictor;
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
    pub use algorithm_spi::{Result, TsError};
}
