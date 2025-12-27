//! Time series prediction algorithms
//!
//! This crate provides implementations of various time series forecasting algorithms:
//!
//! - [`smoothing`]: Exponential smoothing, moving averages
//! - [`regression`]: Linear regression, ARIMA
//! - [`ml`]: Machine learning approaches (KNN)

pub mod ml;
pub mod regression;
pub mod smoothing;

// Re-export from core
pub use predictor_core::{utils, Result, TsError};

// Re-export traits from SPI
pub use predictor_spi::{IncrementalPredictor, Predictor};

// Re-export implementations for convenience
pub use ml::*;
pub use regression::*;
pub use smoothing::*;

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
