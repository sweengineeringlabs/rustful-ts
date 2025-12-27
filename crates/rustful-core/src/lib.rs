//! # rustful-core
//!
//! High-performance time series prediction algorithms implemented in Rust.
//! This is the core library of the rustful-ts framework.
//!
//! ## Supported Algorithms
//!
//! ### Classical Statistical Methods
//! - **ARIMA** - AutoRegressive Integrated Moving Average
//! - **Exponential Smoothing** - Simple, Double, and Triple (Holt-Winters)
//! - **Moving Average** - Simple and Weighted
//!
//! ### Machine Learning Methods
//! - **Linear Regression** - With seasonal decomposition
//! - **K-Nearest Neighbors** - Time series adapted KNN
//!
//! ## Example
//!
//! ```rust
//! use rustful_core::prelude::*;
//!
//! let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
//! let mut arima = Arima::new(1, 1, 0).unwrap();
//! arima.fit(&data).unwrap();
//! let forecast = arima.predict(3).unwrap();
//! assert_eq!(forecast.len(), 3);
//! ```

pub mod algorithms;
pub mod utils;
mod error;

#[cfg(feature = "fetch")]
pub mod data;

pub use error::{TsError, Result};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::algorithms::Predictor;
    pub use crate::algorithms::arima::Arima;
    pub use crate::algorithms::exponential_smoothing::{
        SimpleExponentialSmoothing,
        DoubleExponentialSmoothing,
        HoltWinters,
        SeasonalType,
    };
    pub use crate::algorithms::moving_average::{SimpleMovingAverage, WeightedMovingAverage};
    pub use crate::algorithms::linear_regression::{LinearRegression, SeasonalLinearRegression};
    pub use crate::algorithms::knn::{TimeSeriesKNN, DistanceMetric};
    pub use crate::error::{TsError, Result};
}
