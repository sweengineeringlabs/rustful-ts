//! Smoothing algorithms for time series data
//!
//! This module contains algorithms that smooth noisy data and extract trends.
//!
//! ## Algorithms
//!
//! - **Exponential Smoothing**: SES, Holt (double), Holt-Winters (triple)
//! - **Moving Averages**: Simple (SMA), Weighted (WMA)

pub mod exponential;
pub mod moving_average;

pub use exponential::{
    DoubleExponentialSmoothing, HoltWinters, SeasonalType, SimpleExponentialSmoothing,
};
pub use moving_average::{SimpleMovingAverage, WeightedMovingAverage};
