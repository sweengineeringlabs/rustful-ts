//! Regression and statistical models for time series forecasting
//!
//! This module contains algorithms that fit mathematical models to data.
//!
//! ## Algorithms
//!
//! - **Linear Regression**: Basic and seasonal variants
//! - **ARIMA**: AutoRegressive Integrated Moving Average

pub mod arima;
pub mod linear;

pub use arima::Arima;
pub use linear::{LinearRegression, SeasonalLinearRegression};
