//! Machine learning algorithms for time series
//!
//! This module contains ML-based approaches for pattern recognition and forecasting.
//!
//! ## Algorithms
//!
//! - **KNN**: K-Nearest Neighbors for pattern-based prediction

pub mod knn;

pub use knn::{DistanceMetric, TimeSeriesKNN};
