//! Contract traits for algorithm implementations
//!
//! This module provides the core trait interfaces that all algorithm
//! implementations must adhere to:
//!
//! - [`Predictor`]: The primary trait for time series prediction
//! - [`IncrementalPredictor`]: Extension for online learning algorithms

mod predictor;

pub use predictor::{IncrementalPredictor, Predictor};
