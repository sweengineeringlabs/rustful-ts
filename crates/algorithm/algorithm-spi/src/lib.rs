//! Algorithm Service Provider Interface
//!
//! Defines core traits and error types for time series prediction algorithms.
//!
//! This crate provides the foundational abstractions that all algorithm
//! implementations must adhere to:
//!
//! - [`Predictor`]: The primary trait for time series prediction
//! - [`IncrementalPredictor`]: Extension for online learning algorithms
//! - [`TsError`]: Standardized error type for all algorithm operations
//! - [`Result`]: Convenient result type alias

pub mod contract;
pub mod error;
pub mod model;

// Re-export all public items at crate root for convenience
pub use contract::{IncrementalPredictor, Predictor};
pub use error::{Result, TsError};
