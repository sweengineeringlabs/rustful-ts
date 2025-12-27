//! Predictor Facade
//!
//! High-level API for time series prediction. Re-exports all public types
//! from the predictor stack for convenient usage.

// Re-export everything from API (which includes SPI and core)
pub use predictor_api::*;

// Explicit re-exports for documentation
pub use predictor_api::prelude;
pub use predictor_core::utils;
