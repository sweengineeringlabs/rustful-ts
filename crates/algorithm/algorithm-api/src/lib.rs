//! Algorithm Consumer API
//!
//! This crate provides consumer-facing types and configurations for the
//! algorithm domain. Currently re-exports the SPI traits and types.
//!
//! # Re-exports
//!
//! All types from [`algorithm_spi`] are re-exported for convenience:
//!
//! - [`Predictor`]: Core prediction trait
//! - [`IncrementalPredictor`]: Incremental update trait
//! - [`TsError`]: Error type
//! - [`Result`]: Result type alias

// Re-export all SPI types
pub use algorithm_spi::{IncrementalPredictor, Predictor, Result, TsError};
