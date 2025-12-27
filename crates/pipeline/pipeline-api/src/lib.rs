//! Pipeline step implementations
//!
//! This crate provides implementations of various pipeline steps:
//!
//! - [`NormalizeStep`]: Normalize data to [0, 1] range
//! - [`StandardizeStep`]: Standardize to zero mean, unit variance
//! - [`DifferenceStep`]: Apply differencing

mod normalize;
mod standardize;
mod difference;

// Re-export from core
pub use pipeline_core::{Pipeline, PipelineError, Result};

// Re-export traits from SPI
pub use pipeline_spi::PipelineStep;

// Re-export implementations
pub use normalize::NormalizeStep;
pub use standardize::StandardizeStep;
pub use difference::DifferenceStep;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::PipelineStep;
    pub use crate::{NormalizeStep, StandardizeStep, DifferenceStep};
    pub use crate::{Pipeline, Result, PipelineError};
}
