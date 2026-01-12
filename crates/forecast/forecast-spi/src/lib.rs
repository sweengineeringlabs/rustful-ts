//! Forecast Service Provider Interface
//!
//! Defines traits for forecasting pipeline steps, decomposition, and seasonality detection.

pub mod contract;
pub mod error;
pub mod model;

// Re-export all public items at crate root for convenience
pub use contract::{
    ConfidenceIntervalComputer, Decomposer, PipelineStep, SeasonalityDetector,
};
pub use error::{ForecastError, Result};
pub use model::{ConfidenceInterval, DecompositionResult};
