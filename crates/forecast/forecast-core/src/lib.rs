//! Forecast Core
//!
//! Core implementations for forecasting pipelines, decomposition,
//! seasonality detection, and confidence intervals.

pub mod confidence;
pub mod decomposition;
pub mod pipeline;
pub mod seasonality;

// Re-export SPI traits for implementations
pub use forecast_spi::{
    ConfidenceInterval, ConfidenceIntervalComputer, DecompositionResult, Decomposer,
    ForecastError, PipelineStep, Result, SeasonalityDetector,
};

// Re-export main types
pub use confidence::ForecastWithConfidence;
pub use decomposition::Decomposition;
pub use pipeline::Pipeline;
