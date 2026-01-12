//! Forecast Facade
//!
//! High-level API for forecasting pipelines. Re-exports all public types
//! from the forecast stack for convenient usage.

// Re-export everything from API (which includes SPI and core)
pub use forecast_api::*;

// Explicit re-exports for documentation
pub use forecast_api::prelude;

// Re-export core modules for direct access
pub use forecast_core::{confidence, decomposition, pipeline, seasonality};

// Re-export pipeline types at root for backward compatibility
pub use forecast_core::pipeline::{
    DifferenceStep, NormalizeStep, Pipeline, StandardizeStep,
};

// Re-export SPI traits
pub use forecast_spi::{
    ConfidenceInterval, ConfidenceIntervalComputer, DecompositionResult, Decomposer,
    ForecastError, PipelineStep, SeasonalityDetector,
};
