//! Forecast Consumer API
//!
//! Consumer configurations and builder APIs for forecasting pipelines.
//!
//! This crate provides:
//! - Configuration types for pipeline steps
//! - Builder patterns for constructing forecasting pipelines
//! - Re-exports from SPI and core for convenience

// Re-export from core
pub use forecast_core::{
    confidence, decomposition, pipeline, seasonality, Decomposition, ForecastWithConfidence,
    Pipeline,
};

// Re-export traits from SPI
pub use forecast_spi::{
    ConfidenceInterval, ConfidenceIntervalComputer, DecompositionResult, Decomposer,
    ForecastError, PipelineStep, Result, SeasonalityDetector,
};

use serde::{Deserialize, Serialize};

/// Configuration for pipeline construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Whether to normalize data
    pub normalize: bool,
    /// Whether to standardize data
    pub standardize: bool,
    /// Differencing order (0 for none)
    pub difference_order: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            normalize: false,
            standardize: false,
            difference_order: 0,
        }
    }
}

/// Configuration for decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionConfig {
    /// Seasonality period
    pub period: usize,
    /// Whether to use multiplicative decomposition
    pub multiplicative: bool,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            period: 12,
            multiplicative: false,
        }
    }
}

/// Configuration for confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceConfig {
    /// Confidence level (e.g., 0.95 for 95%)
    pub level: f64,
    /// Whether to use bootstrap method
    pub bootstrap: bool,
    /// Number of bootstrap samples (if bootstrap is true)
    pub n_samples: usize,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            level: 0.95,
            bootstrap: false,
            n_samples: 1000,
        }
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{ConfidenceConfig, DecompositionConfig, PipelineConfig};
    pub use forecast_core::{
        confidence, decomposition, pipeline, seasonality, Decomposition, ForecastWithConfidence,
        Pipeline,
    };
    pub use forecast_spi::{
        ConfidenceInterval, ConfidenceIntervalComputer, DecompositionResult, Decomposer,
        ForecastError, PipelineStep, Result, SeasonalityDetector,
    };
}
