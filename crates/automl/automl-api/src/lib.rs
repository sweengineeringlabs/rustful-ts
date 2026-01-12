//! AutoML Consumer API
//!
//! Configuration types and DTOs for AutoML consumers.

use serde::{Deserialize, Serialize};

// Re-export SPI types
pub use automl_spi::{
    AutoMLError, EnsembleCombiner, HyperparameterOptimizer, ModelSelectionResult, ModelSelector,
    ModelType, Result, SelectedModel,
};

/// Optimization metric for model selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum OptimizationMetric {
    /// Mean Absolute Error
    #[default]
    MAE,
    /// Root Mean Squared Error
    RMSE,
    /// Mean Absolute Percentage Error
    MAPE,
}

/// Ensemble combination method
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum EnsembleMethod {
    /// Simple average of predictions
    #[default]
    Average,
    /// Weighted average of predictions
    WeightedAverage,
    /// Median of predictions
    Median,
}

/// Configuration for AutoML model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoMLConfig {
    /// Metric to optimize during model selection
    pub metric: OptimizationMetric,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Train/test split ratio (portion used for testing)
    pub test_ratio: f64,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            metric: OptimizationMetric::MAE,
            cv_folds: 5,
            max_iterations: 100,
            test_ratio: 0.2,
        }
    }
}

impl AutoMLConfig {
    /// Create a new configuration with the specified metric
    pub fn with_metric(metric: OptimizationMetric) -> Self {
        Self {
            metric,
            ..Default::default()
        }
    }

    /// Set the number of cross-validation folds
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds.max(2);
        self
    }

    /// Set the maximum optimization iterations
    pub fn max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations.max(1);
        self
    }

    /// Set the test ratio for train/test split
    pub fn test_ratio(mut self, ratio: f64) -> Self {
        self.test_ratio = ratio.clamp(0.1, 0.5);
        self
    }
}

/// Configuration for grid search optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSearchConfig {
    /// Metric to optimize
    pub metric: OptimizationMetric,
    /// Train/test split ratio
    pub test_ratio: f64,
}

impl Default for GridSearchConfig {
    fn default() -> Self {
        Self {
            metric: OptimizationMetric::MAE,
            test_ratio: 0.2,
        }
    }
}

impl GridSearchConfig {
    /// Create a new configuration with the specified metric
    pub fn with_metric(metric: OptimizationMetric) -> Self {
        Self {
            metric,
            ..Default::default()
        }
    }

    /// Set the test ratio
    pub fn test_ratio(mut self, ratio: f64) -> Self {
        self.test_ratio = ratio.clamp(0.1, 0.5);
        self
    }
}
