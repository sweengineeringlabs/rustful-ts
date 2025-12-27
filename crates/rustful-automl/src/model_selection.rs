//! Automatic model selection

use serde::{Deserialize, Serialize};

/// Model type for AutoML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Arima { p: usize, d: usize, q: usize },
    SES { alpha: f64 },
    Holt { alpha: f64, beta: f64 },
    HoltWinters { alpha: f64, beta: f64, gamma: f64, period: usize },
    LinearRegression,
    KNN { k: usize, window: usize },
}

/// Optimization metric
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationMetric {
    MAE,
    RMSE,
    MAPE,
}

/// AutoML configuration
#[derive(Debug, Clone)]
pub struct AutoMLConfig {
    pub metric: OptimizationMetric,
    pub cv_folds: usize,
    pub max_iterations: usize,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            metric: OptimizationMetric::MAE,
            cv_folds: 5,
            max_iterations: 100,
        }
    }
}

/// Result of model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionResult {
    pub best_model: ModelType,
    pub score: f64,
    pub all_scores: Vec<(ModelType, f64)>,
}
