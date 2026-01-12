//! AutoML Core
//!
//! Core implementations for automatic machine learning:
//! - Ensemble methods for combining predictions
//! - Grid search hyperparameter optimization
//! - Automatic model selection

mod ensemble;
mod hyperopt;
mod model_selection;

pub use ensemble::{combine_predictions, EnsembleAverager, EnsembleMedian, EnsembleWeighted};
pub use hyperopt::GridSearch;
pub use model_selection::AutoML;

// Re-export from API for convenience
pub use automl_api::{
    AutoMLConfig, AutoMLError, EnsembleMethod, GridSearchConfig, ModelType, OptimizationMetric,
    Result, SelectedModel,
};

// Re-export SPI traits
pub use automl_spi::{EnsembleCombiner, HyperparameterOptimizer, ModelSelectionResult, ModelSelector};
