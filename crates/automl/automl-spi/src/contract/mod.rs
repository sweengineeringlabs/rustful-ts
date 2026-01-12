//! Contract module containing AutoML traits.
//!
//! This module defines the core traits for automatic machine learning:
//! - [`ModelSelector`] - Model selection strategies
//! - [`HyperparameterOptimizer`] - Parameter optimization
//! - [`EnsembleCombiner`] - Ensemble methods

mod ensemble_combiner;
mod hyperparameter_optimizer;
mod model_selector;

pub use ensemble_combiner::EnsembleCombiner;
pub use hyperparameter_optimizer::HyperparameterOptimizer;
pub use model_selector::ModelSelector;
