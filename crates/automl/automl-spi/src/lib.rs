//! AutoML Service Provider Interface
//!
//! Defines traits for automatic machine learning:
//! - Model selection
//! - Hyperparameter optimization
//! - Ensemble methods

pub mod contract;
pub mod error;
pub mod model;

// Re-export all public items at the crate root for backward compatibility
pub use contract::{EnsembleCombiner, HyperparameterOptimizer, ModelSelector};
pub use error::AutoMLError;
pub use model::{ModelSelectionResult, ModelType, SelectedModel};

/// Result type for AutoML operations.
pub type Result<T> = std::result::Result<T, AutoMLError>;
