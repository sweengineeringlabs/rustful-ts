//! Model selection trait for AutoML.

use crate::error::AutoMLError;
use crate::model::ModelSelectionResult;

/// Result type for model selector operations.
pub type Result<T> = std::result::Result<T, AutoMLError>;

/// Trait for model selection strategies.
pub trait ModelSelector {
    /// Select the best model for the given data.
    fn select(&self, data: &[f64], horizon: usize) -> Result<ModelSelectionResult>;
}
