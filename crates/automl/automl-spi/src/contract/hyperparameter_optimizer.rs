//! Hyperparameter optimization trait for AutoML.

use crate::error::AutoMLError;

/// Result type for hyperparameter optimizer operations.
pub type Result<T> = std::result::Result<T, AutoMLError>;

/// Trait for hyperparameter optimizers.
pub trait HyperparameterOptimizer {
    /// The type of parameters being optimized.
    type Params;

    /// Optimize parameters for the given data.
    fn optimize(&self, data: &[f64], horizon: usize) -> Result<(Self::Params, f64)>;
}
