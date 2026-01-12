//! Trait for time series decomposition

use crate::model::DecompositionResult;

/// Trait for time series decomposition
pub trait Decomposer: Send + Sync {
    /// Decompose a time series into trend, seasonal, and residual components
    fn decompose(&self, data: &[f64], period: usize) -> DecompositionResult;
}
