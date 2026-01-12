//! Ensemble combination trait for AutoML.

/// Trait for ensemble combiners.
pub trait EnsembleCombiner {
    /// Combine multiple predictions into a single prediction.
    fn combine(&self, predictions: &[Vec<f64>], weights: Option<&[f64]>) -> Vec<f64>;
}
