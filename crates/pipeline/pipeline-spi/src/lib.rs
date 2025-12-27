//! Pipeline Service Provider Interface
//!
//! Defines traits for data transformation pipelines.

use std::error::Error;

/// Result type for pipeline operations
pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Common trait for pipeline transformation steps
pub trait PipelineStep: Send + Sync {
    /// Fit the step to data (learn parameters)
    fn fit(&mut self, data: &[f64]);

    /// Transform data forward
    fn transform(&self, data: &[f64]) -> Result<Vec<f64>>;

    /// Inverse transform (undo the transformation)
    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>>;

    /// Name of this step
    fn name(&self) -> &str;
}
