//! Pipeline step trait for composable preprocessing

use crate::error::Result;

/// Pipeline step trait for composable preprocessing
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
