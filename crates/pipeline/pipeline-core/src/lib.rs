//! Pipeline Core
//!
//! Core types, error handling, and utilities for pipelines.

pub use pipeline_spi::{PipelineStep, Result as SpiResult};
use thiserror::Error;

/// Result type for pipeline operations
pub type Result<T> = std::result::Result<T, PipelineError>;

/// Errors that can occur during pipeline operations
#[derive(Error, Debug)]
pub enum PipelineError {
    /// Insufficient data points for the operation
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// Invalid parameter value
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Transformation failed
    #[error("Transformation failed: {0}")]
    TransformError(String),
}

/// Composable forecasting pipeline
pub struct Pipeline {
    steps: Vec<Box<dyn PipelineStep>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn add_step(&mut self, step: Box<dyn PipelineStep>) {
        self.steps.push(step);
    }

    pub fn fit_transform(&mut self, data: &[f64]) -> SpiResult<Vec<f64>> {
        let mut result = data.to_vec();
        for step in &mut self.steps {
            step.fit(&result);
            result = step.transform(&result)?;
        }
        Ok(result)
    }

    pub fn inverse_transform(&self, data: &[f64]) -> SpiResult<Vec<f64>> {
        let mut result = data.to_vec();
        for step in self.steps.iter().rev() {
            result = step.inverse_transform(&result)?;
        }
        Ok(result)
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}
