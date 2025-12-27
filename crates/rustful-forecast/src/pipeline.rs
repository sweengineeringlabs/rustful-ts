//! Forecasting pipeline

use rustful_core::Result;

/// Pipeline step trait
pub trait PipelineStep: Send + Sync {
    /// Transform data forward
    fn transform(&self, data: &[f64]) -> Result<Vec<f64>>;

    /// Inverse transform (undo the transformation)
    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>>;

    /// Name of this step
    fn name(&self) -> &str;
}

/// Normalize step
pub struct NormalizeStep {
    min: f64,
    max: f64,
}

impl NormalizeStep {
    pub fn new() -> Self {
        Self { min: 0.0, max: 1.0 }
    }

    pub fn fit(&mut self, data: &[f64]) {
        self.min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        self.max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }
}

impl Default for NormalizeStep {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStep for NormalizeStep {
    fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let range = self.max - self.min;
        if range == 0.0 {
            return Ok(vec![0.5; data.len()]);
        }
        Ok(data.iter().map(|&x| (x - self.min) / range).collect())
    }

    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let range = self.max - self.min;
        Ok(data.iter().map(|&x| x * range + self.min).collect())
    }

    fn name(&self) -> &str {
        "normalize"
    }
}

/// Difference step
pub struct DifferenceStep {
    order: usize,
    initial_values: Vec<f64>,
}

impl DifferenceStep {
    pub fn new(order: usize) -> Self {
        Self {
            order,
            initial_values: Vec::new(),
        }
    }
}

impl PipelineStep for DifferenceStep {
    fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        Ok(rustful_core::utils::preprocessing::difference(data, self.order))
    }

    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        // Note: proper undifferencing requires stored initial values
        Ok(data.to_vec())
    }

    fn name(&self) -> &str {
        "difference"
    }
}
