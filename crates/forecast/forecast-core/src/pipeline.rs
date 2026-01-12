//! Forecasting pipeline implementations
//!
//! Composable preprocessing steps for time series forecasting.

use forecast_spi::{PipelineStep, Result};

/// Normalize step - scales data to [0, 1] range
pub struct NormalizeStep {
    min: f64,
    max: f64,
}

impl NormalizeStep {
    pub fn new() -> Self {
        Self { min: 0.0, max: 1.0 }
    }
}

impl Default for NormalizeStep {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStep for NormalizeStep {
    fn fit(&mut self, data: &[f64]) {
        self.min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        self.max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    }

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

/// Difference step - computes n-th order differences
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
    fn fit(&mut self, data: &[f64]) {
        // Store initial values needed for inverse transform
        self.initial_values = data.iter().take(self.order).copied().collect();
    }

    fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mut result = data.to_vec();
        for _ in 0..self.order {
            if result.len() < 2 {
                return Ok(vec![]);
            }
            result = result.windows(2).map(|w| w[1] - w[0]).collect();
        }
        Ok(result)
    }

    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        // Note: proper undifferencing requires stored initial values
        // This is a simplified implementation
        Ok(data.to_vec())
    }

    fn name(&self) -> &str {
        "difference"
    }
}

/// Standardize step - zero mean, unit variance
pub struct StandardizeStep {
    mean: f64,
    std_dev: f64,
}

impl StandardizeStep {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
        }
    }
}

impl Default for StandardizeStep {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStep for StandardizeStep {
    fn fit(&mut self, data: &[f64]) {
        let n = data.len() as f64;
        self.mean = data.iter().sum::<f64>() / n;
        self.std_dev = (data.iter().map(|x| (x - self.mean).powi(2)).sum::<f64>() / n).sqrt();
    }

    fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        if self.std_dev == 0.0 {
            return Ok(vec![0.0; data.len()]);
        }
        Ok(data
            .iter()
            .map(|&x| (x - self.mean) / self.std_dev)
            .collect())
    }

    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        Ok(data
            .iter()
            .map(|&x| x * self.std_dev + self.mean)
            .collect())
    }

    fn name(&self) -> &str {
        "standardize"
    }
}

/// Composable forecasting pipeline
pub struct Pipeline {
    steps: Vec<Box<dyn PipelineStep>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step to the pipeline
    pub fn add_step(&mut self, step: Box<dyn PipelineStep>) {
        self.steps.push(step);
    }

    /// Fit all steps and transform data
    pub fn fit_transform(&mut self, data: &[f64]) -> Result<Vec<f64>> {
        let mut result = data.to_vec();
        for step in &mut self.steps {
            step.fit(&result);
            result = step.transform(&result)?;
        }
        Ok(result)
    }

    /// Transform data (steps must be fitted first)
    pub fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mut result = data.to_vec();
        for step in &self.steps {
            result = step.transform(&result)?;
        }
        Ok(result)
    }

    /// Inverse transform data (undo all transformations in reverse order)
    pub fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mut result = data.to_vec();
        for step in self.steps.iter().rev() {
            result = step.inverse_transform(&result)?;
        }
        Ok(result)
    }

    /// Get the number of steps in the pipeline
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_step() {
        let mut step = NormalizeStep::new();
        let data = vec![0.0, 50.0, 100.0];
        step.fit(&data);

        let transformed = step.transform(&data).unwrap();
        assert_eq!(transformed, vec![0.0, 0.5, 1.0]);

        let inverse = step.inverse_transform(&transformed).unwrap();
        assert_eq!(inverse, data);
    }

    #[test]
    fn test_standardize_step() {
        let mut step = StandardizeStep::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        step.fit(&data);

        let transformed = step.transform(&data).unwrap();
        let mean: f64 = transformed.iter().sum::<f64>() / transformed.len() as f64;
        assert!((mean).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline() {
        let mut pipeline = Pipeline::new();
        pipeline.add_step(Box::new(NormalizeStep::new()));

        let data = vec![0.0, 50.0, 100.0];
        let transformed = pipeline.fit_transform(&data).unwrap();
        assert_eq!(transformed, vec![0.0, 0.5, 1.0]);
    }
}
