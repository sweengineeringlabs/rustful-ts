//! Standardize step

use pipeline_spi::{PipelineStep, Result as SpiResult};
use serde::{Deserialize, Serialize};

/// Standardize data to zero mean and unit variance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardizeStep {
    mean: f64,
    std_dev: f64,
}

impl StandardizeStep {
    pub fn new() -> Self {
        Self { mean: 0.0, std_dev: 1.0 }
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn std_dev(&self) -> f64 {
        self.std_dev
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

    fn transform(&self, data: &[f64]) -> SpiResult<Vec<f64>> {
        if self.std_dev == 0.0 {
            return Ok(vec![0.0; data.len()]);
        }
        Ok(data.iter().map(|&x| (x - self.mean) / self.std_dev).collect())
    }

    fn inverse_transform(&self, data: &[f64]) -> SpiResult<Vec<f64>> {
        Ok(data.iter().map(|&x| x * self.std_dev + self.mean).collect())
    }

    fn name(&self) -> &str {
        "standardize"
    }
}
