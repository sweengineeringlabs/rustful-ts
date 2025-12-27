//! Normalize step

use pipeline_spi::{PipelineStep, Result as SpiResult};
use serde::{Deserialize, Serialize};

/// Normalize data to [0, 1] range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizeStep {
    min: f64,
    max: f64,
}

impl NormalizeStep {
    pub fn new() -> Self {
        Self { min: 0.0, max: 1.0 }
    }

    pub fn min(&self) -> f64 {
        self.min
    }

    pub fn max(&self) -> f64 {
        self.max
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

    fn transform(&self, data: &[f64]) -> SpiResult<Vec<f64>> {
        let range = self.max - self.min;
        if range == 0.0 {
            return Ok(vec![0.5; data.len()]);
        }
        Ok(data.iter().map(|&x| (x - self.min) / range).collect())
    }

    fn inverse_transform(&self, data: &[f64]) -> SpiResult<Vec<f64>> {
        let range = self.max - self.min;
        Ok(data.iter().map(|&x| x * range + self.min).collect())
    }

    fn name(&self) -> &str {
        "normalize"
    }
}
