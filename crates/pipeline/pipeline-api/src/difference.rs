//! Difference step

use pipeline_spi::{PipelineStep, Result as SpiResult};
use serde::{Deserialize, Serialize};

/// Apply differencing to time series
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    pub fn order(&self) -> usize {
        self.order
    }

    fn difference_once(data: &[f64]) -> Vec<f64> {
        if data.len() < 2 {
            return Vec::new();
        }
        data.windows(2).map(|w| w[1] - w[0]).collect()
    }
}

impl PipelineStep for DifferenceStep {
    fn fit(&mut self, data: &[f64]) {
        // Store initial values for inverse transform
        self.initial_values = data.iter().take(self.order).cloned().collect();
    }

    fn transform(&self, data: &[f64]) -> SpiResult<Vec<f64>> {
        let mut result = data.to_vec();
        for _ in 0..self.order {
            if result.len() < 2 {
                break;
            }
            result = Self::difference_once(&result);
        }
        Ok(result)
    }

    fn inverse_transform(&self, data: &[f64]) -> SpiResult<Vec<f64>> {
        // Note: Proper undifferencing requires stored initial values
        // This is a simplified version
        Ok(data.to_vec())
    }

    fn name(&self) -> &str {
        "difference"
    }
}
