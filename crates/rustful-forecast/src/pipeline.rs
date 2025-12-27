//! Forecasting pipeline

use rustful_core::Result;

/// Pipeline step trait
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

/// Difference step
pub struct DifferenceStep {
    order: usize,
}

impl DifferenceStep {
    pub fn new(order: usize) -> Self {
        Self { order }
    }
}

impl PipelineStep for DifferenceStep {
    fn fit(&mut self, _data: &[f64]) {
        // No parameters to fit for differencing
    }

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

/// Standardize step (zero mean, unit variance)
pub struct StandardizeStep {
    mean: f64,
    std_dev: f64,
}

impl StandardizeStep {
    pub fn new() -> Self {
        Self { mean: 0.0, std_dev: 1.0 }
    }

    pub fn fit(&mut self, data: &[f64]) {
        let n = data.len() as f64;
        self.mean = data.iter().sum::<f64>() / n;
        self.std_dev = (data.iter().map(|x| (x - self.mean).powi(2)).sum::<f64>() / n).sqrt();
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
        Ok(data.iter().map(|&x| (x - self.mean) / self.std_dev).collect())
    }

    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        Ok(data.iter().map(|&x| x * self.std_dev + self.mean).collect())
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

    pub fn add_step(&mut self, step: Box<dyn PipelineStep>) {
        self.steps.push(step);
    }

    pub fn fit_transform(&mut self, data: &[f64]) -> Result<Vec<f64>> {
        let mut result = data.to_vec();
        for step in &mut self.steps {
            step.fit(&result);
            result = step.transform(&result)?;
        }
        Ok(result)
    }

    pub fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
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
