//! Standard Deviation implementation.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Standard Deviation.
///
/// Measures the dispersion of values from their mean over a rolling window.
/// Higher values indicate greater volatility.
#[derive(Debug, Clone)]
pub struct StandardDeviation {
    period: usize,
    /// Use population standard deviation (divide by n) vs sample (divide by n-1)
    population: bool,
}

impl StandardDeviation {
    /// Create a new StandardDeviation indicator with sample standard deviation.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            population: true, // Use population std dev by default for trading
        }
    }

    /// Create with population standard deviation (divide by n).
    pub fn population(period: usize) -> Self {
        Self {
            period,
            population: true,
        }
    }

    /// Create with sample standard deviation (divide by n-1).
    pub fn sample(period: usize) -> Self {
        Self {
            period,
            population: false,
        }
    }

    /// Calculate standard deviation values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Calculate mean
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            // Calculate variance
            let sum_sq: f64 = window.iter().map(|x| (x - mean).powi(2)).sum();
            let divisor = if self.population {
                self.period as f64
            } else {
                (self.period - 1) as f64
            };

            let std_dev = (sum_sq / divisor).sqrt();
            result.push(std_dev);
        }

        result
    }
}

impl TechnicalIndicator for StandardDeviation {
    fn name(&self) -> &str {
        "StandardDeviation"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_dev() {
        let std_dev = StandardDeviation::new(3);
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = std_dev.calculate(&data);

        // First two values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // Check that values are reasonable
        for i in 2..data.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_std_dev_constant() {
        let std_dev = StandardDeviation::new(3);
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let result = std_dev.calculate(&data);

        // Standard deviation of constant values should be 0
        for i in 2..data.len() {
            assert!((result[i] - 0.0).abs() < 1e-10);
        }
    }
}
