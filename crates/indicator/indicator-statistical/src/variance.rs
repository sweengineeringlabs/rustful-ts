//! Variance implementation.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Variance.
///
/// Measures the dispersion of values from their mean (squared).
/// The square of standard deviation.
#[derive(Debug, Clone)]
pub struct Variance {
    period: usize,
    /// Use population variance (divide by n) vs sample (divide by n-1)
    population: bool,
}

impl Variance {
    /// Create a new Variance indicator with population variance.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            population: true,
        }
    }

    /// Create with population variance (divide by n).
    pub fn population(period: usize) -> Self {
        Self {
            period,
            population: true,
        }
    }

    /// Create with sample variance (divide by n-1).
    pub fn sample(period: usize) -> Self {
        Self {
            period,
            population: false,
        }
    }

    /// Calculate variance values.
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

            let variance = sum_sq / divisor;
            result.push(variance);
        }

        result
    }
}

impl TechnicalIndicator for Variance {
    fn name(&self) -> &str {
        "Variance"
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
    fn test_variance() {
        let variance = Variance::new(3);
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = variance.calculate(&data);

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
    fn test_variance_constant() {
        let variance = Variance::new(3);
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let result = variance.calculate(&data);

        // Variance of constant values should be 0
        for i in 2..data.len() {
            assert!((result[i] - 0.0).abs() < 1e-10);
        }
    }
}
