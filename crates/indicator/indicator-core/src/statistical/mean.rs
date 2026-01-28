//! Rolling Mean (Arithmetic Mean) implementation.
//!
//! Calculates the rolling arithmetic mean over a specified period.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Rolling Mean - Arithmetic mean over a period.
///
/// The arithmetic mean is the sum of values divided by the count.
/// This is equivalent to a Simple Moving Average (SMA) but categorized
/// as a statistical measure.
#[derive(Debug, Clone)]
pub struct Mean {
    period: usize,
}

impl Mean {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling mean.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        // Initial sum
        let mut sum: f64 = data[..self.period].iter().sum();
        result.push(sum / self.period as f64);

        // Rolling calculation
        for i in self.period..n {
            sum = sum - data[i - self.period] + data[i];
            result.push(sum / self.period as f64);
        }

        result
    }
}

impl TechnicalIndicator for Mean {
    fn name(&self) -> &str {
        "Mean"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mean = Mean::new(3);
        let result = mean.calculate(&data);

        assert_eq!(result.len(), 10);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 0.001); // (1+2+3)/3 = 2
        assert!((result[3] - 3.0).abs() < 0.001); // (2+3+4)/3 = 3
        assert!((result[9] - 9.0).abs() < 0.001); // (8+9+10)/3 = 9
    }

    #[test]
    fn test_mean_single_period() {
        let data = vec![5.0, 10.0, 15.0, 20.0];
        let mean = Mean::new(1);
        let result = mean.calculate(&data);

        assert_eq!(result.len(), 4);
        assert!((result[0] - 5.0).abs() < 0.001);
        assert!((result[3] - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_mean_full_period() {
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let mean = Mean::new(5);
        let result = mean.calculate(&data);

        assert_eq!(result.len(), 5);
        assert!(result[3].is_nan());
        assert!((result[4] - 6.0).abs() < 0.001); // (2+4+6+8+10)/5 = 6
    }
}
