//! Simple Moving Average implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::SMAConfig;

/// Simple Moving Average (SMA).
///
/// Calculates the unweighted mean of the previous n data points.
#[derive(Debug, Clone)]
pub struct SMA {
    period: usize,
}

impl SMA {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: SMAConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate SMA values.
    #[inline]
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        #[cfg(feature = "simd")]
        {
            return crate::simd::sma_simd(data, self.period);
        }

        #[cfg(not(feature = "simd"))]
        {
            self.calculate_scalar(data)
        }
    }

    /// Scalar implementation.
    fn calculate_scalar(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.period || self.period == 0 {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; self.period - 1];
        let mut sum: f64 = data[0..self.period].iter().sum();
        result.push(sum / self.period as f64);

        for i in self.period..data.len() {
            sum = sum - data[i - self.period] + data[i];
            result.push(sum / self.period as f64);
        }

        result
    }
}

impl TechnicalIndicator for SMA {
    fn name(&self) -> &str {
        "SMA"
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
    fn test_sma() {
        let sma = SMA::new(3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma.calculate(&data);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }
}
