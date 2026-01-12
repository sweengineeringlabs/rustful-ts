//! Volume Weighted Moving Average (VWMA) implementation.

use crate::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};

/// Volume Weighted Moving Average.
///
/// VWMA weights prices by volume, giving more importance to prices
/// with higher volume.
///
/// VWMA = Sum(Price * Volume) / Sum(Volume) over the period
#[derive(Debug, Clone)]
pub struct VWMA {
    period: usize,
}

impl VWMA {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate VWMA values.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < self.period {
            return result;
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let mut sum_pv = 0.0;
            let mut sum_v = 0.0;

            for j in start..=i {
                sum_pv += close[j] * volume[j];
                sum_v += volume[j];
            }

            result[i] = if sum_v > 0.0 { sum_pv / sum_v } else { close[i] };
        }

        result
    }
}

impl Default for VWMA {
    fn default() -> Self {
        Self { period: 20 }
    }
}

impl TechnicalIndicator for VWMA {
    fn name(&self) -> &str {
        "VWMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
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
    fn test_vwma() {
        let vwma = VWMA::new(3);
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let volume = vec![1000.0, 2000.0, 1500.0, 1800.0, 1200.0];

        let result = vwma.calculate(&close, &volume);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // First valid value at index 2
        assert!(!result[2].is_nan());
        // VWMA should be between min and max price in window
        assert!(result[2] >= 100.0 && result[2] <= 102.0);
    }

    #[test]
    fn test_vwma_equal_volume() {
        let vwma = VWMA::new(3);
        let close = vec![100.0, 102.0, 104.0];
        let volume = vec![1000.0, 1000.0, 1000.0];

        let result = vwma.calculate(&close, &volume);

        // With equal volume, VWMA = simple average
        let expected = (100.0 + 102.0 + 104.0) / 3.0;
        assert!((result[2] - expected).abs() < 1e-10);
    }
}
