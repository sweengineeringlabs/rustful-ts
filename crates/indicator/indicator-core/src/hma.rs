//! Hull Moving Average (HMA) implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::HMAConfig;
use crate::WMA;

/// Hull Moving Average.
///
/// HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
/// Provides very smooth averaging with minimal lag.
#[derive(Debug, Clone)]
pub struct HMA {
    period: usize,
}

impl HMA {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: HMAConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate HMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();

        if self.period <= 1 {
            return vec![f64::NAN; n];
        }

        let half_period = self.period / 2;
        let sqrt_period = (self.period as f64).sqrt() as usize;

        let wma_half = WMA::new(half_period.max(1));
        let wma_full = WMA::new(self.period);
        let wma_sqrt = WMA::new(sqrt_period.max(1));

        let wma_half_values = wma_half.calculate(data);
        let wma_full_values = wma_full.calculate(data);

        // Calculate 2 * WMA(n/2) - WMA(n)
        let mut diff = Vec::with_capacity(n);
        for i in 0..n {
            if wma_half_values[i].is_nan() || wma_full_values[i].is_nan() {
                diff.push(f64::NAN);
            } else {
                diff.push(2.0 * wma_half_values[i] - wma_full_values[i]);
            }
        }

        // Apply final WMA with sqrt(period)
        wma_sqrt.calculate(&diff)
    }
}

impl Default for HMA {
    fn default() -> Self {
        Self::from_config(HMAConfig::default())
    }
}

impl TechnicalIndicator for HMA {
    fn name(&self) -> &str {
        "HMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.period + (self.period as f64).sqrt() as usize;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + (self.period as f64).sqrt() as usize
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hma() {
        let hma = HMA::new(9);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = hma.calculate(&data);

        // Should have NaN initially
        assert!(result[0].is_nan());
        // After enough periods, should have values
        assert!(!result[29].is_nan());
        // HMA should be responsive to uptrend
        assert!(result[29] > 120.0);
    }
}
