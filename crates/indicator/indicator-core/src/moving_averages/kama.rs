//! Kaufman Adaptive Moving Average (KAMA) implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::KAMAConfig;

/// Kaufman Adaptive Moving Average.
///
/// KAMA adjusts its smoothing based on market efficiency.
/// In trending markets, it follows price closely.
/// In ranging markets, it filters out noise.
#[derive(Debug, Clone)]
pub struct KAMA {
    period: usize,
    fast_period: usize,
    slow_period: usize,
}

impl KAMA {
    pub fn new(period: usize, fast: usize, slow: usize) -> Self {
        Self {
            period,
            fast_period: fast,
            slow_period: slow,
        }
    }

    pub fn from_config(config: KAMAConfig) -> Self {
        Self {
            period: config.period,
            fast_period: config.fast_period,
            slow_period: config.slow_period,
        }
    }

    /// Calculate KAMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n <= self.period {
            return result;
        }

        let fast_sc = 2.0 / (self.fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (self.slow_period as f64 + 1.0);

        // Initialize KAMA with first valid price
        result[self.period] = data[self.period];

        for i in (self.period + 1)..n {
            // Calculate Efficiency Ratio
            let change = (data[i] - data[i - self.period]).abs();
            let mut volatility = 0.0;
            for j in (i - self.period + 1)..=i {
                volatility += (data[j] - data[j - 1]).abs();
            }

            let er = if volatility != 0.0 { change / volatility } else { 0.0 };

            // Calculate smoothing constant
            let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);

            // Calculate KAMA
            result[i] = result[i - 1] + sc * (data[i] - result[i - 1]);
        }

        result
    }
}

impl Default for KAMA {
    fn default() -> Self {
        Self::from_config(KAMAConfig::default())
    }
}

impl TechnicalIndicator for KAMA {
    fn name(&self) -> &str {
        "KAMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() <= self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kama() {
        let kama = KAMA::default();
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let result = kama.calculate(&data);

        // Should have NaN initially
        assert!(result[0].is_nan());
        // After period, should have values
        assert!(!result[49].is_nan());
        // In a strong trend, KAMA should track price closely
        assert!(result[49] > 130.0);
    }
}
