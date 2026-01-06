//! Double Exponential Moving Average (DEMA) implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::DEMAConfig;
use crate::EMA;

/// Double Exponential Moving Average.
///
/// DEMA = 2 * EMA - EMA(EMA)
/// Reduces lag compared to a standard EMA.
#[derive(Debug, Clone)]
pub struct DEMA {
    period: usize,
}

impl DEMA {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: DEMAConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate DEMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let ema = EMA::new(self.period);

        let ema1 = ema.calculate(data);
        let ema2 = ema.calculate(&ema1);

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            if ema1[i].is_nan() || ema2[i].is_nan() {
                result.push(f64::NAN);
            } else {
                result.push(2.0 * ema1[i] - ema2[i]);
            }
        }

        result
    }
}

impl Default for DEMA {
    fn default() -> Self {
        Self::from_config(DEMAConfig::default())
    }
}

impl TechnicalIndicator for DEMA {
    fn name(&self) -> &str {
        "DEMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period * 2 {
            return Err(IndicatorError::InsufficientData {
                required: self.period * 2,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period * 2
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dema() {
        let dema = DEMA::new(5);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = dema.calculate(&data);

        // DEMA should track the uptrend closely
        assert!(result[0].is_nan());
        // After enough periods, should have values
        assert!(!result[29].is_nan());
        // DEMA should be close to recent prices in uptrend
        assert!(result[29] > 120.0);
    }
}
