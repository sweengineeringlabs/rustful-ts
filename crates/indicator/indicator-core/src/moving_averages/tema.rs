//! Triple Exponential Moving Average (TEMA) implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::TEMAConfig;
use crate::EMA;

/// Triple Exponential Moving Average.
///
/// TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
/// Further reduces lag compared to DEMA.
#[derive(Debug, Clone)]
pub struct TEMA {
    period: usize,
}

impl TEMA {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: TEMAConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate TEMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let ema = EMA::new(self.period);

        let ema1 = ema.calculate(data);
        let ema2 = ema.calculate(&ema1);
        let ema3 = ema.calculate(&ema2);

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            if ema1[i].is_nan() || ema2[i].is_nan() || ema3[i].is_nan() {
                result.push(f64::NAN);
            } else {
                result.push(3.0 * ema1[i] - 3.0 * ema2[i] + ema3[i]);
            }
        }

        result
    }
}

impl Default for TEMA {
    fn default() -> Self {
        Self::from_config(TEMAConfig::default())
    }
}

impl TechnicalIndicator for TEMA {
    fn name(&self) -> &str {
        "TEMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period * 3 {
            return Err(IndicatorError::InsufficientData {
                required: self.period * 3,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period * 3
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tema() {
        let tema = TEMA::new(5);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = tema.calculate(&data);

        // Should have NaN initially
        assert!(result[0].is_nan());
        // After enough periods, should have values
        assert!(!result[29].is_nan());
        // TEMA should track trends closely
        assert!(result[29] > 120.0);
    }
}
