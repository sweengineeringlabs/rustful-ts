//! TRIX (Triple Exponential Average Rate of Change) implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};
use indicator_api::TRIXConfig;
use crate::EMA;

/// TRIX - Triple Exponential Average Rate of Change.
///
/// TRIX = 1-period percent change of triple EMA
/// - Positive values indicate bullish momentum
/// - Negative values indicate bearish momentum
/// - Zero crossovers generate signals
#[derive(Debug, Clone)]
pub struct TRIX {
    period: usize,
}

impl TRIX {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: TRIXConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate TRIX values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let ema = EMA::new(self.period);

        // Calculate triple EMA
        let ema1 = ema.calculate(data);
        let ema2 = ema.calculate(&ema1);
        let ema3 = ema.calculate(&ema2);

        let mut result = vec![f64::NAN; n];

        // First value is NaN (need previous value for rate of change)
        for i in 1..n {
            if ema3[i].is_nan() || ema3[i - 1].is_nan() || ema3[i - 1] == 0.0 {
                result[i] = f64::NAN;
            } else {
                result[i] = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 100.0;
            }
        }

        result
    }
}

impl Default for TRIX {
    fn default() -> Self {
        Self::from_config(TRIXConfig::default())
    }
}

impl TechnicalIndicator for TRIX {
    fn name(&self) -> &str {
        "TRIX"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period * 3 + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period * 3 + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period * 3 + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for TRIX {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let n = values.len();

        if n >= 2 {
            let curr = values[n - 1];
            let prev = values[n - 2];

            if !curr.is_nan() && !prev.is_nan() {
                // Zero crossover signals
                if prev < 0.0 && curr >= 0.0 {
                    return Ok(IndicatorSignal::Bullish);
                } else if prev > 0.0 && curr <= 0.0 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral; values.len()];

        for i in 1..values.len() {
            let curr = values[i];
            let prev = values[i - 1];

            if !curr.is_nan() && !prev.is_nan() {
                if prev < 0.0 && curr >= 0.0 {
                    signals[i] = IndicatorSignal::Bullish;
                } else if prev > 0.0 && curr <= 0.0 {
                    signals[i] = IndicatorSignal::Bearish;
                }
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trix() {
        let trix = TRIX::new(5);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let result = trix.calculate(&data);

        // Should have NaN initially
        assert!(result[0].is_nan());
        // After enough periods, should have values
        assert!(!result[49].is_nan());
        // In uptrend, TRIX should be positive
        assert!(result[49] > 0.0);
    }
}
