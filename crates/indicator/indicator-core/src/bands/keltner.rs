//! Keltner Channels implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::KeltnerConfig;
use crate::{EMA, ATR};

/// Keltner Channels.
///
/// Volatility-based envelope indicator.
/// - Middle Band: EMA of close
/// - Upper Band: Middle + (multiplier × ATR)
/// - Lower Band: Middle - (multiplier × ATR)
#[derive(Debug, Clone)]
pub struct KeltnerChannels {
    ema_period: usize,
    atr_period: usize,
    multiplier: f64,
}

impl KeltnerChannels {
    pub fn new(ema_period: usize, atr_period: usize, multiplier: f64) -> Self {
        Self { ema_period, atr_period, multiplier }
    }

    pub fn from_config(config: KeltnerConfig) -> Self {
        Self {
            ema_period: config.ema_period,
            atr_period: config.atr_period,
            multiplier: config.multiplier,
        }
    }

    /// Calculate Keltner Channels (middle, upper, lower).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        // Calculate EMA of close (middle band)
        let ema = EMA::new(self.ema_period);
        let middle = ema.calculate(close);

        // Calculate ATR
        let atr = ATR::new(self.atr_period);
        let atr_values = atr.calculate(high, low, close);

        // Calculate upper and lower bands
        let mut upper = Vec::with_capacity(n);
        let mut lower = Vec::with_capacity(n);

        for i in 0..n {
            if middle[i].is_nan() || atr_values[i].is_nan() {
                upper.push(f64::NAN);
                lower.push(f64::NAN);
            } else {
                upper.push(middle[i] + self.multiplier * atr_values[i]);
                lower.push(middle[i] - self.multiplier * atr_values[i]);
            }
        }

        (middle, upper, lower)
    }
}

impl Default for KeltnerChannels {
    fn default() -> Self {
        Self::from_config(KeltnerConfig::default())
    }
}

impl TechnicalIndicator for KeltnerChannels {
    fn name(&self) -> &str {
        "KeltnerChannels"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.ema_period.max(self.atr_period);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.ema_period.max(self.atr_period)
    }

    fn output_features(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keltner_channels() {
        let kc = KeltnerChannels::new(10, 10, 2.0);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (middle, upper, lower) = kc.calculate(&high, &low, &close);

        // Check bands exist after warmup
        for i in 10..n {
            if !middle[i].is_nan() {
                assert!(upper[i] > middle[i]);
                assert!(lower[i] < middle[i]);
            }
        }
    }
}
