//! Elder Ray implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::ElderRayConfig;
use crate::EMA;

/// Elder Ray output.
#[derive(Debug, Clone)]
pub struct ElderRayOutput {
    pub bull_power: Vec<f64>,
    pub bear_power: Vec<f64>,
}

/// Elder Ray Index (Bull and Bear Power).
///
/// Measures buying and selling pressure.
/// - Bull Power = High - EMA(Close)
/// - Bear Power = Low - EMA(Close)
#[derive(Debug, Clone)]
pub struct ElderRay {
    period: usize,
}

impl ElderRay {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: ElderRayConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate Elder Ray values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> ElderRayOutput {
        let n = close.len();

        // Calculate EMA of close
        let ema = EMA::new(self.period);
        let ema_values = ema.calculate(close);

        let mut bull_power = Vec::with_capacity(n);
        let mut bear_power = Vec::with_capacity(n);

        for i in 0..n {
            if ema_values[i].is_nan() {
                bull_power.push(f64::NAN);
                bear_power.push(f64::NAN);
            } else {
                bull_power.push(high[i] - ema_values[i]);
                bear_power.push(low[i] - ema_values[i]);
            }
        }

        ElderRayOutput { bull_power, bear_power }
    }
}

impl Default for ElderRay {
    fn default() -> Self {
        Self::from_config(ElderRayConfig::default())
    }
}

impl TechnicalIndicator for ElderRay {
    fn name(&self) -> &str {
        "ElderRay"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(result.bull_power, result.bear_power))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elder_ray() {
        let er = ElderRay::new(13);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64 * 0.5).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64 * 0.5).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect();

        let result = er.calculate(&high, &low, &close);

        // Bull power should be positive (high > ema in uptrend)
        // Bear power should be negative (low < ema)
        for i in 13..n {
            if !result.bull_power[i].is_nan() {
                assert!(result.bull_power[i] > 0.0);
                assert!(result.bear_power[i] < 0.0);
            }
        }
    }
}
