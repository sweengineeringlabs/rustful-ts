//! Average Directional Index (ADX) implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};
use indicator_api::ADXConfig;
use crate::ATR;

/// Average Directional Index output.
#[derive(Debug, Clone)]
pub struct ADXOutput {
    pub adx: Vec<f64>,
    pub plus_di: Vec<f64>,
    pub minus_di: Vec<f64>,
}

/// Average Directional Index.
///
/// Measures trend strength regardless of direction.
/// - ADX > 25: Strong trend
/// - ADX < 20: Weak trend/ranging
/// - +DI > -DI: Bullish trend
/// - -DI > +DI: Bearish trend
#[derive(Debug, Clone)]
pub struct ADX {
    period: usize,
    strong_trend: f64,
}

impl ADX {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            strong_trend: 25.0,
        }
    }

    pub fn from_config(config: ADXConfig) -> Self {
        Self {
            period: config.period,
            strong_trend: config.strong_trend,
        }
    }

    /// Calculate ADX values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> ADXOutput {
        let n = close.len();
        let mut adx = vec![f64::NAN; n];
        let mut plus_di = vec![f64::NAN; n];
        let mut minus_di = vec![f64::NAN; n];

        if n < self.period * 2 {
            return ADXOutput { adx, plus_di, minus_di };
        }

        // First calculate ATR
        let atr_indicator = ATR::new(self.period);
        let atr_values = atr_indicator.calculate(high, low, close);

        // Calculate +DM and -DM
        let mut plus_dm = vec![0.0; n];
        let mut minus_dm = vec![0.0; n];

        for i in 1..n {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];

            plus_dm[i] = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
            minus_dm[i] = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };
        }

        // Smooth +DM and -DM using Wilder's method
        let mut smooth_plus_dm = 0.0;
        let mut smooth_minus_dm = 0.0;

        // Initialize
        for i in 1..=self.period {
            smooth_plus_dm += plus_dm[i];
            smooth_minus_dm += minus_dm[i];
        }

        // Calculate DX values
        let mut dx_values = vec![0.0; n];

        for i in self.period..n {
            if i > self.period {
                smooth_plus_dm = smooth_plus_dm - smooth_plus_dm / self.period as f64 + plus_dm[i];
                smooth_minus_dm = smooth_minus_dm - smooth_minus_dm / self.period as f64 + minus_dm[i];
            }

            let atr_val = atr_values[i];
            if atr_val > 0.0 && !atr_val.is_nan() {
                plus_di[i] = 100.0 * smooth_plus_dm / atr_val / self.period as f64;
                minus_di[i] = 100.0 * smooth_minus_dm / atr_val / self.period as f64;

                let di_sum = plus_di[i] + minus_di[i];
                dx_values[i] = if di_sum > 0.0 {
                    100.0 * (plus_di[i] - minus_di[i]).abs() / di_sum
                } else {
                    0.0
                };
            }
        }

        // Calculate ADX as smoothed average of DX
        let mut adx_val = 0.0;
        for i in self.period..(self.period * 2) {
            adx_val += dx_values[i];
        }
        adx_val /= self.period as f64;
        adx[self.period * 2 - 1] = adx_val;

        for i in (self.period * 2)..n {
            adx_val = (adx_val * (self.period - 1) as f64 + dx_values[i]) / self.period as f64;
            adx[i] = adx_val;
        }

        ADXOutput { adx, plus_di, minus_di }
    }

    /// Calculate just the ADX line.
    pub fn calculate_adx(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        self.calculate(high, low, close).adx
    }
}

impl Default for ADX {
    fn default() -> Self {
        Self::from_config(ADXConfig::default())
    }
}

impl TechnicalIndicator for ADX {
    fn name(&self) -> &str {
        "ADX"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period * 2 {
            return Err(IndicatorError::InsufficientData {
                required: self.period * 2,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(result.adx, result.plus_di, result.minus_di))
    }

    fn min_periods(&self) -> usize {
        self.period * 2
    }

    fn output_features(&self) -> usize {
        3  // ADX, +DI, -DI
    }
}

impl SignalIndicator for ADX {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let n = result.adx.len();

        if n > 0 {
            let adx_val = result.adx[n - 1];
            let plus = result.plus_di[n - 1];
            let minus = result.minus_di[n - 1];

            if !adx_val.is_nan() && adx_val > self.strong_trend {
                if plus > minus {
                    return Ok(IndicatorSignal::Bullish);
                } else {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let n = result.adx.len();
        let mut signals = vec![IndicatorSignal::Neutral; n];

        for i in 0..n {
            let adx_val = result.adx[i];
            let plus = result.plus_di[i];
            let minus = result.minus_di[i];

            if !adx_val.is_nan() && adx_val > self.strong_trend {
                if !plus.is_nan() && !minus.is_nan() {
                    if plus > minus {
                        signals[i] = IndicatorSignal::Bullish;
                    } else {
                        signals[i] = IndicatorSignal::Bearish;
                    }
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
    fn test_adx() {
        let adx = ADX::new(14);
        let n = 50;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let result = adx.calculate(&high, &low, &close);

        // Should have NaN initially
        assert!(result.adx[0].is_nan());
        // After 2*period, should have values
        assert!(!result.adx[49].is_nan());
        // ADX should be between 0 and 100
        assert!(result.adx[49] >= 0.0 && result.adx[49] <= 100.0);
    }
}
