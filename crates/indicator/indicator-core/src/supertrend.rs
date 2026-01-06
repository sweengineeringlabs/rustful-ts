//! SuperTrend implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_api::SuperTrendConfig;
use crate::ATR;

/// SuperTrend output.
#[derive(Debug, Clone)]
pub struct SuperTrendOutput {
    pub supertrend: Vec<f64>,
    pub direction: Vec<f64>, // 1.0 = bullish, -1.0 = bearish
}

/// SuperTrend indicator.
///
/// Trend-following indicator based on ATR.
/// - Upper Band = HL2 + (multiplier × ATR)
/// - Lower Band = HL2 - (multiplier × ATR)
/// - SuperTrend = Lower Band when bullish, Upper Band when bearish
#[derive(Debug, Clone)]
pub struct SuperTrend {
    period: usize,
    multiplier: f64,
}

impl SuperTrend {
    pub fn new(period: usize, multiplier: f64) -> Self {
        Self { period, multiplier }
    }

    pub fn from_config(config: SuperTrendConfig) -> Self {
        Self {
            period: config.period,
            multiplier: config.multiplier,
        }
    }

    /// Calculate SuperTrend values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> SuperTrendOutput {
        let n = high.len();
        if n < self.period {
            return SuperTrendOutput {
                supertrend: vec![f64::NAN; n],
                direction: vec![f64::NAN; n],
            };
        }

        // Calculate ATR
        let atr = ATR::new(self.period);
        let atr_values = atr.calculate(high, low, close);

        let mut supertrend = vec![f64::NAN; n];
        let mut direction = vec![f64::NAN; n];
        let mut upper_band = vec![0.0; n];
        let mut lower_band = vec![0.0; n];

        // Calculate basic bands
        for i in (self.period - 1)..n {
            let hl2 = (high[i] + low[i]) / 2.0;
            upper_band[i] = hl2 + self.multiplier * atr_values[i];
            lower_band[i] = hl2 - self.multiplier * atr_values[i];
        }

        // Initialize at first valid position
        supertrend[self.period - 1] = upper_band[self.period - 1];
        direction[self.period - 1] = -1.0; // Start bearish

        for i in self.period..n {
            // Adjust lower band
            if lower_band[i] > lower_band[i - 1] || close[i - 1] < lower_band[i - 1] {
                // Keep current
            } else {
                lower_band[i] = lower_band[i - 1];
            }

            // Adjust upper band
            if upper_band[i] < upper_band[i - 1] || close[i - 1] > upper_band[i - 1] {
                // Keep current
            } else {
                upper_band[i] = upper_band[i - 1];
            }

            // Determine trend direction
            if direction[i - 1] < 0.0 {
                // Previous was bearish
                if close[i] > upper_band[i - 1] {
                    direction[i] = 1.0; // Switch to bullish
                    supertrend[i] = lower_band[i];
                } else {
                    direction[i] = -1.0;
                    supertrend[i] = upper_band[i];
                }
            } else {
                // Previous was bullish
                if close[i] < lower_band[i - 1] {
                    direction[i] = -1.0; // Switch to bearish
                    supertrend[i] = upper_band[i];
                } else {
                    direction[i] = 1.0;
                    supertrend[i] = lower_band[i];
                }
            }
        }

        SuperTrendOutput { supertrend, direction }
    }
}

impl Default for SuperTrend {
    fn default() -> Self {
        Self::from_config(SuperTrendConfig::default())
    }
}

impl TechnicalIndicator for SuperTrend {
    fn name(&self) -> &str {
        "SuperTrend"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(result.supertrend, result.direction))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for SuperTrend {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let last_dir = result.direction.last().copied().unwrap_or(f64::NAN);

        if last_dir.is_nan() {
            Ok(IndicatorSignal::Neutral)
        } else if last_dir > 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Bearish)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let signals = result.direction.iter().map(|&d| {
            if d.is_nan() {
                IndicatorSignal::Neutral
            } else if d > 0.0 {
                IndicatorSignal::Bullish
            } else {
                IndicatorSignal::Bearish
            }
        }).collect();
        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supertrend() {
        let st = SuperTrend::new(10, 3.0);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.2).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.2).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.2).sin() * 5.0).collect();

        let result = st.calculate(&high, &low, &close);

        assert_eq!(result.supertrend.len(), n);
        assert_eq!(result.direction.len(), n);

        // Check direction values are valid
        for i in 9..n {
            let d = result.direction[i];
            assert!(d == 1.0 || d == -1.0);
        }
    }
}
