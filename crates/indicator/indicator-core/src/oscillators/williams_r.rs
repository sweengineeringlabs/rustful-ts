//! Williams %R implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};
use indicator_api::WilliamsRConfig;

/// Williams %R.
///
/// %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
/// Ranges from -100 to 0.
/// - Above -20: Overbought
/// - Below -80: Oversold
#[derive(Debug, Clone)]
pub struct WilliamsR {
    period: usize,
    overbought: f64,
    oversold: f64,
}

impl WilliamsR {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: -20.0,
            oversold: -80.0,
        }
    }

    pub fn from_config(config: WilliamsRConfig) -> Self {
        Self {
            period: config.period,
            overbought: config.overbought,
            oversold: config.oversold,
        }
    }

    /// Calculate Williams %R values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < self.period {
            return result;
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let highest_high = high[start..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lowest_low = low[start..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);

            let range = highest_high - lowest_low;
            result[i] = if range > 0.0 {
                (highest_high - close[i]) / range * -100.0
            } else {
                -50.0
            };
        }

        result
    }
}

impl Default for WilliamsR {
    fn default() -> Self {
        Self::from_config(WilliamsRConfig::default())
    }
}

impl TechnicalIndicator for WilliamsR {
    fn name(&self) -> &str {
        "Williams%R"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for WilliamsR {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last > self.overbought {
                    return Ok(IndicatorSignal::Bearish);
                } else if last < self.oversold {
                    return Ok(IndicatorSignal::Bullish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(values.iter().map(|&v| {
            if v.is_nan() {
                IndicatorSignal::Neutral
            } else if v > self.overbought {
                IndicatorSignal::Bearish
            } else if v < self.oversold {
                IndicatorSignal::Bullish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_williams_r() {
        let wr = WilliamsR::new(14);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let result = wr.calculate(&high, &low, &close);

        // Should have NaN initially
        assert!(result[0].is_nan());
        // After period, values should be between -100 and 0
        for i in 14..n {
            assert!(!result[i].is_nan());
            assert!(result[i] >= -100.0 && result[i] <= 0.0);
        }
    }
}
