//! Commodity Channel Index (CCI) implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};
use indicator_api::CCIConfig;
use crate::SMA;

/// Commodity Channel Index.
///
/// CCI = (Typical Price - SMA(TP)) / (0.015 * Mean Deviation)
/// - Above +100: Overbought/Strong uptrend
/// - Below -100: Oversold/Strong downtrend
#[derive(Debug, Clone)]
pub struct CCI {
    period: usize,
    overbought: f64,
    oversold: f64,
}

impl CCI {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: 100.0,
            oversold: -100.0,
        }
    }

    pub fn from_config(config: CCIConfig) -> Self {
        Self {
            period: config.period,
            overbought: config.overbought,
            oversold: config.oversold,
        }
    }

    /// Calculate CCI values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < self.period {
            return result;
        }

        // Calculate Typical Price
        let tp: Vec<f64> = (0..n)
            .map(|i| (high[i] + low[i] + close[i]) / 3.0)
            .collect();

        // Calculate SMA of Typical Price
        let sma = SMA::new(self.period);
        let tp_sma = sma.calculate(&tp);

        // Calculate CCI
        for i in (self.period - 1)..n {
            let mean = tp_sma[i];

            // Calculate Mean Deviation
            let start = i + 1 - self.period;
            let mean_dev: f64 = tp[start..=i]
                .iter()
                .map(|&x| (x - mean).abs())
                .sum::<f64>() / self.period as f64;

            result[i] = if mean_dev > 0.0 {
                (tp[i] - mean) / (0.015 * mean_dev)
            } else {
                0.0
            };
        }

        result
    }
}

impl Default for CCI {
    fn default() -> Self {
        Self::from_config(CCIConfig::default())
    }
}

impl TechnicalIndicator for CCI {
    fn name(&self) -> &str {
        "CCI"
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

impl SignalIndicator for CCI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last > self.overbought {
                    return Ok(IndicatorSignal::Bullish);
                } else if last < self.oversold {
                    return Ok(IndicatorSignal::Bearish);
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
                IndicatorSignal::Bullish
            } else if v < self.oversold {
                IndicatorSignal::Bearish
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
    fn test_cci() {
        let cci = CCI::new(20);
        let n = 50;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64 * 0.5).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64 * 0.5).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect();

        let result = cci.calculate(&high, &low, &close);

        // Should have NaN initially
        assert!(result[0].is_nan());
        // After period, should have values
        assert!(!result[49].is_nan());
    }
}
