//! Relative Strength Index implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_api::RSIConfig;

/// Relative Strength Index (RSI).
///
/// Momentum oscillator measuring speed and magnitude of price changes.
/// Values range from 0 to 100.
#[derive(Debug, Clone)]
pub struct RSI {
    period: usize,
    overbought: f64,
    oversold: f64,
}

impl RSI {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: 70.0,
            oversold: 30.0,
        }
    }

    pub fn from_config(config: RSIConfig) -> Self {
        Self {
            period: config.period,
            overbought: config.overbought,
            oversold: config.oversold,
        }
    }

    /// Calculate RSI values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        // Calculate price changes
        let mut gains = Vec::with_capacity(n - 1);
        let mut losses = Vec::with_capacity(n - 1);

        for i in 1..n {
            let change = data[i] - data[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let mut result = vec![f64::NAN; self.period];

        // Initial average gain/loss (SMA)
        let mut avg_gain: f64 = gains[0..self.period].iter().sum::<f64>() / self.period as f64;
        let mut avg_loss: f64 = losses[0..self.period].iter().sum::<f64>() / self.period as f64;

        // First RSI value
        let rsi = if avg_loss == 0.0 {
            100.0
        } else {
            100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        };
        result.push(rsi);

        // Smoothed RSI calculation (Wilder's smoothing)
        for i in self.period..(n - 1) {
            avg_gain = (avg_gain * (self.period - 1) as f64 + gains[i]) / self.period as f64;
            avg_loss = (avg_loss * (self.period - 1) as f64 + losses[i]) / self.period as f64;

            let rsi = if avg_loss == 0.0 {
                100.0
            } else {
                100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            };
            result.push(rsi);
        }

        result
    }
}

impl TechnicalIndicator for RSI {
    fn name(&self) -> &str {
        "RSI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
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
}

impl SignalIndicator for RSI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if last >= self.overbought {
            Ok(IndicatorSignal::Bearish)
        } else if last <= self.oversold {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let signals = values.iter().map(|&rsi| {
            if rsi.is_nan() {
                IndicatorSignal::Neutral
            } else if rsi >= self.overbought {
                IndicatorSignal::Bearish
            } else if rsi <= self.oversold {
                IndicatorSignal::Bullish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect();
        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi_basic() {
        let rsi = RSI::new(14);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result = rsi.calculate(&data);

        // Check that RSI values are in range [0, 100]
        for (i, &val) in result.iter().enumerate() {
            if i >= 14 && !val.is_nan() {
                assert!(val >= 0.0 && val <= 100.0);
            }
        }
    }

    #[test]
    fn test_rsi_uptrend() {
        let rsi = RSI::new(5);
        // Consistent uptrend should give high RSI
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let result = rsi.calculate(&data);

        let last = result.last().unwrap();
        assert!(*last > 70.0); // Should be overbought in strong uptrend
    }
}
