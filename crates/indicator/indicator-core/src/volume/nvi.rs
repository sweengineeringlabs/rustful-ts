//! Negative Volume Index (NVI) implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Negative Volume Index.
///
/// NVI tracks price changes on days when volume decreases.
/// The theory is that "smart money" trades on low-volume days.
///
/// - If Volume < Previous Volume: NVI = Previous NVI + (Price Change %)
/// - If Volume >= Previous Volume: NVI = Previous NVI
///
/// Typically compared against its EMA(255) for signals.
#[derive(Debug, Clone)]
pub struct NVI {
    signal_period: usize,
}

impl NVI {
    pub fn new() -> Self {
        Self { signal_period: 255 }
    }

    pub fn with_signal(signal_period: usize) -> Self {
        Self { signal_period }
    }

    /// Calculate NVI values.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);
        result.push(1000.0); // Standard starting value

        for i in 1..n {
            if volume[i] < volume[i - 1] && close[i - 1] > 0.0 {
                let pct_change = (close[i] - close[i - 1]) / close[i - 1] * 100.0;
                result.push(result[i - 1] + pct_change);
            } else {
                result.push(result[i - 1]);
            }
        }

        result
    }

    /// Calculate NVI with signal line (EMA).
    pub fn calculate_with_signal(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let nvi = self.calculate(close, volume);
        let signal = self.ema(&nvi, self.signal_period);
        (nvi, signal)
    }

    /// Calculate EMA.
    fn ema(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let alpha = 2.0 / (period as f64 + 1.0);

        // Calculate initial SMA
        let sum: f64 = values[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        // Apply EMA
        for i in period..n {
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }
}

impl Default for NVI {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for NVI {
    fn name(&self) -> &str {
        "NVI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let (nvi, signal) = self.calculate_with_signal(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(nvi, signal))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for NVI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (nvi, signal) = self.calculate_with_signal(&data.close, &data.volume);

        if let (Some(&nvi_val), Some(&sig_val)) = (nvi.last(), signal.last()) {
            if !sig_val.is_nan() {
                if nvi_val > sig_val {
                    return Ok(IndicatorSignal::Bullish);
                } else if nvi_val < sig_val {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (nvi, signal) = self.calculate_with_signal(&data.close, &data.volume);

        Ok(nvi
            .iter()
            .zip(signal.iter())
            .map(|(&n, &s)| {
                if s.is_nan() {
                    IndicatorSignal::Neutral
                } else if n > s {
                    IndicatorSignal::Bullish
                } else if n < s {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvi() {
        let nvi = NVI::new();
        let close = vec![100.0, 102.0, 101.0, 103.0, 102.0];
        // Volume decreasing on days 1, 3, 4
        let volume = vec![1000.0, 800.0, 900.0, 700.0, 600.0];

        let result = nvi.calculate(&close, &volume);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 1000.0).abs() < 1e-10);
        // Day 1: volume decreased, price +2%
        assert!(result[1] > result[0]);
        // Day 2: volume increased, NVI unchanged
        assert!((result[2] - result[1]).abs() < 1e-10);
    }

    #[test]
    fn test_nvi_unchanged_on_volume_increase() {
        let nvi = NVI::new();
        let close = vec![100.0, 105.0];
        let volume = vec![1000.0, 1500.0]; // Volume increased

        let result = nvi.calculate(&close, &volume);

        // NVI should be unchanged despite price increase
        assert!((result[1] - result[0]).abs() < 1e-10);
    }
}
