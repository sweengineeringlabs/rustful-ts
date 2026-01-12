//! Positive Volume Index (PVI) implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Positive Volume Index.
///
/// PVI tracks price changes on days when volume increases.
/// The theory is that the "uninformed crowd" trades on high-volume days.
///
/// - If Volume > Previous Volume: PVI = Previous PVI + (Price Change %)
/// - If Volume <= Previous Volume: PVI = Previous PVI
///
/// Typically compared against its EMA(255) for signals.
#[derive(Debug, Clone)]
pub struct PVI {
    signal_period: usize,
}

impl PVI {
    pub fn new() -> Self {
        Self { signal_period: 255 }
    }

    pub fn with_signal(signal_period: usize) -> Self {
        Self { signal_period }
    }

    /// Calculate PVI values.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);
        result.push(1000.0); // Standard starting value

        for i in 1..n {
            if volume[i] > volume[i - 1] && close[i - 1] > 0.0 {
                let pct_change = (close[i] - close[i - 1]) / close[i - 1] * 100.0;
                result.push(result[i - 1] + pct_change);
            } else {
                result.push(result[i - 1]);
            }
        }

        result
    }

    /// Calculate PVI with signal line (EMA).
    pub fn calculate_with_signal(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let pvi = self.calculate(close, volume);
        let signal = self.ema(&pvi, self.signal_period);
        (pvi, signal)
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

impl Default for PVI {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for PVI {
    fn name(&self) -> &str {
        "PVI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let (pvi, signal) = self.calculate_with_signal(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(pvi, signal))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for PVI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (pvi, signal) = self.calculate_with_signal(&data.close, &data.volume);

        if let (Some(&pvi_val), Some(&sig_val)) = (pvi.last(), signal.last()) {
            if !sig_val.is_nan() {
                if pvi_val > sig_val {
                    return Ok(IndicatorSignal::Bullish);
                } else if pvi_val < sig_val {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (pvi, signal) = self.calculate_with_signal(&data.close, &data.volume);

        Ok(pvi
            .iter()
            .zip(signal.iter())
            .map(|(&p, &s)| {
                if s.is_nan() {
                    IndicatorSignal::Neutral
                } else if p > s {
                    IndicatorSignal::Bullish
                } else if p < s {
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
    fn test_pvi() {
        let pvi = PVI::new();
        let close = vec![100.0, 102.0, 101.0, 103.0, 102.0];
        // Volume increasing on days 2
        let volume = vec![1000.0, 800.0, 1200.0, 700.0, 600.0];

        let result = pvi.calculate(&close, &volume);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 1000.0).abs() < 1e-10);
        // Day 1: volume decreased, PVI unchanged
        assert!((result[1] - result[0]).abs() < 1e-10);
        // Day 2: volume increased, price -1%
        assert!(result[2] < result[1]);
    }

    #[test]
    fn test_pvi_unchanged_on_volume_decrease() {
        let pvi = PVI::new();
        let close = vec![100.0, 105.0];
        let volume = vec![1500.0, 1000.0]; // Volume decreased

        let result = pvi.calculate(&close, &volume);

        // PVI should be unchanged despite price increase
        assert!((result[1] - result[0]).abs() < 1e-10);
    }
}
