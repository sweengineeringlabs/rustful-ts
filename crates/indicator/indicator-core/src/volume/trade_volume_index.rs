//! Trade Volume Index (TVI) implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Trade Volume Index (TVI).
///
/// A tick-based accumulation indicator that measures the flow of volume
/// based on price changes. Volume is accumulated on upticks and distributed
/// on downticks.
///
/// - If close > previous close (uptick): TVI = previous TVI + volume
/// - If close < previous close (downtick): TVI = previous TVI - volume
/// - If close = previous close: Uses minimum tick value threshold
///
/// A rising TVI suggests accumulation (buying pressure), while a falling
/// TVI suggests distribution (selling pressure).
#[derive(Debug, Clone)]
pub struct TradeVolumeIndex {
    /// Minimum price change to consider as uptick/downtick
    min_tick: f64,
    /// Signal line period for smoothing
    signal_period: usize,
}

impl TradeVolumeIndex {
    /// Create a new Trade Volume Index indicator.
    ///
    /// # Arguments
    /// * `min_tick` - Minimum price change to consider as uptick/downtick (default 0.5)
    /// * `signal_period` - Period for signal line EMA (default 14)
    pub fn new(min_tick: f64, signal_period: usize) -> Self {
        Self {
            min_tick,
            signal_period,
        }
    }

    /// Calculate EMA for signal line.
    fn ema(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let alpha = 2.0 / (period as f64 + 1.0);

        // Initial SMA
        let mut sum = 0.0;
        for i in 0..period {
            sum += values[i];
        }
        result[period - 1] = sum / period as f64;

        // EMA calculation
        for i in period..n {
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate TVI values.
    /// Returns (TVI values, Signal line)
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();

        if n == 0 {
            return (vec![], vec![]);
        }

        let mut tvi = Vec::with_capacity(n);
        let mut direction = 1.0; // Track last significant direction (1 = up, -1 = down)

        tvi.push(volume[0]); // First TVI equals first volume

        for i in 1..n {
            let price_change = close[i] - close[i - 1];

            // Determine direction based on price change and min_tick threshold
            if price_change > self.min_tick {
                direction = 1.0;
            } else if price_change < -self.min_tick {
                direction = -1.0;
            }
            // If within min_tick range, keep previous direction

            tvi.push(tvi[i - 1] + direction * volume[i]);
        }

        // Calculate signal line
        let signal = self.ema(&tvi, self.signal_period);

        (tvi, signal)
    }
}

impl Default for TradeVolumeIndex {
    fn default() -> Self {
        Self {
            min_tick: 0.5,
            signal_period: 14,
        }
    }
}

impl TechnicalIndicator for TradeVolumeIndex {
    fn name(&self) -> &str {
        "Trade Volume Index"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let (tvi, signal) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(tvi, signal))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for TradeVolumeIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (tvi, signal) = self.calculate(&data.close, &data.volume);

        if tvi.len() < 2 || signal.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = tvi.len();
        let curr_tvi = tvi[n - 1];
        let prev_tvi = tvi[n - 2];
        let curr_sig = signal[n - 1];
        let prev_sig = signal[n - 2];

        if curr_sig.is_nan() || prev_sig.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: TVI crosses above signal line
        if prev_tvi <= prev_sig && curr_tvi > curr_sig {
            return Ok(IndicatorSignal::Bullish);
        }
        // Bearish: TVI crosses below signal line
        if prev_tvi >= prev_sig && curr_tvi < curr_sig {
            return Ok(IndicatorSignal::Bearish);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (tvi, signal) = self.calculate(&data.close, &data.volume);

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..tvi.len() {
            if signal[i].is_nan() || signal[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if tvi[i - 1] <= signal[i - 1] && tvi[i] > signal[i] {
                signals.push(IndicatorSignal::Bullish);
            } else if tvi[i - 1] >= signal[i - 1] && tvi[i] < signal[i] {
                signals.push(IndicatorSignal::Bearish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tvi_uptrend() {
        let tvi = TradeVolumeIndex::new(0.1, 5);
        // Uptrend: each close higher than previous
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0];
        let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 1400.0, 1600.0, 1300.0, 1700.0, 1100.0, 1900.0];

        let (tvi_values, signal_values) = tvi.calculate(&close, &volume);

        assert_eq!(tvi_values.len(), 10);
        assert_eq!(signal_values.len(), 10);

        // TVI should be increasing in uptrend (all volume added)
        assert!(tvi_values[9] > tvi_values[0]);

        // Expected: 1000 + 1500 + 1200 + 1800 + 1400 + 1600 + 1300 + 1700 + 1100 + 1900 = 14500
        assert!((tvi_values[9] - 14500.0).abs() < 1e-10);
    }

    #[test]
    fn test_tvi_downtrend() {
        let tvi = TradeVolumeIndex::new(0.1, 5);
        // Downtrend: each close lower than previous
        let close = vec![109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0];
        let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 1400.0, 1600.0, 1300.0, 1700.0, 1100.0, 1900.0];

        let (tvi_values, _) = tvi.calculate(&close, &volume);

        // TVI should decrease (volume subtracted after first bar)
        // Expected: 1000 - 1500 - 1200 - 1800 - 1400 - 1600 - 1300 - 1700 - 1100 - 1900 = -12500
        assert!((tvi_values[9] - (-12500.0)).abs() < 1e-10);
    }

    #[test]
    fn test_tvi_mixed() {
        let tvi = TradeVolumeIndex::new(0.1, 3);
        let close = vec![100.0, 101.0, 100.0, 99.0, 100.0];
        let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 1400.0];

        let (tvi_values, _) = tvi.calculate(&close, &volume);

        // 1000 + 1500 (up) - 1200 (down) - 1800 (down) + 1400 (up) = 900
        assert!((tvi_values[4] - 900.0).abs() < 1e-10);
    }

    #[test]
    fn test_tvi_min_tick_threshold() {
        let tvi = TradeVolumeIndex::new(1.0, 3);
        // Price changes less than min_tick keep previous direction
        let close = vec![100.0, 100.5, 100.3, 100.8, 99.0];
        let volume = vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0];

        let (tvi_values, _) = tvi.calculate(&close, &volume);

        // First bar: 1000
        // Second: +0.5 < 1.0, direction stays positive (from init): 2000
        // Third: -0.2 < 1.0, direction stays positive: 3000
        // Fourth: +0.5 < 1.0, direction stays positive: 4000
        // Fifth: -1.8 > 1.0, direction becomes negative: 3000
        assert!((tvi_values[4] - 3000.0).abs() < 1e-10);
    }

    #[test]
    fn test_tvi_empty() {
        let tvi = TradeVolumeIndex::default();
        let (values, signal) = tvi.calculate(&[], &[]);
        assert!(values.is_empty());
        assert!(signal.is_empty());
    }

    #[test]
    fn test_tvi_signal() {
        let tvi = TradeVolumeIndex::new(0.1, 3);
        // Create data that will generate a crossover
        let close = vec![100.0, 99.0, 98.0, 97.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0];
        let volume = vec![1000.0; 10];

        let signals = tvi.signals(&OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 1.0).collect(),
            low: close.iter().map(|x| x - 1.0).collect(),
            close: close.clone(),
            volume: volume.clone(),
        }).unwrap();

        assert_eq!(signals.len(), 10);
        // There should be a bullish crossover when TVI starts rising
    }

    #[test]
    fn test_tvi_technical_indicator() {
        let tvi = TradeVolumeIndex::default();
        assert_eq!(tvi.name(), "Trade Volume Index");
        assert_eq!(tvi.min_periods(), 1);
        assert_eq!(tvi.output_features(), 2);
    }
}
