//! Volume Weighted MACD (VWMACD) implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Volume Weighted MACD (VWMACD).
///
/// A variation of the traditional MACD that uses Volume Weighted Moving Averages
/// (VWMA) instead of Exponential Moving Averages (EMA). This gives more weight
/// to price levels with higher trading activity.
///
/// Calculation:
/// - VWMA(fast) of close prices
/// - VWMA(slow) of close prices
/// - VWMACD Line = VWMA(fast) - VWMA(slow)
/// - Signal Line = EMA of VWMACD Line
/// - Histogram = VWMACD Line - Signal Line
///
/// Interpretation:
/// - Similar to traditional MACD but volume-weighted
/// - VWMACD above zero: Bullish trend
/// - VWMACD below zero: Bearish trend
/// - VWMACD crossing signal: Buy/Sell signals
/// - Histogram divergence: Potential trend changes
#[derive(Debug, Clone)]
pub struct VWMACD {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
}

impl VWMACD {
    /// Create a new VWMACD indicator.
    ///
    /// # Arguments
    /// * `fast_period` - Fast VWMA period (typically 12)
    /// * `slow_period` - Slow VWMA period (typically 26)
    /// * `signal_period` - Signal line EMA period (typically 9)
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            signal_period,
        }
    }

    /// Calculate VWMA.
    fn vwma(&self, close: &[f64], volume: &[f64], period: usize) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        for i in (period - 1)..n {
            let start = i + 1 - period;
            let mut sum_pv = 0.0;
            let mut sum_v = 0.0;

            for j in start..=i {
                sum_pv += close[j] * volume[j];
                sum_v += volume[j];
            }

            result[i] = if sum_v > 0.0 { sum_pv / sum_v } else { close[i] };
        }

        result
    }

    /// Calculate EMA for signal line.
    fn ema(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let alpha = 2.0 / (period as f64 + 1.0);

        // Find first non-NaN value
        let mut first_valid = 0;
        while first_valid < n && values[first_valid].is_nan() {
            first_valid += 1;
        }

        if first_valid + period > n {
            return result;
        }

        // Initial SMA
        let mut sum = 0.0;
        for i in first_valid..(first_valid + period) {
            sum += values[i];
        }
        result[first_valid + period - 1] = sum / period as f64;

        // EMA calculation
        for i in (first_valid + period)..n {
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate VWMACD values.
    /// Returns (VWMACD Line, Signal Line, Histogram)
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        if n < self.slow_period {
            return (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate VWMAs
        let vwma_fast = self.vwma(close, volume, self.fast_period);
        let vwma_slow = self.vwma(close, volume, self.slow_period);

        // Calculate VWMACD Line
        let mut vwmacd_line = vec![f64::NAN; n];
        for i in 0..n {
            if !vwma_fast[i].is_nan() && !vwma_slow[i].is_nan() {
                vwmacd_line[i] = vwma_fast[i] - vwma_slow[i];
            }
        }

        // Calculate Signal Line (EMA of VWMACD)
        let signal_line = self.ema(&vwmacd_line, self.signal_period);

        // Calculate Histogram
        let mut histogram = vec![f64::NAN; n];
        for i in 0..n {
            if !vwmacd_line[i].is_nan() && !signal_line[i].is_nan() {
                histogram[i] = vwmacd_line[i] - signal_line[i];
            }
        }

        (vwmacd_line, signal_line, histogram)
    }
}

impl Default for VWMACD {
    fn default() -> Self {
        Self {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
        }
    }
}

impl TechnicalIndicator for VWMACD {
    fn name(&self) -> &str {
        "VWMACD"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.slow_period + self.signal_period - 1;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (vwmacd, signal, histogram) = self.calculate(&data.close, &data.volume);

        // Return all three components: VWMACD line, signal line, histogram
        Ok(IndicatorOutput::triple(vwmacd, signal, histogram))
    }

    fn min_periods(&self) -> usize {
        self.slow_period + self.signal_period - 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for VWMACD {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (vwmacd, signal, _) = self.calculate(&data.close, &data.volume);

        if vwmacd.len() < 2 || signal.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = vwmacd.len();
        let curr_macd = vwmacd[n - 1];
        let prev_macd = vwmacd[n - 2];
        let curr_sig = signal[n - 1];
        let prev_sig = signal[n - 2];

        if curr_macd.is_nan() || curr_sig.is_nan() || prev_macd.is_nan() || prev_sig.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: VWMACD crosses above signal
        if prev_macd <= prev_sig && curr_macd > curr_sig {
            return Ok(IndicatorSignal::Bullish);
        }
        // Bearish: VWMACD crosses below signal
        if prev_macd >= prev_sig && curr_macd < curr_sig {
            return Ok(IndicatorSignal::Bearish);
        }

        // Additional: Zero line crossover
        if prev_macd <= 0.0 && curr_macd > 0.0 {
            return Ok(IndicatorSignal::Bullish);
        }
        if prev_macd >= 0.0 && curr_macd < 0.0 {
            return Ok(IndicatorSignal::Bearish);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (vwmacd, signal, _) = self.calculate(&data.close, &data.volume);

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..vwmacd.len() {
            if vwmacd[i].is_nan() || signal[i].is_nan() ||
               vwmacd[i - 1].is_nan() || signal[i - 1].is_nan()
            {
                signals.push(IndicatorSignal::Neutral);
            } else if vwmacd[i - 1] <= signal[i - 1] && vwmacd[i] > signal[i] {
                // Bullish signal crossover
                signals.push(IndicatorSignal::Bullish);
            } else if vwmacd[i - 1] >= signal[i - 1] && vwmacd[i] < signal[i] {
                // Bearish signal crossover
                signals.push(IndicatorSignal::Bearish);
            } else if vwmacd[i - 1] <= 0.0 && vwmacd[i] > 0.0 {
                // Bullish zero crossover
                signals.push(IndicatorSignal::Bullish);
            } else if vwmacd[i - 1] >= 0.0 && vwmacd[i] < 0.0 {
                // Bearish zero crossover
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
    fn test_vwmacd_basic() {
        let vwmacd = VWMACD::new(5, 10, 3);
        let n = 30;
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        let volume = vec![1000.0; n];

        let (macd_line, signal_line, histogram) = vwmacd.calculate(&close, &volume);

        assert_eq!(macd_line.len(), n);
        assert_eq!(signal_line.len(), n);
        assert_eq!(histogram.len(), n);

        // VWMACD should be positive in uptrend (fast > slow)
        for i in 15..n {
            assert!(!macd_line[i].is_nan());
            assert!(macd_line[i] > 0.0, "MACD line should be positive in uptrend");
        }
    }

    #[test]
    fn test_vwmacd_downtrend() {
        let vwmacd = VWMACD::new(5, 10, 3);
        let n = 30;
        let close: Vec<f64> = (0..n).map(|i| 150.0 - (i as f64 * 0.5)).collect();
        let volume = vec![1000.0; n];

        let (macd_line, _, _) = vwmacd.calculate(&close, &volume);

        // VWMACD should be negative in downtrend (fast < slow)
        for i in 15..n {
            assert!(!macd_line[i].is_nan());
            assert!(macd_line[i] < 0.0, "MACD line should be negative in downtrend");
        }
    }

    #[test]
    fn test_vwmacd_with_varying_volume() {
        let vwmacd = VWMACD::new(5, 10, 3);
        let n = 30;
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        // Higher volume on higher prices
        let volume: Vec<f64> = close.iter().map(|c| c * 10.0).collect();

        let (macd_line, signal_line, histogram) = vwmacd.calculate(&close, &volume);

        // All outputs should be valid
        for i in 15..n {
            assert!(!macd_line[i].is_nan());
            assert!(!signal_line[i].is_nan());
            assert!(!histogram[i].is_nan());
        }
    }

    #[test]
    fn test_vwmacd_histogram() {
        let vwmacd = VWMACD::new(5, 10, 3);
        let n = 30;
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        let volume = vec![1000.0; n];

        let (macd_line, signal_line, histogram) = vwmacd.calculate(&close, &volume);

        // Histogram = MACD - Signal
        for i in 15..n {
            if !histogram[i].is_nan() {
                let expected = macd_line[i] - signal_line[i];
                assert!((histogram[i] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_vwmacd_insufficient_data() {
        let vwmacd = VWMACD::new(12, 26, 9);
        let close = vec![100.0; 20];
        let volume = vec![1000.0; 20];

        let (macd_line, _, _) = vwmacd.calculate(&close, &volume);

        // All NaN with insufficient data
        for val in macd_line {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_vwmacd_signal_bullish_crossover() {
        let vwmacd = VWMACD::new(3, 6, 3);
        // Downtrend followed by uptrend for crossover
        let mut close: Vec<f64> = (0..10).map(|i| 110.0 - i as f64).collect();
        close.extend((0..15).map(|i| 100.0 + i as f64 * 2.0));
        let volume = vec![1000.0; close.len()];

        let signals = vwmacd.signals(&OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 1.0).collect(),
            low: close.iter().map(|x| x - 1.0).collect(),
            close: close.clone(),
            volume: volume.clone(),
        }).unwrap();

        // Should have some bullish signals
        let bullish_count = signals.iter().filter(|s| **s == IndicatorSignal::Bullish).count();
        assert!(bullish_count > 0, "Should have bullish signals during reversal");
    }

    #[test]
    fn test_vwmacd_signal_bearish_crossover() {
        let vwmacd = VWMACD::new(3, 6, 3);
        // Uptrend followed by downtrend for crossover
        let mut close: Vec<f64> = (0..10).map(|i| 100.0 + i as f64).collect();
        close.extend((0..15).map(|i| 110.0 - i as f64 * 2.0));
        let volume = vec![1000.0; close.len()];

        let signals = vwmacd.signals(&OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 1.0).collect(),
            low: close.iter().map(|x| x - 1.0).collect(),
            close: close.clone(),
            volume: volume.clone(),
        }).unwrap();

        // Should have some bearish signals
        let bearish_count = signals.iter().filter(|s| **s == IndicatorSignal::Bearish).count();
        assert!(bearish_count > 0, "Should have bearish signals during reversal");
    }

    #[test]
    fn test_vwmacd_technical_indicator() {
        let vwmacd = VWMACD::default();
        assert_eq!(vwmacd.name(), "VWMACD");
        assert_eq!(vwmacd.min_periods(), 26 + 9 - 1);
        assert_eq!(vwmacd.output_features(), 3);
    }

    #[test]
    fn test_vwmacd_empty() {
        let vwmacd = VWMACD::default();
        let (macd, signal, hist) = vwmacd.calculate(&[], &[]);
        assert!(macd.is_empty());
        assert!(signal.is_empty());
        assert!(hist.is_empty());
    }

    #[test]
    fn test_vwmacd_equal_volume() {
        let vwmacd = VWMACD::new(5, 10, 3);
        let n = 30;
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let volume = vec![1000.0; n];

        let (macd_line_vw, _, _) = vwmacd.calculate(&close, &volume);

        // With equal volume, VWMA = SMA, so VWMACD should behave like regular MACD
        // Just verify it produces valid output
        for i in 15..n {
            assert!(!macd_line_vw[i].is_nan());
        }
    }
}
