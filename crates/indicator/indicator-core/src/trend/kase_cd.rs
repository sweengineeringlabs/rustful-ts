//! Kase Convergence/Divergence (Kase CD) - IND-184
//!
//! Developed by Cynthia Kase, the Kase CD is a momentum-based trend indicator
//! derived from the Kase Peak Oscillator methodology.
//!
//! The indicator measures trend momentum by analyzing the relationship between
//! price range and statistical volatility, then comparing fast and slow
//! smoothed versions similar to MACD.
//!
//! Components:
//! - Kase Peak Oscillator: Normalized range measure
//! - Kase CD Line: Fast KPO - Slow KPO
//! - Signal Line: EMA of Kase CD
//! - Histogram: Kase CD - Signal
//!
//! Signals:
//! - Bullish: Kase CD crosses above signal line
//! - Bearish: Kase CD crosses below signal line
//! - Divergence: Price makes new high/low but CD doesn't confirm

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Kase CD Output containing CD line, signal line, and histogram.
#[derive(Debug, Clone)]
pub struct KaseCDOutput {
    /// Kase CD line (fast - slow KPO)
    pub cd: Vec<f64>,
    /// Signal line (EMA of CD)
    pub signal: Vec<f64>,
    /// Histogram (CD - Signal)
    pub histogram: Vec<f64>,
}

/// Kase Convergence/Divergence - IND-184
///
/// A momentum trend indicator based on Cynthia Kase's methodology.
/// Similar to MACD but uses range-normalized volatility measures.
#[derive(Debug, Clone)]
pub struct KaseCD {
    /// Period for True Range / ATR calculation
    atr_period: usize,
    /// Fast period for KPO smoothing
    fast_period: usize,
    /// Slow period for KPO smoothing
    slow_period: usize,
    /// Signal line period
    signal_period: usize,
}

impl KaseCD {
    /// Create new Kase CD indicator.
    ///
    /// # Arguments
    /// * `atr_period` - Period for ATR calculation (default: 14)
    /// * `fast_period` - Fast smoothing period (default: 8)
    /// * `slow_period` - Slow smoothing period (default: 21)
    /// * `signal_period` - Signal line EMA period (default: 9)
    pub fn new(atr_period: usize, fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            atr_period,
            fast_period,
            slow_period,
            signal_period,
        }
    }

    /// Calculate Kase CD values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> KaseCDOutput {
        let n = high.len();
        let min_len = self.atr_period + self.slow_period + self.signal_period;

        if n < min_len {
            return KaseCDOutput {
                cd: vec![f64::NAN; n],
                signal: vec![f64::NAN; n],
                histogram: vec![f64::NAN; n],
            };
        }

        // Step 1: Calculate True Range
        let tr = self.calculate_true_range(high, low, close);

        // Step 2: Calculate ATR using EMA
        let atr = self.ema(&tr, self.atr_period);

        // Step 3: Calculate Kase Peak Oscillator
        let kpo = self.calculate_kase_peak(high, low, &atr);

        // Step 4: Calculate fast and slow smoothed KPO
        let fast_kpo = self.ema(&kpo, self.fast_period);
        let slow_kpo = self.ema(&kpo, self.slow_period);

        // Step 5: Kase CD = Fast KPO - Slow KPO
        let mut cd = vec![f64::NAN; n];
        for i in 0..n {
            if !fast_kpo[i].is_nan() && !slow_kpo[i].is_nan() {
                cd[i] = fast_kpo[i] - slow_kpo[i];
            }
        }

        // Step 6: Signal line = EMA of CD
        let signal = self.ema(&cd, self.signal_period);

        // Step 7: Histogram = CD - Signal
        let mut histogram = vec![f64::NAN; n];
        for i in 0..n {
            if !cd[i].is_nan() && !signal[i].is_nan() {
                histogram[i] = cd[i] - signal[i];
            }
        }

        KaseCDOutput { cd, signal, histogram }
    }

    /// Calculate True Range.
    fn calculate_true_range(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut tr = vec![high[0] - low[0]];

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        tr
    }

    /// Calculate Kase Peak Oscillator values.
    ///
    /// Peak = max(RWH, RWL) where:
    /// - RWH = (High - Low[atr_period]) / (ATR * sqrt(atr_period))
    /// - RWL = (High[atr_period] - Low) / (ATR * sqrt(atr_period))
    fn calculate_kase_peak(&self, high: &[f64], low: &[f64], atr: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut peak = vec![f64::NAN; n];
        let sqrt_period = (self.atr_period as f64).sqrt();

        for i in self.atr_period..n {
            if atr[i].is_nan() || atr[i] == 0.0 {
                continue;
            }

            let divisor = atr[i] * sqrt_period;

            // Range Weighted High: current high vs past low
            let rwh = (high[i] - low[i - self.atr_period]) / divisor;

            // Range Weighted Low: past high vs current low
            let rwl = (high[i - self.atr_period] - low[i]) / divisor;

            // Peak is the maximum absolute deviation
            peak[i] = rwh.max(rwl);
        }

        peak
    }

    /// Calculate Exponential Moving Average.
    fn ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);

        // Find first valid value for SMA seed
        let mut sum = 0.0;
        let mut count = 0;
        let mut first_valid = None;

        for i in 0..n {
            if !data[i].is_nan() {
                sum += data[i];
                count += 1;
                if count == period {
                    first_valid = Some(i);
                    break;
                }
            }
        }

        if let Some(start) = first_valid {
            result[start] = sum / period as f64;

            for i in (start + 1)..n {
                if !data[i].is_nan() && !result[i - 1].is_nan() {
                    result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                } else if !result[i - 1].is_nan() {
                    result[i] = result[i - 1];
                }
            }
        }

        result
    }
}

impl Default for KaseCD {
    fn default() -> Self {
        Self::new(14, 8, 21, 9)
    }
}

impl TechnicalIndicator for KaseCD {
    fn name(&self) -> &str {
        "KaseCD"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_len = self.atr_period + self.slow_period + self.signal_period;
        if data.high.len() < min_len {
            return Err(IndicatorError::InsufficientData {
                required: min_len,
                got: data.high.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(result.cd, result.signal, result.histogram))
    }

    fn min_periods(&self) -> usize {
        self.atr_period + self.slow_period + self.signal_period
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for KaseCD {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low, &data.close);

        if result.cd.len() < 2 || result.signal.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let cd_last = result.cd[result.cd.len() - 1];
        let cd_prev = result.cd[result.cd.len() - 2];
        let sig_last = result.signal[result.signal.len() - 1];
        let sig_prev = result.signal[result.signal.len() - 2];

        if cd_last.is_nan() || cd_prev.is_nan() || sig_last.is_nan() || sig_prev.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish crossover: CD crosses above signal
        if cd_last > sig_last && cd_prev <= sig_prev {
            Ok(IndicatorSignal::Bullish)
        }
        // Bearish crossover: CD crosses below signal
        else if cd_last < sig_last && cd_prev >= sig_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..result.cd.len().min(result.signal.len()) {
            let cd_curr = result.cd[i];
            let cd_prev = result.cd[i - 1];
            let sig_curr = result.signal[i];
            let sig_prev = result.signal[i - 1];

            if cd_curr.is_nan() || cd_prev.is_nan() || sig_curr.is_nan() || sig_prev.is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if cd_curr > sig_curr && cd_prev <= sig_prev {
                signals.push(IndicatorSignal::Bullish);
            } else if cd_curr < sig_curr && cd_prev >= sig_prev {
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

    fn generate_uptrend_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0 + i as f64;
            high.push(base + 2.0);
            low.push(base - 2.0);
            close.push(base + 1.0);
        }

        (high, low, close)
    }

    fn generate_volatile_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0 + (i as f64 * 0.3).sin() * 10.0;
            high.push(base + 3.0);
            low.push(base - 3.0);
            close.push(base + (i as f64 * 0.5).sin());
        }

        (high, low, close)
    }

    #[test]
    fn test_kase_cd_basic() {
        let kase = KaseCD::default();
        let (high, low, close) = generate_uptrend_data(100);
        let result = kase.calculate(&high, &low, &close);

        assert_eq!(result.cd.len(), 100);
        assert_eq!(result.signal.len(), 100);
        assert_eq!(result.histogram.len(), 100);
    }

    #[test]
    fn test_kase_cd_nan_prefix() {
        let kase = KaseCD::new(14, 8, 21, 9);
        let (high, low, close) = generate_uptrend_data(100);
        let result = kase.calculate(&high, &low, &close);

        // First several values should be NaN
        assert!(result.cd[0].is_nan());
        assert!(result.signal[0].is_nan());

        // Later values should be valid
        let last_cd = result.cd.last().unwrap();
        assert!(!last_cd.is_nan(), "Last CD value should be valid");
    }

    #[test]
    fn test_kase_cd_volatile_data() {
        let kase = KaseCD::default();
        let (high, low, close) = generate_volatile_data(100);
        let result = kase.calculate(&high, &low, &close);

        // Should have valid values
        let valid_cd: Vec<_> = result.cd.iter().filter(|x| !x.is_nan()).collect();
        assert!(!valid_cd.is_empty(), "Should have valid CD values");
    }

    #[test]
    fn test_kase_cd_output_length() {
        let kase = KaseCD::new(10, 5, 15, 7);
        let (high, low, close) = generate_uptrend_data(50);
        let result = kase.calculate(&high, &low, &close);

        // All outputs should have same length as input
        assert_eq!(result.cd.len(), 50);
        assert_eq!(result.signal.len(), 50);
        assert_eq!(result.histogram.len(), 50);
    }

    #[test]
    fn test_kase_cd_histogram_consistency() {
        let kase = KaseCD::default();
        let (high, low, close) = generate_uptrend_data(100);
        let result = kase.calculate(&high, &low, &close);

        // Verify histogram = CD - Signal where both are valid
        for i in 0..result.cd.len() {
            if !result.cd[i].is_nan() && !result.signal[i].is_nan() {
                let expected = result.cd[i] - result.signal[i];
                assert!(
                    (result.histogram[i] - expected).abs() < 1e-10,
                    "Histogram mismatch at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_kase_cd_default() {
        let kase = KaseCD::default();
        assert_eq!(kase.atr_period, 14);
        assert_eq!(kase.fast_period, 8);
        assert_eq!(kase.slow_period, 21);
        assert_eq!(kase.signal_period, 9);
    }

    #[test]
    fn test_technical_indicator_trait() {
        let kase = KaseCD::default();
        assert_eq!(kase.name(), "KaseCD");
        assert_eq!(kase.min_periods(), 14 + 21 + 9);
        assert_eq!(kase.output_features(), 3);
    }

    #[test]
    fn test_signal_indicator_crossover() {
        let kase = KaseCD::default();
        let (high, low, close) = generate_volatile_data(150);

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 150],
        };

        let signals = kase.signals(&data).unwrap();

        // Should have some signals (not all Neutral)
        let non_neutral: Vec<_> = signals
            .iter()
            .filter(|s| **s != IndicatorSignal::Neutral)
            .collect();

        // In volatile data, we should see some crossovers
        // (not guaranteed but likely)
        println!("Non-neutral signals: {}", non_neutral.len());
    }

    #[test]
    fn test_insufficient_data() {
        let kase = KaseCD::default();
        let data = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![101.0; 10],
            low: vec![99.0; 10],
            close: vec![100.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = kase.compute(&data);
        assert!(result.is_err());
    }
}
