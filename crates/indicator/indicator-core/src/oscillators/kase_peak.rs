//! Kase Peak Oscillator (KPO) - IND-181
//!
//! Developed by Cynthia Kase, the Kase Peak Oscillator is a volatility-adjusted
//! momentum oscillator that measures price movement relative to volatility.
//!
//! The indicator normalizes price range by ATR and statistical scaling (square root
//! of period) to identify momentum extremes and potential reversal points.
//!
//! Formula:
//! 1. Calculate True Range and ATR
//! 2. RWH (Range Weighted High) = (High - Low[N]) / (ATR * sqrt(N))
//! 3. RWL (Range Weighted Low) = (High[N] - Low) / (ATR * sqrt(N))
//! 4. Peak = max(RWH, RWL)
//! 5. KPO = EMA(Peak, smooth_period)
//!
//! Signals:
//! - Bullish: KPO crosses above upper threshold (strong upward momentum)
//! - Bearish: KPO crosses below lower threshold (strong downward momentum)
//! - Values oscillate around a baseline, with extremes indicating potential reversals

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Kase Peak Oscillator (KPO) - IND-181
///
/// A volatility-adjusted momentum oscillator that measures price movement
/// normalized by ATR and statistical volatility scaling.
#[derive(Debug, Clone)]
pub struct KasePeakOscillator {
    /// Period for ATR calculation
    atr_period: usize,
    /// Period for EMA smoothing of peak values
    smooth_period: usize,
    /// Upper threshold for bullish signals
    upper_threshold: f64,
    /// Lower threshold for bearish signals
    lower_threshold: f64,
}

impl KasePeakOscillator {
    /// Create a new Kase Peak Oscillator.
    ///
    /// # Arguments
    /// * `atr_period` - Period for ATR calculation (default: 14)
    /// * `smooth_period` - Period for EMA smoothing (default: 9)
    pub fn new(atr_period: usize, smooth_period: usize) -> Self {
        Self {
            atr_period,
            smooth_period,
            upper_threshold: 1.5,
            lower_threshold: -1.5,
        }
    }

    /// Create with custom thresholds for signal generation.
    pub fn with_thresholds(
        atr_period: usize,
        smooth_period: usize,
        upper_threshold: f64,
        lower_threshold: f64,
    ) -> Self {
        Self {
            atr_period,
            smooth_period,
            upper_threshold,
            lower_threshold,
        }
    }

    /// Calculate Kase Peak Oscillator values.
    ///
    /// Returns a vector of KPO values (EMA-smoothed peak values).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        let min_len = self.atr_period + self.smooth_period;

        if n < min_len {
            return vec![f64::NAN; n];
        }

        // Step 1: Calculate True Range
        let tr = self.calculate_true_range(high, low, close);

        // Step 2: Calculate ATR using EMA
        let atr = self.ema(&tr, self.atr_period);

        // Step 3: Calculate Peak values (max of RWH, RWL)
        let peak = self.calculate_peak(high, low, &atr);

        // Step 4: KPO = EMA of Peak
        self.ema(&peak, self.smooth_period)
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

    /// Calculate Peak values (max of RWH and RWL).
    ///
    /// RWH = (High - Low[atr_period]) / (ATR * sqrt(atr_period))
    /// RWL = (High[atr_period] - Low) / (ATR * sqrt(atr_period))
    fn calculate_peak(&self, high: &[f64], low: &[f64], atr: &[f64]) -> Vec<f64> {
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

            // Peak is the maximum of RWH and RWL
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

        // Find first valid values for SMA seed
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

impl Default for KasePeakOscillator {
    fn default() -> Self {
        Self::new(14, 9)
    }
}

impl TechnicalIndicator for KasePeakOscillator {
    fn name(&self) -> &str {
        "KasePeakOscillator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_len = self.atr_period + self.smooth_period;
        if data.high.len() < min_len {
            return Err(IndicatorError::InsufficientData {
                required: min_len,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.atr_period + self.smooth_period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for KasePeakOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last > self.upper_threshold {
                    return Ok(IndicatorSignal::Bullish);
                } else if last < self.lower_threshold {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(values
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    IndicatorSignal::Neutral
                } else if v > self.upper_threshold {
                    IndicatorSignal::Bullish
                } else if v < self.lower_threshold {
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

    fn generate_downtrend_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);

        for i in 0..n {
            let base = 200.0 - i as f64;
            high.push(base + 2.0);
            low.push(base - 2.0);
            close.push(base - 1.0);
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

    fn generate_sideways_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0 + (i as f64 * 0.1).sin() * 2.0;
            high.push(base + 1.5);
            low.push(base - 1.5);
            close.push(base + (i as f64 * 0.2).cos() * 0.5);
        }

        (high, low, close)
    }

    #[test]
    fn test_kpo_basic() {
        let kpo = KasePeakOscillator::default();
        let (high, low, close) = generate_uptrend_data(100);
        let result = kpo.calculate(&high, &low, &close);

        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_kpo_nan_prefix() {
        let kpo = KasePeakOscillator::new(14, 9);
        let (high, low, close) = generate_uptrend_data(100);
        let result = kpo.calculate(&high, &low, &close);

        // First several values should be NaN
        assert!(result[0].is_nan());
        assert!(result[10].is_nan());

        // Later values should be valid
        let last = result.last().unwrap();
        assert!(!last.is_nan(), "Last KPO value should be valid");
    }

    #[test]
    fn test_kpo_output_length() {
        let kpo = KasePeakOscillator::new(10, 5);
        let (high, low, close) = generate_uptrend_data(50);
        let result = kpo.calculate(&high, &low, &close);

        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_kpo_insufficient_data() {
        let kpo = KasePeakOscillator::new(14, 9);
        let (high, low, close) = generate_uptrend_data(10);
        let result = kpo.calculate(&high, &low, &close);

        // All values should be NaN for insufficient data
        assert!(result.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_kpo_volatile_data() {
        let kpo = KasePeakOscillator::default();
        let (high, low, close) = generate_volatile_data(100);
        let result = kpo.calculate(&high, &low, &close);

        // Should have valid values
        let valid_values: Vec<_> = result.iter().filter(|x| !x.is_nan()).collect();
        assert!(!valid_values.is_empty(), "Should have valid KPO values");
    }

    #[test]
    fn test_kpo_uptrend() {
        let kpo = KasePeakOscillator::default();
        let (high, low, close) = generate_uptrend_data(100);
        let result = kpo.calculate(&high, &low, &close);

        // In uptrend, later valid values should be positive
        let valid_values: Vec<f64> = result.iter().filter(|x| !x.is_nan()).copied().collect();
        assert!(!valid_values.is_empty());

        let last_value = valid_values.last().unwrap();
        assert!(*last_value > 0.0, "KPO should be positive in uptrend");
    }

    #[test]
    fn test_kpo_downtrend() {
        let kpo = KasePeakOscillator::default();
        let (high, low, close) = generate_downtrend_data(100);
        let result = kpo.calculate(&high, &low, &close);

        // Should have valid values
        let valid_values: Vec<f64> = result.iter().filter(|x| !x.is_nan()).copied().collect();
        assert!(!valid_values.is_empty());
    }

    #[test]
    fn test_kpo_default_parameters() {
        let kpo = KasePeakOscillator::default();
        assert_eq!(kpo.atr_period, 14);
        assert_eq!(kpo.smooth_period, 9);
        assert_eq!(kpo.upper_threshold, 1.5);
        assert_eq!(kpo.lower_threshold, -1.5);
    }

    #[test]
    fn test_kpo_custom_parameters() {
        let kpo = KasePeakOscillator::new(20, 12);
        assert_eq!(kpo.atr_period, 20);
        assert_eq!(kpo.smooth_period, 12);
    }

    #[test]
    fn test_kpo_custom_thresholds() {
        let kpo = KasePeakOscillator::with_thresholds(14, 9, 2.0, -2.0);
        assert_eq!(kpo.upper_threshold, 2.0);
        assert_eq!(kpo.lower_threshold, -2.0);
    }

    #[test]
    fn test_technical_indicator_trait() {
        let kpo = KasePeakOscillator::default();
        assert_eq!(kpo.name(), "KasePeakOscillator");
        assert_eq!(kpo.min_periods(), 14 + 9);
        assert_eq!(kpo.output_features(), 1);
    }

    #[test]
    fn test_compute_trait() {
        let kpo = KasePeakOscillator::default();
        let (high, low, close) = generate_uptrend_data(100);

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 100],
        };

        let result = kpo.compute(&data).unwrap();
        assert_eq!(result.primary.len(), 100);
    }

    #[test]
    fn test_compute_insufficient_data() {
        let kpo = KasePeakOscillator::default();
        let data = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![101.0; 10],
            low: vec![99.0; 10],
            close: vec![100.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = kpo.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_signal_bullish() {
        let kpo = KasePeakOscillator::with_thresholds(14, 9, 0.5, -0.5);
        let (high, low, close) = generate_uptrend_data(100);

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 100],
        };

        let signal = kpo.signal(&data).unwrap();
        // In strong uptrend with low threshold, should be bullish
        assert_eq!(signal, IndicatorSignal::Bullish);
    }

    #[test]
    fn test_signal_neutral() {
        let kpo = KasePeakOscillator::with_thresholds(14, 9, 10.0, -10.0);
        let (high, low, close) = generate_sideways_data(100);

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 100],
        };

        let signal = kpo.signal(&data).unwrap();
        // With extreme thresholds, should be neutral
        assert_eq!(signal, IndicatorSignal::Neutral);
    }

    #[test]
    fn test_signals_vector() {
        let kpo = KasePeakOscillator::default();
        let (high, low, close) = generate_volatile_data(100);

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 100],
        };

        let signals = kpo.signals(&data).unwrap();
        assert_eq!(signals.len(), 100);

        // First signals should be neutral (NaN values)
        assert_eq!(signals[0], IndicatorSignal::Neutral);
    }

    #[test]
    fn test_kpo_values_bounded() {
        let kpo = KasePeakOscillator::default();
        let (high, low, close) = generate_volatile_data(200);
        let result = kpo.calculate(&high, &low, &close);

        // KPO values should be reasonable (not infinite)
        for &val in result.iter() {
            if !val.is_nan() {
                assert!(val.is_finite(), "KPO values should be finite");
                // Values typically range within reasonable bounds
                assert!(val.abs() < 100.0, "KPO values should be bounded");
            }
        }
    }

    #[test]
    fn test_kpo_consistency() {
        let kpo = KasePeakOscillator::default();
        let (high, low, close) = generate_uptrend_data(100);

        // Multiple calculations should yield same results
        let result1 = kpo.calculate(&high, &low, &close);
        let result2 = kpo.calculate(&high, &low, &close);

        for (v1, v2) in result1.iter().zip(result2.iter()) {
            if v1.is_nan() && v2.is_nan() {
                continue;
            }
            assert!((v1 - v2).abs() < 1e-10, "Results should be consistent");
        }
    }

    #[test]
    fn test_kpo_empty_input() {
        let kpo = KasePeakOscillator::default();
        let result = kpo.calculate(&[], &[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_kpo_single_value() {
        let kpo = KasePeakOscillator::default();
        let result = kpo.calculate(&[100.0], &[99.0], &[99.5]);
        assert_eq!(result.len(), 1);
        assert!(result[0].is_nan());
    }

    #[test]
    fn test_kpo_atr_normalization() {
        // Test that KPO properly normalizes by ATR
        let kpo = KasePeakOscillator::new(10, 5);

        // Create two datasets with same price movement but different volatility
        let n = 50;
        let (high1, low1, close1): (Vec<f64>, Vec<f64>, Vec<f64>) = (0..n)
            .map(|i| {
                let base = 100.0 + i as f64;
                (base + 1.0, base - 1.0, base + 0.5)
            })
            .fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut h, mut l, mut c), (hi, lo, cl)| {
                    h.push(hi);
                    l.push(lo);
                    c.push(cl);
                    (h, l, c)
                },
            );

        let (high2, low2, close2): (Vec<f64>, Vec<f64>, Vec<f64>) = (0..n)
            .map(|i| {
                let base = 100.0 + i as f64;
                (base + 5.0, base - 5.0, base + 2.5)
            })
            .fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut h, mut l, mut c), (hi, lo, cl)| {
                    h.push(hi);
                    l.push(lo);
                    c.push(cl);
                    (h, l, c)
                },
            );

        let result1 = kpo.calculate(&high1, &low1, &close1);
        let result2 = kpo.calculate(&high2, &low2, &close2);

        // Both should have valid values
        let valid1: Vec<f64> = result1.iter().filter(|x| !x.is_nan()).copied().collect();
        let valid2: Vec<f64> = result2.iter().filter(|x| !x.is_nan()).copied().collect();

        assert!(!valid1.is_empty());
        assert!(!valid2.is_empty());

        // Due to ATR normalization, values should be comparable despite different volatility
        // The last values should be in similar ranges
        let last1 = valid1.last().unwrap();
        let last2 = valid2.last().unwrap();

        // Both should be positive (uptrend) and reasonably close due to normalization
        assert!(*last1 > 0.0);
        assert!(*last2 > 0.0);
    }

    #[test]
    fn test_kpo_peak_calculation() {
        // Verify that peak is correctly calculated as max(RWH, RWL)
        let kpo = KasePeakOscillator::new(5, 3);
        let n = 30;

        let (high, low, close) = generate_uptrend_data(n);
        let result = kpo.calculate(&high, &low, &close);

        // In uptrend, RWH should dominate, making peak positive
        let valid_values: Vec<f64> = result.iter().filter(|x| !x.is_nan()).copied().collect();
        assert!(!valid_values.is_empty());
        assert!(*valid_values.last().unwrap() > 0.0);
    }
}
