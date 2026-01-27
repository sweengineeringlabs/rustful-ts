//! Kase Permission Stochastic implementation.
//!
//! A modified stochastic oscillator developed by Cynthia Kase that uses
//! True Range for normalization, providing improved volatility sensitivity.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// Kase Permission Stochastic - IND-183
///
/// A modified stochastic oscillator that uses True Range to expand the
/// high/low range, providing better normalization during volatile periods.
///
/// Formula:
/// 1. True Range = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
/// 2. Modified High = High + (True Range / 2)
/// 3. Modified Low = Low - (True Range / 2)
/// 4. %K = ((Close - Lowest Modified Low) / (Highest Modified High - Lowest Modified Low)) * 100
/// 5. %D = SMA(%K, smooth_period)
#[derive(Debug, Clone)]
pub struct KasePermissionStochastic {
    period: usize,
    smooth_period: usize,
    overbought: f64,
    oversold: f64,
}

impl KasePermissionStochastic {
    /// Create a new Kase Permission Stochastic indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for the stochastic calculation
    /// * `smooth_period` - Smoothing period for %D line (SMA of %K)
    pub fn new(period: usize, smooth_period: usize) -> Self {
        Self {
            period,
            smooth_period,
            overbought: 80.0,
            oversold: 20.0,
        }
    }

    /// Set overbought threshold.
    pub fn with_overbought(mut self, level: f64) -> Self {
        self.overbought = level;
        self
    }

    /// Set oversold threshold.
    pub fn with_oversold(mut self, level: f64) -> Self {
        self.oversold = level;
        self
    }

    /// Calculate True Range for each bar.
    fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n == 0 {
            return vec![];
        }

        let mut tr = Vec::with_capacity(n);
        tr.push(high[0] - low[0]); // First TR is just high-low

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        tr
    }

    /// Simple Moving Average calculation.
    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period || period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];

        // Calculate initial sum
        let mut sum: f64 = data[..period].iter().filter(|x| !x.is_nan()).sum();
        let valid_count = data[..period].iter().filter(|x| !x.is_nan()).count();

        if valid_count == period {
            result.push(sum / period as f64);
        } else {
            result.push(f64::NAN);
        }

        // Rolling sum
        for i in period..n {
            let old = if data[i - period].is_nan() {
                0.0
            } else {
                data[i - period]
            };
            let new = if data[i].is_nan() { 0.0 } else { data[i] };
            sum = sum - old + new;
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate Kase Permission Stochastic values.
    ///
    /// Returns a tuple of (%K line, %D line).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        if n < self.period || self.period == 0 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate True Range
        let tr = Self::true_range(high, low, close);

        // Calculate Modified High and Modified Low
        let modified_high: Vec<f64> = high
            .iter()
            .zip(tr.iter())
            .map(|(&h, &t)| h + t / 2.0)
            .collect();

        let modified_low: Vec<f64> = low
            .iter()
            .zip(tr.iter())
            .map(|(&l, &t)| l - t / 2.0)
            .collect();

        // Calculate %K
        let mut k_line = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window_mod_high = &modified_high[start..=i];
            let window_mod_low = &modified_low[start..=i];

            let highest = window_mod_high
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lowest = window_mod_low.iter().cloned().fold(f64::INFINITY, f64::min);

            let range = highest - lowest;
            if range.abs() < 1e-10 {
                k_line.push(50.0); // Neutral when no range
            } else {
                let k = ((close[i] - lowest) / range) * 100.0;
                // Clamp to [0, 100] to handle floating point precision issues
                k_line.push(k.clamp(0.0, 100.0));
            }
        }

        // Calculate %D (SMA of %K)
        let d_line = Self::sma(&k_line, self.smooth_period);

        (k_line, d_line)
    }
}

impl Default for KasePermissionStochastic {
    fn default() -> Self {
        Self::new(14, 3)
    }
}

impl TechnicalIndicator for KasePermissionStochastic {
    fn name(&self) -> &str {
        "KasePermStoch"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.period + self.smooth_period - 1;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(k_line, d_line))
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period - 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for KasePermissionStochastic {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);

        if k_line.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = k_line.len();
        let k = k_line[n - 1];
        let d = d_line[n - 1];
        let prev_k = k_line[n - 2];
        let prev_d = d_line[n - 2];

        if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: %K crosses above %D in oversold zone
        if prev_k <= prev_d && k > d && k < self.oversold + 10.0 {
            Ok(IndicatorSignal::Bullish)
        }
        // Bearish: %K crosses below %D in overbought zone
        else if prev_k >= prev_d && k < d && k > self.overbought - 10.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);
        let n = k_line.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let k = k_line[i];
            let d = d_line[i];
            let prev_k = k_line[i - 1];
            let prev_d = d_line[i - 1];

            if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            // Bullish: %K crosses above %D in oversold zone
            if prev_k <= prev_d && k > d && k < self.oversold + 10.0 {
                signals.push(IndicatorSignal::Bullish);
            }
            // Bearish: %K crosses below %D in overbought zone
            else if prev_k >= prev_d && k < d && k > self.overbought - 10.0 {
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
    fn test_kase_permission_stochastic_basic() {
        let kps = KasePermissionStochastic::new(14, 3);

        // Create sample OHLC data
        let high: Vec<f64> = (0..30).map(|i| 105.0 + (i as f64).sin() * 5.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 95.0 + (i as f64).sin() * 5.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();

        let (k_line, d_line) = kps.calculate(&high, &low, &close);

        assert_eq!(k_line.len(), 30);
        assert_eq!(d_line.len(), 30);

        // First (period - 1) values should be NaN for %K
        for i in 0..13 {
            assert!(k_line[i].is_nan(), "k_line[{}] should be NaN", i);
        }

        // %K values should be between 0 and 100 after warmup
        for i in 13..30 {
            assert!(
                !k_line[i].is_nan(),
                "k_line[{}] should be valid after warmup",
                i
            );
            assert!(
                k_line[i] >= 0.0 && k_line[i] <= 100.0,
                "k_line[{}] = {} is out of range [0, 100]",
                i,
                k_line[i]
            );
        }

        // %D has initial NaN values from SMA warmup (smooth_period - 1 = 2)
        for i in 0..2 {
            assert!(d_line[i].is_nan(), "d_line[{}] should be NaN", i);
        }

        // After the combined warmup (period + smooth_period - 2 = 15), %D is fully valid
        // with all input %K values being non-NaN
        assert!(
            !d_line[15].is_nan(),
            "d_line[15] should be valid (first fully valid %D)"
        );

        // All %D values after full warmup should be in valid range
        for i in 15..30 {
            assert!(
                d_line[i] >= 0.0 && d_line[i] <= 100.0,
                "d_line[{}] = {} is out of range [0, 100]",
                i,
                d_line[i]
            );
        }
    }

    #[test]
    fn test_kase_permission_stochastic_default() {
        let kps = KasePermissionStochastic::default();
        assert_eq!(kps.period, 14);
        assert_eq!(kps.smooth_period, 3);
        assert_eq!(kps.overbought, 80.0);
        assert_eq!(kps.oversold, 20.0);
    }

    #[test]
    fn test_kase_permission_stochastic_range() {
        let kps = KasePermissionStochastic::default();
        let n = 60;
        let high: Vec<f64> = (0..n)
            .map(|i| 110.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();
        let low: Vec<f64> = (0..n)
            .map(|i| 90.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();

        let (k, d) = kps.calculate(&high, &low, &close);

        // Values should be in range [0, 100]
        for val in k.iter().chain(d.iter()) {
            if !val.is_nan() {
                assert!(
                    *val >= -1e-10 && *val <= 100.0 + 1e-10,
                    "Value {} out of range [0, 100]",
                    val
                );
            }
        }
    }

    #[test]
    fn test_kase_permission_stochastic_with_levels() {
        let kps = KasePermissionStochastic::new(14, 3)
            .with_overbought(70.0)
            .with_oversold(30.0);

        assert_eq!(kps.overbought, 70.0);
        assert_eq!(kps.oversold, 30.0);
    }

    #[test]
    fn test_kase_permission_stochastic_technical_indicator() {
        let kps = KasePermissionStochastic::new(14, 3);

        assert_eq!(kps.name(), "KasePermStoch");
        assert_eq!(kps.min_periods(), 16); // period + smooth_period - 1
        assert_eq!(kps.output_features(), 2);
    }

    #[test]
    fn test_kase_permission_stochastic_compute() {
        let kps = KasePermissionStochastic::new(14, 3);

        let high: Vec<f64> = (0..30).map(|i| 105.0 + (i as f64).sin() * 5.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 95.0 + (i as f64).sin() * 5.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        let volume: Vec<f64> = vec![1000.0; 30];

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let output = kps.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert_eq!(output.secondary.unwrap().len(), 30);
    }

    #[test]
    fn test_kase_permission_stochastic_insufficient_data() {
        let kps = KasePermissionStochastic::new(14, 3);

        let high: Vec<f64> = vec![105.0; 10];
        let low: Vec<f64> = vec![95.0; 10];
        let close: Vec<f64> = vec![100.0; 10];
        let volume: Vec<f64> = vec![1000.0; 10];

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let result = kps.compute(&data);
        assert!(result.is_err());

        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 16);
            assert_eq!(got, 10);
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    #[test]
    fn test_kase_permission_stochastic_signals() {
        let kps = KasePermissionStochastic::new(14, 3);

        let high: Vec<f64> = (0..50).map(|i| 105.0 + (i as f64).sin() * 5.0).collect();
        let low: Vec<f64> = (0..50).map(|i| 95.0 + (i as f64).sin() * 5.0).collect();
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        let volume: Vec<f64> = vec![1000.0; 50];

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let signal = kps.signal(&data);
        assert!(signal.is_ok());

        let signals = kps.signals(&data);
        assert!(signals.is_ok());
        assert_eq!(signals.unwrap().len(), 50);
    }

    #[test]
    fn test_kase_permission_stochastic_true_range_modification() {
        // Test that the modified high/low actually incorporates True Range
        let kps = KasePermissionStochastic::new(5, 2);

        // Create data with varying volatility
        let high = vec![
            102.0, 104.0, 108.0, 106.0, 105.0, 107.0, 109.0, 108.0, 106.0, 105.0,
        ];
        let low = vec![98.0, 96.0, 92.0, 94.0, 95.0, 93.0, 91.0, 92.0, 94.0, 95.0];
        let close = vec![
            100.0, 102.0, 95.0, 104.0, 100.0, 106.0, 93.0, 106.0, 95.0, 103.0,
        ];

        let (k, d) = kps.calculate(&high, &low, &close);

        // Verify output lengths
        assert_eq!(k.len(), 10);
        assert_eq!(d.len(), 10);

        // First 4 values should be NaN for %K
        for i in 0..4 {
            assert!(k[i].is_nan());
        }

        // Valid %K values
        for i in 4..10 {
            assert!(!k[i].is_nan());
            assert!(k[i] >= 0.0 && k[i] <= 100.0);
        }
    }

    #[test]
    fn test_kase_permission_empty_data() {
        let kps = KasePermissionStochastic::new(14, 3);

        let (k, d) = kps.calculate(&[], &[], &[]);

        assert!(k.is_empty());
        assert!(d.is_empty());
    }

    #[test]
    fn test_kase_permission_single_bar() {
        let kps = KasePermissionStochastic::new(14, 3);

        let (k, d) = kps.calculate(&[100.0], &[95.0], &[98.0]);

        assert_eq!(k.len(), 1);
        assert_eq!(d.len(), 1);
        assert!(k[0].is_nan());
        assert!(d[0].is_nan());
    }

    #[test]
    fn test_kase_permission_flat_market() {
        // When market is flat (no movement), should return neutral 50.0
        let kps = KasePermissionStochastic::new(5, 2);

        let high = vec![100.0; 20];
        let low = vec![100.0; 20];
        let close = vec![100.0; 20];

        let (k, d) = kps.calculate(&high, &low, &close);

        // After warmup period, values should be 50.0 (neutral)
        for i in 4..20 {
            assert!(
                (k[i] - 50.0).abs() < 1e-10,
                "Expected 50.0 for flat market, got {}",
                k[i]
            );
        }
    }
}
