//! Kase Bars indicator implementation.
//!
//! Volatility-normalized OHLC bars developed by Cynthia Kase.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries, OHLCV,
};

/// Kase Bars output containing normalized OHLCV data.
#[derive(Debug, Clone)]
pub struct KaseBarsOutput {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
}

impl KaseBarsOutput {
    /// Convert to OHLCVSeries for further analysis.
    pub fn to_series(&self) -> OHLCVSeries {
        OHLCVSeries {
            open: self.open.clone(),
            high: self.high.clone(),
            low: self.low.clone(),
            close: self.close.clone(),
            volume: vec![0.0; self.close.len()],
        }
    }

    /// Get a specific candle at index.
    pub fn candle(&self, idx: usize) -> Option<OHLCV> {
        if idx >= self.close.len() {
            return None;
        }
        Some(OHLCV::new(
            self.open[idx],
            self.high[idx],
            self.low[idx],
            self.close[idx],
            0.0,
        ))
    }

    /// Get the number of bars.
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

/// Kase Bars indicator.
///
/// Developed by Cynthia Kase, Kase Bars normalize price data by volatility
/// to create bars that account for varying market conditions. This allows
/// for better comparison of price movements across different volatility regimes.
///
/// The normalization process:
/// 1. Calculate True Range for each bar
/// 2. Calculate average True Range (ATR) over a lookback period
/// 3. Normalize OHLC values relative to the ATR-adjusted price scale
///
/// Kase Bars help identify:
/// - Trend strength independent of volatility
/// - Overbought/oversold conditions
/// - Pattern recognition with volatility context
///
/// Output:
/// - Primary: Normalized close prices
/// - Secondary: Normalized range (high - low) / ATR
#[derive(Debug, Clone)]
pub struct KaseBars {
    /// Period for ATR calculation.
    period: usize,
    /// Smoothing factor for normalization.
    smoothing: usize,
}

impl KaseBars {
    /// Create a new Kase Bars indicator.
    ///
    /// # Arguments
    /// * `period` - Period for ATR calculation (typically 10-20)
    /// * `smoothing` - Smoothing period for output (typically same as period)
    pub fn new(period: usize, smoothing: usize) -> Self {
        Self {
            period: period.max(1),
            smoothing: smoothing.max(1),
        }
    }

    /// Create with Kase's recommended parameters.
    ///
    /// Period: 10, Smoothing: 10
    pub fn kase_default() -> Self {
        Self::new(10, 10)
    }

    /// Create with custom period and same smoothing.
    pub fn with_period(period: usize) -> Self {
        Self::new(period, period)
    }

    /// Calculate true range for each bar.
    fn calculate_true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n == 0 {
            return vec![];
        }

        let mut tr = Vec::with_capacity(n);
        tr.push(high[0] - low[0]);

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        tr
    }

    /// Calculate exponential moving average.
    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period as f64 + 1.0);

        // Find first non-NaN value
        let mut start_idx = 0;
        while start_idx < n && data[start_idx].is_nan() {
            start_idx += 1;
        }

        if start_idx >= n {
            return result;
        }

        // Use SMA for initial value
        if start_idx + period <= n {
            let mut sum = 0.0;
            let mut count = 0;
            for i in start_idx..(start_idx + period) {
                if !data[i].is_nan() {
                    sum += data[i];
                    count += 1;
                }
            }
            if count > 0 {
                result[start_idx + period - 1] = sum / count as f64;
            }
        }

        // Calculate EMA
        for i in (start_idx + period)..n {
            if !result[i - 1].is_nan() && !data[i].is_nan() {
                result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
            }
        }

        result
    }

    /// Calculate simple moving average.
    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let mut sum: f64 = data[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        for i in period..n {
            sum = sum - data[i - period] + data[i];
            result[i] = sum / period as f64;
        }

        result
    }

    /// Calculate Kase Bars.
    pub fn calculate(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> KaseBarsOutput {
        let n = close.len();
        if n == 0 {
            return KaseBarsOutput {
                open: vec![],
                high: vec![],
                low: vec![],
                close: vec![],
            };
        }

        if n < self.period {
            return KaseBarsOutput {
                open: vec![f64::NAN; n],
                high: vec![f64::NAN; n],
                low: vec![f64::NAN; n],
                close: vec![f64::NAN; n],
            };
        }

        // Calculate ATR
        let tr = Self::calculate_true_range(high, low, close);
        let atr = Self::ema(&tr, self.period);

        // Normalize OHLC by ATR
        let mut kase_open = vec![f64::NAN; n];
        let mut kase_high = vec![f64::NAN; n];
        let mut kase_low = vec![f64::NAN; n];
        let mut kase_close = vec![f64::NAN; n];

        // Calculate typical price as baseline
        let typical: Vec<f64> = (0..n)
            .map(|i| (high[i] + low[i] + close[i]) / 3.0)
            .collect();
        let typical_sma = Self::sma(&typical, self.smoothing);

        for i in 0..n {
            if atr[i].is_nan() || atr[i] < f64::EPSILON || typical_sma[i].is_nan() {
                continue;
            }

            let baseline = typical_sma[i];
            let scale = atr[i];

            // Normalize: center around baseline and scale by ATR
            kase_open[i] = (open[i] - baseline) / scale;
            kase_high[i] = (high[i] - baseline) / scale;
            kase_low[i] = (low[i] - baseline) / scale;
            kase_close[i] = (close[i] - baseline) / scale;
        }

        KaseBarsOutput {
            open: kase_open,
            high: kase_high,
            low: kase_low,
            close: kase_close,
        }
    }

    /// Calculate normalized range (volatility indicator).
    pub fn calculate_normalized_range(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<f64> {
        let n = high.len();
        if n < self.period {
            return vec![f64::NAN; n];
        }

        let tr = Self::calculate_true_range(high, low, close);
        let atr = Self::ema(&tr, self.period);

        (0..n)
            .map(|i| {
                if atr[i].is_nan() || atr[i] < f64::EPSILON {
                    f64::NAN
                } else {
                    (high[i] - low[i]) / atr[i]
                }
            })
            .collect()
    }

    /// Calculate bar type indicator.
    ///
    /// Returns values indicating bar character:
    /// - Positive: Bullish (close > open)
    /// - Negative: Bearish (close < open)
    /// - Zero: Doji (close == open)
    ///
    /// Magnitude indicates relative body size.
    pub fn calculate_bar_type(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<f64> {
        let n = close.len();
        if n < self.period {
            return vec![f64::NAN; n];
        }

        let tr = Self::calculate_true_range(high, low, close);
        let atr = Self::ema(&tr, self.period);

        (0..n)
            .map(|i| {
                if atr[i].is_nan() || atr[i] < f64::EPSILON {
                    return f64::NAN;
                }

                let body = close[i] - open[i];
                let range = high[i] - low[i];

                if range < f64::EPSILON {
                    return 0.0;
                }

                // Normalized body: body size relative to range, scaled by ATR ratio
                let body_ratio = body / range;
                let atr_ratio = range / atr[i];

                body_ratio * atr_ratio
            })
            .collect()
    }

    /// Detect strong momentum bars (normalized).
    ///
    /// A strong bar has:
    /// - Body > 60% of range
    /// - Range > 0.8 ATR
    pub fn detect_strong_bars(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<i32> {
        let n = close.len();
        if n < self.period {
            return vec![0; n];
        }

        let tr = Self::calculate_true_range(high, low, close);
        let atr = Self::ema(&tr, self.period);

        (0..n)
            .map(|i| {
                if atr[i].is_nan() || atr[i] < f64::EPSILON {
                    return 0;
                }

                let body = (close[i] - open[i]).abs();
                let range = high[i] - low[i];

                if range < f64::EPSILON {
                    return 0;
                }

                let body_pct = body / range;
                let range_vs_atr = range / atr[i];

                if body_pct > 0.6 && range_vs_atr > 0.8 {
                    if close[i] > open[i] {
                        1 // Strong bullish
                    } else {
                        -1 // Strong bearish
                    }
                } else {
                    0
                }
            })
            .collect()
    }

    /// Get statistics about recent Kase Bars.
    pub fn statistics(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        lookback: usize,
    ) -> Option<KaseBarsStats> {
        let kase = self.calculate(open, high, low, close);
        let n = kase.close.len();

        if n < lookback {
            return None;
        }

        let start = n - lookback;
        let mut valid_closes = Vec::new();
        let mut valid_ranges = Vec::new();

        for i in start..n {
            if !kase.close[i].is_nan() {
                valid_closes.push(kase.close[i]);
                let range = kase.high[i] - kase.low[i];
                if !range.is_nan() {
                    valid_ranges.push(range);
                }
            }
        }

        if valid_closes.is_empty() {
            return None;
        }

        let close_mean = valid_closes.iter().sum::<f64>() / valid_closes.len() as f64;
        let close_variance = valid_closes.iter()
            .map(|x| (x - close_mean).powi(2))
            .sum::<f64>() / valid_closes.len() as f64;
        let close_std = close_variance.sqrt();

        let range_mean = if valid_ranges.is_empty() {
            f64::NAN
        } else {
            valid_ranges.iter().sum::<f64>() / valid_ranges.len() as f64
        };

        let current_close = kase.close[n - 1];
        let zscore = if close_std > f64::EPSILON && !current_close.is_nan() {
            (current_close - close_mean) / close_std
        } else {
            f64::NAN
        };

        Some(KaseBarsStats {
            mean_normalized_close: close_mean,
            std_normalized_close: close_std,
            mean_normalized_range: range_mean,
            current_zscore: zscore,
        })
    }
}

/// Statistics about Kase Bars.
#[derive(Debug, Clone)]
pub struct KaseBarsStats {
    /// Mean of normalized close prices.
    pub mean_normalized_close: f64,
    /// Standard deviation of normalized close prices.
    pub std_normalized_close: f64,
    /// Mean of normalized ranges.
    pub mean_normalized_range: f64,
    /// Z-score of current normalized close.
    pub current_zscore: f64,
}

impl TechnicalIndicator for KaseBars {
    fn name(&self) -> &str {
        "KaseBars"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.period.max(self.smoothing);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let kase = self.calculate(&data.open, &data.high, &data.low, &data.close);
        let normalized_range = self.calculate_normalized_range(&data.high, &data.low, &data.close);

        Ok(IndicatorOutput::dual(kase.close, normalized_range))
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.smoothing)
    }

    fn output_features(&self) -> usize {
        4 // open, high, low, close (all normalized)
    }
}

impl SignalIndicator for KaseBars {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let kase = self.calculate(&data.open, &data.high, &data.low, &data.close);
        let n = kase.close.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let last_close = kase.close[n - 1];
        let last_open = kase.open[n - 1];

        if last_close.is_nan() || last_open.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal based on normalized bar direction and extreme readings
        if last_close > 1.5 {
            // Extremely bullish (close > 1.5 ATR above baseline)
            Ok(IndicatorSignal::Bullish)
        } else if last_close < -1.5 {
            // Extremely bearish
            Ok(IndicatorSignal::Bearish)
        } else if last_close > last_open {
            Ok(IndicatorSignal::Bullish)
        } else if last_close < last_open {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let kase = self.calculate(&data.open, &data.high, &data.low, &data.close);
        let n = kase.close.len();

        let signals = (0..n)
            .map(|i| {
                let close = kase.close[i];
                let open = kase.open[i];

                if close.is_nan() || open.is_nan() {
                    IndicatorSignal::Neutral
                } else if close > 1.5 {
                    IndicatorSignal::Bullish
                } else if close < -1.5 {
                    IndicatorSignal::Bearish
                } else if close > open {
                    IndicatorSignal::Bullish
                } else if close < open {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> OHLCVSeries {
        let mut data = OHLCVSeries::new();
        for i in 0..30 {
            let base = 100.0 + i as f64 * 0.5;
            let volatility = (i as f64 * 0.3).sin() * 2.0;
            data.open.push(base + volatility * 0.3);
            data.high.push(base + 2.0 + volatility.abs());
            data.low.push(base - 2.0 - volatility.abs());
            data.close.push(base + volatility * 0.5);
            data.volume.push(1000.0);
        }
        data
    }

    #[test]
    fn test_kase_bars_basic() {
        let kb = KaseBars::kase_default();
        let data = create_test_data();

        let kase = kb.calculate(&data.open, &data.high, &data.low, &data.close);

        assert_eq!(kase.open.len(), 30);
        assert_eq!(kase.high.len(), 30);
        assert_eq!(kase.low.len(), 30);
        assert_eq!(kase.close.len(), 30);

        // Values after warmup should be valid
        let warmup = kb.min_periods();
        for i in (warmup + 5)..30 {
            assert!(!kase.close[i].is_nan(), "close[{}] is NaN", i);
            // High should be >= close >= low
            assert!(kase.high[i] >= kase.close[i] || (kase.high[i] - kase.close[i]).abs() < f64::EPSILON,
                "high < close at index {}", i);
            assert!(kase.close[i] >= kase.low[i] || (kase.close[i] - kase.low[i]).abs() < f64::EPSILON,
                "close < low at index {}", i);
        }
    }

    #[test]
    fn test_kase_default_params() {
        let kb = KaseBars::kase_default();
        assert_eq!(kb.period, 10);
        assert_eq!(kb.smoothing, 10);
    }

    #[test]
    fn test_with_period() {
        let kb = KaseBars::with_period(15);
        assert_eq!(kb.period, 15);
        assert_eq!(kb.smoothing, 15);
    }

    #[test]
    fn test_true_range_calculation() {
        let high = vec![102.0, 104.0, 103.0];
        let low = vec![98.0, 100.0, 99.0];
        let close = vec![100.0, 103.0, 101.0];

        let tr = KaseBars::calculate_true_range(&high, &low, &close);

        assert_eq!(tr.len(), 3);
        assert!((tr[0] - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalized_range() {
        let kb = KaseBars::kase_default();
        let data = create_test_data();

        let norm_range = kb.calculate_normalized_range(&data.high, &data.low, &data.close);

        assert_eq!(norm_range.len(), 30);

        // Valid values should be positive
        for i in 15..30 {
            if !norm_range[i].is_nan() {
                assert!(norm_range[i] > 0.0, "normalized range at {} should be positive", i);
            }
        }
    }

    #[test]
    fn test_bar_type() {
        let kb = KaseBars::kase_default();
        let data = create_test_data();

        let bar_type = kb.calculate_bar_type(&data.open, &data.high, &data.low, &data.close);

        assert_eq!(bar_type.len(), 30);

        for i in 15..30 {
            if !bar_type[i].is_nan() {
                // Sign should match close vs open direction
                if data.close[i] > data.open[i] {
                    assert!(bar_type[i] >= 0.0 || bar_type[i].abs() < f64::EPSILON,
                        "bullish bar should have positive type at {}", i);
                } else if data.close[i] < data.open[i] {
                    assert!(bar_type[i] <= 0.0 || bar_type[i].abs() < f64::EPSILON,
                        "bearish bar should have negative type at {}", i);
                }
            }
        }
    }

    #[test]
    fn test_strong_bars_detection() {
        let kb = KaseBars::kase_default();

        // Create data with clear strong bars
        let mut data = OHLCVSeries::new();
        for i in 0..20 {
            let base = 100.0 + i as f64;
            data.open.push(base);
            data.high.push(base + 3.0);
            data.low.push(base - 0.5);
            data.close.push(base + 2.5); // Strong bullish body
            data.volume.push(1000.0);
        }

        let strong = kb.detect_strong_bars(&data.open, &data.high, &data.low, &data.close);

        assert_eq!(strong.len(), 20);
    }

    #[test]
    fn test_statistics() {
        let kb = KaseBars::kase_default();
        let data = create_test_data();

        let stats = kb.statistics(&data.open, &data.high, &data.low, &data.close, 10);

        assert!(stats.is_some());
        let s = stats.unwrap();
        assert!(!s.mean_normalized_close.is_nan());
        assert!(s.std_normalized_close >= 0.0);
    }

    #[test]
    fn test_kase_bars_output_methods() {
        let kb = KaseBars::kase_default();
        let data = create_test_data();

        let kase = kb.calculate(&data.open, &data.high, &data.low, &data.close);

        assert_eq!(kase.len(), 30);
        assert!(!kase.is_empty());

        // Test candle retrieval
        let candle = kase.candle(20);
        assert!(candle.is_some());

        // Test to_series conversion
        let series = kase.to_series();
        assert_eq!(series.len(), 30);
    }

    #[test]
    fn test_technical_indicator_trait() {
        let kb = KaseBars::kase_default();
        let data = create_test_data();

        let output = kb.compute(&data).unwrap();

        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_signal_indicator_trait() {
        let kb = KaseBars::kase_default();
        let data = create_test_data();

        let signal = kb.signal(&data).unwrap();
        assert!(matches!(signal, IndicatorSignal::Neutral | IndicatorSignal::Bullish | IndicatorSignal::Bearish));

        let signals = kb.signals(&data).unwrap();
        assert_eq!(signals.len(), 30);
    }

    #[test]
    fn test_insufficient_data() {
        let kb = KaseBars::kase_default();

        let mut data = OHLCVSeries::new();
        for i in 0..5 {
            data.open.push(100.0 + i as f64);
            data.high.push(102.0 + i as f64);
            data.low.push(98.0 + i as f64);
            data.close.push(101.0 + i as f64);
            data.volume.push(1000.0);
        }

        let result = kb.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_min_periods() {
        let kb = KaseBars::new(15, 20);
        assert_eq!(kb.min_periods(), 20); // max(15, 20) = 20
    }

    #[test]
    fn test_ema_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ema = KaseBars::ema(&data, 3);

        assert_eq!(ema.len(), 10);
        assert!(ema[0].is_nan());
        assert!(ema[1].is_nan());
        // First valid EMA value at index 2
        assert!(!ema[2].is_nan());
        // EMA should follow upward trend
        for i in 3..10 {
            if !ema[i].is_nan() && !ema[i-1].is_nan() {
                assert!(ema[i] > ema[i-1], "EMA should increase in uptrend");
            }
        }
    }

    #[test]
    fn test_sma_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = KaseBars::sma(&data, 3);

        assert_eq!(sma.len(), 5);
        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < f64::EPSILON); // (1+2+3)/3 = 2
        assert!((sma[3] - 3.0).abs() < f64::EPSILON); // (2+3+4)/3 = 3
        assert!((sma[4] - 4.0).abs() < f64::EPSILON); // (3+4+5)/3 = 4
    }

    #[test]
    fn test_normalization_scaling() {
        let kb = KaseBars::kase_default();

        // Create high volatility data
        let mut high_vol_data = OHLCVSeries::new();
        for i in 0..30 {
            let base = 100.0 + i as f64;
            high_vol_data.open.push(base);
            high_vol_data.high.push(base + 10.0); // Large range
            high_vol_data.low.push(base - 10.0);
            high_vol_data.close.push(base + 5.0);
            high_vol_data.volume.push(1000.0);
        }

        // Create low volatility data
        let mut low_vol_data = OHLCVSeries::new();
        for i in 0..30 {
            let base = 100.0 + i as f64;
            low_vol_data.open.push(base);
            low_vol_data.high.push(base + 1.0); // Small range
            low_vol_data.low.push(base - 1.0);
            low_vol_data.close.push(base + 0.5);
            low_vol_data.volume.push(1000.0);
        }

        let high_vol_kase = kb.calculate(
            &high_vol_data.open, &high_vol_data.high,
            &high_vol_data.low, &high_vol_data.close
        );
        let low_vol_kase = kb.calculate(
            &low_vol_data.open, &low_vol_data.high,
            &low_vol_data.low, &low_vol_data.close
        );

        // Both should produce normalized values in similar ranges
        // (that's the point of Kase Bars - normalize for volatility)
        let mut high_vol_range_sum = 0.0;
        let mut low_vol_range_sum = 0.0;
        let mut count = 0;

        for i in 20..30 {
            if !high_vol_kase.close[i].is_nan() && !low_vol_kase.close[i].is_nan() {
                high_vol_range_sum += (high_vol_kase.high[i] - high_vol_kase.low[i]).abs();
                low_vol_range_sum += (low_vol_kase.high[i] - low_vol_kase.low[i]).abs();
                count += 1;
            }
        }

        if count > 0 {
            let high_vol_avg = high_vol_range_sum / count as f64;
            let low_vol_avg = low_vol_range_sum / count as f64;

            // Normalized ranges should be in same ballpark (within 5x of each other)
            assert!(high_vol_avg / low_vol_avg < 5.0 && low_vol_avg / high_vol_avg < 5.0,
                "Normalized ranges should be comparable: high={}, low={}", high_vol_avg, low_vol_avg);
        }
    }

    #[test]
    fn test_empty_data() {
        let kb = KaseBars::kase_default();
        let data = OHLCVSeries::new();

        let kase = kb.calculate(&data.open, &data.high, &data.low, &data.close);
        assert!(kase.is_empty());
    }

    #[test]
    fn test_candle_invalid_index() {
        let kb = KaseBars::kase_default();
        let data = create_test_data();

        let kase = kb.calculate(&data.open, &data.high, &data.low, &data.close);
        assert!(kase.candle(100).is_none());
    }
}
