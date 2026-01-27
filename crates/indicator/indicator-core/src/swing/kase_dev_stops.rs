//! Kase Deviation Stops indicator implementation.
//!
//! Volatility-based stops using standard deviation of true range.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Kase Deviation Stops indicator.
///
/// Developed by Cynthia Kase, this indicator uses the standard deviation of
/// true range to create adaptive stop levels. It provides multiple stop levels
/// based on different deviation multipliers.
///
/// The calculation involves:
/// 1. Calculate True Range for each bar
/// 2. Calculate average True Range (ATR) over the period
/// 3. Calculate standard deviation of True Range
/// 4. Set stop levels at price +/- (ATR + k * StdDev(TR))
///
/// Output:
/// - Primary: Dev Stop 1 (tighter stop using dev_multiplier_1)
/// - Secondary: Dev Stop 2 (wider stop using dev_multiplier_2)
/// - Tertiary: Dev Stop 3 (widest stop using dev_multiplier_3)
#[derive(Debug, Clone)]
pub struct KaseDevStops {
    /// Period for ATR and standard deviation calculation.
    period: usize,
    /// Multiplier for the first (tightest) deviation stop.
    dev_multiplier_1: f64,
    /// Multiplier for the second deviation stop.
    dev_multiplier_2: f64,
    /// Multiplier for the third (widest) deviation stop.
    dev_multiplier_3: f64,
}

impl KaseDevStops {
    /// Create a new Kase Deviation Stops indicator.
    ///
    /// # Arguments
    /// * `period` - Period for calculations (typically 10-30)
    /// * `dev_multiplier_1` - First deviation multiplier (typically 1.0)
    /// * `dev_multiplier_2` - Second deviation multiplier (typically 2.0)
    /// * `dev_multiplier_3` - Third deviation multiplier (typically 3.0)
    pub fn new(
        period: usize,
        dev_multiplier_1: f64,
        dev_multiplier_2: f64,
        dev_multiplier_3: f64,
    ) -> Self {
        Self {
            period: period.max(2),
            dev_multiplier_1: dev_multiplier_1.max(0.0),
            dev_multiplier_2: dev_multiplier_2.max(0.0),
            dev_multiplier_3: dev_multiplier_3.max(0.0),
        }
    }

    /// Create with Kase's recommended default parameters.
    ///
    /// Period: 10
    /// Multipliers: 1.0, 2.0, 3.0
    pub fn kase_default() -> Self {
        Self::new(10, 1.0, 2.0, 3.0)
    }

    /// Create with custom period but standard multipliers.
    pub fn with_period(period: usize) -> Self {
        Self::new(period, 1.0, 2.0, 3.0)
    }

    /// Create with single deviation multiplier.
    pub fn single(period: usize, multiplier: f64) -> Self {
        Self::new(period, multiplier, multiplier, multiplier)
    }

    /// Calculate true range for each bar.
    fn calculate_true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n == 0 {
            return vec![];
        }

        let mut tr = Vec::with_capacity(n);
        tr.push(high[0] - low[0]); // First bar: just high-low

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        tr
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

    /// Calculate standard deviation.
    fn standard_deviation(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        for i in (period - 1)..n {
            let start = i + 1 - period;
            let window = &data[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            result[i] = variance.sqrt();
        }

        result
    }

    /// Calculate Kase Deviation Stops.
    ///
    /// Returns (stop_1, stop_2, stop_3) where each is (long_stop, short_stop).
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> KaseDevStopsOutput {
        let n = high.len();
        if n < self.period {
            return KaseDevStopsOutput {
                dev_stop_1_long: vec![f64::NAN; n],
                dev_stop_1_short: vec![f64::NAN; n],
                dev_stop_2_long: vec![f64::NAN; n],
                dev_stop_2_short: vec![f64::NAN; n],
                dev_stop_3_long: vec![f64::NAN; n],
                dev_stop_3_short: vec![f64::NAN; n],
            };
        }

        let tr = Self::calculate_true_range(high, low, close);
        let atr = Self::sma(&tr, self.period);
        let tr_std = Self::standard_deviation(&tr, self.period);

        let mut output = KaseDevStopsOutput {
            dev_stop_1_long: vec![f64::NAN; n],
            dev_stop_1_short: vec![f64::NAN; n],
            dev_stop_2_long: vec![f64::NAN; n],
            dev_stop_2_short: vec![f64::NAN; n],
            dev_stop_3_long: vec![f64::NAN; n],
            dev_stop_3_short: vec![f64::NAN; n],
        };

        for i in (self.period - 1)..n {
            if atr[i].is_nan() || tr_std[i].is_nan() {
                continue;
            }

            let current_close = close[i];

            // Calculate stop distances
            let stop_dist_1 = atr[i] + self.dev_multiplier_1 * tr_std[i];
            let stop_dist_2 = atr[i] + self.dev_multiplier_2 * tr_std[i];
            let stop_dist_3 = atr[i] + self.dev_multiplier_3 * tr_std[i];

            // Long stops (below price)
            output.dev_stop_1_long[i] = current_close - stop_dist_1;
            output.dev_stop_2_long[i] = current_close - stop_dist_2;
            output.dev_stop_3_long[i] = current_close - stop_dist_3;

            // Short stops (above price)
            output.dev_stop_1_short[i] = current_close + stop_dist_1;
            output.dev_stop_2_short[i] = current_close + stop_dist_2;
            output.dev_stop_3_short[i] = current_close + stop_dist_3;
        }

        output
    }

    /// Calculate trailing stops that only move in favorable direction.
    pub fn calculate_trailing(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> KaseTrailingStops {
        let output = self.calculate(high, low, close);
        let n = close.len();

        let mut trailing = KaseTrailingStops {
            long_stop_1: vec![f64::NAN; n],
            long_stop_2: vec![f64::NAN; n],
            long_stop_3: vec![f64::NAN; n],
            short_stop_1: vec![f64::NAN; n],
            short_stop_2: vec![f64::NAN; n],
            short_stop_3: vec![f64::NAN; n],
        };

        let mut max_long_1 = f64::NEG_INFINITY;
        let mut max_long_2 = f64::NEG_INFINITY;
        let mut max_long_3 = f64::NEG_INFINITY;
        let mut min_short_1 = f64::INFINITY;
        let mut min_short_2 = f64::INFINITY;
        let mut min_short_3 = f64::INFINITY;

        for i in 0..n {
            // Long stops (ratchet up)
            if !output.dev_stop_1_long[i].is_nan() {
                max_long_1 = max_long_1.max(output.dev_stop_1_long[i]);
                trailing.long_stop_1[i] = max_long_1;
            }
            if !output.dev_stop_2_long[i].is_nan() {
                max_long_2 = max_long_2.max(output.dev_stop_2_long[i]);
                trailing.long_stop_2[i] = max_long_2;
            }
            if !output.dev_stop_3_long[i].is_nan() {
                max_long_3 = max_long_3.max(output.dev_stop_3_long[i]);
                trailing.long_stop_3[i] = max_long_3;
            }

            // Short stops (ratchet down)
            if !output.dev_stop_1_short[i].is_nan() {
                min_short_1 = min_short_1.min(output.dev_stop_1_short[i]);
                trailing.short_stop_1[i] = min_short_1;
            }
            if !output.dev_stop_2_short[i].is_nan() {
                min_short_2 = min_short_2.min(output.dev_stop_2_short[i]);
                trailing.short_stop_2[i] = min_short_2;
            }
            if !output.dev_stop_3_short[i].is_nan() {
                min_short_3 = min_short_3.min(output.dev_stop_3_short[i]);
                trailing.short_stop_3[i] = min_short_3;
            }
        }

        trailing
    }

    /// Get current stop levels.
    pub fn current_stops(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Option<KaseCurrentStops> {
        let output = self.calculate(high, low, close);
        let n = close.len();

        if n == 0 {
            return None;
        }

        let last = n - 1;
        if output.dev_stop_1_long[last].is_nan() {
            return None;
        }

        Some(KaseCurrentStops {
            price: close[last],
            long_stop_1: output.dev_stop_1_long[last],
            long_stop_2: output.dev_stop_2_long[last],
            long_stop_3: output.dev_stop_3_long[last],
            short_stop_1: output.dev_stop_1_short[last],
            short_stop_2: output.dev_stop_2_short[last],
            short_stop_3: output.dev_stop_3_short[last],
        })
    }
}

/// Output structure for Kase Deviation Stops.
#[derive(Debug, Clone)]
pub struct KaseDevStopsOutput {
    /// First deviation long stop (tightest)
    pub dev_stop_1_long: Vec<f64>,
    /// First deviation short stop
    pub dev_stop_1_short: Vec<f64>,
    /// Second deviation long stop
    pub dev_stop_2_long: Vec<f64>,
    /// Second deviation short stop
    pub dev_stop_2_short: Vec<f64>,
    /// Third deviation long stop (widest)
    pub dev_stop_3_long: Vec<f64>,
    /// Third deviation short stop
    pub dev_stop_3_short: Vec<f64>,
}

/// Trailing stops output.
#[derive(Debug, Clone)]
pub struct KaseTrailingStops {
    pub long_stop_1: Vec<f64>,
    pub long_stop_2: Vec<f64>,
    pub long_stop_3: Vec<f64>,
    pub short_stop_1: Vec<f64>,
    pub short_stop_2: Vec<f64>,
    pub short_stop_3: Vec<f64>,
}

/// Current stop levels.
#[derive(Debug, Clone)]
pub struct KaseCurrentStops {
    pub price: f64,
    pub long_stop_1: f64,
    pub long_stop_2: f64,
    pub long_stop_3: f64,
    pub short_stop_1: f64,
    pub short_stop_2: f64,
    pub short_stop_3: f64,
}

impl TechnicalIndicator for KaseDevStops {
    fn name(&self) -> &str {
        "KaseDevStops"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let output = self.calculate(&data.high, &data.low, &data.close);

        // Return long stops as primary output (most commonly used)
        Ok(IndicatorOutput::triple(
            output.dev_stop_1_long,
            output.dev_stop_2_long,
            output.dev_stop_3_long,
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        6 // 3 long stops + 3 short stops
    }
}

impl SignalIndicator for KaseDevStops {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate(&data.high, &data.low, &data.close);
        let n = data.close.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let last = n - 1;
        let close = data.close[last];
        let long_stop_2 = output.dev_stop_2_long.get(last).copied().unwrap_or(f64::NAN);
        let short_stop_2 = output.dev_stop_2_short.get(last).copied().unwrap_or(f64::NAN);

        if long_stop_2.is_nan() || short_stop_2.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Use the middle stop (dev_stop_2) for signals
        if close < long_stop_2 {
            Ok(IndicatorSignal::Bearish)
        } else if close > short_stop_2 {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate(&data.high, &data.low, &data.close);
        let n = data.close.len();

        let signals = (0..n)
            .map(|i| {
                let close = data.close[i];
                let long_stop = output.dev_stop_2_long.get(i).copied().unwrap_or(f64::NAN);
                let short_stop = output.dev_stop_2_short.get(i).copied().unwrap_or(f64::NAN);

                if long_stop.is_nan() || short_stop.is_nan() {
                    IndicatorSignal::Neutral
                } else if close < long_stop {
                    IndicatorSignal::Bearish
                } else if close > short_stop {
                    IndicatorSignal::Bullish
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
        // Create trending data with volatility
        for i in 0..30 {
            let base = 100.0 + i as f64 * 0.5;
            let volatility = (i as f64 * 0.5).sin() * 2.0;
            data.open.push(base);
            data.high.push(base + 2.0 + volatility.abs());
            data.low.push(base - 2.0 - volatility.abs());
            data.close.push(base + volatility);
            data.volume.push(1000.0);
        }
        data
    }

    #[test]
    fn test_kase_dev_stops_basic() {
        let kds = KaseDevStops::kase_default();
        let data = create_test_data();

        let output = kds.calculate(&data.high, &data.low, &data.close);

        assert_eq!(output.dev_stop_1_long.len(), 30);
        assert_eq!(output.dev_stop_2_long.len(), 30);
        assert_eq!(output.dev_stop_3_long.len(), 30);

        // First 9 values should be NaN (period=10)
        for i in 0..9 {
            assert!(output.dev_stop_1_long[i].is_nan());
            assert!(output.dev_stop_2_long[i].is_nan());
            assert!(output.dev_stop_3_long[i].is_nan());
        }

        // Values after warmup should be valid and ordered
        for i in 9..30 {
            assert!(!output.dev_stop_1_long[i].is_nan(), "Stop 1 at {} is NaN", i);
            assert!(!output.dev_stop_2_long[i].is_nan(), "Stop 2 at {} is NaN", i);
            assert!(!output.dev_stop_3_long[i].is_nan(), "Stop 3 at {} is NaN", i);

            // Long stops should be below close
            assert!(output.dev_stop_1_long[i] < data.close[i]);
            assert!(output.dev_stop_2_long[i] < data.close[i]);
            assert!(output.dev_stop_3_long[i] < data.close[i]);

            // Stop 3 should be below Stop 2, which is below Stop 1
            assert!(output.dev_stop_3_long[i] <= output.dev_stop_2_long[i],
                "Stop 3 ({}) > Stop 2 ({}) at index {}",
                output.dev_stop_3_long[i], output.dev_stop_2_long[i], i);
            assert!(output.dev_stop_2_long[i] <= output.dev_stop_1_long[i],
                "Stop 2 ({}) > Stop 1 ({}) at index {}",
                output.dev_stop_2_long[i], output.dev_stop_1_long[i], i);
        }
    }

    #[test]
    fn test_short_stops() {
        let kds = KaseDevStops::kase_default();
        let data = create_test_data();

        let output = kds.calculate(&data.high, &data.low, &data.close);

        // Values after warmup
        for i in 9..30 {
            // Short stops should be above close
            assert!(output.dev_stop_1_short[i] > data.close[i]);
            assert!(output.dev_stop_2_short[i] > data.close[i]);
            assert!(output.dev_stop_3_short[i] > data.close[i]);

            // Stop 3 should be above Stop 2, which is above Stop 1
            assert!(output.dev_stop_3_short[i] >= output.dev_stop_2_short[i]);
            assert!(output.dev_stop_2_short[i] >= output.dev_stop_1_short[i]);
        }
    }

    #[test]
    fn test_kase_default_params() {
        let kds = KaseDevStops::kase_default();
        assert_eq!(kds.period, 10);
        assert!((kds.dev_multiplier_1 - 1.0).abs() < f64::EPSILON);
        assert!((kds.dev_multiplier_2 - 2.0).abs() < f64::EPSILON);
        assert!((kds.dev_multiplier_3 - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_with_period() {
        let kds = KaseDevStops::with_period(20);
        assert_eq!(kds.period, 20);
        assert!((kds.dev_multiplier_1 - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_single_multiplier() {
        let kds = KaseDevStops::single(10, 2.0);
        assert_eq!(kds.period, 10);
        assert!((kds.dev_multiplier_1 - 2.0).abs() < f64::EPSILON);
        assert!((kds.dev_multiplier_2 - 2.0).abs() < f64::EPSILON);
        assert!((kds.dev_multiplier_3 - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_true_range_calculation() {
        let high = vec![102.0, 104.0, 103.0];
        let low = vec![98.0, 100.0, 99.0];
        let close = vec![100.0, 103.0, 101.0];

        let tr = KaseDevStops::calculate_true_range(&high, &low, &close);

        assert_eq!(tr.len(), 3);
        // First bar: just H-L
        assert!((tr[0] - 4.0).abs() < f64::EPSILON);
        // Second bar: max(104-100, |104-100|, |100-100|) = 4.0
        assert!((tr[1] - 4.0).abs() < f64::EPSILON);
        // Third bar: max(103-99, |103-103|, |99-103|) = 4.0
        assert!((tr[2] - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trailing_stops() {
        let kds = KaseDevStops::kase_default();
        let data = create_test_data();

        let trailing = kds.calculate_trailing(&data.high, &data.low, &data.close);

        // Verify long stops only move up
        let mut prev_long_1 = f64::NEG_INFINITY;
        for i in 9..30 {
            if !trailing.long_stop_1[i].is_nan() {
                assert!(trailing.long_stop_1[i] >= prev_long_1,
                    "Trailing long stop moved down at index {}", i);
                prev_long_1 = trailing.long_stop_1[i];
            }
        }

        // Verify short stops only move down
        let mut prev_short_1 = f64::INFINITY;
        for i in 9..30 {
            if !trailing.short_stop_1[i].is_nan() {
                assert!(trailing.short_stop_1[i] <= prev_short_1,
                    "Trailing short stop moved up at index {}", i);
                prev_short_1 = trailing.short_stop_1[i];
            }
        }
    }

    #[test]
    fn test_current_stops() {
        let kds = KaseDevStops::kase_default();
        let data = create_test_data();

        let current = kds.current_stops(&data.high, &data.low, &data.close);

        assert!(current.is_some());
        let stops = current.unwrap();

        // Verify ordering
        assert!(stops.long_stop_3 <= stops.long_stop_2);
        assert!(stops.long_stop_2 <= stops.long_stop_1);
        assert!(stops.long_stop_1 < stops.price);
        assert!(stops.price < stops.short_stop_1);
        assert!(stops.short_stop_1 <= stops.short_stop_2);
        assert!(stops.short_stop_2 <= stops.short_stop_3);
    }

    #[test]
    fn test_technical_indicator_trait() {
        let kds = KaseDevStops::kase_default();
        let data = create_test_data();

        let output = kds.compute(&data).unwrap();

        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_signal_indicator_trait() {
        let kds = KaseDevStops::kase_default();
        let data = create_test_data();

        let signal = kds.signal(&data).unwrap();
        assert!(matches!(signal, IndicatorSignal::Neutral | IndicatorSignal::Bullish | IndicatorSignal::Bearish));

        let signals = kds.signals(&data).unwrap();
        assert_eq!(signals.len(), 30);
    }

    #[test]
    fn test_insufficient_data() {
        let kds = KaseDevStops::kase_default();

        let mut data = OHLCVSeries::new();
        for i in 0..5 {
            data.open.push(100.0 + i as f64);
            data.high.push(102.0 + i as f64);
            data.low.push(98.0 + i as f64);
            data.close.push(101.0 + i as f64);
            data.volume.push(1000.0);
        }

        let result = kds.compute(&data);
        assert!(result.is_err());
        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 10);
            assert_eq!(got, 5);
        }
    }

    #[test]
    fn test_min_periods() {
        let kds = KaseDevStops::with_period(15);
        assert_eq!(kds.min_periods(), 15);
    }

    #[test]
    fn test_sma_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = KaseDevStops::sma(&data, 3);

        assert_eq!(sma.len(), 5);
        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < f64::EPSILON); // (1+2+3)/3 = 2
        assert!((sma[3] - 3.0).abs() < f64::EPSILON); // (2+3+4)/3 = 3
        assert!((sma[4] - 4.0).abs() < f64::EPSILON); // (3+4+5)/3 = 4
    }

    #[test]
    fn test_std_calculation() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = KaseDevStops::standard_deviation(&data, 4);

        assert_eq!(std.len(), 8);
        assert!(std[0].is_nan());
        assert!(std[1].is_nan());
        assert!(std[2].is_nan());
        // Index 3: values [2, 4, 4, 4], mean = 3.5, var = 0.75, std ~= 0.866
        assert!(!std[3].is_nan());
        assert!(std[3] > 0.0);
    }

    #[test]
    fn test_multiplier_effect() {
        let kds_low = KaseDevStops::new(10, 0.5, 1.0, 1.5);
        let kds_high = KaseDevStops::new(10, 2.0, 3.0, 4.0);
        let data = create_test_data();

        let output_low = kds_low.calculate(&data.high, &data.low, &data.close);
        let output_high = kds_high.calculate(&data.high, &data.low, &data.close);

        // Higher multipliers should give wider stops (lower long stops)
        for i in 9..30 {
            if !output_low.dev_stop_1_long[i].is_nan() && !output_high.dev_stop_1_long[i].is_nan() {
                assert!(output_high.dev_stop_1_long[i] < output_low.dev_stop_1_long[i],
                    "Higher multiplier should give lower long stop at index {}", i);
            }
        }
    }

    #[test]
    fn test_symmetry() {
        let kds = KaseDevStops::kase_default();
        let data = create_test_data();

        let output = kds.calculate(&data.high, &data.low, &data.close);

        // Long and short stops should be symmetric around close
        for i in 9..30 {
            let close = data.close[i];
            let long_dist = close - output.dev_stop_1_long[i];
            let short_dist = output.dev_stop_1_short[i] - close;

            // Distances should be equal
            assert!((long_dist - short_dist).abs() < 1e-10,
                "Asymmetric stops at index {}: long_dist={}, short_dist={}", i, long_dist, short_dist);
        }
    }
}
