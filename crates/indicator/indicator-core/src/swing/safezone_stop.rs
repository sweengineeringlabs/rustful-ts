//! Elder's SafeZone Stop indicator implementation.
//!
//! A directional stop loss system based on market noise levels.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Elder's SafeZone Stop indicator.
///
/// Dr. Alexander Elder's SafeZone Stop is a volatility-based trailing stop system
/// that adapts to market noise. It calculates stop levels based on directional
/// movement (upward or downward penetration) and a noise coefficient multiplier.
///
/// The indicator calculates:
/// - Downside Penetration (DP): Previous Low - Current Low (when positive)
/// - Upside Penetration (UP): Current High - Previous High (when positive)
/// - Average penetration over the lookback period
/// - Stop distance = Coefficient * Average Penetration
///
/// Output:
/// - Primary: Long stop (below price, for protecting long positions)
/// - Secondary: Short stop (above price, for protecting short positions)
#[derive(Debug, Clone)]
pub struct SafeZoneStop {
    /// Lookback period for averaging penetration.
    period: usize,
    /// Noise coefficient multiplier (typically 2.0-3.0).
    coefficient: f64,
}

impl SafeZoneStop {
    /// Create a new SafeZone Stop indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for averaging penetration (typically 10-22)
    /// * `coefficient` - Noise multiplier (typically 2.0-3.0)
    pub fn new(period: usize, coefficient: f64) -> Self {
        Self {
            period: period.max(1),
            coefficient: coefficient.max(0.1),
        }
    }

    /// Create with Elder's recommended parameters.
    ///
    /// Period: 22 (approximately one trading month)
    /// Coefficient: 2.5
    pub fn elder_default() -> Self {
        Self::new(22, 2.5)
    }

    /// Create with shorter-term parameters.
    ///
    /// Period: 10
    /// Coefficient: 2.0
    pub fn short_term() -> Self {
        Self::new(10, 2.0)
    }

    /// Calculate downside penetration values.
    ///
    /// Downside penetration occurs when the current low is below the previous low.
    fn calculate_downside_penetration(low: &[f64]) -> Vec<f64> {
        let n = low.len();
        if n < 2 {
            return vec![0.0; n];
        }

        let mut dp = vec![0.0; n];
        for i in 1..n {
            let penetration = low[i - 1] - low[i];
            dp[i] = if penetration > 0.0 { penetration } else { 0.0 };
        }
        dp
    }

    /// Calculate upside penetration values.
    ///
    /// Upside penetration occurs when the current high is above the previous high.
    fn calculate_upside_penetration(high: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < 2 {
            return vec![0.0; n];
        }

        let mut up = vec![0.0; n];
        for i in 1..n {
            let penetration = high[i] - high[i - 1];
            up[i] = if penetration > 0.0 { penetration } else { 0.0 };
        }
        up
    }

    /// Calculate average of non-zero penetration values.
    fn calculate_average_penetration(penetration: &[f64], period: usize, index: usize) -> f64 {
        if index < period {
            return 0.0;
        }

        let start = index + 1 - period;
        let end = index + 1;

        let mut sum = 0.0;
        let mut count = 0;

        for i in start..end {
            if penetration[i] > 0.0 {
                sum += penetration[i];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }

    /// Calculate SafeZone Stop levels.
    ///
    /// Returns (long_stop, short_stop) vectors.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n < self.period + 1 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let downside_penetration = Self::calculate_downside_penetration(low);
        let upside_penetration = Self::calculate_upside_penetration(high);

        let mut long_stop = vec![f64::NAN; n];
        let mut short_stop = vec![f64::NAN; n];

        for i in self.period..n {
            // Calculate average downside penetration for long stop
            let avg_dp = Self::calculate_average_penetration(&downside_penetration, self.period, i);
            // Long stop is below the current low
            let stop_distance_long = self.coefficient * avg_dp;
            long_stop[i] = low[i] - stop_distance_long;

            // Calculate average upside penetration for short stop
            let avg_up = Self::calculate_average_penetration(&upside_penetration, self.period, i);
            // Short stop is above the current high
            let stop_distance_short = self.coefficient * avg_up;
            short_stop[i] = high[i] + stop_distance_short;
        }

        (long_stop, short_stop)
    }

    /// Calculate trailing long stop (ratchet mechanism).
    ///
    /// The stop can only move up, never down.
    pub fn calculate_trailing_long_stop(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let (long_stop, _) = self.calculate(high, low);
        let n = long_stop.len();
        let mut trailing = vec![f64::NAN; n];

        let mut max_stop = f64::NEG_INFINITY;

        for i in 0..n {
            if !long_stop[i].is_nan() {
                if long_stop[i] > max_stop {
                    max_stop = long_stop[i];
                }
                trailing[i] = max_stop;
            }
        }

        trailing
    }

    /// Calculate trailing short stop (ratchet mechanism).
    ///
    /// The stop can only move down, never up.
    pub fn calculate_trailing_short_stop(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let (_, short_stop) = self.calculate(high, low);
        let n = short_stop.len();
        let mut trailing = vec![f64::NAN; n];

        let mut min_stop = f64::INFINITY;

        for i in 0..n {
            if !short_stop[i].is_nan() {
                if short_stop[i] < min_stop {
                    min_stop = short_stop[i];
                }
                trailing[i] = min_stop;
            }
        }

        trailing
    }

    /// Get the current stop distance for long positions.
    pub fn long_stop_distance(&self, _high: &[f64], low: &[f64]) -> Option<f64> {
        let n = low.len();
        if n < self.period + 1 {
            return None;
        }

        let downside_penetration = Self::calculate_downside_penetration(low);
        let avg_dp = Self::calculate_average_penetration(&downside_penetration, self.period, n - 1);
        Some(self.coefficient * avg_dp)
    }

    /// Get the current stop distance for short positions.
    pub fn short_stop_distance(&self, high: &[f64], _low: &[f64]) -> Option<f64> {
        let n = high.len();
        if n < self.period + 1 {
            return None;
        }

        let upside_penetration = Self::calculate_upside_penetration(high);
        let avg_up = Self::calculate_average_penetration(&upside_penetration, self.period, n - 1);
        Some(self.coefficient * avg_up)
    }
}

impl TechnicalIndicator for SafeZoneStop {
    fn name(&self) -> &str {
        "SafeZoneStop"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.high.len(),
            });
        }

        let (long_stop, short_stop) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(long_stop, short_stop))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for SafeZoneStop {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (long_stop, short_stop) = self.calculate(&data.high, &data.low);
        let n = data.close.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let last_close = data.close[n - 1];
        let last_long_stop = long_stop.get(n - 1).copied().unwrap_or(f64::NAN);
        let last_short_stop = short_stop.get(n - 1).copied().unwrap_or(f64::NAN);

        if last_long_stop.is_nan() || last_short_stop.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Price above long stop and below short stop: neutral/continue trend
        // Price below long stop: bearish (stop triggered)
        // Price above short stop: bullish (stop triggered)
        if last_close < last_long_stop {
            Ok(IndicatorSignal::Bearish)
        } else if last_close > last_short_stop {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (long_stop, short_stop) = self.calculate(&data.high, &data.low);
        let n = data.close.len();

        let signals = (0..n)
            .map(|i| {
                let close = data.close[i];
                let ls = long_stop.get(i).copied().unwrap_or(f64::NAN);
                let ss = short_stop.get(i).copied().unwrap_or(f64::NAN);

                if ls.is_nan() || ss.is_nan() {
                    IndicatorSignal::Neutral
                } else if close < ls {
                    IndicatorSignal::Bearish
                } else if close > ss {
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

    fn create_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create trending data with some noise
        let high: Vec<f64> = (0..30)
            .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin() * 2.0)
            .collect();
        let low: Vec<f64> = high.iter().map(|h| h - 2.0).collect();
        let close: Vec<f64> = high.iter().zip(low.iter()).map(|(h, l)| (h + l) / 2.0).collect();
        (high, low, close)
    }

    #[test]
    fn test_safezone_stop_basic() {
        let szs = SafeZoneStop::new(10, 2.0);
        let (high, low, _) = create_test_data();

        let (long_stop, short_stop) = szs.calculate(&high, &low);

        assert_eq!(long_stop.len(), 30);
        assert_eq!(short_stop.len(), 30);

        // First 10 values should be NaN
        for i in 0..10 {
            assert!(long_stop[i].is_nan());
            assert!(short_stop[i].is_nan());
        }

        // Values after warmup should be valid
        for i in 10..30 {
            assert!(!long_stop[i].is_nan(), "long_stop[{}] is NaN", i);
            assert!(!short_stop[i].is_nan(), "short_stop[{}] is NaN", i);
            // Long stop should be at or below the low (with penetration it's below)
            assert!(long_stop[i] <= low[i], "long_stop[{}] ({}) > low[{}] ({})", i, long_stop[i], i, low[i]);
            // Short stop should be at or above high
            assert!(short_stop[i] >= high[i], "short_stop[{}] ({}) < high[{}] ({})", i, short_stop[i], i, high[i]);
        }
    }

    #[test]
    fn test_safezone_elder_default() {
        let szs = SafeZoneStop::elder_default();
        assert_eq!(szs.period, 22);
        assert!((szs.coefficient - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_safezone_short_term() {
        let szs = SafeZoneStop::short_term();
        assert_eq!(szs.period, 10);
        assert!((szs.coefficient - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_downside_penetration() {
        let low = vec![100.0, 99.0, 98.5, 99.5, 97.0];
        let dp = SafeZoneStop::calculate_downside_penetration(&low);

        assert_eq!(dp.len(), 5);
        assert!((dp[0] - 0.0).abs() < f64::EPSILON); // First value is always 0
        assert!((dp[1] - 1.0).abs() < f64::EPSILON); // 100 - 99 = 1
        assert!((dp[2] - 0.5).abs() < f64::EPSILON); // 99 - 98.5 = 0.5
        assert!((dp[3] - 0.0).abs() < f64::EPSILON); // 98.5 - 99.5 = -1 (not penetration)
        assert!((dp[4] - 2.5).abs() < f64::EPSILON); // 99.5 - 97 = 2.5
    }

    #[test]
    fn test_upside_penetration() {
        let high = vec![100.0, 101.0, 100.5, 102.0, 101.5];
        let up = SafeZoneStop::calculate_upside_penetration(&high);

        assert_eq!(up.len(), 5);
        assert!((up[0] - 0.0).abs() < f64::EPSILON); // First value is always 0
        assert!((up[1] - 1.0).abs() < f64::EPSILON); // 101 - 100 = 1
        assert!((up[2] - 0.0).abs() < f64::EPSILON); // 100.5 - 101 = -0.5 (not penetration)
        assert!((up[3] - 1.5).abs() < f64::EPSILON); // 102 - 100.5 = 1.5
        assert!((up[4] - 0.0).abs() < f64::EPSILON); // 101.5 - 102 = -0.5 (not penetration)
    }

    #[test]
    fn test_trailing_long_stop() {
        let szs = SafeZoneStop::new(10, 2.0);
        let (high, low, _) = create_test_data();

        let trailing = szs.calculate_trailing_long_stop(&high, &low);

        assert_eq!(trailing.len(), 30);

        // Verify trailing stop only moves up (or stays the same)
        let mut prev_valid = f64::NEG_INFINITY;
        for i in 10..30 {
            if !trailing[i].is_nan() {
                assert!(trailing[i] >= prev_valid, "Trailing stop moved down at index {}", i);
                prev_valid = trailing[i];
            }
        }
    }

    #[test]
    fn test_trailing_short_stop() {
        let szs = SafeZoneStop::new(10, 2.0);
        let (high, low, _) = create_test_data();

        let trailing = szs.calculate_trailing_short_stop(&high, &low);

        assert_eq!(trailing.len(), 30);

        // Verify trailing stop only moves down (or stays the same)
        let mut prev_valid = f64::INFINITY;
        for i in 10..30 {
            if !trailing[i].is_nan() {
                assert!(trailing[i] <= prev_valid, "Trailing stop moved up at index {}", i);
                prev_valid = trailing[i];
            }
        }
    }

    #[test]
    fn test_stop_distances() {
        let szs = SafeZoneStop::new(10, 2.0);
        let (high, low, _) = create_test_data();

        let long_distance = szs.long_stop_distance(&high, &low);
        let short_distance = szs.short_stop_distance(&high, &low);

        assert!(long_distance.is_some());
        assert!(short_distance.is_some());
        assert!(long_distance.unwrap() >= 0.0);
        assert!(short_distance.unwrap() >= 0.0);
    }

    #[test]
    fn test_technical_indicator_trait() {
        let szs = SafeZoneStop::new(10, 2.0);

        let mut data = OHLCVSeries::new();
        for i in 0..30 {
            let base = 100.0 + i as f64 * 0.5;
            data.open.push(base);
            data.high.push(base + 2.0);
            data.low.push(base - 2.0);
            data.close.push(base + 0.5);
            data.volume.push(1000.0);
        }

        let output = szs.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert_eq!(output.secondary.as_ref().unwrap().len(), 30);
    }

    #[test]
    fn test_signal_indicator_trait() {
        let szs = SafeZoneStop::new(10, 2.0);

        let mut data = OHLCVSeries::new();
        for i in 0..30 {
            let base = 100.0 + i as f64 * 0.5;
            data.open.push(base);
            data.high.push(base + 2.0);
            data.low.push(base - 2.0);
            data.close.push(base + 0.5);
            data.volume.push(1000.0);
        }

        let signal = szs.signal(&data).unwrap();
        assert!(matches!(signal, IndicatorSignal::Neutral | IndicatorSignal::Bullish | IndicatorSignal::Bearish));

        let signals = szs.signals(&data).unwrap();
        assert_eq!(signals.len(), 30);
    }

    #[test]
    fn test_insufficient_data() {
        let szs = SafeZoneStop::new(10, 2.0);

        let mut data = OHLCVSeries::new();
        for i in 0..5 {
            data.open.push(100.0 + i as f64);
            data.high.push(102.0 + i as f64);
            data.low.push(98.0 + i as f64);
            data.close.push(101.0 + i as f64);
            data.volume.push(1000.0);
        }

        let result = szs.compute(&data);
        assert!(result.is_err());
        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 11);
            assert_eq!(got, 5);
        }
    }

    #[test]
    fn test_min_periods() {
        let szs = SafeZoneStop::new(15, 2.5);
        assert_eq!(szs.min_periods(), 16);
    }

    #[test]
    fn test_coefficient_effect() {
        let (high, low, _) = create_test_data();

        let szs_low = SafeZoneStop::new(10, 1.5);
        let szs_high = SafeZoneStop::new(10, 3.0);

        let (long_stop_low, _) = szs_low.calculate(&high, &low);
        let (long_stop_high, _) = szs_high.calculate(&high, &low);

        // Higher coefficient should result in wider stops (lower long stop)
        // This is true when there is actual penetration (non-zero average)
        let mut found_difference = false;
        for i in 10..30 {
            if !long_stop_low[i].is_nan() && !long_stop_high[i].is_nan() {
                // When there's no penetration, both will equal low[i]
                // When there is penetration, higher coefficient gives lower stop
                assert!(long_stop_high[i] <= long_stop_low[i],
                    "Higher coefficient should give lower or equal long stop at index {}: high={}, low={}",
                    i, long_stop_high[i], long_stop_low[i]);
                if (long_stop_high[i] - long_stop_low[i]).abs() > f64::EPSILON {
                    found_difference = true;
                }
            }
        }
        // With our test data, we should see some differences
        assert!(found_difference, "Expected at least some difference between stop levels");
    }

    #[test]
    fn test_no_penetration_scenario() {
        let szs = SafeZoneStop::new(5, 2.0);

        // Flat market with no penetration
        let high = vec![100.0; 10];
        let low = vec![99.0; 10];

        let (long_stop, short_stop) = szs.calculate(&high, &low);

        // With no penetration, stops should be at the price level
        for i in 5..10 {
            assert!(!long_stop[i].is_nan());
            assert!(!short_stop[i].is_nan());
            // With zero average penetration, stop distance is zero
            assert!((long_stop[i] - low[i]).abs() < f64::EPSILON);
            assert!((short_stop[i] - high[i]).abs() < f64::EPSILON);
        }
    }
}
