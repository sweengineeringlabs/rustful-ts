//! Tweezer Top and Tweezer Bottom Candlestick Patterns
//!
//! Identifies two-candle reversal patterns with matching highs or lows.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of tweezer pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TweezerType {
    /// Tweezer Top - bearish reversal at resistance
    Top,
    /// Tweezer Bottom - bullish reversal at support
    Bottom,
}

/// Tweezer Top and Bottom pattern indicator.
///
/// Tweezer Bottom (bullish):
/// - Two consecutive candles with matching (or nearly matching) lows
/// - First candle is bearish, second is bullish
/// - Indicates potential support level and reversal
///
/// Tweezer Top (bearish):
/// - Two consecutive candles with matching (or nearly matching) highs
/// - First candle is bullish, second is bearish
/// - Indicates potential resistance level and reversal
#[derive(Debug, Clone)]
pub struct Tweezer {
    /// Tolerance for matching highs/lows as percentage (default: 0.1%)
    tolerance: f64,
    /// Number of bars to determine prior trend (default: 5)
    trend_period: usize,
}

impl Tweezer {
    /// Create a new Tweezer indicator with default parameters.
    pub fn new() -> Self {
        Self {
            tolerance: 0.001,
            trend_period: 5,
        }
    }

    /// Create a Tweezer indicator with custom parameters.
    ///
    /// # Arguments
    /// * `tolerance` - Tolerance for price matching (e.g., 0.001 = 0.1%)
    /// * `trend_period` - Number of bars to determine prior trend
    pub fn with_params(tolerance: f64, trend_period: usize) -> Self {
        Self {
            tolerance,
            trend_period,
        }
    }

    /// Check if two prices are approximately equal within tolerance.
    fn prices_match(&self, price1: f64, price2: f64) -> bool {
        if price1 <= 0.0 || price2 <= 0.0 {
            return false;
        }
        let diff = (price1 - price2).abs();
        let avg = (price1 + price2) / 2.0;
        diff / avg <= self.tolerance
    }

    /// Determine prior trend direction.
    fn prior_trend(&self, close: &[f64], idx: usize) -> Option<bool> {
        if idx < self.trend_period {
            return None;
        }

        let start = idx - self.trend_period;
        let trend_close = close[start];
        let current_close = close[idx - 1];

        if current_close > trend_close * 1.01 {
            Some(true) // Uptrend
        } else if current_close < trend_close * 0.99 {
            Some(false) // Downtrend
        } else {
            None // No clear trend
        }
    }

    /// Detect tweezer pattern between two consecutive candles.
    fn detect_pattern(
        &self,
        first_open: f64, first_high: f64, first_low: f64, first_close: f64,
        second_open: f64, second_high: f64, second_low: f64, second_close: f64,
        prior_uptrend: Option<bool>,
    ) -> Option<TweezerType> {
        let first_is_bullish = first_close > first_open;
        let second_is_bullish = second_close > second_open;

        // Tweezer Bottom: Matching lows, first bearish, second bullish
        if self.prices_match(first_low, second_low) {
            if !first_is_bullish && second_is_bullish {
                // Ideally appears after downtrend
                if prior_uptrend != Some(true) {
                    return Some(TweezerType::Bottom);
                }
            }
        }

        // Tweezer Top: Matching highs, first bullish, second bearish
        if self.prices_match(first_high, second_high) {
            if first_is_bullish && !second_is_bullish {
                // Ideally appears after uptrend
                if prior_uptrend != Some(false) {
                    return Some(TweezerType::Top);
                }
            }
        }

        None
    }

    /// Calculate tweezer patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Tweezer Bottom (bullish)
    /// - -1.0: Tweezer Top (bearish)
    /// - NaN: No pattern
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < 2 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in 1..n {
            let prior_trend = self.prior_trend(close, i);
            if let Some(pattern) = self.detect_pattern(
                open[i - 1], high[i - 1], low[i - 1], close[i - 1],
                open[i], high[i], low[i], close[i],
                prior_trend,
            ) {
                result[i] = match pattern {
                    TweezerType::Bottom => 1.0,
                    TweezerType::Top => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<TweezerType>> {
        let n = close.len();
        if n < 2 {
            return vec![None; n];
        }

        let mut result = vec![None; n];

        for i in 1..n {
            let prior_trend = self.prior_trend(close, i);
            result[i] = self.detect_pattern(
                open[i - 1], high[i - 1], low[i - 1], close[i - 1],
                open[i], high[i], low[i], close[i],
                prior_trend,
            );
        }

        result
    }
}

impl Default for Tweezer {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for Tweezer {
    fn name(&self) -> &str {
        "Tweezer"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 2 {
            return Err(IndicatorError::InsufficientData {
                required: 2,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        2
    }
}

impl SignalIndicator for Tweezer {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = data.close.len();
        let prior_trend = self.prior_trend(&data.close, n - 1);

        if let Some(pattern) = self.detect_pattern(
            data.open[n - 2], data.high[n - 2], data.low[n - 2], data.close[n - 2],
            data.open[n - 1], data.high[n - 1], data.low[n - 1], data.close[n - 1],
            prior_trend,
        ) {
            return match pattern {
                TweezerType::Bottom => Ok(IndicatorSignal::Bullish),
                TweezerType::Top => Ok(IndicatorSignal::Bearish),
            };
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        let signals = values.iter().map(|&v| {
            if v.is_nan() {
                IndicatorSignal::Neutral
            } else if v > 0.0 {
                IndicatorSignal::Bullish
            } else {
                IndicatorSignal::Bearish
            }
        }).collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tweezer_bottom() {
        let tweezer = Tweezer::with_params(0.002, 5);
        // Matching lows, first bearish, second bullish
        let pattern = tweezer.detect_pattern(
            105.0, 106.0, 100.0, 101.0,  // Bearish with low at 100
            101.0, 104.0, 100.0, 103.0,  // Bullish with low at 100
            Some(false), // After downtrend
        );
        assert_eq!(pattern, Some(TweezerType::Bottom));
    }

    #[test]
    fn test_tweezer_top() {
        let tweezer = Tweezer::with_params(0.002, 5);
        // Matching highs, first bullish, second bearish
        let pattern = tweezer.detect_pattern(
            100.0, 110.0, 99.0, 109.0,   // Bullish with high at 110
            109.0, 110.0, 102.0, 103.0,  // Bearish with high at 110
            Some(true), // After uptrend
        );
        assert_eq!(pattern, Some(TweezerType::Top));
    }

    #[test]
    fn test_no_tweezer_different_levels() {
        let tweezer = Tweezer::new();
        // Lows don't match
        let pattern = tweezer.detect_pattern(
            105.0, 106.0, 100.0, 101.0,
            101.0, 104.0, 98.0, 103.0,   // Different low
            Some(false),
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_tweezer_wrong_direction() {
        let tweezer = Tweezer::with_params(0.002, 5);
        // Matching lows but both bullish
        let pattern = tweezer.detect_pattern(
            100.0, 106.0, 99.0, 105.0,   // Bullish
            101.0, 104.0, 99.0, 103.0,   // Bullish
            Some(false),
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_prices_match() {
        let tweezer = Tweezer::new();
        assert!(tweezer.prices_match(100.0, 100.05));
        assert!(!tweezer.prices_match(100.0, 101.0));
    }

    #[test]
    fn test_tweezer_series() {
        let tweezer = Tweezer::with_params(0.01, 3);
        let open = vec![110.0, 108.0, 106.0, 105.0, 101.0];
        let high = vec![111.0, 109.0, 107.0, 106.0, 104.0];
        let low = vec![108.0, 106.0, 104.0, 100.0, 100.0];
        let close = vec![109.0, 107.0, 105.0, 101.0, 103.0];

        let result = tweezer.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 5);
        // Last bar should be tweezer bottom (matching lows, bearish then bullish)
        assert_eq!(result[4], 1.0);
    }
}
