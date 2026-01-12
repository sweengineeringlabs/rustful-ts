//! Rising and Falling Three Methods Candlestick Patterns
//!
//! Identifies five-candle continuation patterns.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of rising/falling methods pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RisingFallingMethodsType {
    /// Rising Three Methods - bullish continuation
    Rising,
    /// Falling Three Methods - bearish continuation
    Falling,
}

/// Rising and Falling Three Methods pattern indicator.
///
/// These are five-candle continuation patterns:
///
/// Rising Three Methods (bullish continuation):
/// 1. First candle: Large bullish candle
/// 2-4. Three small bearish candles within the range of the first candle
/// 5. Fifth candle: Large bullish candle that closes above first candle's close
///
/// Falling Three Methods (bearish continuation):
/// 1. First candle: Large bearish candle
/// 2-4. Three small bullish candles within the range of the first candle
/// 5. Fifth candle: Large bearish candle that closes below first candle's close
///
/// The pattern represents a brief consolidation before trend continuation.
#[derive(Debug, Clone)]
pub struct RisingFallingMethods {
    /// Minimum body ratio for large candles (default: 0.6)
    min_large_body_ratio: f64,
    /// Maximum body ratio for small candles (default: 0.5)
    max_small_body_ratio: f64,
}

impl RisingFallingMethods {
    /// Create a new RisingFallingMethods indicator with default parameters.
    pub fn new() -> Self {
        Self {
            min_large_body_ratio: 0.6,
            max_small_body_ratio: 0.5,
        }
    }

    /// Create a RisingFallingMethods indicator with custom parameters.
    pub fn with_params(min_large_body_ratio: f64, max_small_body_ratio: f64) -> Self {
        Self {
            min_large_body_ratio,
            max_small_body_ratio,
        }
    }

    /// Check if a candle has a large body.
    fn is_large_body(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        let range = high - low;
        if range <= 0.0 {
            return false;
        }
        let body = (close - open).abs();
        body / range >= self.min_large_body_ratio
    }

    /// Check if a candle has a small body.
    fn is_small_body(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        let range = high - low;
        if range <= 0.0 {
            return true; // Very small candle
        }
        let body = (close - open).abs();
        body / range <= self.max_small_body_ratio
    }

    /// Check if candle is within the range of another candle.
    fn is_within_range(&self, candle_high: f64, candle_low: f64, range_high: f64, range_low: f64) -> bool {
        candle_high <= range_high && candle_low >= range_low
    }

    /// Detect rising/falling methods pattern at index (requires 5 candles ending at this index).
    #[allow(clippy::too_many_arguments)]
    fn detect_pattern(
        &self,
        o1: f64, h1: f64, l1: f64, c1: f64,
        o2: f64, h2: f64, l2: f64, c2: f64,
        o3: f64, h3: f64, l3: f64, c3: f64,
        o4: f64, h4: f64, l4: f64, c4: f64,
        o5: f64, h5: f64, l5: f64, c5: f64,
    ) -> Option<RisingFallingMethodsType> {
        let first_is_bullish = c1 > o1;
        let fifth_is_bullish = c5 > o5;

        // Rising Three Methods
        if first_is_bullish && fifth_is_bullish {
            // First and fifth must be large bodied
            if !self.is_large_body(o1, h1, l1, c1) || !self.is_large_body(o5, h5, l5, c5) {
                return None;
            }

            // Three middle candles should be small and bearish
            let second_bearish = c2 < o2;
            let third_bearish = c3 < o3;
            let fourth_bearish = c4 < o4;

            if !second_bearish || !third_bearish || !fourth_bearish {
                return None;
            }

            if !self.is_small_body(o2, h2, l2, c2)
                || !self.is_small_body(o3, h3, l3, c3)
                || !self.is_small_body(o4, h4, l4, c4)
            {
                return None;
            }

            // Middle candles must be within first candle's range
            if !self.is_within_range(h2, l2, h1, l1)
                || !self.is_within_range(h3, l3, h1, l1)
                || !self.is_within_range(h4, l4, h1, l1)
            {
                return None;
            }

            // Fifth candle must close above first candle's close
            if c5 > c1 {
                return Some(RisingFallingMethodsType::Rising);
            }
        }

        // Falling Three Methods
        let first_is_bearish = c1 < o1;
        let fifth_is_bearish = c5 < o5;

        if first_is_bearish && fifth_is_bearish {
            // First and fifth must be large bodied
            if !self.is_large_body(o1, h1, l1, c1) || !self.is_large_body(o5, h5, l5, c5) {
                return None;
            }

            // Three middle candles should be small and bullish
            let second_bullish = c2 > o2;
            let third_bullish = c3 > o3;
            let fourth_bullish = c4 > o4;

            if !second_bullish || !third_bullish || !fourth_bullish {
                return None;
            }

            if !self.is_small_body(o2, h2, l2, c2)
                || !self.is_small_body(o3, h3, l3, c3)
                || !self.is_small_body(o4, h4, l4, c4)
            {
                return None;
            }

            // Middle candles must be within first candle's range
            if !self.is_within_range(h2, l2, h1, l1)
                || !self.is_within_range(h3, l3, h1, l1)
                || !self.is_within_range(h4, l4, h1, l1)
            {
                return None;
            }

            // Fifth candle must close below first candle's close
            if c5 < c1 {
                return Some(RisingFallingMethodsType::Falling);
            }
        }

        None
    }

    /// Calculate rising/falling methods patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Rising Three Methods (bullish continuation)
    /// - -1.0: Falling Three Methods (bearish continuation)
    /// - NaN: No pattern
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < 5 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in 4..n {
            if let Some(pattern) = self.detect_pattern(
                open[i - 4], high[i - 4], low[i - 4], close[i - 4],
                open[i - 3], high[i - 3], low[i - 3], close[i - 3],
                open[i - 2], high[i - 2], low[i - 2], close[i - 2],
                open[i - 1], high[i - 1], low[i - 1], close[i - 1],
                open[i], high[i], low[i], close[i],
            ) {
                result[i] = match pattern {
                    RisingFallingMethodsType::Rising => 1.0,
                    RisingFallingMethodsType::Falling => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<RisingFallingMethodsType>> {
        let n = close.len();
        if n < 5 {
            return vec![None; n];
        }

        let mut result = vec![None; n];

        for i in 4..n {
            result[i] = self.detect_pattern(
                open[i - 4], high[i - 4], low[i - 4], close[i - 4],
                open[i - 3], high[i - 3], low[i - 3], close[i - 3],
                open[i - 2], high[i - 2], low[i - 2], close[i - 2],
                open[i - 1], high[i - 1], low[i - 1], close[i - 1],
                open[i], high[i], low[i], close[i],
            );
        }

        result
    }
}

impl Default for RisingFallingMethods {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for RisingFallingMethods {
    fn name(&self) -> &str {
        "RisingFallingMethods"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 5 {
            return Err(IndicatorError::InsufficientData {
                required: 5,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        5
    }
}

impl SignalIndicator for RisingFallingMethods {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.len() < 5 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = data.close.len();
        if let Some(pattern) = self.detect_pattern(
            data.open[n - 5], data.high[n - 5], data.low[n - 5], data.close[n - 5],
            data.open[n - 4], data.high[n - 4], data.low[n - 4], data.close[n - 4],
            data.open[n - 3], data.high[n - 3], data.low[n - 3], data.close[n - 3],
            data.open[n - 2], data.high[n - 2], data.low[n - 2], data.close[n - 2],
            data.open[n - 1], data.high[n - 1], data.low[n - 1], data.close[n - 1],
        ) {
            return match pattern {
                RisingFallingMethodsType::Rising => Ok(IndicatorSignal::Bullish),
                RisingFallingMethodsType::Falling => Ok(IndicatorSignal::Bearish),
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
    fn test_rising_three_methods() {
        let rfm = RisingFallingMethods::new();
        // First: Large bullish
        // 2-4: Small bearish within first's range
        // Fifth: Large bullish closing above first
        let pattern = rfm.detect_pattern(
            100.0, 115.0, 98.0, 112.0,   // Large bullish (range 98-115, close 112)
            110.0, 112.0, 106.0, 107.0,  // Small bearish within range
            108.0, 110.0, 104.0, 105.0,  // Small bearish within range
            106.0, 108.0, 102.0, 103.0,  // Small bearish within range
            102.0, 120.0, 100.0, 118.0,  // Large bullish closing above 112
        );
        assert_eq!(pattern, Some(RisingFallingMethodsType::Rising));
    }

    #[test]
    fn test_falling_three_methods() {
        let rfm = RisingFallingMethods::new();
        // First: Large bearish
        // 2-4: Small bullish within first's range
        // Fifth: Large bearish closing below first
        let pattern = rfm.detect_pattern(
            112.0, 115.0, 98.0, 100.0,   // Large bearish (range 98-115, close 100)
            102.0, 106.0, 100.0, 105.0,  // Small bullish within range
            104.0, 108.0, 102.0, 107.0,  // Small bullish within range
            106.0, 110.0, 104.0, 109.0,  // Small bullish within range
            110.0, 112.0, 95.0, 97.0,    // Large bearish closing below 100
        );
        assert_eq!(pattern, Some(RisingFallingMethodsType::Falling));
    }

    #[test]
    fn test_no_pattern_middle_not_within_range() {
        let rfm = RisingFallingMethods::new();
        // Middle candle extends beyond first's range
        let pattern = rfm.detect_pattern(
            100.0, 115.0, 98.0, 112.0,
            110.0, 120.0, 106.0, 107.0,  // High exceeds first's high
            108.0, 110.0, 104.0, 105.0,
            106.0, 108.0, 102.0, 103.0,
            102.0, 120.0, 100.0, 118.0,
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_pattern_middle_wrong_direction() {
        let rfm = RisingFallingMethods::new();
        // Rising pattern but middle candles are bullish (should be bearish)
        let pattern = rfm.detect_pattern(
            100.0, 115.0, 98.0, 112.0,
            106.0, 112.0, 104.0, 110.0,  // Bullish, not bearish
            104.0, 110.0, 102.0, 108.0,
            102.0, 108.0, 100.0, 106.0,
            102.0, 120.0, 100.0, 118.0,
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_pattern_fifth_doesnt_confirm() {
        let rfm = RisingFallingMethods::new();
        // Fifth candle doesn't close above first's close
        let pattern = rfm.detect_pattern(
            100.0, 115.0, 98.0, 112.0,
            110.0, 112.0, 106.0, 107.0,
            108.0, 110.0, 104.0, 105.0,
            106.0, 108.0, 102.0, 103.0,
            102.0, 112.0, 100.0, 110.0,  // Closes at 110, below 112
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_rising_falling_methods_series() {
        let rfm = RisingFallingMethods::new();
        let open = vec![100.0, 110.0, 108.0, 106.0, 102.0];
        let high = vec![115.0, 112.0, 110.0, 108.0, 120.0];
        let low = vec![98.0, 106.0, 104.0, 102.0, 100.0];
        let close = vec![112.0, 107.0, 105.0, 103.0, 118.0];

        let result = rfm.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert!(result[3].is_nan());
        assert_eq!(result[4], 1.0); // Rising Three Methods
    }
}
