//! Three Outside Up and Three Outside Down Candlestick Patterns
//!
//! Identifies three-candle reversal patterns based on engulfing confirmation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of three outside pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreeOutsideType {
    /// Three Outside Up - bullish reversal
    Up,
    /// Three Outside Down - bearish reversal
    Down,
}

/// Three Outside Up and Three Outside Down pattern indicator.
///
/// These patterns are confirmations of the engulfing pattern:
///
/// Three Outside Up (bullish):
/// 1. First candle: Bearish candle
/// 2. Second candle: Bullish candle that engulfs first
/// 3. Third candle: Bullish candle that closes above second candle's close
///
/// Three Outside Down (bearish):
/// 1. First candle: Bullish candle
/// 2. Second candle: Bearish candle that engulfs first
/// 3. Third candle: Bearish candle that closes below second candle's close
#[derive(Debug, Clone)]
pub struct ThreeOutside {
    /// Minimum engulf ratio (default: 1.0)
    min_engulf_ratio: f64,
}

impl ThreeOutside {
    /// Create a new ThreeOutside indicator with default parameters.
    pub fn new() -> Self {
        Self {
            min_engulf_ratio: 1.0,
        }
    }

    /// Create a ThreeOutside indicator with custom parameters.
    pub fn with_params(min_engulf_ratio: f64) -> Self {
        Self {
            min_engulf_ratio,
        }
    }

    /// Check if second candle engulfs first (bullish engulfing).
    fn is_bullish_engulfing(&self, first_open: f64, first_close: f64, second_open: f64, second_close: f64) -> bool {
        let first_is_bearish = first_close < first_open;
        let second_is_bullish = second_close > second_open;

        if !first_is_bearish || !second_is_bullish {
            return false;
        }

        let first_body = (first_open - first_close).abs();
        let second_body = (second_close - second_open).abs();

        if first_body < f64::EPSILON {
            return false;
        }

        // Second body engulfs first body
        second_open <= first_close && second_close >= first_open
            && second_body >= first_body * self.min_engulf_ratio
    }

    /// Check if second candle engulfs first (bearish engulfing).
    fn is_bearish_engulfing(&self, first_open: f64, first_close: f64, second_open: f64, second_close: f64) -> bool {
        let first_is_bullish = first_close > first_open;
        let second_is_bearish = second_close < second_open;

        if !first_is_bullish || !second_is_bearish {
            return false;
        }

        let first_body = (first_close - first_open).abs();
        let second_body = (second_open - second_close).abs();

        if first_body < f64::EPSILON {
            return false;
        }

        // Second body engulfs first body
        second_open >= first_close && second_close <= first_open
            && second_body >= first_body * self.min_engulf_ratio
    }

    /// Detect three outside pattern at index (requires 3 candles ending at this index).
    fn detect_pattern(
        &self,
        first_open: f64, _first_high: f64, _first_low: f64, first_close: f64,
        second_open: f64, _second_high: f64, _second_low: f64, second_close: f64,
        _third_open: f64, _third_high: f64, _third_low: f64, third_close: f64,
    ) -> Option<ThreeOutsideType> {
        let third_is_bullish = third_close > _third_open;
        let third_is_bearish = third_close < _third_open;

        // Three Outside Up: Bullish engulfing followed by bullish confirmation
        if self.is_bullish_engulfing(first_open, first_close, second_open, second_close) {
            if third_is_bullish && third_close > second_close {
                return Some(ThreeOutsideType::Up);
            }
        }

        // Three Outside Down: Bearish engulfing followed by bearish confirmation
        if self.is_bearish_engulfing(first_open, first_close, second_open, second_close) {
            if third_is_bearish && third_close < second_close {
                return Some(ThreeOutsideType::Down);
            }
        }

        None
    }

    /// Calculate three outside patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Three Outside Up (bullish)
    /// - -1.0: Three Outside Down (bearish)
    /// - NaN: No pattern
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < 3 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in 2..n {
            if let Some(pattern) = self.detect_pattern(
                open[i - 2], high[i - 2], low[i - 2], close[i - 2],
                open[i - 1], high[i - 1], low[i - 1], close[i - 1],
                open[i], high[i], low[i], close[i],
            ) {
                result[i] = match pattern {
                    ThreeOutsideType::Up => 1.0,
                    ThreeOutsideType::Down => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<ThreeOutsideType>> {
        let n = close.len();
        if n < 3 {
            return vec![None; n];
        }

        let mut result = vec![None; n];

        for i in 2..n {
            result[i] = self.detect_pattern(
                open[i - 2], high[i - 2], low[i - 2], close[i - 2],
                open[i - 1], high[i - 1], low[i - 1], close[i - 1],
                open[i], high[i], low[i], close[i],
            );
        }

        result
    }
}

impl Default for ThreeOutside {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for ThreeOutside {
    fn name(&self) -> &str {
        "ThreeOutside"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 3 {
            return Err(IndicatorError::InsufficientData {
                required: 3,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        3
    }
}

impl SignalIndicator for ThreeOutside {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.len() < 3 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = data.close.len();
        if let Some(pattern) = self.detect_pattern(
            data.open[n - 3], data.high[n - 3], data.low[n - 3], data.close[n - 3],
            data.open[n - 2], data.high[n - 2], data.low[n - 2], data.close[n - 2],
            data.open[n - 1], data.high[n - 1], data.low[n - 1], data.close[n - 1],
        ) {
            return match pattern {
                ThreeOutsideType::Up => Ok(IndicatorSignal::Bullish),
                ThreeOutsideType::Down => Ok(IndicatorSignal::Bearish),
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
    fn test_three_outside_up() {
        let to = ThreeOutside::new();
        // First: Bearish (open=105, close=100)
        // Second: Bullish engulfing (open=99, close=107)
        // Third: Bullish closing above second (close=112)
        let pattern = to.detect_pattern(
            105.0, 106.0, 99.0, 100.0,   // Bearish
            99.0, 108.0, 98.0, 107.0,    // Bullish engulfing
            106.0, 114.0, 105.0, 112.0,  // Bullish confirming
        );
        assert_eq!(pattern, Some(ThreeOutsideType::Up));
    }

    #[test]
    fn test_three_outside_down() {
        let to = ThreeOutside::new();
        // First: Bullish (open=100, close=105)
        // Second: Bearish engulfing (open=106, close=98)
        // Third: Bearish closing below second (close=95)
        let pattern = to.detect_pattern(
            100.0, 106.0, 99.0, 105.0,   // Bullish
            106.0, 107.0, 97.0, 98.0,    // Bearish engulfing
            99.0, 100.0, 94.0, 95.0,     // Bearish confirming
        );
        assert_eq!(pattern, Some(ThreeOutsideType::Down));
    }

    #[test]
    fn test_no_pattern_not_engulfing() {
        let to = ThreeOutside::new();
        // Second doesn't engulf first
        let pattern = to.detect_pattern(
            105.0, 106.0, 99.0, 100.0,
            100.0, 103.0, 99.0, 102.0,   // Doesn't engulf
            101.0, 105.0, 100.0, 104.0,
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_pattern_third_doesnt_confirm() {
        let to = ThreeOutside::new();
        // Third candle doesn't close above second's close
        let pattern = to.detect_pattern(
            105.0, 106.0, 99.0, 100.0,
            99.0, 108.0, 98.0, 107.0,
            106.0, 108.0, 105.0, 106.0,  // Doesn't close above 107
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_bullish_engulfing_check() {
        let to = ThreeOutside::new();
        // First bearish (open=105, close=100), second bullish engulfing
        assert!(to.is_bullish_engulfing(105.0, 100.0, 99.0, 107.0));
        // Not engulfing - second doesn't cover first
        assert!(!to.is_bullish_engulfing(105.0, 100.0, 101.0, 104.0));
    }

    #[test]
    fn test_three_outside_series() {
        let to = ThreeOutside::new();
        let open = vec![105.0, 99.0, 106.0];
        let high = vec![106.0, 108.0, 114.0];
        let low = vec![99.0, 98.0, 105.0];
        let close = vec![100.0, 107.0, 112.0];

        let result = to.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 1.0); // Three Outside Up
    }
}
