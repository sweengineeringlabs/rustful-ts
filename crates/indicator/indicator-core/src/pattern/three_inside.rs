//! Three Inside Up and Three Inside Down Candlestick Patterns
//!
//! Identifies three-candle reversal patterns based on harami confirmation.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of three inside pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreeInsideType {
    /// Three Inside Up - bullish reversal
    Up,
    /// Three Inside Down - bearish reversal
    Down,
}

/// Three Inside Up and Three Inside Down pattern indicator.
///
/// These patterns are confirmations of the harami pattern:
///
/// Three Inside Up (bullish):
/// 1. First candle: Large bearish candle
/// 2. Second candle: Small bullish candle contained within first (bullish harami)
/// 3. Third candle: Bullish candle that closes above first candle's open
///
/// Three Inside Down (bearish):
/// 1. First candle: Large bullish candle
/// 2. Second candle: Small bearish candle contained within first (bearish harami)
/// 3. Third candle: Bearish candle that closes below first candle's open
#[derive(Debug, Clone)]
pub struct ThreeInside {
    /// Minimum body ratio for first candle (default: 0.5)
    min_first_body_ratio: f64,
    /// Maximum body ratio for second candle (default: 0.5)
    max_second_body_ratio: f64,
}

impl ThreeInside {
    /// Create a new ThreeInside indicator with default parameters.
    pub fn new() -> Self {
        Self {
            min_first_body_ratio: 0.5,
            max_second_body_ratio: 0.5,
        }
    }

    /// Create a ThreeInside indicator with custom parameters.
    pub fn with_params(min_first_body_ratio: f64, max_second_body_ratio: f64) -> Self {
        Self {
            min_first_body_ratio,
            max_second_body_ratio,
        }
    }

    /// Check if first candle is a large body candle.
    fn is_large_body(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        let range = high - low;
        if range <= 0.0 {
            return false;
        }
        let body = (close - open).abs();
        body / range >= self.min_first_body_ratio
    }

    /// Check if second candle is contained within first (harami body).
    fn is_inside(&self, first_open: f64, first_close: f64, second_open: f64, second_close: f64) -> bool {
        let first_body_high = first_open.max(first_close);
        let first_body_low = first_open.min(first_close);
        let second_body_high = second_open.max(second_close);
        let second_body_low = second_open.min(second_close);

        // Second body must be completely inside first body
        second_body_high <= first_body_high && second_body_low >= first_body_low
    }

    /// Detect three inside pattern at index (requires 3 candles ending at this index).
    fn detect_pattern(
        &self,
        first_open: f64, first_high: f64, first_low: f64, first_close: f64,
        second_open: f64, _second_high: f64, _second_low: f64, second_close: f64,
        third_open: f64, _third_high: f64, _third_low: f64, third_close: f64,
    ) -> Option<ThreeInsideType> {
        // First candle must be large
        if !self.is_large_body(first_open, first_high, first_low, first_close) {
            return None;
        }

        // Second candle must be inside first (harami)
        if !self.is_inside(first_open, first_close, second_open, second_close) {
            return None;
        }

        // Check second candle body size relative to first
        let first_body = (first_close - first_open).abs();
        let second_body = (second_close - second_open).abs();
        if first_body <= 0.0 || second_body / first_body > self.max_second_body_ratio {
            return None;
        }

        let first_is_bullish = first_close > first_open;
        let second_is_bullish = second_close > second_open;
        let third_is_bullish = third_close > third_open;

        // Three Inside Up: First bearish, second bullish (inside), third bullish (confirms)
        if !first_is_bullish && second_is_bullish && third_is_bullish {
            // Third candle must close above first's open
            if third_close > first_open {
                return Some(ThreeInsideType::Up);
            }
        }

        // Three Inside Down: First bullish, second bearish (inside), third bearish (confirms)
        if first_is_bullish && !second_is_bullish && !third_is_bullish {
            // Third candle must close below first's open
            if third_close < first_open {
                return Some(ThreeInsideType::Down);
            }
        }

        None
    }

    /// Calculate three inside patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Three Inside Up (bullish)
    /// - -1.0: Three Inside Down (bearish)
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
                    ThreeInsideType::Up => 1.0,
                    ThreeInsideType::Down => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<ThreeInsideType>> {
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

impl Default for ThreeInside {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for ThreeInside {
    fn name(&self) -> &str {
        "ThreeInside"
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

impl SignalIndicator for ThreeInside {
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
                ThreeInsideType::Up => Ok(IndicatorSignal::Bullish),
                ThreeInsideType::Down => Ok(IndicatorSignal::Bearish),
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
    fn test_three_inside_up() {
        let ti = ThreeInside::new();
        // First: Large bearish (open=110, close=100)
        // Second: Small bullish inside (open=102, close=106)
        // Third: Bullish closing above first's open (close=112)
        let pattern = ti.detect_pattern(
            110.0, 112.0, 98.0, 100.0,   // Large bearish
            102.0, 107.0, 101.0, 106.0,  // Small bullish inside
            105.0, 114.0, 104.0, 112.0,  // Bullish confirming
        );
        assert_eq!(pattern, Some(ThreeInsideType::Up));
    }

    #[test]
    fn test_three_inside_down() {
        let ti = ThreeInside::new();
        // First: Large bullish (open=100, close=110)
        // Second: Small bearish inside (open=108, close=104)
        // Third: Bearish closing below first's open (close=98)
        let pattern = ti.detect_pattern(
            100.0, 112.0, 98.0, 110.0,   // Large bullish
            108.0, 109.0, 103.0, 104.0,  // Small bearish inside
            105.0, 106.0, 96.0, 98.0,    // Bearish confirming
        );
        assert_eq!(pattern, Some(ThreeInsideType::Down));
    }

    #[test]
    fn test_no_pattern_second_not_inside() {
        let ti = ThreeInside::new();
        // Second candle not fully inside first
        let pattern = ti.detect_pattern(
            110.0, 112.0, 98.0, 100.0,
            99.0, 115.0, 98.0, 114.0,    // Not inside
            105.0, 114.0, 104.0, 112.0,
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_pattern_third_doesnt_confirm() {
        let ti = ThreeInside::new();
        // Third candle doesn't close above first's open
        let pattern = ti.detect_pattern(
            110.0, 112.0, 98.0, 100.0,
            102.0, 107.0, 101.0, 106.0,
            105.0, 108.0, 104.0, 107.0,  // Doesn't close above 110
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_three_inside_series() {
        let ti = ThreeInside::new();
        let open = vec![110.0, 102.0, 105.0];
        let high = vec![112.0, 107.0, 114.0];
        let low = vec![98.0, 101.0, 104.0];
        let close = vec![100.0, 106.0, 112.0];

        let result = ti.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 1.0); // Three Inside Up
    }
}
