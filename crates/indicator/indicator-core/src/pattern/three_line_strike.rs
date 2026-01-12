//! Three Line Strike Candlestick Pattern
//!
//! Identifies four-candle continuation/reversal patterns.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of three line strike pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreeLineStrikeType {
    /// Bullish Three Line Strike - continuation pattern
    Bullish,
    /// Bearish Three Line Strike - continuation pattern
    Bearish,
}

/// Three Line Strike pattern indicator.
///
/// The Three Line Strike is a four-candle pattern:
///
/// Bullish Three Line Strike (bullish continuation):
/// - Three consecutive bullish candles (like three white soldiers)
/// - Fourth candle: Large bearish candle that engulfs all three
/// - The fourth candle "resets" the pattern, continuation expected
///
/// Bearish Three Line Strike (bearish continuation):
/// - Three consecutive bearish candles (like three black crows)
/// - Fourth candle: Large bullish candle that engulfs all three
/// - The fourth candle "resets" the pattern, continuation expected
///
/// Note: Despite appearing as reversal signals, these patterns are
/// statistically more reliable as continuation patterns.
#[derive(Debug, Clone)]
pub struct ThreeLineStrike {
    /// Minimum body ratio for the three consistent candles (default: 0.5)
    min_body_ratio: f64,
}

impl ThreeLineStrike {
    /// Create a new ThreeLineStrike indicator with default parameters.
    pub fn new() -> Self {
        Self {
            min_body_ratio: 0.5,
        }
    }

    /// Create a ThreeLineStrike indicator with custom parameters.
    pub fn with_params(min_body_ratio: f64) -> Self {
        Self {
            min_body_ratio,
        }
    }

    /// Check if a candle has a valid body relative to its range.
    fn is_valid_body(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        let range = high - low;
        if range <= 0.0 {
            return false;
        }
        let body = (close - open).abs();
        body / range >= self.min_body_ratio
    }

    /// Check if candle is bullish with decent body.
    fn is_bullish_body(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        close > open && self.is_valid_body(open, high, low, close)
    }

    /// Check if candle is bearish with decent body.
    fn is_bearish_body(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        close < open && self.is_valid_body(open, high, low, close)
    }

    /// Detect three line strike pattern at index (requires 4 candles ending at this index).
    #[allow(clippy::too_many_arguments)]
    fn detect_pattern(
        &self,
        o1: f64, h1: f64, l1: f64, c1: f64,
        o2: f64, h2: f64, l2: f64, c2: f64,
        o3: f64, h3: f64, l3: f64, c3: f64,
        o4: f64, _h4: f64, _l4: f64, c4: f64,
    ) -> Option<ThreeLineStrikeType> {
        // Bullish Three Line Strike
        if self.is_bullish_body(o1, h1, l1, c1)
            && self.is_bullish_body(o2, h2, l2, c2)
            && self.is_bullish_body(o3, h3, l3, c3)
        {
            // Progressive closes (ascending)
            if c2 > c1 && c3 > c2 {
                // Fourth candle: bearish that engulfs all three (opens above c3, closes below o1)
                let fourth_is_bearish = c4 < o4;
                if fourth_is_bearish && o4 >= c3 && c4 <= o1 {
                    return Some(ThreeLineStrikeType::Bullish);
                }
            }
        }

        // Bearish Three Line Strike
        if self.is_bearish_body(o1, h1, l1, c1)
            && self.is_bearish_body(o2, h2, l2, c2)
            && self.is_bearish_body(o3, h3, l3, c3)
        {
            // Progressive closes (descending)
            if c2 < c1 && c3 < c2 {
                // Fourth candle: bullish that engulfs all three (opens below c3, closes above o1)
                let fourth_is_bullish = c4 > o4;
                if fourth_is_bullish && o4 <= c3 && c4 >= o1 {
                    return Some(ThreeLineStrikeType::Bearish);
                }
            }
        }

        None
    }

    /// Calculate three line strike patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Bullish Three Line Strike
    /// - -1.0: Bearish Three Line Strike
    /// - NaN: No pattern
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < 4 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in 3..n {
            if let Some(pattern) = self.detect_pattern(
                open[i - 3], high[i - 3], low[i - 3], close[i - 3],
                open[i - 2], high[i - 2], low[i - 2], close[i - 2],
                open[i - 1], high[i - 1], low[i - 1], close[i - 1],
                open[i], high[i], low[i], close[i],
            ) {
                result[i] = match pattern {
                    ThreeLineStrikeType::Bullish => 1.0,
                    ThreeLineStrikeType::Bearish => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<ThreeLineStrikeType>> {
        let n = close.len();
        if n < 4 {
            return vec![None; n];
        }

        let mut result = vec![None; n];

        for i in 3..n {
            result[i] = self.detect_pattern(
                open[i - 3], high[i - 3], low[i - 3], close[i - 3],
                open[i - 2], high[i - 2], low[i - 2], close[i - 2],
                open[i - 1], high[i - 1], low[i - 1], close[i - 1],
                open[i], high[i], low[i], close[i],
            );
        }

        result
    }
}

impl Default for ThreeLineStrike {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for ThreeLineStrike {
    fn name(&self) -> &str {
        "ThreeLineStrike"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 4 {
            return Err(IndicatorError::InsufficientData {
                required: 4,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        4
    }
}

impl SignalIndicator for ThreeLineStrike {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.len() < 4 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = data.close.len();
        if let Some(pattern) = self.detect_pattern(
            data.open[n - 4], data.high[n - 4], data.low[n - 4], data.close[n - 4],
            data.open[n - 3], data.high[n - 3], data.low[n - 3], data.close[n - 3],
            data.open[n - 2], data.high[n - 2], data.low[n - 2], data.close[n - 2],
            data.open[n - 1], data.high[n - 1], data.low[n - 1], data.close[n - 1],
        ) {
            return match pattern {
                ThreeLineStrikeType::Bullish => Ok(IndicatorSignal::Bullish),
                ThreeLineStrikeType::Bearish => Ok(IndicatorSignal::Bearish),
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
    fn test_bullish_three_line_strike() {
        let tls = ThreeLineStrike::new();
        // Three bullish candles followed by large bearish engulfing
        let pattern = tls.detect_pattern(
            100.0, 107.0, 99.0, 106.0,   // Bullish 1
            105.0, 112.0, 104.0, 111.0,  // Bullish 2
            110.0, 117.0, 109.0, 116.0,  // Bullish 3
            117.0, 118.0, 98.0, 99.0,    // Bearish engulfing all (opens at 117 >= 116, closes at 99 <= 100)
        );
        assert_eq!(pattern, Some(ThreeLineStrikeType::Bullish));
    }

    #[test]
    fn test_bearish_three_line_strike() {
        let tls = ThreeLineStrike::new();
        // Three bearish candles followed by large bullish engulfing
        let pattern = tls.detect_pattern(
            116.0, 117.0, 109.0, 110.0,  // Bearish 1
            111.0, 112.0, 104.0, 105.0,  // Bearish 2
            106.0, 107.0, 99.0, 100.0,   // Bearish 3
            99.0, 118.0, 98.0, 117.0,    // Bullish engulfing all (opens at 99 <= 100, closes at 117 >= 116)
        );
        assert_eq!(pattern, Some(ThreeLineStrikeType::Bearish));
    }

    #[test]
    fn test_no_pattern_not_progressive() {
        let tls = ThreeLineStrike::new();
        // Three bullish but not progressive
        let pattern = tls.detect_pattern(
            100.0, 107.0, 99.0, 106.0,
            105.0, 112.0, 104.0, 108.0,  // Closes lower than first
            110.0, 117.0, 109.0, 116.0,
            117.0, 118.0, 98.0, 99.0,
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_pattern_fourth_doesnt_engulf() {
        let tls = ThreeLineStrike::new();
        // Fourth candle doesn't fully engulf
        let pattern = tls.detect_pattern(
            100.0, 107.0, 99.0, 106.0,
            105.0, 112.0, 104.0, 111.0,
            110.0, 117.0, 109.0, 116.0,
            115.0, 116.0, 105.0, 106.0,  // Doesn't close below first open
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_three_line_strike_series() {
        let tls = ThreeLineStrike::new();
        let open = vec![100.0, 105.0, 110.0, 117.0];
        let high = vec![107.0, 112.0, 117.0, 118.0];
        let low = vec![99.0, 104.0, 109.0, 98.0];
        let close = vec![106.0, 111.0, 116.0, 99.0];

        let result = tls.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 4);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert_eq!(result[3], 1.0); // Bullish Three Line Strike
    }
}
