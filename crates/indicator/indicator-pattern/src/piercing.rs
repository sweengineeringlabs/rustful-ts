//! Piercing Line and Dark Cloud Cover Candlestick Patterns
//!
//! Identifies two-candle reversal patterns.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of piercing/dark cloud pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiercingType {
    /// Piercing Line - bullish reversal pattern
    PiercingLine,
    /// Dark Cloud Cover - bearish reversal pattern
    DarkCloudCover,
}

/// Piercing Line and Dark Cloud Cover pattern indicator.
///
/// Piercing Line (bullish):
/// - First candle: Large bearish candle
/// - Second candle: Opens below first candle's low, closes above midpoint of first candle
///
/// Dark Cloud Cover (bearish):
/// - First candle: Large bullish candle
/// - Second candle: Opens above first candle's high, closes below midpoint of first candle
#[derive(Debug, Clone)]
pub struct Piercing {
    /// Minimum body to range ratio for candles (default: 0.5)
    min_body_ratio: f64,
    /// Minimum penetration ratio (how far into first candle body) (default: 0.5)
    min_penetration: f64,
}

impl Piercing {
    /// Create a new Piercing indicator with default parameters.
    pub fn new() -> Self {
        Self {
            min_body_ratio: 0.5,
            min_penetration: 0.5,
        }
    }

    /// Create a Piercing indicator with custom parameters.
    ///
    /// # Arguments
    /// * `min_body_ratio` - Minimum body/range ratio for candles
    /// * `min_penetration` - Minimum penetration into first candle body (0.5 = 50%)
    pub fn with_params(min_body_ratio: f64, min_penetration: f64) -> Self {
        Self {
            min_body_ratio,
            min_penetration,
        }
    }

    /// Check if a candle has a sufficiently large body.
    fn is_large_body(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        let range = high - low;
        if range <= 0.0 {
            return false;
        }
        let body = (close - open).abs();
        body / range >= self.min_body_ratio
    }

    /// Detect piercing/dark cloud pattern between two consecutive candles.
    fn detect_pattern(
        &self,
        first_open: f64, first_high: f64, first_low: f64, first_close: f64,
        second_open: f64, second_high: f64, second_low: f64, second_close: f64,
    ) -> Option<PiercingType> {
        // Both candles need sufficient body size
        if !self.is_large_body(first_open, first_high, first_low, first_close)
            || !self.is_large_body(second_open, second_high, second_low, second_close)
        {
            return None;
        }

        let first_is_bullish = first_close > first_open;
        let second_is_bullish = second_close > second_open;
        let _first_midpoint = (first_open + first_close) / 2.0;

        // Piercing Line: First bearish, Second bullish
        if !first_is_bullish && second_is_bullish {
            // Second opens below first's low (gap down)
            if second_open <= first_low {
                // Calculate penetration into first candle's body
                let first_body_low = first_close;
                let first_body_high = first_open;
                let body_range = first_body_high - first_body_low;
                let penetration = second_close - first_body_low;

                // Must penetrate at least min_penetration into the body
                if body_range > 0.0 && penetration / body_range >= self.min_penetration {
                    // But not close above first candle's open (would be engulfing)
                    if second_close < first_open {
                        return Some(PiercingType::PiercingLine);
                    }
                }
            }
        }

        // Dark Cloud Cover: First bullish, Second bearish
        if first_is_bullish && !second_is_bullish {
            // Second opens above first's high (gap up)
            if second_open >= first_high {
                // Calculate penetration into first candle's body
                let first_body_low = first_open;
                let first_body_high = first_close;
                let body_range = first_body_high - first_body_low;
                let penetration = first_body_high - second_close;

                // Must penetrate at least min_penetration into the body
                if body_range > 0.0 && penetration / body_range >= self.min_penetration {
                    // But not close below first candle's open (would be engulfing)
                    if second_close > first_open {
                        return Some(PiercingType::DarkCloudCover);
                    }
                }
            }
        }

        None
    }

    /// Calculate piercing patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Piercing Line (bullish)
    /// - -1.0: Dark Cloud Cover (bearish)
    /// - NaN: No pattern
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < 2 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in 1..n {
            if let Some(pattern) = self.detect_pattern(
                open[i - 1], high[i - 1], low[i - 1], close[i - 1],
                open[i], high[i], low[i], close[i],
            ) {
                result[i] = match pattern {
                    PiercingType::PiercingLine => 1.0,
                    PiercingType::DarkCloudCover => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<PiercingType>> {
        let n = close.len();
        if n < 2 {
            return vec![None; n];
        }

        let mut result = vec![None; n];

        for i in 1..n {
            result[i] = self.detect_pattern(
                open[i - 1], high[i - 1], low[i - 1], close[i - 1],
                open[i], high[i], low[i], close[i],
            );
        }

        result
    }
}

impl Default for Piercing {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for Piercing {
    fn name(&self) -> &str {
        "Piercing"
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

impl SignalIndicator for Piercing {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = data.close.len();
        if let Some(pattern) = self.detect_pattern(
            data.open[n - 2], data.high[n - 2], data.low[n - 2], data.close[n - 2],
            data.open[n - 1], data.high[n - 1], data.low[n - 1], data.close[n - 1],
        ) {
            return match pattern {
                PiercingType::PiercingLine => Ok(IndicatorSignal::Bullish),
                PiercingType::DarkCloudCover => Ok(IndicatorSignal::Bearish),
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
    fn test_piercing_line() {
        let piercing = Piercing::new();
        // First: Large bearish candle (open=110, close=100)
        // Second: Opens below first's low, closes above midpoint (105)
        let pattern = piercing.detect_pattern(
            110.0, 112.0, 98.0, 100.0,  // Bearish candle
            97.0, 108.0, 96.0, 107.0,   // Opens at 97 (below 98), closes at 107 (above 105)
        );
        assert_eq!(pattern, Some(PiercingType::PiercingLine));
    }

    #[test]
    fn test_dark_cloud_cover() {
        let piercing = Piercing::new();
        // First: Large bullish candle (open=100, close=110)
        // Second: Opens above first's high, closes below midpoint (105)
        let pattern = piercing.detect_pattern(
            100.0, 112.0, 98.0, 110.0,  // Bullish candle
            113.0, 114.0, 102.0, 103.0, // Opens at 113 (above 112), closes at 103 (below 105)
        );
        assert_eq!(pattern, Some(PiercingType::DarkCloudCover));
    }

    #[test]
    fn test_no_pattern_insufficient_penetration() {
        let piercing = Piercing::new();
        // Second candle doesn't penetrate enough
        let pattern = piercing.detect_pattern(
            110.0, 112.0, 98.0, 100.0,
            97.0, 103.0, 96.0, 102.0,   // Closes at 102, below midpoint
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_pattern_no_gap() {
        let piercing = Piercing::new();
        // Second candle doesn't gap below first's low
        let pattern = piercing.detect_pattern(
            110.0, 112.0, 98.0, 100.0,
            100.0, 108.0, 99.0, 107.0,  // Opens at 100, not below 98
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_piercing_series() {
        let piercing = Piercing::new();
        let open = vec![110.0, 97.0];
        let high = vec![112.0, 108.0];
        let low = vec![98.0, 96.0];
        let close = vec![100.0, 107.0];

        let result = piercing.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 2);
        assert!(result[0].is_nan());
        assert_eq!(result[1], 1.0);
    }
}
