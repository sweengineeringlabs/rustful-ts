//! Morning Star and Evening Star Candlestick Patterns
//!
//! Identifies three-candle reversal patterns.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of star pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StarType {
    /// Morning Star - bullish reversal at bottom
    MorningStar,
    /// Morning Doji Star - stronger bullish signal with doji middle
    MorningDojiStar,
    /// Evening Star - bearish reversal at top
    EveningStar,
    /// Evening Doji Star - stronger bearish signal with doji middle
    EveningDojiStar,
}

/// Morning Star and Evening Star pattern indicator.
///
/// These are three-candle reversal patterns:
///
/// Morning Star (bullish):
/// 1. First candle: Large bearish candle (confirms downtrend)
/// 2. Second candle: Small body candle (star) that gaps down
/// 3. Third candle: Large bullish candle that closes above first candle's midpoint
///
/// Evening Star (bearish):
/// 1. First candle: Large bullish candle (confirms uptrend)
/// 2. Second candle: Small body candle (star) that gaps up
/// 3. Third candle: Large bearish candle that closes below first candle's midpoint
#[derive(Debug, Clone)]
pub struct MorningStar {
    /// Maximum body ratio for the star candle (default: 0.3)
    star_body_ratio: f64,
    /// Minimum body ratio for first/third candles (default: 0.5)
    large_body_ratio: f64,
    /// Doji threshold for body/range ratio
    doji_threshold: f64,
}

impl MorningStar {
    /// Create a new MorningStar indicator with default parameters.
    pub fn new() -> Self {
        Self {
            star_body_ratio: 0.3,
            large_body_ratio: 0.5,
            doji_threshold: 0.05,
        }
    }

    /// Create a MorningStar indicator with custom parameters.
    pub fn with_params(star_body_ratio: f64, large_body_ratio: f64, doji_threshold: f64) -> Self {
        Self {
            star_body_ratio,
            large_body_ratio,
            doji_threshold,
        }
    }

    /// Check if a candle has a large body relative to its range.
    fn is_large_body(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        let range = high - low;
        if range <= 0.0 {
            return false;
        }
        let body = (close - open).abs();
        body / range >= self.large_body_ratio
    }

    /// Check if a candle is a star (small body).
    fn is_star(&self, open: f64, high: f64, low: f64, close: f64) -> (bool, bool) {
        let range = high - low;
        if range <= 0.0 {
            return (false, false);
        }
        let body = (close - open).abs();
        let body_ratio = body / range;
        let is_star = body_ratio <= self.star_body_ratio;
        let is_doji = body_ratio <= self.doji_threshold;
        (is_star, is_doji)
    }

    /// Detect star pattern at index (requires 3 candles ending at this index).
    fn detect_pattern(
        &self,
        first_open: f64, first_high: f64, first_low: f64, first_close: f64,
        star_open: f64, star_high: f64, star_low: f64, star_close: f64,
        third_open: f64, third_high: f64, third_low: f64, third_close: f64,
    ) -> Option<StarType> {
        // Check first candle is large
        if !self.is_large_body(first_open, first_high, first_low, first_close) {
            return None;
        }

        // Check middle candle is star
        let (is_star, is_doji) = self.is_star(star_open, star_high, star_low, star_close);
        if !is_star {
            return None;
        }

        // Check third candle is large
        if !self.is_large_body(third_open, third_high, third_low, third_close) {
            return None;
        }

        let first_is_bullish = first_close > first_open;
        let third_is_bullish = third_close > third_open;
        let first_midpoint = (first_open + first_close) / 2.0;
        let star_body_center = (star_open + star_close) / 2.0;

        // Morning Star: First bearish, Third bullish, Star gaps down
        if !first_is_bullish && third_is_bullish {
            // Check for gap down (star below first close)
            if star_body_center < first_close && third_close > first_midpoint {
                if is_doji {
                    return Some(StarType::MorningDojiStar);
                }
                return Some(StarType::MorningStar);
            }
        }

        // Evening Star: First bullish, Third bearish, Star gaps up
        if first_is_bullish && !third_is_bullish {
            // Check for gap up (star above first close)
            if star_body_center > first_close && third_close < first_midpoint {
                if is_doji {
                    return Some(StarType::EveningDojiStar);
                }
                return Some(StarType::EveningStar);
            }
        }

        None
    }

    /// Calculate star patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Morning Star
    /// - 2.0: Morning Doji Star
    /// - -1.0: Evening Star
    /// - -2.0: Evening Doji Star
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
                    StarType::MorningStar => 1.0,
                    StarType::MorningDojiStar => 2.0,
                    StarType::EveningStar => -1.0,
                    StarType::EveningDojiStar => -2.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<StarType>> {
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

impl Default for MorningStar {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for MorningStar {
    fn name(&self) -> &str {
        "MorningStar"
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

impl SignalIndicator for MorningStar {
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
                StarType::MorningStar | StarType::MorningDojiStar => Ok(IndicatorSignal::Bullish),
                StarType::EveningStar | StarType::EveningDojiStar => Ok(IndicatorSignal::Bearish),
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
    fn test_morning_star() {
        let ms = MorningStar::new();
        // First: Large bearish, Second: Small star, Third: Large bullish
        let pattern = ms.detect_pattern(
            110.0, 112.0, 98.0, 100.0,   // Large bearish
            99.0, 101.0, 97.0, 99.5,     // Small star
            98.0, 112.0, 97.0, 110.0,    // Large bullish
        );
        assert_eq!(pattern, Some(StarType::MorningStar));
    }

    #[test]
    fn test_evening_star() {
        let ms = MorningStar::new();
        // First: Large bullish, Second: Small star, Third: Large bearish
        let pattern = ms.detect_pattern(
            100.0, 112.0, 98.0, 110.0,   // Large bullish
            111.0, 113.0, 109.0, 111.5,  // Small star
            112.0, 113.0, 98.0, 100.0,   // Large bearish
        );
        assert_eq!(pattern, Some(StarType::EveningStar));
    }

    #[test]
    fn test_not_star() {
        let ms = MorningStar::new();
        // Middle candle too large - not a star
        let pattern = ms.detect_pattern(
            110.0, 112.0, 98.0, 100.0,
            100.0, 110.0, 95.0, 105.0,   // Large middle
            98.0, 112.0, 97.0, 110.0,
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_morning_star_series() {
        let ms = MorningStar::new();
        let open = vec![110.0, 99.0, 98.0];
        let high = vec![112.0, 101.0, 112.0];
        let low = vec![98.0, 97.0, 97.0];
        let close = vec![100.0, 99.5, 110.0];

        let result = ms.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 3);
    }
}
