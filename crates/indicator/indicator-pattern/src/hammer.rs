//! Hammer and Hanging Man Candlestick Patterns
//!
//! Identifies Hammer (bullish reversal) and Hanging Man (bearish reversal) patterns.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of hammer-like pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HammerType {
    /// Hammer - bullish reversal at bottom
    Hammer,
    /// Inverted Hammer - bullish reversal at bottom with upper shadow
    InvertedHammer,
    /// Hanging Man - bearish reversal at top (same shape as hammer)
    HangingMan,
    /// Shooting Star - bearish reversal at top (same shape as inverted hammer)
    ShootingStar,
}

/// Hammer candlestick pattern indicator.
///
/// Hammer patterns are characterized by:
/// - Small body at the upper end of the trading range
/// - Long lower shadow (at least 2x body length)
/// - Little or no upper shadow
///
/// The pattern's signal depends on prior trend context.
#[derive(Debug, Clone)]
pub struct Hammer {
    /// Minimum shadow to body ratio (default: 2.0)
    shadow_ratio: f64,
    /// Maximum body to range ratio (default: 0.3)
    body_ratio_max: f64,
    /// Number of bars to determine prior trend
    trend_period: usize,
}

impl Hammer {
    /// Create a new Hammer indicator with default parameters.
    pub fn new() -> Self {
        Self {
            shadow_ratio: 2.0,
            body_ratio_max: 0.3,
            trend_period: 5,
        }
    }

    /// Create a Hammer indicator with custom parameters.
    pub fn with_params(shadow_ratio: f64, body_ratio_max: f64, trend_period: usize) -> Self {
        Self {
            shadow_ratio,
            body_ratio_max,
            trend_period,
        }
    }

    /// Detect hammer-like pattern at a single bar (without trend context).
    fn detect_pattern(&self, open: f64, high: f64, low: f64, close: f64) -> Option<bool> {
        let range = high - low;
        if range <= 0.0 {
            return None;
        }

        let body = (close - open).abs();
        let body_ratio = body / range;

        if body_ratio > self.body_ratio_max {
            return None;
        }

        let upper_shadow = high - open.max(close);
        let lower_shadow = open.min(close) - low;

        // Hammer-like (long lower shadow)
        if lower_shadow >= body * self.shadow_ratio && upper_shadow < body * 0.5 {
            return Some(true); // true = hammer-like (long lower shadow)
        }

        // Inverted hammer-like (long upper shadow)
        if upper_shadow >= body * self.shadow_ratio && lower_shadow < body * 0.5 {
            return Some(false); // false = inverted hammer-like (long upper shadow)
        }

        None
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

    /// Calculate hammer patterns for all bars.
    ///
    /// Returns numeric values:
    /// - 1.0: Hammer (bullish)
    /// - 2.0: Inverted Hammer (bullish)
    /// - -1.0: Hanging Man (bearish)
    /// - -2.0: Shooting Star (bearish)
    /// - NaN: No pattern
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        for i in 0..n {
            if let Some(is_lower_shadow) = self.detect_pattern(open[i], high[i], low[i], close[i]) {
                let prior_trend = self.prior_trend(close, i);

                match (is_lower_shadow, prior_trend) {
                    (true, Some(false)) => result[i] = 1.0,   // Hammer (after downtrend)
                    (true, Some(true)) => result[i] = -1.0,   // Hanging Man (after uptrend)
                    (false, Some(false)) => result[i] = 2.0,  // Inverted Hammer (after downtrend)
                    (false, Some(true)) => result[i] = -2.0,  // Shooting Star (after uptrend)
                    _ => {} // No clear trend context
                }
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<HammerType>> {
        let values = self.calculate(open, high, low, close);
        values.iter().map(|&v| {
            if v.is_nan() {
                None
            } else if v == 1.0 {
                Some(HammerType::Hammer)
            } else if v == 2.0 {
                Some(HammerType::InvertedHammer)
            } else if v == -1.0 {
                Some(HammerType::HangingMan)
            } else if v == -2.0 {
                Some(HammerType::ShootingStar)
            } else {
                None
            }
        }).collect()
    }
}

impl Default for Hammer {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for Hammer {
    fn name(&self) -> &str {
        "Hammer"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.trend_period + 1
    }
}

impl SignalIndicator for Hammer {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last > 0.0 {
                    return Ok(IndicatorSignal::Bullish);
                } else if last < 0.0 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
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
    fn test_hammer_detection() {
        let hammer = Hammer::new();
        // Classic hammer shape: small body at top, long lower shadow, no upper shadow
        // body = 1, lower shadow = 10, upper shadow = 0, range = 11
        // body_ratio = 1/11 = 0.09 < 0.3 (OK)
        // lower_shadow = 10 >= body * 2 = 2 (OK)
        // upper_shadow = 0 < body * 0.5 = 0.5 (OK)
        let pattern = hammer.detect_pattern(100.0, 101.0, 90.0, 101.0);
        assert_eq!(pattern, Some(true)); // Hammer-like
    }

    #[test]
    fn test_inverted_hammer_detection() {
        let hammer = Hammer::new();
        // Inverted hammer: small body at bottom, long upper shadow, no lower shadow
        // body = 1, upper shadow = 10, lower shadow = 0, range = 11
        let pattern = hammer.detect_pattern(100.0, 111.0, 100.0, 101.0);
        assert_eq!(pattern, Some(false)); // Inverted hammer-like
    }

    #[test]
    fn test_not_hammer() {
        let hammer = Hammer::new();
        // Large body - not a hammer
        let pattern = hammer.detect_pattern(100.0, 110.0, 95.0, 108.0);
        assert!(pattern.is_none());
    }

    #[test]
    fn test_hammer_with_trend() {
        let hammer = Hammer::new();
        // Downtrend followed by hammer
        let open = vec![110.0, 108.0, 105.0, 102.0, 100.0, 98.0, 95.0];
        let high = vec![111.0, 109.0, 106.0, 103.0, 101.0, 99.0, 96.0];
        let low = vec![107.0, 104.0, 101.0, 98.0, 95.0, 92.0, 85.0];
        let close = vec![108.0, 105.0, 102.0, 99.0, 96.0, 93.0, 95.5];

        let result = hammer.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 7);
    }
}
