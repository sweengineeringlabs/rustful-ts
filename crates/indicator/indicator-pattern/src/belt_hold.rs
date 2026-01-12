//! Belt Hold Candlestick Pattern
//!
//! Identifies single-candle reversal patterns with strong open.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of belt hold pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeltHoldType {
    /// Bullish Belt Hold (White Opening Marubozu) - bullish reversal
    Bullish,
    /// Bearish Belt Hold (Black Opening Marubozu) - bearish reversal
    Bearish,
}

/// Belt Hold pattern indicator.
///
/// A Belt Hold is a strong single-candle reversal pattern:
///
/// Bullish Belt Hold:
/// - Opens at or near the low of the session (gaps down)
/// - Has a large bullish body
/// - Little or no lower shadow
/// - Indicates buyers took control from the open
///
/// Bearish Belt Hold:
/// - Opens at or near the high of the session (gaps up)
/// - Has a large bearish body
/// - Little or no upper shadow
/// - Indicates sellers took control from the open
///
/// These patterns are most significant after a trend in the opposite direction.
#[derive(Debug, Clone)]
pub struct BeltHold {
    /// Maximum shadow ratio on the opening side (default: 0.05 = 5%)
    max_shadow_ratio: f64,
    /// Minimum body to range ratio (default: 0.7)
    min_body_ratio: f64,
    /// Number of bars to determine prior trend
    trend_period: usize,
}

impl BeltHold {
    /// Create a new BeltHold indicator with default parameters.
    pub fn new() -> Self {
        Self {
            max_shadow_ratio: 0.05,
            min_body_ratio: 0.7,
            trend_period: 5,
        }
    }

    /// Create a BeltHold indicator with custom parameters.
    pub fn with_params(max_shadow_ratio: f64, min_body_ratio: f64, trend_period: usize) -> Self {
        Self {
            max_shadow_ratio,
            min_body_ratio,
            trend_period,
        }
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

    /// Detect belt hold pattern at a single bar.
    fn detect_single(&self, open: f64, high: f64, low: f64, close: f64, prior_uptrend: Option<bool>) -> Option<BeltHoldType> {
        let range = high - low;
        if range <= 0.0 {
            return None;
        }

        let body = (close - open).abs();
        let body_ratio = body / range;

        // Body must be large
        if body_ratio < self.min_body_ratio {
            return None;
        }

        let is_bullish = close > open;

        if is_bullish {
            // Bullish Belt Hold: Opens at/near low, closes near high
            let lower_shadow = open - low;
            let lower_shadow_ratio = lower_shadow / range;

            if lower_shadow_ratio <= self.max_shadow_ratio {
                // Best after downtrend
                if prior_uptrend != Some(true) {
                    return Some(BeltHoldType::Bullish);
                }
            }
        } else {
            // Bearish Belt Hold: Opens at/near high, closes near low
            let upper_shadow = high - open;
            let upper_shadow_ratio = upper_shadow / range;

            if upper_shadow_ratio <= self.max_shadow_ratio {
                // Best after uptrend
                if prior_uptrend != Some(false) {
                    return Some(BeltHoldType::Bearish);
                }
            }
        }

        None
    }

    /// Calculate belt hold patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Bullish Belt Hold
    /// - -1.0: Bearish Belt Hold
    /// - NaN: No pattern
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        for i in 0..n {
            let prior_trend = self.prior_trend(close, i);
            if let Some(pattern) = self.detect_single(open[i], high[i], low[i], close[i], prior_trend) {
                result[i] = match pattern {
                    BeltHoldType::Bullish => 1.0,
                    BeltHoldType::Bearish => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<BeltHoldType>> {
        let n = close.len();
        let mut result = vec![None; n];

        for i in 0..n {
            let prior_trend = self.prior_trend(close, i);
            result[i] = self.detect_single(open[i], high[i], low[i], close[i], prior_trend);
        }

        result
    }
}

impl Default for BeltHold {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for BeltHold {
    fn name(&self) -> &str {
        "BeltHold"
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
        1
    }
}

impl SignalIndicator for BeltHold {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = data.close.len();
        let prior_trend = self.prior_trend(&data.close, n - 1);

        if let Some(pattern) = self.detect_single(
            data.open[n - 1],
            data.high[n - 1],
            data.low[n - 1],
            data.close[n - 1],
            prior_trend,
        ) {
            return match pattern {
                BeltHoldType::Bullish => Ok(IndicatorSignal::Bullish),
                BeltHoldType::Bearish => Ok(IndicatorSignal::Bearish),
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
    fn test_bullish_belt_hold() {
        let bh = BeltHold::new();
        // Opens at low, closes near high (large bullish body, no lower shadow)
        let pattern = bh.detect_single(100.0, 110.0, 100.0, 109.0, Some(false));
        assert_eq!(pattern, Some(BeltHoldType::Bullish));
    }

    #[test]
    fn test_bearish_belt_hold() {
        let bh = BeltHold::new();
        // Opens at high, closes near low (large bearish body, no upper shadow)
        let pattern = bh.detect_single(110.0, 110.0, 100.0, 101.0, Some(true));
        assert_eq!(pattern, Some(BeltHoldType::Bearish));
    }

    #[test]
    fn test_no_pattern_small_body() {
        let bh = BeltHold::new();
        // Small body
        let pattern = bh.detect_single(100.0, 110.0, 90.0, 102.0, Some(false));
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_pattern_large_shadow() {
        let bh = BeltHold::new();
        // Bullish but has lower shadow
        let pattern = bh.detect_single(105.0, 115.0, 100.0, 114.0, Some(false));
        assert!(pattern.is_none());
    }

    #[test]
    fn test_belt_hold_series() {
        let bh = BeltHold::with_params(0.05, 0.7, 3);
        // Downtrend followed by bullish belt hold
        let open = vec![110.0, 108.0, 105.0, 102.0, 100.0];
        let high = vec![111.0, 109.0, 106.0, 103.0, 110.0];
        let low = vec![108.0, 105.0, 102.0, 99.0, 100.0];
        let close = vec![109.0, 106.0, 103.0, 100.0, 109.0];

        let result = bh.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 5);
        // Last bar should be bullish belt hold after downtrend
        assert_eq!(result[4], 1.0);
    }

    #[test]
    fn test_prior_trend() {
        let bh = BeltHold::new();
        let close = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0];

        // After 5 bars of uptrend
        let trend = bh.prior_trend(&close, 5);
        assert_eq!(trend, Some(true));

        let close_down = vec![110.0, 108.0, 106.0, 104.0, 102.0, 100.0];
        let trend_down = bh.prior_trend(&close_down, 5);
        assert_eq!(trend_down, Some(false));
    }
}
