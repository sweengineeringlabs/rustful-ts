//! Spinning Top Candlestick Pattern
//!
//! Identifies Spinning Top patterns indicating market indecision.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of Spinning Top pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpinningTopType {
    /// Bullish spinning top (white body)
    Bullish,
    /// Bearish spinning top (black body)
    Bearish,
}

/// Spinning Top candlestick pattern indicator.
///
/// A Spinning Top has:
/// - Small body relative to the overall range
/// - Significant upper and lower shadows
/// - Indicates indecision in the market
///
/// Unlike a doji, a spinning top has a visible body, but it's small
/// compared to the shadows. The color (bullish/bearish) is less important
/// than the pattern itself, which signals indecision.
#[derive(Debug, Clone)]
pub struct SpinningTop {
    /// Maximum body to range ratio (default: 0.3)
    max_body_ratio: f64,
    /// Minimum shadow to body ratio for both shadows (default: 0.5)
    min_shadow_ratio: f64,
    /// Minimum body ratio to distinguish from doji (default: 0.05)
    min_body_ratio: f64,
}

impl SpinningTop {
    /// Create a new SpinningTop indicator with default parameters.
    pub fn new() -> Self {
        Self {
            max_body_ratio: 0.3,
            min_shadow_ratio: 0.5,
            min_body_ratio: 0.05,
        }
    }

    /// Create a SpinningTop indicator with custom parameters.
    ///
    /// # Arguments
    /// * `max_body_ratio` - Maximum body/range ratio to qualify
    /// * `min_shadow_ratio` - Minimum shadow/body ratio for both shadows
    /// * `min_body_ratio` - Minimum body/range ratio to distinguish from doji
    pub fn with_params(max_body_ratio: f64, min_shadow_ratio: f64, min_body_ratio: f64) -> Self {
        Self {
            max_body_ratio,
            min_shadow_ratio,
            min_body_ratio,
        }
    }

    /// Detect spinning top pattern at a single bar.
    pub fn detect_single(&self, open: f64, high: f64, low: f64, close: f64) -> Option<SpinningTopType> {
        let range = high - low;
        if range <= 0.0 {
            return None;
        }

        let body = (close - open).abs();
        let body_ratio = body / range;

        // Body must be small but not too small (doji)
        if body_ratio > self.max_body_ratio || body_ratio < self.min_body_ratio {
            return None;
        }

        // Avoid division by zero for very small body
        if body < f64::EPSILON {
            return None;
        }

        let upper_shadow = high - open.max(close);
        let lower_shadow = open.min(close) - low;

        // Both shadows must be significant relative to body
        if upper_shadow / body < self.min_shadow_ratio || lower_shadow / body < self.min_shadow_ratio {
            return None;
        }

        // Determine color
        if close > open {
            Some(SpinningTopType::Bullish)
        } else {
            Some(SpinningTopType::Bearish)
        }
    }

    /// Calculate spinning top patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Bullish spinning top
    /// - -1.0: Bearish spinning top
    /// - NaN: No pattern
    ///
    /// Note: The signal value primarily indicates the presence of indecision.
    /// The bullish/bearish distinction is based on candle color but
    /// both indicate market uncertainty.
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        for i in 0..n {
            if let Some(pattern) = self.detect_single(open[i], high[i], low[i], close[i]) {
                result[i] = match pattern {
                    SpinningTopType::Bullish => 1.0,
                    SpinningTopType::Bearish => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<SpinningTopType>> {
        (0..close.len())
            .map(|i| self.detect_single(open[i], high[i], low[i], close[i]))
            .collect()
    }

    /// Check if spinning top appears at market turning point.
    /// Returns true if the spinning top appears after a trend.
    pub fn is_reversal_candidate(&self, close: &[f64], idx: usize, lookback: usize) -> Option<bool> {
        if idx < lookback {
            return None;
        }

        let start_close = close[idx - lookback];
        let current_close = close[idx];

        // Check if there was a preceding trend
        let trend_change = (current_close - start_close) / start_close;
        if trend_change.abs() >= 0.02 {
            Some(true) // Potential reversal point
        } else {
            Some(false) // No clear preceding trend
        }
    }
}

impl Default for SpinningTop {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for SpinningTop {
    fn name(&self) -> &str {
        "SpinningTop"
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

impl SignalIndicator for SpinningTop {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = data.close.len();
        if let Some(_pattern) = self.detect_single(
            data.open[n - 1],
            data.high[n - 1],
            data.low[n - 1],
            data.close[n - 1],
        ) {
            // Spinning tops primarily signal indecision
            // Return Neutral as neither strongly bullish nor bearish
            return Ok(IndicatorSignal::Neutral);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        // All spinning tops signal indecision (Neutral)
        let signals = values.iter().map(|&v| {
            if v.is_nan() {
                IndicatorSignal::Neutral
            } else {
                // Spinning tops indicate indecision regardless of color
                IndicatorSignal::Neutral
            }
        }).collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bullish_spinning_top() {
        let st = SpinningTop::new();
        // Small bullish body with significant shadows
        // range=20, body=2, upper_shadow=8, lower_shadow=10
        let result = st.detect_single(104.0, 114.0, 94.0, 106.0);
        assert_eq!(result, Some(SpinningTopType::Bullish));
    }

    #[test]
    fn test_bearish_spinning_top() {
        let st = SpinningTop::new();
        // Small bearish body with significant shadows
        // range=20, body=2, upper_shadow=10, lower_shadow=8
        let result = st.detect_single(106.0, 116.0, 96.0, 104.0);
        assert_eq!(result, Some(SpinningTopType::Bearish));
    }

    #[test]
    fn test_not_spinning_top_large_body() {
        let st = SpinningTop::new();
        // Body too large
        let result = st.detect_single(100.0, 115.0, 95.0, 112.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_not_spinning_top_small_shadows() {
        let st = SpinningTop::new();
        // Shadows too small (body = 2, but shadows are tiny)
        let result = st.detect_single(99.0, 101.5, 98.5, 101.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_spinning_top_series() {
        let st = SpinningTop::new();
        let open = vec![104.0, 100.0];
        let high = vec![114.0, 115.0];
        let low = vec![94.0, 95.0];
        let close = vec![106.0, 112.0];

        let result = st.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1.0); // Bullish spinning top
        assert!(result[1].is_nan()); // Large body, not a spinning top
    }

    #[test]
    fn test_reversal_candidate() {
        let st = SpinningTop::new();
        let close = vec![100.0, 102.0, 104.0, 106.0, 108.0, 109.0];

        // After uptrend
        let result = st.is_reversal_candidate(&close, 5, 4);
        assert_eq!(result, Some(true));
    }
}
