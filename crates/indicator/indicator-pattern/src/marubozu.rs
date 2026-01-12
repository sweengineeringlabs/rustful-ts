//! Marubozu Candlestick Pattern
//!
//! Identifies strong directional candlestick patterns with no shadows.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of Marubozu pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarubozuType {
    /// Bullish Marubozu (White) - opens at low, closes at high
    Bullish,
    /// Bearish Marubozu (Black) - opens at high, closes at low
    Bearish,
    /// Opening Marubozu - no shadow on opening side
    OpeningBullish,
    /// Opening Marubozu - no shadow on opening side
    OpeningBearish,
    /// Closing Marubozu - no shadow on closing side
    ClosingBullish,
    /// Closing Marubozu - no shadow on closing side
    ClosingBearish,
}

/// Marubozu candlestick pattern indicator.
///
/// A Marubozu is a candlestick with no shadows (wicks), indicating
/// strong conviction in one direction. The pattern has variations:
///
/// - Full Marubozu: No upper or lower shadow
/// - Opening Marubozu: No shadow on the opening side
/// - Closing Marubozu: No shadow on the closing side
///
/// These patterns indicate strong momentum and conviction.
#[derive(Debug, Clone)]
pub struct Marubozu {
    /// Maximum shadow tolerance as percentage of range (default: 0.01 = 1%)
    shadow_tolerance: f64,
    /// Minimum body to range ratio (default: 0.95)
    min_body_ratio: f64,
}

impl Marubozu {
    /// Create a new Marubozu indicator with default parameters.
    pub fn new() -> Self {
        Self {
            shadow_tolerance: 0.01,
            min_body_ratio: 0.95,
        }
    }

    /// Create a Marubozu indicator with custom tolerance for partial patterns.
    pub fn with_tolerance(shadow_tolerance: f64) -> Self {
        Self {
            shadow_tolerance,
            min_body_ratio: 1.0 - shadow_tolerance * 2.0,
        }
    }

    /// Detect marubozu pattern at a single bar.
    pub fn detect_single(&self, open: f64, high: f64, low: f64, close: f64) -> Option<MarubozuType> {
        let range = high - low;
        if range <= 0.0 {
            return None;
        }

        let body = (close - open).abs();
        let body_ratio = body / range;

        // Minimum body requirement
        if body_ratio < self.min_body_ratio * 0.85 {
            return None;
        }

        let is_bullish = close > open;
        let upper_shadow = if is_bullish { high - close } else { high - open };
        let lower_shadow = if is_bullish { open - low } else { close - low };

        let upper_ratio = upper_shadow / range;
        let lower_ratio = lower_shadow / range;

        let no_upper = upper_ratio <= self.shadow_tolerance;
        let no_lower = lower_ratio <= self.shadow_tolerance;

        if is_bullish {
            if no_upper && no_lower {
                Some(MarubozuType::Bullish)
            } else if no_lower && !no_upper {
                Some(MarubozuType::OpeningBullish)
            } else if no_upper && !no_lower {
                Some(MarubozuType::ClosingBullish)
            } else {
                None
            }
        } else {
            if no_upper && no_lower {
                Some(MarubozuType::Bearish)
            } else if no_upper && !no_lower {
                Some(MarubozuType::OpeningBearish)
            } else if no_lower && !no_upper {
                Some(MarubozuType::ClosingBearish)
            } else {
                None
            }
        }
    }

    /// Calculate marubozu patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Bullish Marubozu (full)
    /// - 0.8: Opening Bullish Marubozu
    /// - 0.6: Closing Bullish Marubozu
    /// - -1.0: Bearish Marubozu (full)
    /// - -0.8: Opening Bearish Marubozu
    /// - -0.6: Closing Bearish Marubozu
    /// - NaN: No pattern
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        for i in 0..n {
            if let Some(pattern) = self.detect_single(open[i], high[i], low[i], close[i]) {
                result[i] = match pattern {
                    MarubozuType::Bullish => 1.0,
                    MarubozuType::OpeningBullish => 0.8,
                    MarubozuType::ClosingBullish => 0.6,
                    MarubozuType::Bearish => -1.0,
                    MarubozuType::OpeningBearish => -0.8,
                    MarubozuType::ClosingBearish => -0.6,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<MarubozuType>> {
        (0..close.len())
            .map(|i| self.detect_single(open[i], high[i], low[i], close[i]))
            .collect()
    }

    /// Check if candle is a full marubozu (no shadows).
    pub fn is_full_marubozu(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        matches!(
            self.detect_single(open, high, low, close),
            Some(MarubozuType::Bullish) | Some(MarubozuType::Bearish)
        )
    }
}

impl Default for Marubozu {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for Marubozu {
    fn name(&self) -> &str {
        "Marubozu"
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

impl SignalIndicator for Marubozu {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = data.close.len();
        if let Some(pattern) = self.detect_single(
            data.open[n - 1],
            data.high[n - 1],
            data.low[n - 1],
            data.close[n - 1],
        ) {
            return match pattern {
                MarubozuType::Bullish | MarubozuType::OpeningBullish | MarubozuType::ClosingBullish => {
                    Ok(IndicatorSignal::Bullish)
                }
                MarubozuType::Bearish | MarubozuType::OpeningBearish | MarubozuType::ClosingBearish => {
                    Ok(IndicatorSignal::Bearish)
                }
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
    fn test_bullish_marubozu() {
        let marubozu = Marubozu::new();
        // Open at low, close at high - perfect bullish marubozu
        let result = marubozu.detect_single(100.0, 110.0, 100.0, 110.0);
        assert_eq!(result, Some(MarubozuType::Bullish));
    }

    #[test]
    fn test_bearish_marubozu() {
        let marubozu = Marubozu::new();
        // Open at high, close at low - perfect bearish marubozu
        let result = marubozu.detect_single(110.0, 110.0, 100.0, 100.0);
        assert_eq!(result, Some(MarubozuType::Bearish));
    }

    #[test]
    fn test_opening_bullish_marubozu() {
        let marubozu = Marubozu::with_tolerance(0.02);
        // No lower shadow, some upper shadow
        let result = marubozu.detect_single(100.0, 112.0, 100.0, 110.0);
        assert_eq!(result, Some(MarubozuType::OpeningBullish));
    }

    #[test]
    fn test_closing_bullish_marubozu() {
        let marubozu = Marubozu::with_tolerance(0.05);
        // Closing bullish: some lower shadow, no upper shadow
        // For bullish: close > open, so upper_shadow = high - close, lower_shadow = open - low
        // open=101, high=110, low=100, close=110 (no upper shadow, small lower shadow)
        // body = 9, range = 10, body_ratio = 0.9 (needs min_body_ratio * 0.85 = 0.765)
        let result = marubozu.detect_single(101.0, 110.0, 100.0, 110.0);
        assert_eq!(result, Some(MarubozuType::ClosingBullish));
    }

    #[test]
    fn test_not_marubozu() {
        let marubozu = Marubozu::new();
        // Significant shadows on both sides
        let result = marubozu.detect_single(105.0, 115.0, 95.0, 108.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_marubozu_series() {
        let marubozu = Marubozu::new();
        // Bar 0: perfect bullish (open=low, close=high)
        // Bar 1: perfect bearish (open=high, close=low)
        let open = vec![100.0, 110.0];
        let high = vec![110.0, 110.0];
        let low = vec![100.0, 100.0];
        let close = vec![110.0, 100.0];

        let result = marubozu.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1.0); // Bullish marubozu
        assert_eq!(result[1], -1.0); // Bearish marubozu
    }
}
