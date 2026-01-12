//! Harami Candlestick Pattern
//!
//! Identifies Bullish and Bearish Harami patterns.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of harami pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HaramiType {
    /// Bullish harami - large bearish candle followed by small bullish candle inside
    Bullish,
    /// Bearish harami - large bullish candle followed by small bearish candle inside
    Bearish,
    /// Harami cross - harami with doji second candle (stronger signal)
    BullishCross,
    /// Bearish harami cross
    BearishCross,
}

/// Harami candlestick pattern indicator.
///
/// A harami (meaning "pregnant" in Japanese) is a two-candle reversal pattern:
/// - The first candle is a large body candle in the direction of the trend
/// - The second candle is a smaller body candle completely contained within the first
///
/// The pattern suggests trend exhaustion and potential reversal.
#[derive(Debug, Clone)]
pub struct Harami {
    /// Maximum ratio of second body to first body (default: 0.5)
    max_body_ratio: f64,
    /// Threshold for doji detection in harami cross (body/range ratio)
    doji_threshold: f64,
}

impl Harami {
    /// Create a new Harami indicator with default parameters.
    pub fn new() -> Self {
        Self {
            max_body_ratio: 0.5,
            doji_threshold: 0.05,
        }
    }

    /// Create a Harami indicator with custom parameters.
    pub fn with_params(max_body_ratio: f64, doji_threshold: f64) -> Self {
        Self {
            max_body_ratio,
            doji_threshold,
        }
    }

    /// Detect harami pattern between two consecutive candles.
    fn detect_pattern(
        &self,
        prev_open: f64,
        _prev_high: f64,
        _prev_low: f64,
        prev_close: f64,
        curr_open: f64,
        curr_high: f64,
        curr_low: f64,
        curr_close: f64,
    ) -> Option<HaramiType> {
        let prev_body = (prev_close - prev_open).abs();
        let curr_body = (curr_close - curr_open).abs();
        let curr_range = curr_high - curr_low;

        // Previous candle must have significant body
        if prev_body < f64::EPSILON {
            return None;
        }

        // Current body must be smaller than previous
        if curr_body >= prev_body * self.max_body_ratio {
            return None;
        }

        let prev_body_high = prev_open.max(prev_close);
        let prev_body_low = prev_open.min(prev_close);
        let curr_body_high = curr_open.max(curr_close);
        let curr_body_low = curr_open.min(curr_close);

        // Current body must be contained within previous body
        if curr_body_low < prev_body_low || curr_body_high > prev_body_high {
            return None;
        }

        let prev_is_bullish = prev_close > prev_open;
        let is_doji = curr_range > 0.0 && curr_body / curr_range < self.doji_threshold;

        // Bullish Harami: Previous bearish, current inside
        if !prev_is_bullish {
            if is_doji {
                return Some(HaramiType::BullishCross);
            }
            return Some(HaramiType::Bullish);
        }

        // Bearish Harami: Previous bullish, current inside
        if prev_is_bullish {
            if is_doji {
                return Some(HaramiType::BearishCross);
            }
            return Some(HaramiType::Bearish);
        }

        None
    }

    /// Calculate harami patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Bullish harami
    /// - 2.0: Bullish harami cross
    /// - -1.0: Bearish harami
    /// - -2.0: Bearish harami cross
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
                    HaramiType::Bullish => 1.0,
                    HaramiType::BullishCross => 2.0,
                    HaramiType::Bearish => -1.0,
                    HaramiType::BearishCross => -2.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<HaramiType>> {
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

impl Default for Harami {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for Harami {
    fn name(&self) -> &str {
        "Harami"
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

impl SignalIndicator for Harami {
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
                HaramiType::Bullish | HaramiType::BullishCross => Ok(IndicatorSignal::Bullish),
                HaramiType::Bearish | HaramiType::BearishCross => Ok(IndicatorSignal::Bearish),
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
    fn test_bullish_harami() {
        let harami = Harami::new();
        // Previous: large bearish candle, Current: small candle inside
        let pattern = harami.detect_pattern(
            110.0, 112.0, 98.0, 100.0,  // Large bearish
            104.0, 106.0, 103.0, 105.0,  // Small inside
        );
        assert_eq!(pattern, Some(HaramiType::Bullish));
    }

    #[test]
    fn test_bearish_harami() {
        let harami = Harami::new();
        // Previous: large bullish candle, Current: small candle inside
        let pattern = harami.detect_pattern(
            100.0, 112.0, 98.0, 110.0,  // Large bullish
            106.0, 107.0, 104.0, 105.0,  // Small inside
        );
        assert_eq!(pattern, Some(HaramiType::Bearish));
    }

    #[test]
    fn test_harami_cross() {
        let harami = Harami::new();
        // Large bearish followed by doji inside
        let pattern = harami.detect_pattern(
            110.0, 112.0, 98.0, 100.0,  // Large bearish
            105.0, 107.0, 103.0, 105.0,  // Doji inside
        );
        assert_eq!(pattern, Some(HaramiType::BullishCross));
    }

    #[test]
    fn test_not_harami() {
        let harami = Harami::new();
        // Second candle not contained within first
        let pattern = harami.detect_pattern(
            100.0, 105.0, 98.0, 102.0,
            101.0, 108.0, 100.0, 106.0,
        );
        assert!(pattern.is_none());
    }
}
