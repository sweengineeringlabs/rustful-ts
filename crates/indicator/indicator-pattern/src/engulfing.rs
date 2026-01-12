//! Engulfing Candlestick Pattern
//!
//! Identifies Bullish and Bearish Engulfing patterns.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of engulfing pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngulfingType {
    /// Bullish engulfing - green candle fully engulfs prior red candle
    Bullish,
    /// Bearish engulfing - red candle fully engulfs prior green candle
    Bearish,
}

/// Engulfing candlestick pattern indicator.
///
/// An engulfing pattern is a two-candle reversal pattern where:
/// - Bullish Engulfing: A bearish candle followed by a larger bullish candle
///   that completely engulfs the prior candle's body
/// - Bearish Engulfing: A bullish candle followed by a larger bearish candle
///   that completely engulfs the prior candle's body
#[derive(Debug, Clone)]
pub struct Engulfing {
    /// Minimum ratio of engulfing body to engulfed body (default: 1.0)
    min_engulf_ratio: f64,
}

impl Engulfing {
    /// Create a new Engulfing indicator with default parameters.
    pub fn new() -> Self {
        Self {
            min_engulf_ratio: 1.0,
        }
    }

    /// Create an Engulfing indicator with custom minimum ratio.
    ///
    /// # Arguments
    /// * `min_engulf_ratio` - Minimum ratio (engulfing body / engulfed body)
    pub fn with_ratio(min_engulf_ratio: f64) -> Self {
        Self { min_engulf_ratio }
    }

    /// Detect engulfing pattern between two consecutive candles.
    fn detect_pattern(
        &self,
        prev_open: f64,
        prev_close: f64,
        curr_open: f64,
        curr_close: f64,
    ) -> Option<EngulfingType> {
        let prev_body = (prev_close - prev_open).abs();
        let curr_body = (curr_close - curr_open).abs();

        // Skip if bodies are too small (potential doji)
        if prev_body < f64::EPSILON || curr_body < f64::EPSILON {
            return None;
        }

        // Check engulfing ratio
        if curr_body < prev_body * self.min_engulf_ratio {
            return None;
        }

        let prev_is_bullish = prev_close > prev_open;
        let curr_is_bullish = curr_close > curr_open;

        // Bullish Engulfing: Previous bearish, current bullish, current engulfs previous
        if !prev_is_bullish && curr_is_bullish {
            let prev_body_high = prev_open;
            let prev_body_low = prev_close;
            let curr_body_high = curr_close;
            let curr_body_low = curr_open;

            if curr_body_low <= prev_body_low && curr_body_high >= prev_body_high {
                return Some(EngulfingType::Bullish);
            }
        }

        // Bearish Engulfing: Previous bullish, current bearish, current engulfs previous
        if prev_is_bullish && !curr_is_bullish {
            let prev_body_high = prev_close;
            let prev_body_low = prev_open;
            let curr_body_high = curr_open;
            let curr_body_low = curr_close;

            if curr_body_low <= prev_body_low && curr_body_high >= prev_body_high {
                return Some(EngulfingType::Bearish);
            }
        }

        None
    }

    /// Calculate engulfing patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Bullish engulfing
    /// - -1.0: Bearish engulfing
    /// - NaN: No pattern
    pub fn calculate(&self, open: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < 2 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in 1..n {
            if let Some(pattern) = self.detect_pattern(open[i - 1], close[i - 1], open[i], close[i]) {
                result[i] = match pattern {
                    EngulfingType::Bullish => 1.0,
                    EngulfingType::Bearish => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], close: &[f64]) -> Vec<Option<EngulfingType>> {
        let n = close.len();
        if n < 2 {
            return vec![None; n];
        }

        let mut result = vec![None; n];

        for i in 1..n {
            result[i] = self.detect_pattern(open[i - 1], close[i - 1], open[i], close[i]);
        }

        result
    }
}

impl Default for Engulfing {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for Engulfing {
    fn name(&self) -> &str {
        "Engulfing"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 2 {
            return Err(IndicatorError::InsufficientData {
                required: 2,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.open, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        2
    }
}

impl SignalIndicator for Engulfing {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = data.close.len();
        if let Some(pattern) = self.detect_pattern(
            data.open[n - 2],
            data.close[n - 2],
            data.open[n - 1],
            data.close[n - 1],
        ) {
            return match pattern {
                EngulfingType::Bullish => Ok(IndicatorSignal::Bullish),
                EngulfingType::Bearish => Ok(IndicatorSignal::Bearish),
            };
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.open, &data.close);
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
    fn test_bullish_engulfing() {
        let engulfing = Engulfing::new();
        // Previous: bearish (open > close), Current: bullish that engulfs
        let pattern = engulfing.detect_pattern(102.0, 98.0, 97.0, 104.0);
        assert_eq!(pattern, Some(EngulfingType::Bullish));
    }

    #[test]
    fn test_bearish_engulfing() {
        let engulfing = Engulfing::new();
        // Previous: bullish (close > open), Current: bearish that engulfs
        let pattern = engulfing.detect_pattern(98.0, 102.0, 103.0, 96.0);
        assert_eq!(pattern, Some(EngulfingType::Bearish));
    }

    #[test]
    fn test_no_engulfing() {
        let engulfing = Engulfing::new();
        // Current candle doesn't fully engulf previous
        let pattern = engulfing.detect_pattern(100.0, 95.0, 96.0, 99.0);
        assert!(pattern.is_none());
    }

    #[test]
    fn test_engulfing_series() {
        let engulfing = Engulfing::new();
        let open = vec![100.0, 102.0, 97.0];
        let close = vec![102.0, 98.0, 104.0];

        let result = engulfing.calculate(&open, &close);
        assert_eq!(result.len(), 3);
        assert!(result[0].is_nan());
        // Index 2 should have bullish engulfing
        assert_eq!(result[2], 1.0);
    }
}
