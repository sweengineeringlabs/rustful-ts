//! Three White Soldiers and Three Black Crows Candlestick Patterns
//!
//! Identifies strong three-candle continuation/reversal patterns.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of three-candle pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreeSoldiersType {
    /// Three White Soldiers - strong bullish reversal
    WhiteSoldiers,
    /// Three Black Crows - strong bearish reversal
    BlackCrows,
}

/// Three White Soldiers and Three Black Crows pattern indicator.
///
/// Three White Soldiers (bullish):
/// - Three consecutive bullish candles
/// - Each opens within the previous body
/// - Each closes progressively higher
/// - Small or no upper shadows
///
/// Three Black Crows (bearish):
/// - Three consecutive bearish candles
/// - Each opens within the previous body
/// - Each closes progressively lower
/// - Small or no lower shadows
#[derive(Debug, Clone)]
pub struct ThreeSoldiers {
    /// Maximum shadow to body ratio (default: 0.3)
    max_shadow_ratio: f64,
    /// Minimum body to range ratio (default: 0.6)
    min_body_ratio: f64,
}

impl ThreeSoldiers {
    /// Create a new ThreeSoldiers indicator with default parameters.
    pub fn new() -> Self {
        Self {
            max_shadow_ratio: 0.3,
            min_body_ratio: 0.6,
        }
    }

    /// Create a ThreeSoldiers indicator with custom parameters.
    pub fn with_params(max_shadow_ratio: f64, min_body_ratio: f64) -> Self {
        Self {
            max_shadow_ratio,
            min_body_ratio,
        }
    }

    /// Check if a candle qualifies as a soldier (bullish with small upper shadow).
    fn is_valid_white_soldier(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        if close <= open {
            return false; // Must be bullish
        }

        let body = close - open;
        let range = high - low;
        let upper_shadow = high - close;

        if range <= 0.0 {
            return false;
        }

        body / range >= self.min_body_ratio && upper_shadow / body <= self.max_shadow_ratio
    }

    /// Check if a candle qualifies as a crow (bearish with small lower shadow).
    fn is_valid_black_crow(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        if close >= open {
            return false; // Must be bearish
        }

        let body = open - close;
        let range = high - low;
        let lower_shadow = close - low;

        if range <= 0.0 {
            return false;
        }

        body / range >= self.min_body_ratio && lower_shadow / body <= self.max_shadow_ratio
    }

    /// Detect three soldiers/crows pattern at index (requires 3 candles ending at this index).
    fn detect_pattern(
        &self,
        o1: f64, h1: f64, l1: f64, c1: f64,
        o2: f64, h2: f64, l2: f64, c2: f64,
        o3: f64, h3: f64, l3: f64, c3: f64,
    ) -> Option<ThreeSoldiersType> {
        // Check for Three White Soldiers
        if self.is_valid_white_soldier(o1, h1, l1, c1)
            && self.is_valid_white_soldier(o2, h2, l2, c2)
            && self.is_valid_white_soldier(o3, h3, l3, c3)
        {
            // Each opens within previous body
            let opens_in_body_2 = o2 >= o1 && o2 <= c1;
            let opens_in_body_3 = o3 >= o2 && o3 <= c2;

            // Progressive closes
            let progressive = c2 > c1 && c3 > c2;

            if opens_in_body_2 && opens_in_body_3 && progressive {
                return Some(ThreeSoldiersType::WhiteSoldiers);
            }
        }

        // Check for Three Black Crows
        if self.is_valid_black_crow(o1, h1, l1, c1)
            && self.is_valid_black_crow(o2, h2, l2, c2)
            && self.is_valid_black_crow(o3, h3, l3, c3)
        {
            // Each opens within previous body
            let opens_in_body_2 = o2 <= o1 && o2 >= c1;
            let opens_in_body_3 = o3 <= o2 && o3 >= c2;

            // Progressive closes
            let progressive = c2 < c1 && c3 < c2;

            if opens_in_body_2 && opens_in_body_3 && progressive {
                return Some(ThreeSoldiersType::BlackCrows);
            }
        }

        None
    }

    /// Calculate three soldiers/crows patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Three White Soldiers (bullish)
    /// - -1.0: Three Black Crows (bearish)
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
                    ThreeSoldiersType::WhiteSoldiers => 1.0,
                    ThreeSoldiersType::BlackCrows => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<ThreeSoldiersType>> {
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

impl Default for ThreeSoldiers {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for ThreeSoldiers {
    fn name(&self) -> &str {
        "ThreeSoldiers"
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

impl SignalIndicator for ThreeSoldiers {
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
                ThreeSoldiersType::WhiteSoldiers => Ok(IndicatorSignal::Bullish),
                ThreeSoldiersType::BlackCrows => Ok(IndicatorSignal::Bearish),
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
    fn test_three_white_soldiers() {
        let ts = ThreeSoldiers::new();
        // Three consecutive bullish candles with progressive closes
        let pattern = ts.detect_pattern(
            100.0, 106.0, 99.0, 105.0,   // First soldier
            104.0, 111.0, 103.0, 110.0,  // Second soldier
            109.0, 116.0, 108.0, 115.0,  // Third soldier
        );
        assert_eq!(pattern, Some(ThreeSoldiersType::WhiteSoldiers));
    }

    #[test]
    fn test_three_black_crows() {
        let ts = ThreeSoldiers::new();
        // Three consecutive bearish candles with progressive closes
        let pattern = ts.detect_pattern(
            115.0, 116.0, 109.0, 110.0,  // First crow
            111.0, 112.0, 104.0, 105.0,  // Second crow
            106.0, 107.0, 99.0, 100.0,   // Third crow
        );
        assert_eq!(pattern, Some(ThreeSoldiersType::BlackCrows));
    }

    #[test]
    fn test_not_soldiers() {
        let ts = ThreeSoldiers::new();
        // Third candle doesn't open within second body
        let pattern = ts.detect_pattern(
            100.0, 106.0, 99.0, 105.0,
            104.0, 111.0, 103.0, 110.0,
            115.0, 120.0, 114.0, 119.0,  // Opens above second close
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_soldiers_series() {
        let ts = ThreeSoldiers::new();
        let open = vec![100.0, 104.0, 109.0];
        let high = vec![106.0, 111.0, 116.0];
        let low = vec![99.0, 103.0, 108.0];
        let close = vec![105.0, 110.0, 115.0];

        let result = ts.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 1.0);
    }
}
