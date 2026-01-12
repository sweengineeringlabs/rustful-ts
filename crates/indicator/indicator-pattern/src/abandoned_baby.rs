//! Abandoned Baby Candlestick Pattern
//!
//! Identifies three-candle gapped reversal patterns.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of abandoned baby pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbandonedBabyType {
    /// Bullish Abandoned Baby - bullish reversal
    Bullish,
    /// Bearish Abandoned Baby - bearish reversal
    Bearish,
}

/// Abandoned Baby pattern indicator.
///
/// The Abandoned Baby is a rare and powerful reversal pattern:
///
/// Bullish Abandoned Baby:
/// 1. First candle: Large bearish candle
/// 2. Second candle: Doji that gaps down (no overlap with first or third candle shadows)
/// 3. Third candle: Large bullish candle that gaps up
///
/// Bearish Abandoned Baby:
/// 1. First candle: Large bullish candle
/// 2. Second candle: Doji that gaps up (no overlap with first or third candle shadows)
/// 3. Third candle: Large bearish candle that gaps down
///
/// The key feature is that the middle doji is completely isolated by gaps on both sides.
#[derive(Debug, Clone)]
pub struct AbandonedBaby {
    /// Maximum body ratio for doji (default: 0.05)
    doji_threshold: f64,
    /// Minimum body ratio for first/third candles (default: 0.5)
    min_body_ratio: f64,
}

impl AbandonedBaby {
    /// Create a new AbandonedBaby indicator with default parameters.
    pub fn new() -> Self {
        Self {
            doji_threshold: 0.05,
            min_body_ratio: 0.5,
        }
    }

    /// Create an AbandonedBaby indicator with custom parameters.
    pub fn with_params(doji_threshold: f64, min_body_ratio: f64) -> Self {
        Self {
            doji_threshold,
            min_body_ratio,
        }
    }

    /// Check if a candle is a doji.
    fn is_doji(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        let range = high - low;
        if range <= 0.0 {
            return true; // Four-price doji
        }
        let body = (close - open).abs();
        body / range <= self.doji_threshold
    }

    /// Check if a candle has a large body.
    fn is_large_body(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        let range = high - low;
        if range <= 0.0 {
            return false;
        }
        let body = (close - open).abs();
        body / range >= self.min_body_ratio
    }

    /// Detect abandoned baby pattern at index (requires 3 candles ending at this index).
    fn detect_pattern(
        &self,
        first_open: f64, first_high: f64, first_low: f64, first_close: f64,
        doji_open: f64, doji_high: f64, doji_low: f64, doji_close: f64,
        third_open: f64, third_high: f64, third_low: f64, third_close: f64,
    ) -> Option<AbandonedBabyType> {
        // Middle candle must be a doji
        if !self.is_doji(doji_open, doji_high, doji_low, doji_close) {
            return None;
        }

        // First and third candles must have large bodies
        if !self.is_large_body(first_open, first_high, first_low, first_close)
            || !self.is_large_body(third_open, third_high, third_low, third_close)
        {
            return None;
        }

        let first_is_bullish = first_close > first_open;
        let third_is_bullish = third_close > third_open;

        // Bullish Abandoned Baby
        if !first_is_bullish && third_is_bullish {
            // Doji must gap down from first (doji high < first low)
            // Doji must gap up to third (doji high < third low)
            if doji_high < first_low && doji_high < third_low {
                return Some(AbandonedBabyType::Bullish);
            }
        }

        // Bearish Abandoned Baby
        if first_is_bullish && !third_is_bullish {
            // Doji must gap up from first (doji low > first high)
            // Doji must gap down to third (doji low > third high)
            if doji_low > first_high && doji_low > third_high {
                return Some(AbandonedBabyType::Bearish);
            }
        }

        None
    }

    /// Calculate abandoned baby patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Bullish Abandoned Baby
    /// - -1.0: Bearish Abandoned Baby
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
                    AbandonedBabyType::Bullish => 1.0,
                    AbandonedBabyType::Bearish => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<AbandonedBabyType>> {
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

impl Default for AbandonedBaby {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for AbandonedBaby {
    fn name(&self) -> &str {
        "AbandonedBaby"
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

impl SignalIndicator for AbandonedBaby {
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
                AbandonedBabyType::Bullish => Ok(IndicatorSignal::Bullish),
                AbandonedBabyType::Bearish => Ok(IndicatorSignal::Bearish),
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
    fn test_bullish_abandoned_baby() {
        let ab = AbandonedBaby::new();
        // First: Large bearish (open=110, close=100, low=98)
        // Doji: Gaps below first low, tiny body (high=97, low=96)
        // Third: Large bullish that gaps above doji (open=98, close=108, low=98)
        let pattern = ab.detect_pattern(
            110.0, 112.0, 98.0, 100.0,   // Large bearish
            96.5, 97.0, 96.0, 96.5,      // Doji with gap (high 97 < first low 98)
            98.0, 110.0, 98.0, 108.0,    // Large bullish (low 98 > doji high 97)
        );
        assert_eq!(pattern, Some(AbandonedBabyType::Bullish));
    }

    #[test]
    fn test_bearish_abandoned_baby() {
        let ab = AbandonedBaby::new();
        // First: Large bullish (open=100, close=110, high=112)
        // Doji: Gaps above first high (low=113)
        // Third: Large bearish that gaps below doji (high=112)
        let pattern = ab.detect_pattern(
            100.0, 112.0, 98.0, 110.0,   // Large bullish
            113.5, 114.0, 113.0, 113.5,  // Doji with gap (low 113 > first high 112)
            112.0, 112.0, 100.0, 102.0,  // Large bearish (high 112 < doji low 113)
        );
        assert_eq!(pattern, Some(AbandonedBabyType::Bearish));
    }

    #[test]
    fn test_no_pattern_no_gap() {
        let ab = AbandonedBaby::new();
        // Doji doesn't gap from first candle
        let pattern = ab.detect_pattern(
            110.0, 112.0, 98.0, 100.0,
            99.0, 100.0, 98.0, 99.0,     // Overlaps with first candle
            98.0, 110.0, 97.0, 108.0,
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_pattern_middle_not_doji() {
        let ab = AbandonedBaby::new();
        // Middle candle has large body
        let pattern = ab.detect_pattern(
            110.0, 112.0, 98.0, 100.0,
            95.0, 97.0, 90.0, 90.5,      // Large body, not doji
            98.0, 110.0, 98.0, 108.0,
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_doji_detection() {
        let ab = AbandonedBaby::new();
        // Doji: tiny body relative to range
        assert!(ab.is_doji(100.0, 105.0, 95.0, 100.2));
        // Not doji: large body
        assert!(!ab.is_doji(100.0, 105.0, 95.0, 104.0));
    }

    #[test]
    fn test_abandoned_baby_series() {
        let ab = AbandonedBaby::new();
        let open = vec![110.0, 96.5, 98.0];
        let high = vec![112.0, 97.0, 110.0];
        let low = vec![98.0, 96.0, 98.0];
        let close = vec![100.0, 96.5, 108.0];

        let result = ab.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 1.0); // Bullish Abandoned Baby
    }
}
