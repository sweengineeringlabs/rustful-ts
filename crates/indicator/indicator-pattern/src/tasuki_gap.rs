//! Tasuki Gap Candlestick Pattern
//!
//! Identifies three-candle continuation patterns with gaps.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of Tasuki gap pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TasukiGapType {
    /// Upside Tasuki Gap - bullish continuation
    Upside,
    /// Downside Tasuki Gap - bearish continuation
    Downside,
}

/// Tasuki Gap pattern indicator.
///
/// The Tasuki Gap is a three-candle continuation pattern:
///
/// Upside Tasuki Gap (bullish continuation):
/// 1. First candle: Bullish candle
/// 2. Second candle: Bullish candle that gaps up (open > first close)
/// 3. Third candle: Bearish candle that opens within second body
///    and closes within the gap (but doesn't close the gap completely)
///
/// Downside Tasuki Gap (bearish continuation):
/// 1. First candle: Bearish candle
/// 2. Second candle: Bearish candle that gaps down (open < first close)
/// 3. Third candle: Bullish candle that opens within second body
///    and closes within the gap (but doesn't close the gap completely)
///
/// The pattern suggests the trend will continue as the gap remains open.
#[derive(Debug, Clone)]
pub struct TasukiGap {
    /// Minimum gap size as percentage of average candle range (default: 0.5%)
    min_gap_ratio: f64,
    /// Minimum body ratio for trend candles (default: 0.5)
    min_body_ratio: f64,
}

impl TasukiGap {
    /// Create a new TasukiGap indicator with default parameters.
    pub fn new() -> Self {
        Self {
            min_gap_ratio: 0.005,
            min_body_ratio: 0.5,
        }
    }

    /// Create a TasukiGap indicator with custom parameters.
    pub fn with_params(min_gap_ratio: f64, min_body_ratio: f64) -> Self {
        Self {
            min_gap_ratio,
            min_body_ratio,
        }
    }

    /// Check if a candle has valid body size.
    fn has_valid_body(&self, open: f64, high: f64, low: f64, close: f64) -> bool {
        let range = high - low;
        if range <= 0.0 {
            return false;
        }
        let body = (close - open).abs();
        body / range >= self.min_body_ratio
    }

    /// Detect tasuki gap pattern at index (requires 3 candles ending at this index).
    fn detect_pattern(
        &self,
        first_open: f64, first_high: f64, first_low: f64, first_close: f64,
        second_open: f64, second_high: f64, second_low: f64, second_close: f64,
        third_open: f64, _third_high: f64, _third_low: f64, third_close: f64,
    ) -> Option<TasukiGapType> {
        let first_is_bullish = first_close > first_open;
        let second_is_bullish = second_close > second_open;
        let third_is_bullish = third_close > third_open;

        // Calculate average price for gap ratio
        let avg_price = (first_close + second_open) / 2.0;

        // Upside Tasuki Gap
        if first_is_bullish && second_is_bullish && !third_is_bullish {
            // Check for gap up (second opens above first close)
            let gap = second_open - first_close;
            if gap > 0.0 && gap / avg_price >= self.min_gap_ratio {
                // Both trend candles should have decent bodies
                if self.has_valid_body(first_open, first_high, first_low, first_close)
                    && self.has_valid_body(second_open, second_high, second_low, second_close)
                {
                    // Third candle opens within second body
                    let second_body_low = second_open;
                    let second_body_high = second_close;
                    if third_open >= second_body_low && third_open <= second_body_high {
                        // Third candle closes within the gap (but gap not fully closed)
                        if third_close <= second_open && third_close > first_close {
                            return Some(TasukiGapType::Upside);
                        }
                    }
                }
            }
        }

        // Downside Tasuki Gap
        if !first_is_bullish && !second_is_bullish && third_is_bullish {
            // Check for gap down (second opens below first close)
            let gap = first_close - second_open;
            if gap > 0.0 && gap / avg_price >= self.min_gap_ratio {
                // Both trend candles should have decent bodies
                if self.has_valid_body(first_open, first_high, first_low, first_close)
                    && self.has_valid_body(second_open, second_high, second_low, second_close)
                {
                    // Third candle opens within second body
                    let second_body_low = second_close;
                    let second_body_high = second_open;
                    if third_open >= second_body_low && third_open <= second_body_high {
                        // Third candle closes within the gap (but gap not fully closed)
                        if third_close >= second_open && third_close < first_close {
                            return Some(TasukiGapType::Downside);
                        }
                    }
                }
            }
        }

        None
    }

    /// Calculate tasuki gap patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Upside Tasuki Gap (bullish continuation)
    /// - -1.0: Downside Tasuki Gap (bearish continuation)
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
                    TasukiGapType::Upside => 1.0,
                    TasukiGapType::Downside => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<TasukiGapType>> {
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

impl Default for TasukiGap {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TasukiGap {
    fn name(&self) -> &str {
        "TasukiGap"
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

impl SignalIndicator for TasukiGap {
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
                TasukiGapType::Upside => Ok(IndicatorSignal::Bullish),
                TasukiGapType::Downside => Ok(IndicatorSignal::Bearish),
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
    fn test_upside_tasuki_gap() {
        let tg = TasukiGap::with_params(0.001, 0.5);
        // First: Bullish (100->106)
        // Second: Bullish gap up (108->114, gap from 106 to 108)
        // Third: Bearish opens in second body (112), closes in gap (107) but above first close (106)
        let pattern = tg.detect_pattern(
            100.0, 107.0, 99.0, 106.0,   // Bullish
            108.0, 115.0, 107.0, 114.0,  // Bullish gap up
            112.0, 113.0, 106.5, 107.0,  // Bearish, closes in gap
        );
        assert_eq!(pattern, Some(TasukiGapType::Upside));
    }

    #[test]
    fn test_downside_tasuki_gap() {
        let tg = TasukiGap::with_params(0.001, 0.5);
        // First: Bearish (110->104)
        // Second: Bearish gap down (102->96, gap from 104 to 102)
        // Third: Bullish opens in second body (100), closes in gap (103) but below first close (104)
        let pattern = tg.detect_pattern(
            110.0, 111.0, 103.0, 104.0,  // Bearish
            102.0, 103.0, 95.0, 96.0,    // Bearish gap down
            100.0, 104.0, 99.0, 103.0,   // Bullish, closes in gap
        );
        assert_eq!(pattern, Some(TasukiGapType::Downside));
    }

    #[test]
    fn test_no_pattern_gap_closed() {
        let tg = TasukiGap::with_params(0.001, 0.5);
        // Third candle closes the gap completely
        let pattern = tg.detect_pattern(
            100.0, 107.0, 99.0, 106.0,
            108.0, 115.0, 107.0, 114.0,
            112.0, 113.0, 104.0, 105.0,  // Closes below first close (gap closed)
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_pattern_no_gap() {
        let tg = TasukiGap::with_params(0.001, 0.5);
        // No gap between first and second
        let pattern = tg.detect_pattern(
            100.0, 107.0, 99.0, 106.0,
            105.0, 112.0, 104.0, 111.0,  // Opens below first close (no gap)
            109.0, 110.0, 106.0, 107.0,
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_tasuki_gap_series() {
        let tg = TasukiGap::with_params(0.001, 0.5);
        let open = vec![100.0, 108.0, 112.0];
        let high = vec![107.0, 115.0, 113.0];
        let low = vec![99.0, 107.0, 106.5];
        let close = vec![106.0, 114.0, 107.0];

        let result = tg.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 3);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 1.0); // Upside Tasuki Gap
    }
}
