//! Kicking Candlestick Pattern
//!
//! Identifies two-candle pattern with opposing Marubozu candles.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of kicking pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KickingType {
    /// Bullish Kicking - strong bullish reversal
    Bullish,
    /// Bearish Kicking - strong bearish reversal
    Bearish,
}

/// Kicking pattern indicator.
///
/// The Kicking pattern is one of the most powerful reversal patterns,
/// consisting of two opposing Marubozu candles with a gap:
///
/// Bullish Kicking:
/// - First candle: Bearish Marubozu (opens at high, closes at low)
/// - Second candle: Bullish Marubozu that gaps up (opens at low above first's high)
///
/// Bearish Kicking:
/// - First candle: Bullish Marubozu (opens at low, closes at high)
/// - Second candle: Bearish Marubozu that gaps down (opens at high below first's low)
///
/// The pattern shows an extreme shift in market sentiment.
#[derive(Debug, Clone)]
pub struct Kicking {
    /// Maximum shadow tolerance as percentage of range (default: 0.02 = 2%)
    shadow_tolerance: f64,
    /// Minimum body to range ratio for Marubozu (default: 0.95)
    min_body_ratio: f64,
}

impl Kicking {
    /// Create a new Kicking indicator with default parameters.
    pub fn new() -> Self {
        Self {
            shadow_tolerance: 0.02,
            min_body_ratio: 0.95,
        }
    }

    /// Create a Kicking indicator with custom parameters.
    pub fn with_params(shadow_tolerance: f64, min_body_ratio: f64) -> Self {
        Self {
            shadow_tolerance,
            min_body_ratio,
        }
    }

    /// Check if a candle is a Marubozu (no shadows).
    fn is_marubozu(&self, open: f64, high: f64, low: f64, close: f64) -> Option<bool> {
        let range = high - low;
        if range <= 0.0 {
            return None;
        }

        let body = (close - open).abs();
        let body_ratio = body / range;

        // Body must be nearly the entire range
        if body_ratio < self.min_body_ratio {
            return None;
        }

        let is_bullish = close > open;
        let upper_shadow = if is_bullish { high - close } else { high - open };
        let lower_shadow = if is_bullish { open - low } else { close - low };

        let upper_ratio = upper_shadow / range;
        let lower_ratio = lower_shadow / range;

        // Both shadows must be minimal
        if upper_ratio <= self.shadow_tolerance && lower_ratio <= self.shadow_tolerance {
            Some(is_bullish)
        } else {
            None
        }
    }

    /// Detect kicking pattern between two consecutive candles.
    fn detect_pattern(
        &self,
        first_open: f64, first_high: f64, first_low: f64, first_close: f64,
        second_open: f64, second_high: f64, second_low: f64, second_close: f64,
    ) -> Option<KickingType> {
        // Both candles must be Marubozu
        let first_bullish = self.is_marubozu(first_open, first_high, first_low, first_close)?;
        let second_bullish = self.is_marubozu(second_open, second_high, second_low, second_close)?;

        // Must be opposing directions
        if first_bullish == second_bullish {
            return None;
        }

        // Bullish Kicking: First bearish Marubozu, second bullish Marubozu with gap up
        if !first_bullish && second_bullish {
            // Second must gap up above first's high
            if second_low > first_high {
                return Some(KickingType::Bullish);
            }
        }

        // Bearish Kicking: First bullish Marubozu, second bearish Marubozu with gap down
        if first_bullish && !second_bullish {
            // Second must gap down below first's low
            if second_high < first_low {
                return Some(KickingType::Bearish);
            }
        }

        None
    }

    /// Calculate kicking patterns for all bars.
    ///
    /// Returns:
    /// - 1.0: Bullish Kicking
    /// - -1.0: Bearish Kicking
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
                    KickingType::Bullish => 1.0,
                    KickingType::Bearish => -1.0,
                };
            }
        }

        result
    }

    /// Get detailed pattern types.
    pub fn pattern_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<KickingType>> {
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

impl Default for Kicking {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for Kicking {
    fn name(&self) -> &str {
        "Kicking"
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

impl SignalIndicator for Kicking {
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
                KickingType::Bullish => Ok(IndicatorSignal::Bullish),
                KickingType::Bearish => Ok(IndicatorSignal::Bearish),
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
    fn test_bullish_kicking() {
        let kicking = Kicking::new();
        // First: Bearish Marubozu (open=high=110, close=low=100)
        // Second: Bullish Marubozu that gaps up (low=112 > first high=110)
        let pattern = kicking.detect_pattern(
            110.0, 110.0, 100.0, 100.0,  // Bearish Marubozu
            112.0, 122.0, 112.0, 122.0,  // Bullish Marubozu with gap up
        );
        assert_eq!(pattern, Some(KickingType::Bullish));
    }

    #[test]
    fn test_bearish_kicking() {
        let kicking = Kicking::new();
        // First: Bullish Marubozu (open=low=100, close=high=110)
        // Second: Bearish Marubozu that gaps down (high=98 < first low=100)
        let pattern = kicking.detect_pattern(
            100.0, 110.0, 100.0, 110.0,  // Bullish Marubozu
            98.0, 98.0, 88.0, 88.0,      // Bearish Marubozu with gap down
        );
        assert_eq!(pattern, Some(KickingType::Bearish));
    }

    #[test]
    fn test_no_pattern_no_gap() {
        let kicking = Kicking::new();
        // Opposing Marubozu but no gap
        let pattern = kicking.detect_pattern(
            110.0, 110.0, 100.0, 100.0,
            105.0, 115.0, 105.0, 115.0,  // No gap (low=105 < first high=110)
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_pattern_not_marubozu() {
        let kicking = Kicking::new();
        // First candle has shadows (not Marubozu)
        let pattern = kicking.detect_pattern(
            108.0, 112.0, 98.0, 100.0,   // Has shadows
            112.0, 122.0, 112.0, 122.0,
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_no_pattern_same_direction() {
        let kicking = Kicking::new();
        // Both bullish Marubozu
        let pattern = kicking.detect_pattern(
            100.0, 110.0, 100.0, 110.0,
            112.0, 122.0, 112.0, 122.0,  // Also bullish
        );
        assert!(pattern.is_none());
    }

    #[test]
    fn test_is_marubozu() {
        let kicking = Kicking::new();
        // Perfect bullish Marubozu
        assert_eq!(kicking.is_marubozu(100.0, 110.0, 100.0, 110.0), Some(true));
        // Perfect bearish Marubozu
        assert_eq!(kicking.is_marubozu(110.0, 110.0, 100.0, 100.0), Some(false));
        // Not Marubozu (has shadows)
        assert!(kicking.is_marubozu(102.0, 112.0, 98.0, 108.0).is_none());
    }

    #[test]
    fn test_kicking_series() {
        let kicking = Kicking::new();
        let open = vec![110.0, 112.0];
        let high = vec![110.0, 122.0];
        let low = vec![100.0, 112.0];
        let close = vec![100.0, 122.0];

        let result = kicking.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 2);
        assert!(result[0].is_nan());
        assert_eq!(result[1], 1.0); // Bullish Kicking
    }
}
