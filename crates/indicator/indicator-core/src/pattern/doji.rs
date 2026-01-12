//! Doji Candlestick Pattern
//!
//! Identifies Doji candlestick patterns indicating market indecision.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Type of Doji pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DojiType {
    /// Standard doji - open equals close
    Standard,
    /// Long-legged doji - long upper and lower shadows
    LongLegged,
    /// Dragonfly doji - long lower shadow, no upper shadow
    Dragonfly,
    /// Gravestone doji - long upper shadow, no lower shadow
    Gravestone,
    /// Four-price doji - open = high = low = close
    FourPrice,
}

/// Doji candlestick pattern indicator.
///
/// A Doji is formed when the open and close are virtually equal,
/// indicating indecision in the market. Different variations have
/// different implications based on the shadow lengths.
#[derive(Debug, Clone)]
pub struct Doji {
    /// Maximum body size as percentage of total range (default: 5%)
    body_threshold: f64,
    /// Minimum shadow ratio for long-legged doji
    long_shadow_ratio: f64,
}

impl Doji {
    /// Create a new Doji indicator with default thresholds.
    pub fn new() -> Self {
        Self {
            body_threshold: 0.05,
            long_shadow_ratio: 0.4,
        }
    }

    /// Create a Doji indicator with custom thresholds.
    ///
    /// # Arguments
    /// * `body_threshold` - Max body/range ratio to qualify as doji (0.0-1.0)
    /// * `long_shadow_ratio` - Min shadow/range ratio for long shadows
    pub fn with_thresholds(body_threshold: f64, long_shadow_ratio: f64) -> Self {
        Self {
            body_threshold,
            long_shadow_ratio,
        }
    }

    /// Detect doji pattern at a single bar.
    pub fn detect_single(&self, open: f64, high: f64, low: f64, close: f64) -> Option<DojiType> {
        let range = high - low;
        if range <= 0.0 {
            // Four-price doji
            return Some(DojiType::FourPrice);
        }

        let body = (close - open).abs();
        let body_ratio = body / range;

        if body_ratio > self.body_threshold {
            return None; // Not a doji
        }

        let upper_shadow = high - open.max(close);
        let lower_shadow = open.min(close) - low;
        let upper_ratio = upper_shadow / range;
        let lower_ratio = lower_shadow / range;

        // Classify doji type
        if upper_ratio < 0.1 && lower_ratio >= self.long_shadow_ratio {
            Some(DojiType::Dragonfly)
        } else if lower_ratio < 0.1 && upper_ratio >= self.long_shadow_ratio {
            Some(DojiType::Gravestone)
        } else if upper_ratio >= self.long_shadow_ratio && lower_ratio >= self.long_shadow_ratio {
            Some(DojiType::LongLegged)
        } else {
            Some(DojiType::Standard)
        }
    }

    /// Calculate doji patterns for all bars.
    ///
    /// Returns numeric values:
    /// - 1.0: Standard doji
    /// - 2.0: Long-legged doji
    /// - 3.0: Dragonfly doji (bullish)
    /// - 4.0: Gravestone doji (bearish)
    /// - 5.0: Four-price doji
    /// - NaN: No doji
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        for i in 0..n {
            if let Some(doji_type) = self.detect_single(open[i], high[i], low[i], close[i]) {
                result[i] = match doji_type {
                    DojiType::Standard => 1.0,
                    DojiType::LongLegged => 2.0,
                    DojiType::Dragonfly => 3.0,
                    DojiType::Gravestone => 4.0,
                    DojiType::FourPrice => 5.0,
                };
            }
        }

        result
    }

    /// Get doji types for all bars.
    pub fn doji_types(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<Option<DojiType>> {
        (0..close.len())
            .map(|i| self.detect_single(open[i], high[i], low[i], close[i]))
            .collect()
    }
}

impl Default for Doji {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for Doji {
    fn name(&self) -> &str {
        "Doji"
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

impl SignalIndicator for Doji {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = data.close.len();
        if let Some(doji_type) = self.detect_single(
            data.open[n - 1],
            data.high[n - 1],
            data.low[n - 1],
            data.close[n - 1],
        ) {
            match doji_type {
                DojiType::Dragonfly => return Ok(IndicatorSignal::Bullish),
                DojiType::Gravestone => return Ok(IndicatorSignal::Bearish),
                _ => return Ok(IndicatorSignal::Neutral), // Indecision
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let types = self.doji_types(&data.open, &data.high, &data.low, &data.close);
        let signals = types.iter().map(|t| {
            match t {
                Some(DojiType::Dragonfly) => IndicatorSignal::Bullish,
                Some(DojiType::Gravestone) => IndicatorSignal::Bearish,
                _ => IndicatorSignal::Neutral,
            }
        }).collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doji_standard() {
        let doji = Doji::new();
        // Open = Close (small body)
        let result = doji.detect_single(100.0, 105.0, 95.0, 100.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_doji_dragonfly() {
        let doji = Doji::new();
        // Long lower shadow, tiny upper shadow
        let result = doji.detect_single(100.0, 100.5, 90.0, 100.0);
        assert_eq!(result, Some(DojiType::Dragonfly));
    }

    #[test]
    fn test_doji_gravestone() {
        let doji = Doji::new();
        // Long upper shadow, tiny lower shadow
        let result = doji.detect_single(100.0, 110.0, 99.5, 100.0);
        assert_eq!(result, Some(DojiType::Gravestone));
    }

    #[test]
    fn test_not_doji() {
        let doji = Doji::new();
        // Large body - not a doji
        let result = doji.detect_single(100.0, 110.0, 95.0, 108.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_four_price_doji() {
        let doji = Doji::new();
        // All prices equal
        let result = doji.detect_single(100.0, 100.0, 100.0, 100.0);
        assert_eq!(result, Some(DojiType::FourPrice));
    }
}
