//! ZigZag Pattern Indicator
//!
//! Identifies significant price swings by filtering out minor price movements.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// ZigZag indicator for identifying significant price swings.
///
/// Filters out minor price movements and highlights major trend reversals.
/// The threshold parameter determines the minimum percentage change required
/// for a new pivot point.
#[derive(Debug, Clone)]
pub struct ZigZag {
    /// Minimum percentage change to register a new pivot (e.g., 5.0 = 5%)
    threshold: f64,
}

impl ZigZag {
    /// Create a new ZigZag indicator with the specified threshold percentage.
    ///
    /// # Arguments
    /// * `threshold` - Minimum percentage change for a new pivot (e.g., 5.0 = 5%)
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Calculate ZigZag values.
    ///
    /// Returns a vector where:
    /// - Positive values indicate swing highs
    /// - Negative values indicate swing lows
    /// - NaN values indicate no pivot at that index
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < 2 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];
        let threshold_mult = self.threshold / 100.0;

        // Track last confirmed pivot
        let mut last_pivot_idx = 0;
        let mut last_pivot_value = (high[0] + low[0]) / 2.0;
        let mut last_pivot_is_high = true;

        // Initialize with first bar
        result[0] = last_pivot_value;

        for i in 1..n {
            let current_high = high[i];
            let current_low = low[i];

            if last_pivot_is_high {
                // Looking for lower low or higher high
                let change_down = (last_pivot_value - current_low) / last_pivot_value;
                let change_up = (current_high - last_pivot_value) / last_pivot_value;

                if change_down >= threshold_mult {
                    // New swing low confirmed
                    result[i] = -current_low;
                    last_pivot_idx = i;
                    last_pivot_value = current_low;
                    last_pivot_is_high = false;
                } else if change_up > 0.0 && current_high > last_pivot_value {
                    // Update the high pivot
                    result[last_pivot_idx] = current_high;
                    last_pivot_value = current_high;
                }
            } else {
                // Looking for higher high or lower low
                let change_up = (current_high - last_pivot_value) / last_pivot_value;
                let change_down = (last_pivot_value - current_low) / last_pivot_value;

                if change_up >= threshold_mult {
                    // New swing high confirmed
                    result[i] = current_high;
                    last_pivot_idx = i;
                    last_pivot_value = current_high;
                    last_pivot_is_high = true;
                } else if change_down > 0.0 && current_low < last_pivot_value {
                    // Update the low pivot
                    result[last_pivot_idx] = -current_low;
                    last_pivot_value = current_low;
                }
            }
        }

        result
    }

    /// Get pivot points as (index, value, is_high) tuples.
    pub fn pivots(&self, high: &[f64], low: &[f64]) -> Vec<(usize, f64, bool)> {
        let zigzag = self.calculate(high, low);
        let mut pivots = Vec::new();

        for (i, &val) in zigzag.iter().enumerate() {
            if !val.is_nan() {
                let is_high = val > 0.0;
                let price = val.abs();
                pivots.push((i, price, is_high));
            }
        }

        pivots
    }
}

impl TechnicalIndicator for ZigZag {
    fn name(&self) -> &str {
        "ZigZag"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < 2 {
            return Err(IndicatorError::InsufficientData {
                required: 2,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        2
    }
}

impl SignalIndicator for ZigZag {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low);

        // Find the last non-NaN value
        for val in values.iter().rev() {
            if !val.is_nan() {
                return if *val > 0.0 {
                    // Last pivot was a high - potential bearish signal
                    Ok(IndicatorSignal::Bearish)
                } else {
                    // Last pivot was a low - potential bullish signal
                    Ok(IndicatorSignal::Bullish)
                };
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low);
        let signals = values.iter().map(|&val| {
            if val.is_nan() {
                IndicatorSignal::Neutral
            } else if val > 0.0 {
                IndicatorSignal::Bearish // Swing high = potential reversal down
            } else {
                IndicatorSignal::Bullish // Swing low = potential reversal up
            }
        }).collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_basic() {
        let zigzag = ZigZag::new(5.0);
        let high = vec![100.0, 105.0, 110.0, 108.0, 103.0, 100.0, 105.0, 115.0];
        let low = vec![95.0, 100.0, 105.0, 103.0, 98.0, 95.0, 100.0, 110.0];

        let result = zigzag.calculate(&high, &low);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_zigzag_pivots() {
        let zigzag = ZigZag::new(10.0);
        let high = vec![100.0, 110.0, 105.0, 95.0, 90.0, 100.0, 115.0];
        let low = vec![90.0, 100.0, 95.0, 85.0, 80.0, 90.0, 105.0];

        let pivots = zigzag.pivots(&high, &low);
        assert!(!pivots.is_empty());
    }
}
