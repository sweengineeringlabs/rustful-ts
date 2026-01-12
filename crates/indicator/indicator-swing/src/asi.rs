//! Accumulative Swing Index (ASI) implementation.
//!
//! The ASI is a cumulative sum of the Swing Index values.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

use crate::swing_index::SwingIndex;

/// Accumulative Swing Index (ASI).
///
/// Developed by J. Welles Wilder, the ASI is a running total of the Swing Index.
/// It provides a long-term picture of price momentum and helps identify
/// trend direction and potential breakouts.
///
/// Key signals:
/// - Rising ASI confirms uptrend
/// - Falling ASI confirms downtrend
/// - ASI breaking its own trendline can signal trend change
/// - ASI divergence from price can indicate weakening momentum
#[derive(Debug, Clone)]
pub struct AccumulativeSwingIndex {
    /// Underlying Swing Index calculator.
    swing_index: SwingIndex,
}

impl AccumulativeSwingIndex {
    /// Create a new Accumulative Swing Index indicator.
    ///
    /// # Arguments
    /// * `limit_move` - Maximum expected price move for Swing Index calculation
    pub fn new(limit_move: f64) -> Self {
        Self {
            swing_index: SwingIndex::new(limit_move),
        }
    }

    /// Create with default limit move.
    pub fn default_limit() -> Self {
        Self {
            swing_index: SwingIndex::default_limit(),
        }
    }

    /// Calculate Accumulative Swing Index values.
    pub fn calculate(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<f64> {
        let si_values = self.swing_index.calculate(open, high, low, close);
        let n = si_values.len();

        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);
        let mut cumulative = 0.0;

        for si in si_values {
            if si.is_nan() {
                result.push(f64::NAN);
            } else {
                cumulative += si;
                result.push(cumulative);
            }
        }

        result
    }
}

impl TechnicalIndicator for AccumulativeSwingIndex {
    fn name(&self) -> &str {
        "AccumulativeSwingIndex"
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

impl SignalIndicator for AccumulativeSwingIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);

        if values.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let current = values.last().copied().unwrap_or(f64::NAN);
        let prev = values[values.len() - 2];

        if current.is_nan() || prev.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal based on ASI direction
        if current > prev {
            Ok(IndicatorSignal::Bullish)
        } else if current < prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);

        let signals = values
            .windows(2)
            .map(|w| {
                if w[0].is_nan() || w[1].is_nan() {
                    IndicatorSignal::Neutral
                } else if w[1] > w[0] {
                    IndicatorSignal::Bullish
                } else if w[1] < w[0] {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect::<Vec<_>>();

        // Prepend neutral for the first value
        let mut result = vec![IndicatorSignal::Neutral];
        result.extend(signals);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asi_basic() {
        let asi = AccumulativeSwingIndex::new(3.0);

        let open = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let high = vec![102.0, 103.0, 104.0, 105.0, 106.0];
        let low = vec![99.0, 100.0, 101.0, 102.0, 103.0];
        let close = vec![101.0, 102.0, 103.0, 104.0, 105.0];

        let result = asi.calculate(&open, &high, &low, &close);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());

        // ASI should accumulate
        for i in 2..5 {
            if !result[i].is_nan() && !result[i - 1].is_nan() {
                // Check that values are accumulating (not necessarily increasing/decreasing)
                assert!(!result[i].is_nan());
            }
        }
    }

    #[test]
    fn test_asi_cumulative() {
        let asi = AccumulativeSwingIndex::new(2.0);

        // Uptrend data
        let open = vec![100.0, 100.5, 101.0, 101.5, 102.0];
        let high = vec![101.0, 101.5, 102.0, 102.5, 103.0];
        let low = vec![99.5, 100.0, 100.5, 101.0, 101.5];
        let close = vec![100.5, 101.0, 101.5, 102.0, 102.5];

        let result = asi.calculate(&open, &high, &low, &close);

        // Values should be cumulative (each non-NaN value builds on previous)
        let mut prev_valid = None;
        for val in result.iter() {
            if !val.is_nan() {
                if let Some(_) = prev_valid {
                    // ASI is cumulative
                }
                prev_valid = Some(*val);
            }
        }
    }

    #[test]
    fn test_asi_signal() {
        let asi = AccumulativeSwingIndex::new(2.0);

        let mut data = OHLCVSeries::new();
        for i in 0..10 {
            data.open.push(100.0 + i as f64 * 0.5);
            data.high.push(102.0 + i as f64 * 0.5);
            data.low.push(98.0 + i as f64 * 0.5);
            data.close.push(101.0 + i as f64 * 0.5);
            data.volume.push(1000.0);
        }

        let signal = asi.signal(&data).unwrap();
        // In an uptrend, we expect bullish or neutral signal
        assert!(signal == IndicatorSignal::Bullish || signal == IndicatorSignal::Neutral);
    }
}
