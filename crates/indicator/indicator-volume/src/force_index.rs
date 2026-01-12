//! Force Index implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Force Index.
///
/// The Force Index measures the force (power) behind price movements
/// by combining price change and volume.
///
/// Force Index = (Close - Previous Close) * Volume
///
/// A smoothed version (EMA) is typically used for analysis.
/// - Positive: Buying pressure
/// - Negative: Selling pressure
#[derive(Debug, Clone)]
pub struct ForceIndex {
    period: usize,
}

impl ForceIndex {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate raw Force Index values.
    pub fn calculate_raw(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n < 2 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in 1..n {
            result[i] = (close[i] - close[i - 1]) * volume[i];
        }

        result
    }

    /// Calculate smoothed Force Index with EMA.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let raw = self.calculate_raw(close, volume);
        let n = raw.len();

        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];
        let alpha = 2.0 / (self.period as f64 + 1.0);

        // First valid raw value is at index 1, need period values for first EMA
        let first_valid = self.period;

        // Calculate initial SMA for EMA seed
        let mut sum = 0.0;
        for i in 1..=self.period {
            if !raw[i].is_nan() {
                sum += raw[i];
            }
        }
        result[first_valid] = sum / self.period as f64;

        // Apply EMA
        for i in (first_valid + 1)..n {
            if !raw[i].is_nan() {
                result[i] = alpha * raw[i] + (1.0 - alpha) * result[i - 1];
            } else {
                result[i] = result[i - 1];
            }
        }

        result
    }
}

impl Default for ForceIndex {
    fn default() -> Self {
        Self { period: 13 }
    }
}

impl TechnicalIndicator for ForceIndex {
    fn name(&self) -> &str {
        "Force Index"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for ForceIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close, &data.volume);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last > 0.0 {
                    return Ok(IndicatorSignal::Bullish);
                } else if last < 0.0 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close, &data.volume);
        Ok(values
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    IndicatorSignal::Neutral
                } else if v > 0.0 {
                    IndicatorSignal::Bullish
                } else if v < 0.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_force_index_raw() {
        let fi = ForceIndex::new(2);
        let close = vec![100.0, 102.0, 101.0, 104.0, 103.0];
        let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 1400.0];

        let result = fi.calculate_raw(&close, &volume);

        assert!(result[0].is_nan());
        // Day 2: (102-100) * 1500 = 3000
        assert!((result[1] - 3000.0).abs() < 1e-10);
        // Day 3: (101-102) * 1200 = -1200
        assert!((result[2] - (-1200.0)).abs() < 1e-10);
    }

    #[test]
    fn test_force_index_smoothed() {
        let fi = ForceIndex::new(2);
        let close = vec![100.0, 102.0, 101.0, 104.0, 103.0];
        let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 1400.0];

        let result = fi.calculate(&close, &volume);

        assert_eq!(result.len(), 5);
        // First valid smoothed value should appear after period
        assert!(!result[2].is_nan());
    }
}
