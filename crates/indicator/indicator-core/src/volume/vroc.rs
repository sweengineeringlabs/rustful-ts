//! Volume Rate of Change (VROC) implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Volume Rate of Change.
///
/// VROC measures the rate of change in volume over a period.
///
/// VROC = ((Current Volume - Volume n periods ago) / Volume n periods ago) * 100
///
/// - Positive: Volume increasing (confirms price move)
/// - Negative: Volume decreasing (weakening trend)
/// - Spikes: Potential trend changes or breakouts
#[derive(Debug, Clone)]
pub struct VROC {
    period: usize,
}

impl VROC {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate VROC values.
    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![f64::NAN; n];

        if n <= self.period {
            return result;
        }

        for i in self.period..n {
            let prev_vol = volume[i - self.period];
            if prev_vol > 0.0 {
                result[i] = ((volume[i] - prev_vol) / prev_vol) * 100.0;
            } else {
                result[i] = 0.0;
            }
        }

        result
    }
}

impl Default for VROC {
    fn default() -> Self {
        Self { period: 14 }
    }
}

impl TechnicalIndicator for VROC {
    fn name(&self) -> &str {
        "VROC"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() <= self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.volume.len(),
            });
        }

        let values = self.calculate(&data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for VROC {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.volume);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                // High positive VROC can indicate strong interest
                if last > 50.0 {
                    return Ok(IndicatorSignal::Bullish);
                } else if last < -50.0 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.volume);
        Ok(values
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    IndicatorSignal::Neutral
                } else if v > 50.0 {
                    IndicatorSignal::Bullish
                } else if v < -50.0 {
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
    fn test_vroc() {
        let vroc = VROC::new(3);
        let volume = vec![1000.0, 1100.0, 1200.0, 1500.0, 1000.0];

        let result = vroc.calculate(&volume);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        // (1500 - 1000) / 1000 * 100 = 50%
        assert!((result[3] - 50.0).abs() < 1e-10);
        // (1000 - 1100) / 1100 * 100 = -9.09%
        assert!((result[4] - (-9.090909090909092)).abs() < 1e-10);
    }

    #[test]
    fn test_vroc_increasing() {
        let vroc = VROC::new(2);
        // Steadily increasing volume
        let volume = vec![1000.0, 1200.0, 1500.0, 1900.0, 2500.0];

        let result = vroc.calculate(&volume);

        // All computed values should be positive
        assert!(result[2] > 0.0);
        assert!(result[3] > 0.0);
        assert!(result[4] > 0.0);
    }
}
