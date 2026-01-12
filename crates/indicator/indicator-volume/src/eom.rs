//! Ease of Movement (EOM) implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Ease of Movement.
///
/// EOM relates price change to volume, indicating how easily
/// price moves on volume.
///
/// Distance Moved = ((High + Low) / 2) - ((Previous High + Previous Low) / 2)
/// Box Ratio = (Volume / divisor) / (High - Low)
/// EOM = Distance Moved / Box Ratio
///
/// High positive values: Price moving up easily
/// High negative values: Price moving down easily
/// Values near zero: Price not moving easily
#[derive(Debug, Clone)]
pub struct EaseOfMovement {
    period: usize,
    divisor: f64,
}

impl EaseOfMovement {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            divisor: 1_000_000.0, // Standard divisor to scale volume
        }
    }

    pub fn with_divisor(period: usize, divisor: f64) -> Self {
        Self { period, divisor }
    }

    /// Calculate raw EOM values.
    pub fn calculate_raw(&self, high: &[f64], low: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = high.len();

        if n < 2 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in 1..n {
            let dm = (high[i] + low[i]) / 2.0 - (high[i - 1] + low[i - 1]) / 2.0;
            let hl_range = high[i] - low[i];

            if hl_range > 0.0 {
                let box_ratio = (volume[i] / self.divisor) / hl_range;
                result[i] = if box_ratio > 0.0 { dm / box_ratio } else { 0.0 };
            } else {
                result[i] = 0.0;
            }
        }

        result
    }

    /// Calculate smoothed EOM (SMA).
    pub fn calculate(&self, high: &[f64], low: &[f64], volume: &[f64]) -> Vec<f64> {
        let raw = self.calculate_raw(high, low, volume);
        self.sma(&raw, self.period)
    }

    /// Simple moving average.
    fn sma(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        if n < period + 1 {
            return result;
        }

        // Start from index 1 (first valid raw value)
        for i in period..n {
            let mut sum = 0.0;
            let mut count = 0;

            for j in (i + 1 - period)..=i {
                if !values[j].is_nan() {
                    sum += values[j];
                    count += 1;
                }
            }

            if count == period {
                result[i] = sum / period as f64;
            }
        }

        result
    }
}

impl Default for EaseOfMovement {
    fn default() -> Self {
        Self {
            period: 14,
            divisor: 1_000_000.0,
        }
    }
}

impl TechnicalIndicator for EaseOfMovement {
    fn name(&self) -> &str {
        "Ease of Movement"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for EaseOfMovement {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.volume);

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
        let values = self.calculate(&data.high, &data.low, &data.volume);
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
    fn test_eom_raw() {
        let eom = EaseOfMovement::new(3);
        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![95.0, 96.0, 97.0, 98.0, 99.0];
        let volume = vec![1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0];

        let result = eom.calculate_raw(&high, &low, &volume);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        // Uptrend: EOM should be positive
        assert!(result[1] > 0.0);
    }

    #[test]
    fn test_eom_smoothed() {
        let eom = EaseOfMovement::new(3);
        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![95.0, 96.0, 97.0, 98.0, 99.0];
        let volume = vec![1000000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0];

        let result = eom.calculate(&high, &low, &volume);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert!(!result[3].is_nan());
    }
}
