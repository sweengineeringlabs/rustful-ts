//! Volume Oscillator implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Volume Oscillator.
///
/// Measures the difference between two volume moving averages as a percentage.
///
/// Volume Oscillator = ((Short EMA - Long EMA) / Long EMA) * 100
///
/// - Positive: Short-term volume above long-term (increasing activity)
/// - Negative: Short-term volume below long-term (decreasing activity)
/// - Zero crossings: Potential trend changes
#[derive(Debug, Clone)]
pub struct VolumeOscillator {
    short_period: usize,
    long_period: usize,
}

impl VolumeOscillator {
    pub fn new(short_period: usize, long_period: usize) -> Self {
        Self {
            short_period,
            long_period,
        }
    }

    /// Calculate EMA.
    fn ema(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let alpha = 2.0 / (period as f64 + 1.0);

        // Calculate initial SMA
        let sum: f64 = values[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        // Apply EMA
        for i in period..n {
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate Volume Oscillator values.
    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![f64::NAN; n];

        if n < self.long_period {
            return result;
        }

        let short_ema = self.ema(volume, self.short_period);
        let long_ema = self.ema(volume, self.long_period);

        for i in 0..n {
            if !short_ema[i].is_nan() && !long_ema[i].is_nan() && long_ema[i] > 0.0 {
                result[i] = ((short_ema[i] - long_ema[i]) / long_ema[i]) * 100.0;
            }
        }

        result
    }
}

impl Default for VolumeOscillator {
    fn default() -> Self {
        Self {
            short_period: 5,
            long_period: 10,
        }
    }
}

impl TechnicalIndicator for VolumeOscillator {
    fn name(&self) -> &str {
        "Volume Oscillator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.volume.len() < self.long_period {
            return Err(IndicatorError::InsufficientData {
                required: self.long_period,
                got: data.volume.len(),
            });
        }

        let values = self.calculate(&data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.long_period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for VolumeOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.volume);

        if values.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = values.len();
        let current = values[n - 1];
        let previous = values[n - 2];

        if current.is_nan() || previous.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: crossing above zero
        if previous <= 0.0 && current > 0.0 {
            return Ok(IndicatorSignal::Bullish);
        }
        // Bearish: crossing below zero
        if previous >= 0.0 && current < 0.0 {
            return Ok(IndicatorSignal::Bearish);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.volume);

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..values.len() {
            if values[i].is_nan() || values[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if values[i - 1] <= 0.0 && values[i] > 0.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if values[i - 1] >= 0.0 && values[i] < 0.0 {
                signals.push(IndicatorSignal::Bearish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_oscillator() {
        let vo = VolumeOscillator::new(3, 5);
        let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0];

        let result = vo.calculate(&volume);

        assert_eq!(result.len(), 7);
        // First valid value should appear at long_period - 1
        assert!(!result[4].is_nan());
        // With increasing volume, short EMA > long EMA, so oscillator > 0
        assert!(result[6] > 0.0);
    }

    #[test]
    fn test_volume_oscillator_decreasing() {
        let vo = VolumeOscillator::new(3, 5);
        // Decreasing volume
        let volume = vec![1600.0, 1500.0, 1400.0, 1300.0, 1200.0, 1100.0, 1000.0];

        let result = vo.calculate(&volume);

        // With decreasing volume, short EMA < long EMA, so oscillator < 0
        assert!(result[6] < 0.0);
    }
}
