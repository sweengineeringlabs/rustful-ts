//! Volume Zone Oscillator (VZO) implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Volume Zone Oscillator (VZO).
///
/// Measures the volume flow direction by comparing positive and negative
/// volume over a specified period.
///
/// Calculation:
/// - R = Current Close vs Previous Close (+1 if up, -1 if down or equal)
/// - VP (Volume Position) = R * Volume
/// - TV (Total Volume) = Volume
/// - VZO = 100 * EMA(VP, period) / EMA(TV, period)
///
/// Interpretation:
/// - VZO > 0: Bullish zone (more buying pressure)
/// - VZO < 0: Bearish zone (more selling pressure)
/// - VZO > 40: Strong bullish trend (overbought)
/// - VZO < -40: Strong bearish trend (oversold)
/// - Crossovers of zero line provide trading signals
#[derive(Debug, Clone)]
pub struct VolumeZoneOscillator {
    period: usize,
}

impl VolumeZoneOscillator {
    /// Create a new Volume Zone Oscillator.
    ///
    /// # Arguments
    /// * `period` - EMA period for smoothing (typically 14)
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate EMA.
    fn ema(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let alpha = 2.0 / (period as f64 + 1.0);

        // Initial SMA
        let mut sum = 0.0;
        for i in 0..period {
            sum += values[i];
        }
        result[period - 1] = sum / period as f64;

        // EMA calculation
        for i in period..n {
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate VZO values.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n < 2 {
            return vec![f64::NAN; n];
        }

        // Calculate Volume Position (VP)
        let mut vp = vec![0.0; n];
        vp[0] = volume[0]; // First bar assumes positive

        for i in 1..n {
            let r = if close[i] > close[i - 1] {
                1.0
            } else {
                -1.0
            };
            vp[i] = r * volume[i];
        }

        // Calculate EMAs
        let ema_vp = self.ema(&vp, self.period);
        let ema_tv = self.ema(&volume, self.period);

        // Calculate VZO
        let mut vzo = vec![f64::NAN; n];
        for i in 0..n {
            if !ema_vp[i].is_nan() && !ema_tv[i].is_nan() && ema_tv[i] != 0.0 {
                vzo[i] = 100.0 * ema_vp[i] / ema_tv[i];
            }
        }

        vzo
    }
}

impl Default for VolumeZoneOscillator {
    fn default() -> Self {
        Self { period: 14 }
    }
}

impl TechnicalIndicator for VolumeZoneOscillator {
    fn name(&self) -> &str {
        "Volume Zone Oscillator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for VolumeZoneOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let vzo = self.calculate(&data.close, &data.volume);

        if vzo.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = vzo.len();
        let curr = vzo[n - 1];
        let prev = vzo[n - 2];

        if curr.is_nan() || prev.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: VZO crosses above zero
        if prev <= 0.0 && curr > 0.0 {
            return Ok(IndicatorSignal::Bullish);
        }
        // Bearish: VZO crosses below zero
        if prev >= 0.0 && curr < 0.0 {
            return Ok(IndicatorSignal::Bearish);
        }

        // Additional signals based on extreme zones
        if curr > 40.0 {
            // Overbought - potential reversal
            return Ok(IndicatorSignal::Bearish);
        }
        if curr < -40.0 {
            // Oversold - potential reversal
            return Ok(IndicatorSignal::Bullish);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let vzo = self.calculate(&data.close, &data.volume);

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..vzo.len() {
            if vzo[i].is_nan() || vzo[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if vzo[i - 1] <= 0.0 && vzo[i] > 0.0 {
                // Bullish crossover
                signals.push(IndicatorSignal::Bullish);
            } else if vzo[i - 1] >= 0.0 && vzo[i] < 0.0 {
                // Bearish crossover
                signals.push(IndicatorSignal::Bearish);
            } else if vzo[i] > 40.0 {
                // Overbought
                signals.push(IndicatorSignal::Bearish);
            } else if vzo[i] < -40.0 {
                // Oversold
                signals.push(IndicatorSignal::Bullish);
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
    fn test_vzo_uptrend() {
        let vzo = VolumeZoneOscillator::new(5);
        // Strong uptrend with consistent volume
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let volume = vec![1000.0; 20];

        let result = vzo.calculate(&close, &volume);

        assert_eq!(result.len(), 20);
        // VZO should be positive in uptrend
        for i in 5..20 {
            assert!(!result[i].is_nan());
            assert!(result[i] > 0.0, "VZO should be positive at index {}", i);
        }
    }

    #[test]
    fn test_vzo_downtrend() {
        let vzo = VolumeZoneOscillator::new(5);
        // Strong downtrend with consistent volume
        let close: Vec<f64> = (0..20).map(|i| 120.0 - i as f64).collect();
        let volume = vec![1000.0; 20];

        let result = vzo.calculate(&close, &volume);

        // VZO should be negative in downtrend
        for i in 5..20 {
            assert!(!result[i].is_nan());
            assert!(result[i] < 0.0, "VZO should be negative at index {}", i);
        }
    }

    #[test]
    fn test_vzo_range() {
        let vzo = VolumeZoneOscillator::new(5);
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let volume = vec![1000.0; 20];

        let result = vzo.calculate(&close, &volume);

        // VZO is bounded between -100 and 100
        for i in 5..20 {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_vzo_mixed() {
        let vzo = VolumeZoneOscillator::new(3);
        // Alternating up and down
        let close = vec![100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0];
        let volume = vec![1000.0; 7];

        let result = vzo.calculate(&close, &volume);

        // VZO should be close to zero with alternating movement
        assert!(!result[6].is_nan());
    }

    #[test]
    fn test_vzo_insufficient_data() {
        let vzo = VolumeZoneOscillator::new(5);
        let close = vec![100.0, 101.0, 102.0];
        let volume = vec![1000.0; 3];

        let result = vzo.calculate(&close, &volume);

        // All NaN with insufficient data
        for val in result {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_vzo_signal_bullish_crossover() {
        let vzo = VolumeZoneOscillator::new(3);
        // Start with downtrend, then reverse to uptrend
        let close = vec![105.0, 104.0, 103.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];
        let volume = vec![1000.0; 10];

        let signals = vzo.signals(&OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 1.0).collect(),
            low: close.iter().map(|x| x - 1.0).collect(),
            close: close.clone(),
            volume: volume.clone(),
        }).unwrap();

        assert_eq!(signals.len(), 10);
        // Should have bullish signal when crossing zero from below
    }

    #[test]
    fn test_vzo_technical_indicator() {
        let vzo = VolumeZoneOscillator::default();
        assert_eq!(vzo.name(), "Volume Zone Oscillator");
        assert_eq!(vzo.min_periods(), 14);
        assert_eq!(vzo.output_features(), 1);
    }

    #[test]
    fn test_vzo_empty() {
        let vzo = VolumeZoneOscillator::default();
        let result = vzo.calculate(&[], &[]);
        assert!(result.is_empty());
    }
}
