//! Elder's Thermometer applied to Volume implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Elder's Thermometer Applied to Volume.
///
/// Based on Dr. Alexander Elder's thermometer concept, this indicator measures
/// volume volatility by comparing current volume to a smoothed average.
///
/// Calculation:
/// - Volume Change = abs(Volume - Previous Volume)
/// - EMA of Volume Change over period
/// - Thermometer = Volume Change / EMA(Volume Change)
///
/// A high thermometer reading (> 2.0) suggests unusual volume activity,
/// which often precedes significant price movements.
///
/// Interpretation:
/// - Thermometer > 2.0: Unusually high volume, potential breakout or reversal
/// - Thermometer < 0.5: Unusually low volume, consolidation
/// - Rising thermometer with rising price: Bullish confirmation
/// - Rising thermometer with falling price: Bearish confirmation
#[derive(Debug, Clone)]
pub struct ElderThermometer {
    period: usize,
    threshold_high: f64,
    threshold_low: f64,
}

impl ElderThermometer {
    /// Create a new Elder's Thermometer indicator.
    ///
    /// # Arguments
    /// * `period` - EMA period for smoothing volume change (typically 22)
    /// * `threshold_high` - High threshold for unusual activity (typically 2.0)
    /// * `threshold_low` - Low threshold for consolidation (typically 0.5)
    pub fn new(period: usize, threshold_high: f64, threshold_low: f64) -> Self {
        Self {
            period,
            threshold_high,
            threshold_low,
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

        // Find first non-NaN value
        let mut first_valid = 0;
        while first_valid < n && values[first_valid].is_nan() {
            first_valid += 1;
        }

        if first_valid + period > n {
            return result;
        }

        // Initial SMA
        let mut sum = 0.0;
        for i in first_valid..(first_valid + period) {
            sum += values[i];
        }
        result[first_valid + period - 1] = sum / period as f64;

        // EMA calculation
        for i in (first_valid + period)..n {
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate Elder Thermometer values.
    /// Returns (Thermometer values, EMA of Volume Change for reference)
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();

        if n < 2 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate absolute volume change
        let mut vol_change = vec![f64::NAN; n];
        for i in 1..n {
            vol_change[i] = (volume[i] - volume[i - 1]).abs();
        }

        // Calculate EMA of volume change
        let ema_vol_change = self.ema(&vol_change, self.period);

        // Calculate thermometer
        let mut thermometer = vec![f64::NAN; n];
        for i in 0..n {
            if !ema_vol_change[i].is_nan() && !vol_change[i].is_nan() {
                if ema_vol_change[i] > 0.0 {
                    thermometer[i] = vol_change[i] / ema_vol_change[i];
                } else {
                    // When EMA of volume change is 0 (constant volume), thermometer is 0
                    thermometer[i] = 0.0;
                }
            }
        }

        (thermometer, ema_vol_change)
    }
}

impl Default for ElderThermometer {
    fn default() -> Self {
        Self {
            period: 22,
            threshold_high: 2.0,
            threshold_low: 0.5,
        }
    }
}

impl TechnicalIndicator for ElderThermometer {
    fn name(&self) -> &str {
        "Elder Thermometer"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let (thermometer, ema_vol_change) = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::dual(thermometer, ema_vol_change))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for ElderThermometer {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (thermometer, _) = self.calculate(&data.close, &data.volume);

        if thermometer.len() < 2 || data.close.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = thermometer.len();
        let curr_therm = thermometer[n - 1];
        let curr_close = data.close[n - 1];
        let prev_close = data.close[n - 2];

        if curr_therm.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // High volume with price movement
        if curr_therm > self.threshold_high {
            if curr_close > prev_close {
                // High volume with rising price - bullish
                return Ok(IndicatorSignal::Bullish);
            } else if curr_close < prev_close {
                // High volume with falling price - bearish
                return Ok(IndicatorSignal::Bearish);
            }
        }

        // Low volume consolidation - neutral
        if curr_therm < self.threshold_low {
            return Ok(IndicatorSignal::Neutral);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (thermometer, _) = self.calculate(&data.close, &data.volume);

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..thermometer.len() {
            if thermometer[i].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if thermometer[i] > self.threshold_high {
                if data.close[i] > data.close[i - 1] {
                    signals.push(IndicatorSignal::Bullish);
                } else if data.close[i] < data.close[i - 1] {
                    signals.push(IndicatorSignal::Bearish);
                } else {
                    signals.push(IndicatorSignal::Neutral);
                }
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
    fn test_elder_thermometer_basic() {
        let therm = ElderThermometer::new(5, 2.0, 0.5);
        let close: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        let volume: Vec<f64> = (0..20).map(|i| 1000.0 + (i as f64 * 100.0)).collect();

        let (thermometer, ema_vol) = therm.calculate(&close, &volume);

        assert_eq!(thermometer.len(), 20);
        assert_eq!(ema_vol.len(), 20);

        // First values should be NaN
        assert!(thermometer[0].is_nan());

        // Later values should be valid
        for i in 6..20 {
            assert!(!thermometer[i].is_nan());
            assert!(thermometer[i] > 0.0);
        }
    }

    #[test]
    fn test_elder_thermometer_spike() {
        let therm = ElderThermometer::new(5, 2.0, 0.5);
        // Normal volume then a sudden spike
        let mut volume = vec![1000.0; 15];
        volume[14] = 5000.0; // Volume spike

        let close: Vec<f64> = (0..15).map(|i| 100.0 + i as f64).collect();

        let (thermometer, _) = therm.calculate(&close, &volume);

        // The spike should cause a high thermometer reading
        assert!(!thermometer[14].is_nan());
        assert!(thermometer[14] > 2.0, "Volume spike should cause high thermometer");
    }

    #[test]
    fn test_elder_thermometer_constant_volume() {
        let therm = ElderThermometer::new(5, 2.0, 0.5);
        // Constant volume should give thermometer around 0
        let volume = vec![1000.0; 15];
        let close: Vec<f64> = (0..15).map(|i| 100.0 + i as f64).collect();

        let (thermometer, _) = therm.calculate(&close, &volume);

        // With constant volume, changes are 0, so thermometer should be 0
        for i in 6..15 {
            assert!(!thermometer[i].is_nan());
            // Thermometer should be very low or 0 with constant volume
            assert!(thermometer[i] < 0.1);
        }
    }

    #[test]
    fn test_elder_thermometer_signal_bullish() {
        let therm = ElderThermometer::new(5, 2.0, 0.5);
        // Create scenario with volume spike on up day
        let mut volume = vec![1000.0; 15];
        volume[14] = 5000.0;

        let mut close: Vec<f64> = (0..15).map(|i| 100.0 + i as f64 * 0.5).collect();
        close[14] = close[13] + 5.0; // Big up move

        let signal = therm.signal(&OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 1.0).collect(),
            low: close.iter().map(|x| x - 1.0).collect(),
            close: close.clone(),
            volume: volume.clone(),
        }).unwrap();

        assert_eq!(signal, IndicatorSignal::Bullish);
    }

    #[test]
    fn test_elder_thermometer_signal_bearish() {
        let therm = ElderThermometer::new(5, 2.0, 0.5);
        // Create scenario with volume spike on down day
        let mut volume = vec![1000.0; 15];
        volume[14] = 5000.0;

        let mut close: Vec<f64> = (0..15).map(|i| 100.0 + i as f64 * 0.5).collect();
        close[14] = close[13] - 5.0; // Big down move

        let signal = therm.signal(&OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 1.0).collect(),
            low: close.iter().map(|x| x - 1.0).collect(),
            close: close.clone(),
            volume: volume.clone(),
        }).unwrap();

        assert_eq!(signal, IndicatorSignal::Bearish);
    }

    #[test]
    fn test_elder_thermometer_technical_indicator() {
        let therm = ElderThermometer::default();
        assert_eq!(therm.name(), "Elder Thermometer");
        assert_eq!(therm.min_periods(), 23);
        assert_eq!(therm.output_features(), 2);
    }

    #[test]
    fn test_elder_thermometer_empty() {
        let therm = ElderThermometer::default();
        let (thermometer, ema) = therm.calculate(&[], &[]);
        assert!(thermometer.is_empty());
        assert!(ema.is_empty());
    }

    #[test]
    fn test_elder_thermometer_insufficient() {
        let therm = ElderThermometer::new(5, 2.0, 0.5);
        let (thermometer, _) = therm.calculate(&[100.0, 101.0], &[1000.0, 1100.0]);
        // All NaN with insufficient data
        for val in thermometer {
            assert!(val.is_nan());
        }
    }
}
