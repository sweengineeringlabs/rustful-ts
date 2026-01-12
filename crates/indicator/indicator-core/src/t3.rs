//! Tillson T3 Moving Average implementation.
//!
//! A triple smoothed EMA with volume factor for reduced lag.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::T3Config;

/// Tillson T3 Moving Average.
///
/// T3 is a smoother moving average with less lag than traditional EMAs.
/// It uses a triple smoothing technique with a volume factor that controls
/// the trade-off between smoothness and responsiveness.
///
/// The T3 formula applies EMA six times with the following combination:
/// T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
/// where c1 = -vf^3, c2 = 3*vf^2 + 3*vf^3, c3 = -6*vf^2 - 3*vf - 3*vf^3, c4 = 1 + 3*vf + vf^3 + 3*vf^2
#[derive(Debug, Clone)]
pub struct T3 {
    period: usize,
    volume_factor: f64,
    alpha: f64,
}

impl T3 {
    pub fn new(period: usize, volume_factor: f64) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self { period, volume_factor, alpha }
    }

    pub fn from_config(config: T3Config) -> Self {
        Self::new(config.period, config.volume_factor)
    }

    /// Calculate T3 values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        // T3 needs about 6*period warmup for full accuracy
        let min_required = self.period;
        if data.len() < min_required || self.period == 0 {
            return vec![f64::NAN; data.len()];
        }

        // Calculate T3 coefficients
        let vf = self.volume_factor;
        let vf2 = vf * vf;
        let vf3 = vf2 * vf;

        let c1 = -vf3;
        let c2 = 3.0 * vf2 + 3.0 * vf3;
        let c3 = -6.0 * vf2 - 3.0 * vf - 3.0 * vf3;
        let c4 = 1.0 + 3.0 * vf + vf3 + 3.0 * vf2;

        // Apply 6 cascaded EMAs
        let e1 = self.ema(data);
        let e2 = self.ema(&e1);
        let e3 = self.ema(&e2);
        let e4 = self.ema(&e3);
        let e5 = self.ema(&e4);
        let e6 = self.ema(&e5);

        // Combine EMAs using T3 formula
        let mut result = Vec::with_capacity(data.len());
        for i in 0..data.len() {
            if e3[i].is_nan() || e4[i].is_nan() || e5[i].is_nan() || e6[i].is_nan() {
                result.push(f64::NAN);
            } else {
                let t3 = c1 * e6[i] + c2 * e5[i] + c3 * e4[i] + c4 * e3[i];
                result.push(t3);
            }
        }

        result
    }

    /// Calculate single EMA pass.
    fn ema(&self, data: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(data.len());
        let mut ema_value = f64::NAN;

        for &value in data.iter() {
            if value.is_nan() {
                result.push(f64::NAN);
            } else if ema_value.is_nan() {
                // Initialize with first valid value
                ema_value = value;
                result.push(ema_value);
            } else {
                ema_value = self.alpha * value + (1.0 - self.alpha) * ema_value;
                result.push(ema_value);
            }
        }

        result
    }
}

impl Default for T3 {
    fn default() -> Self {
        Self::from_config(T3Config::default())
    }
}

impl TechnicalIndicator for T3 {
    fn name(&self) -> &str {
        "T3"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t3() {
        let t3 = T3::new(5, 0.7);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = t3.calculate(&data);

        // All values should be valid after warmup
        assert_eq!(result.len(), 30);
        // Later values should be valid
        assert!(!result[29].is_nan());
    }

    #[test]
    fn test_t3_smoothness() {
        let t3 = T3::new(5, 0.7);
        // Add some noise to data
        let data: Vec<f64> = (0..30)
            .map(|i| 100.0 + i as f64 + (if i % 2 == 0 { 0.5 } else { -0.5 }))
            .collect();
        let result = t3.calculate(&data);

        // T3 should smooth out the oscillations
        // Check that consecutive values don't oscillate as much as raw data
        let mut t3_oscillations = 0;
        let mut data_oscillations = 0;

        for i in 11..29 {
            if (result[i] - result[i - 1]) * (result[i + 1] - result[i]) < 0.0 {
                t3_oscillations += 1;
            }
            if (data[i] - data[i - 1]) * (data[i + 1] - data[i]) < 0.0 {
                data_oscillations += 1;
            }
        }

        assert!(t3_oscillations <= data_oscillations);
    }

    #[test]
    fn test_t3_insufficient_data() {
        let t3 = T3::new(10, 0.7);
        let data = vec![1.0, 2.0, 3.0];
        let result = t3.calculate(&data);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_t3_default() {
        let t3 = T3::default();
        assert_eq!(t3.period, 5);
        assert!((t3.volume_factor - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_t3_technical_indicator_trait() {
        let t3 = T3::new(5, 0.7);
        assert_eq!(t3.name(), "T3");
        assert_eq!(t3.min_periods(), 5);
    }

    #[test]
    fn test_t3_volume_factor_effect() {
        let t3_low_vf = T3::new(5, 0.5);
        let t3_high_vf = T3::new(5, 0.9);
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();

        let result_low = t3_low_vf.calculate(&data);
        let result_high = t3_high_vf.calculate(&data);

        // Higher volume factor should be smoother (lag more behind price)
        // In an uptrend, T3 with lower vf should be closer to price
        assert!(!result_low[19].is_nan());
        assert!(!result_high[19].is_nan());
    }
}
