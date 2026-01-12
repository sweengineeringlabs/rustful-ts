//! Arnaud Legoux Moving Average (ALMA) implementation.
//!
//! A Gaussian-weighted moving average with customizable offset.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::ALMAConfig;

/// Arnaud Legoux Moving Average (ALMA).
///
/// ALMA uses a Gaussian distribution curve as the weighting function,
/// with adjustable offset and sigma parameters. The offset controls the
/// trade-off between smoothness and responsiveness, while sigma controls
/// the width of the Gaussian curve.
///
/// - Offset: 0.85 (default) places the curve closer to recent prices
/// - Sigma: Controls the shape of the Gaussian curve (default: 6)
#[derive(Debug, Clone)]
pub struct ALMA {
    period: usize,
    offset: f64,
    sigma: f64,
}

impl ALMA {
    pub fn new(period: usize, offset: f64, sigma: f64) -> Self {
        Self { period, offset, sigma }
    }

    pub fn from_config(config: ALMAConfig) -> Self {
        Self {
            period: config.period,
            offset: config.offset,
            sigma: config.sigma,
        }
    }

    /// Calculate ALMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.period || self.period == 0 {
            return vec![f64::NAN; data.len()];
        }

        // Pre-calculate weights
        let weights = self.calculate_weights();
        let weight_sum: f64 = weights.iter().sum();

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..data.len() {
            let window_start = i + 1 - self.period;
            let mut weighted_sum = 0.0;

            for j in 0..self.period {
                weighted_sum += weights[j] * data[window_start + j];
            }

            result.push(weighted_sum / weight_sum);
        }

        result
    }

    /// Calculate Gaussian weights for ALMA.
    fn calculate_weights(&self) -> Vec<f64> {
        let m = self.offset * (self.period - 1) as f64;
        let s = self.period as f64 / self.sigma;

        (0..self.period)
            .map(|i| {
                let diff = i as f64 - m;
                (-diff * diff / (2.0 * s * s)).exp()
            })
            .collect()
    }
}

impl Default for ALMA {
    fn default() -> Self {
        Self::from_config(ALMAConfig::default())
    }
}

impl TechnicalIndicator for ALMA {
    fn name(&self) -> &str {
        "ALMA"
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
    fn test_alma() {
        let alma = ALMA::new(5, 0.85, 6.0);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let result = alma.calculate(&data);

        // First 4 values should be NaN
        for i in 0..4 {
            assert!(result[i].is_nan());
        }
        // Subsequent values should be valid
        assert!(!result[4].is_nan());
        assert!(!result[5].is_nan());
        assert!(!result[6].is_nan());
    }

    #[test]
    fn test_alma_weights() {
        let alma = ALMA::new(5, 0.85, 6.0);
        let weights = alma.calculate_weights();

        assert_eq!(weights.len(), 5);
        // Weights should all be positive
        assert!(weights.iter().all(|w| *w > 0.0));
        // With offset 0.85, later weights should be larger
        assert!(weights[4] > weights[0]);
    }

    #[test]
    fn test_alma_insufficient_data() {
        let alma = ALMA::new(10, 0.85, 6.0);
        let data = vec![1.0, 2.0, 3.0];
        let result = alma.calculate(&data);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_alma_default() {
        let alma = ALMA::default();
        assert_eq!(alma.period, 9);
        assert!((alma.offset - 0.85).abs() < 1e-10);
        assert!((alma.sigma - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_alma_technical_indicator_trait() {
        let alma = ALMA::new(5, 0.85, 6.0);
        assert_eq!(alma.name(), "ALMA");
        assert_eq!(alma.min_periods(), 5);
    }
}
