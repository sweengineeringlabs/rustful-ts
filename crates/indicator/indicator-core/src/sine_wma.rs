//! Sine Weighted Moving Average (Sine WMA) implementation.
//!
//! A weighted moving average using sine function for weight distribution.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::SineWMAConfig;
use std::f64::consts::PI;

/// Sine Weighted Moving Average (Sine WMA).
///
/// Uses sine function values as weights, creating a smooth bell-shaped
/// weight distribution. The weights follow a sine curve from 0 to PI,
/// giving maximum weight to middle values and lower weights to edges.
///
/// Weight(i) = sin(PI * i / period) for i = 1 to period
#[derive(Debug, Clone)]
pub struct SineWMA {
    period: usize,
    weights: Vec<f64>,
    weight_sum: f64,
}

impl SineWMA {
    pub fn new(period: usize) -> Self {
        let weights = Self::calculate_weights(period);
        let weight_sum = weights.iter().sum();
        Self { period, weights, weight_sum }
    }

    pub fn from_config(config: SineWMAConfig) -> Self {
        Self::new(config.period)
    }

    /// Calculate sine weights for the given period.
    fn calculate_weights(period: usize) -> Vec<f64> {
        (1..=period)
            .map(|i| (PI * i as f64 / (period as f64 + 1.0)).sin())
            .collect()
    }

    /// Calculate Sine WMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.period || self.period == 0 {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..data.len() {
            let window_start = i + 1 - self.period;
            let mut weighted_sum = 0.0;

            for j in 0..self.period {
                weighted_sum += self.weights[j] * data[window_start + j];
            }

            result.push(weighted_sum / self.weight_sum);
        }

        result
    }
}

impl Default for SineWMA {
    fn default() -> Self {
        Self::from_config(SineWMAConfig::default())
    }
}

impl TechnicalIndicator for SineWMA {
    fn name(&self) -> &str {
        "SineWMA"
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
    fn test_sine_wma() {
        let sine_wma = SineWMA::new(5);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let result = sine_wma.calculate(&data);

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
    fn test_sine_weights() {
        let sine_wma = SineWMA::new(5);

        // Weights should form a bell curve (increasing then decreasing)
        assert!(sine_wma.weights[0] < sine_wma.weights[2]);
        assert!(sine_wma.weights[4] < sine_wma.weights[2]);
        // Middle weight should be highest (or close to it)
        assert!(sine_wma.weights[2] >= sine_wma.weights[0]);
        assert!(sine_wma.weights[2] >= sine_wma.weights[4]);
    }

    #[test]
    fn test_sine_wma_symmetry() {
        let sine_wma = SineWMA::new(7);

        // For odd periods, weights should be symmetric
        let tolerance = 1e-10;
        assert!((sine_wma.weights[0] - sine_wma.weights[6]).abs() < tolerance);
        assert!((sine_wma.weights[1] - sine_wma.weights[5]).abs() < tolerance);
        assert!((sine_wma.weights[2] - sine_wma.weights[4]).abs() < tolerance);
    }

    #[test]
    fn test_sine_wma_constant_data() {
        let sine_wma = SineWMA::new(5);
        let data = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let result = sine_wma.calculate(&data);

        // For constant data, Sine WMA should equal the constant
        assert!((result[4] - 10.0).abs() < 1e-10);
        assert!((result[5] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sine_wma_insufficient_data() {
        let sine_wma = SineWMA::new(10);
        let data = vec![1.0, 2.0, 3.0];
        let result = sine_wma.calculate(&data);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_sine_wma_default() {
        let sine_wma = SineWMA::default();
        assert_eq!(sine_wma.period, 14);
    }

    #[test]
    fn test_sine_wma_technical_indicator_trait() {
        let sine_wma = SineWMA::new(5);
        assert_eq!(sine_wma.name(), "SineWMA");
        assert_eq!(sine_wma.min_periods(), 5);
    }

    #[test]
    fn test_sine_wma_vs_sma() {
        // Sine WMA should give more weight to middle values
        let sine_wma = SineWMA::new(5);
        // Data with high value in the middle
        let data = vec![1.0, 1.0, 10.0, 1.0, 1.0];
        let result = sine_wma.calculate(&data);

        // SMA would be 2.8, Sine WMA should be higher due to middle weighting
        let sma = 2.8;
        assert!(result[4] > sma);
    }
}
