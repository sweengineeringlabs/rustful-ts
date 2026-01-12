//! Kalman Filter implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::KalmanConfig;

/// Kalman Filter for financial time series.
///
/// A simplified 1D Kalman filter that adaptively smooths price data.
/// - process_noise: Expected variance in the system
/// - measurement_noise: Expected variance in measurements
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    process_noise: f64,
    measurement_noise: f64,
}

impl KalmanFilter {
    pub fn new(process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            process_noise,
            measurement_noise,
        }
    }

    pub fn from_config(config: KalmanConfig) -> Self {
        Self {
            process_noise: config.process_noise,
            measurement_noise: config.measurement_noise,
        }
    }

    /// Calculate Kalman filtered values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();

        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);

        // Initialize state
        let mut estimate = data[0];
        let mut error_estimate = 1.0;

        result.push(estimate);

        for i in 1..n {
            // Prediction step
            let error_predict = error_estimate + self.process_noise;

            // Update step
            let kalman_gain = error_predict / (error_predict + self.measurement_noise);
            estimate = estimate + kalman_gain * (data[i] - estimate);
            error_estimate = (1.0 - kalman_gain) * error_predict;

            result.push(estimate);
        }

        result
    }
}

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::from_config(KalmanConfig::default())
    }
}

impl TechnicalIndicator for KalmanFilter {
    fn name(&self) -> &str {
        "Kalman"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman() {
        let kalman = KalmanFilter::default();
        // Add some noise to a linear trend
        let data: Vec<f64> = (0..30)
            .map(|i| 100.0 + i as f64 + if i % 2 == 0 { 2.0 } else { -2.0 })
            .collect();
        let result = kalman.calculate(&data);

        // Should smooth out the noise
        assert_eq!(result.len(), data.len());
        // First value should equal input
        assert!((result[0] - data[0]).abs() < 1e-10);
        // Should track the trend
        assert!(result[29] > result[0]);
    }
}
