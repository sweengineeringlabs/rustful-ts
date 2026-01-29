//! Autoencoder Anomaly Detection Indicator (IND-293)
//!
//! Reconstruction error proxy using statistical methods to simulate
//! autoencoder-based anomaly detection without actual neural network training.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Autoencoder Anomaly - Reconstruction error proxy for anomaly detection
///
/// This indicator simulates autoencoder reconstruction error using statistical
/// methods. It measures how "unusual" current market conditions are compared
/// to recent history by computing normalized deviations across multiple features.
///
/// # Interpretation
/// - High values (> 2.0) indicate anomalous market conditions
/// - Low values (< 1.0) indicate normal market behavior
/// - Spikes often precede significant market moves
#[derive(Debug, Clone)]
pub struct AutoencoderAnomaly {
    /// Lookback period for computing baseline statistics
    period: usize,
    /// Number of features to use in reconstruction
    num_features: usize,
    /// Smoothing period for final output
    smooth_period: usize,
}

impl AutoencoderAnomaly {
    /// Create a new AutoencoderAnomaly indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for baseline (minimum 10)
    /// * `num_features` - Number of market features to analyze (2-6)
    /// * `smooth_period` - Smoothing period for output (minimum 2)
    pub fn new(period: usize, num_features: usize, smooth_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if num_features < 2 || num_features > 6 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_features".to_string(),
                reason: "must be between 2 and 6".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, num_features, smooth_period })
    }

    /// Calculate reconstruction error proxy
    ///
    /// Features used:
    /// 1. Returns
    /// 2. Volatility (range/close)
    /// 3. Volume deviation
    /// 4. Close location in range
    /// 5. Gap size
    /// 6. Price momentum
    pub fn calculate(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate feature vectors
        let mut features: Vec<Vec<f64>> = vec![vec![0.0; n]; self.num_features];

        for i in 1..n {
            // Feature 1: Returns
            features[0][i] = if close[i - 1] > 1e-10 {
                (close[i] / close[i - 1] - 1.0) * 100.0
            } else {
                0.0
            };

            // Feature 2: Range volatility
            if self.num_features >= 2 {
                let range = high[i] - low[i];
                features[1][i] = if close[i] > 1e-10 {
                    range / close[i] * 100.0
                } else {
                    0.0
                };
            }

            // Feature 3: Volume deviation (vs rolling mean)
            if self.num_features >= 3 {
                let start = i.saturating_sub(self.period);
                let vol_mean = volume[start..i].iter().sum::<f64>() / (i - start).max(1) as f64;
                features[2][i] = if vol_mean > 1e-10 {
                    (volume[i] / vol_mean - 1.0) * 100.0
                } else {
                    0.0
                };
            }

            // Feature 4: Close location in range
            if self.num_features >= 4 {
                let range = high[i] - low[i];
                features[3][i] = if range > 1e-10 {
                    ((close[i] - low[i]) / range - 0.5) * 100.0
                } else {
                    0.0
                };
            }

            // Feature 5: Gap size
            if self.num_features >= 5 {
                features[4][i] = if close[i - 1] > 1e-10 {
                    ((open[i] - close[i - 1]) / close[i - 1]) * 100.0
                } else {
                    0.0
                };
            }

            // Feature 6: Short-term momentum
            if self.num_features >= 6 && i >= 5 {
                features[5][i] = if close[i - 5] > 1e-10 {
                    (close[i] / close[i - 5] - 1.0) * 100.0
                } else {
                    0.0
                };
            }
        }

        // Calculate reconstruction error as normalized Mahalanobis-like distance
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut total_error = 0.0;

            for f in 0..self.num_features {
                // Calculate mean and std for this feature over lookback
                let values: Vec<f64> = (start..i).map(|j| features[f][j]).collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>() / values.len() as f64;
                let std = variance.sqrt().max(1e-10);

                // Z-score for current value
                let z_score = (features[f][i] - mean).abs() / std;
                total_error += z_score.powi(2);
            }

            // Root mean squared error across features
            result[i] = (total_error / self.num_features as f64).sqrt();
        }

        // Apply exponential smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }
}

impl TechnicalIndicator for AutoencoderAnomaly {
    fn name(&self) -> &str {
        "Autoencoder Anomaly"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(
            &data.open,
            &data.high,
            &data.low,
            &data.close,
            &data.volume,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let n = 50;
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.3 + (i as f64 * 0.2).sin() * 3.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.5).collect();
        let open: Vec<f64> = close.iter().enumerate()
            .map(|(i, c)| if i > 0 { close[i - 1] + 0.1 } else { *c })
            .collect();
        let volume: Vec<f64> = (0..n)
            .map(|i| 1000.0 + (i as f64 * 0.3).sin() * 300.0)
            .collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_autoencoder_anomaly_basic() {
        let data = make_test_data();
        let indicator = AutoencoderAnomaly::new(20, 4, 5).unwrap();
        let result = indicator.calculate(
            &data.open,
            &data.high,
            &data.low,
            &data.close,
            &data.volume,
        );

        assert_eq!(result.len(), data.close.len());
        // Reconstruction error should be non-negative
        for i in 25..result.len() {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_autoencoder_anomaly_detects_anomaly() {
        let mut data = make_test_data();
        // Inject anomaly: large spike in price and volume
        data.close[40] = data.close[40] * 1.15; // 15% spike
        data.volume[40] = data.volume[40] * 3.0; // 3x volume

        let indicator = AutoencoderAnomaly::new(15, 4, 3).unwrap();
        let result = indicator.calculate(
            &data.open,
            &data.high,
            &data.low,
            &data.close,
            &data.volume,
        );

        // Anomaly point should have elevated reconstruction error
        let normal_avg: f64 = result[20..35].iter().sum::<f64>() / 15.0;
        // The smoothing will spread the effect, so check nearby points too
        let anomaly_region_max = result[40..45].iter().fold(0.0_f64, |a, &b| a.max(b));
        assert!(anomaly_region_max > normal_avg);
    }

    #[test]
    fn test_autoencoder_anomaly_technical_indicator_trait() {
        let data = make_test_data();
        let indicator = AutoencoderAnomaly::new(15, 3, 5).unwrap();

        assert_eq!(indicator.name(), "Autoencoder Anomaly");
        assert_eq!(indicator.min_periods(), 20);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.values.is_empty());
    }

    #[test]
    fn test_autoencoder_anomaly_parameter_validation() {
        assert!(AutoencoderAnomaly::new(5, 4, 5).is_err()); // period too small
        assert!(AutoencoderAnomaly::new(20, 1, 5).is_err()); // num_features too small
        assert!(AutoencoderAnomaly::new(20, 7, 5).is_err()); // num_features too large
        assert!(AutoencoderAnomaly::new(20, 4, 1).is_err()); // smooth_period too small
    }

    #[test]
    fn test_autoencoder_anomaly_different_feature_counts() {
        let data = make_test_data();

        for num_features in 2..=6 {
            let indicator = AutoencoderAnomaly::new(15, num_features, 3).unwrap();
            let result = indicator.calculate(
                &data.open,
                &data.high,
                &data.low,
                &data.close,
                &data.volume,
            );

            assert_eq!(result.len(), data.close.len());
        }
    }
}
