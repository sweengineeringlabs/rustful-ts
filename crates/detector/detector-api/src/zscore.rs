//! Z-Score based anomaly detector

use detector_core::{DetectorError, Result};
use detector_spi::{AnomalyDetector, DetectionResult, Result as SpiResult};
use serde::{Deserialize, Serialize};

/// Z-Score based anomaly detector
///
/// Detects anomalies based on how many standard deviations
/// a data point is from the mean.
///
/// @algorithm ZScore
/// @category StatisticalDetector
/// @complexity O(n) fit, O(n) detect
/// @thread_safe false
/// @since 0.2.0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZScoreDetector {
    threshold: f64,
    mean: f64,
    std_dev: f64,
    fitted: bool,
}

impl ZScoreDetector {
    /// Create a new Z-Score detector
    ///
    /// # Arguments
    ///
    /// * `threshold` - Number of standard deviations for anomaly threshold
    pub fn new(threshold: f64) -> Result<Self> {
        if threshold <= 0.0 {
            return Err(DetectorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        Ok(Self {
            threshold,
            mean: 0.0,
            std_dev: 1.0,
            fitted: false,
        })
    }

    /// Get the threshold
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get the fitted mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the fitted standard deviation
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }
}

impl Default for ZScoreDetector {
    fn default() -> Self {
        Self::new(3.0).unwrap()
    }
}

impl AnomalyDetector for ZScoreDetector {
    fn fit(&mut self, data: &[f64]) -> SpiResult<()> {
        if data.len() < 2 {
            return Err(DetectorError::InsufficientData {
                required: 2,
                actual: data.len(),
            }
            .into());
        }

        let n = data.len() as f64;
        self.mean = data.iter().sum::<f64>() / n;
        self.std_dev = (data.iter().map(|x| (x - self.mean).powi(2)).sum::<f64>() / n).sqrt();
        self.fitted = true;
        Ok(())
    }

    fn detect(&self, data: &[f64]) -> SpiResult<DetectionResult> {
        if !self.fitted {
            return Err(DetectorError::NotFitted.into());
        }

        let scores = self.score(data)?;
        let is_anomaly = scores.iter().map(|&s| s.abs() > self.threshold).collect();

        Ok(DetectionResult {
            is_anomaly,
            scores,
            threshold: self.threshold,
        })
    }

    fn score(&self, data: &[f64]) -> SpiResult<Vec<f64>> {
        if !self.fitted {
            return Err(DetectorError::NotFitted.into());
        }

        if self.std_dev == 0.0 {
            return Ok(vec![0.0; data.len()]);
        }

        Ok(data
            .iter()
            .map(|&x| (x - self.mean) / self.std_dev)
            .collect())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}
