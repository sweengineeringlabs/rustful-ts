//! IQR-based anomaly detector

use detector_core::{DetectorError, Result};
use detector_spi::{AnomalyDetector, DetectionResult, Result as SpiResult};
use serde::{Deserialize, Serialize};

/// Interquartile Range (IQR) based anomaly detector
///
/// Detects anomalies based on the IQR method, commonly used
/// in box plots to identify outliers.
///
/// @algorithm IQR
/// @category StatisticalDetector
/// @complexity O(n log n) fit, O(n) detect
/// @thread_safe false
/// @since 0.2.0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IQRDetector {
    multiplier: f64,
    q1: f64,
    q3: f64,
    fitted: bool,
}

impl IQRDetector {
    /// Create a new IQR detector
    ///
    /// # Arguments
    ///
    /// * `multiplier` - IQR multiplier for outlier bounds (typically 1.5)
    pub fn new(multiplier: f64) -> Result<Self> {
        if multiplier <= 0.0 {
            return Err(DetectorError::InvalidParameter {
                name: "multiplier".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        Ok(Self {
            multiplier,
            q1: 0.0,
            q3: 0.0,
            fitted: false,
        })
    }

    /// Get the multiplier
    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Get the first quartile
    pub fn q1(&self) -> f64 {
        self.q1
    }

    /// Get the third quartile
    pub fn q3(&self) -> f64 {
        self.q3
    }

    /// Get the IQR
    pub fn iqr(&self) -> f64 {
        self.q3 - self.q1
    }
}

impl Default for IQRDetector {
    fn default() -> Self {
        Self::new(1.5).unwrap()
    }
}

impl AnomalyDetector for IQRDetector {
    fn fit(&mut self, data: &[f64]) -> SpiResult<()> {
        if data.len() < 4 {
            return Err(DetectorError::InsufficientData {
                required: 4,
                actual: data.len(),
            }
            .into());
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        self.q1 = sorted[n / 4];
        self.q3 = sorted[3 * n / 4];
        self.fitted = true;
        Ok(())
    }

    fn detect(&self, data: &[f64]) -> SpiResult<DetectionResult> {
        if !self.fitted {
            return Err(DetectorError::NotFitted.into());
        }

        let iqr = self.q3 - self.q1;
        let lower = self.q1 - self.multiplier * iqr;
        let upper = self.q3 + self.multiplier * iqr;

        let is_anomaly: Vec<bool> = data.iter().map(|&x| x < lower || x > upper).collect();
        let scores: Vec<f64> = data
            .iter()
            .map(|&x| {
                if iqr == 0.0 {
                    0.0
                } else if x < lower {
                    (lower - x) / iqr
                } else if x > upper {
                    (x - upper) / iqr
                } else {
                    0.0
                }
            })
            .collect();

        Ok(DetectionResult {
            is_anomaly,
            scores,
            threshold: self.multiplier,
        })
    }

    fn score(&self, data: &[f64]) -> SpiResult<Vec<f64>> {
        if !self.fitted {
            return Err(DetectorError::NotFitted.into());
        }

        let iqr = self.q3 - self.q1;
        Ok(data
            .iter()
            .map(|&x| {
                if iqr == 0.0 {
                    0.0
                } else {
                    ((x - self.q1) / iqr).abs()
                }
            })
            .collect())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}
