//! Anomaly detector implementations.

use anomaly_api::{ZScoreConfig, IQRConfig};
use anomaly_spi::{AnomalyDetector, AnomalyResult, Result};

// ============================================================================
// Z-Score Detector
// ============================================================================

/// Z-Score based anomaly detector.
///
/// Detects anomalies based on standard deviation from the mean.
#[derive(Debug, Clone)]
pub struct ZScoreDetector {
    threshold: f64,
    mean: f64,
    std_dev: f64,
    fitted: bool,
}

impl ZScoreDetector {
    /// Create a new Z-Score detector with given threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            mean: 0.0,
            std_dev: 1.0,
            fitted: false,
        }
    }

    /// Create from configuration.
    pub fn from_config(config: ZScoreConfig) -> Self {
        Self::new(config.threshold)
    }
}

impl Default for ZScoreDetector {
    fn default() -> Self {
        Self::new(3.0)
    }
}

impl AnomalyDetector for ZScoreDetector {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        let n = data.len() as f64;
        self.mean = data.iter().sum::<f64>() / n;
        self.std_dev = (data.iter().map(|x| (x - self.mean).powi(2)).sum::<f64>() / n).sqrt();
        self.fitted = true;
        Ok(())
    }

    fn detect(&self, data: &[f64]) -> Result<AnomalyResult> {
        let scores = self.score(data)?;
        let is_anomaly = scores.iter().map(|&s| s.abs() > self.threshold).collect();
        Ok(AnomalyResult::new(is_anomaly, scores, self.threshold))
    }

    fn score(&self, data: &[f64]) -> Result<Vec<f64>> {
        if self.std_dev == 0.0 {
            return Ok(vec![0.0; data.len()]);
        }
        Ok(data.iter().map(|&x| (x - self.mean) / self.std_dev).collect())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// ============================================================================
// IQR Detector
// ============================================================================

/// IQR-based anomaly detector.
///
/// Detects anomalies based on interquartile range.
#[derive(Debug, Clone)]
pub struct IQRDetector {
    multiplier: f64,
    q1: f64,
    q3: f64,
    fitted: bool,
}

impl IQRDetector {
    /// Create a new IQR detector with given multiplier.
    pub fn new(multiplier: f64) -> Self {
        Self {
            multiplier,
            q1: 0.0,
            q3: 0.0,
            fitted: false,
        }
    }

    /// Create from configuration.
    pub fn from_config(config: IQRConfig) -> Self {
        Self::new(config.multiplier)
    }
}

impl Default for IQRDetector {
    fn default() -> Self {
        Self::new(1.5)
    }
}

impl AnomalyDetector for IQRDetector {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        self.q1 = sorted[n / 4];
        self.q3 = sorted[3 * n / 4];
        self.fitted = true;
        Ok(())
    }

    fn detect(&self, data: &[f64]) -> Result<AnomalyResult> {
        let iqr = self.q3 - self.q1;
        let lower = self.q1 - self.multiplier * iqr;
        let upper = self.q3 + self.multiplier * iqr;

        let is_anomaly: Vec<bool> = data.iter().map(|&x| x < lower || x > upper).collect();
        let scores: Vec<f64> = data
            .iter()
            .map(|&x| {
                if x < lower {
                    (lower - x) / iqr
                } else if x > upper {
                    (x - upper) / iqr
                } else {
                    0.0
                }
            })
            .collect();

        Ok(AnomalyResult::new(is_anomaly, scores, self.multiplier))
    }

    fn score(&self, data: &[f64]) -> Result<Vec<f64>> {
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
