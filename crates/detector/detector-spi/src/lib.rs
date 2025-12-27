//! Detector Service Provider Interface
//!
//! Defines traits for anomaly detection algorithms.

use std::error::Error;

/// Result type for detector operations
pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Common trait for all anomaly detectors
pub trait AnomalyDetector {
    /// Fit the detector to training data
    fn fit(&mut self, data: &[f64]) -> Result<()>;

    /// Detect anomalies in the given data
    fn detect(&self, data: &[f64]) -> Result<DetectionResult>;

    /// Score each point (higher = more anomalous)
    fn score(&self, data: &[f64]) -> Result<Vec<f64>>;

    /// Check if the detector has been fitted
    fn is_fitted(&self) -> bool;
}

/// Result of anomaly detection
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Boolean flags indicating anomaly status for each point
    pub is_anomaly: Vec<bool>,
    /// Anomaly scores for each point
    pub scores: Vec<f64>,
    /// Threshold used for detection
    pub threshold: f64,
}
