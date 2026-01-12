//! Anomaly detector trait definition.

use crate::error::Result;
use crate::model::{Alert, AnomalyResult};

/// Anomaly detector trait.
///
/// Implementations detect anomalies in time series data.
pub trait AnomalyDetector: Send + Sync {
    /// Fit the detector to training data.
    fn fit(&mut self, data: &[f64]) -> Result<()>;

    /// Detect anomalies in data.
    fn detect(&self, data: &[f64]) -> Result<AnomalyResult>;

    /// Compute anomaly scores without thresholding.
    fn score(&self, data: &[f64]) -> Result<Vec<f64>>;

    /// Check if detector has been fitted.
    fn is_fitted(&self) -> bool;
}

/// Real-time monitoring trait.
pub trait MonitoringStream<D: AnomalyDetector>: Send + Sync {
    /// Push a new value and check for anomalies.
    fn push(&mut self, value: f64) -> Result<Option<Alert>>;

    /// Get current buffer contents.
    fn buffer(&self) -> &[f64];

    /// Reset the monitor state.
    fn reset(&mut self);
}
