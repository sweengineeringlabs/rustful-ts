//! Detector Core
//!
//! Core types, error handling, and utilities for anomaly detectors.

pub use detector_spi::{AnomalyDetector, DetectionResult};
use thiserror::Error;
use serde::{Deserialize, Serialize};

/// Re-export SPI Result type for trait implementations
pub use detector_spi::Result as SpiResult;

/// Result type for detector operations
pub type Result<T> = std::result::Result<T, DetectorError>;

/// Errors that can occur during anomaly detection
#[derive(Error, Debug)]
pub enum DetectorError {
    /// Insufficient data points for the operation
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// Invalid parameter value
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Detector has not been fitted yet
    #[error("Detector must be fitted before detection")]
    NotFitted,

    /// Invalid input data
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Warning,
    Critical,
}

/// An alert triggered by anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub timestamp: u64,
    pub value: f64,
    pub score: f64,
    pub severity: AlertSeverity,
    pub message: String,
}

impl Alert {
    pub fn new(value: f64, score: f64) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let severity = if score.abs() > 5.0 {
            AlertSeverity::Critical
        } else {
            AlertSeverity::Warning
        };

        let message = format!(
            "Anomaly detected: value={:.4}, score={:.4}",
            value, score
        );

        Self {
            timestamp,
            value,
            score,
            severity,
            message,
        }
    }
}

/// Real-time monitor for streaming data
pub struct Monitor<D: AnomalyDetector> {
    detector: D,
    buffer: Vec<f64>,
    buffer_size: usize,
}

impl<D: AnomalyDetector> Monitor<D> {
    pub fn new(detector: D, buffer_size: usize) -> Self {
        Self {
            detector,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
        }
    }

    /// Push a new value and check for anomalies
    pub fn push(&mut self, value: f64) -> SpiResult<Option<Alert>> {
        self.buffer.push(value);
        if self.buffer.len() > self.buffer_size {
            self.buffer.remove(0);
        }

        if self.buffer.len() >= self.buffer_size {
            let result = self.detector.detect(&self.buffer)?;
            if let Some(&is_anomaly) = result.is_anomaly.last() {
                if is_anomaly {
                    let score = result.scores.last().copied().unwrap_or(0.0);
                    return Ok(Some(Alert::new(value, score)));
                }
            }
        }
        Ok(None)
    }

    /// Get current buffer contents
    pub fn buffer(&self) -> &[f64] {
        &self.buffer
    }

    /// Clear the buffer
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}
