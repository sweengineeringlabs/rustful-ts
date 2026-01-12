//! Anomaly Detection API
//!
//! Configuration types and builders for anomaly detection.

use serde::{Deserialize, Serialize};

// Re-export SPI types
pub use anomaly_spi::{AnomalyError, AnomalyResult, Alert, AlertSeverity, Result};

// ============================================================================
// Detector Configuration
// ============================================================================

/// Z-Score detector configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZScoreConfig {
    /// Z-score threshold for anomaly detection (default: 3.0).
    pub threshold: f64,
}

impl Default for ZScoreConfig {
    fn default() -> Self {
        Self { threshold: 3.0 }
    }
}

impl ZScoreConfig {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

/// IQR detector configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IQRConfig {
    /// IQR multiplier for outlier bounds (default: 1.5).
    pub multiplier: f64,
}

impl Default for IQRConfig {
    fn default() -> Self {
        Self { multiplier: 1.5 }
    }
}

impl IQRConfig {
    pub fn new(multiplier: f64) -> Self {
        Self { multiplier }
    }
}

/// Monitor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Buffer size for streaming detection.
    pub buffer_size: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self { buffer_size: 100 }
    }
}

impl MonitorConfig {
    pub fn new(buffer_size: usize) -> Self {
        Self { buffer_size }
    }
}
