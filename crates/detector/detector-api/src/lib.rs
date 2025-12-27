//! Anomaly detection algorithms
//!
//! This crate provides implementations of various anomaly detection algorithms:
//!
//! - [`ZScoreDetector`]: Statistical z-score based detection
//! - [`IQRDetector`]: Interquartile range based detection

mod zscore;
mod iqr;

// Re-export from core
pub use detector_core::{Alert, AlertSeverity, DetectorError, Monitor, Result};

// Re-export traits from SPI
pub use detector_spi::{AnomalyDetector, DetectionResult};

// Re-export implementations
pub use zscore::ZScoreDetector;
pub use iqr::IQRDetector;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::AnomalyDetector;
    pub use crate::{ZScoreDetector, IQRDetector};
    pub use crate::{DetectionResult, Alert, AlertSeverity};
    pub use crate::{Result, DetectorError};
}
