//! Anomaly Detection Facade
//!
//! Unified re-exports for the anomaly detection module.
//!
//! This facade provides a single entry point to all anomaly detection functionality:
//! - `AnomalyDetector` trait and `AnomalyResult` from SPI
//! - Configuration types from API
//! - Detector implementations (`ZScoreDetector`, `IQRDetector`) from Core
//! - Monitoring and alerting from Core

// Re-export everything from SPI
pub use anomaly_spi::*;

// Re-export everything from API
pub use anomaly_api::*;

// Re-export everything from Core
pub use anomaly_core::*;
