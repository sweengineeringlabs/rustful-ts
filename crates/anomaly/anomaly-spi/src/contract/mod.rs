//! Contract definitions for anomaly detection.
//!
//! This module contains trait definitions that providers must implement.

mod anomaly_detector;

pub use anomaly_detector::{AnomalyDetector, MonitoringStream};
