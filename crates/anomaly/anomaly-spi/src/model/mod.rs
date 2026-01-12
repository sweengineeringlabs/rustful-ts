//! Data models for anomaly detection.
//!
//! This module contains data structures used throughout the anomaly detection system.

mod alert;
mod anomaly_result;

pub use alert::{Alert, AlertSeverity};
pub use anomaly_result::AnomalyResult;
