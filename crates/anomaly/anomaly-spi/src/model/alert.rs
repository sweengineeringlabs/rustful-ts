//! Alert types for anomaly detection.

use serde::{Deserialize, Serialize};

/// Alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Warning,
    Critical,
}

/// An alert triggered by anomaly detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub timestamp: u64,
    pub value: f64,
    pub score: f64,
    pub severity: AlertSeverity,
    pub message: String,
}
