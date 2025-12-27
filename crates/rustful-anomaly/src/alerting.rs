//! Alerting system

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

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
