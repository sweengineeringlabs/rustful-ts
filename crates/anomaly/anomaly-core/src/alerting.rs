//! Alerting system implementation.

use std::time::{SystemTime, UNIX_EPOCH};

use anomaly_spi::{Alert, AlertSeverity};

/// Create a new alert from a detected anomaly.
pub fn create_alert(value: f64, score: f64) -> Alert {
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

    Alert {
        timestamp,
        value,
        score,
        severity,
        message,
    }
}

/// Alert builder for custom alert creation.
#[derive(Debug, Clone)]
pub struct AlertBuilder {
    value: f64,
    score: f64,
    severity: Option<AlertSeverity>,
    message: Option<String>,
}

impl AlertBuilder {
    /// Create a new alert builder.
    pub fn new(value: f64, score: f64) -> Self {
        Self {
            value,
            score,
            severity: None,
            message: None,
        }
    }

    /// Set custom severity.
    pub fn severity(mut self, severity: AlertSeverity) -> Self {
        self.severity = Some(severity);
        self
    }

    /// Set custom message.
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Build the alert.
    pub fn build(self) -> Alert {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let severity = self.severity.unwrap_or_else(|| {
            if self.score.abs() > 5.0 {
                AlertSeverity::Critical
            } else {
                AlertSeverity::Warning
            }
        });

        let message = self.message.unwrap_or_else(|| {
            format!(
                "Anomaly detected: value={:.4}, score={:.4}",
                self.value, self.score
            )
        });

        Alert {
            timestamp,
            value: self.value,
            score: self.score,
            severity,
            message,
        }
    }
}
