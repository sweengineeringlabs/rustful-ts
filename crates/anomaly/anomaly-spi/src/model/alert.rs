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

#[cfg(test)]
mod tests {
    use super::*;

    // ===== AlertSeverity Tests =====

    #[test]
    fn test_alert_severity_warning() {
        let severity = AlertSeverity::Warning;
        assert_eq!(severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_alert_severity_critical() {
        let severity = AlertSeverity::Critical;
        assert_eq!(severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_alert_severity_debug() {
        let warning = AlertSeverity::Warning;
        let critical = AlertSeverity::Critical;
        assert_eq!(format!("{:?}", warning), "Warning");
        assert_eq!(format!("{:?}", critical), "Critical");
    }

    #[test]
    fn test_alert_severity_clone() {
        let severity = AlertSeverity::Warning;
        let cloned = severity.clone();
        assert_eq!(severity, cloned);
    }

    #[test]
    fn test_alert_severity_copy() {
        let severity = AlertSeverity::Critical;
        let copied = severity; // Copy, not move
        assert_eq!(severity, copied);
        // Both can still be used since it's Copy
        assert_eq!(severity, AlertSeverity::Critical);
        assert_eq!(copied, AlertSeverity::Critical);
    }

    #[test]
    fn test_alert_severity_inequality() {
        assert_ne!(AlertSeverity::Warning, AlertSeverity::Critical);
    }

    #[test]
    fn test_alert_severity_serialize() {
        let severity = AlertSeverity::Warning;
        let json = serde_json::to_string(&severity).unwrap();
        assert_eq!(json, "\"Warning\"");

        let severity = AlertSeverity::Critical;
        let json = serde_json::to_string(&severity).unwrap();
        assert_eq!(json, "\"Critical\"");
    }

    #[test]
    fn test_alert_severity_deserialize() {
        let warning: AlertSeverity = serde_json::from_str("\"Warning\"").unwrap();
        assert_eq!(warning, AlertSeverity::Warning);

        let critical: AlertSeverity = serde_json::from_str("\"Critical\"").unwrap();
        assert_eq!(critical, AlertSeverity::Critical);
    }

    #[test]
    fn test_alert_severity_roundtrip() {
        for severity in [AlertSeverity::Warning, AlertSeverity::Critical] {
            let json = serde_json::to_string(&severity).unwrap();
            let deserialized: AlertSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(severity, deserialized);
        }
    }

    // ===== Alert Tests =====

    fn create_test_alert() -> Alert {
        Alert {
            timestamp: 1234567890,
            value: 100.5,
            score: 0.95,
            severity: AlertSeverity::Critical,
            message: "Anomaly detected".to_string(),
        }
    }

    #[test]
    fn test_alert_creation() {
        let alert = create_test_alert();
        assert_eq!(alert.timestamp, 1234567890);
        assert!((alert.value - 100.5).abs() < f64::EPSILON);
        assert!((alert.score - 0.95).abs() < f64::EPSILON);
        assert_eq!(alert.severity, AlertSeverity::Critical);
        assert_eq!(alert.message, "Anomaly detected");
    }

    #[test]
    fn test_alert_with_warning_severity() {
        let alert = Alert {
            timestamp: 999,
            value: 50.0,
            score: 0.7,
            severity: AlertSeverity::Warning,
            message: "Minor anomaly".to_string(),
        };
        assert_eq!(alert.severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_alert_with_zero_timestamp() {
        let alert = Alert {
            timestamp: 0,
            value: 10.0,
            score: 0.5,
            severity: AlertSeverity::Warning,
            message: "Test".to_string(),
        };
        assert_eq!(alert.timestamp, 0);
    }

    #[test]
    fn test_alert_with_max_timestamp() {
        let alert = Alert {
            timestamp: u64::MAX,
            value: 10.0,
            score: 0.5,
            severity: AlertSeverity::Warning,
            message: "Test".to_string(),
        };
        assert_eq!(alert.timestamp, u64::MAX);
    }

    #[test]
    fn test_alert_with_negative_value() {
        let alert = Alert {
            timestamp: 100,
            value: -999.99,
            score: 0.8,
            severity: AlertSeverity::Critical,
            message: "Negative value detected".to_string(),
        };
        assert!((alert.value - (-999.99)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_alert_with_infinity_value() {
        let alert = Alert {
            timestamp: 100,
            value: f64::INFINITY,
            score: 1.0,
            severity: AlertSeverity::Critical,
            message: "Overflow".to_string(),
        };
        assert!(alert.value.is_infinite());
    }

    #[test]
    fn test_alert_with_nan_value() {
        let alert = Alert {
            timestamp: 100,
            value: f64::NAN,
            score: 0.0,
            severity: AlertSeverity::Warning,
            message: "Invalid data".to_string(),
        };
        assert!(alert.value.is_nan());
    }

    #[test]
    fn test_alert_with_zero_score() {
        let alert = Alert {
            timestamp: 100,
            value: 50.0,
            score: 0.0,
            severity: AlertSeverity::Warning,
            message: "Low anomaly score".to_string(),
        };
        assert!((alert.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_alert_with_max_score() {
        let alert = Alert {
            timestamp: 100,
            value: 50.0,
            score: 1.0,
            severity: AlertSeverity::Critical,
            message: "Maximum anomaly".to_string(),
        };
        assert!((alert.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_alert_with_empty_message() {
        let alert = Alert {
            timestamp: 100,
            value: 50.0,
            score: 0.5,
            severity: AlertSeverity::Warning,
            message: String::new(),
        };
        assert!(alert.message.is_empty());
    }

    #[test]
    fn test_alert_with_long_message() {
        let long_message = "a".repeat(10000);
        let alert = Alert {
            timestamp: 100,
            value: 50.0,
            score: 0.5,
            severity: AlertSeverity::Warning,
            message: long_message.clone(),
        };
        assert_eq!(alert.message.len(), 10000);
    }

    #[test]
    fn test_alert_debug() {
        let alert = create_test_alert();
        let debug_str = format!("{:?}", alert);
        assert!(debug_str.contains("Alert"));
        assert!(debug_str.contains("1234567890"));
        assert!(debug_str.contains("Critical"));
    }

    #[test]
    fn test_alert_clone() {
        let alert = create_test_alert();
        let cloned = alert.clone();
        assert_eq!(alert.timestamp, cloned.timestamp);
        assert!((alert.value - cloned.value).abs() < f64::EPSILON);
        assert!((alert.score - cloned.score).abs() < f64::EPSILON);
        assert_eq!(alert.severity, cloned.severity);
        assert_eq!(alert.message, cloned.message);
    }

    #[test]
    fn test_alert_serialize() {
        let alert = Alert {
            timestamp: 100,
            value: 50.5,
            score: 0.75,
            severity: AlertSeverity::Warning,
            message: "Test alert".to_string(),
        };
        let json = serde_json::to_string(&alert).unwrap();
        assert!(json.contains("\"timestamp\":100"));
        assert!(json.contains("\"value\":50.5"));
        assert!(json.contains("\"score\":0.75"));
        assert!(json.contains("\"severity\":\"Warning\""));
        assert!(json.contains("\"message\":\"Test alert\""));
    }

    #[test]
    fn test_alert_deserialize() {
        let json = r#"{
            "timestamp": 200,
            "value": 99.9,
            "score": 0.88,
            "severity": "Critical",
            "message": "Deserialized alert"
        }"#;
        let alert: Alert = serde_json::from_str(json).unwrap();
        assert_eq!(alert.timestamp, 200);
        assert!((alert.value - 99.9).abs() < f64::EPSILON);
        assert!((alert.score - 0.88).abs() < f64::EPSILON);
        assert_eq!(alert.severity, AlertSeverity::Critical);
        assert_eq!(alert.message, "Deserialized alert");
    }

    #[test]
    fn test_alert_roundtrip() {
        let alert = create_test_alert();
        let json = serde_json::to_string(&alert).unwrap();
        let deserialized: Alert = serde_json::from_str(&json).unwrap();
        assert_eq!(alert.timestamp, deserialized.timestamp);
        assert!((alert.value - deserialized.value).abs() < f64::EPSILON);
        assert!((alert.score - deserialized.score).abs() < f64::EPSILON);
        assert_eq!(alert.severity, deserialized.severity);
        assert_eq!(alert.message, deserialized.message);
    }

    #[test]
    fn test_alert_deserialize_invalid_severity() {
        let json = r#"{
            "timestamp": 100,
            "value": 50.0,
            "score": 0.5,
            "severity": "InvalidSeverity",
            "message": "Test"
        }"#;
        let result: Result<Alert, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_alert_deserialize_missing_field() {
        let json = r#"{
            "timestamp": 100,
            "value": 50.0,
            "score": 0.5,
            "severity": "Warning"
        }"#;
        let result: Result<Alert, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_alert_field_modification() {
        let mut alert = create_test_alert();
        alert.timestamp = 999;
        alert.value = -1.0;
        alert.score = 0.0;
        alert.severity = AlertSeverity::Warning;
        alert.message = "Modified".to_string();

        assert_eq!(alert.timestamp, 999);
        assert!((alert.value - (-1.0)).abs() < f64::EPSILON);
        assert!((alert.score - 0.0).abs() < f64::EPSILON);
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.message, "Modified");
    }
}
