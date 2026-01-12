//! Anomaly detection result types.

use serde::{Deserialize, Serialize};

/// Anomaly detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    /// Boolean mask indicating anomalies.
    pub is_anomaly: Vec<bool>,
    /// Anomaly scores for each point.
    pub scores: Vec<f64>,
    /// Threshold used for detection.
    pub threshold: f64,
}

impl AnomalyResult {
    /// Create a new anomaly result.
    pub fn new(is_anomaly: Vec<bool>, scores: Vec<f64>, threshold: f64) -> Self {
        Self {
            is_anomaly,
            scores,
            threshold,
        }
    }

    /// Get indices of detected anomalies.
    pub fn anomaly_indices(&self) -> Vec<usize> {
        self.is_anomaly
            .iter()
            .enumerate()
            .filter_map(|(i, &is_anomaly)| if is_anomaly { Some(i) } else { None })
            .collect()
    }

    /// Count of detected anomalies.
    pub fn anomaly_count(&self) -> usize {
        self.is_anomaly.iter().filter(|&&x| x).count()
    }
}
