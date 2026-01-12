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

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Constructor Tests =====

    #[test]
    fn test_new_basic() {
        let result = AnomalyResult::new(
            vec![false, true, false],
            vec![0.1, 0.9, 0.2],
            0.5,
        );
        assert_eq!(result.is_anomaly, vec![false, true, false]);
        assert_eq!(result.scores, vec![0.1, 0.9, 0.2]);
        assert!((result.threshold - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_new_empty() {
        let result = AnomalyResult::new(vec![], vec![], 0.5);
        assert!(result.is_anomaly.is_empty());
        assert!(result.scores.is_empty());
        assert!((result.threshold - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_new_all_anomalies() {
        let result = AnomalyResult::new(
            vec![true, true, true],
            vec![0.9, 0.95, 0.85],
            0.5,
        );
        assert!(result.is_anomaly.iter().all(|&x| x));
    }

    #[test]
    fn test_new_no_anomalies() {
        let result = AnomalyResult::new(
            vec![false, false, false],
            vec![0.1, 0.2, 0.3],
            0.5,
        );
        assert!(result.is_anomaly.iter().all(|&x| !x));
    }

    #[test]
    fn test_new_zero_threshold() {
        let result = AnomalyResult::new(vec![true], vec![0.1], 0.0);
        assert!((result.threshold - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_new_negative_threshold() {
        let result = AnomalyResult::new(vec![true], vec![-0.5], -1.0);
        assert!((result.threshold - (-1.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_new_large_dataset() {
        let size = 10000;
        let is_anomaly: Vec<bool> = (0..size).map(|i| i % 100 == 0).collect();
        let scores: Vec<f64> = (0..size).map(|i| (i as f64) / (size as f64)).collect();
        let result = AnomalyResult::new(is_anomaly.clone(), scores.clone(), 0.5);
        assert_eq!(result.is_anomaly.len(), size);
        assert_eq!(result.scores.len(), size);
    }

    // ===== anomaly_indices Tests =====

    #[test]
    fn test_anomaly_indices_basic() {
        let result = AnomalyResult::new(
            vec![false, true, false, true, false],
            vec![0.1, 0.9, 0.2, 0.8, 0.3],
            0.5,
        );
        assert_eq!(result.anomaly_indices(), vec![1, 3]);
    }

    #[test]
    fn test_anomaly_indices_empty() {
        let result = AnomalyResult::new(vec![], vec![], 0.5);
        assert!(result.anomaly_indices().is_empty());
    }

    #[test]
    fn test_anomaly_indices_no_anomalies() {
        let result = AnomalyResult::new(
            vec![false, false, false],
            vec![0.1, 0.2, 0.3],
            0.5,
        );
        assert!(result.anomaly_indices().is_empty());
    }

    #[test]
    fn test_anomaly_indices_all_anomalies() {
        let result = AnomalyResult::new(
            vec![true, true, true],
            vec![0.9, 0.8, 0.7],
            0.5,
        );
        assert_eq!(result.anomaly_indices(), vec![0, 1, 2]);
    }

    #[test]
    fn test_anomaly_indices_first_only() {
        let result = AnomalyResult::new(
            vec![true, false, false, false],
            vec![0.9, 0.1, 0.2, 0.3],
            0.5,
        );
        assert_eq!(result.anomaly_indices(), vec![0]);
    }

    #[test]
    fn test_anomaly_indices_last_only() {
        let result = AnomalyResult::new(
            vec![false, false, false, true],
            vec![0.1, 0.2, 0.3, 0.9],
            0.5,
        );
        assert_eq!(result.anomaly_indices(), vec![3]);
    }

    #[test]
    fn test_anomaly_indices_consecutive() {
        let result = AnomalyResult::new(
            vec![false, true, true, true, false],
            vec![0.1, 0.8, 0.9, 0.85, 0.2],
            0.5,
        );
        assert_eq!(result.anomaly_indices(), vec![1, 2, 3]);
    }

    #[test]
    fn test_anomaly_indices_alternating() {
        let result = AnomalyResult::new(
            vec![true, false, true, false, true],
            vec![0.9, 0.1, 0.8, 0.2, 0.7],
            0.5,
        );
        assert_eq!(result.anomaly_indices(), vec![0, 2, 4]);
    }

    #[test]
    fn test_anomaly_indices_single_true() {
        let result = AnomalyResult::new(vec![true], vec![0.9], 0.5);
        assert_eq!(result.anomaly_indices(), vec![0]);
    }

    #[test]
    fn test_anomaly_indices_single_false() {
        let result = AnomalyResult::new(vec![false], vec![0.1], 0.5);
        assert!(result.anomaly_indices().is_empty());
    }

    // ===== anomaly_count Tests =====

    #[test]
    fn test_anomaly_count_basic() {
        let result = AnomalyResult::new(
            vec![false, true, false, true, false],
            vec![0.1, 0.9, 0.2, 0.8, 0.3],
            0.5,
        );
        assert_eq!(result.anomaly_count(), 2);
    }

    #[test]
    fn test_anomaly_count_empty() {
        let result = AnomalyResult::new(vec![], vec![], 0.5);
        assert_eq!(result.anomaly_count(), 0);
    }

    #[test]
    fn test_anomaly_count_no_anomalies() {
        let result = AnomalyResult::new(
            vec![false, false, false, false],
            vec![0.1, 0.2, 0.3, 0.4],
            0.5,
        );
        assert_eq!(result.anomaly_count(), 0);
    }

    #[test]
    fn test_anomaly_count_all_anomalies() {
        let result = AnomalyResult::new(
            vec![true, true, true, true],
            vec![0.9, 0.8, 0.7, 0.6],
            0.5,
        );
        assert_eq!(result.anomaly_count(), 4);
    }

    #[test]
    fn test_anomaly_count_single() {
        let result = AnomalyResult::new(vec![true], vec![0.9], 0.5);
        assert_eq!(result.anomaly_count(), 1);
    }

    #[test]
    fn test_anomaly_count_large() {
        let size = 10000;
        // Every 10th element is anomaly
        let is_anomaly: Vec<bool> = (0..size).map(|i| i % 10 == 0).collect();
        let scores: Vec<f64> = vec![0.5; size];
        let result = AnomalyResult::new(is_anomaly, scores, 0.5);
        assert_eq!(result.anomaly_count(), 1000);
    }

    #[test]
    fn test_anomaly_count_consistency_with_indices() {
        let result = AnomalyResult::new(
            vec![true, false, true, false, true, true],
            vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.95],
            0.5,
        );
        assert_eq!(result.anomaly_count(), result.anomaly_indices().len());
    }

    // ===== Debug, Clone, Serialize/Deserialize Tests =====

    #[test]
    fn test_debug() {
        let result = AnomalyResult::new(
            vec![true, false],
            vec![0.9, 0.1],
            0.5,
        );
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("AnomalyResult"));
        assert!(debug_str.contains("is_anomaly"));
        assert!(debug_str.contains("scores"));
        assert!(debug_str.contains("threshold"));
    }

    #[test]
    fn test_clone() {
        let original = AnomalyResult::new(
            vec![true, false, true],
            vec![0.9, 0.1, 0.8],
            0.5,
        );
        let cloned = original.clone();
        assert_eq!(original.is_anomaly, cloned.is_anomaly);
        assert_eq!(original.scores, cloned.scores);
        assert!((original.threshold - cloned.threshold).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clone_independence() {
        let original = AnomalyResult::new(
            vec![true, false],
            vec![0.9, 0.1],
            0.5,
        );
        let mut cloned = original.clone();
        cloned.is_anomaly[0] = false;
        cloned.scores[0] = 0.0;

        // Original should be unchanged
        assert!(original.is_anomaly[0]);
        assert!((original.scores[0] - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_serialize() {
        let result = AnomalyResult::new(
            vec![true, false],
            vec![0.9, 0.1],
            0.5,
        );
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"is_anomaly\":[true,false]"));
        assert!(json.contains("\"scores\":[0.9,0.1]"));
        assert!(json.contains("\"threshold\":0.5"));
    }

    #[test]
    fn test_deserialize() {
        let json = r#"{
            "is_anomaly": [true, false, true],
            "scores": [0.9, 0.1, 0.8],
            "threshold": 0.75
        }"#;
        let result: AnomalyResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.is_anomaly, vec![true, false, true]);
        assert_eq!(result.scores, vec![0.9, 0.1, 0.8]);
        assert!((result.threshold - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_roundtrip_serialization() {
        let original = AnomalyResult::new(
            vec![true, false, true, false, true],
            vec![0.9, 0.1, 0.8, 0.2, 0.7],
            0.65,
        );
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: AnomalyResult = serde_json::from_str(&json).unwrap();
        assert_eq!(original.is_anomaly, deserialized.is_anomaly);
        assert_eq!(original.scores, deserialized.scores);
        assert!((original.threshold - deserialized.threshold).abs() < f64::EPSILON);
    }

    #[test]
    fn test_deserialize_empty_arrays() {
        let json = r#"{
            "is_anomaly": [],
            "scores": [],
            "threshold": 0.5
        }"#;
        let result: AnomalyResult = serde_json::from_str(json).unwrap();
        assert!(result.is_anomaly.is_empty());
        assert!(result.scores.is_empty());
    }

    #[test]
    fn test_deserialize_missing_field() {
        let json = r#"{
            "is_anomaly": [true, false],
            "scores": [0.9, 0.1]
        }"#;
        let result: Result<AnomalyResult, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // ===== Field Access Tests =====

    #[test]
    fn test_field_access() {
        let result = AnomalyResult::new(
            vec![true, false],
            vec![0.9, 0.1],
            0.5,
        );
        assert_eq!(result.is_anomaly.len(), 2);
        assert_eq!(result.scores.len(), 2);
        assert!(result.is_anomaly[0]);
        assert!(!result.is_anomaly[1]);
        assert!((result.scores[0] - 0.9).abs() < f64::EPSILON);
        assert!((result.scores[1] - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_field_modification() {
        let mut result = AnomalyResult::new(
            vec![true, false],
            vec![0.9, 0.1],
            0.5,
        );
        result.is_anomaly.push(true);
        result.scores.push(0.85);
        result.threshold = 0.6;

        assert_eq!(result.is_anomaly.len(), 3);
        assert_eq!(result.scores.len(), 3);
        assert!((result.threshold - 0.6).abs() < f64::EPSILON);
    }

    // ===== Edge Cases =====

    #[test]
    fn test_special_float_values_in_scores() {
        let result = AnomalyResult::new(
            vec![true, true, true],
            vec![f64::INFINITY, f64::NEG_INFINITY, 0.0],
            0.5,
        );
        assert!(result.scores[0].is_infinite() && result.scores[0].is_sign_positive());
        assert!(result.scores[1].is_infinite() && result.scores[1].is_sign_negative());
        assert!((result.scores[2] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_nan_in_scores() {
        let result = AnomalyResult::new(
            vec![true],
            vec![f64::NAN],
            0.5,
        );
        assert!(result.scores[0].is_nan());
    }

    #[test]
    fn test_very_small_threshold() {
        let result = AnomalyResult::new(
            vec![true],
            vec![1e-300],
            1e-308,
        );
        assert!(result.threshold > 0.0);
        assert!(result.threshold < 1e-307);
    }

    #[test]
    fn test_very_large_threshold() {
        let result = AnomalyResult::new(
            vec![true],
            vec![1e308],
            1e307,
        );
        assert!(result.threshold > 1e306);
    }

    #[test]
    fn test_mismatched_lengths_allowed() {
        // The struct allows mismatched lengths - this is a design choice
        // Tests that it doesn't panic
        let result = AnomalyResult::new(
            vec![true, false, true],
            vec![0.9], // Different length
            0.5,
        );
        assert_eq!(result.is_anomaly.len(), 3);
        assert_eq!(result.scores.len(), 1);
    }
}
