//! Anomaly detector trait definition.

use crate::error::Result;
use crate::model::{Alert, AnomalyResult};

/// Anomaly detector trait.
///
/// Implementations detect anomalies in time series data.
pub trait AnomalyDetector: Send + Sync {
    /// Fit the detector to training data.
    fn fit(&mut self, data: &[f64]) -> Result<()>;

    /// Detect anomalies in data.
    fn detect(&self, data: &[f64]) -> Result<AnomalyResult>;

    /// Compute anomaly scores without thresholding.
    fn score(&self, data: &[f64]) -> Result<Vec<f64>>;

    /// Check if detector has been fitted.
    fn is_fitted(&self) -> bool;
}

/// Real-time monitoring trait.
pub trait MonitoringStream<D: AnomalyDetector>: Send + Sync {
    /// Push a new value and check for anomalies.
    fn push(&mut self, value: f64) -> Result<Option<Alert>>;

    /// Get current buffer contents.
    fn buffer(&self) -> &[f64];

    /// Reset the monitor state.
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AnomalyError;
    use crate::model::AlertSeverity;

    // ===== Mock AnomalyDetector Implementation =====

    /// A simple mock detector for testing the trait.
    struct MockDetector {
        fitted: bool,
        threshold: f64,
        mean: f64,
        std_dev: f64,
    }

    impl MockDetector {
        fn new(threshold: f64) -> Self {
            Self {
                fitted: false,
                threshold,
                mean: 0.0,
                std_dev: 1.0,
            }
        }
    }

    impl AnomalyDetector for MockDetector {
        fn fit(&mut self, data: &[f64]) -> Result<()> {
            if data.len() < 2 {
                return Err(AnomalyError::InsufficientData {
                    required: 2,
                    got: data.len(),
                });
            }
            let n = data.len() as f64;
            self.mean = data.iter().sum::<f64>() / n;
            let variance = data.iter().map(|x| (x - self.mean).powi(2)).sum::<f64>() / n;
            self.std_dev = variance.sqrt();
            if self.std_dev == 0.0 {
                self.std_dev = 1.0;
            }
            self.fitted = true;
            Ok(())
        }

        fn detect(&self, data: &[f64]) -> Result<AnomalyResult> {
            if !self.fitted {
                return Err(AnomalyError::NotFitted);
            }
            let scores = self.score(data)?;
            let is_anomaly: Vec<bool> = scores.iter().map(|&s| s > self.threshold).collect();
            Ok(AnomalyResult::new(is_anomaly, scores, self.threshold))
        }

        fn score(&self, data: &[f64]) -> Result<Vec<f64>> {
            if !self.fitted {
                return Err(AnomalyError::NotFitted);
            }
            Ok(data
                .iter()
                .map(|x| ((x - self.mean) / self.std_dev).abs())
                .collect())
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }
    }

    // ===== Mock MonitoringStream Implementation =====

    struct MockMonitor {
        detector: MockDetector,
        buffer: Vec<f64>,
        buffer_size: usize,
        alert_threshold: f64,
    }

    impl MockMonitor {
        fn new(detector: MockDetector, buffer_size: usize, alert_threshold: f64) -> Self {
            Self {
                detector,
                buffer: Vec::with_capacity(buffer_size),
                buffer_size,
                alert_threshold,
            }
        }
    }

    impl MonitoringStream<MockDetector> for MockMonitor {
        fn push(&mut self, value: f64) -> Result<Option<Alert>> {
            self.buffer.push(value);
            if self.buffer.len() > self.buffer_size {
                self.buffer.remove(0);
            }

            if !self.detector.is_fitted() {
                return Err(AnomalyError::NotFitted);
            }

            let scores = self.detector.score(&[value])?;
            let score = scores[0];

            if score > self.alert_threshold {
                let severity = if score > self.alert_threshold * 2.0 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                };
                Ok(Some(Alert {
                    timestamp: self.buffer.len() as u64,
                    value,
                    score,
                    severity,
                    message: format!("Anomaly detected with score {:.2}", score),
                }))
            } else {
                Ok(None)
            }
        }

        fn buffer(&self) -> &[f64] {
            &self.buffer
        }

        fn reset(&mut self) {
            self.buffer.clear();
        }
    }

    // ===== AnomalyDetector Trait Tests =====

    #[test]
    fn test_detector_initial_state() {
        let detector = MockDetector::new(2.0);
        assert!(!detector.is_fitted());
    }

    #[test]
    fn test_detector_fit_success() {
        let mut detector = MockDetector::new(2.0);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = detector.fit(&data);
        assert!(result.is_ok());
        assert!(detector.is_fitted());
    }

    #[test]
    fn test_detector_fit_insufficient_data() {
        let mut detector = MockDetector::new(2.0);
        let result = detector.fit(&[1.0]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AnomalyError::InsufficientData { required: 2, got: 1 }
        ));
    }

    #[test]
    fn test_detector_fit_empty_data() {
        let mut detector = MockDetector::new(2.0);
        let result = detector.fit(&[]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AnomalyError::InsufficientData { required: 2, got: 0 }
        ));
    }

    #[test]
    fn test_detector_detect_not_fitted() {
        let detector = MockDetector::new(2.0);
        let result = detector.detect(&[1.0, 2.0, 3.0]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AnomalyError::NotFitted));
    }

    #[test]
    fn test_detector_detect_after_fit() {
        let mut detector = MockDetector::new(2.0);
        detector.fit(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let result = detector.detect(&[3.0, 10.0, 3.0]);
        assert!(result.is_ok());
        let anomaly_result = result.unwrap();
        assert_eq!(anomaly_result.is_anomaly.len(), 3);
        assert_eq!(anomaly_result.scores.len(), 3);
    }

    #[test]
    fn test_detector_detect_finds_anomalies() {
        let mut detector = MockDetector::new(2.0);
        // Mean = 3, std = sqrt(2)
        detector.fit(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        // 100.0 is clearly an anomaly
        let result = detector.detect(&[3.0, 100.0]).unwrap();
        assert!(!result.is_anomaly[0]); // 3.0 is the mean, z-score ~0
        assert!(result.is_anomaly[1]); // 100.0 should be anomaly
    }

    #[test]
    fn test_detector_score_not_fitted() {
        let detector = MockDetector::new(2.0);
        let result = detector.score(&[1.0, 2.0]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AnomalyError::NotFitted));
    }

    #[test]
    fn test_detector_score_after_fit() {
        let mut detector = MockDetector::new(2.0);
        detector.fit(&[0.0, 10.0]).unwrap(); // Mean = 5, std = 5
        let scores = detector.score(&[5.0, 0.0, 10.0]).unwrap();
        assert_eq!(scores.len(), 3);
        // Score at mean should be ~0
        assert!(scores[0] < 0.1);
        // Scores at edges should be ~1
        assert!((scores[1] - 1.0).abs() < 0.1);
        assert!((scores[2] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_detector_score_empty_data() {
        let mut detector = MockDetector::new(2.0);
        detector.fit(&[1.0, 2.0, 3.0]).unwrap();
        let scores = detector.score(&[]).unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn test_detector_refit() {
        let mut detector = MockDetector::new(2.0);
        detector.fit(&[1.0, 2.0, 3.0]).unwrap();
        assert!(detector.is_fitted());

        // Refit with new data
        detector.fit(&[10.0, 20.0, 30.0]).unwrap();
        assert!(detector.is_fitted());

        // Scores should reflect new fit
        let scores = detector.score(&[20.0]).unwrap();
        assert!(scores[0] < 0.1); // 20 is the new mean
    }

    #[test]
    fn test_detector_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockDetector>();
    }

    #[test]
    fn test_detector_trait_object() {
        let mut detector: Box<dyn AnomalyDetector> = Box::new(MockDetector::new(2.0));
        assert!(!detector.is_fitted());
        detector.fit(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(detector.is_fitted());

        let result = detector.detect(&[3.0, 100.0]).unwrap();
        assert_eq!(result.is_anomaly.len(), 2);
    }

    // ===== MonitoringStream Trait Tests =====

    fn create_fitted_monitor() -> MockMonitor {
        let mut detector = MockDetector::new(2.0);
        detector.fit(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap();
        MockMonitor::new(detector, 10, 2.0)
    }

    #[test]
    fn test_monitor_initial_state() {
        let monitor = create_fitted_monitor();
        assert!(monitor.buffer().is_empty());
    }

    #[test]
    fn test_monitor_push_normal_value() {
        let mut monitor = create_fitted_monitor();
        let result = monitor.push(5.0); // Near mean
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        assert_eq!(monitor.buffer().len(), 1);
    }

    #[test]
    fn test_monitor_push_anomaly_value() {
        let mut monitor = create_fitted_monitor();
        let result = monitor.push(100.0); // Far from mean
        assert!(result.is_ok());
        let alert = result.unwrap();
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert!((alert.value - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_monitor_push_critical_anomaly() {
        let mut monitor = create_fitted_monitor();
        let result = monitor.push(1000.0); // Very far from mean
        let alert = result.unwrap().unwrap();
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_monitor_push_warning_anomaly() {
        let mut monitor = create_fitted_monitor();
        // Push a value that triggers warning but not critical
        // Need to find a value with score between 2.0 and 4.0
        let result = monitor.push(15.0);
        if let Ok(Some(alert)) = result {
            assert_eq!(alert.severity, AlertSeverity::Warning);
        }
    }

    #[test]
    fn test_monitor_buffer_grows() {
        let mut monitor = create_fitted_monitor();
        for i in 0..5 {
            monitor.push(i as f64).unwrap();
        }
        assert_eq!(monitor.buffer().len(), 5);
    }

    #[test]
    fn test_monitor_buffer_limit() {
        let mut monitor = create_fitted_monitor();
        // Push more than buffer size
        for i in 0..15 {
            monitor.push(i as f64).unwrap();
        }
        assert_eq!(monitor.buffer().len(), 10); // Buffer size is 10
    }

    #[test]
    fn test_monitor_buffer_contents() {
        let mut monitor = create_fitted_monitor();
        monitor.push(1.0).unwrap();
        monitor.push(2.0).unwrap();
        monitor.push(3.0).unwrap();
        let buffer = monitor.buffer();
        assert_eq!(buffer, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_monitor_reset() {
        let mut monitor = create_fitted_monitor();
        monitor.push(1.0).unwrap();
        monitor.push(2.0).unwrap();
        assert!(!monitor.buffer().is_empty());

        monitor.reset();
        assert!(monitor.buffer().is_empty());
    }

    #[test]
    fn test_monitor_push_after_reset() {
        let mut monitor = create_fitted_monitor();
        monitor.push(1.0).unwrap();
        monitor.reset();
        monitor.push(5.0).unwrap();
        assert_eq!(monitor.buffer().len(), 1);
        assert!((monitor.buffer()[0] - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_monitor_not_fitted_error() {
        let detector = MockDetector::new(2.0); // Not fitted
        let mut monitor = MockMonitor::new(detector, 10, 2.0);
        let result = monitor.push(1.0);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AnomalyError::NotFitted));
    }

    #[test]
    fn test_monitor_alert_message() {
        let mut monitor = create_fitted_monitor();
        let result = monitor.push(100.0).unwrap().unwrap();
        assert!(result.message.contains("Anomaly detected"));
        assert!(result.message.contains("score"));
    }

    #[test]
    fn test_monitor_alert_timestamp() {
        let mut monitor = create_fitted_monitor();
        monitor.push(5.0).unwrap();
        monitor.push(5.0).unwrap();
        let alert = monitor.push(100.0).unwrap().unwrap();
        assert_eq!(alert.timestamp, 3); // Third push
    }

    #[test]
    fn test_monitor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockMonitor>();
    }

    // ===== Integration Tests =====

    #[test]
    fn test_detector_monitor_integration() {
        // Create and fit detector
        let mut detector = MockDetector::new(2.0);
        let training_data: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        detector.fit(&training_data).unwrap();

        // Create monitor
        let mut monitor = MockMonitor::new(detector, 20, 2.0);

        // Push normal values
        let mut alerts = Vec::new();
        for i in 0..10 {
            if let Ok(Some(alert)) = monitor.push(i as f64) {
                alerts.push(alert);
            }
        }
        assert!(alerts.is_empty()); // Normal values shouldn't trigger alerts

        // Push anomaly
        if let Ok(Some(alert)) = monitor.push(1000.0) {
            alerts.push(alert);
        }
        assert_eq!(alerts.len(), 1);
    }

    #[test]
    fn test_multiple_detectors() {
        let mut detector1 = MockDetector::new(1.0);
        let mut detector2 = MockDetector::new(3.0);

        detector1.fit(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        detector2.fit(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        let result1 = detector1.detect(&[10.0]).unwrap();
        let result2 = detector2.detect(&[10.0]).unwrap();

        // detector1 with lower threshold should detect more anomalies
        assert!(result1.is_anomaly[0]);
        // detector2 with higher threshold may or may not detect
        // Both should have scores
        assert!(!result1.scores.is_empty());
        assert!(!result2.scores.is_empty());
    }

    #[test]
    fn test_detector_with_constant_data() {
        let mut detector = MockDetector::new(2.0);
        // All same values - std dev will be 0, handled specially
        detector.fit(&[5.0, 5.0, 5.0, 5.0, 5.0]).unwrap();
        let scores = detector.score(&[5.0, 6.0]).unwrap();
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_detector_with_negative_values() {
        let mut detector = MockDetector::new(2.0);
        detector.fit(&[-10.0, -5.0, 0.0, 5.0, 10.0]).unwrap();
        let result = detector.detect(&[-100.0, 0.0, 100.0]).unwrap();
        assert!(result.is_anomaly[0]); // -100 is anomaly
        assert!(!result.is_anomaly[1]); // 0 is the mean
        assert!(result.is_anomaly[2]); // 100 is anomaly
    }
}
