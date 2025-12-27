//! Integration tests for rustful-anomaly

use anomaly::{ZScoreDetector, IQRDetector, AnomalyDetector};

fn normal_data() -> Vec<f64> {
    vec![10.0, 11.0, 10.5, 11.5, 10.2, 11.3, 10.8, 11.1, 10.6, 11.4,
         10.3, 11.2, 10.9, 11.0, 10.7, 11.3, 10.4, 11.1, 10.8, 11.2]
}

fn data_with_anomalies() -> Vec<f64> {
    vec![10.5, 11.0, 25.0, 10.8, 11.2, -5.0, 10.9, 11.1, 10.7, 30.0]
}

#[test]
fn test_zscore_detector_fit() {
    let data = normal_data();
    let mut detector = ZScoreDetector::new(3.0);

    detector.fit(&data).unwrap();

    // After fitting, should be able to detect
    let result = detector.detect(&data).unwrap();
    assert_eq!(result.is_anomaly.len(), data.len());
}

#[test]
fn test_zscore_detector_detects_anomalies() {
    let training = normal_data();
    let test = data_with_anomalies();

    let mut detector = ZScoreDetector::new(3.0);
    detector.fit(&training).unwrap();

    let result = detector.detect(&test).unwrap();

    // Should detect the extreme values (25.0, -5.0, 30.0)
    let anomaly_count: usize = result.is_anomaly.iter().filter(|&&x| x).count();
    assert!(anomaly_count >= 3);
}

#[test]
fn test_zscore_scores() {
    let training = normal_data();
    let test = data_with_anomalies();

    let mut detector = ZScoreDetector::new(3.0);
    detector.fit(&training).unwrap();

    let scores = detector.score(&test).unwrap();

    // Anomalies should have high absolute scores
    assert!(scores[2].abs() > 10.0); // 25.0 is far from mean
    assert!(scores[5].abs() > 10.0); // -5.0 is far from mean
    assert!(scores[9].abs() > 10.0); // 30.0 is far from mean
}

#[test]
fn test_iqr_detector_fit() {
    let data = normal_data();
    let mut detector = IQRDetector::new(1.5);

    detector.fit(&data).unwrap();

    let result = detector.detect(&data).unwrap();
    assert_eq!(result.is_anomaly.len(), data.len());
}

#[test]
fn test_iqr_detector_detects_anomalies() {
    let training = normal_data();
    let test = data_with_anomalies();

    let mut detector = IQRDetector::new(1.5);
    detector.fit(&training).unwrap();

    let result = detector.detect(&test).unwrap();

    // Should detect the extreme values
    let anomaly_count: usize = result.is_anomaly.iter().filter(|&&x| x).count();
    assert!(anomaly_count >= 3);
}

#[test]
fn test_detector_no_false_positives_on_normal() {
    let data = normal_data();

    // Z-Score detector
    let mut zscore = ZScoreDetector::new(3.0);
    zscore.fit(&data).unwrap();
    let result = zscore.detect(&data).unwrap();
    let zscore_anomalies: usize = result.is_anomaly.iter().filter(|&&x| x).count();

    // Should have very few or no false positives on normal data
    assert!(zscore_anomalies <= 1);

    // IQR detector
    let mut iqr = IQRDetector::new(1.5);
    iqr.fit(&data).unwrap();
    let result = iqr.detect(&data).unwrap();
    let iqr_anomalies: usize = result.is_anomaly.iter().filter(|&&x| x).count();

    assert!(iqr_anomalies <= 1);
}

#[test]
fn test_default_thresholds() {
    let zscore = ZScoreDetector::default();
    let iqr = IQRDetector::default();

    // Defaults should be reasonable (3.0 for Z-score, 1.5 for IQR)
    let data = normal_data();

    let mut z = zscore;
    z.fit(&data).unwrap();

    let mut i = iqr;
    i.fit(&data).unwrap();
}
