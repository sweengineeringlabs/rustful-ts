//! End-to-end tests for rustful-anomaly crate
//!
//! Tests complete anomaly detection workflows using only this crate's API.

use anomaly::{AnomalyDetector, ZScoreDetector, IQRDetector};

fn normal_data() -> Vec<f64> {
    (0..100).map(|i| 50.0 + (i as f64 * 0.1)).collect()
}

fn data_with_anomalies() -> Vec<f64> {
    let mut data = normal_data();
    data[10] += 50.0;  // Spike
    data[30] -= 40.0;  // Drop
    data[50] += 60.0;  // Spike
    data[70] -= 45.0;  // Drop
    data[90] += 55.0;  // Spike
    data
}

#[test]
fn e2e_zscore_detection_workflow() {
    let training = normal_data();
    let test = data_with_anomalies();

    let mut detector = ZScoreDetector::new(3.0);

    // Fit on normal data
    detector.fit(&training).unwrap();

    // Detect on data with anomalies
    let result = detector.detect(&test).unwrap();

    assert_eq!(result.is_anomaly.len(), test.len());
    assert_eq!(result.scores.len(), test.len());

    // Count detected anomalies
    let detected: Vec<usize> = result.is_anomaly
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();

    // Should detect the injected anomalies
    assert!(detected.contains(&10) || detected.contains(&50) || detected.contains(&90),
        "Should detect at least one spike");
}

#[test]
fn e2e_iqr_detection_workflow() {
    let training = normal_data();
    let test = data_with_anomalies();

    let mut detector = IQRDetector::new(1.5);

    detector.fit(&training).unwrap();
    let result = detector.detect(&test).unwrap();

    assert_eq!(result.is_anomaly.len(), test.len());

    let anomaly_count: usize = result.is_anomaly.iter().filter(|&&b| b).count();
    assert!(anomaly_count >= 3, "IQR should detect at least 3 anomalies");
}

#[test]
fn e2e_scoring_workflow() {
    let training = normal_data();
    let test = data_with_anomalies();

    let mut detector = ZScoreDetector::new(3.0);
    detector.fit(&training).unwrap();

    let scores = detector.score(&test).unwrap();
    assert_eq!(scores.len(), test.len());

    // Anomaly positions should have high absolute scores
    assert!(scores[10].abs() > 3.0);
    assert!(scores[50].abs() > 3.0);
    assert!(scores[90].abs() > 3.0);

    // Normal positions should have low scores
    assert!(scores[0].abs() < 3.0);
    assert!(scores[5].abs() < 3.0);
}

#[test]
fn e2e_no_false_positives_on_clean_data() {
    let data = normal_data();

    let mut zscore = ZScoreDetector::new(3.0);
    zscore.fit(&data).unwrap();
    let z_result = zscore.detect(&data).unwrap();

    let mut iqr = IQRDetector::new(1.5);
    iqr.fit(&data).unwrap();
    let iqr_result = iqr.detect(&data).unwrap();

    let z_fp: usize = z_result.is_anomaly.iter().filter(|&&b| b).count();
    let iqr_fp: usize = iqr_result.is_anomaly.iter().filter(|&&b| b).count();

    assert!(z_fp <= 2, "Z-Score too many false positives: {}", z_fp);
    assert!(iqr_fp <= 2, "IQR too many false positives: {}", iqr_fp);
}

#[test]
fn e2e_threshold_sensitivity() {
    let test = data_with_anomalies();

    // Strict threshold
    let mut strict = ZScoreDetector::new(4.0);
    strict.fit(&test).unwrap();
    let strict_result = strict.detect(&test).unwrap();

    // Lenient threshold
    let mut lenient = ZScoreDetector::new(2.0);
    lenient.fit(&test).unwrap();
    let lenient_result = lenient.detect(&test).unwrap();

    let strict_count: usize = strict_result.is_anomaly.iter().filter(|&&b| b).count();
    let lenient_count: usize = lenient_result.is_anomaly.iter().filter(|&&b| b).count();

    // Lenient should detect more
    assert!(lenient_count >= strict_count);
}

#[test]
fn e2e_default_detectors() {
    let data = data_with_anomalies();

    let mut zscore = ZScoreDetector::default();
    zscore.fit(&data).unwrap();
    let _ = zscore.detect(&data).unwrap();

    let mut iqr = IQRDetector::default();
    iqr.fit(&data).unwrap();
    let _ = iqr.detect(&data).unwrap();
}

#[test]
fn e2e_incremental_detection() {
    let normal = normal_data();

    let mut detector = ZScoreDetector::new(3.0);
    detector.fit(&normal).unwrap();

    // Simulate streaming data
    let mut detected_count = 0;
    for i in 0..20 {
        let value = if i == 10 { 200.0 } else { 55.0 };
        let single = vec![value];
        let result = detector.detect(&single).unwrap();
        if result.is_anomaly[0] {
            detected_count += 1;
        }
    }

    assert!(detected_count >= 1, "Should detect the spike at position 10");
}
