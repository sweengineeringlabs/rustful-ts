//! Basic example demonstrating anomaly detection
//!
//! Run with: cargo run --example basic -p rustful-anomaly

use anomaly::{ZScoreDetector, IQRDetector, AnomalyDetector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== rustful-anomaly Basic Examples ===\n");

    // Sample data with anomalies
    let training_data = vec![
        10.0, 11.0, 10.5, 11.5, 10.2, 11.3, 10.8, 11.1, 10.6, 11.4,
        10.3, 11.2, 10.9, 11.0, 10.7, 11.3, 10.4, 11.1, 10.8, 11.2,
    ];

    // Test data with some anomalies
    let test_data = vec![
        10.5, 11.0, 25.0, 10.8, 11.2, -5.0, 10.9, 11.1, 10.7, 30.0,
    ];

    println!("Training data (normal): {:?}", &training_data[..10]);
    println!("Test data (with anomalies): {:?}\n", test_data);

    // 1. Z-Score Detector
    println!("1. Z-Score Detector (threshold=3.0)");
    let mut zscore = ZScoreDetector::new(3.0);
    zscore.fit(&training_data)?;
    let zscore_result = zscore.detect(&test_data)?;
    println!("   Anomalies: {:?}", zscore_result.is_anomaly);
    println!("   Scores: {:?}\n", zscore_result.scores.iter().map(|s| format!("{:.2}", s)).collect::<Vec<_>>());

    // 2. IQR Detector
    println!("2. IQR Detector (multiplier=1.5)");
    let mut iqr = IQRDetector::new(1.5);
    iqr.fit(&training_data)?;
    let iqr_result = iqr.detect(&test_data)?;
    println!("   Anomalies: {:?}", iqr_result.is_anomaly);
    println!("   Scores: {:?}\n", iqr_result.scores.iter().map(|s| format!("{:.2}", s)).collect::<Vec<_>>());

    // Count anomalies
    let zscore_count = zscore_result.is_anomaly.iter().filter(|&&x| x).count();
    let iqr_count = iqr_result.is_anomaly.iter().filter(|&&x| x).count();
    println!("Summary:");
    println!("   Z-Score detected {} anomalies", zscore_count);
    println!("   IQR detected {} anomalies", iqr_count);

    println!("\n=== Examples Complete ===");
    Ok(())
}
