//! Performance benchmarks for detector crate

use bench_harness::{bench_print, footer, header, section};
use detector_api::prelude::*;

fn generate_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            100.0 + (t * 0.1).sin() * 10.0
        })
        .collect()
}

fn main() {
    header("Detector Performance Benchmarks");

    let data_1k = generate_data(1_000);
    let data_10k = generate_data(10_000);
    let data_100k = generate_data(100_000);

    // Z-Score benchmarks
    section("Z-Score Detector");
    bench_print("ZScore fit (1K)", 1000, || {
        let mut detector = ZScoreDetector::new(3.0).unwrap();
        detector.fit(&data_1k).unwrap();
        detector
    });
    bench_print("ZScore fit (10K)", 100, || {
        let mut detector = ZScoreDetector::new(3.0).unwrap();
        detector.fit(&data_10k).unwrap();
        detector
    });
    bench_print("ZScore fit (100K)", 10, || {
        let mut detector = ZScoreDetector::new(3.0).unwrap();
        detector.fit(&data_100k).unwrap();
        detector
    });

    let mut zscore = ZScoreDetector::new(3.0).unwrap();
    zscore.fit(&data_10k).unwrap();
    bench_print("ZScore detect (10K)", 1000, || {
        zscore.detect(&data_10k).unwrap()
    });
    bench_print("ZScore score (10K)", 1000, || {
        zscore.score(&data_10k).unwrap()
    });

    // IQR benchmarks
    section("IQR Detector");
    bench_print("IQR fit (1K)", 1000, || {
        let mut detector = IQRDetector::new(1.5).unwrap();
        detector.fit(&data_1k).unwrap();
        detector
    });
    bench_print("IQR fit (10K)", 100, || {
        let mut detector = IQRDetector::new(1.5).unwrap();
        detector.fit(&data_10k).unwrap();
        detector
    });
    bench_print("IQR fit (100K)", 10, || {
        let mut detector = IQRDetector::new(1.5).unwrap();
        detector.fit(&data_100k).unwrap();
        detector
    });

    let mut iqr = IQRDetector::new(1.5).unwrap();
    iqr.fit(&data_10k).unwrap();
    bench_print("IQR detect (10K)", 1000, || {
        iqr.detect(&data_10k).unwrap()
    });

    footer();
}
