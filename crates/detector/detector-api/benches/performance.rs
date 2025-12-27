//! Performance benchmarks for detector crate

use std::hint::black_box;
use std::time::Instant;

use detector_api::prelude::*;

fn generate_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            100.0 + (t * 0.1).sin() * 10.0
        })
        .collect()
}

fn bench<F, R>(name: &str, iterations: u32, mut f: F)
where
    F: FnMut() -> R,
{
    // Warmup
    for _ in 0..3 {
        black_box(f());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(f());
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations;

    println!(
        "{:30} {:>10.2?} total, {:>10.2?}/iter ({} iters)",
        name, elapsed, per_iter, iterations
    );
}

fn main() {
    println!("=== Detector Performance Benchmarks ===\n");

    let data_1k = generate_data(1_000);
    let data_10k = generate_data(10_000);
    let data_100k = generate_data(100_000);

    // Z-Score benchmarks
    println!("--- Z-Score Detector ---");
    bench("ZScore fit (1K)", 1000, || {
        let mut detector = ZScoreDetector::new(3.0).unwrap();
        detector.fit(&data_1k).unwrap();
        detector
    });
    bench("ZScore fit (10K)", 100, || {
        let mut detector = ZScoreDetector::new(3.0).unwrap();
        detector.fit(&data_10k).unwrap();
        detector
    });
    bench("ZScore fit (100K)", 10, || {
        let mut detector = ZScoreDetector::new(3.0).unwrap();
        detector.fit(&data_100k).unwrap();
        detector
    });

    let mut zscore = ZScoreDetector::new(3.0).unwrap();
    zscore.fit(&data_10k).unwrap();
    bench("ZScore detect (10K)", 1000, || {
        zscore.detect(&data_10k).unwrap()
    });
    bench("ZScore score (10K)", 1000, || {
        zscore.score(&data_10k).unwrap()
    });

    // IQR benchmarks
    println!("\n--- IQR Detector ---");
    bench("IQR fit (1K)", 1000, || {
        let mut detector = IQRDetector::new(1.5).unwrap();
        detector.fit(&data_1k).unwrap();
        detector
    });
    bench("IQR fit (10K)", 100, || {
        let mut detector = IQRDetector::new(1.5).unwrap();
        detector.fit(&data_10k).unwrap();
        detector
    });
    bench("IQR fit (100K)", 10, || {
        let mut detector = IQRDetector::new(1.5).unwrap();
        detector.fit(&data_100k).unwrap();
        detector
    });

    let mut iqr = IQRDetector::new(1.5).unwrap();
    iqr.fit(&data_10k).unwrap();
    bench("IQR detect (10K)", 1000, || {
        iqr.detect(&data_10k).unwrap()
    });

    println!("\n=== Benchmark Complete ===");
}
