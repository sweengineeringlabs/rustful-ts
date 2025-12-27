//! Performance benchmarks for algorithm crate

use std::time::Instant;

use algorithm::prelude::*;
use algorithm::utils::preprocessing::{normalize, standardize, difference};
use algorithm::utils::metrics::{mae, mse, rmse};

fn generate_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            100.0 + t * 0.5 + 10.0 * (t * 0.1).sin() + (i as f64 * 0.01).cos() * 5.0
        })
        .collect()
}

fn bench<F>(name: &str, iterations: u32, mut f: F)
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..3 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations;

    println!(
        "{:30} {:>10.2?} total, {:>10.2?}/iter ({} iters)",
        name, elapsed, per_iter, iterations
    );
}

fn main() {
    println!("=== rustful-ts Performance Benchmarks ===\n");

    let data_1k = generate_data(1_000);
    let data_10k = generate_data(10_000);
    let data_100k = generate_data(100_000);

    // Preprocessing benchmarks
    println!("--- Preprocessing (10K points) ---");
    bench("normalize", 1000, || {
        let _ = normalize(&data_10k);
    });
    bench("standardize", 1000, || {
        let _ = standardize(&data_10k);
    });
    bench("difference(1)", 1000, || {
        let _ = difference(&data_10k, 1);
    });
    bench("difference(2)", 1000, || {
        let _ = difference(&data_10k, 2);
    });

    // Metrics benchmarks
    println!("\n--- Metrics (10K points) ---");
    let predicted: Vec<f64> = data_10k.iter().map(|x| x + 1.0).collect();
    bench("mae", 1000, || {
        let _ = mae(&data_10k, &predicted);
    });
    bench("mse", 1000, || {
        let _ = mse(&data_10k, &predicted);
    });
    bench("rmse", 1000, || {
        let _ = rmse(&data_10k, &predicted);
    });

    // SES benchmarks
    println!("\n--- Simple Exponential Smoothing ---");
    bench("SES fit (1K)", 1000, || {
        let mut model = SimpleExponentialSmoothing::new(0.3).unwrap();
        model.fit(&data_1k).unwrap();
    });
    bench("SES fit (10K)", 100, || {
        let mut model = SimpleExponentialSmoothing::new(0.3).unwrap();
        model.fit(&data_10k).unwrap();
    });
    bench("SES fit (100K)", 10, || {
        let mut model = SimpleExponentialSmoothing::new(0.3).unwrap();
        model.fit(&data_100k).unwrap();
    });

    let mut ses = SimpleExponentialSmoothing::new(0.3).unwrap();
    ses.fit(&data_10k).unwrap();
    bench("SES predict(100)", 10000, || {
        let _ = ses.predict(100).unwrap();
    });

    // Holt-Winters benchmarks
    println!("\n--- Holt-Winters ---");
    bench("HW fit (1K, period=12)", 100, || {
        let mut model = HoltWinters::new(0.3, 0.1, 0.1, 12, SeasonalType::Additive).unwrap();
        model.fit(&data_1k).unwrap();
    });
    bench("HW fit (10K, period=12)", 10, || {
        let mut model = HoltWinters::new(0.3, 0.1, 0.1, 12, SeasonalType::Additive).unwrap();
        model.fit(&data_10k).unwrap();
    });

    let mut hw = HoltWinters::new(0.3, 0.1, 0.1, 12, SeasonalType::Additive).unwrap();
    hw.fit(&data_1k).unwrap();
    bench("HW predict(100)", 10000, || {
        let _ = hw.predict(100).unwrap();
    });

    // ARIMA benchmarks
    println!("\n--- ARIMA ---");
    bench("ARIMA(1,1,0) fit (1K)", 100, || {
        let mut model = Arima::new(1, 1, 0).unwrap();
        model.fit(&data_1k).unwrap();
    });
    bench("ARIMA(2,1,1) fit (1K)", 100, || {
        let mut model = Arima::new(2, 1, 1).unwrap();
        model.fit(&data_1k).unwrap();
    });

    let mut arima = Arima::new(1, 1, 0).unwrap();
    arima.fit(&data_1k).unwrap();
    bench("ARIMA predict(100)", 10000, || {
        let _ = arima.predict(100).unwrap();
    });

    // Linear Regression benchmarks
    println!("\n--- Linear Regression ---");
    bench("LinReg fit (1K)", 1000, || {
        let mut model = LinearRegression::new();
        model.fit(&data_1k).unwrap();
    });
    bench("LinReg fit (10K)", 100, || {
        let mut model = LinearRegression::new();
        model.fit(&data_10k).unwrap();
    });
    bench("LinReg fit (100K)", 10, || {
        let mut model = LinearRegression::new();
        model.fit(&data_100k).unwrap();
    });

    // KNN benchmarks
    println!("\n--- KNN ---");
    bench("KNN(k=5,w=10) fit (1K)", 100, || {
        let mut model = TimeSeriesKNN::new(5, 10, DistanceMetric::Euclidean).unwrap();
        model.fit(&data_1k).unwrap();
    });

    let mut knn = TimeSeriesKNN::new(5, 10, DistanceMetric::Euclidean).unwrap();
    knn.fit(&data_1k).unwrap();
    bench("KNN predict(10)", 1000, || {
        let _ = knn.predict(10).unwrap();
    });

    // Moving Average benchmarks
    println!("\n--- Moving Average ---");
    bench("SMA(20) fit (10K)", 1000, || {
        let mut model = SimpleMovingAverage::new(20).unwrap();
        model.fit(&data_10k).unwrap();
    });
    bench("SMA(100) fit (10K)", 1000, || {
        let mut model = SimpleMovingAverage::new(100).unwrap();
        model.fit(&data_10k).unwrap();
    });

    println!("\n=== Benchmark Complete ===");
}
