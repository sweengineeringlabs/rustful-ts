//! Performance benchmarks for algorithm crate

use algorithm::prelude::*;
use algorithm::utils::metrics::{mae, mse, rmse};
use algorithm::utils::preprocessing::{difference, normalize, standardize};
use bench::{bench_print, footer, header, section};

fn generate_data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            100.0 + t * 0.5 + 10.0 * (t * 0.1).sin() + (i as f64 * 0.01).cos() * 5.0
        })
        .collect()
}

fn main() {
    header("rustful-ts Performance Benchmarks");

    let data_1k = generate_data(1_000);
    let data_10k = generate_data(10_000);
    let data_100k = generate_data(100_000);

    // Preprocessing benchmarks
    section("Preprocessing (10K points)");
    bench_print("normalize", 1000, || normalize(&data_10k));
    bench_print("standardize", 1000, || standardize(&data_10k));
    bench_print("difference(1)", 1000, || difference(&data_10k, 1));
    bench_print("difference(2)", 1000, || difference(&data_10k, 2));

    // Metrics benchmarks
    section("Metrics (10K points)");
    let predicted: Vec<f64> = data_10k.iter().map(|x| x + 1.0).collect();
    bench_print("mae", 1000, || mae(&data_10k, &predicted));
    bench_print("mse", 1000, || mse(&data_10k, &predicted));
    bench_print("rmse", 1000, || rmse(&data_10k, &predicted));

    // SES benchmarks
    section("Simple Exponential Smoothing");
    bench_print("SES fit (1K)", 1000, || {
        let mut model = SimpleExponentialSmoothing::new(0.3).unwrap();
        model.fit(&data_1k).unwrap();
        model
    });
    bench_print("SES fit (10K)", 100, || {
        let mut model = SimpleExponentialSmoothing::new(0.3).unwrap();
        model.fit(&data_10k).unwrap();
        model
    });
    bench_print("SES fit (100K)", 10, || {
        let mut model = SimpleExponentialSmoothing::new(0.3).unwrap();
        model.fit(&data_100k).unwrap();
        model
    });

    let mut ses = SimpleExponentialSmoothing::new(0.3).unwrap();
    ses.fit(&data_10k).unwrap();
    bench_print("SES predict(100)", 10000, || ses.predict(100).unwrap());

    // Holt-Winters benchmarks
    section("Holt-Winters");
    bench_print("HW fit (1K, period=12)", 100, || {
        let mut model = HoltWinters::new(0.3, 0.1, 0.1, 12, SeasonalType::Additive).unwrap();
        model.fit(&data_1k).unwrap();
        model
    });
    bench_print("HW fit (10K, period=12)", 10, || {
        let mut model = HoltWinters::new(0.3, 0.1, 0.1, 12, SeasonalType::Additive).unwrap();
        model.fit(&data_10k).unwrap();
        model
    });

    let mut hw = HoltWinters::new(0.3, 0.1, 0.1, 12, SeasonalType::Additive).unwrap();
    hw.fit(&data_1k).unwrap();
    bench_print("HW predict(100)", 10000, || hw.predict(100).unwrap());

    // ARIMA benchmarks
    section("ARIMA");
    bench_print("ARIMA(1,1,0) fit (1K)", 100, || {
        let mut model = Arima::new(1, 1, 0).unwrap();
        model.fit(&data_1k).unwrap();
        model
    });
    bench_print("ARIMA(2,1,1) fit (1K)", 100, || {
        let mut model = Arima::new(2, 1, 1).unwrap();
        model.fit(&data_1k).unwrap();
        model
    });

    let mut arima = Arima::new(1, 1, 0).unwrap();
    arima.fit(&data_1k).unwrap();
    bench_print("ARIMA predict(100)", 10000, || arima.predict(100).unwrap());

    // Linear Regression benchmarks
    section("Linear Regression");
    bench_print("LinReg fit (1K)", 1000, || {
        let mut model = LinearRegression::new();
        model.fit(&data_1k).unwrap();
        model
    });
    bench_print("LinReg fit (10K)", 100, || {
        let mut model = LinearRegression::new();
        model.fit(&data_10k).unwrap();
        model
    });
    bench_print("LinReg fit (100K)", 10, || {
        let mut model = LinearRegression::new();
        model.fit(&data_100k).unwrap();
        model
    });

    // KNN benchmarks
    section("KNN");
    bench_print("KNN(k=5,w=10) fit (1K)", 100, || {
        let mut model = TimeSeriesKNN::new(5, 10, DistanceMetric::Euclidean).unwrap();
        model.fit(&data_1k).unwrap();
        model
    });

    let mut knn = TimeSeriesKNN::new(5, 10, DistanceMetric::Euclidean).unwrap();
    knn.fit(&data_1k).unwrap();
    bench_print("KNN predict(10)", 1000, || knn.predict(10).unwrap());

    // Moving Average benchmarks
    section("Moving Average");
    bench_print("SMA(20) fit (10K)", 1000, || {
        let mut model = SimpleMovingAverage::new(20).unwrap();
        model.fit(&data_10k).unwrap();
        model
    });
    bench_print("SMA(100) fit (10K)", 1000, || {
        let mut model = SimpleMovingAverage::new(100).unwrap();
        model.fit(&data_10k).unwrap();
        model
    });

    footer();
}
