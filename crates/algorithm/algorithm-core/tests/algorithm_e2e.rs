//! End-to-end tests for algorithm crate
//!
//! Tests complete forecasting workflows using only this crate's API.

use algorithm_core::prelude::*;
use algorithm_core::utils::metrics::{mae, rmse, mape};

fn trend_data() -> Vec<f64> {
    (0..100).map(|i| 50.0 + 0.5 * i as f64).collect()
}

fn seasonal_data() -> Vec<f64> {
    (0..60).map(|i| {
        let t = i as f64;
        100.0 + t * 0.3 + 15.0 * (t * std::f64::consts::PI / 6.0).sin()
    }).collect()
}

#[test]
fn e2e_arima_forecast_workflow() {
    let data = trend_data();
    let train = &data[..80];
    let test = &data[80..];

    let mut model = Arima::new(1, 1, 0).unwrap();
    assert!(!model.is_fitted());

    model.fit(train).unwrap();
    assert!(model.is_fitted());

    let predictions = model.predict(test.len()).unwrap();
    assert_eq!(predictions.len(), test.len());

    let error = mae(test, &predictions);
    assert!(error < 5.0, "ARIMA MAE {} too high for linear trend", error);
}

#[test]
fn e2e_exponential_smoothing_workflow() {
    let data = trend_data();
    let train = &data[..80];
    let test = &data[80..];

    // SES
    let mut ses = SimpleExponentialSmoothing::new(0.3).unwrap();
    ses.fit(train).unwrap();
    let ses_pred = ses.predict(test.len()).unwrap();

    // Holt
    let mut holt = DoubleExponentialSmoothing::new(0.3, 0.1).unwrap();
    holt.fit(train).unwrap();
    let holt_pred = holt.predict(test.len()).unwrap();

    // Holt should outperform SES on trending data
    let ses_mae = mae(test, &ses_pred);
    let holt_mae = mae(test, &holt_pred);
    assert!(holt_mae < ses_mae, "Holt should beat SES on trend data");
}

#[test]
fn e2e_holt_winters_seasonal_workflow() {
    let data = seasonal_data();
    let period = 12;
    let train = &data[..48];
    let test = &data[48..];

    let mut hw = HoltWinters::new(0.3, 0.1, 0.2, period, SeasonalType::Additive).unwrap();
    hw.fit(train).unwrap();

    let predictions = hw.predict(test.len()).unwrap();
    assert_eq!(predictions.len(), test.len());

    let seasonal = hw.seasonal_components();
    assert_eq!(seasonal.len(), period);

    let error = mae(test, &predictions);
    assert!(error.is_finite());
}

#[test]
fn e2e_moving_average_workflow() {
    let data = trend_data();
    let train = &data[..80];

    let mut sma = SimpleMovingAverage::new(5).unwrap();
    sma.fit(train).unwrap();

    let smoothed = sma.smoothed_values();
    assert_eq!(smoothed.len(), train.len() - 4); // window - 1

    let predictions = sma.predict(10).unwrap();
    assert_eq!(predictions.len(), 10);
}

#[test]
fn e2e_linear_regression_workflow() {
    let data = trend_data();
    let train = &data[..80];
    let test = &data[80..];

    let mut lr = LinearRegression::new();
    lr.fit(train).unwrap();

    assert!(lr.slope() > 0.4 && lr.slope() < 0.6);
    assert!(lr.r_squared() > 0.99);

    let predictions = lr.predict(test.len()).unwrap();
    let error = mae(test, &predictions);
    assert!(error < 1.0, "Linear regression MAE {} too high", error);
}

#[test]
fn e2e_knn_workflow() {
    let data = trend_data();
    let train = &data[..80];
    let test = &data[80..];

    let mut knn = TimeSeriesKNN::new(5, 10, DistanceMetric::Euclidean).unwrap();
    assert!(!knn.is_fitted());

    knn.fit(train).unwrap();
    assert!(knn.is_fitted());
    assert!(knn.n_patterns() > 0);

    let predictions = knn.predict(test.len()).unwrap();
    assert_eq!(predictions.len(), test.len());
}

#[test]
fn e2e_model_comparison_workflow() {
    let data = trend_data();
    let train = &data[..80];
    let test = &data[80..];

    let models: Vec<(&str, Box<dyn Predictor>)> = vec![
        ("SES", Box::new(SimpleExponentialSmoothing::new(0.3).unwrap())),
        ("Holt", Box::new(DoubleExponentialSmoothing::new(0.3, 0.1).unwrap())),
        ("SMA", Box::new(SimpleMovingAverage::new(5).unwrap())),
        ("Linear", Box::new(LinearRegression::new())),
        ("ARIMA", Box::new(Arima::new(1, 1, 0).unwrap())),
    ];

    let mut results: Vec<(&str, f64)> = Vec::new();

    for (name, mut model) in models {
        model.fit(train).unwrap();
        let preds = model.predict(test.len()).unwrap();
        results.push((name, mae(test, &preds)));
    }

    // All models should produce valid results
    for (name, error) in &results {
        assert!(error.is_finite(), "{} produced invalid MAE", name);
    }
}

#[test]
fn e2e_metrics_workflow() {
    let actual: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
    let predicted: Vec<f64> = actual.iter().map(|x| x + 1.0).collect();

    let m = mae(&actual, &predicted);
    let r = rmse(&actual, &predicted);
    let p = mape(&actual, &predicted);

    assert!((m - 1.0).abs() < 0.001);
    assert!((r - 1.0).abs() < 0.001);
    assert!(p > 0.0 && p < 0.02);
}

#[test]
fn e2e_preprocessing_workflow() {
    use algorithm_core::utils::preprocessing::{normalize, standardize, difference};

    let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();

    // Normalize
    let (normalized, min, max) = normalize(&data);
    assert!(normalized.iter().all(|&x| x >= 0.0 && x <= 1.0));

    // Standardize
    let (standardized, mean, std) = standardize(&data);
    let std_mean: f64 = standardized.iter().sum::<f64>() / standardized.len() as f64;
    assert!(std_mean.abs() < 0.001);

    // Difference
    let differenced = difference(&data, 1);
    assert_eq!(differenced.len(), data.len() - 1);
    assert!(differenced.iter().all(|&x| (x - 2.0).abs() < 0.001));
}
