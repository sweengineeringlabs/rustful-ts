//! Unit tests for algorithm crate
//!
//! Extracted from source modules for better organization.

use algorithm::prelude::*;
use algorithm::utils::metrics::{mae, mse, r_squared};
use algorithm::utils::preprocessing::{normalize, standardize, destandardize, difference};
use algorithm::utils::validation::{expanding_window_split, sliding_window_split, train_test_split};

// ============================================================================
// Validation Tests
// ============================================================================

#[test]
fn test_expanding_window() {
    let splits = expanding_window_split(100, 50, 10, 10);
    assert!(!splits.is_empty());

    // First split should train on 0..50, test on 50..60
    let (train, test) = &splits[0];
    assert_eq!(train.start, 0);
    assert_eq!(train.end, 50);
    assert_eq!(test.start, 50);
    assert_eq!(test.end, 60);
}

#[test]
fn test_sliding_window() {
    let splits = sliding_window_split(100, 30, 10, 10);
    assert!(!splits.is_empty());

    // All training windows should have same size
    for (train, _) in &splits {
        assert_eq!(train.end - train.start, 30);
    }
}

#[test]
fn test_train_test_split() {
    let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let (train, test) = train_test_split(&data, 0.2);

    assert_eq!(train.len(), 80);
    assert_eq!(test.len(), 20);
    assert_eq!(train[0], 0.0);
    assert_eq!(test[0], 80.0);
}

// ============================================================================
// Metrics Tests
// ============================================================================

#[test]
fn test_mae() {
    let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!((mae(&actual, &predicted) - 0.0).abs() < 1e-10);

    let predicted = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    assert!((mae(&actual, &predicted) - 1.0).abs() < 1e-10);
}

#[test]
fn test_mse() {
    let actual = vec![1.0, 2.0, 3.0];
    let predicted = vec![2.0, 3.0, 4.0];
    assert!((mse(&actual, &predicted) - 1.0).abs() < 1e-10);
}

#[test]
fn test_r_squared() {
    let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let predicted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!((r_squared(&actual, &predicted) - 1.0).abs() < 1e-10);
}

// ============================================================================
// Preprocessing Tests
// ============================================================================

#[test]
fn test_normalize() {
    let data = vec![0.0, 5.0, 10.0];
    let (normalized, min, max) = normalize(&data);

    assert!((min - 0.0).abs() < 1e-10);
    assert!((max - 10.0).abs() < 1e-10);
    assert!((normalized[0] - 0.0).abs() < 1e-10);
    assert!((normalized[1] - 0.5).abs() < 1e-10);
    assert!((normalized[2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_standardize() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (standardized, mean, std_dev) = standardize(&data);

    assert!((mean - 3.0).abs() < 1e-10);
    assert!((standardized.iter().sum::<f64>()).abs() < 1e-10);

    let recovered = destandardize(&standardized, mean, std_dev);
    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig - rec).abs() < 1e-10);
    }
}

#[test]
fn test_difference() {
    let data = vec![1.0, 3.0, 6.0, 10.0];
    let diff = difference(&data, 1);
    assert_eq!(diff, vec![2.0, 3.0, 4.0]);

    let diff2 = difference(&data, 2);
    assert_eq!(diff2, vec![1.0, 1.0]);
}

// ============================================================================
// KNN Tests
// ============================================================================

#[test]
fn test_knn_creation() {
    let knn = TimeSeriesKNN::new(3, 5, DistanceMetric::Euclidean);
    assert!(knn.is_ok());

    let knn = TimeSeriesKNN::new(0, 5, DistanceMetric::Euclidean);
    assert!(knn.is_err());
}

#[test]
fn test_knn_fit_predict() {
    // Create periodic data
    let data: Vec<f64> = (0..100)
        .map(|i| (i as f64 * 0.2).sin() * 10.0 + 50.0)
        .collect();

    let mut knn = TimeSeriesKNN::new(3, 10, DistanceMetric::Euclidean).unwrap();
    knn.fit(&data).unwrap();

    let forecast = knn.predict(5);
    assert!(forecast.is_ok());
    assert_eq!(forecast.unwrap().len(), 5);
}

// ============================================================================
// ARIMA Tests
// ============================================================================

#[test]
fn test_arima_creation() {
    let model = Arima::new(1, 1, 1);
    assert!(model.is_ok());

    let model = Arima::new(11, 0, 0);
    assert!(model.is_err());
}

#[test]
fn test_arima_fit_predict() {
    let data: Vec<f64> = (1..=50).map(|x| x as f64 + (x as f64 * 0.1).sin()).collect();
    let mut model = Arima::new(1, 1, 0).unwrap();

    assert!(model.fit(&data).is_ok());
    assert!(model.is_fitted());

    let forecast = model.predict(5);
    assert!(forecast.is_ok());
    assert_eq!(forecast.unwrap().len(), 5);
}

// ============================================================================
// Linear Regression Tests
// ============================================================================

#[test]
fn test_linear_regression() {
    let data: Vec<f64> = (0..10).map(|i| 10.0 + 2.0 * i as f64).collect();
    let mut model = LinearRegression::new();
    model.fit(&data).unwrap();

    assert!((model.slope() - 2.0).abs() < 1e-10);
    assert!((model.intercept() - 10.0).abs() < 1e-10);
    assert!(model.r_squared() > 0.99);

    let forecast = model.predict(3).unwrap();
    assert!((forecast[0] - 30.0).abs() < 1e-10);
    assert!((forecast[1] - 32.0).abs() < 1e-10);
    assert!((forecast[2] - 34.0).abs() < 1e-10);
}

#[test]
fn test_seasonal_linear_regression() {
    let data: Vec<f64> = (0..24)
        .map(|i| 10.0 + 0.5 * i as f64 + [2.0, -1.0, 0.0, -1.0][i % 4])
        .collect();

    let mut model = SeasonalLinearRegression::new(4).unwrap();
    model.fit(&data).unwrap();

    let forecast = model.predict(4).unwrap();
    assert_eq!(forecast.len(), 4);
}

// ============================================================================
// Moving Average Tests
// ============================================================================

#[test]
fn test_sma() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut sma = SimpleMovingAverage::new(3).unwrap();
    sma.fit(&data).unwrap();

    let smoothed = sma.smoothed_values();
    assert_eq!(smoothed.len(), 3);
    assert!((smoothed[0] - 2.0).abs() < 1e-10); // avg(1,2,3)
    assert!((smoothed[1] - 3.0).abs() < 1e-10); // avg(2,3,4)
    assert!((smoothed[2] - 4.0).abs() < 1e-10); // avg(3,4,5)
}

#[test]
fn test_wma_linear() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut wma = WeightedMovingAverage::linear(3).unwrap();
    wma.fit(&data).unwrap();

    let smoothed = wma.smoothed_values();
    assert_eq!(smoothed.len(), 3);
    // weights: [1, 2, 3], sum = 6
    // first: (1*1 + 2*2 + 3*3) / 6 = 14/6 ≈ 2.333
    assert!((smoothed[0] - 14.0 / 6.0).abs() < 1e-10);
}

// ============================================================================
// Exponential Smoothing Tests
// ============================================================================

#[test]
fn test_ses() {
    let data = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0];
    let mut model = SimpleExponentialSmoothing::new(0.3).unwrap();
    model.fit(&data).unwrap();
    let forecast = model.predict(3).unwrap();
    assert_eq!(forecast.len(), 3);
    // All forecasts should be the same (flat)
    assert!((forecast[0] - forecast[1]).abs() < 1e-10);
}

#[test]
fn test_holt() {
    let data: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 2.0).collect();
    let mut model = DoubleExponentialSmoothing::new(0.3, 0.1).unwrap();
    model.fit(&data).unwrap();
    let forecast = model.predict(3).unwrap();
    assert_eq!(forecast.len(), 3);
    // Forecasts should increase (positive trend)
    assert!(forecast[1] > forecast[0]);
}

#[test]
fn test_holt_winters() {
    let data: Vec<f64> = (0..48)
        .map(|i| 100.0 + (i as f64 * 2.0) + 20.0 * ((i as f64 * std::f64::consts::PI / 6.0).sin()))
        .collect();

    let mut model = HoltWinters::new(0.3, 0.1, 0.2, 12, SeasonalType::Additive).unwrap();
    model.fit(&data).unwrap();
    let forecast = model.predict(12).unwrap();
    assert_eq!(forecast.len(), 12);
}
