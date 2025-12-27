//! Integration tests for algorithm crate

use algorithm::{
    ml::{DistanceMetric, TimeSeriesKNN},
    regression::{Arima, LinearRegression},
    smoothing::{
        DoubleExponentialSmoothing, HoltWinters, SeasonalType, SimpleExponentialSmoothing,
        SimpleMovingAverage,
    },
    utils::metrics::{mae, mape, rmse},
    Predictor,
};

fn sample_data() -> Vec<f64> {
    vec![
        10.0, 12.0, 13.0, 15.0, 14.0, 16.0, 18.0, 17.0, 19.0, 21.0, 20.0, 22.0, 24.0, 23.0, 25.0,
        27.0, 26.0, 28.0, 30.0, 29.0,
    ]
}

#[test]
fn test_arima_fit_predict() {
    let data = sample_data();
    let mut model = Arima::new(1, 1, 1).unwrap();

    assert!(!model.is_fitted());
    model.fit(&data).unwrap();
    assert!(model.is_fitted());

    let forecast = model.predict(5).unwrap();
    assert_eq!(forecast.len(), 5);

    // Forecasts should be in reasonable range
    for val in &forecast {
        assert!(*val > 20.0 && *val < 50.0);
    }
}

#[test]
fn test_ses_fit_predict() {
    let data = sample_data();
    let mut model = SimpleExponentialSmoothing::new(0.3).unwrap();

    model.fit(&data).unwrap();
    let forecast = model.predict(5).unwrap();

    assert_eq!(forecast.len(), 5);
    // SES produces constant forecasts
    assert!((forecast[0] - forecast[4]).abs() < 0.001);
}

#[test]
fn test_holt_fit_predict() {
    let data = sample_data();
    let mut model = DoubleExponentialSmoothing::new(0.3, 0.1).unwrap();

    model.fit(&data).unwrap();
    let forecast = model.predict(5).unwrap();

    assert_eq!(forecast.len(), 5);
    // With positive trend, forecasts should increase
    assert!(forecast[4] > forecast[0]);
}

#[test]
fn test_holt_winters_seasonal() {
    // Generate seasonal data
    let data: Vec<f64> = (0..48)
        .map(|i| 10.0 + (i as f64) * 0.5 + 5.0 * ((i as f64) * std::f64::consts::PI / 6.0).sin())
        .collect();

    let mut model = HoltWinters::new(0.3, 0.1, 0.1, 12, SeasonalType::Additive).unwrap();
    model.fit(&data).unwrap();

    let forecast = model.predict(12).unwrap();
    assert_eq!(forecast.len(), 12);
}

#[test]
fn test_sma_fit_predict() {
    let data = sample_data();
    let mut model = SimpleMovingAverage::new(3).unwrap();

    model.fit(&data).unwrap();
    let forecast = model.predict(5).unwrap();

    assert_eq!(forecast.len(), 5);
    // SMA produces constant forecasts
    assert!((forecast[0] - forecast[4]).abs() < 0.001);
}

#[test]
fn test_linear_regression_fit_predict() {
    let data = sample_data();
    let mut model = LinearRegression::new();

    model.fit(&data).unwrap();
    let forecast = model.predict(5).unwrap();

    assert_eq!(forecast.len(), 5);
    assert!(model.slope() > 0.0); // Positive trend
    assert!(model.r_squared() > 0.8); // Good fit
}

#[test]
fn test_knn_fit_predict() {
    let data = sample_data();
    let mut model = TimeSeriesKNN::new(3, 3, DistanceMetric::Euclidean).unwrap();

    assert!(!model.is_fitted());
    model.fit(&data).unwrap();
    assert!(model.is_fitted());

    let forecast = model.predict(5).unwrap();
    assert_eq!(forecast.len(), 5);
}

#[test]
fn test_metrics() {
    let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let predicted = vec![1.1, 2.1, 3.1, 4.1, 5.1];

    let mae_val = mae(&actual, &predicted);
    assert!((mae_val - 0.1).abs() < 0.001);

    let rmse_val = rmse(&actual, &predicted);
    assert!((rmse_val - 0.1).abs() < 0.001);

    let mape_val = mape(&actual, &predicted);
    assert!(mape_val > 0.0 && mape_val < 0.1);
}

#[test]
fn test_invalid_parameters() {
    // SES alpha must be in (0, 1]
    assert!(SimpleExponentialSmoothing::new(1.5).is_err());
    assert!(SimpleExponentialSmoothing::new(-0.1).is_err());

    // SMA window must be > 0
    assert!(SimpleMovingAverage::new(0).is_err());

    // KNN k must be > 0
    assert!(TimeSeriesKNN::new(0, 3, DistanceMetric::Euclidean).is_err());
}
