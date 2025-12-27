//! Integration tests for rustful-automl

use automl::{
    ModelType, OptimizationMetric, AutoMLConfig, EnsembleMethod, combine_predictions,
};

#[test]
fn test_model_types() {
    let arima = ModelType::Arima { p: 1, d: 1, q: 1 };
    let ses = ModelType::SES { alpha: 0.3 };
    let holt = ModelType::Holt { alpha: 0.3, beta: 0.1 };
    let hw = ModelType::HoltWinters { alpha: 0.3, beta: 0.1, gamma: 0.1, period: 12 };
    let lr = ModelType::LinearRegression;
    let knn = ModelType::KNN { k: 3, window: 5 };

    // Test debug formatting works
    assert!(format!("{:?}", arima).contains("Arima"));
    assert!(format!("{:?}", ses).contains("SES"));
    assert!(format!("{:?}", holt).contains("Holt"));
    assert!(format!("{:?}", hw).contains("HoltWinters"));
    assert!(format!("{:?}", lr).contains("LinearRegression"));
    assert!(format!("{:?}", knn).contains("KNN"));
}

#[test]
fn test_automl_config_default() {
    let config = AutoMLConfig::default();

    assert!(matches!(config.metric, OptimizationMetric::MAE));
    assert_eq!(config.cv_folds, 5);
    assert_eq!(config.max_iterations, 100);
}

#[test]
fn test_automl_config_custom() {
    let config = AutoMLConfig {
        metric: OptimizationMetric::RMSE,
        cv_folds: 10,
        max_iterations: 200,
        test_ratio: 0.2,
    };

    assert!(matches!(config.metric, OptimizationMetric::RMSE));
    assert_eq!(config.cv_folds, 10);
    assert_eq!(config.max_iterations, 200);
}

#[test]
fn test_ensemble_average() {
    let predictions = vec![
        vec![10.0, 11.0, 12.0],
        vec![12.0, 13.0, 14.0],
        vec![11.0, 12.0, 13.0],
    ];

    let combined = combine_predictions(&predictions, EnsembleMethod::Average, None);

    assert_eq!(combined.len(), 3);
    assert!((combined[0] - 11.0).abs() < 0.001);
    assert!((combined[1] - 12.0).abs() < 0.001);
    assert!((combined[2] - 13.0).abs() < 0.001);
}

#[test]
fn test_ensemble_weighted_average() {
    let predictions = vec![
        vec![10.0, 10.0, 10.0],
        vec![20.0, 20.0, 20.0],
    ];
    let weights = vec![0.75, 0.25];

    let combined = combine_predictions(&predictions, EnsembleMethod::WeightedAverage, Some(&weights));

    assert_eq!(combined.len(), 3);
    // 10*0.75 + 20*0.25 = 7.5 + 5 = 12.5
    assert!((combined[0] - 12.5).abs() < 0.001);
}

#[test]
fn test_ensemble_median_odd() {
    let predictions = vec![
        vec![10.0],
        vec![20.0],
        vec![15.0],
    ];

    let combined = combine_predictions(&predictions, EnsembleMethod::Median, None);

    assert_eq!(combined.len(), 1);
    assert!((combined[0] - 15.0).abs() < 0.001); // Median of [10, 15, 20]
}

#[test]
fn test_ensemble_median_even() {
    let predictions = vec![
        vec![10.0],
        vec![20.0],
        vec![15.0],
        vec![25.0],
    ];

    let combined = combine_predictions(&predictions, EnsembleMethod::Median, None);

    assert_eq!(combined.len(), 1);
    // Median of [10, 15, 20, 25] = (15 + 20) / 2 = 17.5
    assert!((combined[0] - 17.5).abs() < 0.001);
}

#[test]
fn test_ensemble_empty() {
    let predictions: Vec<Vec<f64>> = vec![];
    let combined = combine_predictions(&predictions, EnsembleMethod::Average, None);

    assert!(combined.is_empty());
}

#[test]
fn test_model_type_clone() {
    let model = ModelType::Arima { p: 2, d: 1, q: 2 };
    let cloned = model.clone();

    assert!(format!("{:?}", cloned).contains("p: 2"));
}

#[test]
fn test_optimization_metrics() {
    let mae = OptimizationMetric::MAE;
    let rmse = OptimizationMetric::RMSE;
    let mape = OptimizationMetric::MAPE;

    assert!(format!("{:?}", mae).contains("MAE"));
    assert!(format!("{:?}", rmse).contains("RMSE"));
    assert!(format!("{:?}", mape).contains("MAPE"));
}
