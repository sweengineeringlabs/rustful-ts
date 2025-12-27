//! End-to-end tests for rustful-automl crate
//!
//! Tests complete AutoML workflows using only this crate's API.

use automl::{
    ModelType, OptimizationMetric, AutoMLConfig,
    EnsembleMethod, combine_predictions,
};

fn sample_predictions() -> Vec<Vec<f64>> {
    vec![
        vec![10.0, 11.0, 12.0, 13.0, 14.0],
        vec![11.0, 12.0, 13.0, 14.0, 15.0],
        vec![9.0, 10.0, 11.0, 12.0, 13.0],
    ]
}

#[test]
fn e2e_ensemble_average_workflow() {
    let predictions = sample_predictions();

    let combined = combine_predictions(&predictions, EnsembleMethod::Average, None);

    assert_eq!(combined.len(), 5);

    // Average of [10,11,9], [11,12,10], etc.
    assert!((combined[0] - 10.0).abs() < 0.001);
    assert!((combined[1] - 11.0).abs() < 0.001);
    assert!((combined[2] - 12.0).abs() < 0.001);
}

#[test]
fn e2e_ensemble_median_workflow() {
    let predictions = sample_predictions();

    let combined = combine_predictions(&predictions, EnsembleMethod::Median, None);

    assert_eq!(combined.len(), 5);

    // Median of [9,10,11] = 10, [10,11,12] = 11, etc.
    assert!((combined[0] - 10.0).abs() < 0.001);
    assert!((combined[1] - 11.0).abs() < 0.001);
}

#[test]
fn e2e_ensemble_weighted_workflow() {
    let predictions = vec![
        vec![10.0, 10.0, 10.0],
        vec![20.0, 20.0, 20.0],
    ];
    let weights = vec![0.75, 0.25];

    let combined = combine_predictions(
        &predictions,
        EnsembleMethod::WeightedAverage,
        Some(&weights),
    );

    assert_eq!(combined.len(), 3);

    // 10*0.75 + 20*0.25 = 12.5
    for &val in &combined {
        assert!((val - 12.5).abs() < 0.001);
    }
}

#[test]
fn e2e_model_type_configuration() {
    // Test all model type configurations
    let arima = ModelType::Arima { p: 2, d: 1, q: 1 };
    let ses = ModelType::SES { alpha: 0.3 };
    let holt = ModelType::Holt { alpha: 0.3, beta: 0.1 };
    let hw = ModelType::HoltWinters { alpha: 0.3, beta: 0.1, gamma: 0.2, period: 12 };
    let lr = ModelType::LinearRegression;
    let knn = ModelType::KNN { k: 5, window: 10 };

    // All should be clonable and debuggable
    let _ = arima.clone();
    let _ = format!("{:?}", ses);
    let _ = format!("{:?}", holt);
    let _ = format!("{:?}", hw);
    let _ = format!("{:?}", lr);
    let _ = format!("{:?}", knn);
}

#[test]
fn e2e_automl_config_workflow() {
    // Default config
    let default_config = AutoMLConfig::default();
    assert!(matches!(default_config.metric, OptimizationMetric::MAE));
    assert_eq!(default_config.cv_folds, 5);
    assert_eq!(default_config.max_iterations, 100);

    // Custom config
    let custom_config = AutoMLConfig {
        metric: OptimizationMetric::RMSE,
        cv_folds: 10,
        max_iterations: 50,
    };
    assert!(matches!(custom_config.metric, OptimizationMetric::RMSE));
    assert_eq!(custom_config.cv_folds, 10);
}

#[test]
fn e2e_ensemble_empty_handling() {
    let empty: Vec<Vec<f64>> = vec![];

    let avg = combine_predictions(&empty, EnsembleMethod::Average, None);
    let med = combine_predictions(&empty, EnsembleMethod::Median, None);

    assert!(avg.is_empty());
    assert!(med.is_empty());
}

#[test]
fn e2e_ensemble_single_model() {
    let single = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];

    let avg = combine_predictions(&single, EnsembleMethod::Average, None);
    let med = combine_predictions(&single, EnsembleMethod::Median, None);

    // Single model: output should match input
    assert_eq!(avg, single[0]);
    assert_eq!(med, single[0]);
}

#[test]
fn e2e_ensemble_many_models() {
    let predictions: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![i as f64; 5])
        .collect();

    let avg = combine_predictions(&predictions, EnsembleMethod::Average, None);
    let med = combine_predictions(&predictions, EnsembleMethod::Median, None);

    assert_eq!(avg.len(), 5);
    assert_eq!(med.len(), 5);

    // Average of 0..9 = 4.5
    assert!((avg[0] - 4.5).abs() < 0.001);

    // Median of 0..9 = 4.5 (even count)
    assert!((med[0] - 4.5).abs() < 0.001);
}

#[test]
fn e2e_optimization_metrics() {
    let mae = OptimizationMetric::MAE;
    let rmse = OptimizationMetric::RMSE;
    let mape = OptimizationMetric::MAPE;

    // All should be debuggable
    assert!(format!("{:?}", mae).contains("MAE"));
    assert!(format!("{:?}", rmse).contains("RMSE"));
    assert!(format!("{:?}", mape).contains("MAPE"));
}

#[test]
fn e2e_weighted_ensemble_normalization() {
    let predictions = vec![
        vec![100.0],
        vec![200.0],
        vec![300.0],
    ];

    // Weights that sum to 1
    let weights1 = vec![0.5, 0.3, 0.2];
    let result1 = combine_predictions(
        &predictions,
        EnsembleMethod::WeightedAverage,
        Some(&weights1),
    );

    // 100*0.5 + 200*0.3 + 300*0.2 = 50 + 60 + 60 = 170
    assert!((result1[0] - 170.0).abs() < 0.001);
}
