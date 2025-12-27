//! End-to-end tests for automl crate
//!
//! Tests complete AutoML workflows using only this crate's API.

use automl::{
    AutoML, AutoMLConfig, EnsembleMethod, GridSearch, ModelSelectionResult, ModelType,
    OptimizationMetric, combine_predictions,
};

fn sample_predictions() -> Vec<Vec<f64>> {
    vec![
        vec![10.0, 11.0, 12.0, 13.0, 14.0],
        vec![11.0, 12.0, 13.0, 14.0, 15.0],
        vec![9.0, 10.0, 11.0, 12.0, 13.0],
    ]
}

fn trending_data() -> Vec<f64> {
    (0..60)
        .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.2).sin())
        .collect()
}

// ============== AutoML Model Selection Tests ==============

#[test]
fn e2e_automl_select_best_model() {
    let data = trending_data();
    let automl = AutoML::with_defaults();

    let result = automl.select_best_model(&data, 5).unwrap();

    assert!(result.score.is_finite());
    assert!(result.score >= 0.0);
    assert!(!result.all_scores.is_empty());

    // Verify all_scores is sorted (best first)
    for window in result.all_scores.windows(2) {
        assert!(window[0].1 <= window[1].1);
    }
}

#[test]
fn e2e_automl_quick_select() {
    let data = trending_data();
    let automl = AutoML::with_defaults();

    let result = automl.quick_select(&data, 5).unwrap();

    assert!(result.score.is_finite());
    assert!(!result.all_scores.is_empty());
}

#[test]
fn e2e_automl_with_custom_metric() {
    let data = trending_data();

    // Test with RMSE metric
    let config = AutoMLConfig {
        metric: OptimizationMetric::RMSE,
        cv_folds: 3,
        max_iterations: 50,
        test_ratio: 0.2,
    };
    let automl = AutoML::new(config);
    let result = automl.select_best_model(&data, 5).unwrap();

    assert!(result.score.is_finite());
}

#[test]
fn e2e_automl_insufficient_data() {
    let short_data: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let automl = AutoML::with_defaults();

    let result = automl.select_best_model(&short_data, 5);
    assert!(result.is_err());
}

#[test]
fn e2e_automl_model_selection_result_structure() {
    let data = trending_data();
    let automl = AutoML::with_defaults();

    let result: ModelSelectionResult = automl.select_best_model(&data, 5).unwrap();

    // Check that best_model is one of the evaluated models
    let best_in_all = result
        .all_scores
        .iter()
        .any(|(m, s)| format!("{:?}", m) == format!("{:?}", result.best_model) && *s == result.score);
    assert!(best_in_all);
}

// ============== Grid Search Tests ==============

#[test]
fn e2e_grid_search_arima() {
    let data = trending_data();
    let grid = GridSearch::new(OptimizationMetric::MAE);

    let (p, d, q, score) = grid.optimize_arima(&data, 5).unwrap();

    assert!(p <= 3);
    assert!(d <= 2);
    assert!(q <= 3);
    assert!(score.is_finite());
}

#[test]
fn e2e_grid_search_ses() {
    let data = trending_data();
    let grid = GridSearch::new(OptimizationMetric::RMSE);

    let (alpha, score) = grid.optimize_ses(&data, 5).unwrap();

    assert!(alpha > 0.0 && alpha < 1.0);
    assert!(score.is_finite());
}

#[test]
fn e2e_grid_search_holt() {
    let data = trending_data();
    let grid = GridSearch::new(OptimizationMetric::MAE);

    let (alpha, beta, score) = grid.optimize_holt(&data, 5).unwrap();

    assert!(alpha > 0.0 && alpha < 1.0);
    assert!(beta > 0.0 && beta < 1.0);
    assert!(score.is_finite());
}

#[test]
fn e2e_grid_search_knn() {
    let data = trending_data();
    let grid = GridSearch::new(OptimizationMetric::MAE);

    let (k, window, score) = grid.optimize_knn(&data, 3).unwrap();

    assert!(k >= 1);
    assert!(window >= 3);
    assert!(score.is_finite());
}

#[test]
fn e2e_grid_search_custom_test_ratio() {
    let data = trending_data();
    let grid = GridSearch::new(OptimizationMetric::MAE).with_test_ratio(0.3);

    let (alpha, score) = grid.optimize_ses(&data, 5).unwrap();

    assert!(alpha > 0.0 && alpha < 1.0);
    assert!(score.is_finite());
}

// ============== Ensemble Tests ==============

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
    let predictions = vec![vec![10.0, 10.0, 10.0], vec![20.0, 20.0, 20.0]];
    let weights = vec![0.75, 0.25];

    let combined = combine_predictions(&predictions, EnsembleMethod::WeightedAverage, Some(&weights));

    assert_eq!(combined.len(), 3);

    // 10*0.75 + 20*0.25 = 12.5
    for &val in &combined {
        assert!((val - 12.5).abs() < 0.001);
    }
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
    let predictions: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64; 5]).collect();

    let avg = combine_predictions(&predictions, EnsembleMethod::Average, None);
    let med = combine_predictions(&predictions, EnsembleMethod::Median, None);

    assert_eq!(avg.len(), 5);
    assert_eq!(med.len(), 5);

    // Average of 0..9 = 4.5
    assert!((avg[0] - 4.5).abs() < 0.001);

    // Median of 0..9 = 4.5 (even count)
    assert!((med[0] - 4.5).abs() < 0.001);
}

// ============== Model Type & Config Tests ==============

#[test]
fn e2e_model_type_configuration() {
    // Test all model type configurations
    let arima = ModelType::Arima { p: 2, d: 1, q: 1 };
    let ses = ModelType::SES { alpha: 0.3 };
    let holt = ModelType::Holt { alpha: 0.3, beta: 0.1 };
    let hw = ModelType::HoltWinters {
        alpha: 0.3,
        beta: 0.1,
        gamma: 0.2,
        period: 12,
    };
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
fn e2e_model_type_display() {
    let arima = ModelType::Arima { p: 1, d: 1, q: 0 };
    assert_eq!(format!("{}", arima), "ARIMA(1,1,0)");

    let ses = ModelType::SES { alpha: 0.3 };
    assert_eq!(format!("{}", ses), "SES(alpha=0.30)");

    let holt = ModelType::Holt { alpha: 0.5, beta: 0.2 };
    assert_eq!(format!("{}", holt), "Holt(alpha=0.50, beta=0.20)");

    let lr = ModelType::LinearRegression;
    assert_eq!(format!("{}", lr), "LinearRegression");

    let knn = ModelType::KNN { k: 3, window: 5 };
    assert_eq!(format!("{}", knn), "KNN(k=3, window=5)");
}

#[test]
fn e2e_automl_config_workflow() {
    // Default config
    let default_config = AutoMLConfig::default();
    assert!(matches!(default_config.metric, OptimizationMetric::MAE));
    assert_eq!(default_config.cv_folds, 5);
    assert_eq!(default_config.max_iterations, 100);
    assert!((default_config.test_ratio - 0.2).abs() < 0.001);

    // Custom config
    let custom_config = AutoMLConfig {
        metric: OptimizationMetric::RMSE,
        cv_folds: 10,
        max_iterations: 50,
        test_ratio: 0.25,
    };
    assert!(matches!(custom_config.metric, OptimizationMetric::RMSE));
    assert_eq!(custom_config.cv_folds, 10);
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

    // Metrics should be comparable
    assert_eq!(mae, OptimizationMetric::MAE);
    assert_ne!(mae, OptimizationMetric::RMSE);
}

#[test]
fn e2e_weighted_ensemble_normalization() {
    let predictions = vec![vec![100.0], vec![200.0], vec![300.0]];

    // Weights that sum to 1
    let weights1 = vec![0.5, 0.3, 0.2];
    let result1 = combine_predictions(&predictions, EnsembleMethod::WeightedAverage, Some(&weights1));

    // 100*0.5 + 200*0.3 + 300*0.2 = 50 + 60 + 60 = 170
    assert!((result1[0] - 170.0).abs() < 0.001);
}
