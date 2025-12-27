//! Basic example demonstrating AutoML features
//!
//! Run with: cargo run --example basic -p rustful-automl

use automl::{
    ModelType, OptimizationMetric, AutoMLConfig, EnsembleMethod, combine_predictions,
};

fn main() {
    println!("=== rustful-automl Basic Examples ===\n");

    // 1. Model Types
    println!("1. Available Model Types");
    let models = vec![
        ModelType::Arima { p: 1, d: 1, q: 1 },
        ModelType::SES { alpha: 0.3 },
        ModelType::Holt { alpha: 0.3, beta: 0.1 },
        ModelType::HoltWinters { alpha: 0.3, beta: 0.1, gamma: 0.1, period: 12 },
        ModelType::LinearRegression,
        ModelType::KNN { k: 3, window: 5 },
    ];
    for model in &models {
        println!("   {:?}", model);
    }
    println!();

    // 2. AutoML Configuration
    println!("2. AutoML Configuration");
    let config = AutoMLConfig {
        metric: OptimizationMetric::RMSE,
        cv_folds: 5,
        max_iterations: 100,
        test_ratio: 0.2,
    };
    println!("   Metric: {:?}", config.metric);
    println!("   CV Folds: {}", config.cv_folds);
    println!("   Max Iterations: {}\n", config.max_iterations);

    // 3. Ensemble Methods
    println!("3. Ensemble Predictions");

    // Simulate predictions from 3 different models
    let predictions = vec![
        vec![10.0, 11.0, 12.0, 13.0, 14.0],  // Model 1
        vec![10.5, 11.5, 12.5, 13.5, 14.5],  // Model 2
        vec![9.5, 10.5, 11.5, 12.5, 13.5],   // Model 3
    ];

    println!("   Model 1 predictions: {:?}", predictions[0]);
    println!("   Model 2 predictions: {:?}", predictions[1]);
    println!("   Model 3 predictions: {:?}\n", predictions[2]);

    // Average ensemble
    let avg = combine_predictions(&predictions, EnsembleMethod::Average, None);
    println!("   Average ensemble: {:?}", avg);

    // Weighted average ensemble
    let weights = vec![0.5, 0.3, 0.2];
    let weighted = combine_predictions(&predictions, EnsembleMethod::WeightedAverage, Some(&weights));
    println!("   Weighted (0.5, 0.3, 0.2): {:?}", weighted);

    // Median ensemble
    let median = combine_predictions(&predictions, EnsembleMethod::Median, None);
    println!("   Median ensemble: {:?}", median);

    println!("\n=== Examples Complete ===");
}
