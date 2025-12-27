//! Automatic model selection

use algorithm::prelude::*;
use algorithm::utils::metrics::{mae, mape, rmse};
use algorithm::utils::validation::train_test_split;
use serde::{Deserialize, Serialize};

use crate::hyperopt::GridSearch;

/// Model type for AutoML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Arima { p: usize, d: usize, q: usize },
    SES { alpha: f64 },
    Holt { alpha: f64, beta: f64 },
    HoltWinters { alpha: f64, beta: f64, gamma: f64, period: usize },
    LinearRegression,
    KNN { k: usize, window: usize },
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Arima { p, d, q } => write!(f, "ARIMA({},{},{})", p, d, q),
            ModelType::SES { alpha } => write!(f, "SES(alpha={:.2})", alpha),
            ModelType::Holt { alpha, beta } => write!(f, "Holt(alpha={:.2}, beta={:.2})", alpha, beta),
            ModelType::HoltWinters { alpha, beta, gamma, period } => {
                write!(f, "HoltWinters(alpha={:.2}, beta={:.2}, gamma={:.2}, period={})", alpha, beta, gamma, period)
            }
            ModelType::LinearRegression => write!(f, "LinearRegression"),
            ModelType::KNN { k, window } => write!(f, "KNN(k={}, window={})", k, window),
        }
    }
}

/// Optimization metric
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OptimizationMetric {
    MAE,
    RMSE,
    MAPE,
}

/// AutoML configuration
#[derive(Debug, Clone)]
pub struct AutoMLConfig {
    pub metric: OptimizationMetric,
    pub cv_folds: usize,
    pub max_iterations: usize,
    pub test_ratio: f64,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            metric: OptimizationMetric::MAE,
            cv_folds: 5,
            max_iterations: 100,
            test_ratio: 0.2,
        }
    }
}

/// Result of model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionResult {
    pub best_model: ModelType,
    pub score: f64,
    pub all_scores: Vec<(ModelType, f64)>,
}

/// Automatic model selection for time series
#[derive(Debug, Clone)]
pub struct AutoML {
    config: AutoMLConfig,
}

impl AutoML {
    /// Create AutoML with custom configuration
    pub fn new(config: AutoMLConfig) -> Self {
        Self { config }
    }

    /// Create AutoML with default configuration
    pub fn with_defaults() -> Self {
        Self {
            config: AutoMLConfig::default(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &AutoMLConfig {
        &self.config
    }

    /// Compute score for predictions (lower is better)
    fn compute_score(&self, actual: &[f64], predicted: &[f64]) -> f64 {
        match self.config.metric {
            OptimizationMetric::MAE => mae(actual, predicted),
            OptimizationMetric::RMSE => rmse(actual, predicted),
            OptimizationMetric::MAPE => mape(actual, predicted),
        }
    }

    /// Evaluate a specific model type on the data
    fn evaluate_model(
        &self,
        model_type: &ModelType,
        train: &[f64],
        test: &[f64],
        horizon: usize,
    ) -> Option<f64> {
        let test_len = test.len().min(horizon);

        match model_type {
            ModelType::Arima { p, d, q } => {
                let mut model = Arima::new(*p, *d, *q).ok()?;
                model.fit(train).ok()?;
                let predictions = model.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
            ModelType::SES { alpha } => {
                let mut model = SimpleExponentialSmoothing::new(*alpha).ok()?;
                model.fit(train).ok()?;
                let predictions = model.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
            ModelType::Holt { alpha, beta } => {
                let mut model = DoubleExponentialSmoothing::new(*alpha, *beta).ok()?;
                model.fit(train).ok()?;
                let predictions = model.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
            ModelType::HoltWinters { alpha, beta, gamma, period } => {
                let mut model = HoltWinters::new(
                    *alpha,
                    *beta,
                    *gamma,
                    *period,
                    SeasonalType::Additive,
                ).ok()?;
                model.fit(train).ok()?;
                let predictions = model.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
            ModelType::LinearRegression => {
                let mut model = LinearRegression::new();
                model.fit(train).ok()?;
                let predictions = model.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
            ModelType::KNN { k, window } => {
                let mut model = TimeSeriesKNN::new(*k, *window, DistanceMetric::Euclidean)
                    .ok()?
                    .with_horizon(horizon);
                model.fit(train).ok()?;
                let predictions = model.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
        }
    }

    /// Select the best model for the given data
    pub fn select_best_model(
        &self,
        data: &[f64],
        horizon: usize,
    ) -> algorithm::Result<ModelSelectionResult> {
        if data.len() < 20 {
            return Err(algorithm::TsError::InsufficientData {
                required: 20,
                actual: data.len(),
            });
        }

        let (train, test) = train_test_split(data, self.config.test_ratio);
        let grid = GridSearch::new(self.config.metric).with_test_ratio(self.config.test_ratio);

        let mut all_scores: Vec<(ModelType, f64)> = Vec::new();

        // Optimize and evaluate ARIMA
        if let Ok((p, d, q, _)) = grid.optimize_arima(data, horizon) {
            let model_type = ModelType::Arima { p, d, q };
            if let Some(score) = self.evaluate_model(&model_type, train, test, horizon) {
                if score.is_finite() {
                    all_scores.push((model_type, score));
                }
            }
        }

        // Optimize and evaluate SES
        if let Ok((alpha, _)) = grid.optimize_ses(data, horizon) {
            let model_type = ModelType::SES { alpha };
            if let Some(score) = self.evaluate_model(&model_type, train, test, horizon) {
                if score.is_finite() {
                    all_scores.push((model_type, score));
                }
            }
        }

        // Optimize and evaluate Holt
        if let Ok((alpha, beta, _)) = grid.optimize_holt(data, horizon) {
            let model_type = ModelType::Holt { alpha, beta };
            if let Some(score) = self.evaluate_model(&model_type, train, test, horizon) {
                if score.is_finite() {
                    all_scores.push((model_type, score));
                }
            }
        }

        // Evaluate LinearRegression (no hyperparameters)
        let lr_model = ModelType::LinearRegression;
        if let Some(score) = self.evaluate_model(&lr_model, train, test, horizon) {
            if score.is_finite() {
                all_scores.push((lr_model, score));
            }
        }

        // Optimize and evaluate KNN
        if let Ok((k, window, _)) = grid.optimize_knn(data, horizon) {
            let model_type = ModelType::KNN { k, window };
            if let Some(score) = self.evaluate_model(&model_type, train, test, horizon) {
                if score.is_finite() {
                    all_scores.push((model_type, score));
                }
            }
        }

        // Sort by score (lower is better)
        all_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if all_scores.is_empty() {
            return Err(algorithm::TsError::NumericalError(
                "No models could be fitted to the data".to_string(),
            ));
        }

        let (best_model, score) = all_scores[0].clone();

        Ok(ModelSelectionResult {
            best_model,
            score,
            all_scores,
        })
    }

    /// Quick selection with fewer model candidates (faster but less thorough)
    pub fn quick_select(
        &self,
        data: &[f64],
        horizon: usize,
    ) -> algorithm::Result<ModelSelectionResult> {
        if data.len() < 15 {
            return Err(algorithm::TsError::InsufficientData {
                required: 15,
                actual: data.len(),
            });
        }

        let (train, test) = train_test_split(data, self.config.test_ratio);
        let mut all_scores: Vec<(ModelType, f64)> = Vec::new();

        // Quick candidates with fixed hyperparameters
        let candidates = vec![
            ModelType::Arima { p: 1, d: 1, q: 0 },
            ModelType::SES { alpha: 0.3 },
            ModelType::Holt { alpha: 0.3, beta: 0.1 },
            ModelType::LinearRegression,
        ];

        for model_type in candidates {
            if let Some(score) = self.evaluate_model(&model_type, train, test, horizon) {
                if score.is_finite() {
                    all_scores.push((model_type, score));
                }
            }
        }

        all_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if all_scores.is_empty() {
            return Err(algorithm::TsError::NumericalError(
                "No models could be fitted to the data".to_string(),
            ));
        }

        let (best_model, score) = all_scores[0].clone();

        Ok(ModelSelectionResult {
            best_model,
            score,
            all_scores,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trending_data() -> Vec<f64> {
        (0..60).map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.2).sin()).collect()
    }

    #[test]
    fn test_automl_select_best_model() {
        let data = trending_data();
        let automl = AutoML::with_defaults();
        let result = automl.select_best_model(&data, 5).unwrap();

        assert!(result.score.is_finite());
        assert!(!result.all_scores.is_empty());
        println!("Best model: {} (score: {:.4})", result.best_model, result.score);
    }

    #[test]
    fn test_automl_quick_select() {
        let data = trending_data();
        let automl = AutoML::with_defaults();
        let result = automl.quick_select(&data, 5).unwrap();

        assert!(result.score.is_finite());
        println!("Quick best: {} (score: {:.4})", result.best_model, result.score);
    }

    #[test]
    fn test_automl_with_custom_config() {
        let data = trending_data();
        let config = AutoMLConfig {
            metric: OptimizationMetric::RMSE,
            cv_folds: 3,
            max_iterations: 50,
            test_ratio: 0.25,
        };
        let automl = AutoML::new(config);
        let result = automl.select_best_model(&data, 5).unwrap();

        assert!(result.score.is_finite());
    }

    #[test]
    fn test_insufficient_data() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let automl = AutoML::with_defaults();
        let result = automl.select_best_model(&data, 5);

        assert!(result.is_err());
    }

    #[test]
    fn test_model_type_display() {
        let arima = ModelType::Arima { p: 1, d: 1, q: 0 };
        assert_eq!(format!("{}", arima), "ARIMA(1,1,0)");

        let ses = ModelType::SES { alpha: 0.3 };
        assert_eq!(format!("{}", ses), "SES(alpha=0.30)");
    }
}
