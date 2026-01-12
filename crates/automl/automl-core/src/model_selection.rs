//! Automatic model selection

use automl_api::{AutoMLConfig, OptimizationMetric};
use automl_spi::{AutoMLError, ModelSelectionResult, ModelSelector, Result, SelectedModel};
use predictor_facade::prelude::*;
use predictor_facade::utils::metrics::{mae, mape, rmse};
use predictor_facade::utils::validation::train_test_split;

use crate::GridSearch;

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
        model: &SelectedModel,
        train: &[f64],
        test: &[f64],
        horizon: usize,
    ) -> Option<f64> {
        let test_len = test.len().min(horizon);

        match model {
            SelectedModel::Arima { p, d, q } => {
                let mut m = Arima::new(*p, *d, *q).ok()?;
                m.fit(train).ok()?;
                let predictions = m.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
            SelectedModel::SES { alpha } => {
                let mut m = SimpleExponentialSmoothing::new(*alpha).ok()?;
                m.fit(train).ok()?;
                let predictions = m.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
            SelectedModel::Holt { alpha, beta } => {
                let mut m = DoubleExponentialSmoothing::new(*alpha, *beta).ok()?;
                m.fit(train).ok()?;
                let predictions = m.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
            SelectedModel::HoltWinters {
                alpha,
                beta,
                gamma,
                period,
            } => {
                let mut m =
                    HoltWinters::new(*alpha, *beta, *gamma, *period, SeasonalType::Additive).ok()?;
                m.fit(train).ok()?;
                let predictions = m.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
            SelectedModel::LinearRegression => {
                let mut m = LinearRegression::new();
                m.fit(train).ok()?;
                let predictions = m.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
            SelectedModel::KNN { k, window } => {
                let mut m = TimeSeriesKNN::new(*k, *window, DistanceMetric::Euclidean)
                    .ok()?
                    .with_horizon(horizon);
                m.fit(train).ok()?;
                let predictions = m.predict(test_len).ok()?;
                Some(self.compute_score(&test[..test_len], &predictions))
            }
        }
    }

    /// Select the best model for the given data (alias for backward compatibility)
    pub fn select_best_model(
        &self,
        data: &[f64],
        horizon: usize,
    ) -> Result<ModelSelectionResult> {
        <Self as ModelSelector>::select(self, data, horizon)
    }

    /// Quick selection with fewer model candidates (faster but less thorough)
    pub fn quick_select(&self, data: &[f64], horizon: usize) -> Result<ModelSelectionResult> {
        if data.len() < 15 {
            return Err(AutoMLError::InsufficientData {
                required: 15,
                actual: data.len(),
            });
        }

        let (train, test) = train_test_split(data, self.config.test_ratio);
        let mut all_scores: Vec<(SelectedModel, f64)> = Vec::new();

        // Quick candidates with fixed hyperparameters
        let candidates = vec![
            SelectedModel::Arima { p: 1, d: 1, q: 0 },
            SelectedModel::SES { alpha: 0.3 },
            SelectedModel::Holt {
                alpha: 0.3,
                beta: 0.1,
            },
            SelectedModel::LinearRegression,
        ];

        for model in candidates {
            if let Some(score) = self.evaluate_model(&model, train, test, horizon) {
                if score.is_finite() {
                    all_scores.push((model, score));
                }
            }
        }

        all_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if all_scores.is_empty() {
            return Err(AutoMLError::NoValidModels(
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

impl ModelSelector for AutoML {
    /// Select the best model for the given data
    fn select(&self, data: &[f64], horizon: usize) -> Result<ModelSelectionResult> {
        if data.len() < 20 {
            return Err(AutoMLError::InsufficientData {
                required: 20,
                actual: data.len(),
            });
        }

        let (train, test) = train_test_split(data, self.config.test_ratio);
        let grid = GridSearch::new(self.config.metric).with_test_ratio(self.config.test_ratio);

        let mut all_scores: Vec<(SelectedModel, f64)> = Vec::new();

        // Optimize and evaluate ARIMA
        if let Ok((p, d, q, _)) = grid.optimize_arima(data, horizon) {
            let model = SelectedModel::Arima { p, d, q };
            if let Some(score) = self.evaluate_model(&model, train, test, horizon) {
                if score.is_finite() {
                    all_scores.push((model, score));
                }
            }
        }

        // Optimize and evaluate SES
        if let Ok((alpha, _)) = grid.optimize_ses(data, horizon) {
            let model = SelectedModel::SES { alpha };
            if let Some(score) = self.evaluate_model(&model, train, test, horizon) {
                if score.is_finite() {
                    all_scores.push((model, score));
                }
            }
        }

        // Optimize and evaluate Holt
        if let Ok((alpha, beta, _)) = grid.optimize_holt(data, horizon) {
            let model = SelectedModel::Holt { alpha, beta };
            if let Some(score) = self.evaluate_model(&model, train, test, horizon) {
                if score.is_finite() {
                    all_scores.push((model, score));
                }
            }
        }

        // Evaluate LinearRegression (no hyperparameters)
        let lr_model = SelectedModel::LinearRegression;
        if let Some(score) = self.evaluate_model(&lr_model, train, test, horizon) {
            if score.is_finite() {
                all_scores.push((lr_model, score));
            }
        }

        // Optimize and evaluate KNN
        if let Ok((k, window, _)) = grid.optimize_knn(data, horizon) {
            let model = SelectedModel::KNN { k, window };
            if let Some(score) = self.evaluate_model(&model, train, test, horizon) {
                if score.is_finite() {
                    all_scores.push((model, score));
                }
            }
        }

        // Sort by score (lower is better)
        all_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if all_scores.is_empty() {
            return Err(AutoMLError::NoValidModels(
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
        (0..60)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.2).sin())
            .collect()
    }

    #[test]
    fn test_automl_select_best_model() {
        let data = trending_data();
        let automl = AutoML::with_defaults();
        let result = automl.select(&data, 5).unwrap();

        assert!(result.score.is_finite());
        assert!(!result.all_scores.is_empty());
        println!(
            "Best model: {} (score: {:.4})",
            result.best_model, result.score
        );
    }

    #[test]
    fn test_automl_quick_select() {
        let data = trending_data();
        let automl = AutoML::with_defaults();
        let result = automl.quick_select(&data, 5).unwrap();

        assert!(result.score.is_finite());
        println!(
            "Quick best: {} (score: {:.4})",
            result.best_model, result.score
        );
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
        let result = automl.select(&data, 5).unwrap();

        assert!(result.score.is_finite());
    }

    #[test]
    fn test_insufficient_data() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let automl = AutoML::with_defaults();
        let result = automl.select(&data, 5);

        assert!(result.is_err());
    }

    #[test]
    fn test_selected_model_display() {
        let arima = SelectedModel::Arima { p: 1, d: 1, q: 0 };
        assert_eq!(format!("{}", arima), "ARIMA(1,1,0)");

        let ses = SelectedModel::SES { alpha: 0.3 };
        assert_eq!(format!("{}", ses), "SES(alpha=0.30)");
    }
}
