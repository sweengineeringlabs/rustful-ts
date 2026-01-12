//! Hyperparameter optimization

use automl_api::{GridSearchConfig, OptimizationMetric};
use automl_spi::Result;
use predictor_facade::prelude::*;
use predictor_facade::utils::metrics::{mae, mape, rmse};
use predictor_facade::utils::validation::train_test_split;

/// Grid search hyperparameter optimizer
#[derive(Debug, Clone)]
pub struct GridSearch {
    metric: OptimizationMetric,
    test_ratio: f64,
}

impl GridSearch {
    /// Create a new grid search optimizer with default settings
    pub fn new(metric: OptimizationMetric) -> Self {
        Self {
            metric,
            test_ratio: 0.2,
        }
    }

    /// Create from a configuration
    pub fn from_config(config: GridSearchConfig) -> Self {
        Self {
            metric: config.metric,
            test_ratio: config.test_ratio,
        }
    }

    /// Set the test ratio for train/test split
    pub fn with_test_ratio(mut self, ratio: f64) -> Self {
        self.test_ratio = ratio.clamp(0.1, 0.5);
        self
    }

    /// Compute score based on configured metric (lower is better)
    fn compute_score(&self, actual: &[f64], predicted: &[f64]) -> f64 {
        match self.metric {
            OptimizationMetric::MAE => mae(actual, predicted),
            OptimizationMetric::RMSE => rmse(actual, predicted),
            OptimizationMetric::MAPE => mape(actual, predicted),
        }
    }

    /// Optimize ARIMA parameters (p, d, q)
    /// Returns (p, d, q, score)
    pub fn optimize_arima(&self, data: &[f64], horizon: usize) -> Result<(usize, usize, usize, f64)> {
        let (train, test) = train_test_split(data, self.test_ratio);
        let test_len = test.len().min(horizon);

        let mut best_params = (1, 1, 0);
        let mut best_score = f64::MAX;

        // Grid: p in [0,3], d in [0,2], q in [0,3]
        for p in 0..=3 {
            for d in 0..=2 {
                for q in 0..=3 {
                    // Skip invalid combinations
                    if p == 0 && q == 0 {
                        continue;
                    }

                    if let Ok(mut model) = Arima::new(p, d, q) {
                        if model.fit(train).is_ok() {
                            if let Ok(predictions) = model.predict(test_len) {
                                let score = self.compute_score(&test[..test_len], &predictions);
                                if score.is_finite() && score < best_score {
                                    best_score = score;
                                    best_params = (p, d, q);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok((best_params.0, best_params.1, best_params.2, best_score))
    }

    /// Optimize SES alpha parameter
    /// Returns (alpha, score)
    pub fn optimize_ses(&self, data: &[f64], horizon: usize) -> Result<(f64, f64)> {
        let (train, test) = train_test_split(data, self.test_ratio);
        let test_len = test.len().min(horizon);

        let mut best_alpha = 0.3;
        let mut best_score = f64::MAX;

        let alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        for &alpha in &alphas {
            if let Ok(mut model) = SimpleExponentialSmoothing::new(alpha) {
                if model.fit(train).is_ok() {
                    if let Ok(predictions) = model.predict(test_len) {
                        let score = self.compute_score(&test[..test_len], &predictions);
                        if score.is_finite() && score < best_score {
                            best_score = score;
                            best_alpha = alpha;
                        }
                    }
                }
            }
        }

        Ok((best_alpha, best_score))
    }

    /// Optimize Holt (Double Exponential Smoothing) parameters
    /// Returns (alpha, beta, score)
    pub fn optimize_holt(&self, data: &[f64], horizon: usize) -> Result<(f64, f64, f64)> {
        let (train, test) = train_test_split(data, self.test_ratio);
        let test_len = test.len().min(horizon);

        let mut best_alpha = 0.3;
        let mut best_beta = 0.1;
        let mut best_score = f64::MAX;

        let params = [0.1, 0.3, 0.5, 0.7, 0.9];

        for &alpha in &params {
            for &beta in &params {
                if let Ok(mut model) = DoubleExponentialSmoothing::new(alpha, beta) {
                    if model.fit(train).is_ok() {
                        if let Ok(predictions) = model.predict(test_len) {
                            let score = self.compute_score(&test[..test_len], &predictions);
                            if score.is_finite() && score < best_score {
                                best_score = score;
                                best_alpha = alpha;
                                best_beta = beta;
                            }
                        }
                    }
                }
            }
        }

        Ok((best_alpha, best_beta, best_score))
    }

    /// Optimize KNN parameters
    /// Returns (k, window_size, score)
    pub fn optimize_knn(&self, data: &[f64], horizon: usize) -> Result<(usize, usize, f64)> {
        let (train, test) = train_test_split(data, self.test_ratio);
        let test_len = test.len().min(horizon);

        let mut best_k = 3;
        let mut best_window = 5;
        let mut best_score = f64::MAX;

        let k_values = [1, 3, 5, 7];
        let window_values = [3, 5, 7, 10];

        for &k in &k_values {
            for &window in &window_values {
                if let Ok(model) = TimeSeriesKNN::new(k, window, DistanceMetric::Euclidean) {
                    let mut model = model.with_horizon(horizon);
                    if model.fit(train).is_ok() {
                        if let Ok(predictions) = model.predict(test_len) {
                            let score = self.compute_score(&test[..test_len], &predictions);
                            if score.is_finite() && score < best_score {
                                best_score = score;
                                best_k = k;
                                best_window = window;
                            }
                        }
                    }
                }
            }
        }

        Ok((best_k, best_window, best_score))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<f64> {
        // Generate trending data with some noise
        (0..50)
            .map(|i| 10.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin())
            .collect()
    }

    #[test]
    fn test_optimize_arima() {
        let data = sample_data();
        let grid = GridSearch::new(OptimizationMetric::MAE);
        let (p, d, q, score) = grid.optimize_arima(&data, 5).unwrap();

        assert!(p <= 3);
        assert!(d <= 2);
        assert!(q <= 3);
        assert!(score.is_finite());
    }

    #[test]
    fn test_optimize_ses() {
        let data = sample_data();
        let grid = GridSearch::new(OptimizationMetric::RMSE);
        let (alpha, score) = grid.optimize_ses(&data, 5).unwrap();

        assert!(alpha > 0.0 && alpha < 1.0);
        assert!(score.is_finite());
    }

    #[test]
    fn test_optimize_holt() {
        let data = sample_data();
        let grid = GridSearch::new(OptimizationMetric::MAE);
        let (alpha, beta, score) = grid.optimize_holt(&data, 5).unwrap();

        assert!(alpha > 0.0 && alpha < 1.0);
        assert!(beta > 0.0 && beta < 1.0);
        assert!(score.is_finite());
    }

    #[test]
    fn test_optimize_knn() {
        let data = sample_data();
        let grid = GridSearch::new(OptimizationMetric::MAE);
        let (k, window, score) = grid.optimize_knn(&data, 3).unwrap();

        assert!(k >= 1);
        assert!(window >= 3);
        assert!(score.is_finite());
    }
}
