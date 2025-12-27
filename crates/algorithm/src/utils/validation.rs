//! Cross-validation utilities for time series models
//!
//! Time series require special validation strategies that respect temporal order.

use crate::Predictor;
use crate::utils::metrics;

/// Time series cross-validation using expanding window
///
/// Trains on [0, i] and tests on [i+1, i+1+horizon] for increasing i.
///
/// # Arguments
///
/// * `data` - Full time series
/// * `min_train_size` - Minimum observations for initial training
/// * `horizon` - Number of steps to forecast
/// * `step` - How many observations to add between folds
///
/// # Returns
///
/// Vector of (train_indices, test_indices) tuples
pub fn expanding_window_split(
    data_len: usize,
    min_train_size: usize,
    horizon: usize,
    step: usize,
) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
    let mut splits = Vec::new();
    let step = step.max(1);

    let mut train_end = min_train_size;
    while train_end + horizon <= data_len {
        let train_range = 0..train_end;
        let test_range = train_end..(train_end + horizon).min(data_len);

        splits.push((train_range, test_range));
        train_end += step;
    }

    splits
}

/// Time series cross-validation using sliding window
///
/// Uses a fixed-size training window that slides forward.
///
/// # Arguments
///
/// * `data_len` - Length of the time series
/// * `train_size` - Fixed size of training window
/// * `horizon` - Number of steps to forecast
/// * `step` - How many observations to slide between folds
pub fn sliding_window_split(
    data_len: usize,
    train_size: usize,
    horizon: usize,
    step: usize,
) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
    let mut splits = Vec::new();
    let step = step.max(1);

    let mut train_start = 0;
    while train_start + train_size + horizon <= data_len {
        let train_range = train_start..(train_start + train_size);
        let test_range = (train_start + train_size)..(train_start + train_size + horizon).min(data_len);

        splits.push((train_range, test_range));
        train_start += step;
    }

    splits
}

/// Results from cross-validation
#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    /// MAE for each fold
    pub mae_scores: Vec<f64>,
    /// RMSE for each fold
    pub rmse_scores: Vec<f64>,
    /// MAPE for each fold
    pub mape_scores: Vec<f64>,
    /// Mean MAE across folds
    pub mean_mae: f64,
    /// Mean RMSE across folds
    pub mean_rmse: f64,
    /// Standard deviation of MAE
    pub std_mae: f64,
}

impl CrossValidationResults {
    /// Create results from fold scores
    pub fn from_scores(mae_scores: Vec<f64>, rmse_scores: Vec<f64>, mape_scores: Vec<f64>) -> Self {
        let n = mae_scores.len() as f64;

        let mean_mae = if mae_scores.is_empty() {
            f64::NAN
        } else {
            mae_scores.iter().sum::<f64>() / n
        };

        let mean_rmse = if rmse_scores.is_empty() {
            f64::NAN
        } else {
            rmse_scores.iter().sum::<f64>() / n
        };

        let std_mae = if mae_scores.len() < 2 {
            f64::NAN
        } else {
            let variance: f64 = mae_scores.iter().map(|x| (x - mean_mae).powi(2)).sum::<f64>() / (n - 1.0);
            variance.sqrt()
        };

        Self {
            mae_scores,
            rmse_scores,
            mape_scores,
            mean_mae,
            mean_rmse,
            std_mae,
        }
    }

    /// Print summary
    pub fn summary(&self) -> String {
        format!(
            "Cross-Validation Results:\n  Folds: {}\n  Mean MAE: {:.4} (+/- {:.4})\n  Mean RMSE: {:.4}",
            self.mae_scores.len(),
            self.mean_mae,
            self.std_mae,
            self.mean_rmse
        )
    }
}

/// Perform cross-validation on a predictor
///
/// # Type Parameters
///
/// * `P` - Predictor type that implements Clone
/// * `F` - Factory function to create new predictor instances
pub fn cross_validate<P, F>(
    data: &[f64],
    create_predictor: F,
    min_train_size: usize,
    horizon: usize,
    step: usize,
) -> crate::Result<CrossValidationResults>
where
    P: Predictor,
    F: Fn() -> crate::Result<P>,
{
    let splits = expanding_window_split(data.len(), min_train_size, horizon, step);

    if splits.is_empty() {
        return Err(crate::TsError::InsufficientData {
            required: min_train_size + horizon,
            actual: data.len(),
        });
    }

    let mut mae_scores = Vec::new();
    let mut rmse_scores = Vec::new();
    let mut mape_scores = Vec::new();

    for (train_range, test_range) in splits {
        let train_data = &data[train_range];
        let test_data = &data[test_range.clone()];

        let mut predictor = create_predictor()?;

        if predictor.fit(train_data).is_ok() {
            if let Ok(predictions) = predictor.predict(test_range.len()) {
                mae_scores.push(metrics::mae(test_data, &predictions));
                rmse_scores.push(metrics::rmse(test_data, &predictions));
                mape_scores.push(metrics::mape(test_data, &predictions));
            }
        }
    }

    Ok(CrossValidationResults::from_scores(mae_scores, rmse_scores, mape_scores))
}

/// Train-test split for time series
///
/// Returns (train_data, test_data) respecting temporal order.
pub fn train_test_split(data: &[f64], test_ratio: f64) -> (&[f64], &[f64]) {
    let test_ratio = test_ratio.clamp(0.1, 0.9);
    let split_idx = ((1.0 - test_ratio) * data.len() as f64) as usize;
    let split_idx = split_idx.max(1).min(data.len() - 1);

    (&data[..split_idx], &data[split_idx..])
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
