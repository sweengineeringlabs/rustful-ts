//! Predictor Core
//!
//! Core types, error handling, and utilities for time series predictors.

pub use predictor_spi::{IncrementalPredictor, Predictor};
use thiserror::Error;

/// Re-export SPI Result type for trait implementations
pub use predictor_spi::Result as SpiResult;

/// Result type for predictor operations
pub type Result<T> = std::result::Result<T, TsError>;

/// Errors that can occur during time series operations
#[derive(Error, Debug)]
pub enum TsError {
    /// Insufficient data points for the operation
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },

    /// Invalid parameter value
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Model has not been fitted yet
    #[error("Model must be fitted before prediction")]
    NotFitted,

    /// Convergence failure during optimization
    #[error("Optimization failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },

    /// Numerical computation error
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Invalid time series data
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

pub mod utils {
    //! Utility functions for time series operations

    pub mod metrics {
        //! Forecast accuracy metrics

        /// Mean Absolute Error
        pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
            if actual.len() != predicted.len() || actual.is_empty() {
                return f64::NAN;
            }
            let sum: f64 = actual
                .iter()
                .zip(predicted.iter())
                .map(|(a, p)| (a - p).abs())
                .sum();
            sum / actual.len() as f64
        }

        /// Mean Squared Error
        pub fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
            if actual.len() != predicted.len() || actual.is_empty() {
                return f64::NAN;
            }
            let sum: f64 = actual
                .iter()
                .zip(predicted.iter())
                .map(|(a, p)| (a - p).powi(2))
                .sum();
            sum / actual.len() as f64
        }

        /// Root Mean Squared Error
        pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
            mse(actual, predicted).sqrt()
        }

        /// Mean Absolute Percentage Error
        pub fn mape(actual: &[f64], predicted: &[f64]) -> f64 {
            if actual.len() != predicted.len() || actual.is_empty() {
                return f64::NAN;
            }
            let valid: Vec<_> = actual
                .iter()
                .zip(predicted.iter())
                .filter(|(&a, _)| a.abs() > 1e-10)
                .collect();
            if valid.is_empty() {
                return f64::NAN;
            }
            let sum: f64 = valid.iter().map(|(&a, &p)| ((a - p) / a).abs()).sum();
            sum / valid.len() as f64
        }

        /// R-squared (coefficient of determination)
        pub fn r_squared(actual: &[f64], predicted: &[f64]) -> f64 {
            if actual.len() != predicted.len() || actual.is_empty() {
                return f64::NAN;
            }
            let mean: f64 = actual.iter().sum::<f64>() / actual.len() as f64;
            let ss_tot: f64 = actual.iter().map(|a| (a - mean).powi(2)).sum();
            let ss_res: f64 = actual
                .iter()
                .zip(predicted.iter())
                .map(|(a, p)| (a - p).powi(2))
                .sum();
            1.0 - (ss_res / ss_tot)
        }
    }

    pub mod preprocessing {
        //! Data preprocessing utilities

        /// Normalize data to [0, 1] range
        pub fn normalize(data: &[f64]) -> (Vec<f64>, f64, f64) {
            let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = max - min;
            let normalized = if range == 0.0 {
                vec![0.5; data.len()]
            } else {
                data.iter().map(|&x| (x - min) / range).collect()
            };
            (normalized, min, max)
        }

        /// Denormalize data from [0, 1] range
        pub fn denormalize(data: &[f64], min: f64, max: f64) -> Vec<f64> {
            let range = max - min;
            data.iter().map(|&x| x * range + min).collect()
        }

        /// Standardize data to zero mean and unit variance
        pub fn standardize(data: &[f64]) -> (Vec<f64>, f64, f64) {
            let n = data.len() as f64;
            let mean = data.iter().sum::<f64>() / n;
            let std_dev = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
            let standardized = if std_dev == 0.0 {
                vec![0.0; data.len()]
            } else {
                data.iter().map(|&x| (x - mean) / std_dev).collect()
            };
            (standardized, mean, std_dev)
        }

        /// Destandardize data
        pub fn destandardize(data: &[f64], mean: f64, std_dev: f64) -> Vec<f64> {
            data.iter().map(|&x| x * std_dev + mean).collect()
        }

        /// First-order differencing
        pub fn difference(data: &[f64], order: usize) -> Vec<f64> {
            let mut result = data.to_vec();
            for _ in 0..order {
                if result.len() < 2 {
                    return vec![];
                }
                result = result.windows(2).map(|w| w[1] - w[0]).collect();
            }
            result
        }
    }

    pub mod validation {
        //! Cross-validation utilities

        /// Train-test split respecting temporal order
        pub fn train_test_split(data: &[f64], test_ratio: f64) -> (&[f64], &[f64]) {
            let ratio = test_ratio.clamp(0.1, 0.9);
            let split_idx = ((1.0 - ratio) * data.len() as f64) as usize;
            let split_idx = split_idx.max(1).min(data.len() - 1);
            (&data[..split_idx], &data[split_idx..])
        }

        /// Expanding window cross-validation splits
        pub fn expanding_window_split(
            data_len: usize,
            min_train_size: usize,
            horizon: usize,
            step: usize,
        ) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
            let mut splits = Vec::new();
            let mut train_end = min_train_size;

            while train_end + horizon <= data_len {
                let train_range = 0..train_end;
                let test_range = train_end..train_end + horizon;
                splits.push((train_range, test_range));
                train_end += step;
            }

            splits
        }

        /// Sliding window cross-validation splits
        pub fn sliding_window_split(
            data_len: usize,
            train_size: usize,
            horizon: usize,
            step: usize,
        ) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
            let mut splits = Vec::new();
            let mut start = 0;

            while start + train_size + horizon <= data_len {
                let train_range = start..start + train_size;
                let test_range = start + train_size..start + train_size + horizon;
                splits.push((train_range, test_range));
                start += step;
            }

            splits
        }
    }
}
