//! K-Nearest Neighbors for time series prediction
//!
//! Adapts the KNN algorithm for time series forecasting by finding similar
//! historical patterns and using them to predict future values.
//!
//! ## How It Works
//!
//! 1. Extract sliding windows (subsequences) from historical data
//! 2. For a new prediction, find the K most similar historical windows
//! 3. Predict based on what happened after those similar windows
//!
//! ## Distance Metrics
//!
//! - Euclidean distance (default)
//! - Dynamic Time Warping (DTW) for time-shifted patterns

use crate::Predictor;
use crate::{Result, TsError};
use serde::{Deserialize, Serialize};

/// Distance metric for comparing time series subsequences
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Standard Euclidean distance
    Euclidean,
    /// Manhattan (L1) distance
    Manhattan,
}

/// K-Nearest Neighbors time series predictor
///
/// @algorithm KNN
/// @category MachineLearning
/// @complexity O(n*k*w) fit, O(n*k) predict
/// @thread_safe false
/// @since 0.1.0
///
/// # Example
///
/// ```rust
/// use algorithm::ml::{TimeSeriesKNN, DistanceMetric};
/// use algorithm::Predictor;
///
/// let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 10.0 + 50.0).collect();
///
/// let mut knn = TimeSeriesKNN::new(5, 10, DistanceMetric::Euclidean).unwrap();
/// knn.fit(&data).unwrap();
/// let forecast = knn.predict(5).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TimeSeriesKNN {
    /// Number of neighbors
    k: usize,
    /// Window size for pattern matching
    window_size: usize,
    /// Distance metric
    metric: DistanceMetric,
    /// Stored windows from training data
    windows: Vec<Vec<f64>>,
    /// Values that followed each window
    next_values: Vec<Vec<f64>>,
    /// Last window from training data (for prediction)
    last_window: Vec<f64>,
    /// Forecast horizon stored during training
    forecast_horizon: usize,
    /// Whether model has been fitted
    fitted: bool,
}

impl TimeSeriesKNN {
    /// Create a new KNN time series predictor
    ///
    /// # Arguments
    ///
    /// * `k` - Number of neighbors to consider
    /// * `window_size` - Size of the pattern window
    /// * `metric` - Distance metric for similarity
    pub fn new(k: usize, window_size: usize, metric: DistanceMetric) -> Result<Self> {
        if k < 1 {
            return Err(TsError::InvalidParameter {
                name: "k".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if window_size < 2 {
            return Err(TsError::InvalidParameter {
                name: "window_size".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }

        Ok(Self {
            k,
            window_size,
            metric,
            windows: Vec::new(),
            next_values: Vec::new(),
            last_window: Vec::new(),
            forecast_horizon: 1,
            fitted: false,
        })
    }

    /// Set the forecast horizon (how many steps after each window to store)
    pub fn with_horizon(mut self, horizon: usize) -> Self {
        self.forecast_horizon = horizon.max(1);
        self
    }

    /// Compute distance between two windows
    fn compute_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }
            DistanceMetric::Manhattan => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs())
                    .sum()
            }
        }
    }

    /// Find K nearest neighbors to a query window
    fn find_neighbors(&self, query: &[f64]) -> Vec<(usize, f64)> {
        let mut distances: Vec<(usize, f64)> = self
            .windows
            .iter()
            .enumerate()
            .map(|(i, window)| (i, self.compute_distance(query, window)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(self.k);

        distances
    }

    /// Get the number of stored patterns
    pub fn n_patterns(&self) -> usize {
        self.windows.len()
    }

    /// Get window size
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

impl Predictor for TimeSeriesKNN {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        let min_required = self.window_size + self.forecast_horizon + self.k;
        if data.len() < min_required {
            return Err(TsError::InsufficientData {
                required: min_required,
                actual: data.len(),
            });
        }

        self.windows.clear();
        self.next_values.clear();

        // Extract all windows and their following values
        let max_start = data.len() - self.window_size - self.forecast_horizon;
        for i in 0..=max_start {
            let window = data[i..i + self.window_size].to_vec();
            let next = data[i + self.window_size..i + self.window_size + self.forecast_horizon].to_vec();

            self.windows.push(window);
            self.next_values.push(next);
        }

        // Store last window for prediction
        self.last_window = data[data.len() - self.window_size..].to_vec();

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, steps: usize) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(TsError::NotFitted);
        }

        if self.windows.is_empty() {
            return Err(TsError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }

        let mut forecasts = Vec::with_capacity(steps);
        let mut current_window = self.last_window.clone();

        for step in 0..steps {
            // Find neighbors
            let neighbors = self.find_neighbors(&current_window);

            if neighbors.is_empty() {
                // Fallback: use mean of all next values
                let mean: f64 = self
                    .next_values
                    .iter()
                    .filter_map(|v| v.first())
                    .sum::<f64>()
                    / self.next_values.len() as f64;
                forecasts.push(mean);
            } else {
                // Weighted average based on inverse distance
                let value_index = step.min(self.forecast_horizon - 1);

                let total_weight: f64 = neighbors
                    .iter()
                    .map(|(_, d)| 1.0 / (d + 1e-10))
                    .sum();

                let weighted_sum: f64 = neighbors
                    .iter()
                    .map(|(idx, d)| {
                        let weight = 1.0 / (d + 1e-10);
                        let value = self.next_values[*idx].get(value_index).copied().unwrap_or(0.0);
                        weight * value
                    })
                    .sum();

                let forecast = weighted_sum / total_weight;
                forecasts.push(forecast);
            }

            // Update window for next step
            if step < steps - 1 {
                current_window.remove(0);
                current_window.push(*forecasts.last().unwrap());
            }
        }

        Ok(forecasts)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// Private method tests must stay here
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metrics() {
        let knn = TimeSeriesKNN::new(1, 3, DistanceMetric::Euclidean).unwrap();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let euclidean = knn.compute_distance(&a, &b);
        assert!((euclidean - (27.0_f64).sqrt()).abs() < 1e-10);

        let knn = TimeSeriesKNN::new(1, 3, DistanceMetric::Manhattan).unwrap();
        let manhattan = knn.compute_distance(&a, &b);
        assert!((manhattan - 9.0).abs() < 1e-10);
    }
}
