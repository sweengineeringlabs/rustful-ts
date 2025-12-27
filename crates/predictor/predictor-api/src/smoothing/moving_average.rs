//! Moving Average methods for time series smoothing and forecasting
//!
//! Moving averages smooth time series data by averaging observations within a window.
//! They are fundamental building blocks for more complex forecasting methods.
//!
//! ## Types
//!
//! - **Simple Moving Average (SMA)**: Equal weights for all observations in window
//! - **Weighted Moving Average (WMA)**: Custom weights, typically more recent = higher weight
//! - **Exponential Moving Average**: See `exponential_smoothing` module

use crate::Predictor;
use crate::{Result, TsError};
use predictor_spi::Result as SpiResult;
use serde::{Deserialize, Serialize};

/// Simple Moving Average (SMA)
///
/// Computes the unweighted mean of the previous `window` observations.
/// Useful for smoothing noisy data and identifying trends.
///
/// @algorithm SMA
/// @category SmoothingMethod
/// @complexity O(n) fit, O(1) predict
/// @thread_safe false
/// @since 0.1.0
///
/// # Example
///
/// ```rust
/// use predictor_api::smoothing::SimpleMovingAverage;
/// use predictor_api::Predictor;
///
/// let data = vec![10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0];
/// let mut sma = SimpleMovingAverage::new(3).unwrap();
/// sma.fit(&data).unwrap();
///
/// // Get smoothed values
/// let smoothed = sma.smoothed_values();
///
/// // Forecast future values
/// let forecast = sma.predict(3).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SimpleMovingAverage {
    /// Window size for averaging
    window: usize,
    /// Smoothed values after fitting
    smoothed: Vec<f64>,
    /// Last window of original values (for prediction)
    last_window: Vec<f64>,
    /// Whether model has been fitted
    fitted: bool,
}

impl SimpleMovingAverage {
    /// Create a new SMA with specified window size
    ///
    /// # Arguments
    ///
    /// * `window` - Number of observations to average (must be >= 2)
    pub fn new(window: usize) -> Result<Self> {
        if window < 2 {
            return Err(TsError::InvalidParameter {
                name: "window".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }

        Ok(Self {
            window,
            smoothed: Vec::new(),
            last_window: Vec::new(),
            fitted: false,
        })
    }

    /// Get the smoothed time series
    pub fn smoothed_values(&self) -> &[f64] {
        &self.smoothed
    }

    /// Get window size
    pub fn window_size(&self) -> usize {
        self.window
    }

    /// Compute SMA for a given data slice
    pub fn compute(data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(data.len() - window + 1);

        // First window
        let mut sum: f64 = data[..window].iter().sum();
        result.push(sum / window as f64);

        // Sliding window
        for i in window..data.len() {
            sum = sum - data[i - window] + data[i];
            result.push(sum / window as f64);
        }

        result
    }
}

impl Predictor for SimpleMovingAverage {
    fn fit(&mut self, data: &[f64]) -> SpiResult<()> {
        if data.len() < self.window {
            return Err(TsError::InsufficientData {
                required: self.window,
                actual: data.len(),
            }
            .into());
        }

        self.smoothed = Self::compute(data, self.window);
        self.last_window = data[data.len() - self.window..].to_vec();
        self.fitted = true;

        Ok(())
    }

    fn predict(&self, steps: usize) -> SpiResult<Vec<f64>> {
        if !self.fitted {
            return Err(TsError::NotFitted.into());
        }

        // SMA forecast is just the last computed average
        let last_avg = *self.smoothed.last().unwrap();
        Ok(vec![last_avg; steps])
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// Weighted Moving Average (WMA)
///
/// Like SMA but with custom weights for each position in the window.
/// Typically used with linearly decreasing weights (most recent has highest weight).
///
/// @algorithm WMA
/// @category SmoothingMethod
/// @complexity O(n*w) fit, O(1) predict
/// @thread_safe false
/// @since 0.1.0
///
/// # Example
///
/// ```rust
/// use predictor_api::smoothing::WeightedMovingAverage;
/// use predictor_api::Predictor;
///
/// let data = vec![10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 18.0];
///
/// // Linear weights: [1, 2, 3] for window of 3
/// let weights = vec![1.0, 2.0, 3.0];
/// let mut wma = WeightedMovingAverage::new(weights).unwrap();
/// wma.fit(&data).unwrap();
/// let forecast = wma.predict(3).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WeightedMovingAverage {
    /// Weights for each position (normalized internally)
    weights: Vec<f64>,
    /// Normalized weights
    normalized_weights: Vec<f64>,
    /// Smoothed values after fitting
    smoothed: Vec<f64>,
    /// Last window of original values
    last_window: Vec<f64>,
    /// Whether model has been fitted
    fitted: bool,
}

impl WeightedMovingAverage {
    /// Create a new WMA with specified weights
    ///
    /// Weights are applied in order: first weight to oldest observation.
    /// Weights will be normalized to sum to 1.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weights for each position in window (must have at least 2 elements)
    pub fn new(weights: Vec<f64>) -> Result<Self> {
        if weights.len() < 2 {
            return Err(TsError::InvalidParameter {
                name: "weights".to_string(),
                reason: "must have at least 2 weights".to_string(),
            });
        }

        if weights.iter().any(|&w| w < 0.0) {
            return Err(TsError::InvalidParameter {
                name: "weights".to_string(),
                reason: "all weights must be non-negative".to_string(),
            });
        }

        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            return Err(TsError::InvalidParameter {
                name: "weights".to_string(),
                reason: "weights must sum to a positive value".to_string(),
            });
        }

        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / sum).collect();

        Ok(Self {
            weights,
            normalized_weights,
            smoothed: Vec::new(),
            last_window: Vec::new(),
            fitted: false,
        })
    }

    /// Create WMA with linear weights (1, 2, 3, ..., n)
    pub fn linear(window: usize) -> Result<Self> {
        if window < 2 {
            return Err(TsError::InvalidParameter {
                name: "window".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }

        let weights: Vec<f64> = (1..=window).map(|i| i as f64).collect();
        Self::new(weights)
    }

    /// Get the smoothed time series
    pub fn smoothed_values(&self) -> &[f64] {
        &self.smoothed
    }

    /// Get weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get window size
    pub fn window_size(&self) -> usize {
        self.weights.len()
    }
}

impl Predictor for WeightedMovingAverage {
    fn fit(&mut self, data: &[f64]) -> SpiResult<()> {
        let window = self.weights.len();
        if data.len() < window {
            return Err(TsError::InsufficientData {
                required: window,
                actual: data.len(),
            }
            .into());
        }

        self.smoothed = Vec::with_capacity(data.len() - window + 1);

        for i in 0..=(data.len() - window) {
            let weighted_sum: f64 = self
                .normalized_weights
                .iter()
                .zip(&data[i..i + window])
                .map(|(w, v)| w * v)
                .sum();
            self.smoothed.push(weighted_sum);
        }

        self.last_window = data[data.len() - window..].to_vec();
        self.fitted = true;

        Ok(())
    }

    fn predict(&self, steps: usize) -> SpiResult<Vec<f64>> {
        if !self.fitted {
            return Err(TsError::NotFitted.into());
        }

        let last_avg = *self.smoothed.last().unwrap();
        Ok(vec![last_avg; steps])
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

