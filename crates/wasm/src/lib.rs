//! # rustful-wasm
//!
//! WebAssembly bindings for the rustful-ts time series library.
//! This crate exposes all core algorithms to JavaScript/TypeScript through WASM.

use wasm_bindgen::prelude::*;
use algorithm::{
    ml::{DistanceMetric, TimeSeriesKNN},
    regression::{Arima, LinearRegression},
    smoothing::{
        DoubleExponentialSmoothing, HoltWinters, SeasonalType, SimpleExponentialSmoothing,
        SimpleMovingAverage,
    },
    Predictor,
};

/// Initialize panic hook for better error messages in WASM
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

// ============================================================================
// Algorithm Wrappers
// ============================================================================

/// ARIMA model for WASM
#[wasm_bindgen]
pub struct WasmArima {
    inner: Arima,
}

#[wasm_bindgen]
impl WasmArima {
    /// Create a new ARIMA(p, d, q) model
    #[wasm_bindgen(constructor)]
    pub fn new(p: usize, d: usize, q: usize) -> Result<WasmArima, JsValue> {
        let inner = Arima::new(p, d, q).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Fit the model to data
    pub fn fit(&mut self, data: &[f64]) -> Result<(), JsValue> {
        self.inner.fit(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }

    /// Predict future values
    pub fn predict(&self, steps: usize) -> Result<Vec<f64>, JsValue> {
        self.inner.predict(steps).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }
}

/// Simple Exponential Smoothing for WASM
#[wasm_bindgen]
pub struct WasmSES {
    inner: SimpleExponentialSmoothing,
}

#[wasm_bindgen]
impl WasmSES {
    /// Create a new SES model
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64) -> Result<WasmSES, JsValue> {
        let inner = SimpleExponentialSmoothing::new(alpha).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Fit the model
    pub fn fit(&mut self, data: &[f64]) -> Result<(), JsValue> {
        self.inner.fit(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }

    /// Predict future values
    pub fn predict(&self, steps: usize) -> Result<Vec<f64>, JsValue> {
        self.inner.predict(steps).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get current level
    pub fn level(&self) -> f64 {
        self.inner.level()
    }
}

/// Double Exponential Smoothing (Holt's Method) for WASM
#[wasm_bindgen]
pub struct WasmHolt {
    inner: DoubleExponentialSmoothing,
}

#[wasm_bindgen]
impl WasmHolt {
    /// Create a new Holt's method model
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64, beta: f64) -> Result<WasmHolt, JsValue> {
        let inner = DoubleExponentialSmoothing::new(alpha, beta).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Fit the model
    pub fn fit(&mut self, data: &[f64]) -> Result<(), JsValue> {
        self.inner.fit(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }

    /// Predict future values
    pub fn predict(&self, steps: usize) -> Result<Vec<f64>, JsValue> {
        self.inner.predict(steps).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Holt-Winters (Triple Exponential Smoothing) for WASM
#[wasm_bindgen]
pub struct WasmHoltWinters {
    inner: HoltWinters,
}

#[wasm_bindgen]
impl WasmHoltWinters {
    /// Create a new Holt-Winters model with additive seasonality
    #[wasm_bindgen(constructor)]
    pub fn new_additive(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
    ) -> Result<WasmHoltWinters, JsValue> {
        let inner = HoltWinters::new(alpha, beta, gamma, period, SeasonalType::Additive)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create with multiplicative seasonality
    pub fn new_multiplicative(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
    ) -> Result<WasmHoltWinters, JsValue> {
        let inner = HoltWinters::new(alpha, beta, gamma, period, SeasonalType::Multiplicative)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Fit the model
    pub fn fit(&mut self, data: &[f64]) -> Result<(), JsValue> {
        self.inner.fit(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }

    /// Predict future values
    pub fn predict(&self, steps: usize) -> Result<Vec<f64>, JsValue> {
        self.inner.predict(steps).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get seasonal components
    pub fn seasonal_components(&self) -> Vec<f64> {
        self.inner.seasonal_components().to_vec()
    }
}

/// Simple Moving Average for WASM
#[wasm_bindgen]
pub struct WasmSMA {
    inner: SimpleMovingAverage,
}

#[wasm_bindgen]
impl WasmSMA {
    /// Create a new SMA
    #[wasm_bindgen(constructor)]
    pub fn new(window: usize) -> Result<WasmSMA, JsValue> {
        let inner = SimpleMovingAverage::new(window).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Fit and get smoothed values
    pub fn fit(&mut self, data: &[f64]) -> Result<(), JsValue> {
        self.inner.fit(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }

    /// Get smoothed values
    pub fn smoothed_values(&self) -> Vec<f64> {
        self.inner.smoothed_values().to_vec()
    }

    /// Predict future values
    pub fn predict(&self, steps: usize) -> Result<Vec<f64>, JsValue> {
        self.inner.predict(steps).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Linear Regression for WASM
#[wasm_bindgen]
pub struct WasmLinearRegression {
    inner: LinearRegression,
}

#[wasm_bindgen]
impl WasmLinearRegression {
    /// Create a new linear regression model
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmLinearRegression {
        Self {
            inner: LinearRegression::new(),
        }
    }

    /// Fit the model
    pub fn fit(&mut self, data: &[f64]) -> Result<(), JsValue> {
        self.inner.fit(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }

    /// Predict future values
    pub fn predict(&self, steps: usize) -> Result<Vec<f64>, JsValue> {
        self.inner.predict(steps).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get slope
    pub fn slope(&self) -> f64 {
        self.inner.slope()
    }

    /// Get intercept
    pub fn intercept(&self) -> f64 {
        self.inner.intercept()
    }

    /// Get R-squared
    pub fn r_squared(&self) -> f64 {
        self.inner.r_squared()
    }
}

impl Default for WasmLinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// K-Nearest Neighbors for time series prediction
#[wasm_bindgen]
pub struct WasmKNN {
    inner: TimeSeriesKNN,
}

#[wasm_bindgen]
impl WasmKNN {
    /// Create a new KNN model with Euclidean distance
    #[wasm_bindgen(constructor)]
    pub fn new(k: usize, window_size: usize) -> Result<WasmKNN, JsValue> {
        let inner = TimeSeriesKNN::new(k, window_size, DistanceMetric::Euclidean)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create with Manhattan distance
    pub fn new_manhattan(k: usize, window_size: usize) -> Result<WasmKNN, JsValue> {
        let inner = TimeSeriesKNN::new(k, window_size, DistanceMetric::Manhattan)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Fit the model
    pub fn fit(&mut self, data: &[f64]) -> Result<(), JsValue> {
        self.inner.fit(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }

    /// Predict future values
    pub fn predict(&self, steps: usize) -> Result<Vec<f64>, JsValue> {
        self.inner.predict(steps).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    /// Get number of stored patterns
    pub fn n_patterns(&self) -> usize {
        self.inner.n_patterns()
    }

    /// Get window size
    pub fn window_size(&self) -> usize {
        self.inner.window_size()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute Mean Absolute Error
#[wasm_bindgen]
pub fn compute_mae(actual: &[f64], predicted: &[f64]) -> f64 {
    algorithm::utils::metrics::mae(actual, predicted)
}

/// Compute Root Mean Squared Error
#[wasm_bindgen]
pub fn compute_rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    algorithm::utils::metrics::rmse(actual, predicted)
}

/// Compute Mean Absolute Percentage Error
#[wasm_bindgen]
pub fn compute_mape(actual: &[f64], predicted: &[f64]) -> f64 {
    algorithm::utils::metrics::mape(actual, predicted)
}

/// Normalize data to [0, 1]
#[wasm_bindgen]
pub fn normalize_data(data: &[f64]) -> Vec<f64> {
    algorithm::utils::preprocessing::normalize(data).0
}

/// Standardize data to zero mean, unit variance
#[wasm_bindgen]
pub fn standardize_data(data: &[f64]) -> Vec<f64> {
    algorithm::utils::preprocessing::standardize(data).0
}

/// Compute first-order differences
#[wasm_bindgen]
pub fn difference_data(data: &[f64], order: usize) -> Vec<f64> {
    algorithm::utils::preprocessing::difference(data, order)
}

// ============================================================================
// Anomaly Detection Bindings
// ============================================================================

use anomaly::{ZScoreDetector, IQRDetector, AnomalyDetector};

/// Z-Score anomaly detector for WASM
#[wasm_bindgen]
pub struct WasmZScoreDetector {
    inner: ZScoreDetector,
}

#[wasm_bindgen]
impl WasmZScoreDetector {
    /// Create a new Z-Score detector with given threshold (default: 3.0)
    #[wasm_bindgen(constructor)]
    pub fn new(threshold: f64) -> WasmZScoreDetector {
        Self {
            inner: ZScoreDetector::new(threshold),
        }
    }

    /// Fit the detector to training data
    pub fn fit(&mut self, data: &[f64]) -> Result<(), JsValue> {
        self.inner.fit(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Detect anomalies and return boolean array
    pub fn detect(&self, data: &[f64]) -> Result<Vec<u8>, JsValue> {
        let result = self.inner.detect(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(result.is_anomaly.iter().map(|&b| if b { 1 } else { 0 }).collect())
    }

    /// Get anomaly scores
    pub fn score(&self, data: &[f64]) -> Result<Vec<f64>, JsValue> {
        self.inner.score(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// IQR anomaly detector for WASM
#[wasm_bindgen]
pub struct WasmIQRDetector {
    inner: IQRDetector,
}

#[wasm_bindgen]
impl WasmIQRDetector {
    /// Create a new IQR detector with given multiplier (default: 1.5)
    #[wasm_bindgen(constructor)]
    pub fn new(multiplier: f64) -> WasmIQRDetector {
        Self {
            inner: IQRDetector::new(multiplier),
        }
    }

    /// Fit the detector to training data
    pub fn fit(&mut self, data: &[f64]) -> Result<(), JsValue> {
        self.inner.fit(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Detect anomalies and return boolean array
    pub fn detect(&self, data: &[f64]) -> Result<Vec<u8>, JsValue> {
        let result = self.inner.detect(data).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(result.is_anomaly.iter().map(|&b| if b { 1 } else { 0 }).collect())
    }

    /// Get anomaly scores
    pub fn score(&self, data: &[f64]) -> Result<Vec<f64>, JsValue> {
        self.inner.score(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

// ============================================================================
// Financial Risk Metrics
// ============================================================================

/// Calculate Value at Risk (historical method)
#[wasm_bindgen]
pub fn compute_var(returns: &[f64], confidence: f64) -> f64 {
    financial::risk::var_historical(returns, confidence)
}

/// Calculate Sharpe ratio
#[wasm_bindgen]
pub fn compute_sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    financial::risk::sharpe_ratio(returns, risk_free_rate)
}

/// Calculate maximum drawdown
#[wasm_bindgen]
pub fn compute_max_drawdown(equity_curve: &[f64]) -> f64 {
    financial::risk::max_drawdown(equity_curve)
}

/// Calculate Sortino ratio
#[wasm_bindgen]
pub fn compute_sortino_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    financial::risk::sortino_ratio(returns, risk_free_rate)
}

// ============================================================================
// Pipeline Bindings
// ============================================================================

use forecast::{Pipeline, NormalizeStep, DifferenceStep, StandardizeStep};

/// Forecast pipeline for WASM
#[wasm_bindgen]
pub struct WasmPipeline {
    inner: Pipeline,
}

#[wasm_bindgen]
impl WasmPipeline {
    /// Create a new empty pipeline
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmPipeline {
        Self {
            inner: Pipeline::new(),
        }
    }

    /// Add normalization step
    pub fn add_normalize(&mut self) {
        self.inner.add_step(Box::new(NormalizeStep::new()));
    }

    /// Add standardization step
    pub fn add_standardize(&mut self) {
        self.inner.add_step(Box::new(StandardizeStep::new()));
    }

    /// Add differencing step
    pub fn add_difference(&mut self, order: usize) {
        self.inner.add_step(Box::new(DifferenceStep::new(order)));
    }

    /// Transform data through pipeline
    pub fn transform(&mut self, data: &[f64]) -> Result<Vec<f64>, JsValue> {
        self.inner.fit_transform(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Inverse transform data
    pub fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>, JsValue> {
        self.inner.inverse_transform(data).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for WasmPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AutoML / Ensemble Bindings
// ============================================================================

use automl::{EnsembleMethod, combine_predictions};

/// Combine predictions using average
#[wasm_bindgen]
pub fn ensemble_average(predictions: JsValue) -> Result<Vec<f64>, JsValue> {
    let preds: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(predictions)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(combine_predictions(&preds, EnsembleMethod::Average, None))
}

/// Combine predictions using weighted average
#[wasm_bindgen]
pub fn ensemble_weighted_average(predictions: JsValue, weights: &[f64]) -> Result<Vec<f64>, JsValue> {
    let preds: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(predictions)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(combine_predictions(&preds, EnsembleMethod::WeightedAverage, Some(weights)))
}

/// Combine predictions using median
#[wasm_bindgen]
pub fn ensemble_median(predictions: JsValue) -> Result<Vec<f64>, JsValue> {
    let preds: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(predictions)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(combine_predictions(&preds, EnsembleMethod::Median, None))
}
