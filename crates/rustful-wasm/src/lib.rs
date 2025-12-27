//! # rustful-wasm
//!
//! WebAssembly bindings for the rustful-ts time series library.
//! This crate exposes all core algorithms to JavaScript/TypeScript through WASM.

use wasm_bindgen::prelude::*;
use rustful_core::algorithms::{
    Predictor,
    arima::Arima,
    exponential_smoothing::*,
    moving_average::*,
    linear_regression::*,
    knn::*,
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
    rustful_core::utils::metrics::mae(actual, predicted)
}

/// Compute Root Mean Squared Error
#[wasm_bindgen]
pub fn compute_rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    rustful_core::utils::metrics::rmse(actual, predicted)
}

/// Compute Mean Absolute Percentage Error
#[wasm_bindgen]
pub fn compute_mape(actual: &[f64], predicted: &[f64]) -> f64 {
    rustful_core::utils::metrics::mape(actual, predicted)
}

/// Normalize data to [0, 1]
#[wasm_bindgen]
pub fn normalize_data(data: &[f64]) -> Vec<f64> {
    rustful_core::utils::preprocessing::normalize(data).0
}

/// Standardize data to zero mean, unit variance
#[wasm_bindgen]
pub fn standardize_data(data: &[f64]) -> Vec<f64> {
    rustful_core::utils::preprocessing::standardize(data).0
}

/// Compute first-order differences
#[wasm_bindgen]
pub fn difference_data(data: &[f64], order: usize) -> Vec<f64> {
    rustful_core::utils::preprocessing::difference(data, order)
}
