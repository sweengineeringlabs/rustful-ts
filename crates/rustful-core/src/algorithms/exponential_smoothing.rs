//! Exponential Smoothing methods for time series forecasting
//!
//! Exponential smoothing methods assign exponentially decreasing weights to past
//! observations. This module implements three variants:
//!
//! - **Simple (SES)**: For data without trend or seasonality
//! - **Double (Holt's)**: For data with trend but no seasonality
//! - **Triple (Holt-Winters)**: For data with both trend and seasonality
//!
//! ## Choosing Parameters
//!
//! - `alpha` (level): Higher values = more responsive to recent changes (0.1-0.3 typical)
//! - `beta` (trend): Controls trend smoothing (0.1-0.2 typical)
//! - `gamma` (seasonal): Controls seasonal smoothing (0.1-0.3 typical)

use crate::algorithms::Predictor;
use crate::error::{Result, TsError};
use serde::{Deserialize, Serialize};

// ============================================================================
// Simple Exponential Smoothing (SES)
// ============================================================================

/// Simple Exponential Smoothing for stationary time series
///
/// Best for: Data without clear trend or seasonal pattern
///
/// Formula: `S_t = α * Y_t + (1 - α) * S_{t-1}`
///
/// # Example
///
/// ```rust
/// use rustful_ts::algorithms::exponential_smoothing::SimpleExponentialSmoothing;
/// use rustful_ts::algorithms::Predictor;
///
/// let data = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0];
/// let mut model = SimpleExponentialSmoothing::new(0.3).unwrap();
/// model.fit(&data).unwrap();
/// let forecast = model.predict(3).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleExponentialSmoothing {
    /// Smoothing parameter (0 < alpha < 1)
    alpha: f64,
    /// Current level estimate
    level: f64,
    /// Whether model has been fitted
    fitted: bool,
}

impl SimpleExponentialSmoothing {
    /// Create a new SES model
    ///
    /// # Arguments
    ///
    /// * `alpha` - Smoothing parameter (0 < alpha < 1)
    ///             Higher values give more weight to recent observations
    pub fn new(alpha: f64) -> Result<Self> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(TsError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be between 0 and 1 (exclusive)".to_string(),
            });
        }

        Ok(Self {
            alpha,
            level: 0.0,
            fitted: false,
        })
    }

    /// Create SES with automatic alpha selection using grid search
    pub fn auto(data: &[f64]) -> Result<Self> {
        let mut best_alpha = 0.5;
        let mut best_mse = f64::MAX;

        for alpha_int in 1..100 {
            let alpha = alpha_int as f64 / 100.0;
            let mut model = Self::new(alpha)?;
            if model.fit(data).is_ok() {
                let mse = model.compute_mse(data);
                if mse < best_mse {
                    best_mse = mse;
                    best_alpha = alpha;
                }
            }
        }

        let mut model = Self::new(best_alpha)?;
        model.fit(data)?;
        Ok(model)
    }

    fn compute_mse(&self, data: &[f64]) -> f64 {
        let mut level = data[0];
        let mut sse = 0.0;

        for i in 1..data.len() {
            let error = data[i] - level;
            sse += error * error;
            level = self.alpha * data[i] + (1.0 - self.alpha) * level;
        }

        sse / (data.len() - 1) as f64
    }

    /// Get the current level
    pub fn level(&self) -> f64 {
        self.level
    }

    /// Get alpha parameter
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl Predictor for SimpleExponentialSmoothing {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        if data.len() < 2 {
            return Err(TsError::InsufficientData {
                required: 2,
                actual: data.len(),
            });
        }

        // Initialize level with first observation
        self.level = data[0];

        // Update level through all observations
        for &value in &data[1..] {
            self.level = self.alpha * value + (1.0 - self.alpha) * self.level;
        }

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, steps: usize) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(TsError::NotFitted);
        }

        // SES produces flat forecasts
        Ok(vec![self.level; steps])
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// ============================================================================
// Double Exponential Smoothing (Holt's Method)
// ============================================================================

/// Double Exponential Smoothing (Holt's Linear Trend Method)
///
/// Extends SES to capture linear trends in the data.
///
/// Best for: Data with trend but no seasonality
///
/// # Example
///
/// ```rust
/// use rustful_ts::algorithms::exponential_smoothing::DoubleExponentialSmoothing;
/// use rustful_ts::algorithms::Predictor;
///
/// let data = vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0];
/// let mut model = DoubleExponentialSmoothing::new(0.3, 0.1).unwrap();
/// model.fit(&data).unwrap();
/// let forecast = model.predict(3).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleExponentialSmoothing {
    /// Level smoothing parameter
    alpha: f64,
    /// Trend smoothing parameter
    beta: f64,
    /// Current level
    level: f64,
    /// Current trend
    trend: f64,
    /// Whether model has been fitted
    fitted: bool,
}

impl DoubleExponentialSmoothing {
    /// Create a new Holt's method model
    ///
    /// # Arguments
    ///
    /// * `alpha` - Level smoothing (0 < alpha < 1)
    /// * `beta` - Trend smoothing (0 < beta < 1)
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(TsError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be between 0 and 1 (exclusive)".to_string(),
            });
        }
        if !(0.0 < beta && beta < 1.0) {
            return Err(TsError::InvalidParameter {
                name: "beta".to_string(),
                reason: "must be between 0 and 1 (exclusive)".to_string(),
            });
        }

        Ok(Self {
            alpha,
            beta,
            level: 0.0,
            trend: 0.0,
            fitted: false,
        })
    }

    /// Get current level and trend
    pub fn components(&self) -> (f64, f64) {
        (self.level, self.trend)
    }
}

impl Predictor for DoubleExponentialSmoothing {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        if data.len() < 3 {
            return Err(TsError::InsufficientData {
                required: 3,
                actual: data.len(),
            });
        }

        // Initialize level and trend
        self.level = data[0];
        self.trend = data[1] - data[0];

        // Update through all observations
        for &value in &data[1..] {
            let prev_level = self.level;
            self.level = self.alpha * value + (1.0 - self.alpha) * (self.level + self.trend);
            self.trend = self.beta * (self.level - prev_level) + (1.0 - self.beta) * self.trend;
        }

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, steps: usize) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(TsError::NotFitted);
        }

        let mut forecasts = Vec::with_capacity(steps);
        for h in 1..=steps {
            forecasts.push(self.level + h as f64 * self.trend);
        }

        Ok(forecasts)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// ============================================================================
// Triple Exponential Smoothing (Holt-Winters)
// ============================================================================

/// Seasonal type for Holt-Winters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SeasonalType {
    /// Additive seasonality: Y_t = Level + Trend + Season + Error
    Additive,
    /// Multiplicative seasonality: Y_t = (Level + Trend) * Season * Error
    Multiplicative,
}

/// Triple Exponential Smoothing (Holt-Winters Method)
///
/// Extends double exponential smoothing to capture seasonality.
///
/// Best for: Data with both trend and seasonal patterns
///
/// # Example
///
/// ```rust
/// use rustful_ts::algorithms::exponential_smoothing::{HoltWinters, SeasonalType};
/// use rustful_ts::algorithms::Predictor;
///
/// // Monthly data with yearly seasonality
/// let data: Vec<f64> = (0..36).map(|i| {
///     100.0 + (i as f64 * 2.0) + 20.0 * ((i as f64 * std::f64::consts::PI / 6.0).sin())
/// }).collect();
///
/// let mut model = HoltWinters::new(0.3, 0.1, 0.2, 12, SeasonalType::Additive).unwrap();
/// model.fit(&data).unwrap();
/// let forecast = model.predict(12).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoltWinters {
    /// Level smoothing parameter
    alpha: f64,
    /// Trend smoothing parameter
    beta: f64,
    /// Seasonal smoothing parameter
    gamma: f64,
    /// Seasonal period length
    period: usize,
    /// Type of seasonality
    seasonal_type: SeasonalType,
    /// Current level
    level: f64,
    /// Current trend
    trend: f64,
    /// Seasonal components
    seasonal: Vec<f64>,
    /// Whether model has been fitted
    fitted: bool,
}

impl HoltWinters {
    /// Create a new Holt-Winters model
    ///
    /// # Arguments
    ///
    /// * `alpha` - Level smoothing (0 < alpha < 1)
    /// * `beta` - Trend smoothing (0 < beta < 1)
    /// * `gamma` - Seasonal smoothing (0 < gamma < 1)
    /// * `period` - Number of observations per seasonal cycle
    /// * `seasonal_type` - Additive or Multiplicative seasonality
    pub fn new(
        alpha: f64,
        beta: f64,
        gamma: f64,
        period: usize,
        seasonal_type: SeasonalType,
    ) -> Result<Self> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(TsError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be between 0 and 1 (exclusive)".to_string(),
            });
        }
        if !(0.0 < beta && beta < 1.0) {
            return Err(TsError::InvalidParameter {
                name: "beta".to_string(),
                reason: "must be between 0 and 1 (exclusive)".to_string(),
            });
        }
        if !(0.0 < gamma && gamma < 1.0) {
            return Err(TsError::InvalidParameter {
                name: "gamma".to_string(),
                reason: "must be between 0 and 1 (exclusive)".to_string(),
            });
        }
        if period < 2 {
            return Err(TsError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }

        Ok(Self {
            alpha,
            beta,
            gamma,
            period,
            seasonal_type,
            level: 0.0,
            trend: 0.0,
            seasonal: vec![0.0; period],
            fitted: false,
        })
    }

    /// Get seasonal components
    pub fn seasonal_components(&self) -> &[f64] {
        &self.seasonal
    }

    /// Get all components: (level, trend, seasonal)
    pub fn components(&self) -> (f64, f64, &[f64]) {
        (self.level, self.trend, &self.seasonal)
    }

    fn initialize_additive(&mut self, data: &[f64]) {
        // Initialize level as mean of first season
        self.level = data[..self.period].iter().sum::<f64>() / self.period as f64;

        // Initialize trend using first two seasons
        if data.len() >= 2 * self.period {
            let first_season_avg: f64 =
                data[..self.period].iter().sum::<f64>() / self.period as f64;
            let second_season_avg: f64 =
                data[self.period..2 * self.period].iter().sum::<f64>() / self.period as f64;
            self.trend = (second_season_avg - first_season_avg) / self.period as f64;
        } else {
            self.trend = 0.0;
        }

        // Initialize seasonal factors
        for i in 0..self.period {
            self.seasonal[i] = data[i] - self.level;
        }
    }

    fn initialize_multiplicative(&mut self, data: &[f64]) {
        // Initialize level as mean of first season
        self.level = data[..self.period].iter().sum::<f64>() / self.period as f64;

        // Initialize trend
        if data.len() >= 2 * self.period {
            let first_season_avg: f64 =
                data[..self.period].iter().sum::<f64>() / self.period as f64;
            let second_season_avg: f64 =
                data[self.period..2 * self.period].iter().sum::<f64>() / self.period as f64;
            self.trend = (second_season_avg - first_season_avg) / self.period as f64;
        } else {
            self.trend = 0.0;
        }

        // Initialize seasonal factors as ratios
        for i in 0..self.period {
            self.seasonal[i] = if self.level.abs() > 1e-10 {
                data[i] / self.level
            } else {
                1.0
            };
        }
    }
}

impl Predictor for HoltWinters {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        let min_required = self.period * 2;
        if data.len() < min_required {
            return Err(TsError::InsufficientData {
                required: min_required,
                actual: data.len(),
            });
        }

        // Initialize components
        match self.seasonal_type {
            SeasonalType::Additive => self.initialize_additive(data),
            SeasonalType::Multiplicative => self.initialize_multiplicative(data),
        }

        // Update through observations
        for (i, &value) in data.iter().enumerate().skip(self.period) {
            let season_idx = i % self.period;
            let prev_level = self.level;
            let prev_seasonal = self.seasonal[season_idx];

            match self.seasonal_type {
                SeasonalType::Additive => {
                    self.level = self.alpha * (value - prev_seasonal)
                        + (1.0 - self.alpha) * (self.level + self.trend);
                    self.trend =
                        self.beta * (self.level - prev_level) + (1.0 - self.beta) * self.trend;
                    self.seasonal[season_idx] = self.gamma * (value - self.level)
                        + (1.0 - self.gamma) * prev_seasonal;
                }
                SeasonalType::Multiplicative => {
                    let deseasonalized = if prev_seasonal.abs() > 1e-10 {
                        value / prev_seasonal
                    } else {
                        value
                    };
                    self.level = self.alpha * deseasonalized
                        + (1.0 - self.alpha) * (self.level + self.trend);
                    self.trend =
                        self.beta * (self.level - prev_level) + (1.0 - self.beta) * self.trend;
                    self.seasonal[season_idx] = if self.level.abs() > 1e-10 {
                        self.gamma * (value / self.level) + (1.0 - self.gamma) * prev_seasonal
                    } else {
                        prev_seasonal
                    };
                }
            }
        }

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, steps: usize) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(TsError::NotFitted);
        }

        let mut forecasts = Vec::with_capacity(steps);

        for h in 1..=steps {
            let season_idx = (h - 1) % self.period;
            let forecast = match self.seasonal_type {
                SeasonalType::Additive => {
                    self.level + h as f64 * self.trend + self.seasonal[season_idx]
                }
                SeasonalType::Multiplicative => {
                    (self.level + h as f64 * self.trend) * self.seasonal[season_idx]
                }
            };
            forecasts.push(forecast);
        }

        Ok(forecasts)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ses() {
        let data = vec![10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0];
        let mut model = SimpleExponentialSmoothing::new(0.3).unwrap();
        model.fit(&data).unwrap();
        let forecast = model.predict(3).unwrap();
        assert_eq!(forecast.len(), 3);
        // All forecasts should be the same (flat)
        assert!((forecast[0] - forecast[1]).abs() < 1e-10);
    }

    #[test]
    fn test_holt() {
        let data: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 2.0).collect();
        let mut model = DoubleExponentialSmoothing::new(0.3, 0.1).unwrap();
        model.fit(&data).unwrap();
        let forecast = model.predict(3).unwrap();
        assert_eq!(forecast.len(), 3);
        // Forecasts should increase (positive trend)
        assert!(forecast[1] > forecast[0]);
    }

    #[test]
    fn test_holt_winters() {
        let data: Vec<f64> = (0..48)
            .map(|i| 100.0 + (i as f64 * 2.0) + 20.0 * ((i as f64 * std::f64::consts::PI / 6.0).sin()))
            .collect();

        let mut model = HoltWinters::new(0.3, 0.1, 0.2, 12, SeasonalType::Additive).unwrap();
        model.fit(&data).unwrap();
        let forecast = model.predict(12).unwrap();
        assert_eq!(forecast.len(), 12);
    }
}
