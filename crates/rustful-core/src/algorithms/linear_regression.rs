//! Linear Regression for time series forecasting
//!
//! Uses ordinary least squares (OLS) to fit a linear trend to time series data.
//! Can be extended with seasonal dummy variables for seasonal decomposition.
//!
//! ## When to Use
//!
//! - Data shows a clear linear trend
//! - Quick baseline model
//! - When interpretability is important

use crate::algorithms::Predictor;
use crate::error::{Result, TsError};
use serde::{Deserialize, Serialize};

/// Linear Regression model for time series
///
/// Fits y = intercept + slope * t where t is the time index.
///
/// # Example
///
/// ```rust
/// use rustful_ts::algorithms::linear_regression::LinearRegression;
/// use rustful_ts::algorithms::Predictor;
///
/// let data = vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
/// let mut model = LinearRegression::new();
/// model.fit(&data).unwrap();
///
/// let forecast = model.predict(3).unwrap();
/// // Should predict approximately [22, 24, 26]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    /// Y-intercept
    intercept: f64,
    /// Slope (trend per time unit)
    slope: f64,
    /// Number of observations used in fitting
    n_observations: usize,
    /// R-squared value
    r_squared: f64,
    /// Whether model has been fitted
    fitted: bool,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegression {
    /// Create a new linear regression model
    pub fn new() -> Self {
        Self {
            intercept: 0.0,
            slope: 0.0,
            n_observations: 0,
            r_squared: 0.0,
            fitted: false,
        }
    }

    /// Get the slope (trend per time unit)
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// Get the intercept
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Get R-squared (coefficient of determination)
    pub fn r_squared(&self) -> f64 {
        self.r_squared
    }

    /// Predict value at a specific time index
    pub fn predict_at(&self, t: f64) -> Result<f64> {
        if !self.fitted {
            return Err(TsError::NotFitted);
        }
        Ok(self.intercept + self.slope * t)
    }

    /// Get residuals from the fit
    pub fn residuals(&self, data: &[f64]) -> Vec<f64> {
        if !self.fitted {
            return Vec::new();
        }

        data.iter()
            .enumerate()
            .map(|(i, &y)| y - (self.intercept + self.slope * i as f64))
            .collect()
    }
}

impl Predictor for LinearRegression {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        if data.len() < 2 {
            return Err(TsError::InsufficientData {
                required: 2,
                actual: data.len(),
            });
        }

        let n = data.len() as f64;
        self.n_observations = data.len();

        // Time indices: 0, 1, 2, ...
        let sum_t: f64 = (0..data.len()).map(|i| i as f64).sum();
        let sum_y: f64 = data.iter().sum();
        let sum_t2: f64 = (0..data.len()).map(|i| (i * i) as f64).sum();
        let sum_ty: f64 = data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();

        // OLS formulas
        let denominator = n * sum_t2 - sum_t * sum_t;
        if denominator.abs() < 1e-10 {
            return Err(TsError::NumericalError(
                "Singular matrix in regression".to_string(),
            ));
        }

        self.slope = (n * sum_ty - sum_t * sum_y) / denominator;
        self.intercept = (sum_y - self.slope * sum_t) / n;

        // Calculate R-squared
        let mean_y = sum_y / n;
        let ss_tot: f64 = data.iter().map(|&y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = data
            .iter()
            .enumerate()
            .map(|(i, &y)| {
                let predicted = self.intercept + self.slope * i as f64;
                (y - predicted).powi(2)
            })
            .sum();

        self.r_squared = if ss_tot > 1e-10 {
            1.0 - ss_res / ss_tot
        } else {
            1.0
        };

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, steps: usize) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(TsError::NotFitted);
        }

        let mut forecasts = Vec::with_capacity(steps);
        for i in 0..steps {
            let t = (self.n_observations + i) as f64;
            forecasts.push(self.intercept + self.slope * t);
        }

        Ok(forecasts)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// Linear Regression with seasonal dummy variables
///
/// Extends basic linear regression to capture seasonal patterns.
///
/// # Example
///
/// ```rust
/// use rustful_ts::algorithms::linear_regression::SeasonalLinearRegression;
/// use rustful_ts::algorithms::Predictor;
///
/// // Monthly data with yearly seasonality
/// let data: Vec<f64> = (0..36).map(|i| {
///     10.0 + (i as f64 * 0.5) + 5.0 * ((i % 12) as f64 / 12.0 * std::f64::consts::PI * 2.0).sin()
/// }).collect();
///
/// let mut model = SeasonalLinearRegression::new(12).unwrap();
/// model.fit(&data).unwrap();
/// let forecast = model.predict(12).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalLinearRegression {
    /// Seasonal period
    period: usize,
    /// Base trend model
    trend: LinearRegression,
    /// Seasonal factors (additive)
    seasonal_factors: Vec<f64>,
    /// Whether model has been fitted
    fitted: bool,
}

impl SeasonalLinearRegression {
    /// Create a new seasonal linear regression model
    ///
    /// # Arguments
    ///
    /// * `period` - Number of observations per seasonal cycle
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(TsError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }

        Ok(Self {
            period,
            trend: LinearRegression::new(),
            seasonal_factors: vec![0.0; period],
            fitted: false,
        })
    }

    /// Get seasonal factors
    pub fn seasonal_factors(&self) -> &[f64] {
        &self.seasonal_factors
    }

    /// Get the underlying trend model
    pub fn trend(&self) -> &LinearRegression {
        &self.trend
    }
}

impl Predictor for SeasonalLinearRegression {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        let min_required = self.period * 2;
        if data.len() < min_required {
            return Err(TsError::InsufficientData {
                required: min_required,
                actual: data.len(),
            });
        }

        // Step 1: Fit trend
        self.trend.fit(data)?;

        // Step 2: Detrend
        let detrended: Vec<f64> = data
            .iter()
            .enumerate()
            .map(|(i, &y)| y - (self.trend.intercept + self.trend.slope * i as f64))
            .collect();

        // Step 3: Compute seasonal factors as mean of detrended values for each season
        for s in 0..self.period {
            let values: Vec<f64> = detrended
                .iter()
                .enumerate()
                .filter(|(i, _)| i % self.period == s)
                .map(|(_, &v)| v)
                .collect();

            self.seasonal_factors[s] = if !values.is_empty() {
                values.iter().sum::<f64>() / values.len() as f64
            } else {
                0.0
            };
        }

        // Normalize seasonal factors to sum to 0
        let mean_factor: f64 = self.seasonal_factors.iter().sum::<f64>() / self.period as f64;
        for factor in &mut self.seasonal_factors {
            *factor -= mean_factor;
        }

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, steps: usize) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(TsError::NotFitted);
        }

        let trend_forecasts = self.trend.predict(steps)?;

        let forecasts: Vec<f64> = trend_forecasts
            .iter()
            .enumerate()
            .map(|(i, &trend_val)| {
                let season_idx = (self.trend.n_observations + i) % self.period;
                trend_val + self.seasonal_factors[season_idx]
            })
            .collect();

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
    fn test_linear_regression() {
        let data: Vec<f64> = (0..10).map(|i| 10.0 + 2.0 * i as f64).collect();
        let mut model = LinearRegression::new();
        model.fit(&data).unwrap();

        assert!((model.slope() - 2.0).abs() < 1e-10);
        assert!((model.intercept() - 10.0).abs() < 1e-10);
        assert!(model.r_squared() > 0.99);

        let forecast = model.predict(3).unwrap();
        assert!((forecast[0] - 30.0).abs() < 1e-10);
        assert!((forecast[1] - 32.0).abs() < 1e-10);
        assert!((forecast[2] - 34.0).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_linear_regression() {
        let data: Vec<f64> = (0..24)
            .map(|i| 10.0 + 0.5 * i as f64 + [2.0, -1.0, 0.0, -1.0][i % 4])
            .collect();

        let mut model = SeasonalLinearRegression::new(4).unwrap();
        model.fit(&data).unwrap();

        let forecast = model.predict(4).unwrap();
        assert_eq!(forecast.len(), 4);
    }
}
