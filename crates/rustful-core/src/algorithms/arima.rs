//! ARIMA (AutoRegressive Integrated Moving Average) implementation
//!
//! ARIMA models are one of the most widely used approaches for time series forecasting.
//! The model combines three components:
//!
//! - **AR (AutoRegressive)**: Uses past values to predict future values
//! - **I (Integrated)**: Differencing to achieve stationarity
//! - **MA (Moving Average)**: Uses past forecast errors
//!
//! ## Parameters
//!
//! - `p`: Order of the autoregressive part
//! - `d`: Degree of differencing
//! - `q`: Order of the moving average part
//!
//! ## Example
//!
//! ```rust
//! use rustful_ts::algorithms::{arima::Arima, Predictor};
//!
//! let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
//! let mut model = Arima::new(1, 1, 0).unwrap();
//! model.fit(&data).unwrap();
//! let forecast = model.predict(3).unwrap();
//! assert_eq!(forecast.len(), 3);
//! ```

use crate::error::{Result, TsError};
use crate::algorithms::Predictor;
use serde::{Deserialize, Serialize};

/// ARIMA model for time series forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arima {
    /// AR order (p)
    p: usize,
    /// Differencing order (d)
    d: usize,
    /// MA order (q)
    q: usize,
    /// AR coefficients
    ar_coeffs: Vec<f64>,
    /// MA coefficients
    ma_coeffs: Vec<f64>,
    /// Constant term
    constant: f64,
    /// Original data (for undifferencing)
    original_data: Vec<f64>,
    /// Differenced data
    differenced_data: Vec<f64>,
    /// Residuals from fitting
    residuals: Vec<f64>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl Arima {
    /// Create a new ARIMA model with specified orders
    ///
    /// # Arguments
    ///
    /// * `p` - Order of autoregressive component (0-10)
    /// * `d` - Degree of differencing (0-2)
    /// * `q` - Order of moving average component (0-10)
    ///
    /// # Returns
    ///
    /// A new unfitted ARIMA model
    pub fn new(p: usize, d: usize, q: usize) -> Result<Self> {
        if p > 10 {
            return Err(TsError::InvalidParameter {
                name: "p".to_string(),
                reason: "AR order must be <= 10".to_string(),
            });
        }
        if d > 2 {
            return Err(TsError::InvalidParameter {
                name: "d".to_string(),
                reason: "Differencing order must be <= 2".to_string(),
            });
        }
        if q > 10 {
            return Err(TsError::InvalidParameter {
                name: "q".to_string(),
                reason: "MA order must be <= 10".to_string(),
            });
        }

        Ok(Self {
            p,
            d,
            q,
            ar_coeffs: vec![0.0; p],
            ma_coeffs: vec![0.0; q],
            constant: 0.0,
            original_data: Vec::new(),
            differenced_data: Vec::new(),
            residuals: Vec::new(),
            fitted: false,
        })
    }

    /// Apply differencing to make series stationary
    fn difference(data: &[f64], order: usize) -> Vec<f64> {
        let mut result = data.to_vec();
        for _ in 0..order {
            let mut differenced = Vec::with_capacity(result.len().saturating_sub(1));
            for i in 1..result.len() {
                differenced.push(result[i] - result[i - 1]);
            }
            result = differenced;
        }
        result
    }

    /// Reverse differencing to get original scale
    fn undifference(&self, forecasts: &[f64]) -> Vec<f64> {
        if self.d == 0 {
            return forecasts.to_vec();
        }

        let mut result = forecasts.to_vec();
        let n = self.original_data.len();

        for _ in 0..self.d {
            let last_value = self.original_data[n - 1];
            let mut cumsum = vec![last_value + result[0]];
            for i in 1..result.len() {
                cumsum.push(cumsum[i - 1] + result[i]);
            }
            result = cumsum;
        }

        result
    }

    /// Estimate AR coefficients using Yule-Walker equations
    fn estimate_ar_coefficients(&self, data: &[f64]) -> Vec<f64> {
        if self.p == 0 {
            return Vec::new();
        }

        let n = data.len();
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();

        // Compute autocorrelations
        let mut autocorr = vec![0.0; self.p + 1];
        for k in 0..=self.p {
            let mut sum = 0.0;
            for i in k..n {
                sum += centered[i] * centered[i - k];
            }
            autocorr[k] = sum / n as f64;
        }

        // Solve Yule-Walker using Levinson-Durbin
        let mut coeffs = vec![0.0; self.p];
        if autocorr[0].abs() > 1e-10 {
            coeffs[0] = autocorr[1] / autocorr[0];

            for k in 1..self.p {
                let mut sum = autocorr[k + 1];
                for j in 0..k {
                    sum -= coeffs[j] * autocorr[k - j];
                }

                let mut denom = autocorr[0];
                for j in 0..k {
                    denom -= coeffs[j] * autocorr[j + 1];
                }

                if denom.abs() > 1e-10 {
                    let new_coeff = sum / denom;
                    let old_coeffs = coeffs.clone();
                    coeffs[k] = new_coeff;
                    for j in 0..k {
                        coeffs[j] = old_coeffs[j] - new_coeff * old_coeffs[k - 1 - j];
                    }
                }
            }
        }

        coeffs
    }

    /// Estimate MA coefficients from residuals
    fn estimate_ma_coefficients(&self, residuals: &[f64]) -> Vec<f64> {
        if self.q == 0 || residuals.is_empty() {
            return vec![0.0; self.q];
        }

        // Simple estimation using autocorrelation of residuals
        let n = residuals.len();
        let mean: f64 = residuals.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = residuals.iter().map(|x| x - mean).collect();

        let mut coeffs = vec![0.0; self.q];
        let var: f64 = centered.iter().map(|x| x * x).sum::<f64>() / n as f64;

        if var.abs() > 1e-10 {
            for k in 0..self.q {
                let mut sum = 0.0;
                for i in (k + 1)..n {
                    sum += centered[i] * centered[i - k - 1];
                }
                coeffs[k] = (sum / n as f64) / var;
                // Bound coefficients for stability
                coeffs[k] = coeffs[k].clamp(-0.99, 0.99);
            }
        }

        coeffs
    }

    /// Get model parameters
    pub fn params(&self) -> (usize, usize, usize) {
        (self.p, self.d, self.q)
    }

    /// Get AR coefficients
    pub fn ar_coefficients(&self) -> &[f64] {
        &self.ar_coeffs
    }

    /// Get MA coefficients
    pub fn ma_coefficients(&self) -> &[f64] {
        &self.ma_coeffs
    }
}

impl Predictor for Arima {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        let min_required = self.p + self.d + self.q + 10;
        if data.len() < min_required {
            return Err(TsError::InsufficientData {
                required: min_required,
                actual: data.len(),
            });
        }

        // Check for invalid values
        if data.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(TsError::InvalidData(
                "Data contains NaN or infinite values".to_string(),
            ));
        }

        self.original_data = data.to_vec();
        self.differenced_data = Self::difference(data, self.d);

        // Estimate AR coefficients
        self.ar_coeffs = self.estimate_ar_coefficients(&self.differenced_data);

        // Compute residuals
        let n = self.differenced_data.len();
        self.residuals = vec![0.0; n];
        let mean: f64 = self.differenced_data.iter().sum::<f64>() / n as f64;
        self.constant = mean;

        for i in self.p..n {
            let mut prediction = self.constant;
            for j in 0..self.p {
                prediction += self.ar_coeffs[j] * (self.differenced_data[i - j - 1] - mean);
            }
            self.residuals[i] = self.differenced_data[i] - prediction;
        }

        // Estimate MA coefficients
        self.ma_coeffs = self.estimate_ma_coefficients(&self.residuals);

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, steps: usize) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(TsError::NotFitted);
        }

        if steps == 0 {
            return Ok(Vec::new());
        }

        let n = self.differenced_data.len();
        let mut extended = self.differenced_data.clone();
        let mut extended_residuals = self.residuals.clone();

        // Generate forecasts on differenced scale
        for _ in 0..steps {
            let mut forecast = self.constant;

            // AR component
            for j in 0..self.p {
                let idx = extended.len() - j - 1;
                forecast += self.ar_coeffs[j] * (extended[idx] - self.constant);
            }

            // MA component
            for j in 0..self.q {
                if extended_residuals.len() > j {
                    let idx = extended_residuals.len() - j - 1;
                    forecast += self.ma_coeffs[j] * extended_residuals[idx];
                }
            }

            extended.push(forecast);
            extended_residuals.push(0.0); // Future residuals are 0
        }

        // Extract forecasts and undifference
        let forecasts: Vec<f64> = extended[n..].to_vec();
        Ok(self.undifference(&forecasts))
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arima_creation() {
        let model = Arima::new(1, 1, 1);
        assert!(model.is_ok());

        let model = Arima::new(11, 0, 0);
        assert!(model.is_err());
    }

    #[test]
    fn test_arima_fit_predict() {
        let data: Vec<f64> = (1..=50).map(|x| x as f64 + (x as f64 * 0.1).sin()).collect();
        let mut model = Arima::new(1, 1, 0).unwrap();

        assert!(model.fit(&data).is_ok());
        assert!(model.is_fitted());

        let forecast = model.predict(5);
        assert!(forecast.is_ok());
        assert_eq!(forecast.unwrap().len(), 5);
    }
}
