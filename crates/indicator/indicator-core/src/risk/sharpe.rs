//! Sharpe Ratio implementation.
//!
//! Measures risk-adjusted returns relative to the risk-free rate.

use crate::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Sharpe Ratio indicator.
///
/// Calculates the risk-adjusted return by measuring excess return per unit of volatility.
/// Formula: (Portfolio Return - Risk-Free Rate) / Portfolio Std Dev
///
/// Higher values indicate better risk-adjusted performance.
#[derive(Debug, Clone)]
pub struct SharpeRatio {
    /// Rolling window period for calculation.
    period: usize,
    /// Annualized risk-free rate (e.g., 0.02 for 2%).
    risk_free_rate: f64,
    /// Annualization factor (252 for daily, 52 for weekly, 12 for monthly).
    annualization_factor: f64,
}

impl SharpeRatio {
    /// Create a new Sharpe Ratio indicator with default parameters.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Self {
        Self {
            period,
            risk_free_rate: 0.0,
            annualization_factor: 252.0,
        }
    }

    /// Create a new Sharpe Ratio with custom risk-free rate.
    ///
    /// # Arguments
    /// * `period` - Rolling window period
    /// * `risk_free_rate` - Annualized risk-free rate (e.g., 0.02 for 2%)
    pub fn with_risk_free_rate(period: usize, risk_free_rate: f64) -> Self {
        Self {
            period,
            risk_free_rate,
            annualization_factor: 252.0,
        }
    }

    /// Create a new Sharpe Ratio with full configuration.
    ///
    /// # Arguments
    /// * `period` - Rolling window period
    /// * `risk_free_rate` - Annualized risk-free rate
    /// * `annualization_factor` - Factor for annualizing (252 daily, 52 weekly, 12 monthly)
    pub fn with_config(period: usize, risk_free_rate: f64, annualization_factor: f64) -> Self {
        Self {
            period,
            risk_free_rate,
            annualization_factor,
        }
    }

    /// Calculate returns from price series.
    fn calculate_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }
        prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate mean of a slice.
    fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return f64::NAN;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Calculate standard deviation of a slice.
    fn std_dev(data: &[f64], mean: f64) -> f64 {
        if data.len() < 2 {
            return f64::NAN;
        }
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate Sharpe Ratio values.
    pub fn calculate(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let returns = Self::calculate_returns(prices);
        let mut result = vec![f64::NAN; self.period];

        // Daily risk-free rate
        let daily_rf = self.risk_free_rate / self.annualization_factor;

        for i in (self.period - 1)..returns.len() {
            let window = &returns[(i + 1 - self.period)..=i];
            let mean_return = Self::mean(window);
            let std_return = Self::std_dev(window, mean_return);

            if std_return == 0.0 || std_return.is_nan() {
                result.push(f64::NAN);
            } else {
                // Annualized Sharpe Ratio
                let excess_return = mean_return - daily_rf;
                let sharpe = (excess_return / std_return) * self.annualization_factor.sqrt();
                result.push(sharpe);
            }
        }

        result
    }
}

impl TechnicalIndicator for SharpeRatio {
    fn name(&self) -> &str {
        "SharpeRatio"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_basic() {
        let sharpe = SharpeRatio::new(20);
        // Generate increasing prices (positive trend)
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result = sharpe.calculate(&prices);

        // Should have valid values after warm-up period
        assert!(!result[30].is_nan());
        // Positive trend should give positive Sharpe
        assert!(result[30] > 0.0);
    }

    #[test]
    fn test_sharpe_with_risk_free() {
        let sharpe = SharpeRatio::with_risk_free_rate(20, 0.02);
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result = sharpe.calculate(&prices);

        // Should have valid values
        assert!(!result[30].is_nan());
    }

    #[test]
    fn test_sharpe_insufficient_data() {
        let sharpe = SharpeRatio::new(20);
        let prices: Vec<f64> = vec![100.0, 101.0, 102.0];
        let result = sharpe.calculate(&prices);

        // All values should be NaN
        assert!(result.iter().all(|v| v.is_nan()));
    }
}
