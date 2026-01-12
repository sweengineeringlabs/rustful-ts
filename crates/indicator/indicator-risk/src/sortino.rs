//! Sortino Ratio implementation.
//!
//! Measures risk-adjusted returns using downside deviation instead of total volatility.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Sortino Ratio indicator.
///
/// Similar to Sharpe Ratio but only penalizes downside volatility.
/// Formula: (Portfolio Return - Target Return) / Downside Deviation
///
/// Higher values indicate better risk-adjusted performance with focus on downside risk.
#[derive(Debug, Clone)]
pub struct SortinoRatio {
    /// Rolling window period for calculation.
    period: usize,
    /// Target return (Minimum Acceptable Return), often risk-free rate.
    target_return: f64,
    /// Annualization factor (252 for daily, 52 for weekly, 12 for monthly).
    annualization_factor: f64,
}

impl SortinoRatio {
    /// Create a new Sortino Ratio indicator with default parameters.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Self {
        Self {
            period,
            target_return: 0.0,
            annualization_factor: 252.0,
        }
    }

    /// Create a new Sortino Ratio with custom target return.
    ///
    /// # Arguments
    /// * `period` - Rolling window period
    /// * `target_return` - Minimum acceptable return (annualized)
    pub fn with_target_return(period: usize, target_return: f64) -> Self {
        Self {
            period,
            target_return,
            annualization_factor: 252.0,
        }
    }

    /// Create a new Sortino Ratio with full configuration.
    pub fn with_config(period: usize, target_return: f64, annualization_factor: f64) -> Self {
        Self {
            period,
            target_return,
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

    /// Calculate downside deviation (semi-deviation below target).
    fn downside_deviation(data: &[f64], target: f64) -> f64 {
        if data.len() < 2 {
            return f64::NAN;
        }

        let downside_variance: f64 = data
            .iter()
            .map(|&r| {
                let diff = r - target;
                if diff < 0.0 { diff.powi(2) } else { 0.0 }
            })
            .sum::<f64>() / (data.len() - 1) as f64;

        downside_variance.sqrt()
    }

    /// Calculate Sortino Ratio values.
    pub fn calculate(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let returns = Self::calculate_returns(prices);
        let mut result = vec![f64::NAN; self.period];

        // Daily target return
        let daily_target = self.target_return / self.annualization_factor;

        for i in (self.period - 1)..returns.len() {
            let window = &returns[(i + 1 - self.period)..=i];
            let mean_return = Self::mean(window);
            let downside_dev = Self::downside_deviation(window, daily_target);

            if downside_dev == 0.0 || downside_dev.is_nan() {
                // No downside deviation means excellent performance or insufficient data
                if mean_return > daily_target {
                    result.push(f64::INFINITY);
                } else {
                    result.push(f64::NAN);
                }
            } else {
                // Annualized Sortino Ratio
                let excess_return = mean_return - daily_target;
                let sortino = (excess_return / downside_dev) * self.annualization_factor.sqrt();
                result.push(sortino);
            }
        }

        result
    }
}

impl TechnicalIndicator for SortinoRatio {
    fn name(&self) -> &str {
        "SortinoRatio"
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
    fn test_sortino_basic() {
        let sortino = SortinoRatio::new(20);
        // Generate increasing prices (positive trend)
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result = sortino.calculate(&prices);

        // Should have valid values after warm-up period
        assert!(!result[30].is_nan());
    }

    #[test]
    fn test_sortino_with_target() {
        let sortino = SortinoRatio::with_target_return(20, 0.02);
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let result = sortino.calculate(&prices);

        // Should have valid values
        assert!(!result[30].is_nan());
    }

    #[test]
    fn test_sortino_no_downside() {
        let sortino = SortinoRatio::new(10);
        // Strictly increasing prices - no downside
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64)).collect();
        let result = sortino.calculate(&prices);

        // With no downside, Sortino should be infinity
        assert!(result[15].is_infinite() || result[15] > 0.0);
    }
}
