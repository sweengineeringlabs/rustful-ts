//! Conditional Value at Risk (CVaR / Expected Shortfall) implementation.
//!
//! Measures the expected loss given that VaR is exceeded.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Conditional Value at Risk (CVaR) indicator.
///
/// Also known as Expected Shortfall (ES), CVaR measures the expected loss
/// given that the loss exceeds VaR. It addresses VaR's limitation of not
/// considering tail risk beyond the VaR threshold.
///
/// CVaR is always >= VaR for the same confidence level.
#[derive(Debug, Clone)]
pub struct ConditionalVaR {
    /// Rolling window period for calculation.
    period: usize,
    /// Confidence level (e.g., 0.95 for 95% CVaR).
    confidence_level: f64,
}

impl ConditionalVaR {
    /// Create a new CVaR indicator with default 95% confidence.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Self {
        Self {
            period,
            confidence_level: 0.95,
        }
    }

    /// Create a new CVaR indicator with custom confidence level.
    ///
    /// # Arguments
    /// * `period` - Rolling window period
    /// * `confidence_level` - Confidence level (e.g., 0.95, 0.99)
    pub fn with_confidence(period: usize, confidence_level: f64) -> Self {
        Self {
            period,
            confidence_level,
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

    /// Calculate percentile using linear interpolation.
    fn percentile(sorted_data: &[f64], percentile: f64) -> f64 {
        if sorted_data.is_empty() {
            return f64::NAN;
        }
        if sorted_data.len() == 1 {
            return sorted_data[0];
        }

        let n = sorted_data.len() as f64;
        let index = percentile * (n - 1.0);
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper || upper >= sorted_data.len() {
            sorted_data[lower]
        } else {
            let fraction = index - lower as f64;
            sorted_data[lower] * (1.0 - fraction) + sorted_data[upper] * fraction
        }
    }

    /// Calculate CVaR for a window of returns.
    fn calculate_cvar(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return f64::NAN;
        }

        let mut sorted: Vec<f64> = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // VaR threshold
        let var_percentile = 1.0 - self.confidence_level;
        let var_threshold = Self::percentile(&sorted, var_percentile);

        // CVaR is the average of all returns below VaR threshold
        let tail_returns: Vec<f64> = sorted
            .iter()
            .filter(|&&r| r <= var_threshold)
            .copied()
            .collect();

        if tail_returns.is_empty() {
            // If no returns exceed VaR, return VaR itself
            return -var_threshold.min(0.0);
        }

        let avg_tail_loss = tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;

        // Return as positive loss
        -avg_tail_loss.min(0.0)
    }

    /// Calculate CVaR values.
    pub fn calculate(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let returns = Self::calculate_returns(prices);
        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..returns.len() {
            let window = &returns[(i + 1 - self.period)..=i];
            let cvar = self.calculate_cvar(window);
            result.push(cvar);
        }

        result
    }
}

impl TechnicalIndicator for ConditionalVaR {
    fn name(&self) -> &str {
        "CVaR"
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
    use crate::var::ValueAtRisk;

    #[test]
    fn test_cvar_basic() {
        let cvar = ConditionalVaR::new(20);
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.5).sin() * 2.0)
            .collect();
        let result = cvar.calculate(&prices);

        // Should have valid values after warm-up period
        assert!(!result[30].is_nan());
        // CVaR should be positive (representing potential loss)
        assert!(result[30] >= 0.0);
    }

    #[test]
    fn test_cvar_greater_than_var() {
        let var = ValueAtRisk::with_confidence(20, 0.95);
        let cvar = ConditionalVaR::with_confidence(20, 0.95);

        // Prices with significant volatility
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.5).sin() * 5.0)
            .collect();

        let var_result = var.calculate(&prices);
        let cvar_result = cvar.calculate(&prices);

        // CVaR should be >= VaR
        for i in 25..prices.len() {
            if !var_result[i].is_nan() && !cvar_result[i].is_nan() {
                assert!(
                    cvar_result[i] >= var_result[i] - 0.001, // Small tolerance
                    "CVaR {} should be >= VaR {} at index {}",
                    cvar_result[i],
                    var_result[i],
                    i
                );
            }
        }
    }

    #[test]
    fn test_cvar_confidence_levels() {
        let cvar_95 = ConditionalVaR::with_confidence(20, 0.95);
        let cvar_99 = ConditionalVaR::with_confidence(20, 0.99);

        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.5).sin() * 3.0)
            .collect();

        let result_95 = cvar_95.calculate(&prices);
        let result_99 = cvar_99.calculate(&prices);

        // 99% CVaR should generally be >= 95% CVaR
        let idx = 30;
        if !result_95[idx].is_nan() && !result_99[idx].is_nan() {
            assert!(result_99[idx] >= result_95[idx] - 0.001);
        }
    }
}
