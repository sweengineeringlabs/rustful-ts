//! Value at Risk (VaR) implementation.
//!
//! Measures the potential loss at a given confidence level.

use crate::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Value at Risk (VaR) calculation method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VaRMethod {
    /// Historical simulation (percentile-based).
    Historical,
    /// Parametric (assumes normal distribution).
    Parametric,
}

/// Value at Risk indicator.
///
/// VaR estimates the maximum expected loss over a specific time period
/// at a given confidence level under normal market conditions.
///
/// For example, 95% VaR of 0.02 means there's a 5% chance of losing
/// more than 2% of portfolio value.
#[derive(Debug, Clone)]
pub struct ValueAtRisk {
    /// Rolling window period for calculation.
    period: usize,
    /// Confidence level (e.g., 0.95 for 95% VaR).
    confidence_level: f64,
    /// VaR calculation method.
    method: VaRMethod,
}

impl ValueAtRisk {
    /// Create a new VaR indicator with default 95% confidence.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Self {
        Self {
            period,
            confidence_level: 0.95,
            method: VaRMethod::Historical,
        }
    }

    /// Create a new VaR indicator with custom confidence level.
    ///
    /// # Arguments
    /// * `period` - Rolling window period
    /// * `confidence_level` - Confidence level (e.g., 0.95, 0.99)
    pub fn with_confidence(period: usize, confidence_level: f64) -> Self {
        Self {
            period,
            confidence_level,
            method: VaRMethod::Historical,
        }
    }

    /// Create a new VaR indicator with full configuration.
    pub fn with_config(period: usize, confidence_level: f64, method: VaRMethod) -> Self {
        Self {
            period,
            confidence_level,
            method,
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

    /// Inverse standard normal CDF approximation (for parametric VaR).
    fn inv_norm(p: f64) -> f64 {
        // Approximation using rational function
        // More accurate methods exist but this is sufficient for VaR
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }

        // Coefficients for approximation
        let a = [
            -3.969683028665376e1,
            2.209460984245205e2,
            -2.759285104469687e2,
            1.383577518672690e2,
            -3.066479806614716e1,
            2.506628277459239e0,
        ];
        let b = [
            -5.447609879822406e1,
            1.615858368580409e2,
            -1.556989798598866e2,
            6.680131188771972e1,
            -1.328068155288572e1,
        ];
        let c = [
            -7.784894002430293e-3,
            -3.223964580411365e-1,
            -2.400758277161838e0,
            -2.549732539343734e0,
            4.374664141464968e0,
            2.938163982698783e0,
        ];
        let d = [
            7.784695709041462e-3,
            3.224671290700398e-1,
            2.445134137142996e0,
            3.754408661907416e0,
        ];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        let result = if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        };

        result
    }

    /// Calculate historical VaR (percentile method).
    fn historical_var(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return f64::NAN;
        }

        let mut sorted: Vec<f64> = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // VaR is the loss at the (1 - confidence) percentile
        let var_percentile = 1.0 - self.confidence_level;
        let var = Self::percentile(&sorted, var_percentile);

        // Return as positive loss (negate if return is negative)
        -var.min(0.0)
    }

    /// Calculate parametric VaR (assumes normal distribution).
    fn parametric_var(&self, returns: &[f64]) -> f64 {
        let mean = Self::mean(returns);
        let std = Self::std_dev(returns, mean);

        if std.is_nan() || std == 0.0 {
            return f64::NAN;
        }

        // z-score for the confidence level
        let z = Self::inv_norm(1.0 - self.confidence_level);

        // VaR = -(mean + z * std)
        // Return as positive loss
        -(mean + z * std)
    }

    /// Calculate VaR values.
    pub fn calculate(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let returns = Self::calculate_returns(prices);
        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..returns.len() {
            let window = &returns[(i + 1 - self.period)..=i];

            let var = match self.method {
                VaRMethod::Historical => self.historical_var(window),
                VaRMethod::Parametric => self.parametric_var(window),
            };

            result.push(var);
        }

        result
    }
}

impl TechnicalIndicator for ValueAtRisk {
    fn name(&self) -> &str {
        "VaR"
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
    fn test_var_basic() {
        let var = ValueAtRisk::new(20);
        // Generate prices with some volatility
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.5).sin() * 2.0)
            .collect();
        let result = var.calculate(&prices);

        // Should have valid values after warm-up period
        assert!(!result[30].is_nan());
        // VaR should be positive (representing potential loss)
        assert!(result[30] >= 0.0);
    }

    #[test]
    fn test_var_confidence_levels() {
        let var_95 = ValueAtRisk::with_confidence(20, 0.95);
        let var_99 = ValueAtRisk::with_confidence(20, 0.99);

        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.5).sin() * 2.0)
            .collect();

        let result_95 = var_95.calculate(&prices);
        let result_99 = var_99.calculate(&prices);

        // 99% VaR should generally be >= 95% VaR
        let idx = 30;
        assert!(result_99[idx] >= result_95[idx] - 0.001); // Small tolerance for numerical issues
    }

    #[test]
    fn test_var_parametric() {
        let var = ValueAtRisk::with_config(20, 0.95, VaRMethod::Parametric);
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.5).sin() * 2.0)
            .collect();
        let result = var.calculate(&prices);

        assert!(!result[30].is_nan());
    }
}
