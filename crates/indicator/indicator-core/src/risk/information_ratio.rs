//! Information Ratio implementation.
//!
//! Measures the consistency of excess returns relative to a benchmark.

use crate::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Information Ratio indicator.
///
/// Measures the risk-adjusted excess return relative to a benchmark,
/// using tracking error as the risk measure.
///
/// Formula: (Portfolio Return - Benchmark Return) / Tracking Error
///
/// Where Tracking Error = Std Dev of (Portfolio Return - Benchmark Return)
///
/// Higher values indicate more consistent outperformance.
/// - IR > 0.5 is generally considered good
/// - IR > 1.0 is considered exceptional
#[derive(Debug, Clone)]
pub struct InformationRatio {
    /// Rolling window period for calculation.
    period: usize,
    /// Annualization factor (252 for daily, 52 for weekly, 12 for monthly).
    annualization_factor: f64,
    /// Benchmark/market returns (must be set before calculation).
    benchmark_returns: Vec<f64>,
}

impl InformationRatio {
    /// Create a new Information Ratio indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Self {
        Self {
            period,
            annualization_factor: 252.0,
            benchmark_returns: Vec::new(),
        }
    }

    /// Create with custom annualization factor.
    pub fn with_annualization(period: usize, annualization_factor: f64) -> Self {
        Self {
            period,
            annualization_factor,
            benchmark_returns: Vec::new(),
        }
    }

    /// Set benchmark/market returns.
    pub fn with_benchmark(mut self, benchmark: &[f64]) -> Self {
        self.benchmark_returns = Self::calculate_returns(benchmark);
        self
    }

    /// Set benchmark returns directly.
    pub fn with_benchmark_returns(mut self, returns: Vec<f64>) -> Self {
        self.benchmark_returns = returns;
        self
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

    /// Calculate active returns (portfolio - benchmark).
    fn calculate_active_returns(asset_returns: &[f64], bench_returns: &[f64]) -> Vec<f64> {
        asset_returns
            .iter()
            .zip(bench_returns.iter())
            .map(|(a, b)| a - b)
            .collect()
    }

    /// Calculate Information Ratio values.
    pub fn calculate(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let asset_returns = Self::calculate_returns(prices);

        if self.benchmark_returns.len() != asset_returns.len() {
            return vec![f64::NAN; n];
        }

        let active_returns = Self::calculate_active_returns(&asset_returns, &self.benchmark_returns);
        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..active_returns.len() {
            let start = i + 1 - self.period;
            let window = &active_returns[start..=i];

            let mean_active = Self::mean(window);
            let tracking_error = Self::std_dev(window, mean_active);

            if tracking_error == 0.0 || tracking_error.is_nan() {
                result.push(f64::NAN);
            } else {
                // Annualized Information Ratio
                let ir = (mean_active / tracking_error) * self.annualization_factor.sqrt();
                result.push(ir);
            }
        }

        result
    }

    /// Calculate Information Ratio given two price series.
    pub fn calculate_with_benchmark(&self, asset_prices: &[f64], benchmark_prices: &[f64]) -> Vec<f64> {
        let n = asset_prices.len();
        if n < self.period + 1 || benchmark_prices.len() != n {
            return vec![f64::NAN; n];
        }

        let asset_returns = Self::calculate_returns(asset_prices);
        let bench_returns = Self::calculate_returns(benchmark_prices);
        let active_returns = Self::calculate_active_returns(&asset_returns, &bench_returns);

        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..active_returns.len() {
            let start = i + 1 - self.period;
            let window = &active_returns[start..=i];

            let mean_active = Self::mean(window);
            let tracking_error = Self::std_dev(window, mean_active);

            if tracking_error == 0.0 || tracking_error.is_nan() {
                result.push(f64::NAN);
            } else {
                let ir = (mean_active / tracking_error) * self.annualization_factor.sqrt();
                result.push(ir);
            }
        }

        result
    }
}

impl TechnicalIndicator for InformationRatio {
    fn name(&self) -> &str {
        "InformationRatio"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        if self.benchmark_returns.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "benchmark".to_string(),
                reason: "Benchmark returns must be set before computing Information Ratio".to_string(),
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
    fn test_ir_basic() {
        let benchmark: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.3)
            .collect();
        let asset: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.5)
            .collect();

        let ir = InformationRatio::new(20);
        let result = ir.calculate_with_benchmark(&asset, &benchmark);

        // Should have valid values after warm-up
        assert!(!result[30].is_nan());
        // Consistently outperforming should give positive IR
        assert!(result[30] > 0.0);
    }

    #[test]
    fn test_ir_underperformance() {
        let benchmark: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.5)
            .collect();
        let asset: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.2)
            .collect();

        let ir = InformationRatio::new(20);
        let result = ir.calculate_with_benchmark(&asset, &benchmark);

        // Underperformance should give negative IR
        assert!(result[30] < 0.0);
    }

    #[test]
    fn test_ir_same_series() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let ir = InformationRatio::new(20);
        let result = ir.calculate_with_benchmark(&prices, &prices);

        // Same series means zero active returns, so IR should be NaN (no tracking error)
        // or zero depending on implementation
        let last = result.last().unwrap();
        assert!(last.is_nan() || last.abs() < 0.001);
    }
}
