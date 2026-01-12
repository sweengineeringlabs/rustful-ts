//! Beta coefficient implementation.
//!
//! Measures an asset's volatility relative to the market.

use crate::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Beta coefficient indicator.
///
/// Measures the sensitivity of an asset's returns to market returns.
/// - Beta = 1: Asset moves with the market
/// - Beta > 1: Asset is more volatile than the market
/// - Beta < 1: Asset is less volatile than the market
/// - Beta < 0: Asset moves inversely to the market
///
/// Formula: Cov(Asset, Market) / Var(Market)
#[derive(Debug, Clone)]
pub struct Beta {
    /// Rolling window period for calculation.
    period: usize,
    /// Benchmark/market returns (must be set before calculation).
    benchmark_returns: Vec<f64>,
}

impl Beta {
    /// Create a new Beta indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Self {
        Self {
            period,
            benchmark_returns: Vec::new(),
        }
    }

    /// Set benchmark/market returns for beta calculation.
    ///
    /// # Arguments
    /// * `benchmark` - Market/benchmark price series
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

    /// Calculate variance of a slice.
    fn variance(data: &[f64], mean: f64) -> f64 {
        if data.len() < 2 {
            return f64::NAN;
        }
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
    }

    /// Calculate covariance between two series.
    fn covariance(x: &[f64], y: &[f64], mean_x: f64, mean_y: f64) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return f64::NAN;
        }
        x.iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>() / (x.len() - 1) as f64
    }

    /// Calculate Beta values.
    pub fn calculate(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let asset_returns = Self::calculate_returns(prices);

        if self.benchmark_returns.len() != asset_returns.len() {
            // If no benchmark set, return NaN
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..asset_returns.len() {
            let start = i + 1 - self.period;
            let asset_window = &asset_returns[start..=i];
            let bench_window = &self.benchmark_returns[start..=i];

            let mean_asset = Self::mean(asset_window);
            let mean_bench = Self::mean(bench_window);
            let var_bench = Self::variance(bench_window, mean_bench);
            let cov = Self::covariance(asset_window, bench_window, mean_asset, mean_bench);

            if var_bench == 0.0 || var_bench.is_nan() {
                result.push(f64::NAN);
            } else {
                let beta = cov / var_bench;
                result.push(beta);
            }
        }

        result
    }

    /// Calculate Beta given two price series.
    pub fn calculate_with_benchmark(&self, asset_prices: &[f64], benchmark_prices: &[f64]) -> Vec<f64> {
        let n = asset_prices.len();
        if n < self.period + 1 || benchmark_prices.len() != n {
            return vec![f64::NAN; n];
        }

        let asset_returns = Self::calculate_returns(asset_prices);
        let bench_returns = Self::calculate_returns(benchmark_prices);

        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..asset_returns.len() {
            let start = i + 1 - self.period;
            let asset_window = &asset_returns[start..=i];
            let bench_window = &bench_returns[start..=i];

            let mean_asset = Self::mean(asset_window);
            let mean_bench = Self::mean(bench_window);
            let var_bench = Self::variance(bench_window, mean_bench);
            let cov = Self::covariance(asset_window, bench_window, mean_asset, mean_bench);

            if var_bench == 0.0 || var_bench.is_nan() {
                result.push(f64::NAN);
            } else {
                let beta = cov / var_bench;
                result.push(beta);
            }
        }

        result
    }
}

impl TechnicalIndicator for Beta {
    fn name(&self) -> &str {
        "Beta"
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
                reason: "Benchmark returns must be set before computing Beta".to_string(),
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
    fn test_beta_with_same_series() {
        let asset_prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let beta = Beta::new(20);
        let result = beta.calculate_with_benchmark(&asset_prices, &asset_prices);

        // Beta of an asset with itself should be 1.0
        let last = result.last().unwrap();
        assert!((last - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_beta_double_volatility() {
        // Asset has twice the volatility of benchmark with variation
        let benchmark: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.5 + ((i as f64) * 0.3).sin() * 2.0)
            .collect();
        let asset: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 1.0 + ((i as f64) * 0.3).sin() * 4.0) // 2x the movement
            .collect();

        let beta = Beta::new(20);
        let result = beta.calculate_with_benchmark(&asset, &benchmark);

        // Beta should be positive and > 1 (higher volatility)
        let last = result.last().unwrap();
        assert!(!last.is_nan());
        assert!(*last > 1.0);
    }

    #[test]
    fn test_beta_inverse() {
        // Asset moves inversely to benchmark with variation
        let benchmark: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.5 + ((i as f64) * 0.3).sin() * 2.0)
            .collect();
        let asset: Vec<f64> = (0..50)
            .map(|i| 200.0 - (i as f64) * 0.5 - ((i as f64) * 0.3).sin() * 2.0)
            .collect();

        let beta = Beta::new(20);
        let result = beta.calculate_with_benchmark(&asset, &benchmark);

        // Beta should be negative (inverse relationship)
        let last = result.last().unwrap();
        assert!(!last.is_nan());
        assert!(*last < 0.0);
    }
}
