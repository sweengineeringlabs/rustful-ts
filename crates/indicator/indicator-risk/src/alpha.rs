//! Alpha (Jensen's Alpha) implementation.
//!
//! Measures the excess return of an investment relative to its expected return based on Beta.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Alpha (Jensen's Alpha) indicator.
///
/// Measures the risk-adjusted excess return of an investment.
/// Alpha represents the portion of returns not explained by market movements.
///
/// Formula: Asset Return - (Risk-Free Rate + Beta * (Market Return - Risk-Free Rate))
///
/// - Positive Alpha: Outperforming the market on a risk-adjusted basis
/// - Negative Alpha: Underperforming the market
/// - Zero Alpha: Returns explained entirely by market exposure
#[derive(Debug, Clone)]
pub struct Alpha {
    /// Rolling window period for calculation.
    period: usize,
    /// Annualized risk-free rate.
    risk_free_rate: f64,
    /// Annualization factor (252 for daily, 52 for weekly, 12 for monthly).
    annualization_factor: f64,
    /// Benchmark/market returns (must be set before calculation).
    benchmark_returns: Vec<f64>,
}

impl Alpha {
    /// Create a new Alpha indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Self {
        Self {
            period,
            risk_free_rate: 0.0,
            annualization_factor: 252.0,
            benchmark_returns: Vec::new(),
        }
    }

    /// Create a new Alpha indicator with risk-free rate.
    pub fn with_risk_free_rate(period: usize, risk_free_rate: f64) -> Self {
        Self {
            period,
            risk_free_rate,
            annualization_factor: 252.0,
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

    /// Calculate Beta coefficient.
    fn calculate_beta(
        asset_returns: &[f64],
        bench_returns: &[f64],
        mean_asset: f64,
        mean_bench: f64,
    ) -> f64 {
        let var_bench = Self::variance(bench_returns, mean_bench);
        let cov = Self::covariance(asset_returns, bench_returns, mean_asset, mean_bench);

        if var_bench == 0.0 || var_bench.is_nan() {
            f64::NAN
        } else {
            cov / var_bench
        }
    }

    /// Calculate Alpha values.
    pub fn calculate(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let asset_returns = Self::calculate_returns(prices);

        if self.benchmark_returns.len() != asset_returns.len() {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period];
        let daily_rf = self.risk_free_rate / self.annualization_factor;

        for i in (self.period - 1)..asset_returns.len() {
            let start = i + 1 - self.period;
            let asset_window = &asset_returns[start..=i];
            let bench_window = &self.benchmark_returns[start..=i];

            let mean_asset = Self::mean(asset_window);
            let mean_bench = Self::mean(bench_window);
            let beta = Self::calculate_beta(asset_window, bench_window, mean_asset, mean_bench);

            if beta.is_nan() {
                result.push(f64::NAN);
            } else {
                // Jensen's Alpha: R_a - (R_f + Beta * (R_m - R_f))
                let expected_return = daily_rf + beta * (mean_bench - daily_rf);
                let alpha = mean_asset - expected_return;

                // Annualize alpha
                let annualized_alpha = alpha * self.annualization_factor;
                result.push(annualized_alpha);
            }
        }

        result
    }

    /// Calculate Alpha given two price series.
    pub fn calculate_with_benchmark(&self, asset_prices: &[f64], benchmark_prices: &[f64]) -> Vec<f64> {
        let n = asset_prices.len();
        if n < self.period + 1 || benchmark_prices.len() != n {
            return vec![f64::NAN; n];
        }

        let asset_returns = Self::calculate_returns(asset_prices);
        let bench_returns = Self::calculate_returns(benchmark_prices);

        let mut result = vec![f64::NAN; self.period];
        let daily_rf = self.risk_free_rate / self.annualization_factor;

        for i in (self.period - 1)..asset_returns.len() {
            let start = i + 1 - self.period;
            let asset_window = &asset_returns[start..=i];
            let bench_window = &bench_returns[start..=i];

            let mean_asset = Self::mean(asset_window);
            let mean_bench = Self::mean(bench_window);
            let beta = Self::calculate_beta(asset_window, bench_window, mean_asset, mean_bench);

            if beta.is_nan() {
                result.push(f64::NAN);
            } else {
                let expected_return = daily_rf + beta * (mean_bench - daily_rf);
                let alpha = mean_asset - expected_return;
                let annualized_alpha = alpha * self.annualization_factor;
                result.push(annualized_alpha);
            }
        }

        result
    }
}

impl TechnicalIndicator for Alpha {
    fn name(&self) -> &str {
        "Alpha"
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
                reason: "Benchmark returns must be set before computing Alpha".to_string(),
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
    fn test_alpha_with_same_series() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let alpha = Alpha::new(20);
        let result = alpha.calculate_with_benchmark(&prices, &prices);

        // Alpha of an asset with itself as benchmark should be 0
        let last = result.last().unwrap();
        assert!(last.abs() < 0.001);
    }

    #[test]
    fn test_alpha_outperformance() {
        // Asset outperforms benchmark with similar volatility pattern
        let benchmark: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.2 + ((i as f64) * 0.3).sin() * 2.0)
            .collect();
        let asset: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.8 + ((i as f64) * 0.3).sin() * 2.0) // Much higher returns
            .collect();

        let alpha = Alpha::new(20);
        let result = alpha.calculate_with_benchmark(&asset, &benchmark);

        // Alpha should be positive (outperformance beyond beta exposure)
        let last = result.last().unwrap();
        assert!(!last.is_nan());
        // With higher returns and same volatility pattern, should have positive alpha
        assert!(*last > 0.0, "Expected positive alpha, got {}", last);
    }

    #[test]
    fn test_alpha_underperformance() {
        // Asset underperforms benchmark with similar volatility pattern
        let benchmark: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.8 + ((i as f64) * 0.3).sin() * 2.0)
            .collect();
        let asset: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.2 + ((i as f64) * 0.3).sin() * 2.0) // Much lower returns
            .collect();

        let alpha = Alpha::new(20);
        let result = alpha.calculate_with_benchmark(&asset, &benchmark);

        // Alpha should be negative (underperformance)
        let last = result.last().unwrap();
        assert!(!last.is_nan());
        assert!(*last < 0.0, "Expected negative alpha, got {}", last);
    }
}
