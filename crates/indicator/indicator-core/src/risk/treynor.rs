//! Treynor Ratio implementation.
//!
//! Measures the excess return per unit of systematic risk (Beta).

use crate::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Treynor Ratio indicator.
///
/// Similar to Sharpe Ratio but uses Beta (systematic risk) instead of
/// total volatility. Measures excess return per unit of market risk.
///
/// Formula: (Portfolio Return - Risk-Free Rate) / Beta
///
/// Higher values indicate better risk-adjusted performance relative to market risk.
/// Best used for well-diversified portfolios where unsystematic risk is minimal.
#[derive(Debug, Clone)]
pub struct TreynorRatio {
    /// Rolling window period for calculation.
    period: usize,
    /// Annualized risk-free rate.
    risk_free_rate: f64,
    /// Annualization factor (252 for daily, 52 for weekly, 12 for monthly).
    annualization_factor: f64,
    /// Benchmark/market returns (must be set before calculation).
    benchmark_returns: Vec<f64>,
}

impl TreynorRatio {
    /// Create a new Treynor Ratio indicator.
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

    /// Create with custom risk-free rate.
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

    /// Calculate Treynor Ratio values.
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

            if beta == 0.0 || beta.is_nan() {
                result.push(f64::NAN);
            } else {
                // Treynor Ratio: (R_a - R_f) / Beta
                let excess_return = mean_asset - daily_rf;
                // Annualize
                let annualized_excess = excess_return * self.annualization_factor;
                let treynor = annualized_excess / beta;
                result.push(treynor);
            }
        }

        result
    }

    /// Calculate Treynor Ratio given two price series.
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

            if beta == 0.0 || beta.is_nan() {
                result.push(f64::NAN);
            } else {
                let excess_return = mean_asset - daily_rf;
                let annualized_excess = excess_return * self.annualization_factor;
                let treynor = annualized_excess / beta;
                result.push(treynor);
            }
        }

        result
    }
}

impl TechnicalIndicator for TreynorRatio {
    fn name(&self) -> &str {
        "TreynorRatio"
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
                reason: "Benchmark returns must be set before computing Treynor Ratio".to_string(),
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
    fn test_treynor_basic() {
        let benchmark: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.3)
            .collect();
        let asset: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.5)
            .collect();

        let treynor = TreynorRatio::new(20);
        let result = treynor.calculate_with_benchmark(&asset, &benchmark);

        // Should have valid values after warm-up
        assert!(!result[30].is_nan());
    }

    #[test]
    fn test_treynor_with_same_series() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let treynor = TreynorRatio::new(20);
        let result = treynor.calculate_with_benchmark(&prices, &prices);

        // With same series, beta = 1, so Treynor = excess return
        let last = result.last().unwrap();
        assert!(!last.is_nan());
    }

    #[test]
    fn test_treynor_high_beta() {
        // Asset with beta > 1 and same excess return should have lower Treynor
        let benchmark: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.3)
            .collect();
        let high_beta_asset: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.6) // Higher beta
            .collect();
        let low_beta_asset: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.15) // Lower beta
            .collect();

        let treynor = TreynorRatio::new(20);
        let high_result = treynor.calculate_with_benchmark(&high_beta_asset, &benchmark);
        let low_result = treynor.calculate_with_benchmark(&low_beta_asset, &benchmark);

        // Both should have valid results
        assert!(!high_result[30].is_nan());
        assert!(!low_result[30].is_nan());
    }
}
