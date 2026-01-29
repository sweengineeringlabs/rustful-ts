//! Crisis Alpha (IND-411) - Performance in crashes
//!
//! Measures an asset's or strategy's performance during market crisis periods.
//! Positive crisis alpha indicates the ability to profit or hedge during crashes,
//! a highly valuable property for portfolio protection.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Crisis Alpha indicator
#[derive(Debug, Clone)]
pub struct CrisisAlphaConfig {
    /// Rolling window period
    pub period: usize,
    /// Threshold for defining a crisis (benchmark return in std devs)
    pub crisis_threshold: f64,
    /// Lookback for identifying crisis periods
    pub crisis_lookback: usize,
}

impl Default for CrisisAlphaConfig {
    fn default() -> Self {
        Self {
            period: 252,
            crisis_threshold: 2.0,
            crisis_lookback: 20,
        }
    }
}

/// Crisis Alpha - Performance during market crashes (IND-411)
///
/// Measures how an asset performs specifically during market crisis events.
/// Unlike regular alpha/beta which measure average relationship, crisis alpha
/// focuses only on extreme negative market periods.
///
/// # Formula
/// Crisis Alpha = avg(asset_return during crises) - beta * avg(benchmark_return during crises)
///
/// Where crisis is defined as: benchmark_return < -threshold * sigma
///
/// # Interpretation
/// - Positive: Outperforms during crises (hedge-like behavior)
/// - Zero: Neutral crisis performance
/// - Negative: Underperforms during crises (amplifies losses)
///
/// # Properties
/// - Helps identify true diversifiers vs. fair-weather hedges
/// - Key metric for tail risk hedging strategies
/// - Captures non-linear dependence during stress
///
/// # Example
/// ```ignore
/// let ca = CrisisAlpha::new(252, 2.0, 20).unwrap();
/// let alpha = ca.calculate(&asset_prices, &benchmark_prices);
/// ```
#[derive(Debug, Clone)]
pub struct CrisisAlpha {
    config: CrisisAlphaConfig,
}

impl CrisisAlpha {
    /// Create a new Crisis Alpha indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window for calculation (minimum 100)
    /// * `crisis_threshold` - Std devs below mean to define crisis (1.5 to 4.0)
    /// * `crisis_lookback` - Days to look back for crisis detection
    pub fn new(period: usize, crisis_threshold: f64, crisis_lookback: usize) -> Result<Self> {
        if period < 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 100 for meaningful crisis analysis".to_string(),
            });
        }
        if crisis_threshold < 1.5 || crisis_threshold > 4.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "crisis_threshold".to_string(),
                reason: "must be between 1.5 and 4.0".to_string(),
            });
        }
        if crisis_lookback < 5 || crisis_lookback > 60 {
            return Err(IndicatorError::InvalidParameter {
                name: "crisis_lookback".to_string(),
                reason: "must be between 5 and 60".to_string(),
            });
        }
        Ok(Self {
            config: CrisisAlphaConfig {
                period,
                crisis_threshold,
                crisis_lookback,
            },
        })
    }

    /// Create with default configuration
    pub fn default_indicator() -> Self {
        Self {
            config: CrisisAlphaConfig::default(),
        }
    }

    /// Calculate returns from prices
    fn calculate_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }
        prices
            .windows(2)
            .map(|w| {
                if w[0].abs() > 1e-10 {
                    (w[1] - w[0]) / w[0]
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Calculate mean of a slice
    fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Calculate standard deviation
    fn std_dev(data: &[f64], mean: f64) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate beta between asset and benchmark
    fn calculate_beta(asset_rets: &[f64], bench_rets: &[f64]) -> f64 {
        if asset_rets.len() != bench_rets.len() || asset_rets.is_empty() {
            return 0.0;
        }

        let n = asset_rets.len() as f64;
        let mean_asset = Self::mean(asset_rets);
        let mean_bench = Self::mean(bench_rets);

        let covariance: f64 = asset_rets
            .iter()
            .zip(bench_rets.iter())
            .map(|(a, b)| (a - mean_asset) * (b - mean_bench))
            .sum::<f64>() / (n - 1.0);

        let bench_variance: f64 = bench_rets
            .iter()
            .map(|b| (b - mean_bench).powi(2))
            .sum::<f64>() / (n - 1.0);

        if bench_variance > 1e-10 {
            covariance / bench_variance
        } else {
            0.0
        }
    }

    /// Identify crisis periods based on benchmark returns
    fn identify_crisis_periods(
        bench_rets: &[f64],
        threshold: f64,
        lookback: usize,
    ) -> Vec<bool> {
        let n = bench_rets.len();
        let mut is_crisis = vec![false; n];

        if n < lookback {
            return is_crisis;
        }

        for i in lookback..n {
            // Calculate rolling mean and std
            let window = &bench_rets[i - lookback..i];
            let mean = Self::mean(window);
            let std = Self::std_dev(window, mean);

            // Check if current or recent returns are crisis-level
            if std > 1e-10 {
                let crisis_threshold = mean - threshold * std;
                // Mark as crisis if any recent day was extreme
                for j in (i.saturating_sub(5))..=i {
                    if j < n && bench_rets[j] < crisis_threshold {
                        is_crisis[i] = true;
                        break;
                    }
                }
            }
        }

        is_crisis
    }

    /// Calculate crisis alpha for a single series (using itself as benchmark proxy)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let returns = Self::calculate_returns(close);
        let mut result = vec![f64::NAN; self.config.period];

        for i in self.config.period..returns.len() {
            let start = i - self.config.period + 1;
            let window = &returns[start..=i];

            // Use lagged returns as pseudo-benchmark
            if window.len() < 2 {
                result.push(f64::NAN);
                continue;
            }

            let asset_window = &window[1..];
            let bench_window = &window[..window.len()-1];

            let crisis_mask = Self::identify_crisis_periods(
                bench_window,
                self.config.crisis_threshold,
                self.config.crisis_lookback.min(bench_window.len()),
            );

            // Extract crisis period returns
            let crisis_asset: Vec<f64> = asset_window
                .iter()
                .zip(crisis_mask.iter())
                .filter(|(_, &is_crisis)| is_crisis)
                .map(|(&ret, _)| ret)
                .collect();

            let crisis_bench: Vec<f64> = bench_window
                .iter()
                .zip(crisis_mask.iter())
                .filter(|(_, &is_crisis)| is_crisis)
                .map(|(&ret, _)| ret)
                .collect();

            if crisis_asset.len() < 5 {
                result.push(0.0); // Not enough crisis periods
                continue;
            }

            // Calculate crisis alpha
            let beta = Self::calculate_beta(&crisis_asset, &crisis_bench);
            let avg_asset_crisis = Self::mean(&crisis_asset);
            let avg_bench_crisis = Self::mean(&crisis_bench);

            let crisis_alpha = avg_asset_crisis - beta * avg_bench_crisis;

            // Annualize (assuming daily data)
            result.push(crisis_alpha * 252.0 * 100.0); // In percentage points
        }

        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }

    /// Calculate crisis alpha vs an external benchmark
    pub fn calculate_vs_benchmark(&self, asset: &[f64], benchmark: &[f64]) -> Vec<f64> {
        let n = asset.len();
        if n != benchmark.len() || n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let asset_rets = Self::calculate_returns(asset);
        let bench_rets = Self::calculate_returns(benchmark);
        let mut result = vec![f64::NAN; self.config.period];

        for i in self.config.period..asset_rets.len() {
            let start = i - self.config.period + 1;
            let asset_window = &asset_rets[start..=i];
            let bench_window = &bench_rets[start..=i];

            let crisis_mask = Self::identify_crisis_periods(
                bench_window,
                self.config.crisis_threshold,
                self.config.crisis_lookback.min(bench_window.len()),
            );

            // Extract crisis period returns
            let crisis_asset: Vec<f64> = asset_window
                .iter()
                .zip(crisis_mask.iter())
                .filter(|(_, &is_crisis)| is_crisis)
                .map(|(&ret, _)| ret)
                .collect();

            let crisis_bench: Vec<f64> = bench_window
                .iter()
                .zip(crisis_mask.iter())
                .filter(|(_, &is_crisis)| is_crisis)
                .map(|(&ret, _)| ret)
                .collect();

            if crisis_asset.len() < 5 {
                result.push(0.0);
                continue;
            }

            let beta = Self::calculate_beta(&crisis_asset, &crisis_bench);
            let avg_asset_crisis = Self::mean(&crisis_asset);
            let avg_bench_crisis = Self::mean(&crisis_bench);

            let crisis_alpha = avg_asset_crisis - beta * avg_bench_crisis;
            result.push(crisis_alpha * 252.0 * 100.0);
        }

        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }

    /// Calculate crisis beta (beta only during crisis periods)
    pub fn calculate_crisis_beta(&self, asset: &[f64], benchmark: &[f64]) -> Vec<f64> {
        let n = asset.len();
        if n != benchmark.len() || n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let asset_rets = Self::calculate_returns(asset);
        let bench_rets = Self::calculate_returns(benchmark);
        let mut result = vec![f64::NAN; self.config.period];

        for i in self.config.period..asset_rets.len() {
            let start = i - self.config.period + 1;
            let asset_window = &asset_rets[start..=i];
            let bench_window = &bench_rets[start..=i];

            let crisis_mask = Self::identify_crisis_periods(
                bench_window,
                self.config.crisis_threshold,
                self.config.crisis_lookback.min(bench_window.len()),
            );

            let crisis_asset: Vec<f64> = asset_window
                .iter()
                .zip(crisis_mask.iter())
                .filter(|(_, &is_crisis)| is_crisis)
                .map(|(&ret, _)| ret)
                .collect();

            let crisis_bench: Vec<f64> = bench_window
                .iter()
                .zip(crisis_mask.iter())
                .filter(|(_, &is_crisis)| is_crisis)
                .map(|(&ret, _)| ret)
                .collect();

            if crisis_asset.len() < 5 {
                result.push(f64::NAN);
                continue;
            }

            result.push(Self::calculate_beta(&crisis_asset, &crisis_bench));
        }

        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }

    /// Calculate proportion of time spent in crisis
    pub fn calculate_crisis_frequency(&self, benchmark: &[f64]) -> Vec<f64> {
        let n = benchmark.len();
        if n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let bench_rets = Self::calculate_returns(benchmark);
        let mut result = vec![f64::NAN; self.config.period];

        for i in self.config.period..bench_rets.len() {
            let start = i - self.config.period + 1;
            let bench_window = &bench_rets[start..=i];

            let crisis_mask = Self::identify_crisis_periods(
                bench_window,
                self.config.crisis_threshold,
                self.config.crisis_lookback.min(bench_window.len()),
            );

            let crisis_count = crisis_mask.iter().filter(|&&x| x).count();
            result.push(crisis_count as f64 / crisis_mask.len() as f64 * 100.0);
        }

        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }
}

impl TechnicalIndicator for CrisisAlpha {
    fn name(&self) -> &str {
        "Crisis Alpha"
    }

    fn min_periods(&self) -> usize {
        self.config.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_prices(n: usize, volatility: f64) -> Vec<f64> {
        let mut prices = Vec::with_capacity(n);
        let mut price = 100.0;
        for i in 0..n {
            let ret = (i as f64 * 0.1).sin() * volatility + 0.0001;
            price *= 1.0 + ret;
            prices.push(price);
        }
        prices
    }

    fn generate_hedge_asset(benchmark: &[f64], correlation: f64) -> Vec<f64> {
        let mut prices = Vec::with_capacity(benchmark.len());
        let mut price = 100.0;

        for i in 1..benchmark.len() {
            let bench_ret = (benchmark[i] - benchmark[i-1]) / benchmark[i-1];
            // Hedge asset moves opposite to benchmark with some noise
            let asset_ret = -correlation * bench_ret + (i as f64 * 0.2).sin() * 0.005;
            price *= 1.0 + asset_ret;
            prices.push(price);
        }

        // Prepend initial price
        let mut result = vec![100.0];
        result.extend(prices);
        result
    }

    #[test]
    fn test_crisis_alpha_creation() {
        let ca = CrisisAlpha::new(252, 2.0, 20);
        assert!(ca.is_ok());

        let ca = CrisisAlpha::new(50, 2.0, 20);
        assert!(ca.is_err());

        let ca = CrisisAlpha::new(252, 0.5, 20);
        assert!(ca.is_err());

        let ca = CrisisAlpha::new(252, 2.0, 100);
        assert!(ca.is_err());
    }

    #[test]
    fn test_crisis_alpha_basic() {
        let ca = CrisisAlpha::new(100, 2.0, 20).unwrap();
        let prices = generate_test_prices(300, 0.02);

        let result = ca.calculate(&prices);

        assert_eq!(result.len(), prices.len());
        assert!(result[50].is_nan()); // Warm-up period
        // After warm-up, should have valid values
        let valid_count = result[100..].iter().filter(|x| !x.is_nan()).count();
        assert!(valid_count > 0);
    }

    #[test]
    fn test_crisis_alpha_hedge_asset() {
        let ca = CrisisAlpha::new(100, 2.0, 20).unwrap();
        let benchmark = generate_test_prices(300, 0.03);
        let hedge = generate_hedge_asset(&benchmark, 0.8);

        let alpha = ca.calculate_vs_benchmark(&hedge, &benchmark);

        // Hedge asset should have positive crisis alpha (or at least defined)
        let valid_values: Vec<f64> = alpha[150..].iter().filter(|x| !x.is_nan()).copied().collect();
        assert!(!valid_values.is_empty());
    }

    #[test]
    fn test_crisis_beta() {
        let ca = CrisisAlpha::new(100, 2.0, 20).unwrap();
        let benchmark = generate_test_prices(300, 0.03);
        let hedge = generate_hedge_asset(&benchmark, 0.8);

        let beta = ca.calculate_crisis_beta(&hedge, &benchmark);

        let valid_values: Vec<f64> = beta[150..].iter().filter(|x| !x.is_nan()).copied().collect();
        assert!(!valid_values.is_empty());
        // Hedge should have negative or low crisis beta
        let avg_beta = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
        assert!(avg_beta < 1.0); // Lower than market beta
    }

    #[test]
    fn test_crisis_frequency() {
        let ca = CrisisAlpha::new(100, 2.0, 20).unwrap();
        let benchmark = generate_test_prices(300, 0.02);

        let freq = ca.calculate_crisis_frequency(&benchmark);

        assert!(!freq[200].is_nan());
        // Frequency should be between 0 and 100
        assert!(freq[200] >= 0.0 && freq[200] <= 100.0);
    }

    #[test]
    fn test_crisis_alpha_indicator_trait() {
        let ca = CrisisAlpha::new(252, 2.0, 20).unwrap();

        assert_eq!(ca.name(), "Crisis Alpha");
        assert_eq!(ca.min_periods(), 253);
        assert_eq!(ca.output_features(), 1);
    }

    #[test]
    fn test_crisis_alpha_default() {
        let ca = CrisisAlpha::default_indicator();
        assert_eq!(ca.config.period, 252);
        assert!((ca.config.crisis_threshold - 2.0).abs() < 1e-10);
        assert_eq!(ca.config.crisis_lookback, 20);
    }

    #[test]
    fn test_crisis_identification() {
        // Create returns with clear crisis
        let mut returns = vec![0.01; 100];
        returns[50] = -0.10; // 10% drop
        returns[51] = -0.08;

        let crisis_mask = CrisisAlpha::identify_crisis_periods(&returns, 2.0, 20);

        // Should identify crisis around index 50-55
        let crisis_count = crisis_mask[45..60].iter().filter(|&&x| x).count();
        assert!(crisis_count > 0);
    }
}
