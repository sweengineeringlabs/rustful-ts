//! Tail Dependence (IND-407) - Copula tail measure
//!
//! Measures the degree of dependence in the tails of joint distributions
//! using copula-based methods. This is crucial for understanding crash risk
//! and portfolio diversification during extreme events.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Tail Dependence indicator
#[derive(Debug, Clone)]
pub struct TailDependenceConfig {
    /// Rolling window period
    pub period: usize,
    /// Tail threshold (percentile, e.g., 0.05 for 5%)
    pub tail_threshold: f64,
    /// Whether to calculate lower tail (crashes) or upper tail (rallies)
    pub lower_tail: bool,
}

impl Default for TailDependenceConfig {
    fn default() -> Self {
        Self {
            period: 100,
            tail_threshold: 0.10,
            lower_tail: true,
        }
    }
}

/// Tail Dependence - Copula-based tail dependence measure (IND-407)
///
/// Measures the probability that one asset experiences an extreme event
/// given that another asset also experiences an extreme event. This captures
/// dependence structure in the tails that correlation cannot capture.
///
/// # Formula
/// Lower Tail Dependence Coefficient:
/// lambda_L = lim(u->0) P(U <= u | V <= u)
///
/// Estimated empirically as:
/// lambda = count(both_in_tail) / count(benchmark_in_tail)
///
/// # Interpretation
/// - 0: No tail dependence (assets independent in extremes)
/// - 1: Perfect tail dependence (always move together in extremes)
/// - High lower tail: Diversification fails in crashes
/// - Compare with correlation to detect non-linear dependence
///
/// # Example
/// ```ignore
/// let td = TailDependenceCopula::new(100, 0.10, true).unwrap();
/// let coefficients = td.calculate(&asset_returns, &benchmark_returns);
/// ```
#[derive(Debug, Clone)]
pub struct TailDependenceCopula {
    config: TailDependenceConfig,
}

impl TailDependenceCopula {
    /// Create a new Tail Dependence indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window for calculation (minimum 50)
    /// * `tail_threshold` - Percentile threshold (0.01 to 0.25)
    /// * `lower_tail` - True for lower tail (crashes), false for upper tail
    pub fn new(period: usize, tail_threshold: f64, lower_tail: bool) -> Result<Self> {
        if period < 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 50 for meaningful tail estimation".to_string(),
            });
        }
        if tail_threshold < 0.01 || tail_threshold > 0.25 {
            return Err(IndicatorError::InvalidParameter {
                name: "tail_threshold".to_string(),
                reason: "must be between 0.01 and 0.25".to_string(),
            });
        }
        Ok(Self {
            config: TailDependenceConfig {
                period,
                tail_threshold,
                lower_tail,
            },
        })
    }

    /// Create with default configuration
    pub fn default_indicator() -> Self {
        Self {
            config: TailDependenceConfig::default(),
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

    /// Find the percentile threshold value
    fn percentile_threshold(data: &[f64], percentile: f64) -> f64 {
        if data.is_empty() {
            return f64::NAN;
        }
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((data.len() as f64) * percentile).floor() as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Calculate empirical tail dependence coefficient for a window
    fn calculate_tail_coefficient(&self, asset_rets: &[f64], bench_rets: &[f64]) -> f64 {
        if asset_rets.len() != bench_rets.len() || asset_rets.is_empty() {
            return f64::NAN;
        }

        let (asset_thresh, bench_thresh) = if self.config.lower_tail {
            // Lower tail: find the low percentile thresholds
            (
                Self::percentile_threshold(asset_rets, self.config.tail_threshold),
                Self::percentile_threshold(bench_rets, self.config.tail_threshold),
            )
        } else {
            // Upper tail: find the high percentile thresholds
            (
                Self::percentile_threshold(asset_rets, 1.0 - self.config.tail_threshold),
                Self::percentile_threshold(bench_rets, 1.0 - self.config.tail_threshold),
            )
        };

        let mut bench_in_tail = 0;
        let mut both_in_tail = 0;

        for i in 0..asset_rets.len() {
            let bench_extreme = if self.config.lower_tail {
                bench_rets[i] <= bench_thresh
            } else {
                bench_rets[i] >= bench_thresh
            };

            if bench_extreme {
                bench_in_tail += 1;
                let asset_extreme = if self.config.lower_tail {
                    asset_rets[i] <= asset_thresh
                } else {
                    asset_rets[i] >= asset_thresh
                };
                if asset_extreme {
                    both_in_tail += 1;
                }
            }
        }

        if bench_in_tail == 0 {
            0.0
        } else {
            both_in_tail as f64 / bench_in_tail as f64
        }
    }

    /// Calculate tail dependence using close prices (asset vs itself as benchmark proxy)
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

            // For single series, use lagged returns as pseudo-benchmark
            if window.len() > 1 {
                let asset_window = &window[1..];
                let bench_window = &window[..window.len()-1];
                result.push(self.calculate_tail_coefficient(asset_window, bench_window));
            } else {
                result.push(f64::NAN);
            }
        }

        // Pad to match original length
        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }

    /// Calculate tail dependence between two series
    pub fn calculate_between(&self, asset: &[f64], benchmark: &[f64]) -> Vec<f64> {
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
            result.push(self.calculate_tail_coefficient(asset_window, bench_window));
        }

        // Pad to match original length
        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }
}

impl TechnicalIndicator for TailDependenceCopula {
    fn name(&self) -> &str {
        "Tail Dependence (Copula)"
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

    #[test]
    fn test_tail_dependence_creation() {
        let td = TailDependenceCopula::new(100, 0.10, true);
        assert!(td.is_ok());

        let td = TailDependenceCopula::new(30, 0.10, true);
        assert!(td.is_err());

        let td = TailDependenceCopula::new(100, 0.30, true);
        assert!(td.is_err());
    }

    #[test]
    fn test_tail_dependence_basic() {
        let td = TailDependenceCopula::new(50, 0.10, true).unwrap();
        let prices = generate_test_prices(150, 0.02);

        let result = td.calculate(&prices);

        assert_eq!(result.len(), prices.len());
        // Check warm-up period produces NaN
        assert!(result[10].is_nan());
        // Check values after warm-up are valid
        assert!(!result[100].is_nan());
        assert!(result[100] >= 0.0 && result[100] <= 1.0);
    }

    #[test]
    fn test_tail_dependence_upper_tail() {
        let td = TailDependenceCopula::new(50, 0.10, false).unwrap();
        let prices = generate_test_prices(150, 0.02);

        let result = td.calculate(&prices);

        assert!(!result[100].is_nan());
        assert!(result[100] >= 0.0 && result[100] <= 1.0);
    }

    #[test]
    fn test_tail_dependence_between_series() {
        let td = TailDependenceCopula::new(50, 0.10, true).unwrap();
        let prices1 = generate_test_prices(150, 0.02);
        let prices2 = generate_test_prices(150, 0.025);

        let result = td.calculate_between(&prices1, &prices2);

        assert_eq!(result.len(), prices1.len());
        assert!(!result[100].is_nan());
        assert!(result[100] >= 0.0 && result[100] <= 1.0);
    }

    #[test]
    fn test_tail_dependence_indicator_trait() {
        let td = TailDependenceCopula::new(50, 0.10, true).unwrap();

        assert_eq!(td.name(), "Tail Dependence (Copula)");
        assert_eq!(td.min_periods(), 51);
        assert_eq!(td.output_features(), 1);
    }

    #[test]
    fn test_tail_dependence_default() {
        let td = TailDependenceCopula::default_indicator();
        assert_eq!(td.config.period, 100);
        assert!((td.config.tail_threshold - 0.10).abs() < 1e-10);
        assert!(td.config.lower_tail);
    }
}
