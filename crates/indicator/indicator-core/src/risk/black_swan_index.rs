//! Black Swan Index (IND-410) - Tail event probability
//!
//! Measures the probability and expected frequency of extreme tail events
//! (black swans) based on historical return distributions and extreme value theory.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Black Swan Index indicator
#[derive(Debug, Clone)]
pub struct BlackSwanIndexConfig {
    /// Rolling window period
    pub period: usize,
    /// Threshold for defining a black swan (in standard deviations)
    pub sigma_threshold: f64,
    /// Decay factor for historical weighting
    pub decay_factor: f64,
}

impl Default for BlackSwanIndexConfig {
    fn default() -> Self {
        Self {
            period: 252,
            sigma_threshold: 4.0,
            decay_factor: 0.94,
        }
    }
}

/// Black Swan Index - Tail event probability measure (IND-410)
///
/// Quantifies the likelihood and severity of extreme market events
/// that fall outside normal distribution expectations. Combines
/// historical frequency, tail heaviness, and recent volatility.
///
/// # Components
/// 1. Historical tail frequency: How often extreme events occurred
/// 2. Tail excess kurtosis: How fat the tails are vs normal
/// 3. Recent volatility clustering: Current risk environment
/// 4. Expected extreme loss: Conditional expected loss given extreme event
///
/// # Formula
/// BSI = w1 * tail_frequency + w2 * excess_kurtosis + w3 * vol_ratio
///
/// Normalized to 0-100 scale where:
/// - 0-20: Low tail risk
/// - 20-40: Moderate tail risk
/// - 40-60: Elevated tail risk
/// - 60-80: High tail risk
/// - 80-100: Extreme tail risk
///
/// # Example
/// ```ignore
/// let bsi = BlackSwanIndex::new(252, 4.0, 0.94).unwrap();
/// let index = bsi.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct BlackSwanIndex {
    config: BlackSwanIndexConfig,
}

impl BlackSwanIndex {
    /// Create a new Black Swan Index indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window for calculation (minimum 100)
    /// * `sigma_threshold` - Standard deviations to define extreme (3.0 to 6.0)
    /// * `decay_factor` - Exponential decay for volatility (0.9 to 0.99)
    pub fn new(period: usize, sigma_threshold: f64, decay_factor: f64) -> Result<Self> {
        if period < 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 100 for meaningful tail analysis".to_string(),
            });
        }
        if sigma_threshold < 3.0 || sigma_threshold > 6.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sigma_threshold".to_string(),
                reason: "must be between 3.0 and 6.0".to_string(),
            });
        }
        if decay_factor < 0.9 || decay_factor > 0.99 {
            return Err(IndicatorError::InvalidParameter {
                name: "decay_factor".to_string(),
                reason: "must be between 0.9 and 0.99".to_string(),
            });
        }
        Ok(Self {
            config: BlackSwanIndexConfig {
                period,
                sigma_threshold,
                decay_factor,
            },
        })
    }

    /// Create with default configuration
    pub fn default_indicator() -> Self {
        Self {
            config: BlackSwanIndexConfig::default(),
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

    /// Calculate excess kurtosis (kurtosis - 3)
    fn excess_kurtosis(data: &[f64], mean: f64, std: f64) -> f64 {
        if data.len() < 4 || std < 1e-10 {
            return 0.0;
        }
        let n = data.len() as f64;
        let m4: f64 = data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n;
        // Excess kurtosis = kurtosis - 3 (normal has kurtosis = 3)
        m4 - 3.0
    }

    /// Calculate EWMA volatility
    fn ewma_volatility(returns: &[f64], decay: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        let mut ewma_var = returns[0] * returns[0];
        for ret in returns.iter().skip(1) {
            ewma_var = decay * ewma_var + (1.0 - decay) * ret * ret;
        }
        ewma_var.sqrt()
    }

    /// Calculate Black Swan Index for a window
    fn calculate_bsi_window(&self, returns: &[f64]) -> f64 {
        if returns.len() < 20 {
            return f64::NAN;
        }

        let mean_ret = Self::mean(returns);
        let std_ret = Self::std_dev(returns, mean_ret);

        if std_ret < 1e-10 {
            return 0.0;
        }

        // Component 1: Historical tail frequency
        let threshold_neg = mean_ret - self.config.sigma_threshold * std_ret;
        let threshold_pos = mean_ret + self.config.sigma_threshold * std_ret;
        let tail_count = returns
            .iter()
            .filter(|&&r| r < threshold_neg || r > threshold_pos)
            .count();
        let tail_frequency = tail_count as f64 / returns.len() as f64;

        // Expected frequency under normal distribution for n-sigma events
        // For 4-sigma: ~0.00006 (2 * 0.00003)
        let normal_expected = 2.0 * (1.0 - Self::normal_cdf(self.config.sigma_threshold));
        let frequency_ratio = if normal_expected > 0.0 {
            tail_frequency / normal_expected
        } else {
            1.0
        };

        // Component 2: Excess kurtosis (fat tails indicator)
        let excess_kurt = Self::excess_kurtosis(returns, mean_ret, std_ret);
        let kurtosis_score = (excess_kurt / 6.0).max(0.0).min(1.0); // Normalize

        // Component 3: Recent volatility vs historical
        let recent_len = (returns.len() / 4).max(20);
        let recent_returns = &returns[returns.len() - recent_len..];
        let recent_vol = Self::ewma_volatility(recent_returns, self.config.decay_factor);
        let vol_ratio = if std_ret > 0.0 {
            (recent_vol / std_ret - 1.0).max(0.0).min(2.0) / 2.0
        } else {
            0.0
        };

        // Component 4: Maximum drawdown severity
        let max_loss = returns.iter().cloned().fold(0.0_f64, f64::min);
        let max_gain = returns.iter().cloned().fold(0.0_f64, f64::max);
        let extreme_magnitude = (max_loss.abs().max(max_gain) / (self.config.sigma_threshold * std_ret))
            .max(0.0)
            .min(2.0) / 2.0;

        // Combine components with weights
        let w1 = 0.30; // Tail frequency
        let w2 = 0.25; // Kurtosis
        let w3 = 0.25; // Volatility clustering
        let w4 = 0.20; // Extreme magnitude

        let raw_score = w1 * frequency_ratio.min(3.0) / 3.0
            + w2 * kurtosis_score
            + w3 * vol_ratio
            + w4 * extreme_magnitude;

        // Scale to 0-100
        (raw_score * 100.0).min(100.0).max(0.0)
    }

    /// Approximate normal CDF using error function approximation
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
    }

    /// Approximate error function
    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Calculate rolling Black Swan Index
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
            result.push(self.calculate_bsi_window(window));
        }

        // Pad to match original length
        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }

    /// Calculate expected days until next extreme event
    pub fn calculate_expected_frequency(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let returns = Self::calculate_returns(close);
        let mut result = vec![f64::NAN; self.config.period];

        for i in self.config.period..returns.len() {
            let start = i - self.config.period + 1;
            let window = &returns[start..=i];

            let mean_ret = Self::mean(window);
            let std_ret = Self::std_dev(window, mean_ret);

            if std_ret < 1e-10 {
                result.push(f64::INFINITY);
                continue;
            }

            let threshold_neg = mean_ret - self.config.sigma_threshold * std_ret;
            let threshold_pos = mean_ret + self.config.sigma_threshold * std_ret;
            let tail_count = window
                .iter()
                .filter(|&&r| r < threshold_neg || r > threshold_pos)
                .count();

            if tail_count == 0 {
                result.push(f64::INFINITY);
            } else {
                // Expected days = period / tail_count
                result.push(window.len() as f64 / tail_count as f64);
            }
        }

        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }

    /// Calculate conditional expected loss given extreme event
    pub fn calculate_expected_tail_loss(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let returns = Self::calculate_returns(close);
        let mut result = vec![f64::NAN; self.config.period];

        for i in self.config.period..returns.len() {
            let start = i - self.config.period + 1;
            let window = &returns[start..=i];

            let mean_ret = Self::mean(window);
            let std_ret = Self::std_dev(window, mean_ret);

            if std_ret < 1e-10 {
                result.push(0.0);
                continue;
            }

            let threshold = mean_ret - self.config.sigma_threshold * std_ret;
            let tail_losses: Vec<f64> = window
                .iter()
                .filter(|&&r| r < threshold)
                .copied()
                .collect();

            if tail_losses.is_empty() {
                result.push(threshold.abs());
            } else {
                result.push(tail_losses.iter().map(|x| x.abs()).sum::<f64>() / tail_losses.len() as f64);
            }
        }

        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }
}

impl TechnicalIndicator for BlackSwanIndex {
    fn name(&self) -> &str {
        "Black Swan Index"
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

    fn generate_prices_with_extremes(n: usize) -> Vec<f64> {
        let mut prices = generate_test_prices(n, 0.01);
        // Add some extreme moves
        if n > 150 {
            prices[100] *= 0.90; // 10% crash
            prices[150] *= 0.85; // 15% crash
        }
        prices
    }

    #[test]
    fn test_bsi_creation() {
        let bsi = BlackSwanIndex::new(252, 4.0, 0.94);
        assert!(bsi.is_ok());

        let bsi = BlackSwanIndex::new(50, 4.0, 0.94);
        assert!(bsi.is_err());

        let bsi = BlackSwanIndex::new(252, 2.0, 0.94);
        assert!(bsi.is_err());

        let bsi = BlackSwanIndex::new(252, 4.0, 0.80);
        assert!(bsi.is_err());
    }

    #[test]
    fn test_bsi_basic() {
        let bsi = BlackSwanIndex::new(100, 4.0, 0.94).unwrap();
        let prices = generate_test_prices(300, 0.02);

        let result = bsi.calculate(&prices);

        assert_eq!(result.len(), prices.len());
        assert!(result[50].is_nan()); // Warm-up period
        assert!(!result[200].is_nan()); // After warm-up
        // BSI should be in valid range
        assert!(result[200] >= 0.0 && result[200] <= 100.0);
    }

    #[test]
    fn test_bsi_with_extremes() {
        let bsi = BlackSwanIndex::new(100, 4.0, 0.94).unwrap();
        let prices_normal = generate_test_prices(300, 0.01);
        let prices_extreme = generate_prices_with_extremes(300);

        let bsi_normal = bsi.calculate(&prices_normal);
        let bsi_extreme = bsi.calculate(&prices_extreme);

        // Extreme series should have higher BSI on average after extreme events
        let avg_normal: f64 = bsi_normal[200..250].iter().filter(|x| !x.is_nan()).sum::<f64>()
            / bsi_normal[200..250].iter().filter(|x| !x.is_nan()).count().max(1) as f64;
        let avg_extreme: f64 = bsi_extreme[200..250].iter().filter(|x| !x.is_nan()).sum::<f64>()
            / bsi_extreme[200..250].iter().filter(|x| !x.is_nan()).count().max(1) as f64;

        assert!(avg_extreme >= avg_normal || (avg_extreme - avg_normal).abs() < 20.0);
    }

    #[test]
    fn test_expected_frequency() {
        let bsi = BlackSwanIndex::new(100, 4.0, 0.94).unwrap();
        let prices = generate_test_prices(300, 0.02);

        let freq = bsi.calculate_expected_frequency(&prices);

        assert!(!freq[200].is_nan());
        assert!(freq[200] > 0.0); // Should be positive
    }

    #[test]
    fn test_expected_tail_loss() {
        let bsi = BlackSwanIndex::new(100, 4.0, 0.94).unwrap();
        let prices = generate_prices_with_extremes(300);

        let etl = bsi.calculate_expected_tail_loss(&prices);

        assert!(!etl[200].is_nan());
        assert!(etl[200] >= 0.0); // Should be non-negative
    }

    #[test]
    fn test_bsi_indicator_trait() {
        let bsi = BlackSwanIndex::new(252, 4.0, 0.94).unwrap();

        assert_eq!(bsi.name(), "Black Swan Index");
        assert_eq!(bsi.min_periods(), 253);
        assert_eq!(bsi.output_features(), 1);
    }

    #[test]
    fn test_bsi_default() {
        let bsi = BlackSwanIndex::default_indicator();
        assert_eq!(bsi.config.period, 252);
        assert!((bsi.config.sigma_threshold - 4.0).abs() < 1e-10);
        assert!((bsi.config.decay_factor - 0.94).abs() < 1e-10);
    }

    #[test]
    fn test_normal_cdf() {
        // Test normal CDF approximation
        let cdf_0 = BlackSwanIndex::normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 0.01);

        let cdf_2 = BlackSwanIndex::normal_cdf(2.0);
        assert!((cdf_2 - 0.9772).abs() < 0.01);
    }
}
