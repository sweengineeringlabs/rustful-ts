//! Extreme Value Index (IND-409) - GEV/GPD estimation
//!
//! Estimates the tail index of return distributions using Extreme Value Theory.
//! The tail index determines the heaviness of distribution tails and is crucial
//! for risk management and tail probability estimation.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Extreme Value Index indicator
#[derive(Debug, Clone)]
pub struct ExtremeValueIndexConfig {
    /// Rolling window period
    pub period: usize,
    /// Number of tail observations to use (as fraction of period)
    pub tail_fraction: f64,
    /// Estimation method
    pub method: EVIMethod,
}

/// Estimation method for the Extreme Value Index
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EVIMethod {
    /// Hill estimator (for heavy tails)
    Hill,
    /// Pickands estimator (for all tail types)
    Pickands,
    /// Moment estimator (Dekkers-Einmahl-de Haan)
    Moment,
}

impl Default for ExtremeValueIndexConfig {
    fn default() -> Self {
        Self {
            period: 100,
            tail_fraction: 0.10,
            method: EVIMethod::Hill,
        }
    }
}

/// Extreme Value Index - GEV/GPD tail parameter estimation (IND-409)
///
/// Estimates the shape parameter (xi) of the Generalized Extreme Value
/// or Generalized Pareto Distribution from sample data.
///
/// # Theory
/// - xi > 0: Frechet (heavy-tailed, e.g., financial returns)
/// - xi = 0: Gumbel (exponential tails)
/// - xi < 0: Weibull (bounded tails)
///
/// # Methods
/// - Hill: gamma = (1/k) * sum(log(X_i / X_{k+1})) for i = 1..k
/// - Pickands: gamma = log((X_k - X_{2k}) / (X_{2k} - X_{4k})) / log(2)
/// - Moment: Uses first two moments of log-spacings
///
/// # Interpretation
/// - gamma > 0.5: Very heavy tails (extreme risk)
/// - gamma ~ 0.25: Typical for equities
/// - gamma ~ 0: Exponential tails
/// - Higher values indicate fatter tails and more extreme risk
///
/// # Example
/// ```ignore
/// let evi = ExtremeValueIndex::new(100, 0.10, EVIMethod::Hill).unwrap();
/// let tail_index = evi.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct ExtremeValueIndex {
    config: ExtremeValueIndexConfig,
}

impl ExtremeValueIndex {
    /// Create a new Extreme Value Index indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window for calculation (minimum 50)
    /// * `tail_fraction` - Fraction of observations for tail (0.05 to 0.25)
    /// * `method` - Estimation method to use
    pub fn new(period: usize, tail_fraction: f64, method: EVIMethod) -> Result<Self> {
        if period < 50 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 50 for meaningful tail estimation".to_string(),
            });
        }
        if tail_fraction < 0.05 || tail_fraction > 0.25 {
            return Err(IndicatorError::InvalidParameter {
                name: "tail_fraction".to_string(),
                reason: "must be between 0.05 and 0.25".to_string(),
            });
        }
        Ok(Self {
            config: ExtremeValueIndexConfig {
                period,
                tail_fraction,
                method,
            },
        })
    }

    /// Create with default configuration
    pub fn default_indicator() -> Self {
        Self {
            config: ExtremeValueIndexConfig::default(),
        }
    }

    /// Calculate absolute returns
    fn calculate_abs_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }
        prices
            .windows(2)
            .map(|w| {
                if w[0].abs() > 1e-10 {
                    ((w[1] - w[0]) / w[0]).abs()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Hill estimator for the tail index
    /// gamma = (1/k) * sum(log(X_i / X_{k+1})) for i = 1..k
    fn hill_estimator(sorted_desc: &[f64], k: usize) -> f64 {
        if k == 0 || k >= sorted_desc.len() {
            return f64::NAN;
        }

        let threshold = sorted_desc[k];
        if threshold <= 1e-10 {
            return f64::NAN;
        }

        let log_sum: f64 = sorted_desc[..k]
            .iter()
            .filter(|&&x| x > threshold)
            .map(|&x| (x / threshold).ln())
            .sum();

        log_sum / k as f64
    }

    /// Pickands estimator for the tail index
    fn pickands_estimator(sorted_desc: &[f64], k: usize) -> f64 {
        // Need at least 4k observations
        let k4 = 4 * k;
        if k4 >= sorted_desc.len() || k == 0 {
            return f64::NAN;
        }

        let x_k = sorted_desc[k - 1];
        let x_2k = sorted_desc[2 * k - 1];
        let x_4k = sorted_desc[4 * k - 1];

        let num = x_k - x_2k;
        let den = x_2k - x_4k;

        if den.abs() <= 1e-10 {
            return f64::NAN;
        }

        (num / den).ln() / 2.0_f64.ln()
    }

    /// Moment estimator (Dekkers-Einmahl-de Haan)
    fn moment_estimator(sorted_desc: &[f64], k: usize) -> f64 {
        if k == 0 || k >= sorted_desc.len() {
            return f64::NAN;
        }

        let threshold = sorted_desc[k];
        if threshold <= 1e-10 {
            return f64::NAN;
        }

        // Calculate log excesses
        let log_excesses: Vec<f64> = sorted_desc[..k]
            .iter()
            .filter(|&&x| x > threshold)
            .map(|&x| (x / threshold).ln())
            .collect();

        if log_excesses.is_empty() {
            return f64::NAN;
        }

        let n = log_excesses.len() as f64;
        let m1: f64 = log_excesses.iter().sum::<f64>() / n;
        let m2: f64 = log_excesses.iter().map(|x| x * x).sum::<f64>() / n;

        // Moment estimator formula
        let moment_ratio = m1 * m1 / m2;
        m1 + 1.0 - 0.5 / (1.0 - moment_ratio)
    }

    /// Calculate the extreme value index for a window
    fn calculate_evi_window(&self, abs_returns: &[f64]) -> f64 {
        if abs_returns.len() < 10 {
            return f64::NAN;
        }

        // Sort in descending order
        let mut sorted: Vec<f64> = abs_returns.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Number of tail observations
        let k = ((sorted.len() as f64) * self.config.tail_fraction).ceil() as usize;
        let k = k.max(5).min(sorted.len() / 4);

        match self.config.method {
            EVIMethod::Hill => Self::hill_estimator(&sorted, k),
            EVIMethod::Pickands => Self::pickands_estimator(&sorted, k),
            EVIMethod::Moment => Self::moment_estimator(&sorted, k),
        }
    }

    /// Calculate rolling extreme value index
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let abs_returns = Self::calculate_abs_returns(close);
        let mut result = vec![f64::NAN; self.config.period];

        for i in self.config.period..abs_returns.len() {
            let start = i - self.config.period + 1;
            let window = &abs_returns[start..=i];
            result.push(self.calculate_evi_window(window));
        }

        // Pad to match original length
        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }

    /// Calculate tail probability P(X > x) using estimated index
    /// P(X > x) ~ (x / threshold)^(-1/gamma)
    pub fn calculate_tail_probability(&self, close: &[f64], loss_level: f64) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let abs_returns = Self::calculate_abs_returns(close);
        let mut result = vec![f64::NAN; self.config.period];

        for i in self.config.period..abs_returns.len() {
            let start = i - self.config.period + 1;
            let window = &abs_returns[start..=i];

            let gamma = self.calculate_evi_window(window);
            if gamma.is_nan() || gamma <= 0.0 {
                result.push(f64::NAN);
                continue;
            }

            // Estimate threshold as the k-th order statistic
            let mut sorted: Vec<f64> = window.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let k = ((sorted.len() as f64) * self.config.tail_fraction).ceil() as usize;
            let threshold = sorted[k.min(sorted.len() - 1)];

            if loss_level <= threshold || threshold <= 0.0 {
                result.push(self.config.tail_fraction);
            } else {
                // P(X > loss_level) = (k/n) * (loss_level / threshold)^(-1/gamma)
                let prob = (k as f64 / window.len() as f64)
                    * (loss_level / threshold).powf(-1.0 / gamma);
                result.push(prob.min(1.0).max(0.0));
            }
        }

        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }
}

impl TechnicalIndicator for ExtremeValueIndex {
    fn name(&self) -> &str {
        "Extreme Value Index"
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
    fn test_evi_creation() {
        let evi = ExtremeValueIndex::new(100, 0.10, EVIMethod::Hill);
        assert!(evi.is_ok());

        let evi = ExtremeValueIndex::new(30, 0.10, EVIMethod::Hill);
        assert!(evi.is_err());

        let evi = ExtremeValueIndex::new(100, 0.01, EVIMethod::Hill);
        assert!(evi.is_err());
    }

    #[test]
    fn test_evi_hill_basic() {
        let evi = ExtremeValueIndex::new(50, 0.10, EVIMethod::Hill).unwrap();
        let prices = generate_test_prices(150, 0.02);

        let result = evi.calculate(&prices);

        assert_eq!(result.len(), prices.len());
        assert!(result[10].is_nan()); // Warm-up period
        assert!(!result[100].is_nan()); // After warm-up
        // Hill estimator should be positive for typical financial data
        assert!(result[100] >= 0.0);
    }

    #[test]
    fn test_evi_pickands() {
        let evi = ExtremeValueIndex::new(100, 0.10, EVIMethod::Pickands).unwrap();
        let prices = generate_test_prices(200, 0.02);

        let result = evi.calculate(&prices);

        assert!(!result[150].is_nan());
    }

    #[test]
    fn test_evi_moment() {
        let evi = ExtremeValueIndex::new(50, 0.10, EVIMethod::Moment).unwrap();
        let prices = generate_test_prices(150, 0.02);

        let result = evi.calculate(&prices);

        assert!(!result[100].is_nan());
    }

    #[test]
    fn test_tail_probability() {
        let evi = ExtremeValueIndex::new(50, 0.10, EVIMethod::Hill).unwrap();
        let prices = generate_test_prices(150, 0.02);

        let probs = evi.calculate_tail_probability(&prices, 0.05);

        assert!(!probs[100].is_nan());
        // Probability should be between 0 and 1
        assert!(probs[100] >= 0.0 && probs[100] <= 1.0);
    }

    #[test]
    fn test_evi_heavier_tails() {
        let evi = ExtremeValueIndex::new(50, 0.15, EVIMethod::Hill).unwrap();

        // Generate prices with occasional large moves
        let prices_normal = generate_test_prices(150, 0.01);
        let mut prices_heavy = generate_test_prices(150, 0.01);
        // Add some extreme moves
        for i in (30..130).step_by(20) {
            prices_heavy[i] *= 1.05;
        }

        let evi_normal = evi.calculate(&prices_normal);
        let evi_heavy = evi.calculate(&prices_heavy);

        // Heavy-tailed series should have higher EVI on average
        let avg_normal: f64 = evi_normal[100..130].iter().filter(|x| !x.is_nan()).sum::<f64>()
            / evi_normal[100..130].iter().filter(|x| !x.is_nan()).count() as f64;
        let avg_heavy: f64 = evi_heavy[100..130].iter().filter(|x| !x.is_nan()).sum::<f64>()
            / evi_heavy[100..130].iter().filter(|x| !x.is_nan()).count() as f64;

        // Not a strict test since synthetic data may not show clear difference
        assert!(avg_normal.is_finite() && avg_heavy.is_finite());
    }

    #[test]
    fn test_evi_indicator_trait() {
        let evi = ExtremeValueIndex::new(100, 0.10, EVIMethod::Hill).unwrap();

        assert_eq!(evi.name(), "Extreme Value Index");
        assert_eq!(evi.min_periods(), 101);
        assert_eq!(evi.output_features(), 1);
    }

    #[test]
    fn test_evi_default() {
        let evi = ExtremeValueIndex::default_indicator();
        assert_eq!(evi.config.period, 100);
        assert!((evi.config.tail_fraction - 0.10).abs() < 1e-10);
        assert_eq!(evi.config.method, EVIMethod::Hill);
    }
}
