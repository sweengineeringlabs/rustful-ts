//! Stress Index (IND-412) - Market stress composite
//!
//! A comprehensive market stress indicator that combines multiple
//! stress signals into a single composite measure. Useful for
//! identifying periods of elevated market stress and tail risk.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Stress Index indicator
#[derive(Debug, Clone)]
pub struct StressIndexConfig {
    /// Rolling window period for calculations
    pub period: usize,
    /// Short-term volatility period
    pub short_vol_period: usize,
    /// Long-term volatility period
    pub long_vol_period: usize,
    /// Correlation lookback period
    pub correlation_period: usize,
}

impl Default for StressIndexConfig {
    fn default() -> Self {
        Self {
            period: 252,
            short_vol_period: 10,
            long_vol_period: 60,
            correlation_period: 20,
        }
    }
}

/// Stress Index - Market stress composite indicator (IND-412)
///
/// Combines multiple stress indicators into a single composite measure:
/// 1. Volatility ratio (short/long)
/// 2. Return momentum (recent vs historical)
/// 3. Drawdown intensity
/// 4. Volatility of volatility
/// 5. Tail risk measure
///
/// # Formula
/// StressIndex = weighted_sum(normalized_stress_components)
///
/// Normalized to 0-100 scale where:
/// - 0-20: Low stress (calm markets)
/// - 20-40: Moderate stress
/// - 40-60: Elevated stress
/// - 60-80: High stress
/// - 80-100: Extreme stress (crisis levels)
///
/// # Components
/// - Volatility regime: Short vs long-term volatility ratio
/// - Price stress: Magnitude and persistence of declines
/// - Volatility clustering: Recent vol-of-vol
/// - Tail behavior: Frequency of extreme moves
///
/// # Example
/// ```ignore
/// let si = StressIndex::new(252, 10, 60, 20).unwrap();
/// let stress = si.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct StressIndex {
    config: StressIndexConfig,
}

impl StressIndex {
    /// Create a new Stress Index indicator
    ///
    /// # Arguments
    /// * `period` - Overall lookback period (minimum 100)
    /// * `short_vol_period` - Short-term volatility window (5 to 30)
    /// * `long_vol_period` - Long-term volatility window (30 to 252)
    /// * `correlation_period` - Correlation calculation window (10 to 60)
    pub fn new(
        period: usize,
        short_vol_period: usize,
        long_vol_period: usize,
        correlation_period: usize,
    ) -> Result<Self> {
        if period < 100 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 100".to_string(),
            });
        }
        if short_vol_period < 5 || short_vol_period > 30 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_vol_period".to_string(),
                reason: "must be between 5 and 30".to_string(),
            });
        }
        if long_vol_period < 30 || long_vol_period > 252 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_vol_period".to_string(),
                reason: "must be between 30 and 252".to_string(),
            });
        }
        if short_vol_period >= long_vol_period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_vol_period".to_string(),
                reason: "must be less than long_vol_period".to_string(),
            });
        }
        if correlation_period < 10 || correlation_period > 60 {
            return Err(IndicatorError::InvalidParameter {
                name: "correlation_period".to_string(),
                reason: "must be between 10 and 60".to_string(),
            });
        }
        Ok(Self {
            config: StressIndexConfig {
                period,
                short_vol_period,
                long_vol_period,
                correlation_period,
            },
        })
    }

    /// Create with default configuration
    pub fn default_indicator() -> Self {
        Self {
            config: StressIndexConfig::default(),
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

    /// Calculate volatility (standard deviation of returns)
    fn volatility(returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / (returns.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate rolling volatility
    fn rolling_volatility(returns: &[f64], period: usize) -> Vec<f64> {
        let n = returns.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];
        for i in (period - 1)..n {
            let window = &returns[(i + 1 - period)..=i];
            result.push(Self::volatility(window));
        }
        result
    }

    /// Calculate current drawdown
    fn calculate_drawdown(prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        let mut drawdowns = vec![0.0; n];
        let mut peak = prices[0];

        for i in 0..n {
            if prices[i] > peak {
                peak = prices[i];
            }
            drawdowns[i] = if peak > 0.0 {
                (peak - prices[i]) / peak
            } else {
                0.0
            };
        }
        drawdowns
    }

    /// Component 1: Volatility regime stress
    fn volatility_stress(&self, returns: &[f64], idx: usize) -> f64 {
        if idx < self.config.long_vol_period {
            return 0.0;
        }

        let short_start = idx + 1 - self.config.short_vol_period;
        let long_start = idx + 1 - self.config.long_vol_period;

        let short_vol = Self::volatility(&returns[short_start..=idx]);
        let long_vol = Self::volatility(&returns[long_start..=idx]);

        if long_vol > 1e-10 {
            let ratio = short_vol / long_vol;
            // Normalize: ratio of 1 = no stress, ratio of 2+ = high stress
            ((ratio - 1.0).max(0.0) / 2.0).min(1.0)
        } else {
            0.0
        }
    }

    /// Component 2: Price momentum stress (negative momentum = stress)
    fn momentum_stress(&self, returns: &[f64], idx: usize) -> f64 {
        if idx < self.config.correlation_period {
            return 0.0;
        }

        let window = &returns[(idx + 1 - self.config.correlation_period)..=idx];
        let cumulative_return: f64 = window.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;

        // Normalize: positive return = no stress, negative return = stress
        (-cumulative_return * 5.0).max(0.0).min(1.0)
    }

    /// Component 3: Drawdown stress
    fn drawdown_stress(&self, drawdowns: &[f64], idx: usize) -> f64 {
        let current_dd = drawdowns[idx];
        // Normalize: 0% = no stress, 20%+ = high stress
        (current_dd * 5.0).min(1.0)
    }

    /// Component 4: Volatility of volatility stress
    fn vol_of_vol_stress(&self, returns: &[f64], idx: usize) -> f64 {
        if idx < self.config.long_vol_period + self.config.short_vol_period {
            return 0.0;
        }

        // Calculate rolling volatilities
        let mut vols = Vec::new();
        let start = idx + 1 - self.config.long_vol_period;

        for i in start..=idx {
            if i >= self.config.short_vol_period {
                let vol_window = &returns[(i + 1 - self.config.short_vol_period)..=i];
                vols.push(Self::volatility(vol_window));
            }
        }

        if vols.len() < 10 {
            return 0.0;
        }

        // Vol of vol
        let vov = Self::volatility(&vols);
        let mean_vol = vols.iter().sum::<f64>() / vols.len() as f64;

        if mean_vol > 1e-10 {
            // Coefficient of variation of volatility
            (vov / mean_vol).min(1.0)
        } else {
            0.0
        }
    }

    /// Component 5: Tail stress (frequency of extreme moves)
    fn tail_stress(&self, returns: &[f64], idx: usize) -> f64 {
        if idx < self.config.period {
            return 0.0;
        }

        let window = &returns[(idx + 1 - self.config.period)..=idx];
        let mean = window.iter().sum::<f64>() / window.len() as f64;
        let std = Self::volatility(window);

        if std < 1e-10 {
            return 0.0;
        }

        // Count returns beyond 2 standard deviations
        let tail_count = window
            .iter()
            .filter(|&&r| (r - mean).abs() > 2.0 * std)
            .count();

        // Expected frequency under normal: ~5%
        let tail_frequency = tail_count as f64 / window.len() as f64;

        // Normalize: 5% = no excess, 15%+ = high tail stress
        ((tail_frequency - 0.05) / 0.10).max(0.0).min(1.0)
    }

    /// Calculate composite stress index
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let returns = Self::calculate_returns(close);
        let drawdowns = Self::calculate_drawdown(close);
        let mut result = vec![f64::NAN; self.config.period];

        // Component weights
        let w_vol = 0.25;
        let w_momentum = 0.20;
        let w_drawdown = 0.25;
        let w_vov = 0.15;
        let w_tail = 0.15;

        for i in self.config.period..returns.len() {
            let vol_stress = self.volatility_stress(&returns, i);
            let mom_stress = self.momentum_stress(&returns, i);
            let dd_stress = self.drawdown_stress(&drawdowns, i + 1); // +1 for price index
            let vov_stress = self.vol_of_vol_stress(&returns, i);
            let tail_stress = self.tail_stress(&returns, i);

            let composite = w_vol * vol_stress
                + w_momentum * mom_stress
                + w_drawdown * dd_stress
                + w_vov * vov_stress
                + w_tail * tail_stress;

            // Scale to 0-100
            result.push(composite * 100.0);
        }

        // Pad to match original length
        while result.len() < n {
            result.push(f64::NAN);
        }

        result
    }

    /// Calculate individual stress components (for detailed analysis)
    pub fn calculate_components(&self, close: &[f64]) -> StressComponents {
        let n = close.len();
        if n < self.config.period + 1 {
            return StressComponents {
                volatility: vec![f64::NAN; n],
                momentum: vec![f64::NAN; n],
                drawdown: vec![f64::NAN; n],
                vol_of_vol: vec![f64::NAN; n],
                tail: vec![f64::NAN; n],
            };
        }

        let returns = Self::calculate_returns(close);
        let drawdowns = Self::calculate_drawdown(close);

        let mut volatility = vec![f64::NAN; self.config.period];
        let mut momentum = vec![f64::NAN; self.config.period];
        let mut drawdown = vec![f64::NAN; self.config.period];
        let mut vol_of_vol = vec![f64::NAN; self.config.period];
        let mut tail = vec![f64::NAN; self.config.period];

        for i in self.config.period..returns.len() {
            volatility.push(self.volatility_stress(&returns, i) * 100.0);
            momentum.push(self.momentum_stress(&returns, i) * 100.0);
            drawdown.push(self.drawdown_stress(&drawdowns, i + 1) * 100.0);
            vol_of_vol.push(self.vol_of_vol_stress(&returns, i) * 100.0);
            tail.push(self.tail_stress(&returns, i) * 100.0);
        }

        // Pad to match original length
        while volatility.len() < n {
            volatility.push(f64::NAN);
            momentum.push(f64::NAN);
            drawdown.push(f64::NAN);
            vol_of_vol.push(f64::NAN);
            tail.push(f64::NAN);
        }

        StressComponents {
            volatility,
            momentum,
            drawdown,
            vol_of_vol,
            tail,
        }
    }

    /// Calculate stress regime (categorized stress level)
    pub fn calculate_regime(&self, close: &[f64]) -> Vec<StressRegime> {
        self.calculate(close)
            .into_iter()
            .map(|stress| {
                if stress.is_nan() {
                    StressRegime::Unknown
                } else if stress < 20.0 {
                    StressRegime::Low
                } else if stress < 40.0 {
                    StressRegime::Moderate
                } else if stress < 60.0 {
                    StressRegime::Elevated
                } else if stress < 80.0 {
                    StressRegime::High
                } else {
                    StressRegime::Extreme
                }
            })
            .collect()
    }
}

/// Individual stress components
#[derive(Debug, Clone)]
pub struct StressComponents {
    /// Volatility regime stress (0-100)
    pub volatility: Vec<f64>,
    /// Momentum stress (0-100)
    pub momentum: Vec<f64>,
    /// Drawdown stress (0-100)
    pub drawdown: Vec<f64>,
    /// Volatility of volatility stress (0-100)
    pub vol_of_vol: Vec<f64>,
    /// Tail stress (0-100)
    pub tail: Vec<f64>,
}

/// Stress regime categorization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StressRegime {
    /// Unknown/insufficient data
    Unknown,
    /// Low stress (0-20)
    Low,
    /// Moderate stress (20-40)
    Moderate,
    /// Elevated stress (40-60)
    Elevated,
    /// High stress (60-80)
    High,
    /// Extreme stress (80-100)
    Extreme,
}

impl TechnicalIndicator for StressIndex {
    fn name(&self) -> &str {
        "Stress Index"
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

    fn generate_stressed_prices(n: usize) -> Vec<f64> {
        let mut prices = generate_test_prices(n, 0.01);
        // Add a stress period with high volatility and drawdown
        if n > 200 {
            for i in 150..180 {
                prices[i] *= 0.98; // Consistent decline
            }
        }
        prices
    }

    #[test]
    fn test_stress_index_creation() {
        let si = StressIndex::new(252, 10, 60, 20);
        assert!(si.is_ok());

        let si = StressIndex::new(50, 10, 60, 20);
        assert!(si.is_err());

        let si = StressIndex::new(252, 3, 60, 20);
        assert!(si.is_err());

        let si = StressIndex::new(252, 10, 300, 20);
        assert!(si.is_err());

        let si = StressIndex::new(252, 50, 40, 20);
        assert!(si.is_err());
    }

    #[test]
    fn test_stress_index_basic() {
        let si = StressIndex::new(100, 10, 60, 20).unwrap();
        let prices = generate_test_prices(300, 0.02);

        let result = si.calculate(&prices);

        assert_eq!(result.len(), prices.len());
        assert!(result[50].is_nan()); // Warm-up period
        assert!(!result[200].is_nan()); // After warm-up
        // Stress should be in valid range
        assert!(result[200] >= 0.0 && result[200] <= 100.0);
    }

    #[test]
    fn test_stress_index_during_stress() {
        let si = StressIndex::new(100, 10, 60, 20).unwrap();
        let prices_calm = generate_test_prices(300, 0.01);
        let prices_stressed = generate_stressed_prices(300);

        let stress_calm = si.calculate(&prices_calm);
        let stress_stressed = si.calculate(&prices_stressed);

        // Stressed series should have higher stress after the stress period
        let avg_calm: f64 = stress_calm[180..220].iter().filter(|x| !x.is_nan()).sum::<f64>()
            / stress_calm[180..220].iter().filter(|x| !x.is_nan()).count().max(1) as f64;
        let avg_stressed: f64 = stress_stressed[180..220].iter().filter(|x| !x.is_nan()).sum::<f64>()
            / stress_stressed[180..220].iter().filter(|x| !x.is_nan()).count().max(1) as f64;

        assert!(avg_stressed >= avg_calm);
    }

    #[test]
    fn test_stress_components() {
        let si = StressIndex::new(100, 10, 60, 20).unwrap();
        let prices = generate_test_prices(300, 0.02);

        let components = si.calculate_components(&prices);

        // All components should have same length
        assert_eq!(components.volatility.len(), prices.len());
        assert_eq!(components.momentum.len(), prices.len());
        assert_eq!(components.drawdown.len(), prices.len());
        assert_eq!(components.vol_of_vol.len(), prices.len());
        assert_eq!(components.tail.len(), prices.len());

        // Check valid ranges for non-NaN values
        for i in 150..200 {
            if !components.volatility[i].is_nan() {
                assert!(components.volatility[i] >= 0.0 && components.volatility[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_stress_regime() {
        let si = StressIndex::new(100, 10, 60, 20).unwrap();
        let prices = generate_test_prices(300, 0.02);

        let regimes = si.calculate_regime(&prices);

        assert_eq!(regimes.len(), prices.len());

        // Check regime categorization
        assert_eq!(regimes[50], StressRegime::Unknown); // Warm-up period
        assert_ne!(regimes[200], StressRegime::Unknown); // After warm-up
    }

    #[test]
    fn test_stress_index_indicator_trait() {
        let si = StressIndex::new(252, 10, 60, 20).unwrap();

        assert_eq!(si.name(), "Stress Index");
        assert_eq!(si.min_periods(), 253);
        assert_eq!(si.output_features(), 1);
    }

    #[test]
    fn test_stress_index_default() {
        let si = StressIndex::default_indicator();
        assert_eq!(si.config.period, 252);
        assert_eq!(si.config.short_vol_period, 10);
        assert_eq!(si.config.long_vol_period, 60);
        assert_eq!(si.config.correlation_period, 20);
    }

    #[test]
    fn test_volatility_calculation() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02];
        let vol = StressIndex::volatility(&returns);
        assert!(vol > 0.0);
        assert!(vol < 0.1); // Reasonable range for these returns
    }

    #[test]
    fn test_drawdown_calculation() {
        let prices = vec![100.0, 105.0, 103.0, 108.0, 100.0];
        let drawdowns = StressIndex::calculate_drawdown(&prices);

        assert_eq!(drawdowns.len(), prices.len());
        assert_eq!(drawdowns[0], 0.0); // Start with no drawdown
        assert!(drawdowns[4] > 0.0); // Final price is below peak
    }
}
