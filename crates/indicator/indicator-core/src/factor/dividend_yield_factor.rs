//! Dividend Yield Factor implementation (IND-260).
//!
//! Yield ranking factor for dividend-focused investing.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Dividend Yield Factor configuration.
#[derive(Debug, Clone)]
pub struct DividendYieldFactorConfig {
    /// Period for calculating dividend yield metrics.
    pub period: usize,
    /// Period for ranking/percentile calculation.
    pub ranking_period: usize,
    /// Whether to use trailing or forward yield estimate.
    pub use_trailing: bool,
    /// Smoothing period for yield calculation.
    pub smoothing_period: usize,
}

impl Default for DividendYieldFactorConfig {
    fn default() -> Self {
        Self {
            period: 252, // Annual for dividends
            ranking_period: 252,
            use_trailing: true,
            smoothing_period: 20,
        }
    }
}

/// Dividend Yield Factor (IND-260)
///
/// Calculates a dividend yield factor for yield-based ranking.
/// When actual dividend data is not available, uses price stability
/// and income-proxy metrics.
///
/// # Calculation
/// 1. If dividend data available: Yield = Annual Dividend / Price
/// 2. If not: Use price stability proxy (lower volatility = income-like)
/// 3. Rank yield relative to historical values
/// 4. Apply smoothing to reduce noise
///
/// # Interpretation
/// - Higher scores indicate higher dividend yield
/// - Dividend yield factor captures income premium
/// - High-yield stocks often provide defensive characteristics
#[derive(Debug, Clone)]
pub struct DividendYieldFactor {
    config: DividendYieldFactorConfig,
}

impl DividendYieldFactor {
    /// Create a new DividendYieldFactor with default configuration.
    pub fn new() -> Self {
        Self {
            config: DividendYieldFactorConfig::default(),
        }
    }

    /// Create a new DividendYieldFactor with the specified period.
    ///
    /// # Arguments
    /// * `period` - Period for yield calculation
    pub fn with_period(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self {
            config: DividendYieldFactorConfig {
                period,
                ..Default::default()
            },
        })
    }

    /// Create a new DividendYieldFactor with full configuration.
    ///
    /// # Arguments
    /// * `config` - Full configuration options
    pub fn with_config(config: DividendYieldFactorConfig) -> Result<Self> {
        if config.period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if config.smoothing_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Calculate dividend yield factor with actual dividend data.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    /// * `dividends` - Slice of dividend payments (can be 0 for no dividend periods)
    ///
    /// # Returns
    /// Vector of dividend yield factor values (percentile rank).
    pub fn calculate_with_dividends(&self, close: &[f64], dividends: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period || dividends.len() != n {
            return vec![f64::NAN; n];
        }

        // Calculate trailing dividend yield
        let mut yields = vec![f64::NAN; n];

        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;
            // Sum dividends over trailing period
            let annual_dividend: f64 = dividends[start..=i].iter().sum();
            if close[i].abs() > 1e-10 {
                yields[i] = (annual_dividend / close[i]) * 100.0; // As percentage
            }
        }

        // Apply smoothing
        let smoothed = self.apply_smoothing(&yields, self.config.smoothing_period);

        // Calculate percentile rank
        self.calc_percentile_rank(&smoothed, self.config.ranking_period)
    }

    /// Calculate dividend yield factor using price proxy (when dividend data unavailable).
    ///
    /// Uses a combination of:
    /// - Price stability (low volatility suggests dividend payer)
    /// - Mean reversion tendency (income stocks tend to revert)
    /// - Relative price level (value stocks often pay dividends)
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of dividend yield proxy factor values.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period {
            return vec![f64::NAN; n];
        }

        // Calculate returns
        let mut returns = vec![0.0; n];
        for i in 1..n {
            if close[i - 1].abs() > 1e-10 {
                returns[i] = (close[i] - close[i - 1]) / close[i - 1];
            }
        }

        let mut yield_proxy = vec![f64::NAN; n];

        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;
            let window_returns = &returns[start..=i];

            // Component 1: Low volatility score (inverse of vol)
            let mean_return = window_returns.iter().sum::<f64>() / window_returns.len() as f64;
            let variance = window_returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / (window_returns.len() - 1) as f64;
            let volatility = variance.sqrt();
            let vol_score = if volatility > 1e-10 {
                1.0 / (1.0 + volatility * 10.0)
            } else {
                1.0
            };

            // Component 2: Mean reversion score
            let window_close = &close[start..=i];
            let mean_price = window_close.iter().sum::<f64>() / window_close.len() as f64;
            let price_deviation = (close[i] - mean_price) / mean_price;
            let reversion_score = 1.0 / (1.0 + price_deviation.abs());

            // Component 3: Value score (price relative to range)
            let max_price = window_close.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_price = window_close.iter().cloned().fold(f64::INFINITY, f64::min);
            let range = max_price - min_price;
            let value_score = if range > 1e-10 {
                1.0 - (close[i] - min_price) / range // Lower in range = higher value score
            } else {
                0.5
            };

            // Combine components (weighted average)
            yield_proxy[i] = 0.4 * vol_score + 0.3 * reversion_score + 0.3 * value_score;
        }

        // Apply smoothing
        let smoothed = self.apply_smoothing(&yield_proxy, self.config.smoothing_period);

        // Calculate percentile rank and scale to 0-100
        let ranked = self.calc_percentile_rank(&smoothed, self.config.ranking_period);

        ranked
    }

    /// Calculate yield score with dividend and price data.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    /// * `high` - Slice of high prices
    /// * `low` - Slice of low prices
    ///
    /// # Returns
    /// Vector of yield factor values.
    pub fn calculate_with_ohlc(&self, close: &[f64], high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period || high.len() != n || low.len() != n {
            return vec![f64::NAN; n];
        }

        // Use price range and stability as yield proxy
        let mut yield_proxy = vec![f64::NAN; n];

        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;

            // Calculate average daily range (lower = more stable = income-like)
            let mut total_range = 0.0;
            for j in start..=i {
                if close[j].abs() > 1e-10 {
                    total_range += (high[j] - low[j]) / close[j];
                }
            }
            let avg_range = total_range / self.config.period as f64;
            let range_score = 1.0 / (1.0 + avg_range * 50.0);

            // Calculate price trend (negative trend often indicates value/income)
            let price_change = (close[i] - close[start]) / close[start];
            let trend_score = 0.5 - (price_change * 2.0).min(0.5).max(-0.5);

            // Combine scores
            yield_proxy[i] = 0.6 * range_score + 0.4 * trend_score;
        }

        // Apply smoothing
        let smoothed = self.apply_smoothing(&yield_proxy, self.config.smoothing_period);

        // Calculate percentile rank
        self.calc_percentile_rank(&smoothed, self.config.ranking_period)
    }

    /// Apply simple moving average smoothing.
    fn apply_smoothing(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if period <= 1 || n < period {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; n];
        for i in (period - 1)..n {
            let start = i + 1 - period;
            let valid: Vec<f64> = data[start..=i].iter().filter(|v| !v.is_nan()).cloned().collect();
            if !valid.is_empty() {
                result[i] = valid.iter().sum::<f64>() / valid.len() as f64;
            }
        }

        result
    }

    /// Calculate percentile rank.
    fn calc_percentile_rank(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in (period - 1)..n {
            let start = i + 1 - period;
            let current = data[i];

            if current.is_nan() {
                continue;
            }

            let window: Vec<f64> = data[start..=i].iter().filter(|v| !v.is_nan()).cloned().collect();

            if window.is_empty() {
                continue;
            }

            let count_below = window.iter().filter(|&&v| v < current).count();
            result[i] = (count_below as f64 / window.len() as f64) * 100.0;
        }

        result
    }
}

impl Default for DividendYieldFactor {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for DividendYieldFactor {
    fn name(&self) -> &str {
        "Dividend Yield Factor"
    }

    fn min_periods(&self) -> usize {
        self.config.period.max(self.config.ranking_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }

        let values = self.calculate_with_ohlc(&data.close, &data.high, &data.low);
        Ok(IndicatorOutput::single(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + ((i as f64) * 0.05).sin() * 5.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|&c| c * 1.01).collect();
        let low: Vec<f64> = close.iter().map(|&c| c * 0.99).collect();
        (close, high, low)
    }

    #[test]
    fn test_dividend_yield_basic() {
        let factor = DividendYieldFactor::with_period(50).unwrap();
        let (close, _, _) = generate_test_data(200);
        let result = factor.calculate(&close);

        assert_eq!(result.len(), 200);
        // Should have valid values after warm-up
        assert!(!result[150].is_nan());
    }

    #[test]
    fn test_dividend_yield_with_dividends() {
        let factor = DividendYieldFactor::with_period(50).unwrap();
        let (close, _, _) = generate_test_data(200);

        // Simulate quarterly dividends
        let mut dividends = vec![0.0; 200];
        for i in (0..200).step_by(63) {
            // Quarterly
            if i < 200 {
                dividends[i] = 0.50; // $0.50 dividend
            }
        }

        let result = factor.calculate_with_dividends(&close, &dividends);

        assert_eq!(result.len(), 200);
        assert!(!result[150].is_nan());
    }

    #[test]
    fn test_dividend_yield_with_ohlc() {
        let factor = DividendYieldFactor::with_period(50).unwrap();
        let (close, high, low) = generate_test_data(200);
        let result = factor.calculate_with_ohlc(&close, &high, &low);

        assert_eq!(result.len(), 200);
        assert!(!result[150].is_nan());
    }

    #[test]
    fn test_dividend_yield_percentile_range() {
        let factor = DividendYieldFactor::with_period(30).unwrap();
        let (close, high, low) = generate_test_data(100);
        let result = factor.calculate_with_ohlc(&close, &high, &low);

        // Values should be in 0-100 range (percentile)
        for &v in result.iter().filter(|v| !v.is_nan()) {
            assert!(v >= 0.0 && v <= 100.0, "Value {} out of range", v);
        }
    }

    #[test]
    fn test_dividend_yield_smoothing() {
        let config = DividendYieldFactorConfig {
            period: 30,
            ranking_period: 50,
            use_trailing: true,
            smoothing_period: 10,
        };
        let factor = DividendYieldFactor::with_config(config).unwrap();
        let (close, _, _) = generate_test_data(100);
        let result = factor.calculate(&close);

        assert!(!result[60].is_nan());
    }

    #[test]
    fn test_dividend_yield_invalid_period() {
        let result = DividendYieldFactor::with_period(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_dividend_yield_insufficient_data() {
        let factor = DividendYieldFactor::with_period(100).unwrap();
        let (close, _, _) = generate_test_data(50);
        let result = factor.calculate(&close);

        // All values should be NaN
        assert!(result.iter().all(|v| v.is_nan()));
    }
}
