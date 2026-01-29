//! Quality Factor implementation (IND-258).
//!
//! Composite quality factor based on ROE, debt ratios, and earnings stability.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Quality Factor configuration.
#[derive(Debug, Clone)]
pub struct QualityFactorConfig {
    /// Period for calculating quality metrics.
    pub period: usize,
    /// Weight for profitability component (ROE proxy).
    pub profitability_weight: f64,
    /// Weight for stability component (earnings stability).
    pub stability_weight: f64,
    /// Weight for leverage component (debt proxy).
    pub leverage_weight: f64,
    /// Whether to normalize output to 0-100 scale.
    pub normalize: bool,
}

impl Default for QualityFactorConfig {
    fn default() -> Self {
        Self {
            period: 60,
            profitability_weight: 0.40,
            stability_weight: 0.35,
            leverage_weight: 0.25,
            normalize: true,
        }
    }
}

/// Quality Factor (IND-258)
///
/// Calculates a composite quality factor based on:
/// - Profitability (ROE proxy using price returns)
/// - Earnings stability (volatility of returns)
/// - Leverage (implied from price behavior)
///
/// Since fundamental data may not always be available, this indicator
/// uses price-based proxies to estimate quality characteristics.
///
/// # Calculation
/// 1. Profitability proxy: Rolling return on price (momentum adjusted)
/// 2. Stability proxy: Inverse of return volatility (lower vol = higher quality)
/// 3. Leverage proxy: Drawdown severity (lower drawdown = lower leverage risk)
/// 4. Combine components with configurable weights
///
/// # Interpretation
/// - Higher values indicate higher quality stocks
/// - Quality factors: consistent earnings, low debt, high ROE
/// - Quality premium: historically quality stocks outperform
#[derive(Debug, Clone)]
pub struct QualityFactor {
    config: QualityFactorConfig,
}

impl QualityFactor {
    /// Create a new QualityFactor with default configuration.
    pub fn new() -> Self {
        Self {
            config: QualityFactorConfig::default(),
        }
    }

    /// Create a new QualityFactor with the specified period.
    ///
    /// # Arguments
    /// * `period` - Period for quality calculation
    pub fn with_period(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            config: QualityFactorConfig {
                period,
                ..Default::default()
            },
        })
    }

    /// Create a new QualityFactor with full configuration.
    ///
    /// # Arguments
    /// * `config` - Full configuration options
    pub fn with_config(config: QualityFactorConfig) -> Result<Self> {
        if config.period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }

        let total_weight =
            config.profitability_weight + config.stability_weight + config.leverage_weight;
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(IndicatorError::InvalidParameter {
                name: "weights".to_string(),
                reason: "weights must sum to 1.0".to_string(),
            });
        }

        Ok(Self { config })
    }

    /// Calculate quality factor from price and volume data.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    /// * `high` - Slice of high prices
    /// * `low` - Slice of low prices
    ///
    /// # Returns
    /// Vector of quality factor values.
    pub fn calculate(&self, close: &[f64], high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period || high.len() != n || low.len() != n {
            return vec![f64::NAN; n];
        }

        // Calculate returns
        let mut returns = vec![0.0; n];
        for i in 1..n {
            if close[i - 1].abs() > 1e-10 {
                returns[i] = (close[i] - close[i - 1]) / close[i - 1];
            }
        }

        let mut result = vec![f64::NAN; n];

        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;

            // 1. Profitability proxy: Risk-adjusted return (Sharpe-like)
            let window_returns = &returns[start..=i];
            let mean_return = window_returns.iter().sum::<f64>() / window_returns.len() as f64;
            let profitability = mean_return * 252.0; // Annualized return as ROE proxy

            // 2. Stability proxy: Inverse of return volatility
            let variance = window_returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / (window_returns.len() - 1) as f64;
            let volatility = variance.sqrt() * (252.0_f64).sqrt();
            let stability = if volatility > 1e-10 {
                1.0 / (1.0 + volatility) // Higher stability for lower vol
            } else {
                1.0
            };

            // 3. Leverage proxy: Drawdown severity (lower is better)
            let mut max_price = close[start];
            let mut max_drawdown = 0.0;
            for j in start..=i {
                if close[j] > max_price {
                    max_price = close[j];
                }
                let drawdown = (max_price - close[j]) / max_price;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
            }
            let leverage_score = 1.0 - max_drawdown; // Higher score for lower drawdown

            // Combine components
            let quality_score = self.config.profitability_weight * profitability
                + self.config.stability_weight * stability
                + self.config.leverage_weight * leverage_score;

            result[i] = quality_score;
        }

        // Normalize if enabled
        if self.config.normalize {
            result = self.normalize_to_scale(&result);
        }

        result
    }

    /// Calculate quality factor with fundamental data.
    ///
    /// # Arguments
    /// * `roe` - Return on equity values
    /// * `debt_ratio` - Debt to equity ratios
    /// * `earnings_growth` - Earnings growth rates
    ///
    /// # Returns
    /// Vector of quality factor values.
    pub fn calculate_with_fundamentals(
        &self,
        roe: &[f64],
        debt_ratio: &[f64],
        earnings_growth: &[f64],
    ) -> Vec<f64> {
        let n = roe.len();
        if n < self.config.period || debt_ratio.len() != n || earnings_growth.len() != n {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;

            // Profitability: Average ROE
            let avg_roe =
                roe[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
            let profitability_score = (avg_roe / 0.20).min(1.0).max(0.0); // Normalize to 20% benchmark

            // Stability: Consistency of earnings growth
            let earnings_window = &earnings_growth[start..=i];
            let mean_growth = earnings_window.iter().sum::<f64>() / earnings_window.len() as f64;
            let growth_variance = earnings_window
                .iter()
                .map(|g| (g - mean_growth).powi(2))
                .sum::<f64>()
                / (earnings_window.len() - 1) as f64;
            let stability_score = 1.0 / (1.0 + growth_variance.sqrt());

            // Leverage: Inverse of debt ratio
            let avg_debt =
                debt_ratio[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
            let leverage_score = 1.0 / (1.0 + avg_debt);

            // Combine components
            let quality_score = self.config.profitability_weight * profitability_score
                + self.config.stability_weight * stability_score
                + self.config.leverage_weight * leverage_score;

            result[i] = quality_score;
        }

        // Normalize if enabled
        if self.config.normalize {
            result = self.normalize_to_scale(&result);
        }

        result
    }

    /// Normalize values to 0-100 scale.
    fn normalize_to_scale(&self, data: &[f64]) -> Vec<f64> {
        let valid_values: Vec<f64> = data.iter().filter(|v| !v.is_nan()).cloned().collect();

        if valid_values.is_empty() {
            return data.to_vec();
        }

        let min_val = valid_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = valid_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range < 1e-10 {
            return data.to_vec();
        }

        data.iter()
            .map(|&v| {
                if v.is_nan() {
                    f64::NAN
                } else {
                    ((v - min_val) / range) * 100.0
                }
            })
            .collect()
    }
}

impl Default for QualityFactor {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for QualityFactor {
    fn name(&self) -> &str {
        "Quality Factor"
    }

    fn min_periods(&self) -> usize {
        self.config.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close, &data.high, &data.low);
        Ok(IndicatorOutput::single(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.1).sin() * 2.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|&c| c * 1.01).collect();
        let low: Vec<f64> = close.iter().map(|&c| c * 0.99).collect();
        (close, high, low)
    }

    #[test]
    fn test_quality_factor_basic() {
        let factor = QualityFactor::with_period(20).unwrap();
        let (close, high, low) = generate_test_data(100);
        let result = factor.calculate(&close, &high, &low);

        assert_eq!(result.len(), 100);
        // First 19 values should be NaN (warm-up period)
        for i in 0..19 {
            assert!(result[i].is_nan());
        }
        // Values after warm-up should be valid
        assert!(!result[20].is_nan());
    }

    #[test]
    fn test_quality_factor_normalized() {
        let factor = QualityFactor::new();
        let (close, high, low) = generate_test_data(100);
        let result = factor.calculate(&close, &high, &low);

        // Check that normalized values are in 0-100 range
        for &v in result.iter().filter(|v| !v.is_nan()) {
            assert!(v >= 0.0 && v <= 100.0, "Value {} out of range", v);
        }
    }

    #[test]
    fn test_quality_factor_with_fundamentals() {
        let factor = QualityFactor::with_period(10).unwrap();

        // Simulate fundamental data
        let roe: Vec<f64> = (0..50).map(|i| 0.15 + (i as f64) * 0.001).collect();
        let debt_ratio: Vec<f64> = (0..50).map(|i| 0.5 - (i as f64) * 0.005).collect();
        let earnings_growth: Vec<f64> = (0..50).map(|i| 0.10 + (i as f64) * 0.002).collect();

        let result = factor.calculate_with_fundamentals(&roe, &debt_ratio, &earnings_growth);

        assert_eq!(result.len(), 50);
        assert!(!result[20].is_nan());
    }

    #[test]
    fn test_quality_factor_custom_weights() {
        let config = QualityFactorConfig {
            period: 20,
            profitability_weight: 0.50,
            stability_weight: 0.30,
            leverage_weight: 0.20,
            normalize: false,
        };
        let factor = QualityFactor::with_config(config).unwrap();
        let (close, high, low) = generate_test_data(50);
        let result = factor.calculate(&close, &high, &low);

        assert!(!result[25].is_nan());
    }

    #[test]
    fn test_quality_factor_invalid_weights() {
        let config = QualityFactorConfig {
            period: 20,
            profitability_weight: 0.50,
            stability_weight: 0.50,
            leverage_weight: 0.50, // Sum > 1
            normalize: true,
        };
        let result = QualityFactor::with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_quality_factor_invalid_period() {
        let result = QualityFactor::with_period(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_quality_factor_insufficient_data() {
        let factor = QualityFactor::with_period(30).unwrap();
        let (close, high, low) = generate_test_data(20);
        let result = factor.calculate(&close, &high, &low);

        // All values should be NaN
        assert!(result.iter().all(|v| v.is_nan()));
    }
}
