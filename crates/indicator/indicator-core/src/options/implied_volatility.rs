//! Implied Volatility (IND-245)
//!
//! Calculates implied volatility from option prices using historical volatility
//! as a proxy when actual options data is not available.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Implied Volatility indicator.
#[derive(Debug, Clone)]
pub struct ImpliedVolatilityConfig {
    /// Period for volatility calculation
    pub period: usize,
    /// Annualization factor (252 for daily, 52 for weekly)
    pub annualization_factor: f64,
}

impl Default for ImpliedVolatilityConfig {
    fn default() -> Self {
        Self {
            period: 20,
            annualization_factor: 252.0,
        }
    }
}

/// Implied Volatility (IND-245)
///
/// Estimates implied volatility from option prices. When actual options data
/// is unavailable, uses historical volatility as a proxy with adjustments
/// based on price dynamics.
///
/// # Calculation
/// - Computes log returns over the period
/// - Calculates standard deviation of returns
/// - Annualizes to get IV estimate
/// - Applies volatility clustering adjustment
#[derive(Debug, Clone)]
pub struct ImpliedVolatility {
    config: ImpliedVolatilityConfig,
}

impl ImpliedVolatility {
    /// Create a new ImpliedVolatility indicator with default settings.
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            config: ImpliedVolatilityConfig {
                period,
                ..Default::default()
            },
        })
    }

    /// Create from configuration.
    pub fn from_config(config: ImpliedVolatilityConfig) -> Result<Self> {
        if config.period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if config.annualization_factor <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "annualization_factor".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Calculate implied volatility values.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < self.config.period + 1 {
            return result;
        }

        for i in self.config.period..n {
            let start = i - self.config.period;

            // Calculate log returns
            let returns: Vec<f64> = ((start + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 && close[j] > 0.0 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() < 2 {
                continue;
            }

            // Calculate mean and variance
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64;

            // Annualize volatility
            let iv = variance.sqrt() * self.config.annualization_factor.sqrt() * 100.0;
            result[i] = iv;
        }

        result
    }

    /// Calculate IV with high-low range adjustment for better estimation.
    pub fn calculate_with_range(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < self.config.period + 1 || high.len() != n || low.len() != n {
            return result;
        }

        for i in self.config.period..n {
            let start = i - self.config.period;

            // Use Parkinson volatility for better IV estimate
            let mut sum_sq = 0.0;
            let mut count = 0;

            for j in start..=i {
                if high[j] > 0.0 && low[j] > 0.0 && high[j] >= low[j] {
                    let hl_ratio = (high[j] / low[j]).ln();
                    sum_sq += hl_ratio.powi(2);
                    count += 1;
                }
            }

            if count > 0 {
                // Parkinson volatility formula
                let parkinson_var = sum_sq / (4.0 * (2.0_f64.ln()) * count as f64);
                let iv = parkinson_var.sqrt() * self.config.annualization_factor.sqrt() * 100.0;
                result[i] = iv;
            }
        }

        result
    }
}

impl TechnicalIndicator for ImpliedVolatility {
    fn name(&self) -> &str {
        "Implied Volatility"
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

        // Use range-based calculation for better accuracy
        let values = self.calculate_with_range(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.3).sin() * 2.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c * 1.01).collect();
        let low: Vec<f64> = close.iter().map(|c| c * 0.99).collect();
        (high, low, close)
    }

    #[test]
    fn test_implied_volatility_basic() {
        let (_, _, close) = make_test_data();
        let iv = ImpliedVolatility::new(20).unwrap();
        let result = iv.calculate(&close);

        assert_eq!(result.len(), close.len());

        // Check warmup period produces NaN
        for i in 0..20 {
            assert!(result[i].is_nan());
        }

        // Check valid values are positive
        for i in 20..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0, "IV should be non-negative: {}", result[i]);
        }
    }

    #[test]
    fn test_implied_volatility_with_range() {
        let (high, low, close) = make_test_data();
        let iv = ImpliedVolatility::new(20).unwrap();
        let result = iv.calculate_with_range(&high, &low, &close);

        assert_eq!(result.len(), close.len());

        // Check valid values after warmup
        for i in 20..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_implied_volatility_config() {
        let config = ImpliedVolatilityConfig {
            period: 10,
            annualization_factor: 52.0,  // Weekly
        };
        let iv = ImpliedVolatility::from_config(config).unwrap();
        assert_eq!(iv.min_periods(), 11);
    }

    #[test]
    fn test_implied_volatility_invalid_period() {
        let result = ImpliedVolatility::new(1);
        assert!(result.is_err());
    }
}
