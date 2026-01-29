//! IV Percentile (IND-247)
//!
//! Calculates the percentage of days when IV was lower than the current level
//! over a specified lookback period.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for IV Percentile indicator.
#[derive(Debug, Clone)]
pub struct IVPercentileConfig {
    /// Period for IV calculation
    pub iv_period: usize,
    /// Lookback period for percentile calculation
    pub percentile_period: usize,
    /// Annualization factor
    pub annualization_factor: f64,
}

impl Default for IVPercentileConfig {
    fn default() -> Self {
        Self {
            iv_period: 20,
            percentile_period: 252,
            annualization_factor: 252.0,
        }
    }
}

/// IV Percentile (IND-247)
///
/// Measures what percentage of days had IV lower than the current IV level.
/// Unlike IV Rank which uses min/max range, IV Percentile counts actual days.
///
/// # Formula
/// IV Percentile = (Number of days with IV < Current IV) / Total Days * 100
///
/// # Interpretation
/// - 0-20: Very low IV (cheap options)
/// - 20-40: Low IV
/// - 40-60: Average IV
/// - 60-80: High IV
/// - 80-100: Very high IV (expensive options)
///
/// # IV Rank vs IV Percentile
/// - IV Rank: Where current IV falls in the min-max range
/// - IV Percentile: Percentage of days with lower IV
#[derive(Debug, Clone)]
pub struct IVPercentile {
    config: IVPercentileConfig,
}

impl IVPercentile {
    /// Create a new IVPercentile indicator.
    pub fn new(percentile_period: usize) -> Result<Self> {
        if percentile_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "percentile_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            config: IVPercentileConfig {
                percentile_period,
                ..Default::default()
            },
        })
    }

    /// Create from configuration.
    pub fn from_config(config: IVPercentileConfig) -> Result<Self> {
        if config.iv_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "iv_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if config.percentile_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "percentile_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Calculate historical volatility as IV proxy.
    fn calculate_iv(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut iv = vec![f64::NAN; n];

        if n < self.config.iv_period + 1 {
            return iv;
        }

        for i in self.config.iv_period..n {
            let start = i - self.config.iv_period;

            let returns: Vec<f64> = ((start + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 && close[j] > 0.0 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 2 {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / (returns.len() - 1) as f64;

                iv[i] = variance.sqrt() * self.config.annualization_factor.sqrt() * 100.0;
            }
        }

        iv
    }

    /// Calculate IV Percentile values.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let iv = self.calculate_iv(close);
        self.calculate_from_iv(&iv)
    }

    /// Calculate percentile from pre-computed IV values.
    pub fn calculate_from_iv(&self, iv: &[f64]) -> Vec<f64> {
        let n = iv.len();
        let mut result = vec![f64::NAN; n];

        if n < self.config.percentile_period {
            return result;
        }

        for i in (self.config.percentile_period - 1)..n {
            let start = i.saturating_sub(self.config.percentile_period - 1);
            let current = iv[i];

            if current.is_nan() {
                continue;
            }

            // Count days with IV lower than current (excluding current day)
            let window = &iv[start..i];
            let valid_values: Vec<f64> = window.iter()
                .filter(|v| !v.is_nan())
                .copied()
                .collect();

            if valid_values.is_empty() {
                continue;
            }

            let count_below = valid_values.iter().filter(|&&v| v < current).count();
            result[i] = (count_below as f64 / valid_values.len() as f64) * 100.0;
        }

        result
    }

    /// Calculate with equal-or-less comparison (inclusive percentile).
    pub fn calculate_inclusive(&self, iv: &[f64]) -> Vec<f64> {
        let n = iv.len();
        let mut result = vec![f64::NAN; n];

        if n < self.config.percentile_period {
            return result;
        }

        for i in (self.config.percentile_period - 1)..n {
            let start = i.saturating_sub(self.config.percentile_period - 1);
            let current = iv[i];

            if current.is_nan() {
                continue;
            }

            let window = &iv[start..i];
            let valid_values: Vec<f64> = window.iter()
                .filter(|v| !v.is_nan())
                .copied()
                .collect();

            if valid_values.is_empty() {
                continue;
            }

            let count_below_or_equal = valid_values.iter()
                .filter(|&&v| v <= current)
                .count();
            result[i] = (count_below_or_equal as f64 / valid_values.len() as f64) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for IVPercentile {
    fn name(&self) -> &str {
        "IV Percentile"
    }

    fn min_periods(&self) -> usize {
        self.config.iv_period + self.config.percentile_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        (0..300)
            .map(|i| {
                let trend = 100.0 + (i as f64) * 0.1;
                let noise = (i as f64 * 0.3).sin() * 2.0;
                trend + noise
            })
            .collect()
    }

    #[test]
    fn test_iv_percentile_basic() {
        let close = make_test_data();
        let iv_pct = IVPercentile::new(100).unwrap();
        let result = iv_pct.calculate(&close);

        assert_eq!(result.len(), close.len());

        // Check values are in [0, 100]
        for i in 120..result.len() {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "IV Percentile {} out of range at index {}", result[i], i);
            }
        }
    }

    #[test]
    fn test_iv_percentile_from_iv() {
        // Create IV data with known distribution
        let iv: Vec<f64> = (0..100)
            .map(|i| i as f64)  // Linear 0-99
            .collect();

        let config = IVPercentileConfig {
            iv_period: 20,
            percentile_period: 50,
            annualization_factor: 252.0,
        };
        let iv_pct = IVPercentile::from_config(config).unwrap();
        let result = iv_pct.calculate_from_iv(&iv);

        // At index 99 (value=99), almost all previous values are lower
        // Percentile should be high
        assert!(!result[99].is_nan());
        assert!(result[99] > 90.0, "Expected high percentile, got {}", result[99]);
    }

    #[test]
    fn test_iv_percentile_low_value() {
        // IV at constant low value
        let mut iv: Vec<f64> = (0..100).map(|i| 20.0 + i as f64).collect();
        // Set last value to be the lowest
        iv[99] = 10.0;

        let iv_pct = IVPercentile::new(50).unwrap();
        let result = iv_pct.calculate_from_iv(&iv);

        // Should have 0 percentile (no values below)
        assert!(!result[99].is_nan());
        assert!(result[99] < 5.0, "Expected low percentile, got {}", result[99]);
    }

    #[test]
    fn test_iv_percentile_inclusive() {
        let iv: Vec<f64> = vec![10.0; 100];  // All same value

        let iv_pct = IVPercentile::new(50).unwrap();
        let exclusive = iv_pct.calculate_from_iv(&iv);
        let inclusive = iv_pct.calculate_inclusive(&iv);

        // Exclusive: no values strictly below, should be 0
        // Inclusive: all values equal or below, should be 100
        assert!(!exclusive[99].is_nan());
        assert!(exclusive[99] < 1.0);
        assert!(!inclusive[99].is_nan());
        assert!(inclusive[99] > 99.0);
    }

    #[test]
    fn test_iv_percentile_config() {
        let config = IVPercentileConfig {
            iv_period: 10,
            percentile_period: 126,
            annualization_factor: 252.0,
        };
        let iv_pct = IVPercentile::from_config(config).unwrap();
        assert_eq!(iv_pct.min_periods(), 136);
    }
}
