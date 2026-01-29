//! IV Rank (IND-246)
//!
//! Calculates IV percentile ranking over a specified lookback period,
//! typically one year (252 trading days).

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for IV Rank indicator.
#[derive(Debug, Clone)]
pub struct IVRankConfig {
    /// Period for IV calculation
    pub iv_period: usize,
    /// Lookback period for rank calculation (typically 252 for 1 year)
    pub rank_period: usize,
    /// Annualization factor
    pub annualization_factor: f64,
}

impl Default for IVRankConfig {
    fn default() -> Self {
        Self {
            iv_period: 20,
            rank_period: 252,
            annualization_factor: 252.0,
        }
    }
}

/// IV Rank (IND-246)
///
/// Measures where current implied volatility stands relative to its
/// historical range over the past year (or specified period).
///
/// # Formula
/// IV Rank = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV) * 100
///
/// # Interpretation
/// - 0-25: Low IV environment (options are cheap)
/// - 25-50: Below average IV
/// - 50-75: Above average IV
/// - 75-100: High IV environment (options are expensive)
#[derive(Debug, Clone)]
pub struct IVRank {
    config: IVRankConfig,
}

impl IVRank {
    /// Create a new IVRank with default one-year lookback.
    pub fn new(rank_period: usize) -> Result<Self> {
        if rank_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "rank_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            config: IVRankConfig {
                rank_period,
                ..Default::default()
            },
        })
    }

    /// Create from configuration.
    pub fn from_config(config: IVRankConfig) -> Result<Self> {
        if config.iv_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "iv_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if config.rank_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "rank_period".to_string(),
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

    /// Calculate IV Rank values.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let iv = self.calculate_iv(close);
        let mut result = vec![f64::NAN; n];

        let warmup = self.config.iv_period + self.config.rank_period - 1;
        if n < warmup {
            return result;
        }

        for i in warmup..n {
            let start = i.saturating_sub(self.config.rank_period - 1);
            let window = &iv[start..=i];

            // Find min and max IV in window (skip NaN values)
            let valid_ivs: Vec<f64> = window.iter()
                .filter(|v| !v.is_nan())
                .copied()
                .collect();

            if valid_ivs.len() < 2 {
                continue;
            }

            let min_iv = valid_ivs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_iv = valid_ivs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            if !iv[i].is_nan() && max_iv > min_iv {
                result[i] = (iv[i] - min_iv) / (max_iv - min_iv) * 100.0;
            } else if !iv[i].is_nan() {
                result[i] = 50.0;  // Neutral when no range
            }
        }

        result
    }

    /// Calculate with pre-computed IV values.
    pub fn calculate_from_iv(&self, iv: &[f64]) -> Vec<f64> {
        let n = iv.len();
        let mut result = vec![f64::NAN; n];

        if n < self.config.rank_period {
            return result;
        }

        for i in (self.config.rank_period - 1)..n {
            let start = i.saturating_sub(self.config.rank_period - 1);
            let window = &iv[start..=i];

            let valid_ivs: Vec<f64> = window.iter()
                .filter(|v| !v.is_nan())
                .copied()
                .collect();

            if valid_ivs.len() < 2 {
                continue;
            }

            let min_iv = valid_ivs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_iv = valid_ivs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            if !iv[i].is_nan() && max_iv > min_iv {
                result[i] = (iv[i] - min_iv) / (max_iv - min_iv) * 100.0;
            } else if !iv[i].is_nan() {
                result[i] = 50.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for IVRank {
    fn name(&self) -> &str {
        "IV Rank"
    }

    fn min_periods(&self) -> usize {
        self.config.iv_period + self.config.rank_period
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
        // Generate price data with varying volatility
        (0..300)
            .map(|i| {
                let trend = 100.0 + (i as f64) * 0.1;
                let vol_factor = if i < 100 { 1.0 } else if i < 200 { 2.0 } else { 0.5 };
                trend + (i as f64 * 0.5).sin() * vol_factor * 2.0
            })
            .collect()
    }

    #[test]
    fn test_iv_rank_basic() {
        let close = make_test_data();
        let iv_rank = IVRank::new(252).unwrap();
        let result = iv_rank.calculate(&close);

        assert_eq!(result.len(), close.len());

        // Check values after warmup are in [0, 100]
        for i in 270..result.len() {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 100.0,
                    "IV Rank {} out of range at index {}", result[i], i);
            }
        }
    }

    #[test]
    fn test_iv_rank_from_iv() {
        // Create synthetic IV data
        let iv: Vec<f64> = (0..100)
            .map(|i| 20.0 + (i as f64 * 0.1).sin() * 10.0)
            .collect();

        let config = IVRankConfig {
            iv_period: 20,
            rank_period: 50,
            annualization_factor: 252.0,
        };
        let iv_rank = IVRank::from_config(config).unwrap();
        let result = iv_rank.calculate_from_iv(&iv);

        assert_eq!(result.len(), iv.len());

        // Check values are in range
        for i in 50..result.len() {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_iv_rank_low_volatility() {
        // Constant low volatility should give low rank
        let close: Vec<f64> = (0..300)
            .map(|i| 100.0 + (i as f64) * 0.01)  // Very low volatility
            .collect();

        let iv_rank = IVRank::new(50).unwrap();
        let result = iv_rank.calculate(&close);

        // Most values should be around 50 (neutral) due to minimal variance
        let valid_values: Vec<f64> = result.iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();

        assert!(!valid_values.is_empty());
    }

    #[test]
    fn test_iv_rank_config() {
        let config = IVRankConfig {
            iv_period: 10,
            rank_period: 126,
            annualization_factor: 252.0,
        };
        let iv_rank = IVRank::from_config(config).unwrap();
        assert_eq!(iv_rank.min_periods(), 136);
    }
}
