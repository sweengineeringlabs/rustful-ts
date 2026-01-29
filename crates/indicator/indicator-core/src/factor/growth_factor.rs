//! Growth Factor implementation (IND-261).
//!
//! Earnings and revenue growth ranking factor for growth investing.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Growth Factor configuration.
#[derive(Debug, Clone)]
pub struct GrowthFactorConfig {
    /// Short-term growth period (e.g., quarterly).
    pub short_period: usize,
    /// Long-term growth period (e.g., annual).
    pub long_period: usize,
    /// Period for ranking calculation.
    pub ranking_period: usize,
    /// Weight for short-term growth.
    pub short_weight: f64,
    /// Weight for long-term growth.
    pub long_weight: f64,
    /// Weight for growth consistency.
    pub consistency_weight: f64,
}

impl Default for GrowthFactorConfig {
    fn default() -> Self {
        Self {
            short_period: 63,   // ~Quarterly
            long_period: 252,   // ~Annual
            ranking_period: 252,
            short_weight: 0.35,
            long_weight: 0.40,
            consistency_weight: 0.25,
        }
    }
}

/// Growth Factor (IND-261)
///
/// Calculates a growth factor based on earnings and revenue growth rates.
/// When fundamental data is unavailable, uses price momentum as a proxy.
///
/// # Calculation
/// 1. Calculate short-term growth rate
/// 2. Calculate long-term growth rate
/// 3. Measure growth consistency (lower variance = more consistent)
/// 4. Combine with configurable weights
/// 5. Rank relative to historical values
///
/// # Interpretation
/// - Higher scores indicate stronger growth characteristics
/// - Growth premium: growth stocks tend to outperform during expansions
/// - Balance between momentum and sustainability
#[derive(Debug, Clone)]
pub struct GrowthFactor {
    config: GrowthFactorConfig,
}

impl GrowthFactor {
    /// Create a new GrowthFactor with default configuration.
    pub fn new() -> Self {
        Self {
            config: GrowthFactorConfig::default(),
        }
    }

    /// Create a new GrowthFactor with specified periods.
    ///
    /// # Arguments
    /// * `short_period` - Period for short-term growth
    /// * `long_period` - Period for long-term growth
    pub fn with_periods(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self {
            config: GrowthFactorConfig {
                short_period,
                long_period,
                ..Default::default()
            },
        })
    }

    /// Create a new GrowthFactor with full configuration.
    ///
    /// # Arguments
    /// * `config` - Full configuration options
    pub fn with_config(config: GrowthFactorConfig) -> Result<Self> {
        if config.short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if config.long_period <= config.short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }

        let total_weight =
            config.short_weight + config.long_weight + config.consistency_weight;
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(IndicatorError::InvalidParameter {
                name: "weights".to_string(),
                reason: "weights must sum to 1.0".to_string(),
            });
        }

        Ok(Self { config })
    }

    /// Calculate growth factor with fundamental earnings data.
    ///
    /// # Arguments
    /// * `earnings` - Slice of earnings per share values
    /// * `revenue` - Slice of revenue values
    ///
    /// # Returns
    /// Vector of growth factor values (percentile rank).
    pub fn calculate_with_fundamentals(&self, earnings: &[f64], revenue: &[f64]) -> Vec<f64> {
        let n = earnings.len();
        if n < self.config.long_period || revenue.len() != n {
            return vec![f64::NAN; n];
        }

        let mut growth_scores = vec![f64::NAN; n];

        for i in (self.config.long_period - 1)..n {
            // Short-term earnings growth
            let short_start = i + 1 - self.config.short_period;
            let short_earnings_growth = if earnings[short_start].abs() > 1e-10 {
                (earnings[i] - earnings[short_start]) / earnings[short_start].abs()
            } else {
                0.0
            };

            // Long-term earnings growth
            let long_start = i + 1 - self.config.long_period;
            let long_earnings_growth = if earnings[long_start].abs() > 1e-10 {
                (earnings[i] - earnings[long_start]) / earnings[long_start].abs()
            } else {
                0.0
            };

            // Short-term revenue growth
            let short_revenue_growth = if revenue[short_start].abs() > 1e-10 {
                (revenue[i] - revenue[short_start]) / revenue[short_start].abs()
            } else {
                0.0
            };

            // Long-term revenue growth
            let long_revenue_growth = if revenue[long_start].abs() > 1e-10 {
                (revenue[i] - revenue[long_start]) / revenue[long_start].abs()
            } else {
                0.0
            };

            // Combine earnings and revenue growth
            let short_growth = (short_earnings_growth + short_revenue_growth) / 2.0;
            let long_growth = (long_earnings_growth + long_revenue_growth) / 2.0;

            // Growth consistency (lower variance = higher score)
            let earnings_window = &earnings[long_start..=i];
            let mean = earnings_window.iter().sum::<f64>() / earnings_window.len() as f64;
            let variance = earnings_window
                .iter()
                .map(|e| (e - mean).powi(2))
                .sum::<f64>()
                / (earnings_window.len() - 1) as f64;
            let consistency = 1.0 / (1.0 + variance.sqrt() / mean.abs().max(1e-10));

            // Combine components
            growth_scores[i] = self.config.short_weight * short_growth
                + self.config.long_weight * long_growth
                + self.config.consistency_weight * consistency;
        }

        // Calculate percentile rank
        self.calc_percentile_rank(&growth_scores, self.config.ranking_period)
    }

    /// Calculate growth factor using price as proxy.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of growth factor values.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.long_period {
            return vec![f64::NAN; n];
        }

        let mut growth_scores = vec![f64::NAN; n];

        for i in (self.config.long_period - 1)..n {
            // Short-term momentum (price growth)
            let short_start = i + 1 - self.config.short_period;
            let short_growth = if close[short_start].abs() > 1e-10 {
                (close[i] - close[short_start]) / close[short_start]
            } else {
                0.0
            };

            // Long-term momentum
            let long_start = i + 1 - self.config.long_period;
            let long_growth = if close[long_start].abs() > 1e-10 {
                (close[i] - close[long_start]) / close[long_start]
            } else {
                0.0
            };

            // Growth consistency (smoothness of price increase)
            let window = &close[long_start..=i];
            let mut positive_periods = 0;
            let mut total_periods = 0;
            for j in 1..window.len() {
                if window[j] > window[j - 1] {
                    positive_periods += 1;
                }
                total_periods += 1;
            }
            let consistency = if total_periods > 0 {
                positive_periods as f64 / total_periods as f64
            } else {
                0.5
            };

            // Combine components
            growth_scores[i] = self.config.short_weight * short_growth
                + self.config.long_weight * long_growth
                + self.config.consistency_weight * consistency;
        }

        // Calculate percentile rank
        self.calc_percentile_rank(&growth_scores, self.config.ranking_period)
    }

    /// Calculate growth factor with volume confirmation.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    /// * `volume` - Slice of volume data
    ///
    /// # Returns
    /// Vector of growth factor values.
    pub fn calculate_with_volume(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.long_period || volume.len() != n {
            return vec![f64::NAN; n];
        }

        let mut growth_scores = vec![f64::NAN; n];

        for i in (self.config.long_period - 1)..n {
            // Price growth components
            let short_start = i + 1 - self.config.short_period;
            let short_growth = if close[short_start].abs() > 1e-10 {
                (close[i] - close[short_start]) / close[short_start]
            } else {
                0.0
            };

            let long_start = i + 1 - self.config.long_period;
            let long_growth = if close[long_start].abs() > 1e-10 {
                (close[i] - close[long_start]) / close[long_start]
            } else {
                0.0
            };

            // Volume growth (expanding volume confirms growth)
            let short_vol_avg = volume[short_start..=i].iter().sum::<f64>()
                / self.config.short_period as f64;
            let long_vol_avg = volume[long_start..=i].iter().sum::<f64>()
                / self.config.long_period as f64;
            let volume_expansion = if long_vol_avg > 1e-10 {
                (short_vol_avg / long_vol_avg - 1.0).max(-0.5).min(0.5)
            } else {
                0.0
            };

            // Growth consistency with volume confirmation
            let window_close = &close[long_start..=i];
            let window_vol = &volume[long_start..=i];
            let mut up_volume = 0.0;
            let mut down_volume = 0.0;
            for j in 1..window_close.len() {
                if window_close[j] > window_close[j - 1] {
                    up_volume += window_vol[j];
                } else {
                    down_volume += window_vol[j];
                }
            }
            let volume_consistency = if up_volume + down_volume > 1e-10 {
                up_volume / (up_volume + down_volume)
            } else {
                0.5
            };

            // Combine components (adjust weights for volume factor)
            let adj_short_weight = self.config.short_weight * 0.9;
            let adj_long_weight = self.config.long_weight * 0.9;
            let adj_consistency_weight = self.config.consistency_weight * 0.8;
            let volume_weight = 1.0 - adj_short_weight - adj_long_weight - adj_consistency_weight;

            growth_scores[i] = adj_short_weight * short_growth
                + adj_long_weight * long_growth
                + adj_consistency_weight * volume_consistency
                + volume_weight * volume_expansion;
        }

        // Calculate percentile rank
        self.calc_percentile_rank(&growth_scores, self.config.ranking_period)
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

impl Default for GrowthFactor {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for GrowthFactor {
    fn name(&self) -> &str {
        "Growth Factor"
    }

    fn min_periods(&self) -> usize {
        self.config.long_period.max(self.config.ranking_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }

        let values = self.calculate_with_volume(&data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_growth_data(n: usize, growth_rate: f64) -> Vec<f64> {
        (0..n)
            .map(|i| 100.0 * (1.0 + growth_rate).powf(i as f64 / 252.0))
            .collect()
    }

    fn generate_volume_data(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 1_000_000.0 + ((i as f64) * 0.1).sin() * 100_000.0)
            .collect()
    }

    #[test]
    fn test_growth_factor_basic() {
        let factor = GrowthFactor::with_periods(20, 60).unwrap();
        let close = generate_growth_data(200, 0.20); // 20% annual growth
        let result = factor.calculate(&close);

        assert_eq!(result.len(), 200);
        assert!(!result[150].is_nan());
    }

    #[test]
    fn test_growth_factor_with_volume() {
        let factor = GrowthFactor::with_periods(20, 60).unwrap();
        let close = generate_growth_data(200, 0.15);
        let volume = generate_volume_data(200);
        let result = factor.calculate_with_volume(&close, &volume);

        assert_eq!(result.len(), 200);
        assert!(!result[150].is_nan());
    }

    #[test]
    fn test_growth_factor_with_fundamentals() {
        let factor = GrowthFactor::with_periods(20, 60).unwrap();

        // Simulate fundamental data with growth
        let earnings: Vec<f64> = (0..200)
            .map(|i| 2.0 * (1.0 + 0.15_f64).powf(i as f64 / 252.0))
            .collect();
        let revenue: Vec<f64> = (0..200)
            .map(|i| 1000.0 * (1.0 + 0.12_f64).powf(i as f64 / 252.0))
            .collect();

        let result = factor.calculate_with_fundamentals(&earnings, &revenue);

        assert_eq!(result.len(), 200);
        assert!(!result[150].is_nan());
    }

    #[test]
    fn test_growth_factor_percentile_range() {
        let factor = GrowthFactor::with_periods(20, 60).unwrap();
        let close = generate_growth_data(200, 0.10);
        let result = factor.calculate(&close);

        // Values should be in 0-100 range
        for &v in result.iter().filter(|v| !v.is_nan()) {
            assert!(v >= 0.0 && v <= 100.0, "Value {} out of range", v);
        }
    }

    #[test]
    fn test_growth_factor_high_vs_low_growth() {
        let factor = GrowthFactor::with_periods(10, 30).unwrap();

        // High growth stock
        let high_growth = generate_growth_data(100, 0.50);
        let result_high = factor.calculate(&high_growth);

        // Low growth stock
        let low_growth = generate_growth_data(100, 0.05);
        let result_low = factor.calculate(&low_growth);

        // Both should produce valid results
        assert!(!result_high[80].is_nan());
        assert!(!result_low[80].is_nan());
    }

    #[test]
    fn test_growth_factor_custom_weights() {
        let config = GrowthFactorConfig {
            short_period: 20,
            long_period: 60,
            ranking_period: 100,
            short_weight: 0.50,
            long_weight: 0.30,
            consistency_weight: 0.20,
        };
        let factor = GrowthFactor::with_config(config).unwrap();
        let close = generate_growth_data(150, 0.15);
        let result = factor.calculate(&close);

        assert!(!result[120].is_nan());
    }

    #[test]
    fn test_growth_factor_invalid_periods() {
        let result = GrowthFactor::with_periods(30, 20); // long < short
        assert!(result.is_err());
    }

    #[test]
    fn test_growth_factor_invalid_weights() {
        let config = GrowthFactorConfig {
            short_period: 20,
            long_period: 60,
            ranking_period: 100,
            short_weight: 0.50,
            long_weight: 0.50,
            consistency_weight: 0.50, // Sum > 1
        };
        let result = GrowthFactor::with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_growth_factor_insufficient_data() {
        let factor = GrowthFactor::with_periods(20, 60).unwrap();
        let close = generate_growth_data(30, 0.15);
        let result = factor.calculate(&close);

        // All values should be NaN
        assert!(result.iter().all(|v| v.is_nan()));
    }
}
