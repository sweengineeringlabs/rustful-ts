//! Liquidity Factor implementation (IND-262).
//!
//! Trading volume and turnover ranking factor for liquidity-based investing.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Liquidity Factor configuration.
#[derive(Debug, Clone)]
pub struct LiquidityFactorConfig {
    /// Period for calculating liquidity metrics.
    pub period: usize,
    /// Period for ranking calculation.
    pub ranking_period: usize,
    /// Weight for volume component.
    pub volume_weight: f64,
    /// Weight for turnover component.
    pub turnover_weight: f64,
    /// Weight for bid-ask spread proxy.
    pub spread_weight: f64,
    /// Weight for price impact proxy.
    pub impact_weight: f64,
}

impl Default for LiquidityFactorConfig {
    fn default() -> Self {
        Self {
            period: 20,
            ranking_period: 252,
            volume_weight: 0.35,
            turnover_weight: 0.30,
            spread_weight: 0.20,
            impact_weight: 0.15,
        }
    }
}

/// Liquidity Factor (IND-262)
///
/// Calculates a liquidity factor based on trading volume, turnover ratio,
/// and market microstructure proxies.
///
/// # Calculation
/// 1. Volume component: Average daily volume relative to historical
/// 2. Turnover component: Volume * Price (dollar volume)
/// 3. Spread proxy: High-low range relative to close (Corwin-Schultz inspired)
/// 4. Price impact proxy: Return per unit volume (Amihud illiquidity inverse)
/// 5. Combine with configurable weights
///
/// # Interpretation
/// - Higher scores indicate higher liquidity
/// - Liquidity premium: less liquid stocks may offer higher returns
/// - Important for portfolio construction and execution costs
#[derive(Debug, Clone)]
pub struct LiquidityFactor {
    config: LiquidityFactorConfig,
}

impl LiquidityFactor {
    /// Create a new LiquidityFactor with default configuration.
    pub fn new() -> Self {
        Self {
            config: LiquidityFactorConfig::default(),
        }
    }

    /// Create a new LiquidityFactor with the specified period.
    ///
    /// # Arguments
    /// * `period` - Period for liquidity calculation
    pub fn with_period(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            config: LiquidityFactorConfig {
                period,
                ..Default::default()
            },
        })
    }

    /// Create a new LiquidityFactor with full configuration.
    ///
    /// # Arguments
    /// * `config` - Full configuration options
    pub fn with_config(config: LiquidityFactorConfig) -> Result<Self> {
        if config.period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }

        let total_weight = config.volume_weight
            + config.turnover_weight
            + config.spread_weight
            + config.impact_weight;
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(IndicatorError::InvalidParameter {
                name: "weights".to_string(),
                reason: "weights must sum to 1.0".to_string(),
            });
        }

        Ok(Self { config })
    }

    /// Calculate liquidity factor.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    /// * `high` - Slice of high prices
    /// * `low` - Slice of low prices
    /// * `volume` - Slice of volume data
    ///
    /// # Returns
    /// Vector of liquidity factor values (percentile rank).
    pub fn calculate(&self, close: &[f64], high: &[f64], low: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period || high.len() != n || low.len() != n || volume.len() != n {
            return vec![f64::NAN; n];
        }

        let mut liquidity_scores = vec![f64::NAN; n];

        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;

            // 1. Volume component: Average volume
            let avg_volume = volume[start..=i].iter().sum::<f64>() / self.config.period as f64;
            let volume_score = avg_volume.ln().max(0.0) / 20.0; // Normalize log volume

            // 2. Turnover component: Dollar volume (price * volume)
            let dollar_volume: f64 = (start..=i)
                .map(|j| close[j] * volume[j])
                .sum::<f64>()
                / self.config.period as f64;
            let turnover_score = dollar_volume.ln().max(0.0) / 25.0; // Normalize log dollar volume

            // 3. Spread proxy: High-low spread (Corwin-Schultz inspired)
            let mut total_spread = 0.0;
            for j in start..=i {
                if close[j].abs() > 1e-10 {
                    total_spread += (high[j] - low[j]) / close[j];
                }
            }
            let avg_spread = total_spread / self.config.period as f64;
            // Lower spread = higher liquidity, so invert
            let spread_score = 1.0 / (1.0 + avg_spread * 20.0);

            // 4. Price impact proxy: Amihud illiquidity measure (inverted)
            // |return| / dollar volume - lower is more liquid
            let mut amihud_sum = 0.0;
            let mut amihud_count = 0;
            for j in (start + 1)..=i {
                let dollar_vol = close[j] * volume[j];
                if dollar_vol > 1e-10 && close[j - 1].abs() > 1e-10 {
                    let ret = ((close[j] - close[j - 1]) / close[j - 1]).abs();
                    amihud_sum += ret / dollar_vol * 1e9; // Scale for readability
                    amihud_count += 1;
                }
            }
            let amihud = if amihud_count > 0 {
                amihud_sum / amihud_count as f64
            } else {
                0.0
            };
            // Lower Amihud = more liquid, so invert
            let impact_score = 1.0 / (1.0 + amihud * 100.0);

            // Combine components
            liquidity_scores[i] = self.config.volume_weight * volume_score
                + self.config.turnover_weight * turnover_score
                + self.config.spread_weight * spread_score
                + self.config.impact_weight * impact_score;
        }

        // Calculate percentile rank
        self.calc_percentile_rank(&liquidity_scores, self.config.ranking_period)
    }

    /// Calculate liquidity factor with just price and volume.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    /// * `volume` - Slice of volume data
    ///
    /// # Returns
    /// Vector of liquidity factor values.
    pub fn calculate_simple(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period || volume.len() != n {
            return vec![f64::NAN; n];
        }

        let mut liquidity_scores = vec![f64::NAN; n];

        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;

            // Volume score
            let avg_volume = volume[start..=i].iter().sum::<f64>() / self.config.period as f64;
            let volume_score = avg_volume.ln().max(0.0) / 20.0;

            // Dollar volume (turnover) score
            let dollar_volume: f64 = (start..=i)
                .map(|j| close[j] * volume[j])
                .sum::<f64>()
                / self.config.period as f64;
            let turnover_score = dollar_volume.ln().max(0.0) / 25.0;

            // Volume stability (consistent volume = more liquid)
            let mean_vol = avg_volume;
            let vol_variance = volume[start..=i]
                .iter()
                .map(|v| (v - mean_vol).powi(2))
                .sum::<f64>()
                / (self.config.period - 1) as f64;
            let vol_cv = if mean_vol > 1e-10 {
                vol_variance.sqrt() / mean_vol
            } else {
                1.0
            };
            let stability_score = 1.0 / (1.0 + vol_cv);

            // Amihud illiquidity (simplified)
            let mut amihud_sum = 0.0;
            let mut amihud_count = 0;
            for j in (start + 1)..=i {
                let dollar_vol = close[j] * volume[j];
                if dollar_vol > 1e-10 && close[j - 1].abs() > 1e-10 {
                    let ret = ((close[j] - close[j - 1]) / close[j - 1]).abs();
                    amihud_sum += ret / dollar_vol * 1e9;
                    amihud_count += 1;
                }
            }
            let amihud = if amihud_count > 0 {
                amihud_sum / amihud_count as f64
            } else {
                0.0
            };
            let impact_score = 1.0 / (1.0 + amihud * 100.0);

            // Combine (adjust weights for simplified version)
            liquidity_scores[i] = 0.35 * volume_score
                + 0.30 * turnover_score
                + 0.20 * stability_score
                + 0.15 * impact_score;
        }

        // Calculate percentile rank
        self.calc_percentile_rank(&liquidity_scores, self.config.ranking_period)
    }

    /// Calculate Amihud illiquidity measure directly.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    /// * `volume` - Slice of volume data
    ///
    /// # Returns
    /// Vector of Amihud illiquidity values (higher = less liquid).
    pub fn calculate_amihud(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period || volume.len() != n {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in self.config.period..n {
            let start = i + 1 - self.config.period;

            let mut amihud_sum = 0.0;
            let mut count = 0;

            for j in (start + 1)..=i {
                let dollar_vol = close[j] * volume[j];
                if dollar_vol > 1e-10 && close[j - 1].abs() > 1e-10 {
                    let ret = ((close[j] - close[j - 1]) / close[j - 1]).abs();
                    amihud_sum += ret / dollar_vol;
                    count += 1;
                }
            }

            if count > 0 {
                result[i] = amihud_sum / count as f64 * 1e6; // Scale for readability
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

impl Default for LiquidityFactor {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for LiquidityFactor {
    fn name(&self) -> &str {
        "Liquidity Factor"
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

        let values = self.calculate(&data.close, &data.high, &data.low, &data.volume);
        Ok(IndicatorOutput::single(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n: usize, base_volume: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + ((i as f64) * 0.05).sin() * 5.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|&c| c * 1.01).collect();
        let low: Vec<f64> = close.iter().map(|&c| c * 0.99).collect();
        let volume: Vec<f64> = (0..n)
            .map(|i| base_volume + ((i as f64) * 0.1).sin() * base_volume * 0.2)
            .collect();
        (close, high, low, volume)
    }

    #[test]
    fn test_liquidity_factor_basic() {
        let factor = LiquidityFactor::with_period(10).unwrap();
        let (close, high, low, volume) = generate_test_data(100, 1_000_000.0);
        let result = factor.calculate(&close, &high, &low, &volume);

        assert_eq!(result.len(), 100);
        // Should have valid values after warm-up
        assert!(!result[50].is_nan());
    }

    #[test]
    fn test_liquidity_factor_simple() {
        let factor = LiquidityFactor::with_period(10).unwrap();
        let (close, _, _, volume) = generate_test_data(100, 1_000_000.0);
        let result = factor.calculate_simple(&close, &volume);

        assert_eq!(result.len(), 100);
        assert!(!result[50].is_nan());
    }

    #[test]
    fn test_liquidity_factor_amihud() {
        let factor = LiquidityFactor::with_period(10).unwrap();
        let (close, _, _, volume) = generate_test_data(100, 1_000_000.0);
        let result = factor.calculate_amihud(&close, &volume);

        assert_eq!(result.len(), 100);
        assert!(!result[50].is_nan());
        // Amihud values should be positive
        for &v in result.iter().filter(|v| !v.is_nan()) {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_liquidity_factor_high_vs_low_volume() {
        let factor = LiquidityFactor::with_period(10).unwrap();

        // High volume (more liquid)
        let (close_h, high_h, low_h, volume_h) = generate_test_data(100, 10_000_000.0);
        let result_high = factor.calculate(&close_h, &high_h, &low_h, &volume_h);

        // Low volume (less liquid)
        let (close_l, high_l, low_l, volume_l) = generate_test_data(100, 100_000.0);
        let result_low = factor.calculate(&close_l, &high_l, &low_l, &volume_l);

        // Both should produce valid results
        assert!(!result_high[50].is_nan());
        assert!(!result_low[50].is_nan());
    }

    #[test]
    fn test_liquidity_factor_percentile_range() {
        let factor = LiquidityFactor::with_period(10).unwrap();
        let (close, high, low, volume) = generate_test_data(100, 1_000_000.0);
        let result = factor.calculate(&close, &high, &low, &volume);

        // Values should be in 0-100 range
        for &v in result.iter().filter(|v| !v.is_nan()) {
            assert!(v >= 0.0 && v <= 100.0, "Value {} out of range", v);
        }
    }

    #[test]
    fn test_liquidity_factor_custom_weights() {
        let config = LiquidityFactorConfig {
            period: 15,
            ranking_period: 50,
            volume_weight: 0.40,
            turnover_weight: 0.30,
            spread_weight: 0.15,
            impact_weight: 0.15,
        };
        let factor = LiquidityFactor::with_config(config).unwrap();
        let (close, high, low, volume) = generate_test_data(100, 1_000_000.0);
        let result = factor.calculate(&close, &high, &low, &volume);

        assert!(!result[60].is_nan());
    }

    #[test]
    fn test_liquidity_factor_invalid_period() {
        let result = LiquidityFactor::with_period(2);
        assert!(result.is_err());
    }

    #[test]
    fn test_liquidity_factor_invalid_weights() {
        let config = LiquidityFactorConfig {
            period: 10,
            ranking_period: 50,
            volume_weight: 0.40,
            turnover_weight: 0.40,
            spread_weight: 0.40,
            impact_weight: 0.40, // Sum > 1
        };
        let result = LiquidityFactor::with_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_liquidity_factor_insufficient_data() {
        let factor = LiquidityFactor::with_period(50).unwrap();
        let (close, high, low, volume) = generate_test_data(30, 1_000_000.0);
        let result = factor.calculate(&close, &high, &low, &volume);

        // All values should be NaN
        assert!(result.iter().all(|v| v.is_nan()));
    }
}
