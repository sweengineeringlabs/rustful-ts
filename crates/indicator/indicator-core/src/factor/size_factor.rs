//! Size Factor implementation (IND-257).
//!
//! Market capitalization ranking factor for factor-based investing.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Size Factor configuration.
#[derive(Debug, Clone)]
pub struct SizeFactorConfig {
    /// Period for calculating average market cap.
    pub period: usize,
    /// Whether to use log transformation for market cap.
    pub use_log: bool,
    /// Z-score normalization period (0 to disable).
    pub zscore_period: usize,
}

impl Default for SizeFactorConfig {
    fn default() -> Self {
        Self {
            period: 20,
            use_log: true,
            zscore_period: 252,
        }
    }
}

/// Size Factor (IND-257)
///
/// Calculates a size factor based on market capitalization proxy.
/// Since direct market cap data may not always be available, this indicator
/// uses price * volume as a proxy for market cap/liquidity.
///
/// # Calculation
/// 1. Calculate market cap proxy: price * volume (or use provided market cap)
/// 2. Optionally apply log transformation for normalization
/// 3. Calculate rolling average over the specified period
/// 4. Optionally apply z-score normalization
///
/// # Interpretation
/// - Higher values indicate larger market cap (large-cap)
/// - Lower values indicate smaller market cap (small-cap)
/// - Can be used for size-based portfolio construction
/// - Small-cap premium: historically small-caps outperform large-caps
#[derive(Debug, Clone)]
pub struct SizeFactor {
    config: SizeFactorConfig,
}

impl SizeFactor {
    /// Create a new SizeFactor with default configuration.
    pub fn new() -> Self {
        Self {
            config: SizeFactorConfig::default(),
        }
    }

    /// Create a new SizeFactor with the specified period.
    ///
    /// # Arguments
    /// * `period` - Period for calculating average market cap proxy
    pub fn with_period(period: usize) -> Result<Self> {
        if period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            config: SizeFactorConfig {
                period,
                ..Default::default()
            },
        })
    }

    /// Create a new SizeFactor with full configuration.
    ///
    /// # Arguments
    /// * `config` - Full configuration options
    pub fn with_config(config: SizeFactorConfig) -> Result<Self> {
        if config.period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Calculate the size factor values using price and volume as market cap proxy.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    /// * `volume` - Slice of volume data
    ///
    /// # Returns
    /// Vector of size factor values.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period || volume.len() != n {
            return vec![f64::NAN; n];
        }

        // Calculate market cap proxy (price * volume)
        let mut market_cap_proxy: Vec<f64> = close
            .iter()
            .zip(volume.iter())
            .map(|(p, v)| {
                let cap = p * v;
                if self.config.use_log && cap > 0.0 {
                    cap.ln()
                } else {
                    cap
                }
            })
            .collect();

        // Calculate rolling average
        let mut result = vec![f64::NAN; self.config.period - 1];
        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;
            let sum: f64 = market_cap_proxy[start..=i].iter().sum();
            let avg = sum / self.config.period as f64;
            result.push(avg);
        }

        // Apply z-score normalization if enabled
        if self.config.zscore_period > 0 && n >= self.config.zscore_period {
            result = self.apply_zscore(&result, self.config.zscore_period);
        }

        result
    }

    /// Calculate size factor with explicit market cap data.
    ///
    /// # Arguments
    /// * `market_cap` - Slice of market capitalization values
    ///
    /// # Returns
    /// Vector of size factor values.
    pub fn calculate_with_market_cap(&self, market_cap: &[f64]) -> Vec<f64> {
        let n = market_cap.len();
        if n < self.config.period {
            return vec![f64::NAN; n];
        }

        // Apply log transformation if enabled
        let transformed: Vec<f64> = market_cap
            .iter()
            .map(|&cap| {
                if self.config.use_log && cap > 0.0 {
                    cap.ln()
                } else {
                    cap
                }
            })
            .collect();

        // Calculate rolling average
        let mut result = vec![f64::NAN; self.config.period - 1];
        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;
            let sum: f64 = transformed[start..=i].iter().sum();
            let avg = sum / self.config.period as f64;
            result.push(avg);
        }

        // Apply z-score normalization if enabled
        if self.config.zscore_period > 0 && n >= self.config.zscore_period {
            result = self.apply_zscore(&result, self.config.zscore_period);
        }

        result
    }

    /// Apply z-score normalization to a series.
    fn apply_zscore(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in (period - 1)..n {
            let start = i + 1 - period;
            let window: Vec<f64> = data[start..=i]
                .iter()
                .filter(|v| !v.is_nan())
                .cloned()
                .collect();

            if window.len() < 2 {
                continue;
            }

            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance =
                window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (window.len() - 1) as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 && !data[i].is_nan() {
                result[i] = (data[i] - mean) / std_dev;
            }
        }

        result
    }
}

impl Default for SizeFactor {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for SizeFactor {
    fn name(&self) -> &str {
        "Size Factor"
    }

    fn min_periods(&self) -> usize {
        self.config.period.max(self.config.zscore_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let volume: Vec<f64> = (0..n).map(|i| 1_000_000.0 + (i as f64) * 10_000.0).collect();
        (close, volume)
    }

    #[test]
    fn test_size_factor_basic() {
        let factor = SizeFactor::with_period(10).unwrap();
        let (close, volume) = generate_test_data(50);
        let result = factor.calculate(&close, &volume);

        assert_eq!(result.len(), 50);
        // First 9 values should be NaN (warm-up period)
        for i in 0..9 {
            assert!(result[i].is_nan());
        }
        // Values after warm-up should be valid
        assert!(!result[10].is_nan());
    }

    #[test]
    fn test_size_factor_increasing_cap() {
        let config = SizeFactorConfig {
            period: 5,
            use_log: false,
            zscore_period: 0,
        };
        let factor = SizeFactor::with_config(config).unwrap();
        let (close, volume) = generate_test_data(20);
        let result = factor.calculate(&close, &volume);

        // Values should be increasing as market cap proxy increases
        for i in 6..result.len() {
            if !result[i].is_nan() && !result[i - 1].is_nan() {
                assert!(result[i] > result[i - 1], "Expected increasing values");
            }
        }
    }

    #[test]
    fn test_size_factor_with_market_cap() {
        let factor = SizeFactor::with_period(5).unwrap();
        let market_cap: Vec<f64> = (0..30)
            .map(|i| 1_000_000_000.0 + (i as f64) * 10_000_000.0)
            .collect();
        let result = factor.calculate_with_market_cap(&market_cap);

        assert_eq!(result.len(), 30);
        assert!(!result[10].is_nan());
    }

    #[test]
    fn test_size_factor_log_transformation() {
        let config_log = SizeFactorConfig {
            period: 5,
            use_log: true,
            zscore_period: 0,
        };
        let config_no_log = SizeFactorConfig {
            period: 5,
            use_log: false,
            zscore_period: 0,
        };
        let factor_log = SizeFactor::with_config(config_log).unwrap();
        let factor_no_log = SizeFactor::with_config(config_no_log).unwrap();
        let (close, volume) = generate_test_data(20);

        let result_log = factor_log.calculate(&close, &volume);
        let result_no_log = factor_no_log.calculate(&close, &volume);

        // Log-transformed values should be smaller
        for i in 5..20 {
            if !result_log[i].is_nan() && !result_no_log[i].is_nan() {
                assert!(result_log[i] < result_no_log[i]);
            }
        }
    }

    #[test]
    fn test_size_factor_insufficient_data() {
        let factor = SizeFactor::with_period(20).unwrap();
        let (close, volume) = generate_test_data(10);
        let result = factor.calculate(&close, &volume);

        // All values should be NaN
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_size_factor_invalid_period() {
        let result = SizeFactor::with_period(0);
        assert!(result.is_err());
    }
}
