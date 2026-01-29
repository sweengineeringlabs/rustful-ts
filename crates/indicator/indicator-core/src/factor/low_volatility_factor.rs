//! Low Volatility Factor implementation (IND-259).
//!
//! Historical volatility ranking factor for low-volatility investing.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Low Volatility Factor configuration.
#[derive(Debug, Clone)]
pub struct LowVolatilityFactorConfig {
    /// Period for calculating volatility.
    pub volatility_period: usize,
    /// Period for ranking/normalization.
    pub ranking_period: usize,
    /// Annualization factor (252 for daily, 52 for weekly).
    pub annualization_factor: f64,
    /// Volatility measure to use.
    pub volatility_type: VolatilityType,
    /// Whether to invert (higher score = lower vol).
    pub invert_score: bool,
}

/// Type of volatility measure to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolatilityType {
    /// Standard deviation of returns.
    StandardDeviation,
    /// Average True Range based.
    ATR,
    /// Parkinson (high-low based).
    Parkinson,
    /// Garman-Klass (OHLC based).
    GarmanKlass,
}

impl Default for LowVolatilityFactorConfig {
    fn default() -> Self {
        Self {
            volatility_period: 20,
            ranking_period: 252,
            annualization_factor: 252.0,
            volatility_type: VolatilityType::StandardDeviation,
            invert_score: true,
        }
    }
}

/// Low Volatility Factor (IND-259)
///
/// Calculates a low volatility factor based on historical volatility ranking.
/// Lower volatility stocks historically provide better risk-adjusted returns
/// (low volatility anomaly).
///
/// # Calculation
/// 1. Calculate rolling volatility using chosen method
/// 2. Optionally annualize the volatility
/// 3. Rank volatility relative to historical values
/// 4. Invert score so higher = lower volatility (if enabled)
///
/// # Interpretation
/// - Higher scores indicate lower volatility (defensive stocks)
/// - Lower scores indicate higher volatility (aggressive stocks)
/// - Low volatility premium: low-vol stocks often outperform high-vol on risk-adjusted basis
#[derive(Debug, Clone)]
pub struct LowVolatilityFactor {
    config: LowVolatilityFactorConfig,
}

impl LowVolatilityFactor {
    /// Create a new LowVolatilityFactor with default configuration.
    pub fn new() -> Self {
        Self {
            config: LowVolatilityFactorConfig::default(),
        }
    }

    /// Create a new LowVolatilityFactor with the specified volatility period.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for volatility calculation
    pub fn with_period(volatility_period: usize) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            config: LowVolatilityFactorConfig {
                volatility_period,
                ..Default::default()
            },
        })
    }

    /// Create a new LowVolatilityFactor with full configuration.
    ///
    /// # Arguments
    /// * `config` - Full configuration options
    pub fn with_config(config: LowVolatilityFactorConfig) -> Result<Self> {
        if config.volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if config.ranking_period < config.volatility_period {
            return Err(IndicatorError::InvalidParameter {
                name: "ranking_period".to_string(),
                reason: "must be at least volatility_period".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Calculate the low volatility factor.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    /// * `high` - Slice of high prices
    /// * `low` - Slice of low prices
    ///
    /// # Returns
    /// Vector of low volatility factor values.
    pub fn calculate(&self, close: &[f64], high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.volatility_period || high.len() != n || low.len() != n {
            return vec![f64::NAN; n];
        }

        // Calculate volatility based on type
        let volatility = match self.config.volatility_type {
            VolatilityType::StandardDeviation => self.calc_std_volatility(close),
            VolatilityType::ATR => self.calc_atr_volatility(close, high, low),
            VolatilityType::Parkinson => self.calc_parkinson_volatility(high, low),
            VolatilityType::GarmanKlass => self.calc_garman_klass_volatility(close, high, low),
        };

        // Calculate percentile rank
        let mut result = self.calc_percentile_rank(&volatility, self.config.ranking_period);

        // Invert if configured (so higher = lower volatility)
        if self.config.invert_score {
            result = result.iter().map(|&v| if v.is_nan() { v } else { 100.0 - v }).collect();
        }

        result
    }

    /// Calculate standard deviation based volatility.
    fn calc_std_volatility(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        // Calculate returns
        let mut returns = vec![0.0; n];
        for i in 1..n {
            if close[i - 1].abs() > 1e-10 {
                returns[i] = (close[i] - close[i - 1]) / close[i - 1];
            }
        }

        // Calculate rolling standard deviation
        for i in self.config.volatility_period..n {
            let start = i + 1 - self.config.volatility_period;
            let window = &returns[start..=i];
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance =
                window.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (window.len() - 1) as f64;
            let std_dev = variance.sqrt();
            // Annualize
            result[i] = std_dev * self.config.annualization_factor.sqrt();
        }

        result
    }

    /// Calculate ATR-based volatility.
    fn calc_atr_volatility(&self, close: &[f64], high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut true_range = vec![0.0; n];

        // Calculate True Range
        true_range[0] = high[0] - low[0];
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            true_range[i] = hl.max(hc).max(lc);
        }

        // Calculate rolling average (ATR)
        let mut result = vec![f64::NAN; n];
        for i in (self.config.volatility_period - 1)..n {
            let start = i + 1 - self.config.volatility_period;
            let atr = true_range[start..=i].iter().sum::<f64>() / self.config.volatility_period as f64;
            // Normalize by price for comparability
            if close[i].abs() > 1e-10 {
                result[i] = (atr / close[i]) * 100.0 * self.config.annualization_factor.sqrt();
            }
        }

        result
    }

    /// Calculate Parkinson volatility (high-low based).
    fn calc_parkinson_volatility(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut log_hl = vec![0.0; n];

        // Calculate log(high/low)^2
        for i in 0..n {
            if low[i] > 0.0 {
                log_hl[i] = (high[i] / low[i]).ln().powi(2);
            }
        }

        // Calculate rolling Parkinson volatility
        let factor = 1.0 / (4.0 * 2.0_f64.ln());
        let mut result = vec![f64::NAN; n];

        for i in (self.config.volatility_period - 1)..n {
            let start = i + 1 - self.config.volatility_period;
            let avg = log_hl[start..=i].iter().sum::<f64>() / self.config.volatility_period as f64;
            let parkinson = (factor * avg).sqrt();
            // Annualize
            result[i] = parkinson * self.config.annualization_factor.sqrt();
        }

        result
    }

    /// Calculate Garman-Klass volatility.
    fn calc_garman_klass_volatility(&self, close: &[f64], high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut gk_values = vec![0.0; n];

        // Calculate GK variance components
        for i in 1..n {
            if close[i - 1] > 0.0 && low[i] > 0.0 {
                let log_hl = (high[i] / low[i]).ln();
                let log_co = (close[i] / close[i - 1]).ln();
                gk_values[i] =
                    0.5 * log_hl.powi(2) - (2.0 * 2.0_f64.ln() - 1.0) * log_co.powi(2);
            }
        }

        // Calculate rolling GK volatility
        let mut result = vec![f64::NAN; n];
        for i in self.config.volatility_period..n {
            let start = i + 1 - self.config.volatility_period;
            let avg = gk_values[start..=i].iter().sum::<f64>() / self.config.volatility_period as f64;
            let gk_vol = if avg > 0.0 { avg.sqrt() } else { 0.0 };
            // Annualize
            result[i] = gk_vol * self.config.annualization_factor.sqrt();
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

impl Default for LowVolatilityFactor {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for LowVolatilityFactor {
    fn name(&self) -> &str {
        "Low Volatility Factor"
    }

    fn min_periods(&self) -> usize {
        self.config.ranking_period.max(self.config.volatility_period)
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

    fn generate_test_data(n: usize, volatility: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + ((i as f64) * 0.1).sin() * volatility)
            .collect();
        let high: Vec<f64> = close.iter().map(|&c| c * (1.0 + volatility * 0.01)).collect();
        let low: Vec<f64> = close.iter().map(|&c| c * (1.0 - volatility * 0.01)).collect();
        (close, high, low)
    }

    #[test]
    fn test_low_volatility_basic() {
        let factor = LowVolatilityFactor::with_period(10).unwrap();
        let (close, high, low) = generate_test_data(100, 5.0);
        let result = factor.calculate(&close, &high, &low);

        assert_eq!(result.len(), 100);
        // Should have valid values after warm-up
        assert!(!result[50].is_nan());
    }

    #[test]
    fn test_low_volatility_inverted_score() {
        let config = LowVolatilityFactorConfig {
            volatility_period: 10,
            ranking_period: 50,
            annualization_factor: 252.0,
            volatility_type: VolatilityType::StandardDeviation,
            invert_score: true,
        };
        let factor = LowVolatilityFactor::with_config(config).unwrap();
        let (close, high, low) = generate_test_data(100, 5.0);
        let result = factor.calculate(&close, &high, &low);

        // Scores should be in 0-100 range
        for &v in result.iter().filter(|v| !v.is_nan()) {
            assert!(v >= 0.0 && v <= 100.0);
        }
    }

    #[test]
    fn test_low_volatility_atr_type() {
        let config = LowVolatilityFactorConfig {
            volatility_period: 14,
            ranking_period: 50,
            annualization_factor: 252.0,
            volatility_type: VolatilityType::ATR,
            invert_score: true,
        };
        let factor = LowVolatilityFactor::with_config(config).unwrap();
        let (close, high, low) = generate_test_data(100, 5.0);
        let result = factor.calculate(&close, &high, &low);

        assert!(!result[60].is_nan());
    }

    #[test]
    fn test_low_volatility_parkinson_type() {
        let config = LowVolatilityFactorConfig {
            volatility_period: 20,
            ranking_period: 50,
            annualization_factor: 252.0,
            volatility_type: VolatilityType::Parkinson,
            invert_score: true,
        };
        let factor = LowVolatilityFactor::with_config(config).unwrap();
        let (close, high, low) = generate_test_data(100, 5.0);
        let result = factor.calculate(&close, &high, &low);

        assert!(!result[60].is_nan());
    }

    #[test]
    fn test_low_volatility_garman_klass_type() {
        let config = LowVolatilityFactorConfig {
            volatility_period: 20,
            ranking_period: 50,
            annualization_factor: 252.0,
            volatility_type: VolatilityType::GarmanKlass,
            invert_score: true,
        };
        let factor = LowVolatilityFactor::with_config(config).unwrap();
        let (close, high, low) = generate_test_data(100, 5.0);
        let result = factor.calculate(&close, &high, &low);

        assert!(!result[60].is_nan());
    }

    #[test]
    fn test_low_volatility_high_vs_low_vol() {
        let config = LowVolatilityFactorConfig {
            volatility_period: 10,
            ranking_period: 30,
            annualization_factor: 252.0,
            volatility_type: VolatilityType::StandardDeviation,
            invert_score: false, // Raw volatility rank
        };

        let factor = LowVolatilityFactor::with_config(config).unwrap();

        // Low volatility data
        let (close_low, high_low, low_low) = generate_test_data(50, 1.0);
        let result_low = factor.calculate(&close_low, &high_low, &low_low);

        // High volatility data
        let (close_high, high_high, low_high) = generate_test_data(50, 10.0);
        let result_high = factor.calculate(&close_high, &high_high, &low_high);

        // Both should produce valid results
        assert!(!result_low[40].is_nan());
        assert!(!result_high[40].is_nan());
    }

    #[test]
    fn test_low_volatility_invalid_period() {
        let result = LowVolatilityFactor::with_period(2);
        assert!(result.is_err());
    }

    #[test]
    fn test_low_volatility_invalid_ranking_period() {
        let config = LowVolatilityFactorConfig {
            volatility_period: 20,
            ranking_period: 10, // Less than volatility_period
            annualization_factor: 252.0,
            volatility_type: VolatilityType::StandardDeviation,
            invert_score: true,
        };
        let result = LowVolatilityFactor::with_config(config);
        assert!(result.is_err());
    }
}
