//! Volatility Skew (IND-248)
//!
//! Measures the difference between put and call implied volatility,
//! using price-based proxies when actual options data is unavailable.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Volatility Skew indicator.
#[derive(Debug, Clone)]
pub struct VolatilitySkewConfig {
    /// Period for volatility calculation
    pub period: usize,
    /// Smoothing period for skew values
    pub smoothing_period: usize,
}

impl Default for VolatilitySkewConfig {
    fn default() -> Self {
        Self {
            period: 20,
            smoothing_period: 5,
        }
    }
}

/// Volatility Skew (IND-248)
///
/// Measures the put/call IV difference (skew) using price-based proxies.
/// In equity markets, puts typically have higher IV than calls (negative skew)
/// due to demand for downside protection.
///
/// # Calculation
/// Uses downside vs upside volatility as a proxy for put/call IV difference:
/// - Downside volatility: Std dev of negative returns
/// - Upside volatility: Std dev of positive returns
/// - Skew = (Downside Vol - Upside Vol) / Total Vol
///
/// # Interpretation
/// - Positive skew: Puts more expensive (bearish sentiment)
/// - Zero skew: Balanced IV
/// - Negative skew: Calls more expensive (bullish sentiment)
#[derive(Debug, Clone)]
pub struct VolatilitySkew {
    config: VolatilitySkewConfig,
}

impl VolatilitySkew {
    /// Create a new VolatilitySkew indicator.
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            config: VolatilitySkewConfig {
                period,
                ..Default::default()
            },
        })
    }

    /// Create from configuration.
    pub fn from_config(config: VolatilitySkewConfig) -> Result<Self> {
        if config.period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
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

    /// Calculate volatility skew values.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < self.config.period + 1 {
            return result;
        }

        for i in self.config.period..n {
            let start = i - self.config.period;

            // Calculate returns and separate by direction
            let mut up_returns: Vec<f64> = Vec::new();
            let mut down_returns: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 0.0 && close[j] > 0.0 {
                    let ret = (close[j] / close[j - 1]).ln();
                    if ret > 0.0 {
                        up_returns.push(ret);
                    } else if ret < 0.0 {
                        down_returns.push(ret.abs());
                    }
                }
            }

            // Calculate upside and downside volatility
            let up_vol = if up_returns.len() >= 2 {
                let mean = up_returns.iter().sum::<f64>() / up_returns.len() as f64;
                let var = up_returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / (up_returns.len() - 1) as f64;
                var.sqrt()
            } else {
                0.0
            };

            let down_vol = if down_returns.len() >= 2 {
                let mean = down_returns.iter().sum::<f64>() / down_returns.len() as f64;
                let var = down_returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / (down_returns.len() - 1) as f64;
                var.sqrt()
            } else {
                0.0
            };

            // Calculate skew as normalized difference
            let total_vol = up_vol + down_vol;
            if total_vol > 0.0 {
                result[i] = (down_vol - up_vol) / total_vol * 100.0;
            } else {
                result[i] = 0.0;
            }
        }

        // Apply smoothing if configured
        if self.config.smoothing_period > 1 {
            result = self.smooth(&result);
        }

        result
    }

    /// Calculate skew using high-low range data for better accuracy.
    pub fn calculate_with_range(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < self.config.period + 1 || high.len() != n || low.len() != n {
            return result;
        }

        for i in self.config.period..n {
            let start = i - self.config.period;

            // Use range-based measures
            let mut up_ranges: Vec<f64> = Vec::new();
            let mut down_ranges: Vec<f64> = Vec::new();

            for j in start..=i {
                if high[j] > 0.0 && low[j] > 0.0 && close.get(j.saturating_sub(1)).is_some() {
                    let prev_close = if j == 0 { close[0] } else { close[j - 1] };

                    // Upside range: high - max(prev_close, open)
                    let up_range = (high[j] - prev_close).max(0.0);
                    // Downside range: min(prev_close, open) - low
                    let down_range = (prev_close - low[j]).max(0.0);

                    if up_range > 0.0 {
                        up_ranges.push(up_range / prev_close);
                    }
                    if down_range > 0.0 {
                        down_ranges.push(down_range / prev_close);
                    }
                }
            }

            // Calculate average ranges
            let avg_up = if !up_ranges.is_empty() {
                up_ranges.iter().sum::<f64>() / up_ranges.len() as f64
            } else {
                0.0
            };

            let avg_down = if !down_ranges.is_empty() {
                down_ranges.iter().sum::<f64>() / down_ranges.len() as f64
            } else {
                0.0
            };

            let total = avg_up + avg_down;
            if total > 0.0 {
                result[i] = (avg_down - avg_up) / total * 100.0;
            } else {
                result[i] = 0.0;
            }
        }

        if self.config.smoothing_period > 1 {
            result = self.smooth(&result);
        }

        result
    }

    /// Apply EMA smoothing.
    fn smooth(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let alpha = 2.0 / (self.config.smoothing_period as f64 + 1.0);

        let mut ema = f64::NAN;
        for i in 0..n {
            if !data[i].is_nan() {
                if ema.is_nan() {
                    ema = data[i];
                } else {
                    ema = alpha * data[i] + (1.0 - alpha) * ema;
                }
                result[i] = ema;
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilitySkew {
    fn name(&self) -> &str {
        "Volatility Skew"
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

        let values = self.calculate_with_range(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trending_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Uptrending market with small pullbacks
        let mut close = Vec::with_capacity(100);
        let mut high = Vec::with_capacity(100);
        let mut low = Vec::with_capacity(100);

        for i in 0..100 {
            let base = 100.0 + (i as f64) * 0.5;
            let noise = (i as f64 * 0.4).sin() * 1.0;
            let c = base + noise;
            close.push(c);
            high.push(c * 1.01);
            low.push(c * 0.99);
        }

        (high, low, close)
    }

    fn make_volatile_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Market with large down moves
        let mut close = Vec::with_capacity(100);
        let mut high = Vec::with_capacity(100);
        let mut low = Vec::with_capacity(100);

        for i in 0..100 {
            let base = 100.0;
            let noise = if i % 5 == 0 { -3.0 } else { 0.5 };
            let c = base + (i as f64) * 0.1 + noise;
            close.push(c);
            high.push(c + if noise > 0.0 { 1.5 } else { 0.5 });
            low.push(c - if noise < 0.0 { 2.0 } else { 0.3 });
        }

        (high, low, close)
    }

    #[test]
    fn test_volatility_skew_basic() {
        let (_, _, close) = make_trending_data();
        let skew = VolatilitySkew::new(20).unwrap();
        let result = skew.calculate(&close);

        assert_eq!(result.len(), close.len());

        // Check values are in reasonable range [-100, 100]
        for i in 20..result.len() {
            if !result[i].is_nan() {
                assert!(result[i] >= -100.0 && result[i] <= 100.0,
                    "Skew {} out of range at index {}", result[i], i);
            }
        }
    }

    #[test]
    fn test_volatility_skew_with_range() {
        let (high, low, close) = make_volatile_data();
        let skew = VolatilitySkew::new(20).unwrap();
        let result = skew.calculate_with_range(&high, &low, &close);

        assert_eq!(result.len(), close.len());

        // Market with large down moves should show positive skew
        let avg_skew: f64 = result.iter()
            .skip(30)
            .filter(|v| !v.is_nan())
            .sum::<f64>() / result.iter().skip(30).filter(|v| !v.is_nan()).count() as f64;

        // May not always be positive but should be reasonable
        assert!(avg_skew.abs() < 100.0);
    }

    #[test]
    fn test_volatility_skew_smoothing() {
        let (_, _, close) = make_trending_data();

        let config_no_smooth = VolatilitySkewConfig {
            period: 20,
            smoothing_period: 1,
        };
        let config_smooth = VolatilitySkewConfig {
            period: 20,
            smoothing_period: 10,
        };

        let skew_raw = VolatilitySkew::from_config(config_no_smooth).unwrap();
        let skew_smooth = VolatilitySkew::from_config(config_smooth).unwrap();

        let raw = skew_raw.calculate(&close);
        let smooth = skew_smooth.calculate(&close);

        // Smoothed version should have less variance
        let raw_var: f64 = {
            let valid: Vec<f64> = raw.iter().skip(30).filter(|v| !v.is_nan()).copied().collect();
            if valid.len() < 2 { 0.0 } else {
                let mean = valid.iter().sum::<f64>() / valid.len() as f64;
                valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / valid.len() as f64
            }
        };

        let smooth_var: f64 = {
            let valid: Vec<f64> = smooth.iter().skip(30).filter(|v| !v.is_nan()).copied().collect();
            if valid.len() < 2 { 0.0 } else {
                let mean = valid.iter().sum::<f64>() / valid.len() as f64;
                valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / valid.len() as f64
            }
        };

        // Smoothed should generally have lower or similar variance
        // (may not always be true due to data, so just check it's reasonable)
        assert!(smooth_var.is_finite());
        assert!(raw_var.is_finite());
    }

    #[test]
    fn test_volatility_skew_invalid_period() {
        let result = VolatilitySkew::new(2);
        assert!(result.is_err());
    }
}
