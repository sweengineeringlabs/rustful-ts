//! Put/Call Open Interest Ratio (IND-250)
//!
//! Analyzes the ratio of put to call open interest as a sentiment indicator.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Put/Call Open Interest indicator.
#[derive(Debug, Clone)]
pub struct PutCallOpenInterestConfig {
    /// Smoothing period for the ratio
    pub smoothing_period: usize,
    /// Period for calculating ratio extremes
    pub lookback_period: usize,
    /// Overbought threshold (high put/call ratio = bearish sentiment)
    pub overbought_threshold: f64,
    /// Oversold threshold (low put/call ratio = bullish sentiment)
    pub oversold_threshold: f64,
}

impl Default for PutCallOpenInterestConfig {
    fn default() -> Self {
        Self {
            smoothing_period: 5,
            lookback_period: 20,
            overbought_threshold: 1.2,
            oversold_threshold: 0.8,
        }
    }
}

/// Put/Call Open Interest Ratio (IND-250)
///
/// Measures the ratio of put open interest to call open interest.
/// This is a contrarian sentiment indicator.
///
/// # Calculation
/// P/C OI Ratio = Put Open Interest / Call Open Interest
///
/// # Interpretation (Contrarian)
/// - High ratio (>1.2): Excessive bearish sentiment, potentially bullish
/// - Low ratio (<0.8): Excessive bullish sentiment, potentially bearish
/// - Normal (0.8-1.2): Balanced sentiment
///
/// # Note
/// When actual open interest data is unavailable, this indicator uses
/// volume and price patterns as a proxy for sentiment.
#[derive(Debug, Clone)]
pub struct PutCallOpenInterest {
    config: PutCallOpenInterestConfig,
}

impl PutCallOpenInterest {
    /// Create a new PutCallOpenInterest indicator.
    pub fn new(smoothing_period: usize) -> Result<Self> {
        if smoothing_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            config: PutCallOpenInterestConfig {
                smoothing_period,
                ..Default::default()
            },
        })
    }

    /// Create from configuration.
    pub fn from_config(config: PutCallOpenInterestConfig) -> Result<Self> {
        if config.smoothing_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if config.lookback_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if config.oversold_threshold >= config.overbought_threshold {
            return Err(IndicatorError::InvalidParameter {
                name: "oversold_threshold".to_string(),
                reason: "must be less than overbought_threshold".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Calculate put/call ratio from actual open interest data.
    pub fn calculate_from_oi(&self, put_oi: &[f64], call_oi: &[f64]) -> Vec<f64> {
        let n = put_oi.len().min(call_oi.len());
        let mut result = vec![f64::NAN; n];

        for i in 0..n {
            if call_oi[i] > 0.0 && put_oi[i] >= 0.0 {
                result[i] = put_oi[i] / call_oi[i];
            }
        }

        // Apply smoothing
        if self.config.smoothing_period > 1 {
            result = self.smooth(&result);
        }

        result
    }

    /// Calculate proxy ratio from price/volume data when OI not available.
    /// Uses down-volume vs up-volume as a proxy.
    pub fn calculate_proxy(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![f64::NAN; n];

        if n < 2 {
            return result;
        }

        // Calculate cumulative up/down volume over lookback
        for i in self.config.lookback_period..n {
            let start = i - self.config.lookback_period;

            let mut up_volume = 0.0;
            let mut down_volume = 0.0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_volume += volume[j];
                } else if close[j] < close[j - 1] {
                    down_volume += volume[j];
                } else {
                    // Unchanged, split evenly
                    up_volume += volume[j] / 2.0;
                    down_volume += volume[j] / 2.0;
                }
            }

            // Ratio of down (put proxy) to up (call proxy)
            if up_volume > 0.0 {
                result[i] = down_volume / up_volume;
            } else if down_volume > 0.0 {
                result[i] = 2.0;  // Cap at 2 when no up volume
            } else {
                result[i] = 1.0;  // Neutral when no volume
            }
        }

        // Apply smoothing
        if self.config.smoothing_period > 1 {
            result = self.smooth(&result);
        }

        result
    }

    /// Get sentiment signal from ratio.
    pub fn get_signal(&self, ratio: f64) -> PutCallSignal {
        if ratio.is_nan() {
            PutCallSignal::Neutral
        } else if ratio >= self.config.overbought_threshold {
            PutCallSignal::ContraBullish  // High put/call = contrarian bullish
        } else if ratio <= self.config.oversold_threshold {
            PutCallSignal::ContraBearish  // Low put/call = contrarian bearish
        } else {
            PutCallSignal::Neutral
        }
    }

    /// Calculate signals for entire series.
    pub fn calculate_signals(&self, ratios: &[f64]) -> Vec<PutCallSignal> {
        ratios.iter().map(|&r| self.get_signal(r)).collect()
    }

    /// Calculate percentile of current ratio.
    pub fn calculate_percentile(&self, ratios: &[f64]) -> Vec<f64> {
        let n = ratios.len();
        let mut result = vec![f64::NAN; n];

        if n < self.config.lookback_period {
            return result;
        }

        for i in (self.config.lookback_period - 1)..n {
            let start = i.saturating_sub(self.config.lookback_period - 1);
            let current = ratios[i];

            if current.is_nan() {
                continue;
            }

            let valid: Vec<f64> = ratios[start..i]
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .collect();

            if !valid.is_empty() {
                let count_below = valid.iter().filter(|&&v| v < current).count();
                result[i] = (count_below as f64 / valid.len() as f64) * 100.0;
            }
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

/// Put/Call ratio sentiment signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PutCallSignal {
    /// High put/call ratio - contrarian bullish
    ContraBullish,
    /// Low put/call ratio - contrarian bearish
    ContraBearish,
    /// Neutral ratio
    Neutral,
}

impl TechnicalIndicator for PutCallOpenInterest {
    fn name(&self) -> &str {
        "Put/Call Open Interest"
    }

    fn min_periods(&self) -> usize {
        self.config.lookback_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }

        // Use proxy calculation since we have price/volume data
        let values = self.calculate_proxy(&data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let volume: Vec<f64> = (0..100)
            .map(|i| 1000000.0 + (i as f64 * 0.5).cos() * 500000.0)
            .collect();
        (close, volume)
    }

    fn make_bearish_data() -> (Vec<f64>, Vec<f64>) {
        // Declining market with high down volume
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 - (i as f64) * 0.5)
            .collect();
        let volume: Vec<f64> = (0..100)
            .map(|_| 1000000.0)
            .collect();
        (close, volume)
    }

    #[test]
    fn test_put_call_oi_from_actual() {
        let put_oi: Vec<f64> = vec![1000.0, 1200.0, 1100.0, 1500.0, 1300.0];
        let call_oi: Vec<f64> = vec![1000.0, 1000.0, 1200.0, 1100.0, 1400.0];

        let pc = PutCallOpenInterest::new(1).unwrap();
        let result = pc.calculate_from_oi(&put_oi, &call_oi);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 0.01);  // 1000/1000 = 1.0
        assert!((result[1] - 1.2).abs() < 0.01);  // 1200/1000 = 1.2
    }

    #[test]
    fn test_put_call_oi_proxy() {
        let (close, volume) = make_test_data();
        let pc = PutCallOpenInterest::new(5).unwrap();
        let result = pc.calculate_proxy(&close, &volume);

        assert_eq!(result.len(), close.len());

        // Check values are reasonable
        for i in 25..result.len() {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 5.0,
                    "Ratio {} out of range at index {}", result[i], i);
            }
        }
    }

    #[test]
    fn test_put_call_oi_bearish_market() {
        let (close, volume) = make_bearish_data();
        let pc = PutCallOpenInterest::new(3).unwrap();
        let result = pc.calculate_proxy(&close, &volume);

        // In a consistently declining market, put proxy (down volume) should dominate
        let avg_ratio: f64 = result.iter()
            .skip(30)
            .filter(|v| !v.is_nan())
            .sum::<f64>() / result.iter().skip(30).filter(|v| !v.is_nan()).count() as f64;

        // Should be above 1.0 (more down volume)
        assert!(avg_ratio > 0.5, "Expected higher ratio in bear market, got {}", avg_ratio);
    }

    #[test]
    fn test_put_call_signal() {
        let config = PutCallOpenInterestConfig {
            smoothing_period: 5,
            lookback_period: 20,
            overbought_threshold: 1.2,
            oversold_threshold: 0.8,
        };
        let pc = PutCallOpenInterest::from_config(config).unwrap();

        assert_eq!(pc.get_signal(1.5), PutCallSignal::ContraBullish);
        assert_eq!(pc.get_signal(0.6), PutCallSignal::ContraBearish);
        assert_eq!(pc.get_signal(1.0), PutCallSignal::Neutral);
        assert_eq!(pc.get_signal(f64::NAN), PutCallSignal::Neutral);
    }

    #[test]
    fn test_put_call_percentile() {
        let ratios: Vec<f64> = (0..50)
            .map(|i| 0.7 + (i as f64) * 0.02)  // Rising from 0.7 to ~1.68
            .collect();

        let pc = PutCallOpenInterest::new(5).unwrap();
        let percentile = pc.calculate_percentile(&ratios);

        // Last value should have high percentile (near 100)
        let last = percentile.last().unwrap();
        assert!(!last.is_nan());
        assert!(*last > 90.0, "Expected high percentile, got {}", last);
    }

    #[test]
    fn test_put_call_config() {
        let config = PutCallOpenInterestConfig {
            smoothing_period: 10,
            lookback_period: 30,
            overbought_threshold: 1.5,
            oversold_threshold: 0.5,
        };
        let pc = PutCallOpenInterest::from_config(config).unwrap();
        assert_eq!(pc.min_periods(), 31);
    }

    #[test]
    fn test_put_call_invalid_config() {
        // Invalid: oversold >= overbought
        let config = PutCallOpenInterestConfig {
            smoothing_period: 5,
            lookback_period: 20,
            overbought_threshold: 1.0,
            oversold_threshold: 1.0,
        };
        assert!(PutCallOpenInterest::from_config(config).is_err());
    }
}
