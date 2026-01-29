//! Social Volume Indicator - IND-277
//!
//! A proxy indicator for social media mention volume.
//! Uses price and volume dynamics to estimate social interest.
//!
//! High volatility + High volume = Likely high social interest
//! Price breakouts = Increased social chatter

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Social Volume output.
#[derive(Debug, Clone)]
pub struct SocialVolumeOutput {
    /// Social volume index (0-100 scale).
    pub volume_index: Vec<f64>,
    /// Normalized social volume.
    pub normalized: Vec<f64>,
    /// Z-score of social volume.
    pub zscore: Vec<f64>,
}

/// Social Volume signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocialVolumeSignal {
    /// Extremely high social activity.
    Extreme,
    /// High social activity.
    High,
    /// Normal social activity.
    Normal,
    /// Low social activity.
    Low,
    /// Very low social activity.
    VeryLow,
}

/// Social Volume Indicator - IND-277
///
/// Estimates social media mention volume from market data.
///
/// # Formula
/// ```text
/// Activity = Volatility * Volume Surge * Price Change Magnitude
/// Social Volume Index = Normalized(Activity) * 100
/// ```
///
/// # Example
/// ```
/// use indicator_core::sentiment::SocialVolume;
///
/// let sv = SocialVolume::new(14, 50).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SocialVolume {
    /// Lookback period for baseline calculation.
    period: usize,
    /// Smoothing period for output.
    smooth_period: usize,
    /// High volume threshold.
    high_threshold: f64,
    /// Low volume threshold.
    low_threshold: f64,
}

impl SocialVolume {
    /// Create a new Social Volume indicator.
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            period,
            smooth_period,
            high_threshold: 70.0,
            low_threshold: 30.0,
        })
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(
        period: usize,
        smooth_period: usize,
        high_threshold: f64,
        low_threshold: f64,
    ) -> Result<Self> {
        let mut sv = Self::new(period, smooth_period)?;
        sv.high_threshold = high_threshold;
        sv.low_threshold = low_threshold;
        Ok(sv)
    }

    /// Calculate social volume from OHLCV data.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> SocialVolumeOutput {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());

        if n < self.period + 1 {
            return SocialVolumeOutput {
                volume_index: vec![0.0; n],
                normalized: vec![0.0; n],
                zscore: vec![f64::NAN; n],
            };
        }

        let mut raw_activity = vec![0.0; n];

        // Calculate raw activity score
        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // 1. Volatility component (ATR-like)
            let mut atr_sum = 0.0;
            for j in (start + 1)..=i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                atr_sum += tr;
            }
            let atr = atr_sum / self.period as f64;
            let atr_pct = if close[i] > 0.0 { atr / close[i] * 100.0 } else { 0.0 };

            // 2. Volume surge component
            let avg_vol: f64 = volume[start..i].iter().sum::<f64>() / (i - start) as f64;
            let vol_surge = if avg_vol > 0.0 { volume[i] / avg_vol } else { 1.0 };

            // 3. Price change magnitude
            let price_change = if close[start] > 0.0 {
                ((close[i] / close[start] - 1.0) * 100.0).abs()
            } else {
                0.0
            };

            // Combine components
            // Social interest increases with volatility, volume, and price movement
            raw_activity[i] = atr_pct * vol_surge * (1.0 + price_change * 0.1);
        }

        // Normalize to 0-100 scale
        let mut volume_index = self.normalize_to_scale(&raw_activity, self.period);

        // Apply smoothing
        if self.smooth_period > 1 {
            let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
            for i in 1..n {
                volume_index[i] = alpha * volume_index[i] + (1.0 - alpha) * volume_index[i - 1];
            }
        }

        // Calculate normalized (min-max over period)
        let normalized = self.calculate_normalized(&raw_activity);

        // Calculate z-score
        let zscore = self.calculate_zscore(&raw_activity);

        SocialVolumeOutput {
            volume_index,
            normalized,
            zscore,
        }
    }

    /// Normalize values to 0-100 scale using rolling percentile.
    fn normalize_to_scale(&self, data: &[f64], lookback: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        for i in lookback..n {
            let start = i.saturating_sub(lookback * 2);
            let window: Vec<f64> = data[start..=i]
                .iter()
                .filter(|x| **x > 0.0)
                .copied()
                .collect();

            if window.is_empty() {
                continue;
            }

            let min = window.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            if max - min > 1e-10 {
                result[i] = ((data[i] - min) / (max - min) * 100.0).clamp(0.0, 100.0);
            } else {
                result[i] = 50.0; // Neutral if no variance
            }
        }

        result
    }

    /// Calculate normalized values (0-1 scale).
    fn calculate_normalized(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let window: Vec<f64> = data[start..=i].to_vec();

            let min = window.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            if max - min > 1e-10 {
                result[i] = (data[i] - min) / (max - min);
            } else {
                result[i] = 0.5;
            }
        }

        result
    }

    /// Calculate z-score.
    fn calculate_zscore(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let window: Vec<f64> = data[start..=i]
                .iter()
                .filter(|x| **x > 0.0)
                .copied()
                .collect();

            if window.len() < 2 {
                continue;
            }

            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / window.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                result[i] = (data[i] - mean) / std_dev;
            }
        }

        result
    }

    /// Get signal interpretation.
    pub fn interpret(&self, volume_index: f64) -> SocialVolumeSignal {
        if volume_index.is_nan() || volume_index <= 0.0 {
            SocialVolumeSignal::VeryLow
        } else if volume_index >= 85.0 {
            SocialVolumeSignal::Extreme
        } else if volume_index >= self.high_threshold {
            SocialVolumeSignal::High
        } else if volume_index <= 15.0 {
            SocialVolumeSignal::VeryLow
        } else if volume_index <= self.low_threshold {
            SocialVolumeSignal::Low
        } else {
            SocialVolumeSignal::Normal
        }
    }
}

impl TechnicalIndicator for SocialVolume {
    fn name(&self) -> &str {
        "Social Volume"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::single(output.volume_index))
    }
}

impl Default for SocialVolume {
    fn default() -> Self {
        Self::new(14, 5).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.5).collect();
        let volume: Vec<f64> = (0..40).map(|i| 1000.0 + (i as f64) * 50.0).collect();
        (high, low, close, volume)
    }

    #[test]
    fn test_social_volume_basic() {
        let (high, low, close, volume) = make_test_data();
        let sv = SocialVolume::new(10, 5).unwrap();
        let output = sv.calculate(&high, &low, &close, &volume);

        assert_eq!(output.volume_index.len(), close.len());
        // Values should be in 0-100 range
        for val in output.volume_index.iter().skip(15) {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_social_volume_spike() {
        let (mut high, mut low, mut close, mut volume) = make_test_data();
        // Create a volume spike
        volume[30] = volume[29] * 5.0;
        high[30] = close[30] + 5.0;
        low[30] = close[30] - 2.0;

        let sv = SocialVolume::new(10, 3).unwrap();
        let output = sv.calculate(&high, &low, &close, &volume);

        // Social volume should increase at the spike
        assert!(output.volume_index[30] > output.volume_index[25]);
    }

    #[test]
    fn test_social_volume_interpretation() {
        let sv = SocialVolume::default();

        assert_eq!(sv.interpret(90.0), SocialVolumeSignal::Extreme);
        assert_eq!(sv.interpret(75.0), SocialVolumeSignal::High);
        assert_eq!(sv.interpret(50.0), SocialVolumeSignal::Normal);
        assert_eq!(sv.interpret(25.0), SocialVolumeSignal::Low);
        assert_eq!(sv.interpret(10.0), SocialVolumeSignal::VeryLow);
    }

    #[test]
    fn test_social_volume_zscore() {
        let (high, low, close, volume) = make_test_data();
        let sv = SocialVolume::new(10, 3).unwrap();
        let output = sv.calculate(&high, &low, &close, &volume);

        // Z-score should be valid after warmup
        for i in 15..output.zscore.len() {
            assert!(!output.zscore[i].is_nan());
        }
    }

    #[test]
    fn test_social_volume_validation() {
        assert!(SocialVolume::new(2, 5).is_err());
        assert!(SocialVolume::new(10, 0).is_err());
        assert!(SocialVolume::new(10, 5).is_ok());
    }

    #[test]
    fn test_technical_indicator_impl() {
        let sv = SocialVolume::default();
        assert_eq!(sv.name(), "Social Volume");
        assert!(sv.min_periods() > 0);
    }
}
