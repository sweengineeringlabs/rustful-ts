//! Platform Activity Indicators
//!
//! Indicators for measuring social media and platform-specific activity metrics.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Reddit/Twitter Activity (IND-281) - Platform-specific metrics proxy
///
/// This indicator creates a proxy for social media activity using
/// price and volume patterns that often correlate with retail sentiment
/// spikes on platforms like Reddit and Twitter.
#[derive(Debug, Clone)]
pub struct RedditTwitterActivity {
    period: usize,
    volatility_weight: f64,
    volume_weight: f64,
}

/// Configuration for RedditTwitterActivity
#[derive(Debug, Clone)]
pub struct RedditTwitterActivityConfig {
    pub period: usize,
    pub volatility_weight: f64,
    pub volume_weight: f64,
}

impl Default for RedditTwitterActivityConfig {
    fn default() -> Self {
        Self {
            period: 14,
            volatility_weight: 0.5,
            volume_weight: 0.5,
        }
    }
}

impl RedditTwitterActivity {
    pub fn new(period: usize) -> Result<Self> {
        Self::with_config(RedditTwitterActivityConfig {
            period,
            ..Default::default()
        })
    }

    pub fn with_config(config: RedditTwitterActivityConfig) -> Result<Self> {
        if config.period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if config.volatility_weight < 0.0 || config.volatility_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.volume_weight < 0.0 || config.volume_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self {
            period: config.period,
            volatility_weight: config.volatility_weight,
            volume_weight: config.volume_weight,
        })
    }

    /// Calculate platform activity proxy (0-100 scale)
    ///
    /// High readings suggest elevated social media activity which often
    /// correlates with retail trading interest on platforms like WSB.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate intraday volatility proxy (retail excitement)
            let mut volatility_sum = 0.0;
            for j in start..=i {
                let range = high[j] - low[j];
                let mid = (high[j] + low[j]) / 2.0;
                if mid > 0.0 {
                    volatility_sum += (range / mid) * 100.0;
                }
            }
            let avg_volatility = volatility_sum / (i - start + 1) as f64;

            // Calculate volume surge (viral attention proxy)
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / (i - start).max(1) as f64;
            let volume_ratio = if avg_volume > 0.0 {
                (volume[i] / avg_volume).min(5.0)
            } else {
                1.0
            };

            // Calculate price momentum (FOMO indicator)
            let momentum = if close[start] > 0.0 {
                ((close[i] / close[start]) - 1.0).abs() * 100.0
            } else {
                0.0
            };

            // Calculate gap frequency (meme stock behavior)
            let mut gap_count = 0;
            for j in (start + 1)..=i {
                let gap = (close[j - 1] - low[j]).abs().max((high[j] - close[j - 1]).abs());
                let typical_range = (high[j] - low[j]).max(0.001);
                if gap > typical_range * 0.5 {
                    gap_count += 1;
                }
            }
            let gap_factor = (gap_count as f64 / self.period as f64) * 100.0;

            // Combine components
            let volatility_component = (avg_volatility * 10.0).min(50.0) * self.volatility_weight;
            let volume_component = ((volume_ratio - 1.0) * 25.0).max(0.0).min(50.0) * self.volume_weight;
            let momentum_component = momentum.min(25.0);
            let gap_component = gap_factor.min(25.0);

            let activity = volatility_component + volume_component + momentum_component * 0.5 + gap_component * 0.5;
            result[i] = activity.min(100.0).max(0.0);
        }
        result
    }
}

impl TechnicalIndicator for RedditTwitterActivity {
    fn name(&self) -> &str {
        "Reddit Twitter Activity"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![105.0, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 109.5, 111.0,
                       110.5, 112.0, 111.5, 113.0, 112.5, 114.0, 113.5, 115.0, 114.5, 116.0];
        let low = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 104.5, 106.0,
                      105.5, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0, 109.5, 111.0];
        let close = vec![104.0, 105.0, 106.0, 105.0, 107.0, 108.0, 107.0, 109.0, 108.0, 110.0,
                        109.0, 111.0, 110.0, 112.0, 111.0, 113.0, 112.0, 114.0, 113.0, 115.0];
        let volume = vec![1000.0, 1200.0, 1100.0, 1300.0, 1500.0, 1400.0, 1600.0, 1800.0, 1700.0, 1900.0,
                         2000.0, 1800.0, 2100.0, 2200.0, 2000.0, 2300.0, 2100.0, 2400.0, 2200.0, 2500.0];
        (high, low, close, volume)
    }

    #[test]
    fn test_reddit_twitter_activity_creation() {
        let indicator = RedditTwitterActivity::new(14);
        assert!(indicator.is_ok());

        let indicator = RedditTwitterActivity::new(3);
        assert!(indicator.is_err());
    }

    #[test]
    fn test_reddit_twitter_activity_calculation() {
        let (high, low, close, volume) = make_test_data();
        let indicator = RedditTwitterActivity::new(10).unwrap();
        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Activity should be between 0 and 100
        for i in 11..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_reddit_twitter_activity_with_config() {
        let config = RedditTwitterActivityConfig {
            period: 14,
            volatility_weight: 0.7,
            volume_weight: 0.3,
        };
        let indicator = RedditTwitterActivity::with_config(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_reddit_twitter_activity_min_periods() {
        let indicator = RedditTwitterActivity::new(14).unwrap();
        assert_eq!(indicator.min_periods(), 15);
    }
}
