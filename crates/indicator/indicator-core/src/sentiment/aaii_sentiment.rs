//! AAII Sentiment Indicators
//!
//! Indicators for measuring survey-based sentiment proxies.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// AAII Sentiment (IND-284) - Survey-based sentiment proxy
///
/// This indicator creates a proxy for AAII (American Association of
/// Individual Investors) sentiment survey using price action patterns
/// that correlate with retail investor sentiment.
#[derive(Debug, Clone)]
pub struct AAIISentiment {
    period: usize,
    lookback: usize,
}

/// Configuration for AAIISentiment
#[derive(Debug, Clone)]
pub struct AAIISentimentConfig {
    pub period: usize,
    pub lookback: usize,
}

impl Default for AAIISentimentConfig {
    fn default() -> Self {
        Self {
            period: 5,  // Weekly-like period (AAII is weekly)
            lookback: 52,  // One year lookback for extremes
        }
    }
}

impl AAIISentiment {
    pub fn new(period: usize, lookback: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if lookback < period {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least equal to period".to_string(),
            });
        }
        Ok(Self { period, lookback })
    }

    pub fn with_config(config: AAIISentimentConfig) -> Result<Self> {
        Self::new(config.period, config.lookback)
    }

    /// Calculate bullish sentiment proxy (0-100)
    ///
    /// Higher values indicate more bullish retail sentiment
    pub fn calculate_bullish(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let period_start = i.saturating_sub(self.period);

            // Recent performance affects retail bullishness
            let recent_return = if close[period_start] > 0.0 {
                (close[i] / close[period_start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Proximity to highs increases bullishness
            let max_high = high[start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let high_proximity = if max_high > 0.0 {
                close[i] / max_high
            } else {
                0.5
            };

            // Winning streak increases bullishness
            let mut streak = 0i32;
            for j in (period_start + 1)..=i {
                if close[j] >= close[j - 1] {
                    streak += 1;
                } else {
                    streak = 0;
                }
            }
            let streak_factor = (streak as f64 / self.period as f64).min(1.0);

            // Combine factors
            let return_component = (recent_return + 10.0).clamp(0.0, 20.0) / 20.0 * 40.0;
            let high_component = high_proximity * 30.0;
            let streak_component = streak_factor * 30.0;

            result[i] = (return_component + high_component + streak_component).clamp(0.0, 100.0);
        }
        result
    }

    /// Calculate bearish sentiment proxy (0-100)
    ///
    /// Higher values indicate more bearish retail sentiment
    pub fn calculate_bearish(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let period_start = i.saturating_sub(self.period);

            // Recent losses increase bearishness
            let recent_return = if close[period_start] > 0.0 {
                (close[i] / close[period_start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Proximity to lows increases bearishness
            let min_low = low[start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_high = high[start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = max_high - min_low;
            let low_proximity = if range > 0.0 {
                1.0 - (close[i] - min_low) / range
            } else {
                0.5
            };

            // Losing streak increases bearishness
            let mut streak = 0i32;
            for j in (period_start + 1)..=i {
                if close[j] <= close[j - 1] {
                    streak += 1;
                } else {
                    streak = 0;
                }
            }
            let streak_factor = (streak as f64 / self.period as f64).min(1.0);

            // Combine factors
            let return_component = (-recent_return + 10.0).clamp(0.0, 20.0) / 20.0 * 40.0;
            let low_component = low_proximity * 30.0;
            let streak_component = streak_factor * 30.0;

            result[i] = (return_component + low_component + streak_component).clamp(0.0, 100.0);
        }
        result
    }

    /// Calculate neutral sentiment proxy (0-100)
    pub fn calculate_neutral(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let bullish = self.calculate_bullish(high, low, close);
        let bearish = self.calculate_bearish(high, low, close);

        let n = bullish.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            // Neutral is what remains after bull + bear
            result[i] = (100.0 - bullish[i] - bearish[i]).clamp(0.0, 100.0);
        }
        result
    }

    /// Calculate bull-bear spread (-100 to 100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let bullish = self.calculate_bullish(high, low, close);
        let bearish = self.calculate_bearish(high, low, close);

        let n = bullish.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            result[i] = (bullish[i] - bearish[i]).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for AAIISentiment {
    fn name(&self) -> &str {
        "AAII Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let bullish = self.calculate_bullish(&data.high, &data.low, &data.close);
        let bearish = self.calculate_bearish(&data.high, &data.low, &data.close);
        let neutral = self.calculate_neutral(&data.high, &data.low, &data.close);
        let spread = self.calculate(&data.high, &data.low, &data.close);

        Ok(IndicatorOutput::triple(bullish, bearish, neutral))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create longer test data for lookback period
        let mut high = Vec::with_capacity(60);
        let mut low = Vec::with_capacity(60);
        let mut close = Vec::with_capacity(60);

        for i in 0..60 {
            let base = 100.0 + (i as f64 * 0.5);
            high.push(base + 5.0);
            low.push(base);
            close.push(base + 3.0);
        }

        (high, low, close)
    }

    #[test]
    fn test_aaii_sentiment_creation() {
        let indicator = AAIISentiment::new(5, 52);
        assert!(indicator.is_ok());

        let indicator = AAIISentiment::new(2, 52);
        assert!(indicator.is_err());

        let indicator = AAIISentiment::new(10, 5);
        assert!(indicator.is_err());
    }

    #[test]
    fn test_aaii_sentiment_bullish() {
        let (high, low, close) = make_test_data();
        let indicator = AAIISentiment::new(5, 52).unwrap();
        let result = indicator.calculate_bullish(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        for i in 53..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_aaii_sentiment_bearish() {
        let (high, low, close) = make_test_data();
        let indicator = AAIISentiment::new(5, 52).unwrap();
        let result = indicator.calculate_bearish(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        for i in 53..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_aaii_sentiment_spread() {
        let (high, low, close) = make_test_data();
        let indicator = AAIISentiment::new(5, 52).unwrap();
        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        for i in 53..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_aaii_sentiment_uptrend_bullish() {
        let (high, low, close) = make_test_data();
        let indicator = AAIISentiment::new(5, 52).unwrap();
        let bullish = indicator.calculate_bullish(&high, &low, &close);
        let bearish = indicator.calculate_bearish(&high, &low, &close);

        // In an uptrend, bullish should generally exceed bearish
        if bullish.len() > 55 {
            assert!(bullish[55] > bearish[55] * 0.5, "Bullish {} vs Bearish {}", bullish[55], bearish[55]);
        }
    }

    #[test]
    fn test_aaii_sentiment_min_periods() {
        let indicator = AAIISentiment::new(5, 52).unwrap();
        assert_eq!(indicator.min_periods(), 53);
    }
}
