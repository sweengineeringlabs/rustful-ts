//! Social Sentiment Indicator - IND-278
//!
//! A proxy indicator for positive/negative social sentiment ratio.
//! Uses price action and momentum to estimate crowd sentiment.
//!
//! Rising prices with increasing volume = Positive sentiment
//! Falling prices with increasing volume = Negative sentiment

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Social Sentiment output.
#[derive(Debug, Clone)]
pub struct SocialSentimentOutput {
    /// Sentiment ratio (-100 to 100 scale).
    pub sentiment_ratio: Vec<f64>,
    /// Positive sentiment proxy (0-100).
    pub positive: Vec<f64>,
    /// Negative sentiment proxy (0-100).
    pub negative: Vec<f64>,
    /// Sentiment momentum (rate of change in sentiment).
    pub momentum: Vec<f64>,
}

/// Social Sentiment signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocialSentimentSignal {
    /// Extremely positive sentiment.
    ExtremelyPositive,
    /// Positive sentiment.
    Positive,
    /// Neutral sentiment.
    Neutral,
    /// Negative sentiment.
    Negative,
    /// Extremely negative sentiment.
    ExtremelyNegative,
}

/// Social Sentiment Indicator - IND-278
///
/// Estimates positive/negative social sentiment ratio from market data.
///
/// # Formula
/// ```text
/// Up Momentum = Sum of positive returns * volume
/// Down Momentum = Sum of negative returns * volume
/// Sentiment Ratio = (Up - Down) / (Up + Down) * 100
/// ```
///
/// # Example
/// ```
/// use indicator_core::sentiment::SocialSentiment;
///
/// let ss = SocialSentiment::new(14, 5).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SocialSentiment {
    /// Lookback period for sentiment calculation.
    period: usize,
    /// Smoothing period.
    smooth_period: usize,
    /// Positive threshold.
    positive_threshold: f64,
    /// Negative threshold.
    negative_threshold: f64,
}

impl SocialSentiment {
    /// Create a new Social Sentiment indicator.
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
            positive_threshold: 30.0,
            negative_threshold: -30.0,
        })
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(
        period: usize,
        smooth_period: usize,
        positive_threshold: f64,
        negative_threshold: f64,
    ) -> Result<Self> {
        let mut ss = Self::new(period, smooth_period)?;
        ss.positive_threshold = positive_threshold;
        ss.negative_threshold = negative_threshold;
        Ok(ss)
    }

    /// Calculate social sentiment from OHLCV data.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> SocialSentimentOutput {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());

        if n < self.period + 1 {
            return SocialSentimentOutput {
                sentiment_ratio: vec![0.0; n],
                positive: vec![0.0; n],
                negative: vec![0.0; n],
                momentum: vec![0.0; n],
            };
        }

        let mut sentiment_ratio = vec![0.0; n];
        let mut positive = vec![0.0; n];
        let mut negative = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let mut up_momentum = 0.0;
            let mut down_momentum = 0.0;
            let mut total_volume = 0.0;

            for j in (start + 1)..=i {
                let ret = close[j] / close[j - 1] - 1.0;
                let vol_weight = volume[j];
                total_volume += vol_weight;

                // Consider candle body as sentiment indicator
                let body = close[j] - close[j - 1];
                let upper_wick = high[j] - close[j].max(close[j - 1]);
                let lower_wick = close[j].min(close[j - 1]) - low[j];

                // Positive sentiment: price up, small upper wick (conviction)
                // Negative sentiment: price down, small lower wick (conviction)
                let conviction = if ret > 0.0 {
                    1.0 + (body.abs() / (upper_wick.abs() + 0.001)).min(2.0) * 0.5
                } else {
                    1.0 + (body.abs() / (lower_wick.abs() + 0.001)).min(2.0) * 0.5
                };

                if ret > 0.0 {
                    up_momentum += ret.abs() * vol_weight * conviction;
                } else if ret < 0.0 {
                    down_momentum += ret.abs() * vol_weight * conviction;
                }
            }

            // Calculate sentiment ratio
            let total_momentum = up_momentum + down_momentum;
            if total_momentum > 1e-10 {
                sentiment_ratio[i] = (up_momentum - down_momentum) / total_momentum * 100.0;
            }

            // Calculate individual positive/negative scores
            if total_volume > 1e-10 {
                positive[i] = (up_momentum / total_volume * 1000.0).min(100.0);
                negative[i] = (down_momentum / total_volume * 1000.0).min(100.0);
            }
        }

        // Apply smoothing
        if self.smooth_period > 1 {
            let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
            for i in 1..n {
                sentiment_ratio[i] = alpha * sentiment_ratio[i] + (1.0 - alpha) * sentiment_ratio[i - 1];
                positive[i] = alpha * positive[i] + (1.0 - alpha) * positive[i - 1];
                negative[i] = alpha * negative[i] + (1.0 - alpha) * negative[i - 1];
            }
        }

        // Calculate momentum (rate of change in sentiment)
        let momentum = self.calculate_momentum(&sentiment_ratio);

        SocialSentimentOutput {
            sentiment_ratio,
            positive,
            negative,
            momentum,
        }
    }

    /// Calculate sentiment momentum.
    fn calculate_momentum(&self, sentiment: &[f64]) -> Vec<f64> {
        let n = sentiment.len();
        let mut result = vec![0.0; n];

        let momentum_period = self.period.min(5);
        for i in momentum_period..n {
            result[i] = sentiment[i] - sentiment[i - momentum_period];
        }

        result
    }

    /// Get signal interpretation.
    pub fn interpret(&self, sentiment_ratio: f64) -> SocialSentimentSignal {
        if sentiment_ratio.is_nan() {
            SocialSentimentSignal::Neutral
        } else if sentiment_ratio >= 60.0 {
            SocialSentimentSignal::ExtremelyPositive
        } else if sentiment_ratio >= self.positive_threshold {
            SocialSentimentSignal::Positive
        } else if sentiment_ratio <= -60.0 {
            SocialSentimentSignal::ExtremelyNegative
        } else if sentiment_ratio <= self.negative_threshold {
            SocialSentimentSignal::Negative
        } else {
            SocialSentimentSignal::Neutral
        }
    }

    /// Interpret sentiment with momentum context.
    pub fn interpret_with_momentum(&self, sentiment: f64, momentum: f64) -> SocialSentimentSignal {
        let base = self.interpret(sentiment);

        // Strengthen signal if momentum confirms
        if momentum.is_nan() {
            return base;
        }

        match base {
            SocialSentimentSignal::Positive if momentum > 10.0 => {
                SocialSentimentSignal::ExtremelyPositive
            }
            SocialSentimentSignal::Negative if momentum < -10.0 => {
                SocialSentimentSignal::ExtremelyNegative
            }
            _ => base,
        }
    }
}

impl TechnicalIndicator for SocialSentiment {
    fn name(&self) -> &str {
        "Social Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::single(output.sentiment_ratio))
    }
}

impl Default for SocialSentiment {
    fn default() -> Self {
        Self::new(14, 5).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uptrend_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64) * 1.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.3).collect();
        let volume: Vec<f64> = vec![1000.0; 40];
        (high, low, close, volume)
    }

    fn make_downtrend_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..40).map(|i| 140.0 - (i as f64) * 1.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.3).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        let volume: Vec<f64> = vec![1000.0; 40];
        (high, low, close, volume)
    }

    #[test]
    fn test_social_sentiment_uptrend() {
        let (high, low, close, volume) = make_uptrend_data();
        let ss = SocialSentiment::new(10, 3).unwrap();
        let output = ss.calculate(&high, &low, &close, &volume);

        assert_eq!(output.sentiment_ratio.len(), close.len());
        // Uptrend should show positive sentiment
        assert!(output.sentiment_ratio[30] > 0.0);
    }

    #[test]
    fn test_social_sentiment_downtrend() {
        let (high, low, close, volume) = make_downtrend_data();
        let ss = SocialSentiment::new(10, 3).unwrap();
        let output = ss.calculate(&high, &low, &close, &volume);

        // Downtrend should show negative sentiment
        assert!(output.sentiment_ratio[30] < 0.0);
    }

    #[test]
    fn test_social_sentiment_range() {
        let (high, low, close, volume) = make_uptrend_data();
        let ss = SocialSentiment::new(10, 3).unwrap();
        let output = ss.calculate(&high, &low, &close, &volume);

        // Sentiment should be in -100 to 100 range
        for val in output.sentiment_ratio.iter().skip(15) {
            assert!(*val >= -100.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_social_sentiment_interpretation() {
        let ss = SocialSentiment::default();

        assert_eq!(ss.interpret(70.0), SocialSentimentSignal::ExtremelyPositive);
        assert_eq!(ss.interpret(40.0), SocialSentimentSignal::Positive);
        assert_eq!(ss.interpret(0.0), SocialSentimentSignal::Neutral);
        assert_eq!(ss.interpret(-40.0), SocialSentimentSignal::Negative);
        assert_eq!(ss.interpret(-70.0), SocialSentimentSignal::ExtremelyNegative);
    }

    #[test]
    fn test_social_sentiment_momentum() {
        let (high, low, close, volume) = make_uptrend_data();
        let ss = SocialSentiment::new(10, 3).unwrap();
        let output = ss.calculate(&high, &low, &close, &volume);

        // Momentum should be calculated
        assert_eq!(output.momentum.len(), close.len());
    }

    #[test]
    fn test_social_sentiment_positive_negative() {
        let (high, low, close, volume) = make_uptrend_data();
        let ss = SocialSentiment::new(10, 3).unwrap();
        let output = ss.calculate(&high, &low, &close, &volume);

        // In uptrend, positive should be higher than negative
        assert!(output.positive[30] > output.negative[30]);
    }

    #[test]
    fn test_social_sentiment_validation() {
        assert!(SocialSentiment::new(2, 5).is_err());
        assert!(SocialSentiment::new(10, 0).is_err());
        assert!(SocialSentiment::new(10, 5).is_ok());
    }

    #[test]
    fn test_technical_indicator_impl() {
        let ss = SocialSentiment::default();
        assert_eq!(ss.name(), "Social Sentiment");
        assert!(ss.min_periods() > 0);
    }

    #[test]
    fn test_interpret_with_momentum() {
        let ss = SocialSentiment::default();

        // Positive sentiment with strong positive momentum -> extremely positive
        assert_eq!(
            ss.interpret_with_momentum(35.0, 15.0),
            SocialSentimentSignal::ExtremelyPositive
        );

        // Negative sentiment with strong negative momentum -> extremely negative
        assert_eq!(
            ss.interpret_with_momentum(-35.0, -15.0),
            SocialSentimentSignal::ExtremelyNegative
        );
    }
}
