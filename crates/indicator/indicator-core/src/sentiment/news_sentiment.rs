//! News Sentiment Indicator - IND-279
//!
//! A proxy indicator for NLP-based news sentiment analysis.
//! Uses volatility clustering and price gaps as proxies for news impact.
//!
//! Gap ups with follow-through = Positive news
//! Gap downs with follow-through = Negative news
//! High volatility without trend = Mixed/conflicting news

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// News Sentiment output.
#[derive(Debug, Clone)]
pub struct NewsSentimentOutput {
    /// News sentiment score (-100 to 100).
    pub sentiment: Vec<f64>,
    /// News impact magnitude (0-100).
    pub impact: Vec<f64>,
    /// News consistency score (0-100, how consistent the signal is).
    pub consistency: Vec<f64>,
    /// Detected news events (1 = positive, -1 = negative, 0 = none).
    pub events: Vec<i8>,
}

/// News Sentiment signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewsSentimentSignal {
    /// Strong positive news impact.
    StrongPositive,
    /// Moderate positive news.
    Positive,
    /// No significant news impact.
    Neutral,
    /// Moderate negative news.
    Negative,
    /// Strong negative news impact.
    StrongNegative,
}

/// News Sentiment Indicator - IND-279
///
/// Estimates news sentiment impact from price/volume patterns.
///
/// # Formula
/// ```text
/// Gap = (Open - Previous Close) / Previous Close
/// Follow-through = (Close - Open) / (High - Low)
/// News Sentiment = Gap * Follow-through * Impact Multiplier
/// ```
///
/// # Example
/// ```
/// use indicator_core::sentiment::NewsSentiment;
///
/// let ns = NewsSentiment::new(14, 0.02).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct NewsSentiment {
    /// Lookback period for baseline.
    period: usize,
    /// Minimum gap threshold (as decimal, e.g., 0.02 = 2%).
    gap_threshold: f64,
    /// Smoothing period.
    smooth_period: usize,
}

impl NewsSentiment {
    /// Create a new News Sentiment indicator.
    pub fn new(period: usize, gap_threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if gap_threshold <= 0.0 || gap_threshold > 0.2 {
            return Err(IndicatorError::InvalidParameter {
                name: "gap_threshold".to_string(),
                reason: "must be between 0 and 0.2 (0-20%)".to_string(),
            });
        }
        Ok(Self {
            period,
            gap_threshold,
            smooth_period: 3,
        })
    }

    /// Create with custom smoothing.
    pub fn with_smoothing(period: usize, gap_threshold: f64, smooth_period: usize) -> Result<Self> {
        let mut ns = Self::new(period, gap_threshold)?;
        ns.smooth_period = smooth_period.max(1);
        Ok(ns)
    }

    /// Calculate news sentiment from OHLCV data.
    pub fn calculate(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> NewsSentimentOutput {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());

        if n < self.period + 1 {
            return NewsSentimentOutput {
                sentiment: vec![0.0; n],
                impact: vec![0.0; n],
                consistency: vec![0.0; n],
                events: vec![0; n],
            };
        }

        let mut sentiment = vec![0.0; n];
        let mut impact = vec![0.0; n];
        let mut consistency = vec![0.0; n];
        let mut events: Vec<i8> = vec![0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate baseline volatility
            let mut vol_sum = 0.0;
            for j in start..i {
                if j > 0 {
                    let ret = (close[j] / close[j - 1] - 1.0).abs();
                    vol_sum += ret;
                }
            }
            let avg_volatility = vol_sum / (i - start) as f64;

            // Calculate gap
            let gap = if close[i - 1] > 0.0 {
                (open[i] - close[i - 1]) / close[i - 1]
            } else {
                0.0
            };

            // Calculate intraday range and follow-through
            let range = high[i] - low[i];
            let body = close[i] - open[i];
            let follow_through = if range > 1e-10 {
                body / range
            } else {
                0.0
            };

            // Calculate volume surge
            let avg_vol: f64 = volume[start..i].iter().sum::<f64>() / (i - start) as f64;
            let vol_surge = if avg_vol > 0.0 { volume[i] / avg_vol } else { 1.0 };

            // Detect news events
            let is_significant_gap = gap.abs() > self.gap_threshold;
            let has_follow_through = (gap > 0.0 && follow_through > 0.3)
                || (gap < 0.0 && follow_through < -0.3);
            let has_volume = vol_surge > 1.5;

            // Calculate raw sentiment
            let mut raw_sentiment = 0.0;
            let mut raw_impact = 0.0;

            if is_significant_gap {
                // Gap-based sentiment
                raw_sentiment = gap * 100.0 * (1.0 + follow_through.abs());

                // Impact based on gap size and volume
                raw_impact = (gap.abs() / self.gap_threshold * 50.0) * vol_surge.sqrt();

                // Event detection
                if has_follow_through && has_volume {
                    events[i] = if gap > 0.0 { 1 } else { -1 };
                }
            } else {
                // No significant gap - use momentum-based approach
                let momentum = if close[i - self.period] > 0.0 {
                    (close[i] / close[i - self.period] - 1.0) * 100.0
                } else {
                    0.0
                };

                // Check for gradual news impact (trend with volume)
                if vol_surge > 1.2 && avg_volatility > 0.0 {
                    let current_vol = (close[i] / close[i - 1] - 1.0).abs();
                    if current_vol > avg_volatility * 1.5 {
                        raw_sentiment = momentum * 0.5;
                        raw_impact = (current_vol / avg_volatility * 20.0).min(50.0);
                    }
                }
            }

            sentiment[i] = raw_sentiment.clamp(-100.0, 100.0);
            impact[i] = raw_impact.min(100.0);

            // Calculate consistency (how consistent sentiment is over period)
            let mut same_direction_count = 0;
            let current_direction = if sentiment[i] > 0.0 { 1 } else if sentiment[i] < 0.0 { -1 } else { 0 };
            for j in (start + 1)..i {
                let prev_dir = if sentiment[j] > 0.0 { 1 } else if sentiment[j] < 0.0 { -1 } else { 0 };
                if prev_dir == current_direction && current_direction != 0 {
                    same_direction_count += 1;
                }
            }
            consistency[i] = (same_direction_count as f64 / (i - start) as f64 * 100.0).min(100.0);
        }

        // Apply smoothing
        if self.smooth_period > 1 {
            let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
            for i in 1..n {
                sentiment[i] = alpha * sentiment[i] + (1.0 - alpha) * sentiment[i - 1];
                impact[i] = alpha * impact[i] + (1.0 - alpha) * impact[i - 1];
            }
        }

        NewsSentimentOutput {
            sentiment,
            impact,
            consistency,
            events,
        }
    }

    /// Get signal interpretation.
    pub fn interpret(&self, sentiment: f64, impact: f64) -> NewsSentimentSignal {
        if sentiment.is_nan() || impact < 10.0 {
            return NewsSentimentSignal::Neutral;
        }

        let strength = sentiment.abs() * (impact / 50.0).sqrt();

        if sentiment > 0.0 {
            if strength >= 50.0 {
                NewsSentimentSignal::StrongPositive
            } else if strength >= 20.0 {
                NewsSentimentSignal::Positive
            } else {
                NewsSentimentSignal::Neutral
            }
        } else {
            if strength >= 50.0 {
                NewsSentimentSignal::StrongNegative
            } else if strength >= 20.0 {
                NewsSentimentSignal::Negative
            } else {
                NewsSentimentSignal::Neutral
            }
        }
    }

    /// Count news events in a period.
    pub fn count_events(&self, events: &[i8]) -> (usize, usize) {
        let positive = events.iter().filter(|&&e| e > 0).count();
        let negative = events.iter().filter(|&&e| e < 0).count();
        (positive, negative)
    }
}

impl TechnicalIndicator for NewsSentiment {
    fn name(&self) -> &str {
        "News Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::single(output.sentiment))
    }
}

impl Default for NewsSentiment {
    fn default() -> Self {
        Self::new(14, 0.015).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_normal_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let open: Vec<f64> = close.iter().map(|c| c - 0.05).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.2).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.2).collect();
        let volume: Vec<f64> = vec![1000.0; 40];
        (open, high, low, close, volume)
    }

    fn make_gap_up_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut close: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let mut open = close.clone();
        let mut high: Vec<f64> = close.iter().map(|c| c + 0.2).collect();
        let mut low: Vec<f64> = close.iter().map(|c| c - 0.2).collect();
        let mut volume: Vec<f64> = vec![1000.0; 40];

        // Create a gap up at index 25
        open[25] = close[24] * 1.03; // 3% gap up
        close[25] = open[25] * 1.01; // Follow-through
        high[25] = close[25] * 1.005;
        low[25] = open[25] * 0.995;
        volume[25] = 3000.0; // Volume surge

        (open, high, low, close, volume)
    }

    #[test]
    fn test_news_sentiment_normal() {
        let (open, high, low, close, volume) = make_normal_data();
        let ns = NewsSentiment::new(10, 0.02).unwrap();
        let output = ns.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(output.sentiment.len(), close.len());
        // No significant gaps = low impact
        for i in 15..output.impact.len() {
            assert!(output.impact[i] < 50.0);
        }
    }

    #[test]
    fn test_news_sentiment_gap_up() {
        let (open, high, low, close, volume) = make_gap_up_data();
        let ns = NewsSentiment::new(10, 0.02).unwrap();
        let output = ns.calculate(&open, &high, &low, &close, &volume);

        // Should detect positive sentiment at gap
        assert!(output.sentiment[25] > 0.0);
        // Should detect news event
        assert_eq!(output.events[25], 1);
    }

    #[test]
    fn test_news_sentiment_range() {
        let (open, high, low, close, volume) = make_gap_up_data();
        let ns = NewsSentiment::new(10, 0.02).unwrap();
        let output = ns.calculate(&open, &high, &low, &close, &volume);

        for val in &output.sentiment {
            assert!(*val >= -100.0 && *val <= 100.0);
        }
        for val in &output.impact {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }

    #[test]
    fn test_news_sentiment_interpretation() {
        let ns = NewsSentiment::default();

        assert_eq!(ns.interpret(50.0, 80.0), NewsSentimentSignal::StrongPositive);
        assert_eq!(ns.interpret(30.0, 40.0), NewsSentimentSignal::Positive);
        assert_eq!(ns.interpret(10.0, 5.0), NewsSentimentSignal::Neutral);
        assert_eq!(ns.interpret(-30.0, 40.0), NewsSentimentSignal::Negative);
        assert_eq!(ns.interpret(-50.0, 80.0), NewsSentimentSignal::StrongNegative);
    }

    #[test]
    fn test_news_sentiment_event_count() {
        let ns = NewsSentiment::default();
        let events = vec![0, 1, -1, 1, 0, -1, -1, 1, 0, 0];

        let (positive, negative) = ns.count_events(&events);
        assert_eq!(positive, 3);
        assert_eq!(negative, 3);
    }

    #[test]
    fn test_news_sentiment_validation() {
        assert!(NewsSentiment::new(2, 0.02).is_err());
        assert!(NewsSentiment::new(10, 0.0).is_err());
        assert!(NewsSentiment::new(10, 0.3).is_err());
        assert!(NewsSentiment::new(10, 0.02).is_ok());
    }

    #[test]
    fn test_technical_indicator_impl() {
        let ns = NewsSentiment::default();
        assert_eq!(ns.name(), "News Sentiment");
        assert!(ns.min_periods() > 0);
    }

    #[test]
    fn test_news_sentiment_consistency() {
        let (open, high, low, close, volume) = make_normal_data();
        let ns = NewsSentiment::new(10, 0.02).unwrap();
        let output = ns.calculate(&open, &high, &low, &close, &volume);

        // Consistency should be in 0-100 range
        for val in &output.consistency {
            assert!(*val >= 0.0 && *val <= 100.0);
        }
    }
}
