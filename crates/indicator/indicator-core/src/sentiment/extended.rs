//! Extended Sentiment Indicators
//!
//! Additional sentiment-based indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Market Momentum Sentiment - Momentum-based sentiment gauge
#[derive(Debug, Clone)]
pub struct MarketMomentumSentiment {
    period: usize,
}

impl MarketMomentumSentiment {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate momentum-based sentiment (-100 to 100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            // Calculate momentum
            let momentum = (close[i] / close[i - self.period] - 1.0) * 100.0;

            // Calculate recent momentum consistency
            let mut up_count = 0;
            let mut down_count = 0;
            for j in (i - self.period + 1)..=i {
                if close[j] > close[j - 1] {
                    up_count += 1;
                } else if close[j] < close[j - 1] {
                    down_count += 1;
                }
            }

            // Sentiment combines momentum and consistency
            let consistency = (up_count as f64 - down_count as f64) / self.period as f64;
            result[i] = (momentum * 2.0 + consistency * 50.0).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for MarketMomentumSentiment {
    fn name(&self) -> &str {
        "Market Momentum Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volatility Sentiment - Volatility-derived sentiment indicator
#[derive(Debug, Clone)]
pub struct VolatilitySentiment {
    period: usize,
}

impl VolatilitySentiment {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volatility-based sentiment (high vol = fear, low vol = complacency)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate rolling volatility stats
        for i in self.period..n {
            let start = i - self.period;

            // Calculate true range volatility
            let mut tr_sum = 0.0;
            for j in start..=i {
                let tr = if j == 0 {
                    high[j] - low[j]
                } else {
                    let tr1 = high[j] - low[j];
                    let tr2 = (high[j] - close[j - 1]).abs();
                    let tr3 = (low[j] - close[j - 1]).abs();
                    tr1.max(tr2).max(tr3)
                };
                tr_sum += tr;
            }
            let avg_tr = tr_sum / (self.period + 1) as f64;

            // Normalize by price
            let normalized_vol = avg_tr / close[i] * 100.0;

            // Invert: high vol = negative sentiment (fear), low vol = positive (complacency)
            // Scale to -100 to 100
            result[i] = (50.0 - normalized_vol * 20.0).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for VolatilitySentiment {
    fn name(&self) -> &str {
        "Volatility Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Trend Sentiment - Trend strength as sentiment
#[derive(Debug, Clone)]
pub struct TrendSentiment {
    short_period: usize,
    long_period: usize,
}

impl TrendSentiment {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self { short_period, long_period })
    }

    /// Calculate trend-based sentiment
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term average
            let short_start = i - self.short_period;
            let short_avg: f64 = close[short_start..=i].iter().sum::<f64>() / (self.short_period + 1) as f64;

            // Long-term average
            let long_start = i - self.long_period;
            let long_avg: f64 = close[long_start..=i].iter().sum::<f64>() / (self.long_period + 1) as f64;

            // Trend strength
            let trend_strength = (short_avg / long_avg - 1.0) * 100.0;

            // Price position relative to averages
            let above_short = if close[i] > short_avg { 1.0 } else { -1.0 };
            let above_long = if close[i] > long_avg { 1.0 } else { -1.0 };

            // Combine trend and position
            result[i] = (trend_strength * 3.0 + (above_short + above_long) * 15.0).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for TrendSentiment {
    fn name(&self) -> &str {
        "Trend Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Reversal Sentiment - Detects potential reversal sentiment
#[derive(Debug, Clone)]
pub struct ReversalSentiment {
    period: usize,
}

impl ReversalSentiment {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate reversal sentiment (-100 bearish reversal, 100 bullish reversal)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Find high and low of period
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            if range > 1e-10 {
                // Position in range
                let position = (close[i] - period_low) / range;

                // Trend into position
                let trend = (close[i] - close[start]) / range;

                // Reversal sentiment:
                // High position after downtrend = bullish reversal
                // Low position after uptrend = bearish reversal
                let reversal_signal = if trend < -0.3 && position > 0.7 {
                    // Downtrend but price recovering = bullish reversal
                    (position - 0.5) * 100.0 * (1.0 - trend)
                } else if trend > 0.3 && position < 0.3 {
                    // Uptrend but price falling = bearish reversal
                    (position - 0.5) * 100.0 * (1.0 + trend)
                } else {
                    0.0
                };

                result[i] = reversal_signal.clamp(-100.0, 100.0);
            }
        }
        result
    }
}

impl TechnicalIndicator for ReversalSentiment {
    fn name(&self) -> &str {
        "Reversal Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Extreme Readings - Detects extreme market readings
#[derive(Debug, Clone)]
pub struct ExtremeReadings {
    period: usize,
    threshold: f64,
}

impl ExtremeReadings {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if threshold <= 0.0 || threshold > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0 and 5".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate extreme readings score (-100 extreme bearish, 100 extreme bullish)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate mean and std dev
            let mean: f64 = close[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let var: f64 = close[start..=i].iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (self.period + 1) as f64;
            let std_dev = var.sqrt();

            if std_dev > 1e-10 {
                // Z-score
                let z_score = (close[i] - mean) / std_dev;

                // Extreme reading based on threshold
                if z_score.abs() >= self.threshold {
                    result[i] = (z_score / self.threshold * 50.0).clamp(-100.0, 100.0);
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for ExtremeReadings {
    fn name(&self) -> &str {
        "Extreme Readings"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Sentiment Oscillator - Composite sentiment oscillator
#[derive(Debug, Clone)]
pub struct SentimentOscillator {
    period: usize,
}

impl SentimentOscillator {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate composite sentiment oscillator
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Price momentum component
            let price_change = (close[i] / close[start] - 1.0) * 100.0;

            // Volume trend component
            let vol_avg: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let vol_ratio = if vol_avg > 0.0 { volume[i] / vol_avg } else { 1.0 };

            // Range component
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;
            let position = if range > 1e-10 {
                (close[i] - period_low) / range
            } else {
                0.5
            };

            // Combine components
            // Price momentum (weight: 0.4), Position (weight: 0.4), Volume confirmation (weight: 0.2)
            let price_component = price_change.clamp(-50.0, 50.0);
            let position_component = (position - 0.5) * 100.0;
            let volume_component = ((vol_ratio - 1.0) * 25.0).clamp(-25.0, 25.0);

            // Volume confirms direction
            let volume_factor = if (price_change > 0.0 && vol_ratio > 1.0) ||
                                   (price_change < 0.0 && vol_ratio > 1.0) {
                1.2
            } else {
                0.8
            };

            result[i] = ((price_component * 0.4 + position_component * 0.4 + volume_component * 0.2) * volume_factor)
                .clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for SentimentOscillator {
    fn name(&self) -> &str {
        "Sentiment Oscillator"
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
        let high = vec![102.0, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0, 108.0, 107.5,
                       109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0, 114.0, 113.5, 115.0,
                       116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0, 119.5, 121.0, 122.0];
        let low = vec![98.0, 99.0, 100.0, 99.5, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5,
                      105.0, 106.0, 105.5, 107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0,
                      112.0, 111.5, 113.0, 114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0];
        let close = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
                        107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
                        114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0];
        let volume = vec![1000.0; 30];
        (high, low, close, volume)
    }

    #[test]
    fn test_market_momentum_sentiment() {
        let (_, _, close, _) = make_test_data();
        let mms = MarketMomentumSentiment::new(10).unwrap();
        let result = mms.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_volatility_sentiment() {
        let (high, low, close, _) = make_test_data();
        let vs = VolatilitySentiment::new(10).unwrap();
        let result = vs.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_trend_sentiment() {
        let (_, _, close, _) = make_test_data();
        let ts = TrendSentiment::new(5, 15).unwrap();
        let result = ts.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[20] >= -100.0 && result[20] <= 100.0);
    }

    #[test]
    fn test_reversal_sentiment() {
        let (high, low, close, _) = make_test_data();
        let rs = ReversalSentiment::new(10).unwrap();
        let result = rs.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Most values should be 0 or small in trending data
    }

    #[test]
    fn test_extreme_readings() {
        let (_, _, close, _) = make_test_data();
        let er = ExtremeReadings::new(10, 2.0).unwrap();
        let result = er.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_sentiment_oscillator() {
        let (high, low, close, volume) = make_test_data();
        let so = SentimentOscillator::new(10).unwrap();
        let result = so.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }
}
