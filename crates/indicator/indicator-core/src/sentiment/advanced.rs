//! Advanced Sentiment Indicators
//!
//! Sophisticated sentiment indicators for comprehensive market analysis.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Price Action Sentiment - Sentiment derived from price action patterns
#[derive(Debug, Clone)]
pub struct PriceActionSentiment {
    period: usize,
}

impl PriceActionSentiment {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate price action sentiment (-100 to 100)
    /// Analyzes candle patterns, body sizes, and wicks to gauge sentiment
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;
            let mut sentiment_score = 0.0;

            for j in start..=i {
                let body = close[j] - open[j];
                let range = high[j] - low[j];

                if range > 1e-10 {
                    // Body ratio: large bodies indicate strong conviction
                    let body_ratio = body.abs() / range;

                    // Upper wick ratio: selling pressure
                    let upper_wick = high[j] - close[j].max(open[j]);
                    let upper_wick_ratio = upper_wick / range;

                    // Lower wick ratio: buying pressure
                    let lower_wick = close[j].min(open[j]) - low[j];
                    let lower_wick_ratio = lower_wick / range;

                    // Bullish candle with small upper wick = bullish sentiment
                    // Bearish candle with small lower wick = bearish sentiment
                    let candle_sentiment = if body > 0.0 {
                        body_ratio * (1.0 - upper_wick_ratio) * 2.0 - 1.0
                    } else {
                        -body_ratio * (1.0 - lower_wick_ratio) * 2.0 + 1.0
                    };

                    sentiment_score += candle_sentiment;
                }
            }

            // Normalize to -100 to 100
            result[i] = (sentiment_score / (self.period + 1) as f64 * 100.0).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for PriceActionSentiment {
    fn name(&self) -> &str {
        "Price Action Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close)))
    }
}

/// Volume Based Sentiment - Sentiment derived from volume patterns
#[derive(Debug, Clone)]
pub struct VolumeBasedSentiment {
    period: usize,
}

impl VolumeBasedSentiment {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume-based sentiment (-100 to 100)
    /// High volume on up moves = bullish, high volume on down moves = bearish
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate average volume
            let avg_volume: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;

            if avg_volume <= 0.0 {
                continue;
            }

            let mut up_volume = 0.0;
            let mut down_volume = 0.0;

            for j in (start + 1)..=i {
                let vol_ratio = volume[j] / avg_volume;
                if close[j] > close[j - 1] {
                    up_volume += vol_ratio;
                } else if close[j] < close[j - 1] {
                    down_volume += vol_ratio;
                }
            }

            let total_volume = up_volume + down_volume;
            if total_volume > 0.0 {
                // Sentiment based on volume-weighted direction
                let sentiment = (up_volume - down_volume) / total_volume * 100.0;
                result[i] = sentiment.clamp(-100.0, 100.0);
            }
        }
        result
    }
}

impl TechnicalIndicator for VolumeBasedSentiment {
    fn name(&self) -> &str {
        "Volume Based Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Momentum Sentiment - Sentiment derived from momentum indicators
#[derive(Debug, Clone)]
pub struct MomentumSentiment {
    period: usize,
    smoothing: usize,
}

impl MomentumSentiment {
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smoothing < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Calculate momentum-based sentiment (-100 to 100)
    /// Combines ROC, RSI-like components for sentiment gauge
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut raw_sentiment = vec![0.0; n];

        // Calculate raw momentum sentiment
        for i in self.period..n {
            // Rate of change component
            let roc = if close[i - self.period] > 0.0 {
                (close[i] / close[i - self.period] - 1.0) * 100.0
            } else {
                0.0
            };

            // RSI-like component (gains vs losses)
            let mut gains = 0.0;
            let mut losses = 0.0;
            for j in (i - self.period + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += change.abs();
                }
            }

            let rsi_component = if gains + losses > 0.0 {
                (gains / (gains + losses) - 0.5) * 200.0
            } else {
                0.0
            };

            // Acceleration component
            let mid = i - self.period / 2;
            let first_half_change = close[mid] - close[i - self.period];
            let second_half_change = close[i] - close[mid];
            let acceleration = if first_half_change.abs() > 1e-10 {
                ((second_half_change - first_half_change) / first_half_change.abs() * 50.0).clamp(-50.0, 50.0)
            } else {
                0.0
            };

            // Combine components
            raw_sentiment[i] = (roc * 0.3 + rsi_component * 0.5 + acceleration * 0.2).clamp(-100.0, 100.0);
        }

        // Apply smoothing
        let total_period = self.period + self.smoothing - 1;
        for i in total_period..n {
            let sum: f64 = raw_sentiment[(i - self.smoothing + 1)..=i].iter().sum();
            result[i] = sum / self.smoothing as f64;
        }

        result
    }
}

impl TechnicalIndicator for MomentumSentiment {
    fn name(&self) -> &str {
        "Momentum Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Extreme Sentiment - Detects extreme sentiment conditions
#[derive(Debug, Clone)]
pub struct ExtremeSentiment {
    period: usize,
    threshold: f64,
}

impl ExtremeSentiment {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if threshold <= 0.0 || threshold > 4.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0 and 4".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate extreme sentiment (-100 to 100)
    /// Detects overbought/oversold extremes using multiple factors
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate price statistics
            let close_slice = &close[start..=i];
            let mean: f64 = close_slice.iter().sum::<f64>() / (self.period + 1) as f64;
            let var: f64 = close_slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (self.period + 1) as f64;
            let std_dev = var.sqrt();

            if std_dev < 1e-10 {
                continue;
            }

            // Price z-score
            let price_zscore = (close[i] - mean) / std_dev;

            // Volume z-score
            let vol_slice = &volume[start..=i];
            let vol_mean: f64 = vol_slice.iter().sum::<f64>() / (self.period + 1) as f64;
            let vol_var: f64 = vol_slice.iter()
                .map(|&x| (x - vol_mean).powi(2))
                .sum::<f64>() / (self.period + 1) as f64;
            let vol_std = vol_var.sqrt();
            let vol_zscore = if vol_std > 1e-10 {
                (volume[i] - vol_mean) / vol_std
            } else {
                0.0
            };

            // Range position
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;
            let range_position = if range > 1e-10 {
                (close[i] - period_low) / range
            } else {
                0.5
            };

            // Extreme detection
            let is_price_extreme = price_zscore.abs() >= self.threshold;
            let is_vol_extreme = vol_zscore >= self.threshold;
            let is_range_extreme = range_position < 0.1 || range_position > 0.9;

            if is_price_extreme || (is_vol_extreme && is_range_extreme) {
                // Calculate extreme sentiment
                let direction = if price_zscore > 0.0 || range_position > 0.5 { 1.0 } else { -1.0 };
                let intensity = (price_zscore.abs() / self.threshold).min(2.0);
                let vol_factor = if is_vol_extreme { 1.3 } else { 1.0 };

                result[i] = (direction * intensity * vol_factor * 50.0).clamp(-100.0, 100.0);
            }
        }
        result
    }
}

impl TechnicalIndicator for ExtremeSentiment {
    fn name(&self) -> &str {
        "Extreme Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Sentiment Divergence - Divergence between price and sentiment
#[derive(Debug, Clone)]
pub struct SentimentDivergence {
    price_period: usize,
    sentiment_period: usize,
}

impl SentimentDivergence {
    pub fn new(price_period: usize, sentiment_period: usize) -> Result<Self> {
        if price_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "price_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if sentiment_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "sentiment_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { price_period, sentiment_period })
    }

    /// Calculate sentiment divergence (-100 to 100)
    /// Positive = bullish divergence (price down, sentiment up)
    /// Negative = bearish divergence (price up, sentiment down)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let max_period = self.price_period.max(self.sentiment_period);

        for i in max_period..n {
            // Calculate price trend
            let price_start = i - self.price_period;
            let price_change = if close[price_start] > 0.0 {
                (close[i] / close[price_start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Calculate internal sentiment (based on volume-weighted close position)
            let sent_start = i - self.sentiment_period;
            let mut sentiment_sum = 0.0;
            let mut vol_sum = 0.0;

            for j in sent_start..=i {
                let range = high[j] - low[j];
                if range > 1e-10 {
                    let position = (close[j] - low[j]) / range;
                    sentiment_sum += (position * 2.0 - 1.0) * volume[j];
                    vol_sum += volume[j];
                }
            }

            let internal_sentiment = if vol_sum > 0.0 {
                sentiment_sum / vol_sum * 100.0
            } else {
                0.0
            };

            // Calculate momentum sentiment
            let mut up_vol = 0.0;
            let mut down_vol = 0.0;
            for j in (sent_start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_vol += volume[j];
                } else if close[j] < close[j - 1] {
                    down_vol += volume[j];
                }
            }
            let vol_sentiment = if up_vol + down_vol > 0.0 {
                (up_vol - down_vol) / (up_vol + down_vol) * 100.0
            } else {
                0.0
            };

            // Combined sentiment
            let combined_sentiment = (internal_sentiment + vol_sentiment) / 2.0;

            // Divergence: price going one way, sentiment going the other
            // Normalize both to similar scales
            let price_direction = price_change.signum();
            let sentiment_direction = combined_sentiment.signum();

            if price_direction != sentiment_direction && price_direction != 0.0 && sentiment_direction != 0.0 {
                // Divergence detected
                let divergence_strength = (price_change.abs() + combined_sentiment.abs()) / 2.0;
                // Bullish divergence: price down, sentiment up
                // Bearish divergence: price up, sentiment down
                result[i] = (-price_direction * divergence_strength).clamp(-100.0, 100.0);
            }
        }
        result
    }
}

impl TechnicalIndicator for SentimentDivergence {
    fn name(&self) -> &str {
        "Sentiment Divergence"
    }

    fn min_periods(&self) -> usize {
        self.price_period.max(self.sentiment_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Composite Sentiment Score - Combined sentiment measurement
#[derive(Debug, Clone)]
pub struct CompositeSentimentScore {
    period: usize,
    weights: SentimentWeights,
}

/// Weights for combining different sentiment components
#[derive(Debug, Clone)]
pub struct SentimentWeights {
    pub price_action: f64,
    pub volume: f64,
    pub momentum: f64,
    pub volatility: f64,
}

impl Default for SentimentWeights {
    fn default() -> Self {
        Self {
            price_action: 0.30,
            volume: 0.25,
            momentum: 0.25,
            volatility: 0.20,
        }
    }
}

impl CompositeSentimentScore {
    pub fn new(period: usize) -> Result<Self> {
        Self::with_weights(period, SentimentWeights::default())
    }

    pub fn with_weights(period: usize, weights: SentimentWeights) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        let total = weights.price_action + weights.volume + weights.momentum + weights.volatility;
        if (total - 1.0).abs() > 0.01 {
            return Err(IndicatorError::InvalidParameter {
                name: "weights".to_string(),
                reason: "must sum to 1.0".to_string(),
            });
        }
        Ok(Self { period, weights })
    }

    /// Calculate composite sentiment score (-100 to 100)
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // 1. Price Action Component
            let mut price_action_score = 0.0;
            for j in start..=i {
                let body = close[j] - open[j];
                let range = high[j] - low[j];
                if range > 1e-10 {
                    let body_ratio = body / range;
                    price_action_score += body_ratio;
                }
            }
            let price_action = (price_action_score / (self.period + 1) as f64 * 100.0).clamp(-100.0, 100.0);

            // 2. Volume Component
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let mut vol_sentiment = 0.0;
            if avg_vol > 0.0 {
                for j in (start + 1)..=i {
                    let vol_ratio = volume[j] / avg_vol;
                    let direction = if close[j] > close[j - 1] { 1.0 } else { -1.0 };
                    vol_sentiment += direction * vol_ratio;
                }
            }
            let volume_score = (vol_sentiment / self.period as f64 * 50.0).clamp(-100.0, 100.0);

            // 3. Momentum Component
            let momentum = if close[start] > 0.0 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };
            let mut gains = 0.0;
            let mut losses = 0.0;
            for j in (start + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += change.abs();
                }
            }
            let rsi_like = if gains + losses > 0.0 {
                (gains / (gains + losses) - 0.5) * 200.0
            } else {
                0.0
            };
            let momentum_score = ((momentum * 2.0 + rsi_like) / 2.0).clamp(-100.0, 100.0);

            // 4. Volatility Component (inverse: low vol = bullish complacency, high vol = fear)
            let mut tr_sum = 0.0;
            for j in (start + 1)..=i {
                let tr1 = high[j] - low[j];
                let tr2 = (high[j] - close[j - 1]).abs();
                let tr3 = (low[j] - close[j - 1]).abs();
                tr_sum += tr1.max(tr2).max(tr3);
            }
            let atr = tr_sum / self.period as f64;
            let normalized_vol = if close[i] > 0.0 { atr / close[i] * 100.0 } else { 0.0 };
            // Invert: low volatility = positive sentiment, high volatility = negative
            let volatility_score = (50.0 - normalized_vol * 25.0).clamp(-100.0, 100.0);

            // Combine with weights
            result[i] = (
                price_action * self.weights.price_action +
                volume_score * self.weights.volume +
                momentum_score * self.weights.momentum +
                volatility_score * self.weights.volatility
            ).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for CompositeSentimentScore {
    fn name(&self) -> &str {
        "Composite Sentiment Score"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let open = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
                       107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
                       114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0];
        let high = vec![102.0, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0, 108.0, 107.5,
                       109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0, 114.0, 113.5, 115.0,
                       116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0, 119.5, 121.0, 122.0];
        let low = vec![98.0, 99.0, 100.0, 99.5, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5,
                      105.0, 106.0, 105.5, 107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0,
                      112.0, 111.5, 113.0, 114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0];
        let close = vec![101.0, 102.0, 103.0, 102.0, 104.0, 105.0, 104.0, 106.0, 107.0, 106.0,
                        108.0, 109.0, 108.0, 110.0, 111.0, 110.0, 112.0, 113.0, 112.0, 114.0,
                        115.0, 114.0, 116.0, 117.0, 116.0, 118.0, 119.0, 118.0, 120.0, 121.0];
        let volume = vec![1000.0, 1200.0, 1100.0, 1300.0, 1500.0, 1400.0, 1600.0, 1800.0, 1700.0, 1900.0,
                         2000.0, 1800.0, 2100.0, 2200.0, 2000.0, 2300.0, 2100.0, 2400.0, 2200.0, 2500.0,
                         2600.0, 2400.0, 2700.0, 2800.0, 2600.0, 2900.0, 3000.0, 2800.0, 3100.0, 3200.0];
        (open, high, low, close, volume)
    }

    #[test]
    fn test_price_action_sentiment() {
        let (open, high, low, close, _) = make_test_data();
        let pas = PriceActionSentiment::new(10).unwrap();
        let result = pas.calculate(&open, &high, &low, &close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_price_action_sentiment_validation() {
        assert!(PriceActionSentiment::new(4).is_err());
        assert!(PriceActionSentiment::new(5).is_ok());
    }

    #[test]
    fn test_volume_based_sentiment() {
        let (_, _, _, close, volume) = make_test_data();
        let vbs = VolumeBasedSentiment::new(10).unwrap();
        let result = vbs.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_volume_based_sentiment_validation() {
        assert!(VolumeBasedSentiment::new(4).is_err());
        assert!(VolumeBasedSentiment::new(5).is_ok());
    }

    #[test]
    fn test_momentum_sentiment() {
        let (_, _, _, close, _) = make_test_data();
        let ms = MomentumSentiment::new(10, 3).unwrap();
        let result = ms.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[20] >= -100.0 && result[20] <= 100.0);
    }

    #[test]
    fn test_momentum_sentiment_validation() {
        assert!(MomentumSentiment::new(4, 3).is_err());
        assert!(MomentumSentiment::new(5, 0).is_err());
        assert!(MomentumSentiment::new(5, 1).is_ok());
    }

    #[test]
    fn test_extreme_sentiment() {
        let (_, high, low, close, volume) = make_test_data();
        let es = ExtremeSentiment::new(10, 2.0).unwrap();
        let result = es.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Values should be within bounds
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_extreme_sentiment_validation() {
        assert!(ExtremeSentiment::new(9, 2.0).is_err());
        assert!(ExtremeSentiment::new(10, 0.0).is_err());
        assert!(ExtremeSentiment::new(10, 5.0).is_err());
        assert!(ExtremeSentiment::new(10, 2.0).is_ok());
    }

    #[test]
    fn test_sentiment_divergence() {
        let (_, high, low, close, volume) = make_test_data();
        let sd = SentimentDivergence::new(10, 10).unwrap();
        let result = sd.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_sentiment_divergence_validation() {
        assert!(SentimentDivergence::new(4, 10).is_err());
        assert!(SentimentDivergence::new(10, 4).is_err());
        assert!(SentimentDivergence::new(5, 5).is_ok());
    }

    #[test]
    fn test_composite_sentiment_score() {
        let (open, high, low, close, volume) = make_test_data();
        let css = CompositeSentimentScore::new(10).unwrap();
        let result = css.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_composite_sentiment_score_with_weights() {
        let (open, high, low, close, volume) = make_test_data();
        let weights = SentimentWeights {
            price_action: 0.40,
            volume: 0.30,
            momentum: 0.20,
            volatility: 0.10,
        };
        let css = CompositeSentimentScore::with_weights(10, weights).unwrap();
        let result = css.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_composite_sentiment_score_validation() {
        assert!(CompositeSentimentScore::new(9).is_err());
        assert!(CompositeSentimentScore::new(10).is_ok());

        let bad_weights = SentimentWeights {
            price_action: 0.50,
            volume: 0.50,
            momentum: 0.50,
            volatility: 0.50,
        };
        assert!(CompositeSentimentScore::with_weights(10, bad_weights).is_err());
    }
}
