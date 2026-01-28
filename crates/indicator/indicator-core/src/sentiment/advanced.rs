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

// ============================================================================
// Additional Advanced Sentiment Indicators
// ============================================================================

/// Sentiment Momentum - Tracks momentum in sentiment changes
#[derive(Debug, Clone)]
pub struct SentimentMomentum {
    period: usize,
    smoothing: usize,
}

impl SentimentMomentum {
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
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

    /// Calculate sentiment momentum (-100 to 100)
    /// Measures the rate of change in underlying sentiment readings
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut raw_sentiment = vec![0.0; n];
        let mut sentiment_momentum = vec![0.0; n];

        // First pass: Calculate raw sentiment based on price action and volume
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Close position in range (-1 to 1)
                let position = (close[i] - low[i]) / range * 2.0 - 1.0;

                // Price change direction
                let price_change = if close[i - 1] > 0.0 {
                    (close[i] / close[i - 1] - 1.0) * 100.0
                } else {
                    0.0
                };

                // Volume factor (relative to simple average)
                let vol_factor = if i >= self.period {
                    let avg_vol: f64 = volume[(i - self.period)..i].iter().sum::<f64>() / self.period as f64;
                    if avg_vol > 0.0 { (volume[i] / avg_vol).min(3.0) } else { 1.0 }
                } else {
                    1.0
                };

                // Raw sentiment combines position, price change, and volume
                raw_sentiment[i] = (position * 30.0 + price_change * 5.0) * vol_factor.sqrt();
            }
        }

        // Second pass: Calculate sentiment momentum (rate of change in sentiment)
        for i in self.period..n {
            let sent_change = raw_sentiment[i] - raw_sentiment[i - self.period];

            // Also consider acceleration
            let mid = i - self.period / 2;
            let first_half = raw_sentiment[mid] - raw_sentiment[i - self.period];
            let second_half = raw_sentiment[i] - raw_sentiment[mid];
            let acceleration = second_half - first_half;

            sentiment_momentum[i] = (sent_change * 0.7 + acceleration * 0.3).clamp(-100.0, 100.0);
        }

        // Third pass: Apply smoothing
        let total_lookback = self.period + self.smoothing - 1;
        for i in total_lookback..n {
            let sum: f64 = sentiment_momentum[(i - self.smoothing + 1)..=i].iter().sum();
            result[i] = (sum / self.smoothing as f64).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for SentimentMomentum {
    fn name(&self) -> &str {
        "Sentiment Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Sentiment Extreme Detector - Detects extreme sentiment conditions
#[derive(Debug, Clone)]
pub struct SentimentExtremeDetector {
    period: usize,
    z_threshold: f64,
}

impl SentimentExtremeDetector {
    pub fn new(period: usize, z_threshold: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if z_threshold <= 0.0 || z_threshold > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "z_threshold".to_string(),
                reason: "must be between 0 and 5".to_string(),
            });
        }
        Ok(Self { period, z_threshold })
    }

    /// Calculate extreme sentiment detection (-100 to 100)
    /// Returns strong signals when sentiment reaches extreme levels
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut sentiment_readings = vec![0.0; n];

        // Calculate sentiment readings
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                let position = (close[i] - low[i]) / range;
                let body = close[i] - close[i - 1];
                let body_ratio = body / range;

                // Volume confirmation
                let vol_spike = if i >= self.period {
                    let avg_vol: f64 = volume[(i - self.period)..i].iter().sum::<f64>() / self.period as f64;
                    if avg_vol > 0.0 && volume[i] > avg_vol * 1.5 { 1.5 } else { 1.0 }
                } else {
                    1.0
                };

                sentiment_readings[i] = (position * 2.0 - 1.0 + body_ratio) * 50.0 * vol_spike;
            }
        }

        // Detect extremes using z-score
        for i in self.period..n {
            let start = i - self.period;
            let slice = &sentiment_readings[start..=i];

            let mean: f64 = slice.iter().sum::<f64>() / (self.period + 1) as f64;
            let variance: f64 = slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (self.period + 1) as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                let z_score = (sentiment_readings[i] - mean) / std_dev;

                // Only output when extreme
                if z_score.abs() >= self.z_threshold {
                    let intensity = (z_score.abs() / self.z_threshold).min(2.0);
                    result[i] = (z_score.signum() * intensity * 50.0).clamp(-100.0, 100.0);
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for SentimentExtremeDetector {
    fn name(&self) -> &str {
        "Sentiment Extreme Detector"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Sentiment Trend Follower - Follows sentiment trends with confirmation
#[derive(Debug, Clone)]
pub struct SentimentTrendFollower {
    fast_period: usize,
    slow_period: usize,
}

impl SentimentTrendFollower {
    pub fn new(fast_period: usize, slow_period: usize) -> Result<Self> {
        if fast_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if slow_period <= fast_period {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_period".to_string(),
                reason: "must be greater than fast_period".to_string(),
            });
        }
        Ok(Self { fast_period, slow_period })
    }

    /// Calculate sentiment trend following signal (-100 to 100)
    /// Positive indicates bullish sentiment trend, negative indicates bearish
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut sentiment_base = vec![0.0; n];

        // Calculate base sentiment
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Close position normalized
                let position = (close[i] - low[i]) / range * 2.0 - 1.0;

                // Momentum component
                let momentum = if close[i - 1] > 0.0 {
                    ((close[i] / close[i - 1]) - 1.0) * 100.0
                } else {
                    0.0
                };

                // Volume weight
                let vol_weight = if i >= 5 {
                    let avg_vol: f64 = volume[(i - 5)..i].iter().sum::<f64>() / 5.0;
                    if avg_vol > 0.0 { (volume[i] / avg_vol).sqrt().min(2.0) } else { 1.0 }
                } else {
                    1.0
                };

                sentiment_base[i] = (position * 40.0 + momentum * 3.0) * vol_weight;
            }
        }

        // Calculate fast and slow sentiment averages
        for i in self.slow_period..n {
            // Fast average
            let fast_start = i - self.fast_period;
            let fast_avg: f64 = sentiment_base[fast_start..=i].iter().sum::<f64>()
                / (self.fast_period + 1) as f64;

            // Slow average
            let slow_start = i - self.slow_period;
            let slow_avg: f64 = sentiment_base[slow_start..=i].iter().sum::<f64>()
                / (self.slow_period + 1) as f64;

            // Trend following: fast above slow = bullish trend
            let trend_diff = fast_avg - slow_avg;

            // Trend strength based on alignment
            let trend_alignment = if (fast_avg > 0.0 && slow_avg > 0.0) ||
                                     (fast_avg < 0.0 && slow_avg < 0.0) {
                1.3  // Aligned trends are stronger
            } else {
                0.8  // Divergent trends are weaker
            };

            result[i] = (trend_diff * 2.0 * trend_alignment).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for SentimentTrendFollower {
    fn name(&self) -> &str {
        "Sentiment Trend Follower"
    }

    fn min_periods(&self) -> usize {
        self.slow_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Sentiment Contrarian Signal - Generates contrarian signals from sentiment extremes
#[derive(Debug, Clone)]
pub struct SentimentContrarianSignal {
    period: usize,
    contrarian_threshold: f64,
}

impl SentimentContrarianSignal {
    pub fn new(period: usize, contrarian_threshold: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if contrarian_threshold <= 0.0 || contrarian_threshold > 100.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "contrarian_threshold".to_string(),
                reason: "must be between 0 and 100".to_string(),
            });
        }
        Ok(Self { period, contrarian_threshold })
    }

    /// Calculate contrarian sentiment signal (-100 to 100)
    /// Positive = contrarian bullish (sentiment was too bearish)
    /// Negative = contrarian bearish (sentiment was too bullish)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut sentiment_readings = vec![0.0; n];

        // Calculate sentiment readings
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Multiple sentiment components
                let position = (close[i] - low[i]) / range;
                let body = (close[i] - close[i - 1]).signum();

                // Volume surge indicator
                let vol_surge = if i >= self.period {
                    let avg_vol: f64 = volume[(i - self.period)..i].iter().sum::<f64>() / self.period as f64;
                    if avg_vol > 0.0 { (volume[i] / avg_vol - 1.0).max(0.0) } else { 0.0 }
                } else {
                    0.0
                };

                // Higher volume on moves increases sentiment reading
                sentiment_readings[i] = ((position * 2.0 - 1.0) * 50.0 + body * 20.0) * (1.0 + vol_surge * 0.5);
            }
        }

        // Generate contrarian signals
        for i in self.period..n {
            let start = i - self.period;

            // Calculate average sentiment over period
            let avg_sentiment: f64 = sentiment_readings[start..=i].iter().sum::<f64>()
                / (self.period + 1) as f64;

            // Check for extreme sentiment that warrants contrarian signal
            if avg_sentiment.abs() >= self.contrarian_threshold {
                // Contrarian: opposite of extreme sentiment
                // If sentiment was extremely bullish (positive), generate bearish contrarian signal
                // If sentiment was extremely bearish (negative), generate bullish contrarian signal
                let excess = avg_sentiment.abs() - self.contrarian_threshold;
                let intensity = (1.0 + excess / 50.0).min(2.0);

                // Contrarian signal is OPPOSITE of current sentiment
                result[i] = (-avg_sentiment.signum() * intensity * 50.0).clamp(-100.0, 100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for SentimentContrarianSignal {
    fn name(&self) -> &str {
        "Sentiment Contrarian Signal"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Sentiment Volatility - Measures volatility in sentiment readings
#[derive(Debug, Clone)]
pub struct SentimentVolatility {
    period: usize,
}

impl SentimentVolatility {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate sentiment volatility (0 to 100)
    /// Higher values indicate more volatile/unstable sentiment
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];
        let mut sentiment_readings = vec![0.0; n];

        // Calculate daily sentiment readings
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Position-based sentiment
                let position = (close[i] - low[i]) / range * 2.0 - 1.0;

                // Change-based sentiment
                let change_sentiment = if close[i - 1] > 0.0 {
                    ((close[i] / close[i - 1]) - 1.0) * 50.0
                } else {
                    0.0
                };

                sentiment_readings[i] = (position * 50.0 + change_sentiment).clamp(-100.0, 100.0);
            }
        }

        // Calculate volatility of sentiment
        for i in self.period..n {
            let start = i - self.period;
            let slice = &sentiment_readings[start..=i];

            // Mean sentiment
            let mean: f64 = slice.iter().sum::<f64>() / (self.period + 1) as f64;

            // Standard deviation of sentiment
            let variance: f64 = slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (self.period + 1) as f64;
            let std_dev = variance.sqrt();

            // Also measure max swing in sentiment
            let max_sent = slice.iter().cloned().fold(f64::MIN, f64::max);
            let min_sent = slice.iter().cloned().fold(f64::MAX, f64::min);
            let sentiment_range = max_sent - min_sent;

            // Combine std dev and range for volatility measure
            // Scale to 0-100 range
            result[i] = ((std_dev * 1.5 + sentiment_range * 0.3) / 2.0).clamp(0.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for SentimentVolatility {
    fn name(&self) -> &str {
        "Sentiment Volatility"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Sentiment Cycle - Tracks sentiment cycles and phases
#[derive(Debug, Clone)]
pub struct SentimentCycle {
    period: usize,
    cycle_length: usize,
}

impl SentimentCycle {
    pub fn new(period: usize, cycle_length: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if cycle_length < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "cycle_length".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, cycle_length })
    }

    /// Calculate sentiment cycle position (-100 to 100)
    /// Tracks where sentiment is in its cycle: -100 = trough, 100 = peak
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut sentiment_readings = vec![0.0; n];
        let max_period = self.period.max(self.cycle_length);

        // Calculate sentiment readings
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                let position = (close[i] - low[i]) / range;
                let momentum = if close[i - 1] > 0.0 {
                    (close[i] / close[i - 1] - 1.0) * 100.0
                } else {
                    0.0
                };

                // Volume factor
                let vol_factor = if i >= self.period {
                    let avg_vol: f64 = volume[(i - self.period)..i].iter().sum::<f64>() / self.period as f64;
                    if avg_vol > 0.0 { (volume[i] / avg_vol).sqrt().min(2.0) } else { 1.0 }
                } else {
                    1.0
                };

                sentiment_readings[i] = ((position * 2.0 - 1.0) * 40.0 + momentum * 3.0) * vol_factor;
            }
        }

        // Calculate cycle position
        for i in max_period..n {
            let cycle_start = i - self.cycle_length;
            let cycle_slice = &sentiment_readings[cycle_start..=i];

            // Find cycle high and low
            let cycle_high = cycle_slice.iter().cloned().fold(f64::MIN, f64::max);
            let cycle_low = cycle_slice.iter().cloned().fold(f64::MAX, f64::min);
            let cycle_range = cycle_high - cycle_low;

            // Current position in cycle
            let current_sentiment = sentiment_readings[i];

            if cycle_range > 1e-10 {
                // Normalize to -100 to 100 based on cycle position
                let cycle_position = (current_sentiment - cycle_low) / cycle_range * 2.0 - 1.0;

                // Detect cycle phase based on recent trajectory
                let recent_avg: f64 = sentiment_readings[(i - self.period)..=i].iter().sum::<f64>()
                    / (self.period + 1) as f64;
                let older_avg: f64 = if i >= self.period * 2 {
                    sentiment_readings[(i - self.period * 2)..(i - self.period)].iter().sum::<f64>()
                        / self.period as f64
                } else {
                    recent_avg
                };

                let trend = recent_avg - older_avg;

                // Combine cycle position with trend for phase detection
                // Rising sentiment near bottom = early cycle (bullish)
                // Falling sentiment near top = late cycle (bearish)
                let phase_adjustment = if cycle_position < 0.0 && trend > 0.0 {
                    10.0  // Recovery phase
                } else if cycle_position > 0.0 && trend < 0.0 {
                    -10.0  // Distribution phase
                } else {
                    0.0
                };

                result[i] = (cycle_position * 100.0 + phase_adjustment).clamp(-100.0, 100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for SentimentCycle {
    fn name(&self) -> &str {
        "Sentiment Cycle"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.cycle_length) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

// ============================================================================
// Extended Advanced Sentiment Indicators
// ============================================================================

/// Sentiment Strength - Overall sentiment strength measure
/// Combines multiple factors to gauge the overall strength of market sentiment
#[derive(Debug, Clone)]
pub struct SentimentStrength {
    period: usize,
    smoothing: usize,
}

impl SentimentStrength {
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

    /// Calculate overall sentiment strength (0 to 100)
    /// Higher values indicate stronger sentiment (either bullish or bearish)
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut raw_strength = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // 1. Price momentum strength
            let price_change = if close[start] > 0.0 {
                ((close[i] / close[start]) - 1.0).abs() * 100.0
            } else {
                0.0
            };

            // 2. Directional consistency (how many bars moved in same direction)
            let mut up_count = 0;
            let mut down_count = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_count += 1;
                } else if close[j] < close[j - 1] {
                    down_count += 1;
                }
            }
            let consistency = (up_count.max(down_count) as f64) / self.period as f64;

            // 3. Volume confirmation strength
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let mut vol_confirmed_moves = 0.0;
            let mut total_moves = 0.0;
            for j in (start + 1)..=i {
                if close[j] != close[j - 1] {
                    total_moves += 1.0;
                    if volume[j] > avg_vol {
                        vol_confirmed_moves += 1.0;
                    }
                }
            }
            let vol_confirmation = if total_moves > 0.0 {
                vol_confirmed_moves / total_moves
            } else {
                0.5
            };

            // 4. Body strength (large bodies indicate conviction)
            let mut body_strength_sum = 0.0;
            for j in start..=i {
                let range = high[j] - low[j];
                if range > 1e-10 {
                    let body = (close[j] - open[j]).abs();
                    body_strength_sum += body / range;
                }
            }
            let body_strength = body_strength_sum / (self.period + 1) as f64;

            // 5. Range expansion (wider ranges indicate stronger sentiment)
            let mut atr_sum = 0.0;
            for j in (start + 1)..=i {
                let tr1 = high[j] - low[j];
                let tr2 = (high[j] - close[j - 1]).abs();
                let tr3 = (low[j] - close[j - 1]).abs();
                atr_sum += tr1.max(tr2).max(tr3);
            }
            let atr = atr_sum / self.period as f64;
            let range_strength = if close[i] > 0.0 {
                (atr / close[i] * 100.0).min(10.0) / 10.0
            } else {
                0.0
            };

            // Combine all components
            raw_strength[i] = (
                price_change.min(50.0) / 50.0 * 25.0 +   // Price momentum (max 25)
                consistency * 25.0 +                       // Directional consistency (max 25)
                vol_confirmation * 20.0 +                  // Volume confirmation (max 20)
                body_strength * 15.0 +                     // Body strength (max 15)
                range_strength * 15.0                      // Range expansion (max 15)
            ).clamp(0.0, 100.0);
        }

        // Apply smoothing
        let total_lookback = self.period + self.smoothing - 1;
        for i in total_lookback..n {
            let sum: f64 = raw_strength[(i - self.smoothing + 1)..=i].iter().sum();
            result[i] = (sum / self.smoothing as f64).clamp(0.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for SentimentStrength {
    fn name(&self) -> &str {
        "Sentiment Strength"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Sentiment Acceleration - Rate of change of sentiment
/// Measures how quickly sentiment is changing (second derivative of sentiment)
#[derive(Debug, Clone)]
pub struct SentimentAcceleration {
    period: usize,
    roc_period: usize,
}

impl SentimentAcceleration {
    pub fn new(period: usize, roc_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if roc_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "roc_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, roc_period })
    }

    /// Calculate sentiment acceleration (-100 to 100)
    /// Positive = sentiment improving at increasing rate
    /// Negative = sentiment deteriorating at increasing rate
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut sentiment = vec![0.0; n];
        let mut sentiment_velocity = vec![0.0; n];

        // First pass: Calculate base sentiment
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Position-based sentiment
                let position = (close[i] - low[i]) / range * 2.0 - 1.0;

                // Momentum-based sentiment
                let momentum = if close[i - 1] > 0.0 {
                    ((close[i] / close[i - 1]) - 1.0) * 100.0
                } else {
                    0.0
                };

                // Volume factor
                let vol_factor = if i >= self.period {
                    let avg_vol: f64 = volume[(i - self.period)..i].iter().sum::<f64>() / self.period as f64;
                    if avg_vol > 0.0 { (volume[i] / avg_vol).sqrt().min(2.0) } else { 1.0 }
                } else {
                    1.0
                };

                sentiment[i] = (position * 40.0 + momentum * 5.0) * vol_factor;
            }
        }

        // Second pass: Calculate sentiment velocity (first derivative)
        for i in self.period..n {
            let start = i - self.period;
            let recent_sentiment: f64 = sentiment[(i - self.period / 2)..=i].iter().sum::<f64>()
                / ((self.period / 2) + 1) as f64;
            let older_sentiment: f64 = sentiment[start..(i - self.period / 2)].iter().sum::<f64>()
                / (self.period / 2) as f64;

            sentiment_velocity[i] = recent_sentiment - older_sentiment;
        }

        // Third pass: Calculate sentiment acceleration (second derivative)
        let total_lookback = self.period + self.roc_period;
        for i in total_lookback..n {
            let current_velocity = sentiment_velocity[i];
            let past_velocity = sentiment_velocity[i - self.roc_period];

            // Acceleration is the change in velocity
            let acceleration = current_velocity - past_velocity;

            // Scale and clamp
            result[i] = (acceleration * 2.0).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for SentimentAcceleration {
    fn name(&self) -> &str {
        "Sentiment Acceleration"
    }

    fn min_periods(&self) -> usize {
        self.period + self.roc_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Sentiment Mean Reversion - Distance from sentiment mean
/// Measures how far current sentiment deviates from its historical average
#[derive(Debug, Clone)]
pub struct SentimentMeanReversion {
    short_period: usize,
    long_period: usize,
}

impl SentimentMeanReversion {
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

    /// Calculate sentiment mean reversion signal (-100 to 100)
    /// Positive = sentiment below mean (potential bullish reversion)
    /// Negative = sentiment above mean (potential bearish reversion)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut sentiment = vec![0.0; n];

        // Calculate base sentiment readings
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Multi-factor sentiment
                let position = (close[i] - low[i]) / range * 2.0 - 1.0;
                let momentum = if close[i - 1] > 0.0 {
                    ((close[i] / close[i - 1]) - 1.0) * 100.0
                } else {
                    0.0
                };

                // Volume weight
                let vol_weight = if i >= 5 {
                    let avg_vol: f64 = volume[(i - 5)..i].iter().sum::<f64>() / 5.0;
                    if avg_vol > 0.0 { (volume[i] / avg_vol).sqrt().min(2.0) } else { 1.0 }
                } else {
                    1.0
                };

                sentiment[i] = (position * 40.0 + momentum * 4.0) * vol_weight;
            }
        }

        // Calculate mean reversion signal
        for i in self.long_period..n {
            // Short-term sentiment average
            let short_start = i - self.short_period;
            let short_avg: f64 = sentiment[short_start..=i].iter().sum::<f64>()
                / (self.short_period + 1) as f64;

            // Long-term sentiment average (the "mean")
            let long_start = i - self.long_period;
            let long_avg: f64 = sentiment[long_start..=i].iter().sum::<f64>()
                / (self.long_period + 1) as f64;

            // Calculate standard deviation of long-term sentiment
            let variance: f64 = sentiment[long_start..=i].iter()
                .map(|&x| (x - long_avg).powi(2))
                .sum::<f64>() / (self.long_period + 1) as f64;
            let std_dev = variance.sqrt();

            // Calculate deviation from mean in terms of standard deviations
            let deviation = if std_dev > 1e-10 {
                (short_avg - long_avg) / std_dev
            } else {
                0.0
            };

            // Mean reversion signal: opposite of deviation
            // When sentiment is too high (deviation > 0), expect bearish reversion (negative signal)
            // When sentiment is too low (deviation < 0), expect bullish reversion (positive signal)
            result[i] = (-deviation * 33.0).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for SentimentMeanReversion {
    fn name(&self) -> &str {
        "Sentiment Mean Reversion"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Crowd Behavior Index - Index of crowd/herd behavior
/// Measures the degree of herding or crowd behavior in market sentiment
#[derive(Debug, Clone)]
pub struct CrowdBehaviorIndex {
    period: usize,
    threshold: f64,
}

impl CrowdBehaviorIndex {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if threshold <= 0.0 || threshold > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate crowd behavior index (0 to 100)
    /// Higher values indicate stronger herd behavior / crowd following
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // 1. Directional unanimity - how much agreement in direction
            let mut up_days = 0;
            let mut down_days = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_days += 1;
                } else if close[j] < close[j - 1] {
                    down_days += 1;
                }
            }
            let total_directional = up_days + down_days;
            let unanimity = if total_directional > 0 {
                (up_days.max(down_days) as f64) / (total_directional as f64)
            } else {
                0.5
            };

            // 2. Volume clustering - high volume when moving in the same direction
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let dominant_direction = if up_days > down_days { 1 } else { -1 };
            let mut vol_in_dominant_direction = 0.0;
            let mut total_vol = 0.0;
            for j in (start + 1)..=i {
                let direction = if close[j] > close[j - 1] { 1 } else if close[j] < close[j - 1] { -1 } else { 0 };
                if direction == dominant_direction && volume[j] > avg_vol * self.threshold {
                    vol_in_dominant_direction += volume[j];
                }
                total_vol += volume[j];
            }
            let vol_clustering = if total_vol > 0.0 {
                vol_in_dominant_direction / total_vol
            } else {
                0.0
            };

            // 3. Price pattern conformity - similar candle patterns
            let mut pattern_similarity = 0.0;
            let mut pattern_count = 0;
            for j in (start + 1)..=i {
                let range = high[j] - low[j];
                if range > 1e-10 {
                    let body_ratio = (close[j] - open[j]) / range;
                    // Check if pattern matches dominant direction
                    if (dominant_direction == 1 && body_ratio > 0.0) ||
                       (dominant_direction == -1 && body_ratio < 0.0) {
                        pattern_similarity += body_ratio.abs();
                    }
                    pattern_count += 1;
                }
            }
            let pattern_conformity = if pattern_count > 0 {
                pattern_similarity / pattern_count as f64
            } else {
                0.0
            };

            // 4. Consecutive moves - runs of same direction
            let mut max_consecutive = 0;
            let mut current_consecutive = 0;
            let mut last_direction = 0;
            for j in (start + 1)..=i {
                let direction = if close[j] > close[j - 1] { 1 } else if close[j] < close[j - 1] { -1 } else { 0 };
                if direction != 0 && direction == last_direction {
                    current_consecutive += 1;
                    max_consecutive = max_consecutive.max(current_consecutive);
                } else if direction != 0 {
                    current_consecutive = 1;
                    last_direction = direction;
                }
            }
            let consecutive_factor = (max_consecutive as f64) / (self.period as f64 / 2.0);

            // 5. Momentum alignment - closing prices consistently near highs or lows
            let mut alignment_sum = 0.0;
            for j in start..=i {
                let range = high[j] - low[j];
                if range > 1e-10 {
                    let position = (close[j] - low[j]) / range;
                    // Extreme positions indicate crowd behavior
                    if position > 0.8 || position < 0.2 {
                        alignment_sum += 1.0;
                    }
                }
            }
            let alignment_factor = alignment_sum / (self.period + 1) as f64;

            // Combine all factors into crowd behavior index
            result[i] = (
                unanimity * 25.0 +
                vol_clustering * 100.0 * 25.0 +
                pattern_conformity * 100.0 * 20.0 +
                consecutive_factor.min(1.0) * 15.0 +
                alignment_factor * 15.0
            ).clamp(0.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for CrowdBehaviorIndex {
    fn name(&self) -> &str {
        "Crowd Behavior Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Sentiment Regime Detector - Detects sentiment regime changes
/// Identifies transitions between bullish, bearish, and neutral sentiment regimes
#[derive(Debug, Clone)]
pub struct SentimentRegimeDetector {
    short_period: usize,
    long_period: usize,
    sensitivity: f64,
}

impl SentimentRegimeDetector {
    pub fn new(short_period: usize, long_period: usize, sensitivity: f64) -> Result<Self> {
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
        if sensitivity <= 0.0 || sensitivity > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sensitivity".to_string(),
                reason: "must be between 0 and 5".to_string(),
            });
        }
        Ok(Self { short_period, long_period, sensitivity })
    }

    /// Calculate sentiment regime (-100 to 100)
    /// Strong positive = bullish regime, Strong negative = bearish regime
    /// Values near 0 = neutral/transitioning regime
    /// Spikes indicate regime changes
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut sentiment = vec![0.0; n];

        // Calculate base sentiment
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                let position = (close[i] - low[i]) / range * 2.0 - 1.0;
                let momentum = if close[i - 1] > 0.0 {
                    ((close[i] / close[i - 1]) - 1.0) * 100.0
                } else {
                    0.0
                };

                let vol_factor = if i >= 5 {
                    let avg_vol: f64 = volume[(i - 5)..i].iter().sum::<f64>() / 5.0;
                    if avg_vol > 0.0 { (volume[i] / avg_vol).sqrt().min(2.0) } else { 1.0 }
                } else {
                    1.0
                };

                sentiment[i] = (position * 40.0 + momentum * 5.0) * vol_factor;
            }
        }

        // Detect regime and regime changes
        for i in self.long_period..n {
            // Calculate short-term regime
            let short_start = i - self.short_period;
            let short_avg: f64 = sentiment[short_start..=i].iter().sum::<f64>()
                / (self.short_period + 1) as f64;
            let short_var: f64 = sentiment[short_start..=i].iter()
                .map(|&x| (x - short_avg).powi(2))
                .sum::<f64>() / (self.short_period + 1) as f64;
            let short_std = short_var.sqrt();

            // Calculate long-term regime baseline
            let long_start = i - self.long_period;
            let long_avg: f64 = sentiment[long_start..=i].iter().sum::<f64>()
                / (self.long_period + 1) as f64;
            let long_var: f64 = sentiment[long_start..=i].iter()
                .map(|&x| (x - long_avg).powi(2))
                .sum::<f64>() / (self.long_period + 1) as f64;
            let long_std = long_var.sqrt();

            // Regime detection based on position relative to long-term average
            let regime_position = if long_std > 1e-10 {
                (short_avg - long_avg) / long_std
            } else {
                0.0
            };

            // Regime change detection - look for volatility expansion and direction shift
            let volatility_ratio = if long_std > 1e-10 {
                short_std / long_std
            } else {
                1.0
            };

            // Check for momentum shift
            let half_short = self.short_period / 2;
            let recent_avg: f64 = sentiment[(i - half_short)..=i].iter().sum::<f64>()
                / (half_short + 1) as f64;
            let older_avg: f64 = sentiment[short_start..(i - half_short)].iter().sum::<f64>()
                / half_short as f64;
            let momentum_shift = recent_avg - older_avg;

            // Combine factors for regime signal
            let base_regime = regime_position * 30.0;
            let change_signal = if volatility_ratio > 1.5 && momentum_shift.abs() > self.sensitivity * 10.0 {
                momentum_shift.signum() * (volatility_ratio - 1.0).min(1.0) * 30.0
            } else {
                0.0
            };

            // Current regime with change emphasis
            result[i] = (base_regime + change_signal + short_avg * 0.5).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for SentimentRegimeDetector {
    fn name(&self) -> &str {
        "Sentiment Regime Detector"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Contra Sentiment Signal - Contrarian signals from sentiment extremes
/// Generates signals when sentiment reaches extremes, suggesting potential reversals
#[derive(Debug, Clone)]
pub struct ContraSentimentSignal {
    period: usize,
    extreme_threshold: f64,
    confirmation_period: usize,
}

impl ContraSentimentSignal {
    pub fn new(period: usize, extreme_threshold: f64, confirmation_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if extreme_threshold <= 0.0 || extreme_threshold > 3.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "extreme_threshold".to_string(),
                reason: "must be between 0 and 3".to_string(),
            });
        }
        if confirmation_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "confirmation_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, extreme_threshold, confirmation_period })
    }

    /// Calculate contra sentiment signal (-100 to 100)
    /// Positive = contrarian bullish (sentiment extremely bearish)
    /// Negative = contrarian bearish (sentiment extremely bullish)
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut sentiment = vec![0.0; n];

        // Calculate comprehensive sentiment readings
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Price action sentiment
                let position = (close[i] - low[i]) / range * 2.0 - 1.0;
                let body = (close[i] - open[i]) / range;

                // Momentum sentiment
                let momentum = if close[i - 1] > 0.0 {
                    ((close[i] / close[i - 1]) - 1.0) * 100.0
                } else {
                    0.0
                };

                // Volume-weighted sentiment
                let vol_factor = if i >= 5 {
                    let avg_vol: f64 = volume[(i - 5)..i].iter().sum::<f64>() / 5.0;
                    if avg_vol > 0.0 {
                        let vol_ratio = volume[i] / avg_vol;
                        vol_ratio.sqrt().min(2.0)
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                sentiment[i] = (position * 30.0 + body * 20.0 + momentum * 3.0) * vol_factor;
            }
        }

        // Generate contrarian signals
        let total_lookback = self.period + self.confirmation_period;
        for i in total_lookback..n {
            let start = i - self.period;

            // Calculate sentiment statistics
            let slice = &sentiment[start..=i];
            let mean: f64 = slice.iter().sum::<f64>() / (self.period + 1) as f64;
            let variance: f64 = slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (self.period + 1) as f64;
            let std_dev = variance.sqrt();

            if std_dev < 1e-10 {
                continue;
            }

            // Calculate z-score of recent sentiment
            let recent_avg: f64 = sentiment[(i - self.confirmation_period)..=i].iter().sum::<f64>()
                / (self.confirmation_period + 1) as f64;
            let z_score = (recent_avg - mean) / std_dev;

            // Check for extreme sentiment
            if z_score.abs() >= self.extreme_threshold {
                // Look for early signs of reversal
                let very_recent = sentiment[i];
                let slightly_older = sentiment[i - self.confirmation_period];
                let reversal_hint = (very_recent - slightly_older).signum() != z_score.signum();

                // Calculate signal strength
                let extreme_intensity = ((z_score.abs() - self.extreme_threshold) / self.extreme_threshold).min(1.0);

                // Volume confirmation for reversal
                let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
                let recent_vol_factor = if avg_vol > 0.0 && volume[i] > avg_vol * 1.3 {
                    1.3
                } else {
                    1.0
                };

                // Generate contrarian signal (opposite of extreme sentiment)
                let base_signal = -z_score.signum() * (50.0 + extreme_intensity * 30.0);
                let reversal_bonus = if reversal_hint { 20.0 * (-z_score.signum()) } else { 0.0 };

                result[i] = ((base_signal + reversal_bonus) * recent_vol_factor).clamp(-100.0, 100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for ContraSentimentSignal {
    fn name(&self) -> &str {
        "Contra Sentiment Signal"
    }

    fn min_periods(&self) -> usize {
        self.period + self.confirmation_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume)))
    }
}

// ============================================================================
// New Sentiment Indicators
// ============================================================================

/// Price Based Sentiment - Sentiment derived from price action patterns
///
/// This indicator analyzes price movement characteristics to gauge market sentiment.
/// It considers price momentum, trend direction, and price position within recent ranges
/// to derive a comprehensive sentiment reading.
///
/// # Calculation
/// - Analyzes price momentum over the period
/// - Evaluates trend direction consistency
/// - Measures price position relative to recent highs/lows
/// - Combines factors into a normalized sentiment score
///
/// # Output Range
/// Returns values from -100 to 100:
/// - Positive values indicate bullish sentiment
/// - Negative values indicate bearish sentiment
/// - Values near 0 indicate neutral sentiment
#[derive(Debug, Clone)]
pub struct PriceBasedSentiment {
    period: usize,
    smoothing: usize,
}

impl PriceBasedSentiment {
    /// Creates a new PriceBasedSentiment indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for analysis (minimum 5)
    /// * `smoothing` - Smoothing period for the final output (minimum 1)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
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

    /// Calculate price-based sentiment (-100 to 100)
    ///
    /// # Arguments
    /// * `high` - Array of high prices
    /// * `low` - Array of low prices
    /// * `close` - Array of closing prices
    ///
    /// # Returns
    /// Vector of sentiment values for each data point
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];
        let mut raw_sentiment = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // 1. Price momentum component
            let momentum = if close[start] > 0.0 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };

            // 2. Trend direction consistency
            let mut up_count = 0;
            let mut down_count = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_count += 1;
                } else if close[j] < close[j - 1] {
                    down_count += 1;
                }
            }
            let trend_consistency = (up_count as f64 - down_count as f64) / self.period as f64;

            // 3. Price position within range
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;
            let position = if range > 1e-10 {
                (close[i] - period_low) / range * 2.0 - 1.0  // -1 to 1
            } else {
                0.0
            };

            // 4. Recent price strength (last few bars)
            let recent_bars = (self.period / 3).max(2);
            let recent_start = i - recent_bars;
            let recent_change = if close[recent_start] > 0.0 {
                (close[i] / close[recent_start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Combine components
            raw_sentiment[i] = (
                momentum * 0.3 +
                trend_consistency * 40.0 +
                position * 30.0 +
                recent_change * 0.4
            ).clamp(-100.0, 100.0);
        }

        // Apply smoothing
        let total_lookback = self.period + self.smoothing - 1;
        for i in total_lookback..n {
            let sum: f64 = raw_sentiment[(i - self.smoothing + 1)..=i].iter().sum();
            result[i] = (sum / self.smoothing as f64).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for PriceBasedSentiment {
    fn name(&self) -> &str {
        "Price Based Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Volume Sentiment Pattern - Sentiment derived from volume patterns
///
/// This indicator analyzes volume patterns in conjunction with price movements
/// to determine market sentiment. High volume on up moves indicates bullish sentiment,
/// while high volume on down moves indicates bearish sentiment.
///
/// # Calculation
/// - Tracks volume on up vs down days
/// - Analyzes volume trends and spikes
/// - Measures volume-price correlation
/// - Identifies accumulation vs distribution patterns
///
/// # Output Range
/// Returns values from -100 to 100:
/// - Positive values indicate bullish volume sentiment (accumulation)
/// - Negative values indicate bearish volume sentiment (distribution)
#[derive(Debug, Clone)]
pub struct VolumeSentimentPattern {
    period: usize,
    sensitivity: f64,
}

impl VolumeSentimentPattern {
    /// Creates a new VolumeSentimentPattern indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for analysis (minimum 5)
    /// * `sensitivity` - Sensitivity to volume changes (0.5 to 3.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, sensitivity: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if sensitivity < 0.5 || sensitivity > 3.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sensitivity".to_string(),
                reason: "must be between 0.5 and 3.0".to_string(),
            });
        }
        Ok(Self { period, sensitivity })
    }

    /// Calculate volume-based sentiment pattern (-100 to 100)
    ///
    /// # Arguments
    /// * `high` - Array of high prices
    /// * `low` - Array of low prices
    /// * `close` - Array of closing prices
    /// * `volume` - Array of volume values
    ///
    /// # Returns
    /// Vector of sentiment values for each data point
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate average volume
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            if avg_vol <= 0.0 {
                continue;
            }

            // 1. Volume-weighted price movement
            let mut up_vol_weighted = 0.0;
            let mut down_vol_weighted = 0.0;
            for j in (start + 1)..=i {
                let vol_ratio = (volume[j] / avg_vol).powf(self.sensitivity);
                if close[j] > close[j - 1] {
                    up_vol_weighted += vol_ratio;
                } else if close[j] < close[j - 1] {
                    down_vol_weighted += vol_ratio;
                }
            }

            // 2. Close position within bar (accumulation/distribution)
            let mut accum_dist_score = 0.0;
            for j in start..=i {
                let range = high[j] - low[j];
                if range > 1e-10 {
                    // Money Flow Multiplier concept
                    let mf_mult = ((close[j] - low[j]) - (high[j] - close[j])) / range;
                    accum_dist_score += mf_mult * (volume[j] / avg_vol);
                }
            }
            accum_dist_score /= (self.period + 1) as f64;

            // 3. Volume trend (increasing or decreasing)
            let first_half_vol: f64 = volume[start..(start + self.period / 2)].iter().sum::<f64>()
                / (self.period / 2) as f64;
            let second_half_vol: f64 = volume[(i - self.period / 2)..=i].iter().sum::<f64>()
                / (self.period / 2 + 1) as f64;
            let vol_trend = if first_half_vol > 0.0 {
                (second_half_vol / first_half_vol - 1.0) * 50.0
            } else {
                0.0
            };

            // 4. Volume spike detection with direction
            let vol_spike = if volume[i] > avg_vol * 1.5 {
                let direction = if close[i] > close[i - 1] { 1.0 } else { -1.0 };
                direction * ((volume[i] / avg_vol) - 1.0).min(2.0) * 20.0
            } else {
                0.0
            };

            // Combine all components
            let total_vol_weighted = up_vol_weighted + down_vol_weighted;
            let vol_direction = if total_vol_weighted > 0.0 {
                (up_vol_weighted - down_vol_weighted) / total_vol_weighted * 40.0
            } else {
                0.0
            };

            result[i] = (
                vol_direction +
                accum_dist_score * 30.0 +
                vol_trend.clamp(-20.0, 20.0) +
                vol_spike
            ).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for VolumeSentimentPattern {
    fn name(&self) -> &str {
        "Volume Sentiment Pattern"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Momentum Sentiment Index - Sentiment derived from momentum indicators
///
/// This indicator combines multiple momentum-based measurements to derive
/// a comprehensive sentiment reading. It analyzes rate of change, momentum
/// consistency, and momentum acceleration.
///
/// # Calculation
/// - Calculates rate of change (ROC)
/// - Measures RSI-like momentum bias
/// - Analyzes momentum acceleration/deceleration
/// - Evaluates momentum consistency over time
///
/// # Output Range
/// Returns values from -100 to 100:
/// - Positive values indicate bullish momentum sentiment
/// - Negative values indicate bearish momentum sentiment
#[derive(Debug, Clone)]
pub struct MomentumSentimentIndex {
    period: usize,
    roc_period: usize,
}

impl MomentumSentimentIndex {
    /// Creates a new MomentumSentimentIndex indicator
    ///
    /// # Arguments
    /// * `period` - Main lookback period for analysis (minimum 5)
    /// * `roc_period` - Period for rate of change calculation (minimum 2)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, roc_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if roc_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "roc_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, roc_period })
    }

    /// Calculate momentum-based sentiment index (-100 to 100)
    ///
    /// # Arguments
    /// * `close` - Array of closing prices
    ///
    /// # Returns
    /// Vector of sentiment index values for each data point
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let max_lookback = self.period.max(self.roc_period);
        let mut result = vec![0.0; n];

        for i in max_lookback..n {
            // 1. Rate of Change component
            let roc = if close[i - self.roc_period] > 0.0 {
                (close[i] / close[i - self.roc_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // 2. RSI-like momentum bias
            let start = i - self.period;
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
            let rsi_bias = if gains + losses > 0.0 {
                (gains / (gains + losses) - 0.5) * 200.0
            } else {
                0.0
            };

            // 3. Momentum acceleration
            let mid = i - self.period / 2;
            let first_half_mom = if close[start] > 0.0 {
                (close[mid] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };
            let second_half_mom = if close[mid] > 0.0 {
                (close[i] / close[mid] - 1.0) * 100.0
            } else {
                0.0
            };
            let acceleration = (second_half_mom - first_half_mom).clamp(-50.0, 50.0);

            // 4. Momentum consistency (streak analysis)
            let mut streak = 0i32;
            let mut max_streak = 0i32;
            let mut last_direction = 0i32;
            for j in (start + 1)..=i {
                let direction = if close[j] > close[j - 1] { 1 } else if close[j] < close[j - 1] { -1 } else { 0 };
                if direction != 0 && direction == last_direction {
                    streak += direction;
                } else if direction != 0 {
                    if streak.abs() > max_streak.abs() {
                        max_streak = streak;
                    }
                    streak = direction;
                    last_direction = direction;
                }
            }
            if streak.abs() > max_streak.abs() {
                max_streak = streak;
            }
            let consistency_score = (max_streak as f64 / (self.period as f64 / 3.0) * 20.0).clamp(-30.0, 30.0);

            // Combine components with weights
            result[i] = (
                roc * 0.25 +
                rsi_bias * 0.35 +
                acceleration * 0.25 +
                consistency_score
            ).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for MomentumSentimentIndex {
    fn name(&self) -> &str {
        "Momentum Sentiment Index"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.roc_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Extreme Sentiment Detector - Detects extreme sentiment levels
///
/// This indicator identifies when market sentiment reaches extreme levels
/// that may indicate potential reversals. It uses statistical methods to
/// detect outliers in sentiment readings.
///
/// # Calculation
/// - Calculates base sentiment from price and volume
/// - Computes rolling statistics (mean, standard deviation)
/// - Identifies readings beyond threshold standard deviations
/// - Outputs signal strength based on extremity level
///
/// # Output Range
/// Returns values from -100 to 100:
/// - Strong positive values indicate extreme bullish sentiment
/// - Strong negative values indicate extreme bearish sentiment
/// - Values near 0 indicate normal/non-extreme sentiment
#[derive(Debug, Clone)]
pub struct ExtremeSentimentDetector {
    period: usize,
    z_threshold: f64,
    lookback_multiple: usize,
}

impl ExtremeSentimentDetector {
    /// Creates a new ExtremeSentimentDetector indicator
    ///
    /// # Arguments
    /// * `period` - Base period for sentiment calculation (minimum 5)
    /// * `z_threshold` - Z-score threshold for extreme detection (1.0 to 4.0)
    /// * `lookback_multiple` - Multiple of period for historical comparison (minimum 2)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, z_threshold: f64, lookback_multiple: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if z_threshold < 1.0 || z_threshold > 4.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "z_threshold".to_string(),
                reason: "must be between 1.0 and 4.0".to_string(),
            });
        }
        if lookback_multiple < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_multiple".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, z_threshold, lookback_multiple })
    }

    /// Calculate extreme sentiment detection (-100 to 100)
    ///
    /// # Arguments
    /// * `high` - Array of high prices
    /// * `low` - Array of low prices
    /// * `close` - Array of closing prices
    /// * `volume` - Array of volume values
    ///
    /// # Returns
    /// Vector of extreme sentiment signals for each data point
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut sentiment_readings = vec![0.0; n];
        let lookback = self.period * self.lookback_multiple;

        // First pass: Calculate base sentiment readings
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Position-based sentiment
                let position = (close[i] - low[i]) / range * 2.0 - 1.0;

                // Momentum-based sentiment
                let momentum = if close[i - 1] > 0.0 {
                    ((close[i] / close[i - 1]) - 1.0) * 100.0
                } else {
                    0.0
                };

                // Volume factor
                let vol_factor = if i >= self.period {
                    let avg_vol: f64 = volume[(i - self.period)..i].iter().sum::<f64>() / self.period as f64;
                    if avg_vol > 0.0 { (volume[i] / avg_vol).sqrt().min(2.0) } else { 1.0 }
                } else {
                    1.0
                };

                sentiment_readings[i] = (position * 40.0 + momentum * 4.0) * vol_factor;
            }
        }

        // Second pass: Detect extremes using z-score
        for i in lookback..n {
            let start = i - lookback;
            let slice = &sentiment_readings[start..=i];

            // Calculate mean and standard deviation
            let mean: f64 = slice.iter().sum::<f64>() / (lookback + 1) as f64;
            let variance: f64 = slice.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (lookback + 1) as f64;
            let std_dev = variance.sqrt();

            if std_dev < 1e-10 {
                continue;
            }

            // Calculate z-score of recent sentiment average
            let recent_avg: f64 = sentiment_readings[(i - self.period)..=i].iter().sum::<f64>()
                / (self.period + 1) as f64;
            let z_score = (recent_avg - mean) / std_dev;

            // Check for extreme
            if z_score.abs() >= self.z_threshold {
                // Calculate signal intensity
                let excess = z_score.abs() - self.z_threshold;
                let intensity = (1.0 + excess / self.z_threshold).min(2.0);

                // Check for potential exhaustion (volume spike at extreme)
                let avg_vol: f64 = volume[(i - self.period)..i].iter().sum::<f64>() / self.period as f64;
                let vol_spike = if avg_vol > 0.0 && volume[i] > avg_vol * 1.5 { 1.2 } else { 1.0 };

                // Output signal indicating extreme level
                result[i] = (z_score.signum() * intensity * vol_spike * 50.0).clamp(-100.0, 100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for ExtremeSentimentDetector {
    fn name(&self) -> &str {
        "Extreme Sentiment Detector"
    }

    fn min_periods(&self) -> usize {
        self.period * self.lookback_multiple + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Sentiment Oscillator - Oscillator showing sentiment shifts
///
/// This indicator creates an oscillating sentiment measure that helps identify
/// sentiment shifts and potential turning points in market psychology.
///
/// # Calculation
/// - Calculates fast and slow sentiment averages
/// - Computes oscillation as the difference between fast and slow
/// - Applies smoothing to reduce noise
/// - Normalizes output to oscillator range
///
/// # Output Range
/// Returns values from -100 to 100:
/// - Rising values indicate improving sentiment
/// - Falling values indicate deteriorating sentiment
/// - Zero crossings indicate sentiment shifts
#[derive(Debug, Clone)]
pub struct SentimentOscillator {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
}

impl SentimentOscillator {
    /// Creates a new SentimentOscillator indicator
    ///
    /// # Arguments
    /// * `fast_period` - Fast sentiment calculation period (minimum 3)
    /// * `slow_period` - Slow sentiment calculation period (must be > fast_period)
    /// * `signal_period` - Signal line smoothing period (minimum 1)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Result<Self> {
        if fast_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if slow_period <= fast_period {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_period".to_string(),
                reason: "must be greater than fast_period".to_string(),
            });
        }
        if signal_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "signal_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { fast_period, slow_period, signal_period })
    }

    /// Calculate sentiment oscillator (-100 to 100)
    ///
    /// # Arguments
    /// * `high` - Array of high prices
    /// * `low` - Array of low prices
    /// * `close` - Array of closing prices
    /// * `volume` - Array of volume values
    ///
    /// # Returns
    /// Vector of oscillator values for each data point
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut base_sentiment = vec![0.0; n];
        let mut oscillator = vec![0.0; n];

        // First pass: Calculate base sentiment for each bar
        for i in 1..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Price position sentiment
                let position = (close[i] - low[i]) / range * 2.0 - 1.0;

                // Momentum sentiment
                let momentum = if close[i - 1] > 0.0 {
                    ((close[i] / close[i - 1]) - 1.0) * 100.0
                } else {
                    0.0
                };

                // Body direction sentiment
                let body = (close[i] - high[i].min(low[i])) / range;

                // Volume weighting
                let vol_weight = if i >= 5 {
                    let avg_vol: f64 = volume[(i.saturating_sub(5))..i].iter().sum::<f64>() / 5.0;
                    if avg_vol > 0.0 { (volume[i] / avg_vol).sqrt().min(2.0) } else { 1.0 }
                } else {
                    1.0
                };

                base_sentiment[i] = (position * 35.0 + momentum * 3.0 + body * 15.0) * vol_weight;
            }
        }

        // Second pass: Calculate oscillator (fast - slow)
        for i in self.slow_period..n {
            // Fast average
            let fast_start = i - self.fast_period;
            let fast_avg: f64 = base_sentiment[fast_start..=i].iter().sum::<f64>()
                / (self.fast_period + 1) as f64;

            // Slow average
            let slow_start = i - self.slow_period;
            let slow_avg: f64 = base_sentiment[slow_start..=i].iter().sum::<f64>()
                / (self.slow_period + 1) as f64;

            // Oscillator is the difference
            oscillator[i] = fast_avg - slow_avg;
        }

        // Third pass: Apply signal smoothing
        let total_lookback = self.slow_period + self.signal_period - 1;
        for i in total_lookback..n {
            let sum: f64 = oscillator[(i - self.signal_period + 1)..=i].iter().sum();
            result[i] = (sum / self.signal_period as f64 * 2.0).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for SentimentOscillator {
    fn name(&self) -> &str {
        "Sentiment Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.slow_period + self.signal_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Composite Sentiment Index - Combined sentiment measure
///
/// This indicator creates a comprehensive sentiment index by combining
/// multiple sentiment factors including price action, volume, momentum,
/// and volatility-adjusted readings.
///
/// # Calculation
/// - Calculates price action sentiment component
/// - Calculates volume sentiment component
/// - Calculates momentum sentiment component
/// - Calculates volatility-adjusted component
/// - Combines with configurable weights
///
/// # Output Range
/// Returns values from -100 to 100:
/// - Positive values indicate overall bullish sentiment
/// - Negative values indicate overall bearish sentiment
/// - Higher absolute values indicate stronger sentiment
#[derive(Debug, Clone)]
pub struct CompositeSentimentIndex {
    period: usize,
    momentum_weight: f64,
    volume_weight: f64,
    price_action_weight: f64,
    volatility_weight: f64,
}

impl CompositeSentimentIndex {
    /// Creates a new CompositeSentimentIndex indicator with default weights
    ///
    /// # Arguments
    /// * `period` - Lookback period for all components (minimum 10)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        Self::with_weights(period, 0.30, 0.25, 0.30, 0.15)
    }

    /// Creates a new CompositeSentimentIndex indicator with custom weights
    ///
    /// # Arguments
    /// * `period` - Lookback period for all components (minimum 10)
    /// * `momentum_weight` - Weight for momentum component (0.0 to 1.0)
    /// * `volume_weight` - Weight for volume component (0.0 to 1.0)
    /// * `price_action_weight` - Weight for price action component (0.0 to 1.0)
    /// * `volatility_weight` - Weight for volatility component (0.0 to 1.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn with_weights(
        period: usize,
        momentum_weight: f64,
        volume_weight: f64,
        price_action_weight: f64,
        volatility_weight: f64,
    ) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        let total = momentum_weight + volume_weight + price_action_weight + volatility_weight;
        if (total - 1.0).abs() > 0.01 {
            return Err(IndicatorError::InvalidParameter {
                name: "weights".to_string(),
                reason: "must sum to 1.0".to_string(),
            });
        }
        Ok(Self {
            period,
            momentum_weight,
            volume_weight,
            price_action_weight,
            volatility_weight,
        })
    }

    /// Calculate composite sentiment index (-100 to 100)
    ///
    /// # Arguments
    /// * `open` - Array of opening prices
    /// * `high` - Array of high prices
    /// * `low` - Array of low prices
    /// * `close` - Array of closing prices
    /// * `volume` - Array of volume values
    ///
    /// # Returns
    /// Vector of composite sentiment values for each data point
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // 1. Momentum Component
            let momentum_roc = if close[start] > 0.0 {
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
            let rsi_sentiment = if gains + losses > 0.0 {
                (gains / (gains + losses) - 0.5) * 200.0
            } else {
                0.0
            };
            let momentum_score = ((momentum_roc * 1.5 + rsi_sentiment) / 2.0).clamp(-100.0, 100.0);

            // 2. Volume Component
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let mut up_vol = 0.0;
            let mut down_vol = 0.0;
            for j in (start + 1)..=i {
                let vol_ratio = if avg_vol > 0.0 { volume[j] / avg_vol } else { 1.0 };
                if close[j] > close[j - 1] {
                    up_vol += vol_ratio;
                } else if close[j] < close[j - 1] {
                    down_vol += vol_ratio;
                }
            }
            let volume_score = if up_vol + down_vol > 0.0 {
                ((up_vol - down_vol) / (up_vol + down_vol) * 100.0).clamp(-100.0, 100.0)
            } else {
                0.0
            };

            // 3. Price Action Component
            let mut price_action_sum = 0.0;
            for j in start..=i {
                let range = high[j] - low[j];
                if range > 1e-10 {
                    let body = close[j] - open[j];
                    let body_ratio = body / range;
                    let position = (close[j] - low[j]) / range * 2.0 - 1.0;
                    price_action_sum += body_ratio * 50.0 + position * 25.0;
                }
            }
            let price_action_score = (price_action_sum / (self.period + 1) as f64).clamp(-100.0, 100.0);

            // 4. Volatility-Adjusted Component
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;
            let position_in_range = if range > 1e-10 {
                (close[i] - period_low) / range * 2.0 - 1.0
            } else {
                0.0
            };

            // ATR-based volatility
            let mut tr_sum = 0.0;
            for j in (start + 1)..=i {
                let tr1 = high[j] - low[j];
                let tr2 = (high[j] - close[j - 1]).abs();
                let tr3 = (low[j] - close[j - 1]).abs();
                tr_sum += tr1.max(tr2).max(tr3);
            }
            let atr = tr_sum / self.period as f64;
            let vol_normalized = if close[i] > 0.0 { atr / close[i] * 100.0 } else { 0.0 };
            // High volatility at extremes intensifies the signal
            let vol_multiplier = if vol_normalized > 2.0 { 1.2 } else { 1.0 };
            let volatility_score = (position_in_range * 100.0 * vol_multiplier).clamp(-100.0, 100.0);

            // Combine all components
            result[i] = (
                momentum_score * self.momentum_weight +
                volume_score * self.volume_weight +
                price_action_score * self.price_action_weight +
                volatility_score * self.volatility_weight
            ).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for CompositeSentimentIndex {
    fn name(&self) -> &str {
        "Composite Sentiment Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume)))
    }
}

// ============================================================================
// Additional NEW Sentiment Indicators
// ============================================================================

/// Fear Greed Proxy - Fear/greed proxy derived from price action
///
/// This indicator analyzes price action characteristics to estimate whether
/// the market is driven by fear (panic selling) or greed (euphoric buying).
/// It combines multiple factors including volatility, momentum, and volume
/// patterns to create a fear/greed scale.
///
/// # Calculation
/// - Analyzes price volatility relative to historical norms
/// - Measures momentum and acceleration of price moves
/// - Evaluates volume patterns during price extremes
/// - Combines factors into a fear-greed scale
///
/// # Output Range
/// Returns values from -100 to 100:
/// - Strong negative values (-100 to -50) indicate extreme fear
/// - Moderate negative values (-50 to -20) indicate fear
/// - Values near 0 (-20 to 20) indicate neutral sentiment
/// - Moderate positive values (20 to 50) indicate greed
/// - Strong positive values (50 to 100) indicate extreme greed
#[derive(Debug, Clone)]
pub struct FearGreedProxy {
    period: usize,
    volatility_period: usize,
    smoothing: usize,
}

impl FearGreedProxy {
    /// Creates a new FearGreedProxy indicator
    ///
    /// # Arguments
    /// * `period` - Main lookback period for analysis (minimum 10)
    /// * `volatility_period` - Period for volatility calculation (minimum 5)
    /// * `smoothing` - Smoothing period for the final output (minimum 1)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, volatility_period: usize, smoothing: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smoothing < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, volatility_period, smoothing })
    }

    /// Calculate fear/greed proxy (-100 to 100)
    ///
    /// # Arguments
    /// * `open` - Array of opening prices
    /// * `high` - Array of high prices
    /// * `low` - Array of low prices
    /// * `close` - Array of closing prices
    /// * `volume` - Array of volume values
    ///
    /// # Returns
    /// Vector of fear/greed proxy values for each data point
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut raw_fg = vec![0.0; n];
        let max_period = self.period.max(self.volatility_period);

        for i in max_period..n {
            let start = i - self.period;
            let vol_start = i - self.volatility_period;

            // 1. Momentum component (greed when strongly positive, fear when negative)
            let momentum = if close[start] > 0.0 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };
            let momentum_score = (momentum * 3.0).clamp(-40.0, 40.0);

            // 2. Volatility component (high volatility often indicates fear)
            let mut tr_sum = 0.0;
            for j in (vol_start + 1)..=i {
                let tr1 = high[j] - low[j];
                let tr2 = (high[j] - close[j - 1]).abs();
                let tr3 = (low[j] - close[j - 1]).abs();
                tr_sum += tr1.max(tr2).max(tr3);
            }
            let current_atr = tr_sum / self.volatility_period as f64;

            // Historical ATR for comparison
            let hist_start = start.saturating_sub(self.volatility_period);
            let mut hist_tr_sum = 0.0;
            let hist_count = (start - hist_start).max(1);
            for j in (hist_start + 1)..=start {
                if j < n {
                    let tr1 = high[j] - low[j];
                    let tr2 = (high[j] - close[j - 1]).abs();
                    let tr3 = (low[j] - close[j - 1]).abs();
                    hist_tr_sum += tr1.max(tr2).max(tr3);
                }
            }
            let hist_atr = if hist_count > 0 { hist_tr_sum / hist_count as f64 } else { current_atr };

            let vol_ratio = if hist_atr > 1e-10 { current_atr / hist_atr } else { 1.0 };
            // High volatility during downtrend = fear, high volatility during uptrend = greed
            let vol_direction = if momentum > 0.0 { 1.0 } else { -1.0 };
            let volatility_score = if vol_ratio > 1.2 {
                vol_direction * (vol_ratio - 1.0).min(1.5) * 15.0
            } else {
                0.0
            };

            // 3. Price position component (near highs = greed, near lows = fear)
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;
            let position = if range > 1e-10 {
                (close[i] - period_low) / range * 2.0 - 1.0  // -1 to 1
            } else {
                0.0
            };
            let position_score = position * 25.0;

            // 4. Volume confirmation (high volume confirms the direction)
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let recent_vol_ratio = if avg_vol > 0.0 { volume[i] / avg_vol } else { 1.0 };
            let vol_confirm = if recent_vol_ratio > 1.3 {
                momentum_score.signum() * (recent_vol_ratio - 1.0).min(1.0) * 10.0
            } else {
                0.0
            };

            // 5. Candle body analysis (large bodies indicate conviction)
            let mut body_sentiment = 0.0;
            for j in (i - 3.min(self.period))..=i {
                let bar_range = high[j] - low[j];
                if bar_range > 1e-10 {
                    let body = close[j] - open[j];
                    let body_ratio = body / bar_range;
                    body_sentiment += body_ratio;
                }
            }
            let body_score = (body_sentiment * 5.0).clamp(-15.0, 15.0);

            // Combine all components
            raw_fg[i] = (momentum_score + volatility_score + position_score + vol_confirm + body_score)
                .clamp(-100.0, 100.0);
        }

        // Apply smoothing
        let total_lookback = max_period + self.smoothing - 1;
        for i in total_lookback..n {
            let sum: f64 = raw_fg[(i - self.smoothing + 1)..=i].iter().sum();
            result[i] = (sum / self.smoothing as f64).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for FearGreedProxy {
    fn name(&self) -> &str {
        "Fear Greed Proxy"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.volatility_period) + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Market Panic Index - Detects panic selling conditions
///
/// This indicator identifies market panic by analyzing rapid price declines,
/// volume spikes, and volatility expansion that characterize panic selling.
/// It helps identify capitulation events and potential market bottoms.
///
/// # Calculation
/// - Measures rate of price decline
/// - Analyzes volume spikes during declines
/// - Evaluates volatility expansion
/// - Tracks consecutive down days with increasing volume
///
/// # Output Range
/// Returns values from 0 to 100:
/// - Values 0-20 indicate no panic
/// - Values 20-50 indicate mild panic
/// - Values 50-75 indicate moderate panic
/// - Values 75-100 indicate extreme panic
#[derive(Debug, Clone)]
pub struct MarketPanicIndex {
    period: usize,
    panic_threshold: f64,
}

impl MarketPanicIndex {
    /// Creates a new MarketPanicIndex indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for panic detection (minimum 5)
    /// * `panic_threshold` - Sensitivity threshold for panic detection (0.5 to 3.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, panic_threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if panic_threshold < 0.5 || panic_threshold > 3.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "panic_threshold".to_string(),
                reason: "must be between 0.5 and 3.0".to_string(),
            });
        }
        Ok(Self { period, panic_threshold })
    }

    /// Calculate market panic index (0 to 100)
    ///
    /// # Arguments
    /// * `high` - Array of high prices
    /// * `low` - Array of low prices
    /// * `close` - Array of closing prices
    /// * `volume` - Array of volume values
    ///
    /// # Returns
    /// Vector of panic index values for each data point
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // 1. Price decline component
            let price_change = if close[start] > 0.0 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };
            // Only consider declines as panic-inducing
            let decline_score = if price_change < 0.0 {
                (price_change.abs() * 2.0).min(40.0)
            } else {
                0.0
            };

            // 2. Consecutive down days
            let mut consecutive_down = 0;
            for j in (start + 1)..=i {
                if close[j] < close[j - 1] {
                    consecutive_down += 1;
                } else {
                    consecutive_down = 0;
                }
            }
            let consecutive_score = (consecutive_down as f64 * 5.0).min(20.0);

            // 3. Volume spike on down days
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let mut panic_vol_score = 0.0;
            for j in (start + 1)..=i {
                if close[j] < close[j - 1] && avg_vol > 0.0 {
                    let vol_ratio = volume[j] / avg_vol;
                    if vol_ratio > self.panic_threshold {
                        panic_vol_score += (vol_ratio - 1.0).min(2.0) * 5.0;
                    }
                }
            }
            panic_vol_score = panic_vol_score.min(25.0);

            // 4. Volatility expansion
            let mut tr_sum = 0.0;
            for j in (start + 1)..=i {
                let tr1 = high[j] - low[j];
                let tr2 = (high[j] - close[j - 1]).abs();
                let tr3 = (low[j] - close[j - 1]).abs();
                tr_sum += tr1.max(tr2).max(tr3);
            }
            let atr = tr_sum / self.period as f64;
            let volatility_pct = if close[i] > 0.0 { atr / close[i] * 100.0 } else { 0.0 };
            let volatility_score = (volatility_pct * 3.0).min(20.0);

            // 5. Gap down detection (panic often creates gaps)
            let mut gap_score = 0.0;
            for j in (start + 1)..=i {
                if high[j] < low[j - 1] {  // Gap down
                    let gap_size = (low[j - 1] - high[j]) / close[j - 1] * 100.0;
                    gap_score += gap_size.min(5.0);
                }
            }
            gap_score = gap_score.min(15.0);

            // Combine components (only positive = panic)
            result[i] = (decline_score + consecutive_score + panic_vol_score + volatility_score + gap_score)
                .clamp(0.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for MarketPanicIndex {
    fn name(&self) -> &str {
        "Market Panic Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Euphoria Detector - Detects euphoric buying conditions
///
/// This indicator identifies market euphoria by analyzing rapid price advances,
/// volume surges during rallies, and extreme optimism patterns. It helps identify
/// potential market tops and excessive bullish sentiment.
///
/// # Calculation
/// - Measures rate of price advance
/// - Analyzes volume surges during advances
/// - Evaluates price position relative to recent range
/// - Tracks consecutive up days with increasing volume
///
/// # Output Range
/// Returns values from 0 to 100:
/// - Values 0-20 indicate no euphoria
/// - Values 20-50 indicate mild euphoria
/// - Values 50-75 indicate moderate euphoria
/// - Values 75-100 indicate extreme euphoria
#[derive(Debug, Clone)]
pub struct EuphoriaDetector {
    period: usize,
    euphoria_threshold: f64,
}

impl EuphoriaDetector {
    /// Creates a new EuphoriaDetector indicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for euphoria detection (minimum 5)
    /// * `euphoria_threshold` - Sensitivity threshold for euphoria detection (0.5 to 3.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, euphoria_threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if euphoria_threshold < 0.5 || euphoria_threshold > 3.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "euphoria_threshold".to_string(),
                reason: "must be between 0.5 and 3.0".to_string(),
            });
        }
        Ok(Self { period, euphoria_threshold })
    }

    /// Calculate euphoria detection (0 to 100)
    ///
    /// # Arguments
    /// * `open` - Array of opening prices
    /// * `high` - Array of high prices
    /// * `low` - Array of low prices
    /// * `close` - Array of closing prices
    /// * `volume` - Array of volume values
    ///
    /// # Returns
    /// Vector of euphoria values for each data point
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // 1. Price advance component
            let price_change = if close[start] > 0.0 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };
            // Only consider advances as euphoria-inducing
            let advance_score = if price_change > 0.0 {
                (price_change * 2.5).min(40.0)
            } else {
                0.0
            };

            // 2. Consecutive up days
            let mut consecutive_up = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    consecutive_up += 1;
                } else {
                    consecutive_up = 0;
                }
            }
            let consecutive_score = (consecutive_up as f64 * 5.0).min(20.0);

            // 3. Volume surge on up days
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let mut euphoric_vol_score = 0.0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] && avg_vol > 0.0 {
                    let vol_ratio = volume[j] / avg_vol;
                    if vol_ratio > self.euphoria_threshold {
                        euphoric_vol_score += (vol_ratio - 1.0).min(2.0) * 5.0;
                    }
                }
            }
            euphoric_vol_score = euphoric_vol_score.min(25.0);

            // 4. Price at period highs (closing near highs indicates euphoria)
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;
            let position = if range > 1e-10 {
                (close[i] - period_low) / range
            } else {
                0.5
            };
            let position_score = if position > 0.8 {
                (position - 0.5) * 30.0
            } else {
                0.0
            };

            // 5. Strong bullish candles
            let mut bullish_candle_score = 0.0;
            for j in start..=i {
                let bar_range = high[j] - low[j];
                if bar_range > 1e-10 {
                    let body = close[j] - open[j];
                    let body_ratio = body / bar_range;
                    if body_ratio > 0.6 {  // Strong bullish candle
                        bullish_candle_score += body_ratio * 3.0;
                    }
                }
            }
            bullish_candle_score = bullish_candle_score.min(15.0);

            // 6. Gap up detection (euphoria often creates gaps up)
            let mut gap_score = 0.0;
            for j in (start + 1)..=i {
                if low[j] > high[j - 1] {  // Gap up
                    let gap_size = (low[j] - high[j - 1]) / close[j - 1] * 100.0;
                    gap_score += gap_size.min(3.0);
                }
            }
            gap_score = gap_score.min(10.0);

            // Combine components
            result[i] = (advance_score + consecutive_score + euphoric_vol_score +
                        position_score + bullish_candle_score + gap_score)
                .clamp(0.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for EuphoriaDetector {
    fn name(&self) -> &str {
        "Euphoria Detector"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Sentiment Strength Indicator - Measures overall sentiment strength
///
/// This indicator measures the overall strength and conviction of market sentiment,
/// regardless of direction. It helps identify periods of strong conviction
/// (whether bullish or bearish) versus periods of indecision.
///
/// # Calculation
/// - Measures directional consistency
/// - Analyzes volume confirmation of moves
/// - Evaluates price momentum strength
/// - Tracks body-to-range ratios for conviction
///
/// # Output Range
/// Returns values from 0 to 100:
/// - Values 0-30 indicate weak sentiment/indecision
/// - Values 30-60 indicate moderate sentiment strength
/// - Values 60-100 indicate strong sentiment conviction
#[derive(Debug, Clone)]
pub struct SentimentStrengthIndicator {
    period: usize,
    smoothing: usize,
}

impl SentimentStrengthIndicator {
    /// Creates a new SentimentStrengthIndicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for analysis (minimum 5)
    /// * `smoothing` - Smoothing period for the final output (minimum 1)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
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

    /// Calculate sentiment strength (0 to 100)
    ///
    /// # Arguments
    /// * `open` - Array of opening prices
    /// * `high` - Array of high prices
    /// * `low` - Array of low prices
    /// * `close` - Array of closing prices
    /// * `volume` - Array of volume values
    ///
    /// # Returns
    /// Vector of sentiment strength values for each data point
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let mut raw_strength = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // 1. Directional consistency (strength of trend regardless of direction)
            let mut up_count = 0;
            let mut down_count = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_count += 1;
                } else if close[j] < close[j - 1] {
                    down_count += 1;
                }
            }
            let dominant_direction = up_count.max(down_count);
            let consistency = (dominant_direction as f64) / self.period as f64;
            let consistency_score = consistency * 30.0;

            // 2. Momentum strength (absolute momentum)
            let momentum = if close[start] > 0.0 {
                (close[i] / close[start] - 1.0).abs() * 100.0
            } else {
                0.0
            };
            let momentum_score = (momentum * 2.0).min(25.0);

            // 3. Volume confirmation strength
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let mut vol_confirmed = 0;
            let mut total_moves = 0;
            for j in (start + 1)..=i {
                if close[j] != close[j - 1] {
                    total_moves += 1;
                    if avg_vol > 0.0 && volume[j] > avg_vol {
                        vol_confirmed += 1;
                    }
                }
            }
            let vol_confirm_ratio = if total_moves > 0 {
                vol_confirmed as f64 / total_moves as f64
            } else {
                0.0
            };
            let vol_score = vol_confirm_ratio * 20.0;

            // 4. Body strength (large bodies indicate conviction)
            let mut body_strength_sum = 0.0;
            for j in start..=i {
                let bar_range = high[j] - low[j];
                if bar_range > 1e-10 {
                    let body = (close[j] - open[j]).abs();
                    body_strength_sum += body / bar_range;
                }
            }
            let body_strength = body_strength_sum / (self.period + 1) as f64;
            let body_score = body_strength * 25.0;

            // 5. Range expansion (larger ranges indicate stronger sentiment)
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;
            let range_pct = if close[i] > 0.0 { range / close[i] * 100.0 } else { 0.0 };
            let range_score = (range_pct * 2.0).min(15.0);

            // Combine components
            raw_strength[i] = (consistency_score + momentum_score + vol_score + body_score + range_score)
                .clamp(0.0, 100.0);
        }

        // Apply smoothing
        let total_lookback = self.period + self.smoothing - 1;
        for i in total_lookback..n {
            let sum: f64 = raw_strength[(i - self.smoothing + 1)..=i].iter().sum();
            result[i] = (sum / self.smoothing as f64).clamp(0.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for SentimentStrengthIndicator {
    fn name(&self) -> &str {
        "Sentiment Strength Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Crowd Behavior Indicator - Measures crowd/herd behavior intensity
///
/// This indicator measures the intensity of crowd or herd behavior in the market.
/// It identifies when market participants are moving in unison, which often
/// precedes market extremes and potential reversals.
///
/// # Calculation
/// - Analyzes directional unanimity among market participants
/// - Measures volume clustering in dominant direction
/// - Evaluates pattern conformity across bars
/// - Tracks momentum alignment
///
/// # Output Range
/// Returns values from 0 to 100:
/// - Values 0-30 indicate low crowd behavior/independence
/// - Values 30-60 indicate moderate crowd behavior
/// - Values 60-100 indicate high crowd/herd behavior
#[derive(Debug, Clone)]
pub struct CrowdBehaviorIndicator {
    period: usize,
    sensitivity: f64,
}

impl CrowdBehaviorIndicator {
    /// Creates a new CrowdBehaviorIndicator
    ///
    /// # Arguments
    /// * `period` - Lookback period for analysis (minimum 10)
    /// * `sensitivity` - Sensitivity to crowd behavior patterns (0.5 to 2.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, sensitivity: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if sensitivity < 0.5 || sensitivity > 2.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sensitivity".to_string(),
                reason: "must be between 0.5 and 2.0".to_string(),
            });
        }
        Ok(Self { period, sensitivity })
    }

    /// Calculate crowd behavior intensity (0 to 100)
    ///
    /// # Arguments
    /// * `open` - Array of opening prices
    /// * `high` - Array of high prices
    /// * `low` - Array of low prices
    /// * `close` - Array of closing prices
    /// * `volume` - Array of volume values
    ///
    /// # Returns
    /// Vector of crowd behavior values for each data point
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // 1. Directional unanimity
            let mut up_days = 0;
            let mut down_days = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_days += 1;
                } else if close[j] < close[j - 1] {
                    down_days += 1;
                }
            }
            let total_directional = up_days + down_days;
            let unanimity = if total_directional > 0 {
                (up_days.max(down_days) as f64) / (total_directional as f64)
            } else {
                0.5
            };
            let unanimity_score = (unanimity - 0.5) * 2.0 * 30.0 * self.sensitivity;

            // 2. Consecutive move streaks
            let mut max_streak = 0;
            let mut current_streak = 0;
            let mut last_direction = 0;
            for j in (start + 1)..=i {
                let direction = if close[j] > close[j - 1] { 1 } else if close[j] < close[j - 1] { -1 } else { 0 };
                if direction != 0 && direction == last_direction {
                    current_streak += 1;
                    max_streak = max_streak.max(current_streak);
                } else if direction != 0 {
                    current_streak = 1;
                    last_direction = direction;
                }
            }
            let streak_score = (max_streak as f64 / (self.period as f64 / 3.0) * 20.0).min(25.0);

            // 3. Volume concentration in dominant direction
            let dominant_direction = if up_days > down_days { 1 } else { -1 };
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let mut dominant_vol = 0.0;
            let mut total_vol = 0.0;
            for j in (start + 1)..=i {
                let dir = if close[j] > close[j - 1] { 1 } else if close[j] < close[j - 1] { -1 } else { 0 };
                total_vol += volume[j];
                if dir == dominant_direction {
                    dominant_vol += volume[j];
                }
            }
            let vol_concentration = if total_vol > 0.0 { dominant_vol / total_vol } else { 0.5 };
            let vol_score = (vol_concentration - 0.5) * 2.0 * 25.0 * self.sensitivity;

            // 4. Pattern conformity (similar candle patterns)
            let mut conforming_candles = 0;
            for j in start..=i {
                let bar_range = high[j] - low[j];
                if bar_range > 1e-10 {
                    let body = close[j] - open[j];
                    let body_ratio = body / bar_range;
                    // Check if candle matches dominant direction
                    if (dominant_direction == 1 && body_ratio > 0.2) ||
                       (dominant_direction == -1 && body_ratio < -0.2) {
                        conforming_candles += 1;
                    }
                }
            }
            let conformity_ratio = conforming_candles as f64 / (self.period + 1) as f64;
            let conformity_score = conformity_ratio * 20.0;

            // 5. Price momentum alignment
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;
            let position = if range > 1e-10 {
                (close[i] - period_low) / range
            } else {
                0.5
            };
            // Extreme positions indicate crowd behavior
            let alignment_score = if position > 0.8 || position < 0.2 {
                15.0
            } else {
                0.0
            };

            // Combine all components
            result[i] = (unanimity_score + streak_score + vol_score + conformity_score + alignment_score)
                .clamp(0.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for CrowdBehaviorIndicator {
    fn name(&self) -> &str {
        "Crowd Behavior Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Smart Money Sentiment - Institutional sentiment proxy
///
/// This indicator attempts to gauge institutional or "smart money" sentiment
/// by analyzing volume patterns, price efficiency, and accumulation/distribution
/// signals that may indicate professional trading activity.
///
/// # Calculation
/// - Analyzes volume distribution during price moves
/// - Measures price efficiency (smart money often trades efficiently)
/// - Evaluates accumulation vs distribution patterns
/// - Tracks end-of-day price positioning
///
/// # Output Range
/// Returns values from -100 to 100:
/// - Strong negative values indicate smart money is bearish/distributing
/// - Values near 0 indicate neutral smart money activity
/// - Strong positive values indicate smart money is bullish/accumulating
#[derive(Debug, Clone)]
pub struct SmartMoneySentiment {
    period: usize,
    efficiency_period: usize,
}

impl SmartMoneySentiment {
    /// Creates a new SmartMoneySentiment indicator
    ///
    /// # Arguments
    /// * `period` - Main lookback period for analysis (minimum 10)
    /// * `efficiency_period` - Period for efficiency calculation (minimum 5)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, efficiency_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if efficiency_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "efficiency_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period, efficiency_period })
    }

    /// Calculate smart money sentiment (-100 to 100)
    ///
    /// # Arguments
    /// * `open` - Array of opening prices
    /// * `high` - Array of high prices
    /// * `low` - Array of low prices
    /// * `close` - Array of closing prices
    /// * `volume` - Array of volume values
    ///
    /// # Returns
    /// Vector of smart money sentiment values for each data point
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];
        let max_period = self.period.max(self.efficiency_period);

        for i in max_period..n {
            let start = i - self.period;
            let eff_start = i - self.efficiency_period;

            // 1. Accumulation/Distribution based on close position
            // Smart money tends to buy on weakness (close near lows) and sell on strength
            let mut accum_dist_score = 0.0;
            for j in start..=i {
                let bar_range = high[j] - low[j];
                if bar_range > 1e-10 {
                    // Money Flow Multiplier
                    let mfm = ((close[j] - low[j]) - (high[j] - close[j])) / bar_range;
                    accum_dist_score += mfm * volume[j];
                }
            }
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let ad_normalized = if avg_vol > 0.0 {
                (accum_dist_score / ((self.period + 1) as f64 * avg_vol) * 40.0).clamp(-40.0, 40.0)
            } else {
                0.0
            };

            // 2. Price efficiency ratio (smart money creates efficient moves)
            let net_change = (close[i] - close[eff_start]).abs();
            let mut total_change = 0.0;
            for j in (eff_start + 1)..=i {
                total_change += (close[j] - close[j - 1]).abs();
            }
            let efficiency = if total_change > 1e-10 {
                net_change / total_change
            } else {
                0.0
            };
            let direction = if close[i] > close[eff_start] { 1.0 } else { -1.0 };
            let efficiency_score = direction * efficiency * 25.0;

            // 3. Volume-weighted price trend (smart money volume profile)
            // High volume at good prices, low volume at poor prices
            let mut vol_weighted_up = 0.0;
            let mut vol_weighted_down = 0.0;
            let avg_vol_period: f64 = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            for j in (start + 1)..=i {
                let vol_weight = if avg_vol_period > 0.0 { volume[j] / avg_vol_period } else { 1.0 };
                if close[j] > close[j - 1] {
                    vol_weighted_up += vol_weight;
                } else if close[j] < close[j - 1] {
                    vol_weighted_down += vol_weight;
                }
            }
            let total_vol_weighted = vol_weighted_up + vol_weighted_down;
            let vol_profile_score = if total_vol_weighted > 0.0 {
                ((vol_weighted_up - vol_weighted_down) / total_vol_weighted * 25.0).clamp(-25.0, 25.0)
            } else {
                0.0
            };

            // 4. Late session strength (smart money often active late in session)
            // Using close vs open as proxy for intraday positioning
            let mut late_strength = 0.0;
            for j in (i - 3.min(self.period / 3))..=i {
                let bar_range = high[j] - low[j];
                if bar_range > 1e-10 {
                    // Close relative to open, weighted by volume
                    let intraday_move = (close[j] - open[j]) / bar_range;
                    let vol_weight = if avg_vol > 0.0 { (volume[j] / avg_vol).min(2.0) } else { 1.0 };
                    late_strength += intraday_move * vol_weight;
                }
            }
            let late_score = (late_strength * 5.0).clamp(-15.0, 15.0);

            // 5. Divergence detection (smart money often acts contrary to price)
            // If price is up but volume pattern suggests distribution = bearish smart money
            let price_trend = if close[start] > 0.0 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };
            let volume_trend = if vol_weighted_up + vol_weighted_down > 0.0 {
                (vol_weighted_up - vol_weighted_down) / total_vol_weighted.max(1.0)
            } else {
                0.0
            };
            // Divergence: price up but volume bearish, or vice versa
            let divergence_score = if (price_trend > 2.0 && volume_trend < -0.1) ||
                                      (price_trend < -2.0 && volume_trend > 0.1) {
                -price_trend.signum() * 10.0  // Contrarian signal
            } else {
                0.0
            };

            // Combine all components
            result[i] = (ad_normalized + efficiency_score + vol_profile_score + late_score + divergence_score)
                .clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for SmartMoneySentiment {
    fn name(&self) -> &str {
        "Smart Money Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.efficiency_period) + 1
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

    // ============================================================================
    // Tests for New Advanced Sentiment Indicators
    // ============================================================================

    #[test]
    fn test_sentiment_momentum() {
        let (_, high, low, close, volume) = make_test_data();
        let sm = SentimentMomentum::new(5, 3).unwrap();
        let result = sm.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Check values are within bounds
        for &val in &result[10..] {
            assert!(val >= -100.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_sentiment_momentum_validation() {
        // period must be at least 2
        assert!(SentimentMomentum::new(1, 3).is_err());
        assert!(SentimentMomentum::new(2, 3).is_ok());

        // smoothing must be at least 1
        assert!(SentimentMomentum::new(5, 0).is_err());
        assert!(SentimentMomentum::new(5, 1).is_ok());
    }

    #[test]
    fn test_sentiment_momentum_min_periods() {
        let sm = SentimentMomentum::new(5, 3).unwrap();
        assert_eq!(sm.min_periods(), 8); // period + smoothing
    }

    #[test]
    fn test_sentiment_extreme_detector() {
        let (_, high, low, close, volume) = make_test_data();
        let sed = SentimentExtremeDetector::new(10, 2.0).unwrap();
        let result = sed.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_sentiment_extreme_detector_validation() {
        // period must be at least 2
        assert!(SentimentExtremeDetector::new(1, 2.0).is_err());
        assert!(SentimentExtremeDetector::new(2, 2.0).is_ok());

        // z_threshold must be between 0 and 5
        assert!(SentimentExtremeDetector::new(10, 0.0).is_err());
        assert!(SentimentExtremeDetector::new(10, 5.1).is_err());
        assert!(SentimentExtremeDetector::new(10, 2.0).is_ok());
        assert!(SentimentExtremeDetector::new(10, 5.0).is_ok());
    }

    #[test]
    fn test_sentiment_extreme_detector_min_periods() {
        let sed = SentimentExtremeDetector::new(10, 2.0).unwrap();
        assert_eq!(sed.min_periods(), 11); // period + 1
    }

    #[test]
    fn test_sentiment_trend_follower() {
        let (_, high, low, close, volume) = make_test_data();
        let stf = SentimentTrendFollower::new(5, 15).unwrap();
        let result = stf.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result[15..] {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_sentiment_trend_follower_validation() {
        // fast_period must be at least 2
        assert!(SentimentTrendFollower::new(1, 10).is_err());
        assert!(SentimentTrendFollower::new(2, 10).is_ok());

        // slow_period must be greater than fast_period
        assert!(SentimentTrendFollower::new(5, 5).is_err());
        assert!(SentimentTrendFollower::new(5, 4).is_err());
        assert!(SentimentTrendFollower::new(5, 6).is_ok());
    }

    #[test]
    fn test_sentiment_trend_follower_min_periods() {
        let stf = SentimentTrendFollower::new(5, 15).unwrap();
        assert_eq!(stf.min_periods(), 16); // slow_period + 1
    }

    #[test]
    fn test_sentiment_contrarian_signal() {
        let (_, high, low, close, volume) = make_test_data();
        let scs = SentimentContrarianSignal::new(10, 30.0).unwrap();
        let result = scs.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_sentiment_contrarian_signal_validation() {
        // period must be at least 2
        assert!(SentimentContrarianSignal::new(1, 30.0).is_err());
        assert!(SentimentContrarianSignal::new(2, 30.0).is_ok());

        // contrarian_threshold must be between 0 and 100
        assert!(SentimentContrarianSignal::new(10, 0.0).is_err());
        assert!(SentimentContrarianSignal::new(10, 101.0).is_err());
        assert!(SentimentContrarianSignal::new(10, 50.0).is_ok());
        assert!(SentimentContrarianSignal::new(10, 100.0).is_ok());
    }

    #[test]
    fn test_sentiment_contrarian_signal_min_periods() {
        let scs = SentimentContrarianSignal::new(10, 30.0).unwrap();
        assert_eq!(scs.min_periods(), 11); // period + 1
    }

    #[test]
    fn test_sentiment_volatility() {
        let (_, high, low, close, _) = make_test_data();
        let sv = SentimentVolatility::new(10).unwrap();
        let result = sv.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Volatility should be non-negative
        for &val in &result[10..] {
            assert!(val >= 0.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_sentiment_volatility_validation() {
        // period must be at least 2
        assert!(SentimentVolatility::new(1).is_err());
        assert!(SentimentVolatility::new(2).is_ok());
    }

    #[test]
    fn test_sentiment_volatility_min_periods() {
        let sv = SentimentVolatility::new(10).unwrap();
        assert_eq!(sv.min_periods(), 11); // period + 1
    }

    #[test]
    fn test_sentiment_cycle() {
        let (_, high, low, close, volume) = make_test_data();
        let sc = SentimentCycle::new(5, 15).unwrap();
        let result = sc.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result[15..] {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_sentiment_cycle_validation() {
        // period must be at least 2
        assert!(SentimentCycle::new(1, 10).is_err());
        assert!(SentimentCycle::new(2, 10).is_ok());

        // cycle_length must be at least 2
        assert!(SentimentCycle::new(5, 1).is_err());
        assert!(SentimentCycle::new(5, 2).is_ok());
    }

    #[test]
    fn test_sentiment_cycle_min_periods() {
        let sc = SentimentCycle::new(5, 15).unwrap();
        assert_eq!(sc.min_periods(), 16); // max(period, cycle_length) + 1

        let sc2 = SentimentCycle::new(20, 10).unwrap();
        assert_eq!(sc2.min_periods(), 21); // max(20, 10) + 1
    }

    #[test]
    fn test_sentiment_momentum_name() {
        let sm = SentimentMomentum::new(5, 3).unwrap();
        assert_eq!(sm.name(), "Sentiment Momentum");
    }

    #[test]
    fn test_sentiment_extreme_detector_name() {
        let sed = SentimentExtremeDetector::new(10, 2.0).unwrap();
        assert_eq!(sed.name(), "Sentiment Extreme Detector");
    }

    #[test]
    fn test_sentiment_trend_follower_name() {
        let stf = SentimentTrendFollower::new(5, 15).unwrap();
        assert_eq!(stf.name(), "Sentiment Trend Follower");
    }

    #[test]
    fn test_sentiment_contrarian_signal_name() {
        let scs = SentimentContrarianSignal::new(10, 30.0).unwrap();
        assert_eq!(scs.name(), "Sentiment Contrarian Signal");
    }

    #[test]
    fn test_sentiment_volatility_name() {
        let sv = SentimentVolatility::new(10).unwrap();
        assert_eq!(sv.name(), "Sentiment Volatility");
    }

    #[test]
    fn test_sentiment_cycle_name() {
        let sc = SentimentCycle::new(5, 15).unwrap();
        assert_eq!(sc.name(), "Sentiment Cycle");
    }

    #[test]
    fn test_sentiment_indicators_with_flat_data() {
        // Test with flat (constant) data
        let high = vec![100.0; 30];
        let low = vec![100.0; 30];
        let close = vec![100.0; 30];
        let volume = vec![1000.0; 30];

        // SentimentMomentum should handle flat data
        let sm = SentimentMomentum::new(5, 3).unwrap();
        let result = sm.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 30);

        // SentimentVolatility should be 0 for flat data
        let sv = SentimentVolatility::new(10).unwrap();
        let result = sv.calculate(&high, &low, &close);
        assert_eq!(result.len(), 30);
    }

    #[test]
    fn test_sentiment_indicators_with_volatile_data() {
        // Test with highly volatile data
        let high: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 5.0).sin() * 20.0 + 5.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 5.0).sin() * 20.0 - 5.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 5.0).sin() * 20.0).collect();
        let volume: Vec<f64> = (0..30).map(|i| 1000.0 + (i as f64 * 3.0).cos() * 500.0).collect();

        // All indicators should handle volatile data without panicking
        let sm = SentimentMomentum::new(5, 3).unwrap();
        let result = sm.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 30);

        let sed = SentimentExtremeDetector::new(5, 1.5).unwrap();
        let result = sed.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 30);

        let stf = SentimentTrendFollower::new(3, 8).unwrap();
        let result = stf.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 30);

        let scs = SentimentContrarianSignal::new(5, 20.0).unwrap();
        let result = scs.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 30);

        let sv = SentimentVolatility::new(5).unwrap();
        let result = sv.calculate(&high, &low, &close);
        assert_eq!(result.len(), 30);

        let sc = SentimentCycle::new(5, 10).unwrap();
        let result = sc.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 30);
    }

    // ============================================================================
    // Tests for Extended Advanced Sentiment Indicators
    // ============================================================================

    #[test]
    fn test_sentiment_strength() {
        let (open, high, low, close, volume) = make_test_data();
        let ss = SentimentStrength::new(10, 3).unwrap();
        let result = ss.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Strength should be non-negative
        for &val in &result[15..] {
            assert!(val >= 0.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_sentiment_strength_validation() {
        // period must be at least 5
        assert!(SentimentStrength::new(4, 3).is_err());
        assert!(SentimentStrength::new(5, 3).is_ok());

        // smoothing must be at least 1
        assert!(SentimentStrength::new(10, 0).is_err());
        assert!(SentimentStrength::new(10, 1).is_ok());
    }

    #[test]
    fn test_sentiment_strength_min_periods() {
        let ss = SentimentStrength::new(10, 3).unwrap();
        assert_eq!(ss.min_periods(), 13); // period + smoothing
    }

    #[test]
    fn test_sentiment_strength_name() {
        let ss = SentimentStrength::new(10, 3).unwrap();
        assert_eq!(ss.name(), "Sentiment Strength");
    }

    #[test]
    fn test_sentiment_strength_uptrend() {
        // Test with strong uptrend data - should show high strength
        let open: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 2.0).collect();
        let high: Vec<f64> = (0..30).map(|i| 102.0 + i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 99.0 + i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 101.5 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..30).map(|i| 1000.0 + i as f64 * 50.0).collect();

        let ss = SentimentStrength::new(10, 3).unwrap();
        let result = ss.calculate(&open, &high, &low, &close, &volume);

        // Strong trend should produce higher strength values
        let avg_strength: f64 = result[20..].iter().sum::<f64>() / 10.0;
        assert!(avg_strength > 20.0, "Expected strong trend to show strength > 20, got {}", avg_strength);
    }

    #[test]
    fn test_sentiment_acceleration() {
        let (_, high, low, close, volume) = make_test_data();
        let sa = SentimentAcceleration::new(10, 5).unwrap();
        let result = sa.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_sentiment_acceleration_validation() {
        // period must be at least 5
        assert!(SentimentAcceleration::new(4, 3).is_err());
        assert!(SentimentAcceleration::new(5, 3).is_ok());

        // roc_period must be at least 2
        assert!(SentimentAcceleration::new(10, 1).is_err());
        assert!(SentimentAcceleration::new(10, 2).is_ok());
    }

    #[test]
    fn test_sentiment_acceleration_min_periods() {
        let sa = SentimentAcceleration::new(10, 5).unwrap();
        assert_eq!(sa.min_periods(), 16); // period + roc_period + 1
    }

    #[test]
    fn test_sentiment_acceleration_name() {
        let sa = SentimentAcceleration::new(10, 5).unwrap();
        assert_eq!(sa.name(), "Sentiment Acceleration");
    }

    #[test]
    fn test_sentiment_acceleration_with_accelerating_trend() {
        // Create data with accelerating momentum
        let mut close = vec![100.0; 40];
        for i in 10..40 {
            // Accelerating gains
            let acceleration = ((i - 10) as f64).powi(2) * 0.01;
            close[i] = close[i - 1] + 0.5 + acceleration;
        }
        let high: Vec<f64> = close.iter().map(|&c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|&c| c - 1.0).collect();
        let volume = vec![1000.0; 40];

        let sa = SentimentAcceleration::new(10, 5).unwrap();
        let result = sa.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 40);
    }

    #[test]
    fn test_sentiment_mean_reversion() {
        let (_, high, low, close, volume) = make_test_data();
        let smr = SentimentMeanReversion::new(10, 25).unwrap();
        let result = smr.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result[25..] {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_sentiment_mean_reversion_validation() {
        // short_period must be at least 5
        assert!(SentimentMeanReversion::new(4, 20).is_err());
        assert!(SentimentMeanReversion::new(5, 20).is_ok());

        // long_period must be greater than short_period
        assert!(SentimentMeanReversion::new(10, 10).is_err());
        assert!(SentimentMeanReversion::new(10, 9).is_err());
        assert!(SentimentMeanReversion::new(10, 11).is_ok());
    }

    #[test]
    fn test_sentiment_mean_reversion_min_periods() {
        let smr = SentimentMeanReversion::new(10, 25).unwrap();
        assert_eq!(smr.min_periods(), 26); // long_period + 1
    }

    #[test]
    fn test_sentiment_mean_reversion_name() {
        let smr = SentimentMeanReversion::new(10, 25).unwrap();
        assert_eq!(smr.name(), "Sentiment Mean Reversion");
    }

    #[test]
    fn test_sentiment_mean_reversion_extreme_deviation() {
        // Create data that deviates from mean then reverts
        let mut close = vec![100.0; 40];
        // First create stable period
        for i in 1..20 {
            close[i] = 100.0 + (i as f64 * 0.1).sin() * 2.0;
        }
        // Then extreme deviation
        for i in 20..30 {
            close[i] = 100.0 + (i - 20) as f64 * 3.0; // Strong upward deviation
        }
        // Then slight reversion
        for i in 30..40 {
            close[i] = 130.0 - (i - 30) as f64 * 1.0;
        }
        let high: Vec<f64> = close.iter().map(|&c| c + 2.0).collect();
        let low: Vec<f64> = close.iter().map(|&c| c - 2.0).collect();
        let volume = vec![1000.0; 40];

        let smr = SentimentMeanReversion::new(5, 15).unwrap();
        let result = smr.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 40);
    }

    #[test]
    fn test_crowd_behavior_index() {
        let (open, high, low, close, volume) = make_test_data();
        let cbi = CrowdBehaviorIndex::new(15, 0.7).unwrap();
        let result = cbi.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Crowd behavior index should be non-negative
        for &val in &result[15..] {
            assert!(val >= 0.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_crowd_behavior_index_validation() {
        // period must be at least 10
        assert!(CrowdBehaviorIndex::new(9, 0.5).is_err());
        assert!(CrowdBehaviorIndex::new(10, 0.5).is_ok());

        // threshold must be between 0 and 1
        assert!(CrowdBehaviorIndex::new(15, 0.0).is_err());
        assert!(CrowdBehaviorIndex::new(15, 1.1).is_err());
        assert!(CrowdBehaviorIndex::new(15, 0.5).is_ok());
        assert!(CrowdBehaviorIndex::new(15, 1.0).is_ok());
    }

    #[test]
    fn test_crowd_behavior_index_min_periods() {
        let cbi = CrowdBehaviorIndex::new(15, 0.7).unwrap();
        assert_eq!(cbi.min_periods(), 16); // period + 1
    }

    #[test]
    fn test_crowd_behavior_index_name() {
        let cbi = CrowdBehaviorIndex::new(15, 0.7).unwrap();
        assert_eq!(cbi.name(), "Crowd Behavior Index");
    }

    #[test]
    fn test_crowd_behavior_index_with_herding() {
        // Create data with strong herding behavior - all days moving same direction
        let open: Vec<f64> = (0..40).map(|i| 100.0 + i as f64).collect();
        let high: Vec<f64> = (0..40).map(|i| 101.5 + i as f64).collect();
        let low: Vec<f64> = (0..40).map(|i| 99.5 + i as f64).collect();
        let close: Vec<f64> = (0..40).map(|i| 101.0 + i as f64).collect();
        let volume: Vec<f64> = (0..40).map(|i| 1000.0 + i as f64 * 20.0).collect();

        let cbi = CrowdBehaviorIndex::new(15, 0.5).unwrap();
        let result = cbi.calculate(&open, &high, &low, &close, &volume);

        // Strong herding should produce high crowd behavior values
        let avg_cbi: f64 = result[20..].iter().sum::<f64>() / 20.0;
        assert!(avg_cbi > 30.0, "Expected high herding to show CBI > 30, got {}", avg_cbi);
    }

    #[test]
    fn test_sentiment_regime_detector() {
        let (_, high, low, close, volume) = make_test_data();
        let srd = SentimentRegimeDetector::new(10, 25, 1.5).unwrap();
        let result = srd.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_sentiment_regime_detector_validation() {
        // short_period must be at least 5
        assert!(SentimentRegimeDetector::new(4, 20, 1.5).is_err());
        assert!(SentimentRegimeDetector::new(5, 20, 1.5).is_ok());

        // long_period must be greater than short_period
        assert!(SentimentRegimeDetector::new(10, 10, 1.5).is_err());
        assert!(SentimentRegimeDetector::new(10, 9, 1.5).is_err());
        assert!(SentimentRegimeDetector::new(10, 11, 1.5).is_ok());

        // sensitivity must be between 0 and 5
        assert!(SentimentRegimeDetector::new(10, 20, 0.0).is_err());
        assert!(SentimentRegimeDetector::new(10, 20, 5.1).is_err());
        assert!(SentimentRegimeDetector::new(10, 20, 2.5).is_ok());
        assert!(SentimentRegimeDetector::new(10, 20, 5.0).is_ok());
    }

    #[test]
    fn test_sentiment_regime_detector_min_periods() {
        let srd = SentimentRegimeDetector::new(10, 25, 1.5).unwrap();
        assert_eq!(srd.min_periods(), 26); // long_period + 1
    }

    #[test]
    fn test_sentiment_regime_detector_name() {
        let srd = SentimentRegimeDetector::new(10, 25, 1.5).unwrap();
        assert_eq!(srd.name(), "Sentiment Regime Detector");
    }

    #[test]
    fn test_sentiment_regime_detector_regime_change() {
        // Create data with clear regime change
        let mut close = vec![100.0; 50];
        // Bullish regime
        for i in 1..25 {
            close[i] = close[i - 1] + 0.5;
        }
        // Regime change to bearish
        for i in 25..50 {
            close[i] = close[i - 1] - 0.5;
        }
        let high: Vec<f64> = close.iter().map(|&c| c + 1.5).collect();
        let low: Vec<f64> = close.iter().map(|&c| c - 1.5).collect();
        let volume = vec![1000.0; 50];

        let srd = SentimentRegimeDetector::new(5, 15, 1.5).unwrap();
        let result = srd.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 50);
        // Check that indicator produces non-zero values during trend periods
        let first_half_avg: f64 = result[20..25].iter().sum::<f64>() / 5.0;
        let second_half_avg: f64 = result[35..45].iter().sum::<f64>() / 10.0;
        // Values should differ between regimes
        assert!((first_half_avg - second_half_avg).abs() > 0.1,
                "Regime detector should produce different values for different regimes");
    }

    #[test]
    fn test_contra_sentiment_signal() {
        let (open, high, low, close, volume) = make_test_data();
        let css = ContraSentimentSignal::new(15, 1.5, 3).unwrap();
        let result = css.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_contra_sentiment_signal_validation() {
        // period must be at least 10
        assert!(ContraSentimentSignal::new(9, 1.5, 3).is_err());
        assert!(ContraSentimentSignal::new(10, 1.5, 3).is_ok());

        // extreme_threshold must be between 0 and 3
        assert!(ContraSentimentSignal::new(15, 0.0, 3).is_err());
        assert!(ContraSentimentSignal::new(15, 3.1, 3).is_err());
        assert!(ContraSentimentSignal::new(15, 1.5, 3).is_ok());
        assert!(ContraSentimentSignal::new(15, 3.0, 3).is_ok());

        // confirmation_period must be at least 1
        assert!(ContraSentimentSignal::new(15, 1.5, 0).is_err());
        assert!(ContraSentimentSignal::new(15, 1.5, 1).is_ok());
    }

    #[test]
    fn test_contra_sentiment_signal_min_periods() {
        let css = ContraSentimentSignal::new(15, 1.5, 3).unwrap();
        assert_eq!(css.min_periods(), 19); // period + confirmation_period + 1
    }

    #[test]
    fn test_contra_sentiment_signal_name() {
        let css = ContraSentimentSignal::new(15, 1.5, 3).unwrap();
        assert_eq!(css.name(), "Contra Sentiment Signal");
    }

    #[test]
    fn test_contra_sentiment_signal_extreme_bullish() {
        // Create extremely bullish data to trigger contrarian bearish signal
        let open: Vec<f64> = (0..40).map(|i| 100.0 + i as f64 * 2.0).collect();
        let high: Vec<f64> = (0..40).map(|i| 102.0 + i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..40).map(|i| 99.5 + i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..40).map(|i| 101.8 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..40).map(|i| 1000.0 + i as f64 * 100.0).collect();

        let css = ContraSentimentSignal::new(10, 1.0, 3).unwrap();
        let result = css.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), 40);
    }

    #[test]
    fn test_extended_indicators_with_flat_data() {
        // Test all new indicators with flat (constant) data
        let open = vec![100.0; 50];
        let high = vec![100.0; 50];
        let low = vec![100.0; 50];
        let close = vec![100.0; 50];
        let volume = vec![1000.0; 50];

        // SentimentStrength
        let ss = SentimentStrength::new(10, 3).unwrap();
        let result = ss.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 50);

        // SentimentAcceleration
        let sa = SentimentAcceleration::new(10, 5).unwrap();
        let result = sa.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);

        // SentimentMeanReversion
        let smr = SentimentMeanReversion::new(10, 25).unwrap();
        let result = smr.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);

        // CrowdBehaviorIndex
        let cbi = CrowdBehaviorIndex::new(15, 0.7).unwrap();
        let result = cbi.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 50);

        // SentimentRegimeDetector
        let srd = SentimentRegimeDetector::new(10, 25, 1.5).unwrap();
        let result = srd.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);

        // ContraSentimentSignal
        let css = ContraSentimentSignal::new(15, 1.5, 3).unwrap();
        let result = css.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_extended_indicators_with_volatile_data() {
        // Test all new indicators with highly volatile data
        let open: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5).sin() * 20.0).collect();
        let high: Vec<f64> = (0..50).map(|i| 105.0 + (i as f64 * 0.5).sin() * 25.0).collect();
        let low: Vec<f64> = (0..50).map(|i| 95.0 + (i as f64 * 0.5).sin() * 15.0).collect();
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5).sin() * 22.0).collect();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 0.3).cos().abs() * 2000.0).collect();

        // SentimentStrength
        let ss = SentimentStrength::new(10, 3).unwrap();
        let result = ss.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
        for &val in &result[15..] {
            assert!(val >= 0.0 && val <= 100.0);
        }

        // SentimentAcceleration
        let sa = SentimentAcceleration::new(10, 5).unwrap();
        let result = sa.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0);
        }

        // SentimentMeanReversion
        let smr = SentimentMeanReversion::new(10, 25).unwrap();
        let result = smr.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
        for &val in &result[25..] {
            assert!(val >= -100.0 && val <= 100.0);
        }

        // CrowdBehaviorIndex
        let cbi = CrowdBehaviorIndex::new(15, 0.7).unwrap();
        let result = cbi.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
        for &val in &result[15..] {
            assert!(val >= 0.0 && val <= 100.0);
        }

        // SentimentRegimeDetector
        let srd = SentimentRegimeDetector::new(10, 25, 1.5).unwrap();
        let result = srd.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0);
        }

        // ContraSentimentSignal
        let css = ContraSentimentSignal::new(15, 1.5, 3).unwrap();
        let result = css.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_extended_indicators_with_short_data() {
        // Test with minimal data length
        let open = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let high = vec![101.0, 102.0, 103.0, 104.0, 105.0];
        let low = vec![99.0, 100.0, 101.0, 102.0, 103.0];
        let close = vec![100.5, 101.5, 102.5, 103.5, 104.5];
        let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0];

        // These should not panic even with short data
        let ss = SentimentStrength::new(5, 1).unwrap();
        let result = ss.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 5);

        let sa = SentimentAcceleration::new(5, 2).unwrap();
        let result = sa.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 5);
    }

    // ============================================================================
    // Tests for New Sentiment Indicators (6 new indicators)
    // ============================================================================

    #[test]
    fn test_price_based_sentiment() {
        let (_, high, low, close, _) = make_test_data();
        let pbs = PriceBasedSentiment::new(10, 3).unwrap();
        let result = pbs.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Check values are within bounds
        for &val in &result[15..] {
            assert!(val >= -100.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_price_based_sentiment_validation() {
        // period must be at least 5
        assert!(PriceBasedSentiment::new(4, 3).is_err());
        assert!(PriceBasedSentiment::new(5, 3).is_ok());

        // smoothing must be at least 1
        assert!(PriceBasedSentiment::new(10, 0).is_err());
        assert!(PriceBasedSentiment::new(10, 1).is_ok());
    }

    #[test]
    fn test_price_based_sentiment_min_periods() {
        let pbs = PriceBasedSentiment::new(10, 3).unwrap();
        assert_eq!(pbs.min_periods(), 13); // period + smoothing
    }

    #[test]
    fn test_price_based_sentiment_name() {
        let pbs = PriceBasedSentiment::new(10, 3).unwrap();
        assert_eq!(pbs.name(), "Price Based Sentiment");
    }

    #[test]
    fn test_price_based_sentiment_uptrend() {
        // Test with strong uptrend data
        let high: Vec<f64> = (0..40).map(|i| 102.0 + i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..40).map(|i| 98.0 + i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..40).map(|i| 101.0 + i as f64 * 2.0).collect();

        let pbs = PriceBasedSentiment::new(10, 3).unwrap();
        let result = pbs.calculate(&high, &low, &close);

        // Strong uptrend should produce positive sentiment
        let avg_sentiment: f64 = result[20..].iter().sum::<f64>() / 20.0;
        assert!(avg_sentiment > 0.0, "Expected positive sentiment for uptrend, got {}", avg_sentiment);
    }

    #[test]
    fn test_price_based_sentiment_downtrend() {
        // Test with strong downtrend data
        let high: Vec<f64> = (0..40).map(|i| 202.0 - i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..40).map(|i| 198.0 - i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..40).map(|i| 199.0 - i as f64 * 2.0).collect();

        let pbs = PriceBasedSentiment::new(10, 3).unwrap();
        let result = pbs.calculate(&high, &low, &close);

        // Strong downtrend should produce negative sentiment
        let avg_sentiment: f64 = result[20..].iter().sum::<f64>() / 20.0;
        assert!(avg_sentiment < 0.0, "Expected negative sentiment for downtrend, got {}", avg_sentiment);
    }

    #[test]
    fn test_volume_sentiment_pattern() {
        let (_, high, low, close, volume) = make_test_data();
        let vsp = VolumeSentimentPattern::new(10, 1.0).unwrap();
        let result = vsp.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result[10..] {
            assert!(val >= -100.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_volume_sentiment_pattern_validation() {
        // period must be at least 5
        assert!(VolumeSentimentPattern::new(4, 1.0).is_err());
        assert!(VolumeSentimentPattern::new(5, 1.0).is_ok());

        // sensitivity must be between 0.5 and 3.0
        assert!(VolumeSentimentPattern::new(10, 0.4).is_err());
        assert!(VolumeSentimentPattern::new(10, 3.1).is_err());
        assert!(VolumeSentimentPattern::new(10, 0.5).is_ok());
        assert!(VolumeSentimentPattern::new(10, 3.0).is_ok());
    }

    #[test]
    fn test_volume_sentiment_pattern_min_periods() {
        let vsp = VolumeSentimentPattern::new(10, 1.0).unwrap();
        assert_eq!(vsp.min_periods(), 11); // period + 1
    }

    #[test]
    fn test_volume_sentiment_pattern_name() {
        let vsp = VolumeSentimentPattern::new(10, 1.0).unwrap();
        assert_eq!(vsp.name(), "Volume Sentiment Pattern");
    }

    #[test]
    fn test_volume_sentiment_pattern_accumulation() {
        // Test accumulation pattern: high volume on up days
        let high: Vec<f64> = (0..40).map(|i| 102.0 + i as f64).collect();
        let low: Vec<f64> = (0..40).map(|i| 98.0 + i as f64).collect();
        let close: Vec<f64> = (0..40).map(|i| 101.0 + i as f64).collect();
        // Higher volume on up days
        let volume: Vec<f64> = (0..40).map(|i| if i % 2 == 0 { 2000.0 } else { 1000.0 }).collect();

        let vsp = VolumeSentimentPattern::new(10, 1.0).unwrap();
        let result = vsp.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 40);
    }

    #[test]
    fn test_momentum_sentiment_index() {
        let (_, _, _, close, _) = make_test_data();
        let msi = MomentumSentimentIndex::new(10, 5).unwrap();
        let result = msi.calculate(&close);

        assert_eq!(result.len(), close.len());
        for &val in &result[10..] {
            assert!(val >= -100.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_momentum_sentiment_index_validation() {
        // period must be at least 5
        assert!(MomentumSentimentIndex::new(4, 3).is_err());
        assert!(MomentumSentimentIndex::new(5, 3).is_ok());

        // roc_period must be at least 2
        assert!(MomentumSentimentIndex::new(10, 1).is_err());
        assert!(MomentumSentimentIndex::new(10, 2).is_ok());
    }

    #[test]
    fn test_momentum_sentiment_index_min_periods() {
        let msi = MomentumSentimentIndex::new(10, 5).unwrap();
        assert_eq!(msi.min_periods(), 11); // max(period, roc_period) + 1
    }

    #[test]
    fn test_momentum_sentiment_index_name() {
        let msi = MomentumSentimentIndex::new(10, 5).unwrap();
        assert_eq!(msi.name(), "Momentum Sentiment Index");
    }

    #[test]
    fn test_momentum_sentiment_index_strong_momentum() {
        // Test with strong upward momentum
        let close: Vec<f64> = (0..40).map(|i| 100.0 + i as f64 * 1.5).collect();

        let msi = MomentumSentimentIndex::new(10, 5).unwrap();
        let result = msi.calculate(&close);

        // Strong momentum should produce positive sentiment
        let avg_sentiment: f64 = result[15..].iter().sum::<f64>() / 25.0;
        assert!(avg_sentiment > 0.0, "Expected positive momentum sentiment, got {}", avg_sentiment);
    }

    #[test]
    fn test_extreme_sentiment_detector() {
        let (_, high, low, close, volume) = make_test_data();
        let esd = ExtremeSentimentDetector::new(10, 2.0, 3).unwrap();
        let result = esd.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_extreme_sentiment_detector_validation() {
        // period must be at least 5
        assert!(ExtremeSentimentDetector::new(4, 2.0, 3).is_err());
        assert!(ExtremeSentimentDetector::new(5, 2.0, 3).is_ok());

        // z_threshold must be between 1.0 and 4.0
        assert!(ExtremeSentimentDetector::new(10, 0.9, 3).is_err());
        assert!(ExtremeSentimentDetector::new(10, 4.1, 3).is_err());
        assert!(ExtremeSentimentDetector::new(10, 1.0, 3).is_ok());
        assert!(ExtremeSentimentDetector::new(10, 4.0, 3).is_ok());

        // lookback_multiple must be at least 2
        assert!(ExtremeSentimentDetector::new(10, 2.0, 1).is_err());
        assert!(ExtremeSentimentDetector::new(10, 2.0, 2).is_ok());
    }

    #[test]
    fn test_extreme_sentiment_detector_min_periods() {
        let esd = ExtremeSentimentDetector::new(10, 2.0, 3).unwrap();
        assert_eq!(esd.min_periods(), 31); // period * lookback_multiple + 1
    }

    #[test]
    fn test_extreme_sentiment_detector_name() {
        let esd = ExtremeSentimentDetector::new(10, 2.0, 3).unwrap();
        assert_eq!(esd.name(), "Extreme Sentiment Detector");
    }

    #[test]
    fn test_extreme_sentiment_detector_with_spike() {
        // Create data with an extreme move
        let mut high = vec![101.0; 50];
        let mut low = vec![99.0; 50];
        let mut close = vec![100.0; 50];
        let mut volume = vec![1000.0; 50];

        // Add extreme move at end
        for i in 40..50 {
            high[i] = 100.0 + (i - 40) as f64 * 5.0;
            low[i] = 98.0 + (i - 40) as f64 * 5.0;
            close[i] = 100.0 + (i - 40) as f64 * 5.0;
            volume[i] = 3000.0;
        }

        let esd = ExtremeSentimentDetector::new(5, 1.5, 3).unwrap();
        let result = esd.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_sentiment_oscillator() {
        let (_, high, low, close, volume) = make_test_data();
        let so = SentimentOscillator::new(5, 15, 3).unwrap();
        let result = so.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result[18..] {
            assert!(val >= -100.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_sentiment_oscillator_validation() {
        // fast_period must be at least 3
        assert!(SentimentOscillator::new(2, 10, 3).is_err());
        assert!(SentimentOscillator::new(3, 10, 3).is_ok());

        // slow_period must be greater than fast_period
        assert!(SentimentOscillator::new(5, 5, 3).is_err());
        assert!(SentimentOscillator::new(5, 4, 3).is_err());
        assert!(SentimentOscillator::new(5, 6, 3).is_ok());

        // signal_period must be at least 1
        assert!(SentimentOscillator::new(5, 10, 0).is_err());
        assert!(SentimentOscillator::new(5, 10, 1).is_ok());
    }

    #[test]
    fn test_sentiment_oscillator_min_periods() {
        let so = SentimentOscillator::new(5, 15, 3).unwrap();
        assert_eq!(so.min_periods(), 18); // slow_period + signal_period
    }

    #[test]
    fn test_sentiment_oscillator_name() {
        let so = SentimentOscillator::new(5, 15, 3).unwrap();
        assert_eq!(so.name(), "Sentiment Oscillator");
    }

    #[test]
    fn test_sentiment_oscillator_crossover() {
        // Test that oscillator can detect sentiment shifts
        let mut close = vec![100.0; 50];
        // Downtrend then uptrend
        for i in 1..25 {
            close[i] = close[i - 1] - 0.5;
        }
        for i in 25..50 {
            close[i] = close[i - 1] + 1.0;
        }
        let high: Vec<f64> = close.iter().map(|&c| c + 2.0).collect();
        let low: Vec<f64> = close.iter().map(|&c| c - 2.0).collect();
        let volume = vec![1000.0; 50];

        let so = SentimentOscillator::new(3, 10, 2).unwrap();
        let result = so.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_composite_sentiment_index() {
        let (open, high, low, close, volume) = make_test_data();
        let csi = CompositeSentimentIndex::new(15).unwrap();
        let result = csi.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result[15..] {
            assert!(val >= -100.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_composite_sentiment_index_with_weights() {
        let (open, high, low, close, volume) = make_test_data();
        let csi = CompositeSentimentIndex::with_weights(15, 0.40, 0.20, 0.30, 0.10).unwrap();
        let result = csi.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result[15..] {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_composite_sentiment_index_validation() {
        // period must be at least 10
        assert!(CompositeSentimentIndex::new(9).is_err());
        assert!(CompositeSentimentIndex::new(10).is_ok());

        // weights must sum to 1.0
        assert!(CompositeSentimentIndex::with_weights(15, 0.50, 0.50, 0.50, 0.50).is_err());
        assert!(CompositeSentimentIndex::with_weights(15, 0.25, 0.25, 0.25, 0.25).is_ok());
    }

    #[test]
    fn test_composite_sentiment_index_min_periods() {
        let csi = CompositeSentimentIndex::new(15).unwrap();
        assert_eq!(csi.min_periods(), 16); // period + 1
    }

    #[test]
    fn test_composite_sentiment_index_name() {
        let csi = CompositeSentimentIndex::new(15).unwrap();
        assert_eq!(csi.name(), "Composite Sentiment Index");
    }

    #[test]
    fn test_composite_sentiment_index_bullish_market() {
        // Test with bullish market conditions
        let open: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 1.5).collect();
        let high: Vec<f64> = (0..50).map(|i| 102.0 + i as f64 * 1.5).collect();
        let low: Vec<f64> = (0..50).map(|i| 99.0 + i as f64 * 1.5).collect();
        let close: Vec<f64> = (0..50).map(|i| 101.5 + i as f64 * 1.5).collect();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + i as f64 * 20.0).collect();

        let csi = CompositeSentimentIndex::new(15).unwrap();
        let result = csi.calculate(&open, &high, &low, &close, &volume);

        // Bullish market should produce positive composite sentiment
        let avg_sentiment: f64 = result[25..].iter().sum::<f64>() / 25.0;
        assert!(avg_sentiment > 0.0, "Expected positive sentiment for bullish market, got {}", avg_sentiment);
    }

    #[test]
    fn test_new_indicators_with_flat_data() {
        // Test all 6 new indicators with flat (constant) data
        let open = vec![100.0; 50];
        let high = vec![100.0; 50];
        let low = vec![100.0; 50];
        let close = vec![100.0; 50];
        let volume = vec![1000.0; 50];

        // PriceBasedSentiment
        let pbs = PriceBasedSentiment::new(10, 3).unwrap();
        let result = pbs.calculate(&high, &low, &close);
        assert_eq!(result.len(), 50);

        // VolumeSentimentPattern
        let vsp = VolumeSentimentPattern::new(10, 1.0).unwrap();
        let result = vsp.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);

        // MomentumSentimentIndex
        let msi = MomentumSentimentIndex::new(10, 5).unwrap();
        let result = msi.calculate(&close);
        assert_eq!(result.len(), 50);

        // ExtremeSentimentDetector
        let esd = ExtremeSentimentDetector::new(5, 2.0, 3).unwrap();
        let result = esd.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);

        // SentimentOscillator
        let so = SentimentOscillator::new(5, 15, 3).unwrap();
        let result = so.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);

        // CompositeSentimentIndex
        let csi = CompositeSentimentIndex::new(15).unwrap();
        let result = csi.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_new_indicators_with_volatile_data() {
        // Test all 6 new indicators with highly volatile data
        let open: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5).sin() * 20.0).collect();
        let high: Vec<f64> = (0..50).map(|i| 105.0 + (i as f64 * 0.5).sin() * 25.0).collect();
        let low: Vec<f64> = (0..50).map(|i| 95.0 + (i as f64 * 0.5).sin() * 15.0).collect();
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.5).sin() * 22.0).collect();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 0.3).cos().abs() * 2000.0).collect();

        // PriceBasedSentiment
        let pbs = PriceBasedSentiment::new(10, 3).unwrap();
        let result = pbs.calculate(&high, &low, &close);
        assert_eq!(result.len(), 50);
        for &val in &result[15..] {
            assert!(val >= -100.0 && val <= 100.0);
        }

        // VolumeSentimentPattern
        let vsp = VolumeSentimentPattern::new(10, 1.5).unwrap();
        let result = vsp.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
        for &val in &result[10..] {
            assert!(val >= -100.0 && val <= 100.0);
        }

        // MomentumSentimentIndex
        let msi = MomentumSentimentIndex::new(10, 5).unwrap();
        let result = msi.calculate(&close);
        assert_eq!(result.len(), 50);
        for &val in &result[10..] {
            assert!(val >= -100.0 && val <= 100.0);
        }

        // ExtremeSentimentDetector
        let esd = ExtremeSentimentDetector::new(5, 1.5, 3).unwrap();
        let result = esd.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
        for &val in &result {
            assert!(val >= -100.0 && val <= 100.0);
        }

        // SentimentOscillator
        let so = SentimentOscillator::new(5, 12, 3).unwrap();
        let result = so.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
        for &val in &result[15..] {
            assert!(val >= -100.0 && val <= 100.0);
        }

        // CompositeSentimentIndex
        let csi = CompositeSentimentIndex::new(12).unwrap();
        let result = csi.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 50);
        for &val in &result[12..] {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_new_indicators_with_short_data() {
        // Test with minimal data length to ensure no panics
        let open = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0];
        let high = vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0];
        let low = vec![99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0];
        let close = vec![100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5];
        let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0];

        // These should not panic with short data
        let pbs = PriceBasedSentiment::new(5, 2).unwrap();
        let result = pbs.calculate(&high, &low, &close);
        assert_eq!(result.len(), 10);

        let vsp = VolumeSentimentPattern::new(5, 1.0).unwrap();
        let result = vsp.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 10);

        let msi = MomentumSentimentIndex::new(5, 3).unwrap();
        let result = msi.calculate(&close);
        assert_eq!(result.len(), 10);
    }

    // ============================================================================
    // Tests for Additional NEW Sentiment Indicators (6 new indicators)
    // ============================================================================

    #[test]
    fn test_fear_greed_proxy() {
        let (open, high, low, close, volume) = make_test_data();
        let fgp = FearGreedProxy::new(15, 10, 3).unwrap();
        let result = fgp.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result[20..] {
            assert!(val >= -100.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_fear_greed_proxy_validation() {
        // period must be at least 10
        assert!(FearGreedProxy::new(9, 5, 3).is_err());
        assert!(FearGreedProxy::new(10, 5, 3).is_ok());

        // volatility_period must be at least 5
        assert!(FearGreedProxy::new(15, 4, 3).is_err());
        assert!(FearGreedProxy::new(15, 5, 3).is_ok());

        // smoothing must be at least 1
        assert!(FearGreedProxy::new(15, 10, 0).is_err());
        assert!(FearGreedProxy::new(15, 10, 1).is_ok());
    }

    #[test]
    fn test_fear_greed_proxy_min_periods() {
        let fgp = FearGreedProxy::new(15, 10, 3).unwrap();
        assert_eq!(fgp.min_periods(), 18); // max(period, volatility_period) + smoothing
    }

    #[test]
    fn test_fear_greed_proxy_name() {
        let fgp = FearGreedProxy::new(15, 10, 3).unwrap();
        assert_eq!(fgp.name(), "Fear Greed Proxy");
    }

    #[test]
    fn test_fear_greed_proxy_bullish_market() {
        // Test with bullish market - should show greed (positive values)
        let open: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 1.5).collect();
        let high: Vec<f64> = (0..50).map(|i| 102.0 + i as f64 * 1.5).collect();
        let low: Vec<f64> = (0..50).map(|i| 99.0 + i as f64 * 1.5).collect();
        let close: Vec<f64> = (0..50).map(|i| 101.5 + i as f64 * 1.5).collect();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + i as f64 * 30.0).collect();

        let fgp = FearGreedProxy::new(15, 10, 3).unwrap();
        let result = fgp.calculate(&open, &high, &low, &close, &volume);

        let avg_fg: f64 = result[25..].iter().sum::<f64>() / 25.0;
        assert!(avg_fg > 0.0, "Expected greed (positive) for bullish market, got {}", avg_fg);
    }

    #[test]
    fn test_fear_greed_proxy_bearish_market() {
        // Test with bearish market - should show fear (negative values)
        let open: Vec<f64> = (0..50).map(|i| 200.0 - i as f64 * 1.5).collect();
        let high: Vec<f64> = (0..50).map(|i| 201.0 - i as f64 * 1.5).collect();
        let low: Vec<f64> = (0..50).map(|i| 198.0 - i as f64 * 1.5).collect();
        let close: Vec<f64> = (0..50).map(|i| 198.5 - i as f64 * 1.5).collect();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + i as f64 * 30.0).collect();

        let fgp = FearGreedProxy::new(15, 10, 3).unwrap();
        let result = fgp.calculate(&open, &high, &low, &close, &volume);

        let avg_fg: f64 = result[25..].iter().sum::<f64>() / 25.0;
        assert!(avg_fg < 0.0, "Expected fear (negative) for bearish market, got {}", avg_fg);
    }

    #[test]
    fn test_market_panic_index() {
        let (_, high, low, close, volume) = make_test_data();
        let mpi = MarketPanicIndex::new(10, 1.5).unwrap();
        let result = mpi.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Panic index should be non-negative
        for &val in &result[10..] {
            assert!(val >= 0.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_market_panic_index_validation() {
        // period must be at least 5
        assert!(MarketPanicIndex::new(4, 1.5).is_err());
        assert!(MarketPanicIndex::new(5, 1.5).is_ok());

        // panic_threshold must be between 0.5 and 3.0
        assert!(MarketPanicIndex::new(10, 0.4).is_err());
        assert!(MarketPanicIndex::new(10, 3.1).is_err());
        assert!(MarketPanicIndex::new(10, 0.5).is_ok());
        assert!(MarketPanicIndex::new(10, 3.0).is_ok());
    }

    #[test]
    fn test_market_panic_index_min_periods() {
        let mpi = MarketPanicIndex::new(10, 1.5).unwrap();
        assert_eq!(mpi.min_periods(), 11); // period + 1
    }

    #[test]
    fn test_market_panic_index_name() {
        let mpi = MarketPanicIndex::new(10, 1.5).unwrap();
        assert_eq!(mpi.name(), "Market Panic Index");
    }

    #[test]
    fn test_market_panic_index_crash_scenario() {
        // Test with crash scenario - should show high panic
        let mut close = vec![100.0; 50];
        let mut high = vec![101.0; 50];
        let mut low = vec![99.0; 50];
        let mut volume = vec![1000.0; 50];

        // Create crash scenario
        for i in 30..50 {
            close[i] = close[i - 1] * 0.95;  // 5% daily drop
            high[i] = close[i] * 1.01;
            low[i] = close[i] * 0.98;
            volume[i] = 3000.0;  // Volume spike
        }

        let mpi = MarketPanicIndex::new(10, 1.0).unwrap();
        let result = mpi.calculate(&high, &low, &close, &volume);

        // Should show elevated panic during crash
        let avg_panic: f64 = result[40..].iter().sum::<f64>() / 10.0;
        assert!(avg_panic > 20.0, "Expected elevated panic during crash, got {}", avg_panic);
    }

    #[test]
    fn test_euphoria_detector() {
        let (open, high, low, close, volume) = make_test_data();
        let ed = EuphoriaDetector::new(10, 1.5).unwrap();
        let result = ed.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Euphoria detector should be non-negative
        for &val in &result[10..] {
            assert!(val >= 0.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_euphoria_detector_validation() {
        // period must be at least 5
        assert!(EuphoriaDetector::new(4, 1.5).is_err());
        assert!(EuphoriaDetector::new(5, 1.5).is_ok());

        // euphoria_threshold must be between 0.5 and 3.0
        assert!(EuphoriaDetector::new(10, 0.4).is_err());
        assert!(EuphoriaDetector::new(10, 3.1).is_err());
        assert!(EuphoriaDetector::new(10, 0.5).is_ok());
        assert!(EuphoriaDetector::new(10, 3.0).is_ok());
    }

    #[test]
    fn test_euphoria_detector_min_periods() {
        let ed = EuphoriaDetector::new(10, 1.5).unwrap();
        assert_eq!(ed.min_periods(), 11); // period + 1
    }

    #[test]
    fn test_euphoria_detector_name() {
        let ed = EuphoriaDetector::new(10, 1.5).unwrap();
        assert_eq!(ed.name(), "Euphoria Detector");
    }

    #[test]
    fn test_euphoria_detector_rally_scenario() {
        // Test with strong rally - should show high euphoria
        let open: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 3.0).collect();
        let high: Vec<f64> = (0..50).map(|i| 103.0 + i as f64 * 3.0).collect();
        let low: Vec<f64> = (0..50).map(|i| 99.5 + i as f64 * 3.0).collect();
        let close: Vec<f64> = (0..50).map(|i| 102.5 + i as f64 * 3.0).collect();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + i as f64 * 50.0).collect();

        let ed = EuphoriaDetector::new(10, 1.0).unwrap();
        let result = ed.calculate(&open, &high, &low, &close, &volume);

        // Should show elevated euphoria during rally
        let avg_euphoria: f64 = result[30..].iter().sum::<f64>() / 20.0;
        assert!(avg_euphoria > 20.0, "Expected elevated euphoria during rally, got {}", avg_euphoria);
    }

    #[test]
    fn test_sentiment_strength_indicator() {
        let (open, high, low, close, volume) = make_test_data();
        let ssi = SentimentStrengthIndicator::new(10, 3).unwrap();
        let result = ssi.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Sentiment strength should be non-negative
        for &val in &result[15..] {
            assert!(val >= 0.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_sentiment_strength_indicator_validation() {
        // period must be at least 5
        assert!(SentimentStrengthIndicator::new(4, 3).is_err());
        assert!(SentimentStrengthIndicator::new(5, 3).is_ok());

        // smoothing must be at least 1
        assert!(SentimentStrengthIndicator::new(10, 0).is_err());
        assert!(SentimentStrengthIndicator::new(10, 1).is_ok());
    }

    #[test]
    fn test_sentiment_strength_indicator_min_periods() {
        let ssi = SentimentStrengthIndicator::new(10, 3).unwrap();
        assert_eq!(ssi.min_periods(), 13); // period + smoothing
    }

    #[test]
    fn test_sentiment_strength_indicator_name() {
        let ssi = SentimentStrengthIndicator::new(10, 3).unwrap();
        assert_eq!(ssi.name(), "Sentiment Strength Indicator");
    }

    #[test]
    fn test_sentiment_strength_indicator_strong_trend() {
        // Test with strong trend - should show high strength
        let open: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let high: Vec<f64> = (0..50).map(|i| 102.0 + i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..50).map(|i| 99.0 + i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..50).map(|i| 101.5 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + i as f64 * 30.0).collect();

        let ssi = SentimentStrengthIndicator::new(10, 3).unwrap();
        let result = ssi.calculate(&open, &high, &low, &close, &volume);

        let avg_strength: f64 = result[20..].iter().sum::<f64>() / 30.0;
        assert!(avg_strength > 30.0, "Expected high strength for strong trend, got {}", avg_strength);
    }

    #[test]
    fn test_crowd_behavior_indicator() {
        let (open, high, low, close, volume) = make_test_data();
        let cbi = CrowdBehaviorIndicator::new(15, 1.0).unwrap();
        let result = cbi.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Crowd behavior should be non-negative
        for &val in &result[15..] {
            assert!(val >= 0.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_crowd_behavior_indicator_validation() {
        // period must be at least 10
        assert!(CrowdBehaviorIndicator::new(9, 1.0).is_err());
        assert!(CrowdBehaviorIndicator::new(10, 1.0).is_ok());

        // sensitivity must be between 0.5 and 2.0
        assert!(CrowdBehaviorIndicator::new(15, 0.4).is_err());
        assert!(CrowdBehaviorIndicator::new(15, 2.1).is_err());
        assert!(CrowdBehaviorIndicator::new(15, 0.5).is_ok());
        assert!(CrowdBehaviorIndicator::new(15, 2.0).is_ok());
    }

    #[test]
    fn test_crowd_behavior_indicator_min_periods() {
        let cbi = CrowdBehaviorIndicator::new(15, 1.0).unwrap();
        assert_eq!(cbi.min_periods(), 16); // period + 1
    }

    #[test]
    fn test_crowd_behavior_indicator_name() {
        let cbi = CrowdBehaviorIndicator::new(15, 1.0).unwrap();
        assert_eq!(cbi.name(), "Crowd Behavior Indicator");
    }

    #[test]
    fn test_crowd_behavior_indicator_herding() {
        // Test with strong herding - all days moving same direction
        let open: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let high: Vec<f64> = (0..50).map(|i| 101.5 + i as f64).collect();
        let low: Vec<f64> = (0..50).map(|i| 99.5 + i as f64).collect();
        let close: Vec<f64> = (0..50).map(|i| 101.0 + i as f64).collect();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + i as f64 * 20.0).collect();

        let cbi = CrowdBehaviorIndicator::new(15, 1.0).unwrap();
        let result = cbi.calculate(&open, &high, &low, &close, &volume);

        let avg_cbi: f64 = result[25..].iter().sum::<f64>() / 25.0;
        assert!(avg_cbi > 30.0, "Expected high crowd behavior for herding, got {}", avg_cbi);
    }

    #[test]
    fn test_smart_money_sentiment() {
        let (open, high, low, close, volume) = make_test_data();
        let sms = SmartMoneySentiment::new(15, 10).unwrap();
        let result = sms.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for &val in &result[15..] {
            assert!(val >= -100.0 && val <= 100.0, "Value {} out of bounds", val);
        }
    }

    #[test]
    fn test_smart_money_sentiment_validation() {
        // period must be at least 10
        assert!(SmartMoneySentiment::new(9, 5).is_err());
        assert!(SmartMoneySentiment::new(10, 5).is_ok());

        // efficiency_period must be at least 5
        assert!(SmartMoneySentiment::new(15, 4).is_err());
        assert!(SmartMoneySentiment::new(15, 5).is_ok());
    }

    #[test]
    fn test_smart_money_sentiment_min_periods() {
        let sms = SmartMoneySentiment::new(15, 10).unwrap();
        assert_eq!(sms.min_periods(), 16); // max(period, efficiency_period) + 1
    }

    #[test]
    fn test_smart_money_sentiment_name() {
        let sms = SmartMoneySentiment::new(15, 10).unwrap();
        assert_eq!(sms.name(), "Smart Money Sentiment");
    }

    #[test]
    fn test_smart_money_sentiment_accumulation() {
        // Test with accumulation pattern - closing near lows with volume
        let mut open = vec![100.0; 50];
        let mut high = vec![102.0; 50];
        let mut low = vec![98.0; 50];
        let mut close = vec![100.0; 50];
        let mut volume = vec![1000.0; 50];

        // Create accumulation pattern - closing near highs with increasing volume
        for i in 20..50 {
            open[i] = 100.0 + (i - 20) as f64 * 0.3;
            high[i] = open[i] + 2.0;
            low[i] = open[i] - 1.0;
            close[i] = open[i] + 1.5;  // Close near high
            volume[i] = 1500.0 + (i - 20) as f64 * 50.0;  // Increasing volume
        }

        let sms = SmartMoneySentiment::new(15, 10).unwrap();
        let result = sms.calculate(&open, &high, &low, &close, &volume);

        // Should show positive smart money sentiment during accumulation
        let avg_sms: f64 = result[35..].iter().sum::<f64>() / 15.0;
        assert!(avg_sms > 0.0, "Expected positive smart money sentiment during accumulation, got {}", avg_sms);
    }

    #[test]
    fn test_additional_new_indicators_with_flat_data() {
        // Test all 6 additional new indicators with flat (constant) data
        let open = vec![100.0; 60];
        let high = vec![100.0; 60];
        let low = vec![100.0; 60];
        let close = vec![100.0; 60];
        let volume = vec![1000.0; 60];

        // FearGreedProxy
        let fgp = FearGreedProxy::new(15, 10, 3).unwrap();
        let result = fgp.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 60);

        // MarketPanicIndex
        let mpi = MarketPanicIndex::new(10, 1.5).unwrap();
        let result = mpi.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 60);

        // EuphoriaDetector
        let ed = EuphoriaDetector::new(10, 1.5).unwrap();
        let result = ed.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 60);

        // SentimentStrengthIndicator
        let ssi = SentimentStrengthIndicator::new(10, 3).unwrap();
        let result = ssi.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 60);

        // CrowdBehaviorIndicator
        let cbi = CrowdBehaviorIndicator::new(15, 1.0).unwrap();
        let result = cbi.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 60);

        // SmartMoneySentiment
        let sms = SmartMoneySentiment::new(15, 10).unwrap();
        let result = sms.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 60);
    }

    #[test]
    fn test_additional_new_indicators_with_volatile_data() {
        // Test all 6 additional new indicators with highly volatile data
        let open: Vec<f64> = (0..60).map(|i| 100.0 + (i as f64 * 0.5).sin() * 20.0).collect();
        let high: Vec<f64> = (0..60).map(|i| 105.0 + (i as f64 * 0.5).sin() * 25.0).collect();
        let low: Vec<f64> = (0..60).map(|i| 95.0 + (i as f64 * 0.5).sin() * 15.0).collect();
        let close: Vec<f64> = (0..60).map(|i| 100.0 + (i as f64 * 0.5).sin() * 22.0).collect();
        let volume: Vec<f64> = (0..60).map(|i| 1000.0 + (i as f64 * 0.3).cos().abs() * 2000.0).collect();

        // FearGreedProxy
        let fgp = FearGreedProxy::new(15, 10, 3).unwrap();
        let result = fgp.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 60);
        for &val in &result[20..] {
            assert!(val >= -100.0 && val <= 100.0);
        }

        // MarketPanicIndex
        let mpi = MarketPanicIndex::new(10, 1.5).unwrap();
        let result = mpi.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 60);
        for &val in &result[10..] {
            assert!(val >= 0.0 && val <= 100.0);
        }

        // EuphoriaDetector
        let ed = EuphoriaDetector::new(10, 1.5).unwrap();
        let result = ed.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 60);
        for &val in &result[10..] {
            assert!(val >= 0.0 && val <= 100.0);
        }

        // SentimentStrengthIndicator
        let ssi = SentimentStrengthIndicator::new(10, 3).unwrap();
        let result = ssi.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 60);
        for &val in &result[15..] {
            assert!(val >= 0.0 && val <= 100.0);
        }

        // CrowdBehaviorIndicator
        let cbi = CrowdBehaviorIndicator::new(15, 1.0).unwrap();
        let result = cbi.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 60);
        for &val in &result[15..] {
            assert!(val >= 0.0 && val <= 100.0);
        }

        // SmartMoneySentiment
        let sms = SmartMoneySentiment::new(15, 10).unwrap();
        let result = sms.calculate(&open, &high, &low, &close, &volume);
        assert_eq!(result.len(), 60);
        for &val in &result[15..] {
            assert!(val >= -100.0 && val <= 100.0);
        }
    }
}
