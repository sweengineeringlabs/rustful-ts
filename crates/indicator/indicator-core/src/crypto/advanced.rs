//! Advanced Crypto & On-Chain Proxy Indicators
//!
//! Cryptocurrency-specific indicators using price/volume as proxies for on-chain metrics.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

// ============================================================================
// HashRateMomentum - Proxy for hash rate momentum using price patterns
// ============================================================================

/// Hash Rate Momentum Proxy
///
/// Uses price momentum and volatility patterns as a proxy for network hash rate trends.
/// Rising hash rate typically correlates with miner confidence and price strength.
#[derive(Debug, Clone)]
pub struct HashRateMomentum {
    period: usize,
    smooth_period: usize,
}

impl HashRateMomentum {
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smooth_period })
    }

    /// Calculate hash rate momentum proxy
    /// Returns positive values for increasing hash rate proxy, negative for decreasing
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Price momentum component
            let price_mom = if close[start] > 1e-10 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Volume trend as mining activity proxy
            let early_vol: f64 = volume[start..(start + self.period / 2).min(i)].iter().sum();
            let late_vol: f64 = volume[(i - self.period / 2)..=i].iter().sum();
            let vol_trend = if early_vol > 1e-10 {
                (late_vol / early_vol - 1.0) * 50.0
            } else {
                0.0
            };

            // Stability component (lower volatility = more stable hash rate)
            let mut vol_sum = 0.0;
            for j in (start + 1)..=i {
                vol_sum += (close[j] / close[j - 1] - 1.0).abs();
            }
            let avg_volatility = vol_sum / self.period as f64;
            let stability = 1.0 / (1.0 + avg_volatility * 10.0);

            // Composite hash rate momentum
            result[i] = (price_mom * 0.5 + vol_trend * 0.3) * stability;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }
}

impl TechnicalIndicator for HashRateMomentum {
    fn name(&self) -> &str {
        "Hash Rate Momentum Proxy"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// MinerCapitulation - Detects miner capitulation patterns
// ============================================================================

/// Miner Capitulation Detector
///
/// Identifies potential miner capitulation events using price decline patterns
/// combined with volume spikes, which often indicate forced selling by miners.
#[derive(Debug, Clone)]
pub struct MinerCapitulation {
    period: usize,
    threshold: f64,
}

impl MinerCapitulation {
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

    /// Calculate miner capitulation score (0-100)
    /// Higher values indicate higher probability of miner capitulation
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Price decline from period high
            let period_high = close[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let price_decline = if period_high > 1e-10 {
                (period_high - close[i]) / period_high
            } else {
                0.0
            };

            // Volume spike detection
            let avg_vol: f64 = volume[start..i].iter().sum::<f64>() / (i - start) as f64;
            let vol_spike = if avg_vol > 1e-10 {
                volume[i] / avg_vol
            } else {
                1.0
            };

            // Consecutive down days
            let mut down_streak = 0;
            for j in (start..=i).rev() {
                if j > 0 && close[j] < close[j - 1] {
                    down_streak += 1;
                } else {
                    break;
                }
            }

            // Capitulation score components
            let decline_score = (price_decline / self.threshold).min(1.0) * 40.0;
            let volume_score = ((vol_spike - 1.0) / 2.0).clamp(0.0, 1.0) * 30.0;
            let streak_score = (down_streak as f64 / 5.0).min(1.0) * 30.0;

            result[i] = decline_score + volume_score + streak_score;
        }

        result
    }

    /// Returns true if current reading suggests capitulation
    pub fn is_capitulating(&self, score: f64) -> bool {
        score >= 70.0
    }
}

impl TechnicalIndicator for MinerCapitulation {
    fn name(&self) -> &str {
        "Miner Capitulation"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// WhaleAccumulation - Detects whale accumulation patterns via volume
// ============================================================================

/// Whale Accumulation Detector
///
/// Identifies potential whale accumulation patterns through analysis of
/// large volume bars during price consolidation or dips.
#[derive(Debug, Clone)]
pub struct WhaleAccumulation {
    period: usize,
    volume_threshold: f64,
}

impl WhaleAccumulation {
    pub fn new(period: usize, volume_threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if volume_threshold <= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_threshold".to_string(),
                reason: "must be greater than 1.0".to_string(),
            });
        }
        Ok(Self { period, volume_threshold })
    }

    /// Calculate whale accumulation score (-100 to +100)
    /// Positive = accumulation, Negative = distribution
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate average volume
            let avg_vol: f64 = volume[start..i].iter().sum::<f64>() / (i - start) as f64;

            // Detect large volume bars
            let mut accumulation_score = 0.0;
            let mut distribution_score = 0.0;

            for j in start..=i {
                if avg_vol > 1e-10 && volume[j] > avg_vol * self.volume_threshold {
                    // Large volume detected
                    let price_change = if j > 0 { close[j] - close[j - 1] } else { 0.0 };
                    let vol_ratio = volume[j] / avg_vol;

                    if price_change >= 0.0 {
                        // Large volume on up/flat move = accumulation
                        accumulation_score += vol_ratio;
                    } else {
                        // Large volume on down move = distribution
                        distribution_score += vol_ratio;
                    }
                }
            }

            // Price position in range (buying at lows = accumulation)
            let period_high = close[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = close[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;
            let price_position = if range > 1e-10 {
                (close[i] - period_low) / range
            } else {
                0.5
            };

            // Accumulation at low prices is bullish
            let position_weight = 1.0 - price_position;

            // Net accumulation score
            let net_score = accumulation_score - distribution_score;
            result[i] = (net_score * (1.0 + position_weight * 0.5)).clamp(-100.0, 100.0);
        }

        result
    }

    /// Returns true if showing accumulation pattern
    pub fn is_accumulating(&self, score: f64) -> bool {
        score > 20.0
    }
}

impl TechnicalIndicator for WhaleAccumulation {
    fn name(&self) -> &str {
        "Whale Accumulation"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// RetailSentimentProxy - Proxy for retail sentiment
// ============================================================================

/// Retail Sentiment Proxy
///
/// Estimates retail trader sentiment using price momentum, volatility patterns,
/// and volume behavior that typically characterize retail trading activity.
#[derive(Debug, Clone)]
pub struct RetailSentimentProxy {
    period: usize,
    momentum_weight: f64,
}

impl RetailSentimentProxy {
    pub fn new(period: usize, momentum_weight: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if momentum_weight <= 0.0 || momentum_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_weight".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { period, momentum_weight })
    }

    /// Calculate retail sentiment score (-100 to +100)
    /// Positive = bullish retail sentiment, Negative = bearish
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Price momentum (retail tends to chase momentum)
            let price_mom = if close[start] > 1e-10 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Recent momentum acceleration (retail piles in at extremes)
            let mid = (start + i) / 2;
            let first_half_return = if close[start] > 1e-10 && mid > start {
                (close[mid] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };
            let second_half_return = if close[mid] > 1e-10 && mid < i {
                (close[i] / close[mid] - 1.0) * 100.0
            } else {
                0.0
            };
            let acceleration = second_half_return - first_half_return;

            // Volume trend (retail volume increases in trends)
            let first_half_vol: f64 = volume[start..mid].iter().sum();
            let second_half_vol: f64 = volume[mid..=i].iter().sum();
            let vol_trend = if first_half_vol > 1e-10 {
                (second_half_vol / first_half_vol - 1.0) * 50.0
            } else {
                0.0
            };

            // FOMO/FUD indicator (extreme moves with volume = retail)
            let fomo_fud = if price_mom.abs() > 5.0 && vol_trend > 0.0 {
                price_mom.signum() * (vol_trend.abs() * 0.2)
            } else {
                0.0
            };

            // Composite retail sentiment
            let momentum_component = price_mom * self.momentum_weight;
            let acceleration_component = acceleration * (1.0 - self.momentum_weight) * 0.5;
            let fomo_component = fomo_fud;

            result[i] = (momentum_component + acceleration_component + fomo_component).clamp(-100.0, 100.0);
        }

        result
    }

    /// Returns sentiment level classification
    pub fn classify_sentiment(&self, score: f64) -> &'static str {
        match score {
            s if s >= 50.0 => "Extreme Greed",
            s if s >= 20.0 => "Greed",
            s if s >= -20.0 => "Neutral",
            s if s >= -50.0 => "Fear",
            _ => "Extreme Fear",
        }
    }
}

impl TechnicalIndicator for RetailSentimentProxy {
    fn name(&self) -> &str {
        "Retail Sentiment Proxy"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// InstitutionalFlowProxy - Proxy for institutional flow patterns
// ============================================================================

/// Institutional Flow Proxy
///
/// Estimates institutional buying/selling activity using patterns that
/// typically characterize large-scale, systematic trading.
#[derive(Debug, Clone)]
pub struct InstitutionalFlowProxy {
    period: usize,
    smooth_period: usize,
}

impl InstitutionalFlowProxy {
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smooth_period })
    }

    /// Calculate institutional flow proxy (-100 to +100)
    /// Positive = institutional buying, Negative = institutional selling
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate average volume and standard deviation
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
            let vol_variance: f64 = volume[start..=i]
                .iter()
                .map(|v| (v - avg_vol).powi(2))
                .sum::<f64>() / (i - start + 1) as f64;
            let vol_std = vol_variance.sqrt();

            // Institutional activity detection
            // Large, consistent volume with controlled price movement
            let mut institutional_score = 0.0;
            let mut bar_count = 0.0;

            for j in start..=i {
                if j > 0 {
                    let price_change_pct = ((close[j] - close[j - 1]) / close[j - 1]).abs() * 100.0;
                    let vol_zscore = if vol_std > 1e-10 {
                        (volume[j] - avg_vol) / vol_std
                    } else {
                        0.0
                    };

                    // Institutional pattern: high volume with controlled moves
                    if vol_zscore > 0.5 && price_change_pct < 3.0 {
                        let direction = if close[j] > close[j - 1] { 1.0 } else { -1.0 };
                        institutional_score += direction * vol_zscore * (1.0 - price_change_pct / 5.0);
                        bar_count += 1.0;
                    }
                }
            }

            // Normalize score
            if bar_count > 0.0 {
                result[i] = (institutional_score / bar_count * 20.0).clamp(-100.0, 100.0);
            }
        }

        // Apply smoothing (institutional activity is typically gradual)
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Returns true if showing institutional accumulation
    pub fn is_institutional_buying(&self, score: f64) -> bool {
        score > 15.0
    }
}

impl TechnicalIndicator for InstitutionalFlowProxy {
    fn name(&self) -> &str {
        "Institutional Flow Proxy"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// NetworkActivityProxy - Proxy for network activity
// ============================================================================

/// Network Activity Proxy
///
/// Estimates on-chain network activity using trading volume patterns
/// and price velocity as proxies for transaction throughput.
#[derive(Debug, Clone)]
pub struct NetworkActivityProxy {
    period: usize,
    baseline_period: usize,
}

impl NetworkActivityProxy {
    pub fn new(period: usize, baseline_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if baseline_period <= period {
            return Err(IndicatorError::InvalidParameter {
                name: "baseline_period".to_string(),
                reason: "must be greater than period".to_string(),
            });
        }
        Ok(Self { period, baseline_period })
    }

    /// Calculate network activity score (0 to 100+)
    /// Higher values indicate higher network activity
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.baseline_period..n {
            let period_start = i.saturating_sub(self.period);
            let baseline_start = i.saturating_sub(self.baseline_period);

            // Current period volume activity
            let current_vol: f64 = volume[period_start..=i].iter().sum();
            let current_avg = current_vol / self.period as f64;

            // Baseline volume
            let baseline_vol: f64 = volume[baseline_start..period_start].iter().sum();
            let baseline_avg = baseline_vol / (period_start - baseline_start).max(1) as f64;

            // Volume ratio (normalized activity)
            let vol_ratio = if baseline_avg > 1e-10 {
                current_avg / baseline_avg
            } else {
                1.0
            };

            // Price velocity as transaction urgency proxy
            let mut price_velocity = 0.0;
            for j in (period_start + 1)..=i {
                price_velocity += (close[j] - close[j - 1]).abs();
            }
            price_velocity /= self.period as f64;

            // Baseline price velocity
            let mut baseline_velocity = 0.0;
            for j in (baseline_start + 1)..period_start {
                baseline_velocity += (close[j] - close[j - 1]).abs();
            }
            baseline_velocity /= (period_start - baseline_start - 1).max(1) as f64;

            let velocity_ratio = if baseline_velocity > 1e-10 {
                price_velocity / baseline_velocity
            } else {
                1.0
            };

            // Transaction count proxy (volume divided by avg trade size proxy)
            let avg_price = close[period_start..=i].iter().sum::<f64>() / self.period as f64;
            let tx_proxy = if avg_price > 1e-10 {
                current_vol / avg_price
            } else {
                0.0
            };

            // Baseline tx proxy
            let baseline_avg_price = close[baseline_start..period_start].iter().sum::<f64>()
                / (period_start - baseline_start).max(1) as f64;
            let baseline_tx = if baseline_avg_price > 1e-10 {
                baseline_vol / baseline_avg_price
            } else {
                0.0
            };

            let tx_ratio = if baseline_tx > 1e-10 {
                tx_proxy / baseline_tx
            } else {
                1.0
            };

            // Composite network activity score
            let activity = (vol_ratio * 0.4 + velocity_ratio * 0.3 + tx_ratio * 0.3) * 50.0;
            result[i] = activity.max(0.0);
        }

        result
    }

    /// Returns activity level classification
    pub fn classify_activity(&self, score: f64) -> &'static str {
        match score {
            s if s >= 80.0 => "Very High",
            s if s >= 60.0 => "High",
            s if s >= 40.0 => "Normal",
            s if s >= 20.0 => "Low",
            _ => "Very Low",
        }
    }
}

impl TechnicalIndicator for NetworkActivityProxy {
    fn name(&self) -> &str {
        "Network Activity Proxy"
    }

    fn min_periods(&self) -> usize {
        self.baseline_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        // Generate realistic price/volume data with trends and reversals
        let mut close = Vec::with_capacity(50);
        let mut volume = Vec::with_capacity(50);

        let mut price = 100.0;
        for i in 0..50 {
            // Add trend and noise
            let trend = if i < 20 { 0.5 } else if i < 35 { -0.3 } else { 0.4 };
            let noise = ((i as f64 * 0.7).sin() * 2.0);
            price += trend + noise;
            price = price.max(50.0);
            close.push(price);

            // Volume correlates with price movement magnitude
            let base_vol = 1000.0 + (i as f64 * 20.0);
            let vol_spike = if i % 7 == 0 { 2.0 } else { 1.0 };
            volume.push(base_vol * vol_spike);
        }

        (close, volume)
    }

    fn make_ohlcv_series() -> OHLCVSeries {
        let (close, volume) = make_test_data();
        let high: Vec<f64> = close.iter().map(|c| c + 1.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.5).collect();
        let open = close.clone();

        OHLCVSeries { open, high, low, close, volume }
    }

    // ========== HashRateMomentum Tests ==========

    #[test]
    fn test_hash_rate_momentum_new() {
        assert!(HashRateMomentum::new(10, 5).is_ok());
        assert!(HashRateMomentum::new(4, 5).is_err()); // period too small
        assert!(HashRateMomentum::new(10, 1).is_err()); // smooth_period too small
    }

    #[test]
    fn test_hash_rate_momentum_calculate() {
        let (close, volume) = make_test_data();
        let hrm = HashRateMomentum::new(10, 5).unwrap();
        let result = hrm.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Should have values after min_periods
        assert!(result[20].abs() > 0.0 || result[25].abs() > 0.0);
    }

    #[test]
    fn test_hash_rate_momentum_trait() {
        let data = make_ohlcv_series();
        let hrm = HashRateMomentum::new(10, 5).unwrap();

        assert_eq!(hrm.name(), "Hash Rate Momentum Proxy");
        assert_eq!(hrm.min_periods(), 15);

        let output = hrm.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== MinerCapitulation Tests ==========

    #[test]
    fn test_miner_capitulation_new() {
        assert!(MinerCapitulation::new(14, 0.3).is_ok());
        assert!(MinerCapitulation::new(5, 0.3).is_err()); // period too small
        assert!(MinerCapitulation::new(14, 0.0).is_err()); // threshold invalid
        assert!(MinerCapitulation::new(14, 1.5).is_err()); // threshold too large
    }

    #[test]
    fn test_miner_capitulation_calculate() {
        let (close, volume) = make_test_data();
        let mc = MinerCapitulation::new(14, 0.3).unwrap();
        let result = mc.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Score should be bounded 0-100
        for i in 20..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_miner_capitulation_is_capitulating() {
        let mc = MinerCapitulation::new(14, 0.3).unwrap();
        assert!(!mc.is_capitulating(50.0));
        assert!(mc.is_capitulating(75.0));
    }

    #[test]
    fn test_miner_capitulation_trait() {
        let data = make_ohlcv_series();
        let mc = MinerCapitulation::new(14, 0.3).unwrap();

        assert_eq!(mc.name(), "Miner Capitulation");
        assert_eq!(mc.min_periods(), 15);

        let output = mc.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== WhaleAccumulation Tests ==========

    #[test]
    fn test_whale_accumulation_new() {
        assert!(WhaleAccumulation::new(14, 1.5).is_ok());
        assert!(WhaleAccumulation::new(5, 1.5).is_err()); // period too small
        assert!(WhaleAccumulation::new(14, 0.5).is_err()); // threshold too small
        assert!(WhaleAccumulation::new(14, 1.0).is_err()); // threshold exactly 1.0
    }

    #[test]
    fn test_whale_accumulation_calculate() {
        let (close, volume) = make_test_data();
        let wa = WhaleAccumulation::new(14, 1.5).unwrap();
        let result = wa.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Score should be bounded -100 to 100
        for i in 20..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_whale_accumulation_is_accumulating() {
        let wa = WhaleAccumulation::new(14, 1.5).unwrap();
        assert!(!wa.is_accumulating(10.0));
        assert!(wa.is_accumulating(30.0));
    }

    #[test]
    fn test_whale_accumulation_trait() {
        let data = make_ohlcv_series();
        let wa = WhaleAccumulation::new(14, 1.5).unwrap();

        assert_eq!(wa.name(), "Whale Accumulation");
        assert_eq!(wa.min_periods(), 15);

        let output = wa.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== RetailSentimentProxy Tests ==========

    #[test]
    fn test_retail_sentiment_new() {
        assert!(RetailSentimentProxy::new(14, 0.6).is_ok());
        assert!(RetailSentimentProxy::new(3, 0.6).is_err()); // period too small
        assert!(RetailSentimentProxy::new(14, 0.0).is_err()); // weight invalid
        assert!(RetailSentimentProxy::new(14, 1.5).is_err()); // weight too large
    }

    #[test]
    fn test_retail_sentiment_calculate() {
        let (close, volume) = make_test_data();
        let rsp = RetailSentimentProxy::new(14, 0.6).unwrap();
        let result = rsp.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Score should be bounded -100 to 100
        for i in 20..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_retail_sentiment_classify() {
        let rsp = RetailSentimentProxy::new(14, 0.6).unwrap();
        assert_eq!(rsp.classify_sentiment(60.0), "Extreme Greed");
        assert_eq!(rsp.classify_sentiment(30.0), "Greed");
        assert_eq!(rsp.classify_sentiment(0.0), "Neutral");
        assert_eq!(rsp.classify_sentiment(-30.0), "Fear");
        assert_eq!(rsp.classify_sentiment(-60.0), "Extreme Fear");
    }

    #[test]
    fn test_retail_sentiment_trait() {
        let data = make_ohlcv_series();
        let rsp = RetailSentimentProxy::new(14, 0.6).unwrap();

        assert_eq!(rsp.name(), "Retail Sentiment Proxy");
        assert_eq!(rsp.min_periods(), 15);

        let output = rsp.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== InstitutionalFlowProxy Tests ==========

    #[test]
    fn test_institutional_flow_new() {
        assert!(InstitutionalFlowProxy::new(14, 5).is_ok());
        assert!(InstitutionalFlowProxy::new(5, 5).is_err()); // period too small
        assert!(InstitutionalFlowProxy::new(14, 1).is_err()); // smooth_period too small
    }

    #[test]
    fn test_institutional_flow_calculate() {
        let (close, volume) = make_test_data();
        let ifp = InstitutionalFlowProxy::new(14, 5).unwrap();
        let result = ifp.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Score should be bounded -100 to 100
        for i in 25..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_institutional_flow_is_buying() {
        let ifp = InstitutionalFlowProxy::new(14, 5).unwrap();
        assert!(!ifp.is_institutional_buying(10.0));
        assert!(ifp.is_institutional_buying(20.0));
    }

    #[test]
    fn test_institutional_flow_trait() {
        let data = make_ohlcv_series();
        let ifp = InstitutionalFlowProxy::new(14, 5).unwrap();

        assert_eq!(ifp.name(), "Institutional Flow Proxy");
        assert_eq!(ifp.min_periods(), 19);

        let output = ifp.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== NetworkActivityProxy Tests ==========

    #[test]
    fn test_network_activity_new() {
        assert!(NetworkActivityProxy::new(7, 21).is_ok());
        assert!(NetworkActivityProxy::new(3, 21).is_err()); // period too small
        assert!(NetworkActivityProxy::new(21, 21).is_err()); // baseline must be > period
        assert!(NetworkActivityProxy::new(21, 14).is_err()); // baseline must be > period
    }

    #[test]
    fn test_network_activity_calculate() {
        let (close, volume) = make_test_data();
        let nap = NetworkActivityProxy::new(7, 21).unwrap();
        let result = nap.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Score should be non-negative
        for i in 25..result.len() {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_network_activity_classify() {
        let nap = NetworkActivityProxy::new(7, 21).unwrap();
        assert_eq!(nap.classify_activity(90.0), "Very High");
        assert_eq!(nap.classify_activity(70.0), "High");
        assert_eq!(nap.classify_activity(50.0), "Normal");
        assert_eq!(nap.classify_activity(30.0), "Low");
        assert_eq!(nap.classify_activity(10.0), "Very Low");
    }

    #[test]
    fn test_network_activity_trait() {
        let data = make_ohlcv_series();
        let nap = NetworkActivityProxy::new(7, 21).unwrap();

        assert_eq!(nap.name(), "Network Activity Proxy");
        assert_eq!(nap.min_periods(), 22);

        let output = nap.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== Edge Cases and Integration Tests ==========

    #[test]
    fn test_empty_data() {
        let close: Vec<f64> = vec![];
        let volume: Vec<f64> = vec![];

        let hrm = HashRateMomentum::new(10, 5).unwrap();
        assert!(hrm.calculate(&close, &volume).is_empty());

        let mc = MinerCapitulation::new(14, 0.3).unwrap();
        assert!(mc.calculate(&close, &volume).is_empty());
    }

    #[test]
    fn test_insufficient_data() {
        let close: Vec<f64> = vec![100.0, 101.0, 102.0];
        let volume: Vec<f64> = vec![1000.0, 1100.0, 1200.0];

        let hrm = HashRateMomentum::new(10, 5).unwrap();
        let result = hrm.calculate(&close, &volume);
        // Should return zeros for insufficient data
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_constant_prices() {
        let close: Vec<f64> = vec![100.0; 50];
        let volume: Vec<f64> = vec![1000.0; 50];

        let rsp = RetailSentimentProxy::new(10, 0.5).unwrap();
        let result = rsp.calculate(&close, &volume);

        // With constant prices, sentiment should be near neutral
        for i in 15..result.len() {
            assert!(result[i].abs() < 50.0);
        }
    }

    #[test]
    fn test_all_indicators_with_same_data() {
        let data = make_ohlcv_series();

        let hrm = HashRateMomentum::new(10, 5).unwrap();
        let mc = MinerCapitulation::new(14, 0.3).unwrap();
        let wa = WhaleAccumulation::new(14, 1.5).unwrap();
        let rsp = RetailSentimentProxy::new(14, 0.6).unwrap();
        let ifp = InstitutionalFlowProxy::new(14, 5).unwrap();
        let nap = NetworkActivityProxy::new(7, 21).unwrap();

        // All should compute successfully
        assert!(hrm.compute(&data).is_ok());
        assert!(mc.compute(&data).is_ok());
        assert!(wa.compute(&data).is_ok());
        assert!(rsp.compute(&data).is_ok());
        assert!(ifp.compute(&data).is_ok());
        assert!(nap.compute(&data).is_ok());
    }
}

// ============================================================================
// OnChainMomentum - Momentum from on-chain metrics proxy
// ============================================================================

/// On-Chain Momentum Indicator
///
/// Derives momentum signals from on-chain metrics proxies including volume patterns,
/// price velocity, and transaction intensity to estimate blockchain activity momentum.
#[derive(Debug, Clone)]
pub struct OnChainMomentum {
    period: usize,
    smooth_period: usize,
}

impl OnChainMomentum {
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smooth_period })
    }

    /// Calculate on-chain momentum (-100 to +100)
    /// Positive = bullish on-chain momentum, Negative = bearish
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Volume momentum (proxy for transaction activity)
            let early_vol: f64 = volume[start..(start + self.period / 2).min(i)].iter().sum();
            let late_vol: f64 = volume[(i - self.period / 2)..=i].iter().sum();
            let vol_momentum = if early_vol > 1e-10 {
                (late_vol / early_vol - 1.0) * 100.0
            } else {
                0.0
            };

            // Price velocity (proxy for value transferred momentum)
            let mut price_velocity = 0.0;
            for j in (start + 1)..=i {
                price_velocity += close[j] - close[j - 1];
            }
            let avg_velocity = price_velocity / self.period as f64;
            let velocity_score = (avg_velocity / close[i].max(1e-10)) * 1000.0;

            // Dollar volume momentum (proxy for on-chain value momentum)
            let early_dv: f64 = (start..(start + self.period / 2).min(i))
                .map(|j| close[j] * volume[j])
                .sum();
            let late_dv: f64 = ((i - self.period / 2)..=i)
                .map(|j| close[j] * volume[j])
                .sum();
            let dv_momentum = if early_dv > 1e-10 {
                (late_dv / early_dv - 1.0) * 50.0
            } else {
                0.0
            };

            // Composite on-chain momentum
            result[i] = (vol_momentum * 0.3 + velocity_score * 0.3 + dv_momentum * 0.4)
                .clamp(-100.0, 100.0);
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Returns momentum classification
    pub fn classify_momentum(&self, score: f64) -> &'static str {
        match score {
            s if s >= 40.0 => "Strong Bullish",
            s if s >= 15.0 => "Bullish",
            s if s >= -15.0 => "Neutral",
            s if s >= -40.0 => "Bearish",
            _ => "Strong Bearish",
        }
    }
}

impl TechnicalIndicator for OnChainMomentum {
    fn name(&self) -> &str {
        "On-Chain Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// NetworkHealthIndex - Composite network health indicator
// ============================================================================

/// Network Health Index
///
/// A composite indicator that evaluates overall network health by combining
/// volume stability, price trend strength, and activity consistency metrics.
#[derive(Debug, Clone)]
pub struct NetworkHealthIndex {
    period: usize,
    baseline_period: usize,
}

impl NetworkHealthIndex {
    pub fn new(period: usize, baseline_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if baseline_period <= period {
            return Err(IndicatorError::InvalidParameter {
                name: "baseline_period".to_string(),
                reason: "must be greater than period".to_string(),
            });
        }
        Ok(Self { period, baseline_period })
    }

    /// Calculate network health index (0 to 100)
    /// Higher values indicate healthier network metrics
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.baseline_period..n {
            let period_start = i.saturating_sub(self.period);
            let baseline_start = i.saturating_sub(self.baseline_period);

            // Volume stability (low variance = healthy)
            let period_vol: Vec<f64> = volume[period_start..=i].to_vec();
            let avg_vol: f64 = period_vol.iter().sum::<f64>() / period_vol.len() as f64;
            let vol_variance: f64 = period_vol.iter()
                .map(|v| (v - avg_vol).powi(2))
                .sum::<f64>() / period_vol.len() as f64;
            let vol_cv = if avg_vol > 1e-10 {
                vol_variance.sqrt() / avg_vol
            } else {
                1.0
            };
            let stability_score = (1.0 - vol_cv.min(1.0)) * 30.0;

            // Trend consistency (smooth trends = healthy)
            let mut up_moves = 0;
            let mut down_moves = 0;
            for j in (period_start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_moves += 1;
                } else if close[j] < close[j - 1] {
                    down_moves += 1;
                }
            }
            let total_moves = up_moves + down_moves;
            let trend_consistency = if total_moves > 0 {
                (up_moves.max(down_moves) as f64 / total_moves as f64 - 0.5) * 2.0
            } else {
                0.5
            };
            let trend_score = trend_consistency * 25.0;

            // Activity level relative to baseline
            let baseline_vol: f64 = volume[baseline_start..period_start].iter().sum::<f64>()
                / (period_start - baseline_start).max(1) as f64;
            let activity_ratio = if baseline_vol > 1e-10 {
                avg_vol / baseline_vol
            } else {
                1.0
            };
            let activity_score = (activity_ratio.min(2.0) / 2.0) * 25.0;

            // Price health (not extreme moves)
            let price_range: f64 = close[period_start..=i].iter()
                .cloned()
                .fold(f64::MIN, f64::max) -
                close[period_start..=i].iter()
                .cloned()
                .fold(f64::MAX, f64::min);
            let avg_price = close[period_start..=i].iter().sum::<f64>() / self.period as f64;
            let range_pct = if avg_price > 1e-10 {
                price_range / avg_price
            } else {
                1.0
            };
            let price_health = (1.0 - (range_pct * 5.0).min(1.0)) * 20.0;

            result[i] = (stability_score + trend_score + activity_score + price_health).clamp(0.0, 100.0);
        }

        result
    }

    /// Returns health classification
    pub fn classify_health(&self, score: f64) -> &'static str {
        match score {
            s if s >= 80.0 => "Excellent",
            s if s >= 60.0 => "Good",
            s if s >= 40.0 => "Fair",
            s if s >= 20.0 => "Poor",
            _ => "Critical",
        }
    }
}

impl TechnicalIndicator for NetworkHealthIndex {
    fn name(&self) -> &str {
        "Network Health Index"
    }

    fn min_periods(&self) -> usize {
        self.baseline_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// HodlerBehaviorIndex - Index of long-term holder behavior
// ============================================================================

/// Hodler Behavior Index
///
/// Estimates long-term holder behavior by analyzing volume patterns, price stability,
/// and accumulation signals that typically characterize patient, long-term investors.
#[derive(Debug, Clone)]
pub struct HodlerBehaviorIndex {
    short_period: usize,
    long_period: usize,
}

impl HodlerBehaviorIndex {
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

    /// Calculate hodler behavior index (-100 to +100)
    /// Positive = accumulation/holding, Negative = distribution/selling
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let short_start = i.saturating_sub(self.short_period);
            let long_start = i.saturating_sub(self.long_period);

            // Long-term price trend (hodlers prefer uptrends)
            let long_return = if close[long_start] > 1e-10 {
                (close[i] / close[long_start] - 1.0) * 100.0
            } else {
                0.0
            };
            let trend_score = long_return.clamp(-50.0, 50.0) * 0.4;

            // Volume decline during downtrends (hodlers don't panic sell)
            let short_vol: f64 = volume[short_start..=i].iter().sum::<f64>() / self.short_period as f64;
            let long_vol: f64 = volume[long_start..short_start].iter().sum::<f64>()
                / (short_start - long_start).max(1) as f64;

            let short_return = if close[short_start] > 1e-10 {
                (close[i] / close[short_start] - 1.0) * 100.0
            } else {
                0.0
            };

            // If price falling but volume decreasing = hodlers holding
            let hodl_signal = if short_return < 0.0 && long_vol > 1e-10 {
                let vol_ratio = short_vol / long_vol;
                if vol_ratio < 1.0 {
                    (1.0 - vol_ratio) * 30.0 // Low volume on drops = holding
                } else {
                    -(vol_ratio - 1.0).min(1.0) * 30.0 // High volume on drops = selling
                }
            } else if short_return > 0.0 && long_vol > 1e-10 {
                let vol_ratio = short_vol / long_vol;
                if vol_ratio < 1.5 {
                    15.0 // Steady volume on rises = accumulation
                } else {
                    -10.0 // Very high volume on rises = profit taking
                }
            } else {
                0.0
            };

            // Price stability score (hodlers create support/resistance)
            let price_slice = &close[long_start..=i];
            let avg_price = price_slice.iter().sum::<f64>() / price_slice.len() as f64;
            let price_dev: f64 = price_slice.iter()
                .map(|p| (p - avg_price).abs())
                .sum::<f64>() / price_slice.len() as f64;
            let stability = if avg_price > 1e-10 {
                1.0 - (price_dev / avg_price * 10.0).min(1.0)
            } else {
                0.5
            };
            let stability_score = stability * 20.0;

            result[i] = (trend_score + hodl_signal + stability_score).clamp(-100.0, 100.0);
        }

        result
    }

    /// Returns behavior classification
    pub fn classify_behavior(&self, score: f64) -> &'static str {
        match score {
            s if s >= 40.0 => "Strong Accumulation",
            s if s >= 15.0 => "Accumulation",
            s if s >= -15.0 => "Neutral",
            s if s >= -40.0 => "Distribution",
            _ => "Strong Distribution",
        }
    }
}

impl TechnicalIndicator for HodlerBehaviorIndex {
    fn name(&self) -> &str {
        "Hodler Behavior Index"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// ExchangeFlowMomentum - Momentum of exchange in/outflows
// ============================================================================

/// Exchange Flow Momentum
///
/// Estimates exchange in/outflow momentum using volume patterns and price action
/// as proxies. High exchange inflows often precede selling, while outflows suggest
/// accumulation to cold storage.
#[derive(Debug, Clone)]
pub struct ExchangeFlowMomentum {
    period: usize,
    sensitivity: f64,
}

impl ExchangeFlowMomentum {
    pub fn new(period: usize, sensitivity: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if sensitivity <= 0.0 || sensitivity > 2.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sensitivity".to_string(),
                reason: "must be between 0 and 2".to_string(),
            });
        }
        Ok(Self { period, sensitivity })
    }

    /// Calculate exchange flow momentum (-100 to +100)
    /// Positive = net outflows (bullish), Negative = net inflows (bearish)
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate average volume for baseline
            let avg_vol: f64 = volume[start..i].iter().sum::<f64>() / (i - start) as f64;

            // Analyze volume/price patterns for flow signals
            let mut inflow_signal = 0.0;
            let mut outflow_signal = 0.0;

            for j in start..=i {
                if j > 0 && avg_vol > 1e-10 {
                    let price_change = close[j] - close[j - 1];
                    let vol_ratio = volume[j] / avg_vol;

                    // Inflow pattern: high volume with price weakness
                    if price_change < 0.0 && vol_ratio > 1.2 {
                        inflow_signal += vol_ratio * price_change.abs() / close[j - 1] * 100.0;
                    }
                    // Outflow pattern: controlled volume with price strength
                    else if price_change > 0.0 && vol_ratio < 1.5 && vol_ratio > 0.5 {
                        outflow_signal += (1.5 - vol_ratio) * price_change / close[j - 1] * 100.0;
                    }
                    // Strong outflow: low volume consolidation (coins moving off exchange)
                    else if price_change.abs() / close[j - 1].max(1e-10) < 0.01 && vol_ratio < 0.7 {
                        outflow_signal += (1.0 - vol_ratio) * 10.0;
                    }
                }
            }

            // Net flow momentum (outflows positive, inflows negative)
            let net_flow = (outflow_signal - inflow_signal) * self.sensitivity;
            result[i] = net_flow.clamp(-100.0, 100.0);
        }

        // Apply light smoothing
        for i in 1..n {
            result[i] = 0.7 * result[i] + 0.3 * result[i - 1];
        }

        result
    }

    /// Returns flow classification
    pub fn classify_flow(&self, score: f64) -> &'static str {
        match score {
            s if s >= 30.0 => "Strong Outflow",
            s if s >= 10.0 => "Net Outflow",
            s if s >= -10.0 => "Balanced",
            s if s >= -30.0 => "Net Inflow",
            _ => "Strong Inflow",
        }
    }
}

impl TechnicalIndicator for ExchangeFlowMomentum {
    fn name(&self) -> &str {
        "Exchange Flow Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// MinerBehaviorIndex - Index of miner selling/holding behavior
// ============================================================================

/// Miner Behavior Index
///
/// Analyzes patterns typically associated with miner behavior including
/// consistent selling pressure, capitulation events, and accumulation phases.
#[derive(Debug, Clone)]
pub struct MinerBehaviorIndex {
    period: usize,
    capitulation_threshold: f64,
}

impl MinerBehaviorIndex {
    pub fn new(period: usize, capitulation_threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if capitulation_threshold <= 0.0 || capitulation_threshold > 0.5 {
            return Err(IndicatorError::InvalidParameter {
                name: "capitulation_threshold".to_string(),
                reason: "must be between 0 and 0.5".to_string(),
            });
        }
        Ok(Self { period, capitulation_threshold })
    }

    /// Calculate miner behavior index (-100 to +100)
    /// Positive = miners holding/accumulating, Negative = miners selling
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Price trend (miners often sell into strength)
            let price_return = if close[start] > 1e-10 {
                (close[i] / close[start] - 1.0)
            } else {
                0.0
            };

            // Period high and drawdown
            let period_high = close[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let drawdown = if period_high > 1e-10 {
                (period_high - close[i]) / period_high
            } else {
                0.0
            };

            // Volume analysis
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
            let recent_vol: f64 = volume[(i - 2).max(start)..=i].iter().sum::<f64>() / 3.0;
            let vol_ratio = if avg_vol > 1e-10 { recent_vol / avg_vol } else { 1.0 };

            // Capitulation detection
            let is_capitulating = drawdown > self.capitulation_threshold && vol_ratio > 1.5;

            // Selling pressure score
            let sell_pressure = if price_return > 0.02 && vol_ratio > 1.2 {
                // Strong price with high volume = miners selling into strength
                -30.0 * vol_ratio.min(2.0)
            } else if is_capitulating {
                // Capitulation = forced selling
                -50.0 - drawdown * 100.0
            } else if price_return < -0.05 && vol_ratio > 1.5 {
                // Decline with high volume = panic selling
                -40.0
            } else {
                0.0
            };

            // Holding/accumulation score
            let hold_score = if price_return < 0.0 && vol_ratio < 0.8 {
                // Price decline with low volume = holding through dip
                20.0 * (1.0 - vol_ratio)
            } else if price_return.abs() < 0.02 && vol_ratio < 0.9 {
                // Consolidation with low volume = accumulation
                15.0
            } else if price_return > 0.0 && vol_ratio < 1.0 {
                // Price rise with normal volume = healthy
                10.0
            } else {
                0.0
            };

            result[i] = (sell_pressure + hold_score).clamp(-100.0, 100.0);
        }

        result
    }

    /// Returns behavior classification
    pub fn classify_behavior(&self, score: f64) -> &'static str {
        match score {
            s if s >= 30.0 => "Strong Holding",
            s if s >= 10.0 => "Holding",
            s if s >= -10.0 => "Neutral",
            s if s >= -30.0 => "Selling",
            _ => "Capitulation",
        }
    }
}

impl TechnicalIndicator for MinerBehaviorIndex {
    fn name(&self) -> &str {
        "Miner Behavior Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// InstitutionalFlowIndex - Tracks institutional flow patterns
// ============================================================================

/// Institutional Flow Index
///
/// Advanced indicator tracking institutional investment flow patterns using
/// analysis of large-scale, systematic trading behavior visible in volume
/// and price action patterns.
#[derive(Debug, Clone)]
pub struct InstitutionalFlowIndex {
    period: usize,
    smooth_period: usize,
    size_threshold: f64,
}

impl InstitutionalFlowIndex {
    pub fn new(period: usize, smooth_period: usize, size_threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if size_threshold <= 1.0 || size_threshold > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "size_threshold".to_string(),
                reason: "must be between 1 and 5".to_string(),
            });
        }
        Ok(Self { period, smooth_period, size_threshold })
    }

    /// Calculate institutional flow index (-100 to +100)
    /// Positive = institutional buying, Negative = institutional selling
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate volume statistics
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
            let vol_variance: f64 = volume[start..=i]
                .iter()
                .map(|v| (v - avg_vol).powi(2))
                .sum::<f64>() / (i - start + 1) as f64;
            let vol_std = vol_variance.sqrt();

            // Detect institutional-sized trades
            let mut institutional_buying = 0.0;
            let mut institutional_selling = 0.0;
            let mut large_trade_count = 0;

            for j in start..=i {
                if j > 0 && vol_std > 1e-10 {
                    let vol_zscore = (volume[j] - avg_vol) / vol_std;

                    // Large volume bar (potential institutional activity)
                    if vol_zscore > self.size_threshold - 1.0 {
                        large_trade_count += 1;
                        let price_change = close[j] - close[j - 1];
                        let price_pct = price_change / close[j - 1].max(1e-10) * 100.0;

                        // Institutional pattern: large volume with controlled price impact
                        let impact_efficiency = if price_pct.abs() > 0.0 {
                            vol_zscore / price_pct.abs()
                        } else {
                            vol_zscore * 2.0 // Very efficient if no price impact
                        };

                        if price_change > 0.0 {
                            institutional_buying += impact_efficiency.min(10.0);
                        } else if price_change < 0.0 {
                            institutional_selling += impact_efficiency.min(10.0);
                        } else {
                            // Flat price with high volume = accumulation
                            institutional_buying += vol_zscore * 0.5;
                        }
                    }
                }
            }

            // TWAP/VWAP pattern detection (institutions use algorithmic execution)
            let mut consistent_direction = 0;
            let mut last_direction = 0;
            for j in (start + 1)..=i {
                let direction = if close[j] > close[j - 1] { 1 } else if close[j] < close[j - 1] { -1 } else { 0 };
                if direction != 0 && direction == last_direction {
                    consistent_direction += 1;
                }
                last_direction = direction;
            }
            let algo_score = if consistent_direction > self.period / 3 {
                last_direction as f64 * (consistent_direction as f64 / self.period as f64) * 20.0
            } else {
                0.0
            };

            // Net institutional flow
            let flow_score = if large_trade_count > 0 {
                (institutional_buying - institutional_selling) / large_trade_count as f64 * 15.0
            } else {
                0.0
            };

            result[i] = (flow_score + algo_score).clamp(-100.0, 100.0);
        }

        // Apply EMA smoothing (institutional activity unfolds gradually)
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Returns flow classification
    pub fn classify_flow(&self, score: f64) -> &'static str {
        match score {
            s if s >= 40.0 => "Heavy Institutional Buying",
            s if s >= 15.0 => "Institutional Buying",
            s if s >= -15.0 => "Neutral",
            s if s >= -40.0 => "Institutional Selling",
            _ => "Heavy Institutional Selling",
        }
    }

    /// Returns true if significant institutional activity detected
    pub fn is_institutional_active(&self, score: f64) -> bool {
        score.abs() > 20.0
    }
}

impl TechnicalIndicator for InstitutionalFlowIndex {
    fn name(&self) -> &str {
        "Institutional Flow Index"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// Unit Tests for New Indicators
// ============================================================================

#[cfg(test)]
mod new_tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        let mut close = Vec::with_capacity(50);
        let mut volume = Vec::with_capacity(50);

        let mut price = 100.0;
        for i in 0..50 {
            let trend = if i < 20 { 0.5 } else if i < 35 { -0.3 } else { 0.4 };
            let noise = ((i as f64 * 0.7).sin() * 2.0);
            price += trend + noise;
            price = price.max(50.0);
            close.push(price);

            let base_vol = 1000.0 + (i as f64 * 20.0);
            let vol_spike = if i % 7 == 0 { 2.0 } else { 1.0 };
            volume.push(base_vol * vol_spike);
        }

        (close, volume)
    }

    fn make_ohlcv_series() -> OHLCVSeries {
        let (close, volume) = make_test_data();
        let high: Vec<f64> = close.iter().map(|c| c + 1.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.5).collect();
        let open = close.clone();

        OHLCVSeries { open, high, low, close, volume }
    }

    // ========== OnChainMomentum Tests ==========

    #[test]
    fn test_on_chain_momentum_new() {
        assert!(OnChainMomentum::new(10, 5).is_ok());
        assert!(OnChainMomentum::new(4, 5).is_err());
        assert!(OnChainMomentum::new(10, 1).is_err());
    }

    #[test]
    fn test_on_chain_momentum_calculate() {
        let (close, volume) = make_test_data();
        let ocm = OnChainMomentum::new(10, 5).unwrap();
        let result = ocm.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 20..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_on_chain_momentum_classify() {
        let ocm = OnChainMomentum::new(10, 5).unwrap();
        assert_eq!(ocm.classify_momentum(50.0), "Strong Bullish");
        assert_eq!(ocm.classify_momentum(20.0), "Bullish");
        assert_eq!(ocm.classify_momentum(0.0), "Neutral");
        assert_eq!(ocm.classify_momentum(-20.0), "Bearish");
        assert_eq!(ocm.classify_momentum(-50.0), "Strong Bearish");
    }

    #[test]
    fn test_on_chain_momentum_trait() {
        let data = make_ohlcv_series();
        let ocm = OnChainMomentum::new(10, 5).unwrap();

        assert_eq!(ocm.name(), "On-Chain Momentum");
        assert_eq!(ocm.min_periods(), 15);

        let output = ocm.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== NetworkHealthIndex Tests ==========

    #[test]
    fn test_network_health_index_new() {
        assert!(NetworkHealthIndex::new(7, 21).is_ok());
        assert!(NetworkHealthIndex::new(3, 21).is_err());
        assert!(NetworkHealthIndex::new(21, 21).is_err());
        assert!(NetworkHealthIndex::new(21, 14).is_err());
    }

    #[test]
    fn test_network_health_index_calculate() {
        let (close, volume) = make_test_data();
        let nhi = NetworkHealthIndex::new(7, 21).unwrap();
        let result = nhi.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_network_health_index_classify() {
        let nhi = NetworkHealthIndex::new(7, 21).unwrap();
        assert_eq!(nhi.classify_health(90.0), "Excellent");
        assert_eq!(nhi.classify_health(70.0), "Good");
        assert_eq!(nhi.classify_health(50.0), "Fair");
        assert_eq!(nhi.classify_health(30.0), "Poor");
        assert_eq!(nhi.classify_health(10.0), "Critical");
    }

    #[test]
    fn test_network_health_index_trait() {
        let data = make_ohlcv_series();
        let nhi = NetworkHealthIndex::new(7, 21).unwrap();

        assert_eq!(nhi.name(), "Network Health Index");
        assert_eq!(nhi.min_periods(), 22);

        let output = nhi.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== HodlerBehaviorIndex Tests ==========

    #[test]
    fn test_hodler_behavior_index_new() {
        assert!(HodlerBehaviorIndex::new(7, 21).is_ok());
        assert!(HodlerBehaviorIndex::new(3, 21).is_err());
        assert!(HodlerBehaviorIndex::new(21, 21).is_err());
        assert!(HodlerBehaviorIndex::new(21, 14).is_err());
    }

    #[test]
    fn test_hodler_behavior_index_calculate() {
        let (close, volume) = make_test_data();
        let hbi = HodlerBehaviorIndex::new(7, 21).unwrap();
        let result = hbi.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 25..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_hodler_behavior_index_classify() {
        let hbi = HodlerBehaviorIndex::new(7, 21).unwrap();
        assert_eq!(hbi.classify_behavior(50.0), "Strong Accumulation");
        assert_eq!(hbi.classify_behavior(20.0), "Accumulation");
        assert_eq!(hbi.classify_behavior(0.0), "Neutral");
        assert_eq!(hbi.classify_behavior(-20.0), "Distribution");
        assert_eq!(hbi.classify_behavior(-50.0), "Strong Distribution");
    }

    #[test]
    fn test_hodler_behavior_index_trait() {
        let data = make_ohlcv_series();
        let hbi = HodlerBehaviorIndex::new(7, 21).unwrap();

        assert_eq!(hbi.name(), "Hodler Behavior Index");
        assert_eq!(hbi.min_periods(), 22);

        let output = hbi.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== ExchangeFlowMomentum Tests ==========

    #[test]
    fn test_exchange_flow_momentum_new() {
        assert!(ExchangeFlowMomentum::new(14, 1.0).is_ok());
        assert!(ExchangeFlowMomentum::new(3, 1.0).is_err());
        assert!(ExchangeFlowMomentum::new(14, 0.0).is_err());
        assert!(ExchangeFlowMomentum::new(14, 2.5).is_err());
    }

    #[test]
    fn test_exchange_flow_momentum_calculate() {
        let (close, volume) = make_test_data();
        let efm = ExchangeFlowMomentum::new(14, 1.0).unwrap();
        let result = efm.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 20..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_exchange_flow_momentum_classify() {
        let efm = ExchangeFlowMomentum::new(14, 1.0).unwrap();
        assert_eq!(efm.classify_flow(40.0), "Strong Outflow");
        assert_eq!(efm.classify_flow(20.0), "Net Outflow");
        assert_eq!(efm.classify_flow(0.0), "Balanced");
        assert_eq!(efm.classify_flow(-20.0), "Net Inflow");
        assert_eq!(efm.classify_flow(-40.0), "Strong Inflow");
    }

    #[test]
    fn test_exchange_flow_momentum_trait() {
        let data = make_ohlcv_series();
        let efm = ExchangeFlowMomentum::new(14, 1.0).unwrap();

        assert_eq!(efm.name(), "Exchange Flow Momentum");
        assert_eq!(efm.min_periods(), 15);

        let output = efm.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== MinerBehaviorIndex Tests ==========

    #[test]
    fn test_miner_behavior_index_new() {
        assert!(MinerBehaviorIndex::new(14, 0.2).is_ok());
        assert!(MinerBehaviorIndex::new(5, 0.2).is_err());
        assert!(MinerBehaviorIndex::new(14, 0.0).is_err());
        assert!(MinerBehaviorIndex::new(14, 0.6).is_err());
    }

    #[test]
    fn test_miner_behavior_index_calculate() {
        let (close, volume) = make_test_data();
        let mbi = MinerBehaviorIndex::new(14, 0.2).unwrap();
        let result = mbi.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 20..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_miner_behavior_index_classify() {
        let mbi = MinerBehaviorIndex::new(14, 0.2).unwrap();
        assert_eq!(mbi.classify_behavior(40.0), "Strong Holding");
        assert_eq!(mbi.classify_behavior(20.0), "Holding");
        assert_eq!(mbi.classify_behavior(0.0), "Neutral");
        assert_eq!(mbi.classify_behavior(-20.0), "Selling");
        assert_eq!(mbi.classify_behavior(-50.0), "Capitulation");
    }

    #[test]
    fn test_miner_behavior_index_trait() {
        let data = make_ohlcv_series();
        let mbi = MinerBehaviorIndex::new(14, 0.2).unwrap();

        assert_eq!(mbi.name(), "Miner Behavior Index");
        assert_eq!(mbi.min_periods(), 15);

        let output = mbi.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== InstitutionalFlowIndex Tests ==========

    #[test]
    fn test_institutional_flow_index_new() {
        assert!(InstitutionalFlowIndex::new(14, 5, 2.0).is_ok());
        assert!(InstitutionalFlowIndex::new(5, 5, 2.0).is_err());
        assert!(InstitutionalFlowIndex::new(14, 1, 2.0).is_err());
        assert!(InstitutionalFlowIndex::new(14, 5, 0.5).is_err());
        assert!(InstitutionalFlowIndex::new(14, 5, 6.0).is_err());
    }

    #[test]
    fn test_institutional_flow_index_calculate() {
        let (close, volume) = make_test_data();
        let ifi = InstitutionalFlowIndex::new(14, 5, 2.0).unwrap();
        let result = ifi.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 25..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_institutional_flow_index_classify() {
        let ifi = InstitutionalFlowIndex::new(14, 5, 2.0).unwrap();
        assert_eq!(ifi.classify_flow(50.0), "Heavy Institutional Buying");
        assert_eq!(ifi.classify_flow(25.0), "Institutional Buying");
        assert_eq!(ifi.classify_flow(0.0), "Neutral");
        assert_eq!(ifi.classify_flow(-25.0), "Institutional Selling");
        assert_eq!(ifi.classify_flow(-50.0), "Heavy Institutional Selling");
    }

    #[test]
    fn test_institutional_flow_index_is_active() {
        let ifi = InstitutionalFlowIndex::new(14, 5, 2.0).unwrap();
        assert!(!ifi.is_institutional_active(15.0));
        assert!(ifi.is_institutional_active(25.0));
        assert!(ifi.is_institutional_active(-25.0));
    }

    #[test]
    fn test_institutional_flow_index_trait() {
        let data = make_ohlcv_series();
        let ifi = InstitutionalFlowIndex::new(14, 5, 2.0).unwrap();

        assert_eq!(ifi.name(), "Institutional Flow Index");
        assert_eq!(ifi.min_periods(), 19);

        let output = ifi.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== Integration Tests ==========

    #[test]
    fn test_all_new_indicators_with_same_data() {
        let data = make_ohlcv_series();

        let ocm = OnChainMomentum::new(10, 5).unwrap();
        let nhi = NetworkHealthIndex::new(7, 21).unwrap();
        let hbi = HodlerBehaviorIndex::new(7, 21).unwrap();
        let efm = ExchangeFlowMomentum::new(14, 1.0).unwrap();
        let mbi = MinerBehaviorIndex::new(14, 0.2).unwrap();
        let ifi = InstitutionalFlowIndex::new(14, 5, 2.0).unwrap();

        assert!(ocm.compute(&data).is_ok());
        assert!(nhi.compute(&data).is_ok());
        assert!(hbi.compute(&data).is_ok());
        assert!(efm.compute(&data).is_ok());
        assert!(mbi.compute(&data).is_ok());
        assert!(ifi.compute(&data).is_ok());
    }

    #[test]
    fn test_new_indicators_empty_data() {
        let close: Vec<f64> = vec![];
        let volume: Vec<f64> = vec![];

        let ocm = OnChainMomentum::new(10, 5).unwrap();
        assert!(ocm.calculate(&close, &volume).is_empty());

        let nhi = NetworkHealthIndex::new(7, 21).unwrap();
        assert!(nhi.calculate(&close, &volume).is_empty());

        let hbi = HodlerBehaviorIndex::new(7, 21).unwrap();
        assert!(hbi.calculate(&close, &volume).is_empty());

        let efm = ExchangeFlowMomentum::new(14, 1.0).unwrap();
        assert!(efm.calculate(&close, &volume).is_empty());

        let mbi = MinerBehaviorIndex::new(14, 0.2).unwrap();
        assert!(mbi.calculate(&close, &volume).is_empty());

        let ifi = InstitutionalFlowIndex::new(14, 5, 2.0).unwrap();
        assert!(ifi.calculate(&close, &volume).is_empty());
    }
}

// ============================================================================
// CryptoTrendStrength - Crypto-specific trend strength indicator
// ============================================================================

/// Crypto Trend Strength Indicator
///
/// Measures the strength of price trends in cryptocurrency markets using
/// a combination of momentum, volume confirmation, and volatility-adjusted
/// price movement. Unlike traditional trend strength indicators, this accounts
/// for the high volatility and 24/7 nature of crypto markets.
///
/// The indicator produces values from 0 to 100:
/// - 0-20: No trend / Ranging market
/// - 20-40: Weak trend
/// - 40-60: Moderate trend
/// - 60-80: Strong trend
/// - 80-100: Very strong trend (potentially parabolic)
#[derive(Debug, Clone)]
pub struct CryptoTrendStrength {
    /// Main lookback period for trend analysis
    period: usize,
    /// Smoothing period for noise reduction
    smooth_period: usize,
    /// Volatility adjustment factor
    volatility_factor: f64,
}

impl CryptoTrendStrength {
    /// Creates a new CryptoTrendStrength indicator
    ///
    /// # Arguments
    /// * `period` - Main lookback period (minimum 5)
    /// * `smooth_period` - Smoothing period for EMA (minimum 2)
    /// * `volatility_factor` - Factor for volatility adjustment (0.1 to 3.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error
    pub fn new(period: usize, smooth_period: usize, volatility_factor: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volatility_factor <= 0.1 || volatility_factor > 3.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_factor".to_string(),
                reason: "must be between 0.1 and 3.0".to_string(),
            });
        }
        Ok(Self { period, smooth_period, volatility_factor })
    }

    /// Calculate crypto trend strength (0 to 100)
    ///
    /// Higher values indicate stronger trends with volume confirmation.
    /// Uses directional movement, volume momentum, and volatility normalization.
    pub fn calculate(&self, close: &[f64], high: &[f64], low: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // 1. Directional Movement Component (ADX-like)
            let mut plus_dm_sum = 0.0;
            let mut minus_dm_sum = 0.0;
            let mut tr_sum = 0.0;

            for j in (start + 1)..=i {
                let high_diff = high[j] - high[j - 1];
                let low_diff = low[j - 1] - low[j];

                // Plus directional movement
                if high_diff > low_diff && high_diff > 0.0 {
                    plus_dm_sum += high_diff;
                }
                // Minus directional movement
                if low_diff > high_diff && low_diff > 0.0 {
                    minus_dm_sum += low_diff;
                }

                // True range
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                tr_sum += tr;
            }

            let plus_di = if tr_sum > 1e-10 { plus_dm_sum / tr_sum * 100.0 } else { 0.0 };
            let minus_di = if tr_sum > 1e-10 { minus_dm_sum / tr_sum * 100.0 } else { 0.0 };
            let di_diff = (plus_di - minus_di).abs();
            let di_sum = plus_di + minus_di;
            let dx = if di_sum > 1e-10 { di_diff / di_sum * 100.0 } else { 0.0 };

            // 2. Price Momentum Component
            let price_return = if close[start] > 1e-10 {
                ((close[i] / close[start]) - 1.0).abs() * 100.0
            } else {
                0.0
            };

            // 3. Volume Confirmation Component
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
            let recent_vol: f64 = volume[(i - 2).max(start)..=i].iter().sum::<f64>() / 3.0;
            let vol_ratio = if avg_vol > 1e-10 { recent_vol / avg_vol } else { 1.0 };
            let volume_score = (vol_ratio - 0.5).max(0.0).min(2.0) * 25.0;

            // 4. Volatility-Adjusted Trend Consistency
            let mut consecutive_up = 0;
            let mut consecutive_down = 0;
            let mut max_streak = 0;
            let mut current_streak = 0;
            let mut last_direction = 0;

            for j in (start + 1)..=i {
                let direction = if close[j] > close[j - 1] { 1 } else if close[j] < close[j - 1] { -1 } else { 0 };
                if direction == last_direction && direction != 0 {
                    current_streak += 1;
                    max_streak = max_streak.max(current_streak);
                } else {
                    current_streak = 1;
                }
                last_direction = direction;
                if direction > 0 { consecutive_up += 1; }
                else if direction < 0 { consecutive_down += 1; }
            }

            let consistency = max_streak as f64 / self.period as f64 * 30.0;

            // 5. Volatility Normalization
            let atr = tr_sum / self.period as f64;
            let avg_price = close[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
            let volatility_pct = if avg_price > 1e-10 { atr / avg_price } else { 0.0 };
            let vol_adj = 1.0 / (1.0 + volatility_pct * self.volatility_factor * 10.0);

            // Composite trend strength
            let raw_strength = (dx * 0.35 + price_return * 0.25 + volume_score * 0.20 + consistency * 0.20) * vol_adj;
            result[i] = raw_strength.clamp(0.0, 100.0);
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Returns trend strength classification
    pub fn classify_strength(&self, score: f64) -> &'static str {
        match score {
            s if s >= 80.0 => "Very Strong (Parabolic)",
            s if s >= 60.0 => "Strong",
            s if s >= 40.0 => "Moderate",
            s if s >= 20.0 => "Weak",
            _ => "No Trend",
        }
    }

    /// Returns true if trend is considered tradeable (strength > 40)
    pub fn is_trending(&self, score: f64) -> bool {
        score >= 40.0
    }
}

impl TechnicalIndicator for CryptoTrendStrength {
    fn name(&self) -> &str {
        "Crypto Trend Strength"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(
            &data.close,
            &data.high,
            &data.low,
            &data.volume,
        )))
    }
}

// ============================================================================
// VolatilityRegimeIndex - Identifies crypto volatility regimes
// ============================================================================

/// Volatility Regime Index
///
/// Identifies the current volatility regime in cryptocurrency markets.
/// Crypto markets cycle through distinct volatility regimes which have
/// significant implications for trading strategies.
///
/// Regime Classification:
/// - Low Volatility (0-25): Consolidation, range-bound
/// - Normal Volatility (25-50): Typical market conditions
/// - High Volatility (50-75): Active trading, trending
/// - Extreme Volatility (75-100): Crisis or euphoria
#[derive(Debug, Clone)]
pub struct VolatilityRegimeIndex {
    /// Short-term volatility period
    short_period: usize,
    /// Long-term volatility period for comparison
    long_period: usize,
    /// Number of regimes to consider
    regime_sensitivity: f64,
}

impl VolatilityRegimeIndex {
    /// Creates a new VolatilityRegimeIndex
    ///
    /// # Arguments
    /// * `short_period` - Short-term volatility window (minimum 5)
    /// * `long_period` - Long-term baseline window (must be > short_period)
    /// * `regime_sensitivity` - Sensitivity to regime changes (0.5 to 2.0)
    pub fn new(short_period: usize, long_period: usize, regime_sensitivity: f64) -> Result<Self> {
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
        if regime_sensitivity < 0.5 || regime_sensitivity > 2.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "regime_sensitivity".to_string(),
                reason: "must be between 0.5 and 2.0".to_string(),
            });
        }
        Ok(Self { short_period, long_period, regime_sensitivity })
    }

    /// Calculate volatility regime index (0 to 100)
    ///
    /// Combines multiple volatility measures including realized volatility,
    /// intraday range, and volatility trend.
    pub fn calculate(&self, close: &[f64], high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let short_start = i.saturating_sub(self.short_period);
            let long_start = i.saturating_sub(self.long_period);

            // 1. Realized Volatility (close-to-close returns)
            let mut short_returns_sq = 0.0;
            let mut long_returns_sq = 0.0;

            for j in (short_start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    let ret = (close[j] / close[j - 1]).ln();
                    short_returns_sq += ret * ret;
                }
            }
            for j in (long_start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    let ret = (close[j] / close[j - 1]).ln();
                    long_returns_sq += ret * ret;
                }
            }

            let short_rv = (short_returns_sq / self.short_period as f64).sqrt() * (252.0_f64).sqrt() * 100.0;
            let long_rv = (long_returns_sq / self.long_period as f64).sqrt() * (252.0_f64).sqrt() * 100.0;

            // 2. Intraday Range Volatility (Parkinson)
            let mut short_range_sq = 0.0;
            for j in short_start..=i {
                if low[j] > 1e-10 {
                    let range_ratio = (high[j] / low[j]).ln();
                    short_range_sq += range_ratio * range_ratio;
                }
            }
            let parkinson_vol = (short_range_sq / (4.0 * 0.693 * self.short_period as f64)).sqrt() * (252.0_f64).sqrt() * 100.0;

            // 3. Volatility Ratio (short vs long)
            let vol_ratio = if long_rv > 1e-10 {
                short_rv / long_rv
            } else {
                1.0
            };

            // 4. ATR-based volatility
            let mut atr_sum = 0.0;
            for j in (short_start + 1)..=i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                atr_sum += tr;
            }
            let atr = atr_sum / self.short_period as f64;
            let avg_price = close[short_start..=i].iter().sum::<f64>() / (i - short_start + 1) as f64;
            let atr_pct = if avg_price > 1e-10 { atr / avg_price * 100.0 } else { 0.0 };

            // 5. Volatility acceleration
            let mid_point = (short_start + i) / 2;
            let first_half_vol = if mid_point > short_start {
                let mut sq = 0.0;
                for j in (short_start + 1)..=mid_point {
                    if close[j - 1] > 1e-10 {
                        let ret = (close[j] / close[j - 1]).ln();
                        sq += ret * ret;
                    }
                }
                (sq / (mid_point - short_start) as f64).sqrt()
            } else { 0.0 };

            let second_half_vol = if i > mid_point {
                let mut sq = 0.0;
                for j in (mid_point + 1)..=i {
                    if close[j - 1] > 1e-10 {
                        let ret = (close[j] / close[j - 1]).ln();
                        sq += ret * ret;
                    }
                }
                (sq / (i - mid_point) as f64).sqrt()
            } else { 0.0 };

            let vol_acceleration = if first_half_vol > 1e-10 {
                second_half_vol / first_half_vol - 1.0
            } else {
                0.0
            };

            // Composite regime score
            let rv_component = (short_rv / 100.0).min(1.0) * 30.0; // Normalize typical crypto vol
            let range_component = (parkinson_vol / 100.0).min(1.0) * 20.0;
            let ratio_component = ((vol_ratio - 0.5).max(0.0) * 2.0).min(1.0) * 25.0;
            let atr_component = (atr_pct / 5.0).min(1.0) * 15.0; // 5% daily ATR = max
            let accel_component = (vol_acceleration.abs() * 2.0).min(1.0) * 10.0;

            let raw_score = (rv_component + range_component + ratio_component + atr_component + accel_component)
                * self.regime_sensitivity;
            result[i] = raw_score.clamp(0.0, 100.0);
        }

        result
    }

    /// Returns volatility regime classification
    pub fn classify_regime(&self, score: f64) -> &'static str {
        match score {
            s if s >= 75.0 => "Extreme Volatility",
            s if s >= 50.0 => "High Volatility",
            s if s >= 25.0 => "Normal Volatility",
            _ => "Low Volatility",
        }
    }

    /// Returns true if in a high or extreme volatility regime
    pub fn is_high_volatility(&self, score: f64) -> bool {
        score >= 50.0
    }

    /// Returns suggested position size multiplier based on regime
    pub fn position_size_multiplier(&self, score: f64) -> f64 {
        match score {
            s if s >= 75.0 => 0.25,
            s if s >= 50.0 => 0.5,
            s if s >= 25.0 => 1.0,
            _ => 1.25,
        }
    }
}

impl TechnicalIndicator for VolatilityRegimeIndex {
    fn name(&self) -> &str {
        "Volatility Regime Index"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.high, &data.low)))
    }
}

// ============================================================================
// CryptoMomentumRank - Momentum ranking for cryptocurrency
// ============================================================================

/// Crypto Momentum Rank
///
/// Ranks momentum across multiple timeframes specifically designed for
/// cryptocurrency markets. Combines short, medium, and long-term momentum
/// with volume confirmation and trend quality assessment.
///
/// Output ranges from -100 (extreme bearish) to +100 (extreme bullish):
/// - +80 to +100: Extreme bullish momentum
/// - +40 to +80: Strong bullish momentum
/// - +10 to +40: Moderate bullish momentum
/// - -10 to +10: Neutral momentum
/// - -40 to -10: Moderate bearish momentum
/// - -80 to -40: Strong bearish momentum
/// - -100 to -80: Extreme bearish momentum
#[derive(Debug, Clone)]
pub struct CryptoMomentumRank {
    /// Short-term momentum period
    short_period: usize,
    /// Medium-term momentum period
    medium_period: usize,
    /// Long-term momentum period
    long_period: usize,
    /// Volume weight in calculation
    volume_weight: f64,
}

impl CryptoMomentumRank {
    /// Creates a new CryptoMomentumRank indicator
    ///
    /// # Arguments
    /// * `short_period` - Short-term lookback (minimum 3)
    /// * `medium_period` - Medium-term lookback (must be > short_period)
    /// * `long_period` - Long-term lookback (must be > medium_period)
    /// * `volume_weight` - Weight for volume confirmation (0.0 to 1.0)
    pub fn new(short_period: usize, medium_period: usize, long_period: usize, volume_weight: f64) -> Result<Self> {
        if short_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if medium_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if long_period <= medium_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than medium_period".to_string(),
            });
        }
        if volume_weight < 0.0 || volume_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self { short_period, medium_period, long_period, volume_weight })
    }

    /// Calculate crypto momentum rank (-100 to +100)
    ///
    /// Combines rate of change across multiple timeframes with volume
    /// momentum and price strength relative to recent range.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let short_start = i.saturating_sub(self.short_period);
            let medium_start = i.saturating_sub(self.medium_period);
            let long_start = i.saturating_sub(self.long_period);

            // 1. Multi-timeframe Rate of Change
            let short_roc = if close[short_start] > 1e-10 {
                (close[i] / close[short_start] - 1.0) * 100.0
            } else { 0.0 };

            let medium_roc = if close[medium_start] > 1e-10 {
                (close[i] / close[medium_start] - 1.0) * 100.0
            } else { 0.0 };

            let long_roc = if close[long_start] > 1e-10 {
                (close[i] / close[long_start] - 1.0) * 100.0
            } else { 0.0 };

            // 2. Momentum Acceleration
            let mid_point = (short_start + i) / 2;
            let first_half_roc = if close[short_start] > 1e-10 && mid_point > short_start {
                (close[mid_point] / close[short_start] - 1.0) * 100.0
            } else { 0.0 };

            let second_half_roc = if close[mid_point] > 1e-10 && mid_point < i {
                (close[i] / close[mid_point] - 1.0) * 100.0
            } else { 0.0 };

            let acceleration = second_half_roc - first_half_roc;

            // 3. Volume Momentum Confirmation
            let early_vol: f64 = volume[long_start..(long_start + self.long_period / 2).min(i)].iter().sum();
            let late_vol: f64 = volume[(i - self.long_period / 2)..=i].iter().sum();
            let vol_momentum = if early_vol > 1e-10 {
                (late_vol / early_vol - 1.0)
            } else { 0.0 };

            // Volume confirms direction
            let vol_confirm = if (short_roc > 0.0 && vol_momentum > 0.0) || (short_roc < 0.0 && vol_momentum < 0.0) {
                1.0 + vol_momentum.abs().min(0.5) // Boost up to 50%
            } else {
                1.0 - vol_momentum.abs().min(0.3) // Reduce up to 30%
            };

            // 4. Price Position in Range
            let period_high = close[long_start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = close[long_start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;
            let position = if range > 1e-10 {
                (close[i] - period_low) / range * 2.0 - 1.0 // -1 to +1
            } else { 0.0 };

            // 5. Composite Momentum Rank
            let price_momentum = (self.volume_weight * 0.0 + (1.0 - self.volume_weight) * 1.0) * (
                short_roc * 0.4 + medium_roc * 0.35 + long_roc * 0.25
            );

            let base_rank = price_momentum + acceleration * 0.2 + position * 10.0;
            let volume_adjusted = base_rank * vol_confirm * self.volume_weight + base_rank * (1.0 - self.volume_weight);

            result[i] = volume_adjusted.clamp(-100.0, 100.0);
        }

        // Apply light smoothing
        for i in 1..n {
            result[i] = 0.8 * result[i] + 0.2 * result[i - 1];
        }

        result
    }

    /// Returns momentum rank classification
    pub fn classify_momentum(&self, rank: f64) -> &'static str {
        match rank {
            r if r >= 80.0 => "Extreme Bullish",
            r if r >= 40.0 => "Strong Bullish",
            r if r >= 10.0 => "Moderate Bullish",
            r if r >= -10.0 => "Neutral",
            r if r >= -40.0 => "Moderate Bearish",
            r if r >= -80.0 => "Strong Bearish",
            _ => "Extreme Bearish",
        }
    }

    /// Returns true if momentum suggests bullish bias
    pub fn is_bullish(&self, rank: f64) -> bool {
        rank > 10.0
    }

    /// Returns true if momentum suggests bearish bias
    pub fn is_bearish(&self, rank: f64) -> bool {
        rank < -10.0
    }
}

impl TechnicalIndicator for CryptoMomentumRank {
    fn name(&self) -> &str {
        "Crypto Momentum Rank"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// MarketSentimentProxy - Proxy for crypto market sentiment
// ============================================================================

/// Market Sentiment Proxy
///
/// Estimates overall crypto market sentiment using price action, volume patterns,
/// and market behavior proxies. This indicator attempts to capture the collective
/// emotional state of market participants without requiring external sentiment data.
///
/// Output ranges from -100 (extreme fear) to +100 (extreme greed):
/// - +70 to +100: Extreme Greed
/// - +30 to +70: Greed
/// - -30 to +30: Neutral
/// - -70 to -30: Fear
/// - -100 to -70: Extreme Fear
#[derive(Debug, Clone)]
pub struct MarketSentimentProxy {
    /// Main calculation period
    period: usize,
    /// Longer baseline period for comparison
    baseline_period: usize,
    /// Smoothing factor
    smooth_period: usize,
}

impl MarketSentimentProxy {
    /// Creates a new MarketSentimentProxy indicator
    ///
    /// # Arguments
    /// * `period` - Main calculation period (minimum 5)
    /// * `baseline_period` - Longer baseline for comparison (must be > period)
    /// * `smooth_period` - Smoothing period (minimum 2)
    pub fn new(period: usize, baseline_period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if baseline_period <= period {
            return Err(IndicatorError::InvalidParameter {
                name: "baseline_period".to_string(),
                reason: "must be greater than period".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, baseline_period, smooth_period })
    }

    /// Calculate market sentiment proxy (-100 to +100)
    ///
    /// Combines momentum chasing behavior, volume extremes, price position,
    /// and volatility patterns to estimate market sentiment.
    pub fn calculate(&self, close: &[f64], high: &[f64], low: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.baseline_period..n {
            let period_start = i.saturating_sub(self.period);
            let baseline_start = i.saturating_sub(self.baseline_period);

            // 1. Momentum Component (greed follows gains, fear follows losses)
            let price_return = if close[period_start] > 1e-10 {
                (close[i] / close[period_start] - 1.0) * 100.0
            } else { 0.0 };
            let momentum_sentiment = (price_return * 3.0).clamp(-40.0, 40.0);

            // 2. Volume Behavior (high volume in uptrends = greed, in downtrends = fear)
            let avg_vol: f64 = volume[baseline_start..period_start].iter().sum::<f64>()
                / (period_start - baseline_start).max(1) as f64;
            let recent_vol: f64 = volume[period_start..=i].iter().sum::<f64>()
                / (i - period_start + 1) as f64;
            let vol_ratio = if avg_vol > 1e-10 { recent_vol / avg_vol } else { 1.0 };

            let volume_sentiment = if price_return > 0.0 {
                (vol_ratio - 1.0).clamp(-0.5, 1.0) * 20.0 // High vol + up = greed
            } else {
                -(vol_ratio - 1.0).clamp(-0.5, 1.0) * 20.0 // High vol + down = fear
            };

            // 3. Price Position in Range (buying highs = greed, selling lows = fear)
            let baseline_high = close[baseline_start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let baseline_low = close[baseline_start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = baseline_high - baseline_low;
            let position = if range > 1e-10 {
                (close[i] - baseline_low) / range
            } else { 0.5 };
            let position_sentiment = (position * 2.0 - 1.0) * 15.0; // High price = greed

            // 4. Volatility Pattern (contracting = complacency, expanding = fear/greed)
            let mut recent_atr = 0.0;
            for j in (period_start + 1)..=i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                recent_atr += tr;
            }
            recent_atr /= (i - period_start) as f64;

            let mut baseline_atr = 0.0;
            for j in (baseline_start + 1)..period_start {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                baseline_atr += tr;
            }
            baseline_atr /= (period_start - baseline_start - 1).max(1) as f64;

            let vol_change = if baseline_atr > 1e-10 {
                recent_atr / baseline_atr - 1.0
            } else { 0.0 };

            // Expanding volatility in uptrend = euphoria, in downtrend = panic
            let volatility_sentiment = if price_return > 0.0 {
                vol_change.clamp(-0.5, 1.0) * 10.0
            } else {
                -vol_change.clamp(-0.5, 1.0) * 10.0
            };

            // 5. Consecutive Move Analysis (streaks indicate sentiment extremes)
            let mut streak = 0;
            let mut streak_direction = 0;
            for j in (period_start + 1..=i).rev() {
                if close[j] > close[j - 1] {
                    if streak_direction >= 0 {
                        streak += 1;
                        streak_direction = 1;
                    } else { break; }
                } else if close[j] < close[j - 1] {
                    if streak_direction <= 0 {
                        streak += 1;
                        streak_direction = -1;
                    } else { break; }
                }
            }
            let streak_sentiment = streak_direction as f64 * (streak as f64 / 5.0).min(1.0) * 15.0;

            // Composite sentiment
            let raw_sentiment = momentum_sentiment + volume_sentiment + position_sentiment
                + volatility_sentiment + streak_sentiment;
            result[i] = raw_sentiment.clamp(-100.0, 100.0);
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Returns sentiment classification
    pub fn classify_sentiment(&self, score: f64) -> &'static str {
        match score {
            s if s >= 70.0 => "Extreme Greed",
            s if s >= 30.0 => "Greed",
            s if s >= -30.0 => "Neutral",
            s if s >= -70.0 => "Fear",
            _ => "Extreme Fear",
        }
    }

    /// Returns true if sentiment is at a contrarian extreme
    pub fn is_contrarian_signal(&self, score: f64) -> bool {
        score >= 70.0 || score <= -70.0
    }

    /// Returns suggested action based on contrarian view
    pub fn contrarian_action(&self, score: f64) -> &'static str {
        match score {
            s if s >= 70.0 => "Consider Taking Profits",
            s if s <= -70.0 => "Consider Accumulating",
            _ => "No Clear Contrarian Signal",
        }
    }
}

impl TechnicalIndicator for MarketSentimentProxy {
    fn name(&self) -> &str {
        "Market Sentiment Proxy"
    }

    fn min_periods(&self) -> usize {
        self.baseline_period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(
            &data.close,
            &data.high,
            &data.low,
            &data.volume,
        )))
    }
}

// ============================================================================
// CryptoCyclePhase - Identifies crypto market cycle phase
// ============================================================================

/// Crypto Cycle Phase Indicator
///
/// Identifies the current phase of the crypto market cycle based on price
/// trends, momentum characteristics, and volume patterns. Crypto markets
/// tend to move in cyclical patterns with distinct phases.
///
/// Phases (encoded as values):
/// - Accumulation (10-30): After downtrend, before uptrend
/// - Markup (30-60): Bull market, uptrend
/// - Distribution (60-80): After uptrend, before downtrend
/// - Markdown (0-10 or 80-100): Bear market, downtrend
#[derive(Debug, Clone)]
pub struct CryptoCyclePhase {
    /// Short-term trend period
    short_period: usize,
    /// Medium-term trend period
    medium_period: usize,
    /// Long-term trend period
    long_period: usize,
}

impl CryptoCyclePhase {
    /// Creates a new CryptoCyclePhase indicator
    ///
    /// # Arguments
    /// * `short_period` - Short-term trend window (minimum 5)
    /// * `medium_period` - Medium-term trend window (must be > short_period)
    /// * `long_period` - Long-term trend window (must be > medium_period)
    pub fn new(short_period: usize, medium_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if medium_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if long_period <= medium_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than medium_period".to_string(),
            });
        }
        Ok(Self { short_period, medium_period, long_period })
    }

    /// Calculate crypto cycle phase (0 to 100)
    ///
    /// The value encodes the market cycle phase with overlapping ranges.
    /// Use classify_phase() to get the textual interpretation.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let short_start = i.saturating_sub(self.short_period);
            let medium_start = i.saturating_sub(self.medium_period);
            let long_start = i.saturating_sub(self.long_period);

            // 1. Multi-timeframe trend analysis
            let short_trend = if close[short_start] > 1e-10 {
                (close[i] / close[short_start] - 1.0) * 100.0
            } else { 0.0 };

            let medium_trend = if close[medium_start] > 1e-10 {
                (close[i] / close[medium_start] - 1.0) * 100.0
            } else { 0.0 };

            let long_trend = if close[long_start] > 1e-10 {
                (close[i] / close[long_start] - 1.0) * 100.0
            } else { 0.0 };

            // 2. Moving average alignment
            let short_ma: f64 = close[(i - self.short_period + 1)..=i].iter().sum::<f64>()
                / self.short_period as f64;
            let medium_ma: f64 = close[(i - self.medium_period + 1)..=i].iter().sum::<f64>()
                / self.medium_period as f64;
            let long_ma: f64 = close[(i - self.long_period + 1)..=i].iter().sum::<f64>()
                / self.long_period as f64;

            // MA alignment score
            let bullish_alignment = if close[i] > short_ma && short_ma > medium_ma && medium_ma > long_ma {
                1.0
            } else if close[i] > short_ma && short_ma > medium_ma {
                0.7
            } else if close[i] > short_ma {
                0.4
            } else {
                0.0
            };

            let bearish_alignment = if close[i] < short_ma && short_ma < medium_ma && medium_ma < long_ma {
                1.0
            } else if close[i] < short_ma && short_ma < medium_ma {
                0.7
            } else if close[i] < short_ma {
                0.4
            } else {
                0.0
            };

            // 3. Volume trend analysis
            let long_avg_vol: f64 = volume[long_start..medium_start].iter().sum::<f64>()
                / (medium_start - long_start).max(1) as f64;
            let recent_avg_vol: f64 = volume[short_start..=i].iter().sum::<f64>()
                / (i - short_start + 1) as f64;
            let vol_trend = if long_avg_vol > 1e-10 {
                recent_avg_vol / long_avg_vol
            } else { 1.0 };

            // 4. Price position in long-term range
            let lt_high = close[long_start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let lt_low = close[long_start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let lt_range = lt_high - lt_low;
            let price_position = if lt_range > 1e-10 {
                (close[i] - lt_low) / lt_range
            } else { 0.5 };

            // 5. Trend momentum (acceleration/deceleration)
            let recent_momentum = short_trend - medium_trend;
            let trend_momentum = if medium_trend > 0.0 {
                if recent_momentum > 0.0 { 1.0 } else { -0.5 } // Accelerating/decelerating up
            } else {
                if recent_momentum < 0.0 { -1.0 } else { 0.5 } // Accelerating/decelerating down
            };

            // Phase determination logic
            let mut phase_score;

            if long_trend < -10.0 && bearish_alignment > 0.5 {
                // Markdown phase
                if short_trend > medium_trend && vol_trend < 1.0 {
                    // Potential accumulation starting
                    phase_score = 15.0 + price_position * 10.0;
                } else {
                    phase_score = 5.0 + (1.0 - bearish_alignment) * 5.0;
                }
            } else if long_trend > 10.0 && bullish_alignment > 0.5 {
                // Markup or distribution
                if short_trend < medium_trend && vol_trend > 1.2 {
                    // Potential distribution
                    phase_score = 65.0 + price_position * 15.0;
                } else {
                    // Markup
                    phase_score = 35.0 + bullish_alignment * 20.0 + (1.0 - vol_trend.min(2.0) / 2.0) * 5.0;
                }
            } else if price_position < 0.3 && short_trend > 0.0 {
                // Accumulation
                phase_score = 15.0 + short_trend.min(20.0) + (1.0 - vol_trend.min(1.5)) * 10.0;
            } else if price_position > 0.7 && short_trend < 0.0 {
                // Distribution
                phase_score = 65.0 - short_trend.min(20.0).abs() + vol_trend.min(2.0) * 5.0;
            } else {
                // Transitional
                phase_score = 40.0 + trend_momentum * 10.0 + (price_position - 0.5) * 20.0;
            }

            result[i] = phase_score.clamp(0.0, 100.0);
        }

        // Light smoothing to reduce noise
        for i in 1..n {
            result[i] = 0.7 * result[i] + 0.3 * result[i - 1];
        }

        result
    }

    /// Returns cycle phase classification
    pub fn classify_phase(&self, score: f64) -> &'static str {
        match score {
            s if s < 15.0 => "Markdown (Bear)",
            s if s < 35.0 => "Accumulation",
            s if s < 65.0 => "Markup (Bull)",
            s if s < 85.0 => "Distribution",
            _ => "Late Distribution/Early Markdown",
        }
    }

    /// Returns suggested strategy for current phase
    pub fn phase_strategy(&self, score: f64) -> &'static str {
        match score {
            s if s < 15.0 => "Defensive, Wait for Accumulation Signs",
            s if s < 35.0 => "Begin Accumulation, Build Positions",
            s if s < 65.0 => "Hold Long Positions, Trail Stops",
            s if s < 85.0 => "Take Profits, Reduce Exposure",
            _ => "Maximum Caution, Consider Hedging",
        }
    }

    /// Returns true if in favorable accumulation phase
    pub fn is_accumulation(&self, score: f64) -> bool {
        score >= 15.0 && score < 35.0
    }

    /// Returns true if in distribution phase
    pub fn is_distribution(&self, score: f64) -> bool {
        score >= 65.0 && score < 85.0
    }
}

impl TechnicalIndicator for CryptoCyclePhase {
    fn name(&self) -> &str {
        "Crypto Cycle Phase"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

// ============================================================================
// AdaptiveCryptoMA - Moving average adapted for crypto volatility
// ============================================================================

/// Adaptive Crypto Moving Average
///
/// A moving average that automatically adjusts its smoothing factor based on
/// cryptocurrency market volatility. During high volatility periods, it becomes
/// more responsive; during low volatility, it becomes smoother to filter noise.
///
/// This is particularly useful for crypto markets which experience
/// rapid regime changes between trending and ranging conditions.
#[derive(Debug, Clone)]
pub struct AdaptiveCryptoMA {
    /// Base period for the moving average
    period: usize,
    /// Fast period for high volatility
    fast_period: usize,
    /// Slow period for low volatility
    slow_period: usize,
    /// Volatility lookback for adaptation
    volatility_period: usize,
}

impl AdaptiveCryptoMA {
    /// Creates a new AdaptiveCryptoMA indicator
    ///
    /// # Arguments
    /// * `period` - Base calculation period (minimum 5)
    /// * `fast_period` - Fastest smoothing period (minimum 2)
    /// * `slow_period` - Slowest smoothing period (must be > fast_period)
    /// * `volatility_period` - Period for volatility calculation (minimum 5)
    pub fn new(period: usize, fast_period: usize, slow_period: usize, volatility_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
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
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period, fast_period, slow_period, volatility_period })
    }

    /// Calculate the adaptive crypto moving average
    ///
    /// Returns the adaptive MA values. The smoothing constant automatically
    /// adjusts based on current volatility relative to historical volatility.
    pub fn calculate(&self, close: &[f64], high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        if n == 0 {
            return result;
        }

        // Initialize with simple average
        let init_end = self.period.min(n);
        let init_sum: f64 = close[0..init_end].iter().sum();
        result[init_end - 1] = init_sum / init_end as f64;

        // Calculate historical volatility baseline
        let max_lookback = self.period.max(self.volatility_period);

        for i in max_lookback..n {
            let vol_start = i.saturating_sub(self.volatility_period);

            // 1. Current volatility (ATR-based)
            let mut atr_sum = 0.0;
            for j in (vol_start + 1)..=i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                atr_sum += tr;
            }
            let current_atr = atr_sum / self.volatility_period as f64;

            // 2. Recent price range volatility
            let period_high = close[vol_start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = close[vol_start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let avg_price = close[vol_start..=i].iter().sum::<f64>() / (i - vol_start + 1) as f64;
            let range_volatility = if avg_price > 1e-10 {
                (period_high - period_low) / avg_price
            } else { 0.0 };

            // 3. Return volatility (standard deviation of returns)
            let mut returns = Vec::with_capacity(self.volatility_period);
            for j in (vol_start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push((close[j] / close[j - 1] - 1.0) * 100.0);
                }
            }
            let return_mean: f64 = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
            let return_variance: f64 = returns.iter()
                .map(|r| (r - return_mean).powi(2))
                .sum::<f64>() / returns.len().max(1) as f64;
            let return_vol = return_variance.sqrt();

            // 4. Efficiency Ratio (directional movement / total movement)
            let direction = (close[i] - close[vol_start]).abs();
            let mut total_movement = 0.0;
            for j in (vol_start + 1)..=i {
                total_movement += (close[j] - close[j - 1]).abs();
            }
            let efficiency_ratio = if total_movement > 1e-10 {
                direction / total_movement
            } else { 0.0 };

            // 5. Combined volatility score (0 = low vol, 1 = high vol)
            let atr_pct = if avg_price > 1e-10 { current_atr / avg_price } else { 0.0 };
            let vol_score = (
                (atr_pct * 20.0).min(1.0) * 0.3 +     // ATR component
                (range_volatility * 5.0).min(1.0) * 0.3 + // Range component
                (return_vol / 5.0).min(1.0) * 0.2 +    // Return vol component
                (1.0 - efficiency_ratio) * 0.2         // Noise component
            ).clamp(0.0, 1.0);

            // 6. Calculate adaptive smoothing constant
            // High volatility -> use fast period, Low volatility -> use slow period
            let fast_alpha = 2.0 / (self.fast_period as f64 + 1.0);
            let slow_alpha = 2.0 / (self.slow_period as f64 + 1.0);

            // Adaptive alpha: interpolate between slow and fast based on volatility
            // Also factor in efficiency ratio (trending = faster response)
            let vol_adjusted_alpha = slow_alpha + vol_score * (fast_alpha - slow_alpha);
            let efficiency_boost = efficiency_ratio * (fast_alpha - vol_adjusted_alpha) * 0.5;
            let adaptive_alpha = (vol_adjusted_alpha + efficiency_boost).clamp(slow_alpha, fast_alpha);

            // 7. Calculate adaptive MA
            let prev_ma = if i > 0 && result[i - 1] > 1e-10 {
                result[i - 1]
            } else {
                close[i]
            };

            result[i] = adaptive_alpha * close[i] + (1.0 - adaptive_alpha) * prev_ma;
        }

        // Fill in early values with simple average
        for i in 0..max_lookback {
            let start = 0;
            let end = i + 1;
            result[i] = close[start..end].iter().sum::<f64>() / end as f64;
        }

        result
    }

    /// Calculate the current adaptation factor (0 = slow, 1 = fast)
    pub fn get_adaptation_factor(&self, close: &[f64], high: &[f64], low: &[f64], index: usize) -> f64 {
        if index < self.volatility_period {
            return 0.5;
        }

        let vol_start = index.saturating_sub(self.volatility_period);

        // ATR-based volatility
        let mut atr_sum = 0.0;
        for j in (vol_start + 1)..=index {
            let tr = (high[j] - low[j])
                .max((high[j] - close[j - 1]).abs())
                .max((low[j] - close[j - 1]).abs());
            atr_sum += tr;
        }
        let avg_price = close[vol_start..=index].iter().sum::<f64>()
            / (index - vol_start + 1) as f64;
        let atr_pct = if avg_price > 1e-10 {
            atr_sum / self.volatility_period as f64 / avg_price
        } else { 0.0 };

        (atr_pct * 20.0).clamp(0.0, 1.0)
    }

    /// Returns the current regime description
    pub fn describe_regime(&self, adaptation_factor: f64) -> &'static str {
        match adaptation_factor {
            f if f >= 0.75 => "High Volatility - Fast Response",
            f if f >= 0.5 => "Moderate Volatility - Balanced",
            f if f >= 0.25 => "Low Volatility - Smooth Response",
            _ => "Very Low Volatility - Maximum Smoothing",
        }
    }
}

impl TechnicalIndicator for AdaptiveCryptoMA {
    fn name(&self) -> &str {
        "Adaptive Crypto MA"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.volatility_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.high, &data.low)))
    }
}

// ============================================================================
// Unit Tests for Six New Crypto Indicators
// ============================================================================

#[cfg(test)]
mod crypto_indicator_tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Generate realistic OHLCV data with trends and reversals
        let mut close = Vec::with_capacity(60);
        let mut high = Vec::with_capacity(60);
        let mut low = Vec::with_capacity(60);
        let mut volume = Vec::with_capacity(60);

        let mut price = 100.0;
        for i in 0..60 {
            // Add trend and noise
            let trend = if i < 20 { 0.8 } else if i < 40 { -0.5 } else { 0.6 };
            let noise = ((i as f64 * 0.7).sin() * 2.0);
            price += trend + noise;
            price = price.max(50.0);

            let volatility = 1.5 + ((i as f64 * 0.3).sin().abs() * 2.0);
            close.push(price);
            high.push(price + volatility);
            low.push(price - volatility);

            // Volume correlates with price movement magnitude
            let base_vol = 1000.0 + (i as f64 * 20.0);
            let vol_spike = if i % 7 == 0 { 2.5 } else { 1.0 };
            volume.push(base_vol * vol_spike);
        }

        (close, high, low, volume)
    }

    fn make_ohlcv_series() -> OHLCVSeries {
        let (close, high, low, volume) = make_test_data();
        let open = close.clone();

        OHLCVSeries { open, high, low, close, volume }
    }

    // ========== CryptoTrendStrength Tests ==========

    #[test]
    fn test_crypto_trend_strength_new() {
        assert!(CryptoTrendStrength::new(14, 5, 1.0).is_ok());
        assert!(CryptoTrendStrength::new(4, 5, 1.0).is_err()); // period too small
        assert!(CryptoTrendStrength::new(14, 1, 1.0).is_err()); // smooth_period too small
        assert!(CryptoTrendStrength::new(14, 5, 0.05).is_err()); // volatility_factor too small
        assert!(CryptoTrendStrength::new(14, 5, 3.5).is_err()); // volatility_factor too large
    }

    #[test]
    fn test_crypto_trend_strength_calculate() {
        let (close, high, low, volume) = make_test_data();
        let cts = CryptoTrendStrength::new(14, 5, 1.0).unwrap();
        let result = cts.calculate(&close, &high, &low, &volume);

        assert_eq!(result.len(), close.len());
        // Values should be bounded 0-100
        for i in 20..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_crypto_trend_strength_classify() {
        let cts = CryptoTrendStrength::new(14, 5, 1.0).unwrap();
        assert_eq!(cts.classify_strength(85.0), "Very Strong (Parabolic)");
        assert_eq!(cts.classify_strength(70.0), "Strong");
        assert_eq!(cts.classify_strength(50.0), "Moderate");
        assert_eq!(cts.classify_strength(30.0), "Weak");
        assert_eq!(cts.classify_strength(10.0), "No Trend");
    }

    #[test]
    fn test_crypto_trend_strength_is_trending() {
        let cts = CryptoTrendStrength::new(14, 5, 1.0).unwrap();
        assert!(!cts.is_trending(30.0));
        assert!(cts.is_trending(50.0));
    }

    #[test]
    fn test_crypto_trend_strength_trait() {
        let data = make_ohlcv_series();
        let cts = CryptoTrendStrength::new(14, 5, 1.0).unwrap();

        assert_eq!(cts.name(), "Crypto Trend Strength");
        assert_eq!(cts.min_periods(), 19);

        let output = cts.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== VolatilityRegimeIndex Tests ==========

    #[test]
    fn test_volatility_regime_index_new() {
        assert!(VolatilityRegimeIndex::new(7, 21, 1.0).is_ok());
        assert!(VolatilityRegimeIndex::new(3, 21, 1.0).is_err()); // short_period too small
        assert!(VolatilityRegimeIndex::new(21, 21, 1.0).is_err()); // long_period not > short
        assert!(VolatilityRegimeIndex::new(7, 21, 0.3).is_err()); // sensitivity too low
        assert!(VolatilityRegimeIndex::new(7, 21, 2.5).is_err()); // sensitivity too high
    }

    #[test]
    fn test_volatility_regime_index_calculate() {
        let (close, high, low, _) = make_test_data();
        let vri = VolatilityRegimeIndex::new(7, 21, 1.0).unwrap();
        let result = vri.calculate(&close, &high, &low);

        assert_eq!(result.len(), close.len());
        // Values should be bounded 0-100
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_volatility_regime_index_classify() {
        let vri = VolatilityRegimeIndex::new(7, 21, 1.0).unwrap();
        assert_eq!(vri.classify_regime(80.0), "Extreme Volatility");
        assert_eq!(vri.classify_regime(60.0), "High Volatility");
        assert_eq!(vri.classify_regime(35.0), "Normal Volatility");
        assert_eq!(vri.classify_regime(15.0), "Low Volatility");
    }

    #[test]
    fn test_volatility_regime_index_helpers() {
        let vri = VolatilityRegimeIndex::new(7, 21, 1.0).unwrap();
        assert!(!vri.is_high_volatility(40.0));
        assert!(vri.is_high_volatility(60.0));

        assert_eq!(vri.position_size_multiplier(80.0), 0.25);
        assert_eq!(vri.position_size_multiplier(60.0), 0.5);
        assert_eq!(vri.position_size_multiplier(35.0), 1.0);
        assert_eq!(vri.position_size_multiplier(15.0), 1.25);
    }

    #[test]
    fn test_volatility_regime_index_trait() {
        let data = make_ohlcv_series();
        let vri = VolatilityRegimeIndex::new(7, 21, 1.0).unwrap();

        assert_eq!(vri.name(), "Volatility Regime Index");
        assert_eq!(vri.min_periods(), 22);

        let output = vri.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== CryptoMomentumRank Tests ==========

    #[test]
    fn test_crypto_momentum_rank_new() {
        assert!(CryptoMomentumRank::new(5, 14, 30, 0.5).is_ok());
        assert!(CryptoMomentumRank::new(2, 14, 30, 0.5).is_err()); // short_period too small
        assert!(CryptoMomentumRank::new(5, 5, 30, 0.5).is_err()); // medium_period not > short
        assert!(CryptoMomentumRank::new(5, 14, 14, 0.5).is_err()); // long_period not > medium
        assert!(CryptoMomentumRank::new(5, 14, 30, -0.1).is_err()); // volume_weight negative
        assert!(CryptoMomentumRank::new(5, 14, 30, 1.5).is_err()); // volume_weight > 1
    }

    #[test]
    fn test_crypto_momentum_rank_calculate() {
        let (close, _, _, volume) = make_test_data();
        let cmr = CryptoMomentumRank::new(5, 14, 30, 0.5).unwrap();
        let result = cmr.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Values should be bounded -100 to 100
        for i in 35..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_crypto_momentum_rank_classify() {
        let cmr = CryptoMomentumRank::new(5, 14, 30, 0.5).unwrap();
        assert_eq!(cmr.classify_momentum(85.0), "Extreme Bullish");
        assert_eq!(cmr.classify_momentum(60.0), "Strong Bullish");
        assert_eq!(cmr.classify_momentum(25.0), "Moderate Bullish");
        assert_eq!(cmr.classify_momentum(0.0), "Neutral");
        assert_eq!(cmr.classify_momentum(-25.0), "Moderate Bearish");
        assert_eq!(cmr.classify_momentum(-60.0), "Strong Bearish");
        assert_eq!(cmr.classify_momentum(-85.0), "Extreme Bearish");
    }

    #[test]
    fn test_crypto_momentum_rank_direction() {
        let cmr = CryptoMomentumRank::new(5, 14, 30, 0.5).unwrap();
        assert!(!cmr.is_bullish(5.0));
        assert!(cmr.is_bullish(20.0));
        assert!(!cmr.is_bearish(-5.0));
        assert!(cmr.is_bearish(-20.0));
    }

    #[test]
    fn test_crypto_momentum_rank_trait() {
        let data = make_ohlcv_series();
        let cmr = CryptoMomentumRank::new(5, 14, 30, 0.5).unwrap();

        assert_eq!(cmr.name(), "Crypto Momentum Rank");
        assert_eq!(cmr.min_periods(), 31);

        let output = cmr.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== MarketSentimentProxy Tests ==========

    #[test]
    fn test_market_sentiment_proxy_new() {
        assert!(MarketSentimentProxy::new(7, 21, 5).is_ok());
        assert!(MarketSentimentProxy::new(3, 21, 5).is_err()); // period too small
        assert!(MarketSentimentProxy::new(7, 7, 5).is_err()); // baseline not > period
        assert!(MarketSentimentProxy::new(7, 21, 1).is_err()); // smooth_period too small
    }

    #[test]
    fn test_market_sentiment_proxy_calculate() {
        let (close, high, low, volume) = make_test_data();
        let msp = MarketSentimentProxy::new(7, 21, 5).unwrap();
        let result = msp.calculate(&close, &high, &low, &volume);

        assert_eq!(result.len(), close.len());
        // Values should be bounded -100 to 100
        for i in 30..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_market_sentiment_proxy_classify() {
        let msp = MarketSentimentProxy::new(7, 21, 5).unwrap();
        assert_eq!(msp.classify_sentiment(80.0), "Extreme Greed");
        assert_eq!(msp.classify_sentiment(50.0), "Greed");
        assert_eq!(msp.classify_sentiment(0.0), "Neutral");
        assert_eq!(msp.classify_sentiment(-50.0), "Fear");
        assert_eq!(msp.classify_sentiment(-80.0), "Extreme Fear");
    }

    #[test]
    fn test_market_sentiment_proxy_contrarian() {
        let msp = MarketSentimentProxy::new(7, 21, 5).unwrap();
        assert!(!msp.is_contrarian_signal(50.0));
        assert!(msp.is_contrarian_signal(75.0));
        assert!(msp.is_contrarian_signal(-75.0));

        assert_eq!(msp.contrarian_action(80.0), "Consider Taking Profits");
        assert_eq!(msp.contrarian_action(-80.0), "Consider Accumulating");
        assert_eq!(msp.contrarian_action(0.0), "No Clear Contrarian Signal");
    }

    #[test]
    fn test_market_sentiment_proxy_trait() {
        let data = make_ohlcv_series();
        let msp = MarketSentimentProxy::new(7, 21, 5).unwrap();

        assert_eq!(msp.name(), "Market Sentiment Proxy");
        assert_eq!(msp.min_periods(), 26);

        let output = msp.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== CryptoCyclePhase Tests ==========

    #[test]
    fn test_crypto_cycle_phase_new() {
        assert!(CryptoCyclePhase::new(7, 21, 50).is_ok());
        assert!(CryptoCyclePhase::new(3, 21, 50).is_err()); // short_period too small
        assert!(CryptoCyclePhase::new(7, 7, 50).is_err()); // medium_period not > short
        assert!(CryptoCyclePhase::new(7, 21, 21).is_err()); // long_period not > medium
    }

    #[test]
    fn test_crypto_cycle_phase_calculate() {
        let (close, _, _, volume) = make_test_data();
        let ccp = CryptoCyclePhase::new(7, 21, 50).unwrap();
        let result = ccp.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Values should be bounded 0-100
        for i in 55..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_crypto_cycle_phase_classify() {
        let ccp = CryptoCyclePhase::new(7, 21, 50).unwrap();
        assert_eq!(ccp.classify_phase(10.0), "Markdown (Bear)");
        assert_eq!(ccp.classify_phase(25.0), "Accumulation");
        assert_eq!(ccp.classify_phase(50.0), "Markup (Bull)");
        assert_eq!(ccp.classify_phase(75.0), "Distribution");
        assert_eq!(ccp.classify_phase(90.0), "Late Distribution/Early Markdown");
    }

    #[test]
    fn test_crypto_cycle_phase_strategy() {
        let ccp = CryptoCyclePhase::new(7, 21, 50).unwrap();
        assert_eq!(ccp.phase_strategy(10.0), "Defensive, Wait for Accumulation Signs");
        assert_eq!(ccp.phase_strategy(25.0), "Begin Accumulation, Build Positions");
        assert_eq!(ccp.phase_strategy(50.0), "Hold Long Positions, Trail Stops");
        assert_eq!(ccp.phase_strategy(75.0), "Take Profits, Reduce Exposure");
        assert_eq!(ccp.phase_strategy(90.0), "Maximum Caution, Consider Hedging");
    }

    #[test]
    fn test_crypto_cycle_phase_helpers() {
        let ccp = CryptoCyclePhase::new(7, 21, 50).unwrap();
        assert!(!ccp.is_accumulation(10.0));
        assert!(ccp.is_accumulation(25.0));
        assert!(!ccp.is_accumulation(50.0));

        assert!(!ccp.is_distribution(50.0));
        assert!(ccp.is_distribution(70.0));
        assert!(!ccp.is_distribution(90.0));
    }

    #[test]
    fn test_crypto_cycle_phase_trait() {
        let data = make_ohlcv_series();
        let ccp = CryptoCyclePhase::new(7, 21, 50).unwrap();

        assert_eq!(ccp.name(), "Crypto Cycle Phase");
        assert_eq!(ccp.min_periods(), 51);

        let output = ccp.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== AdaptiveCryptoMA Tests ==========

    #[test]
    fn test_adaptive_crypto_ma_new() {
        assert!(AdaptiveCryptoMA::new(14, 3, 30, 10).is_ok());
        assert!(AdaptiveCryptoMA::new(3, 3, 30, 10).is_err()); // period too small
        assert!(AdaptiveCryptoMA::new(14, 1, 30, 10).is_err()); // fast_period too small
        assert!(AdaptiveCryptoMA::new(14, 3, 3, 10).is_err()); // slow_period not > fast
        assert!(AdaptiveCryptoMA::new(14, 3, 30, 3).is_err()); // volatility_period too small
    }

    #[test]
    fn test_adaptive_crypto_ma_calculate() {
        let (close, high, low, _) = make_test_data();
        let acma = AdaptiveCryptoMA::new(14, 3, 30, 10).unwrap();
        let result = acma.calculate(&close, &high, &low);

        assert_eq!(result.len(), close.len());
        // MA should be positive and follow price
        for i in 15..result.len() {
            assert!(result[i] > 0.0, "MA at {} should be positive", i);
            // MA should be reasonably close to price
            let deviation = (result[i] - close[i]).abs() / close[i];
            assert!(deviation < 0.3, "MA deviation at {} is {}", i, deviation);
        }
    }

    #[test]
    fn test_adaptive_crypto_ma_smoothness() {
        let (close, high, low, _) = make_test_data();
        let acma = AdaptiveCryptoMA::new(14, 3, 30, 10).unwrap();
        let result = acma.calculate(&close, &high, &low);

        // MA should be smoother than price
        let mut price_changes = 0.0;
        let mut ma_changes = 0.0;

        for i in 20..result.len() {
            price_changes += (close[i] - close[i - 1]).abs();
            ma_changes += (result[i] - result[i - 1]).abs();
        }

        assert!(ma_changes < price_changes, "MA should be smoother than price");
    }

    #[test]
    fn test_adaptive_crypto_ma_adaptation_factor() {
        let (close, high, low, _) = make_test_data();
        let acma = AdaptiveCryptoMA::new(14, 3, 30, 10).unwrap();

        let factor = acma.get_adaptation_factor(&close, &high, &low, 30);
        assert!(factor >= 0.0 && factor <= 1.0);
    }

    #[test]
    fn test_adaptive_crypto_ma_describe_regime() {
        let acma = AdaptiveCryptoMA::new(14, 3, 30, 10).unwrap();
        assert_eq!(acma.describe_regime(0.8), "High Volatility - Fast Response");
        assert_eq!(acma.describe_regime(0.6), "Moderate Volatility - Balanced");
        assert_eq!(acma.describe_regime(0.3), "Low Volatility - Smooth Response");
        assert_eq!(acma.describe_regime(0.1), "Very Low Volatility - Maximum Smoothing");
    }

    #[test]
    fn test_adaptive_crypto_ma_trait() {
        let data = make_ohlcv_series();
        let acma = AdaptiveCryptoMA::new(14, 3, 30, 10).unwrap();

        assert_eq!(acma.name(), "Adaptive Crypto MA");
        assert_eq!(acma.min_periods(), 14);

        let output = acma.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========== Integration and Edge Case Tests ==========

    #[test]
    fn test_all_six_indicators_with_same_data() {
        let data = make_ohlcv_series();

        let cts = CryptoTrendStrength::new(14, 5, 1.0).unwrap();
        let vri = VolatilityRegimeIndex::new(7, 21, 1.0).unwrap();
        let cmr = CryptoMomentumRank::new(5, 14, 30, 0.5).unwrap();
        let msp = MarketSentimentProxy::new(7, 21, 5).unwrap();
        let ccp = CryptoCyclePhase::new(7, 21, 50).unwrap();
        let acma = AdaptiveCryptoMA::new(14, 3, 30, 10).unwrap();

        // All should compute successfully
        assert!(cts.compute(&data).is_ok());
        assert!(vri.compute(&data).is_ok());
        assert!(cmr.compute(&data).is_ok());
        assert!(msp.compute(&data).is_ok());
        assert!(ccp.compute(&data).is_ok());
        assert!(acma.compute(&data).is_ok());
    }

    #[test]
    fn test_six_indicators_empty_data() {
        let close: Vec<f64> = vec![];
        let high: Vec<f64> = vec![];
        let low: Vec<f64> = vec![];
        let volume: Vec<f64> = vec![];

        let cts = CryptoTrendStrength::new(14, 5, 1.0).unwrap();
        assert!(cts.calculate(&close, &high, &low, &volume).is_empty());

        let vri = VolatilityRegimeIndex::new(7, 21, 1.0).unwrap();
        assert!(vri.calculate(&close, &high, &low).is_empty());

        let cmr = CryptoMomentumRank::new(5, 14, 30, 0.5).unwrap();
        assert!(cmr.calculate(&close, &volume).is_empty());

        let msp = MarketSentimentProxy::new(7, 21, 5).unwrap();
        assert!(msp.calculate(&close, &high, &low, &volume).is_empty());

        let ccp = CryptoCyclePhase::new(7, 21, 50).unwrap();
        assert!(ccp.calculate(&close, &volume).is_empty());

        let acma = AdaptiveCryptoMA::new(14, 3, 30, 10).unwrap();
        assert!(acma.calculate(&close, &high, &low).is_empty());
    }

    #[test]
    fn test_six_indicators_insufficient_data() {
        let close: Vec<f64> = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let high: Vec<f64> = vec![101.0, 102.0, 103.0, 104.0, 105.0];
        let low: Vec<f64> = vec![99.0, 100.0, 101.0, 102.0, 103.0];
        let volume: Vec<f64> = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0];

        let cts = CryptoTrendStrength::new(14, 5, 1.0).unwrap();
        let result = cts.calculate(&close, &high, &low, &volume);
        // Should return data but early values will be zero/smoothed
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_six_indicators_constant_prices() {
        let close: Vec<f64> = vec![100.0; 60];
        let high: Vec<f64> = vec![101.0; 60];
        let low: Vec<f64> = vec![99.0; 60];
        let volume: Vec<f64> = vec![1000.0; 60];

        let cts = CryptoTrendStrength::new(14, 5, 1.0).unwrap();
        let result = cts.calculate(&close, &high, &low, &volume);
        // With constant prices, trend strength should be low
        for i in 20..result.len() {
            assert!(result[i] < 50.0, "Trend strength should be low for constant prices");
        }

        let acma = AdaptiveCryptoMA::new(14, 3, 30, 10).unwrap();
        let ma_result = acma.calculate(&close, &high, &low);
        // MA should converge to constant price
        for i in 20..ma_result.len() {
            assert!((ma_result[i] - 100.0).abs() < 1.0, "MA should be close to constant price");
        }
    }

    #[test]
    fn test_six_indicators_trending_up() {
        // Strong uptrend
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let volume: Vec<f64> = (0..60).map(|i| 1000.0 + i as f64 * 50.0).collect();

        let cmr = CryptoMomentumRank::new(5, 14, 30, 0.5).unwrap();
        let result = cmr.calculate(&close, &volume);
        // In strong uptrend, momentum should be positive
        for i in 35..result.len() {
            assert!(result[i] > 0.0, "Momentum should be positive in uptrend at {}", i);
        }
    }

    #[test]
    fn test_six_indicators_trending_down() {
        // Strong downtrend
        let close: Vec<f64> = (0..60).map(|i| 200.0 - i as f64 * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let volume: Vec<f64> = (0..60).map(|i| 1000.0 + i as f64 * 50.0).collect();

        let cmr = CryptoMomentumRank::new(5, 14, 30, 0.5).unwrap();
        let result = cmr.calculate(&close, &volume);
        // In strong downtrend, momentum should be negative
        for i in 35..result.len() {
            assert!(result[i] < 0.0, "Momentum should be negative in downtrend at {}", i);
        }
    }

    #[test]
    fn test_high_volatility_detection() {
        // High volatility data
        let mut close = Vec::with_capacity(60);
        let mut high = Vec::with_capacity(60);
        let mut low = Vec::with_capacity(60);

        let mut price = 100.0;
        for i in 0..60 {
            // Large swings
            price += ((i as f64 * 0.5).sin() * 10.0);
            price = price.max(50.0);
            close.push(price);
            high.push(price + 5.0); // Large range
            low.push(price - 5.0);
        }

        let vri = VolatilityRegimeIndex::new(7, 21, 1.0).unwrap();
        let result = vri.calculate(&close, &high, &low);

        // With high volatility, regime index should be elevated
        let avg_vol: f64 = result[25..].iter().sum::<f64>() / (result.len() - 25) as f64;
        assert!(avg_vol > 20.0, "Volatility regime should detect high volatility");
    }
}
