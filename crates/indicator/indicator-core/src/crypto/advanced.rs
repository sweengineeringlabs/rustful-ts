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
