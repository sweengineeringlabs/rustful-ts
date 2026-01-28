//! Market Sentiment Indicators
//!
//! Indicators for measuring market sentiment and crowd behavior.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Fear Index - Inverse of complacency
#[derive(Debug, Clone)]
pub struct FearIndex {
    period: usize,
}

impl FearIndex {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate fear index (0-100 scale)
    /// Uses volatility and drawdown as fear proxy
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate volatility component
            let returns: Vec<f64> = ((start + 1)..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();

            if returns.is_empty() {
                continue;
            }

            let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean_ret).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let vol = variance.sqrt() * (252.0_f64).sqrt() * 100.0;

            // Calculate drawdown component
            let max_high = high[start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let drawdown = if max_high > 0.0 {
                (max_high - close[i]) / max_high * 100.0
            } else {
                0.0
            };

            // Fear index: weighted combination of vol and drawdown
            let fear = (vol * 0.6 + drawdown * 4.0).min(100.0);
            result[i] = fear;
        }
        result
    }
}

impl TechnicalIndicator for FearIndex {
    fn name(&self) -> &str {
        "Fear Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Greed Index - Inverse of fear
#[derive(Debug, Clone)]
pub struct GreedIndex {
    period: usize,
}

impl GreedIndex {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate greed index (0-100 scale)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Momentum component
            let momentum = if close[start] > 0.0 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };

            // New high proximity
            let max_high = high[start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let high_proximity = if max_high > 0.0 {
                (close[i] / max_high) * 100.0
            } else {
                0.0
            };

            // Greed: weighted combination
            let greed = ((momentum * 2.0).max(0.0) + high_proximity) / 2.0;
            result[i] = greed.min(100.0).max(0.0);
        }
        result
    }
}

impl TechnicalIndicator for GreedIndex {
    fn name(&self) -> &str {
        "Greed Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Crowd Psychology - Herd behavior indicator
#[derive(Debug, Clone)]
pub struct CrowdPsychology {
    period: usize,
}

impl CrowdPsychology {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate crowd psychology score
    /// Positive = bullish herd, Negative = bearish herd
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Count consecutive moves in same direction
            let mut streak = 0i32;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    if streak >= 0 { streak += 1; } else { streak = 1; }
                } else if close[j] < close[j - 1] {
                    if streak <= 0 { streak -= 1; } else { streak = -1; }
                }
            }

            // Volume confirmation
            let avg_vol = volume[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;
            let vol_factor = if avg_vol > 0.0 {
                (volume[i] / avg_vol).min(2.0)
            } else {
                1.0
            };

            result[i] = streak as f64 * vol_factor * 10.0;
        }
        result
    }
}

impl TechnicalIndicator for CrowdPsychology {
    fn name(&self) -> &str {
        "Crowd Psychology"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Market Euphoria - Extreme optimism detection
#[derive(Debug, Clone)]
pub struct MarketEuphoria {
    period: usize,
    threshold: f64,
}

impl MarketEuphoria {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate euphoria level (0-100)
    pub fn calculate(&self, high: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut euphoria: f64 = 0.0;

            // New highs count
            let mut new_high_count = 0;
            let mut running_max = high[start];
            for j in (start + 1)..=i {
                if high[j] > running_max {
                    new_high_count += 1;
                    running_max = high[j];
                }
            }
            euphoria += (new_high_count as f64 / self.period as f64) * 30.0;

            // Volume surge
            let avg_vol = volume[start..(i - 1).max(start + 1)].iter().sum::<f64>()
                / ((i - 1).max(start + 1) - start).max(1) as f64;
            if avg_vol > 0.0 && volume[i] > avg_vol * self.threshold {
                euphoria += 30.0;
            }

            // Momentum strength
            let momentum = (close[i] / close[start] - 1.0) * 100.0;
            euphoria += (momentum * 2.0).max(0.0).min(40.0);

            result[i] = euphoria.min(100.0);
        }
        result
    }
}

impl TechnicalIndicator for MarketEuphoria {
    fn name(&self) -> &str {
        "Market Euphoria"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.close, &data.volume)))
    }
}

/// Capitulation - Extreme pessimism detection
#[derive(Debug, Clone)]
pub struct Capitulation {
    period: usize,
    vol_threshold: f64,
}

impl Capitulation {
    pub fn new(period: usize, vol_threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period, vol_threshold })
    }

    /// Calculate capitulation level (0-100)
    pub fn calculate(&self, low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut capitulation: f64 = 0.0;

            // New lows count
            let mut new_low_count = 0;
            let mut running_min = low[start];
            for j in (start + 1)..=i {
                if low[j] < running_min {
                    new_low_count += 1;
                    running_min = low[j];
                }
            }
            capitulation += (new_low_count as f64 / self.period as f64) * 30.0;

            // Volume spike (panic selling)
            let avg_vol = volume[start..(i - 1).max(start + 1)].iter().sum::<f64>()
                / ((i - 1).max(start + 1) - start).max(1) as f64;
            if avg_vol > 0.0 && volume[i] > avg_vol * self.vol_threshold {
                capitulation += 30.0;
            }

            // Momentum weakness
            let momentum = (close[i] / close[start] - 1.0) * 100.0;
            capitulation += (-momentum * 2.0).max(0.0).min(40.0);

            result[i] = capitulation.min(100.0);
        }
        result
    }
}

impl TechnicalIndicator for Capitulation {
    fn name(&self) -> &str {
        "Capitulation"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.low, &data.close, &data.volume)))
    }
}

/// Smart Money Confidence - Institutional behavior proxy
#[derive(Debug, Clone)]
pub struct SmartMoneyConfidence {
    period: usize,
}

impl SmartMoneyConfidence {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate smart money confidence (-100 to 100)
    /// Uses volume-weighted close location
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let mut weighted_position = 0.0;
            let mut total_volume = 0.0;

            for j in start..=i {
                let range = high[j] - low[j];
                if range > 0.0 {
                    // Close position: 0 = low, 1 = high
                    let position = (close[j] - low[j]) / range;
                    // Smart money accumulates at lows, distributes at highs
                    // Strong close (near high) with high volume = smart money buying
                    weighted_position += (position * 2.0 - 1.0) * volume[j];
                    total_volume += volume[j];
                }
            }

            if total_volume > 0.0 {
                result[i] = (weighted_position / total_volume) * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for SmartMoneyConfidence {
    fn name(&self) -> &str {
        "Smart Money Confidence"
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
    fn test_fear_index() {
        let (high, low, close, _) = make_test_data();
        let fi = FearIndex::new(10).unwrap();
        let result = fi.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Fear should be between 0 and 100
        assert!(result[15] >= 0.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_greed_index() {
        let (high, low, close, _) = make_test_data();
        let gi = GreedIndex::new(10).unwrap();
        let result = gi.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Greed should be between 0 and 100
        assert!(result[15] >= 0.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_crowd_psychology() {
        let (_, _, close, volume) = make_test_data();
        let cp = CrowdPsychology::new(10).unwrap();
        let result = cp.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // In uptrend, should be positive
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_market_euphoria() {
        let (high, _, close, volume) = make_test_data();
        let me = MarketEuphoria::new(10, 1.5).unwrap();
        let result = me.calculate(&high, &close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= 0.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_capitulation() {
        let (_, low, close, volume) = make_test_data();
        let cap = Capitulation::new(10, 1.5).unwrap();
        let result = cap.calculate(&low, &close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= 0.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_smart_money_confidence() {
        let (high, low, close, volume) = make_test_data();
        let smc = SmartMoneyConfidence::new(10).unwrap();
        let result = smc.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Should be between -100 and 100
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }
}
