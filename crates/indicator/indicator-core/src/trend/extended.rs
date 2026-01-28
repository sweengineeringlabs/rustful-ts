//! Extended Trend Indicators
//!
//! Additional trend-following and trend analysis indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Composite Trend Score - Multi-factor trend measurement
#[derive(Debug, Clone)]
pub struct CompositeTrendScore {
    short_period: usize,
    long_period: usize,
}

impl CompositeTrendScore {
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

    /// Calculate trend score (-100 to +100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let mut score = 0.0;

            // Price above/below short MA
            let short_start = i.saturating_sub(self.short_period);
            let short_ma: f64 = close[short_start..=i].iter().sum::<f64>() / self.short_period as f64;
            if close[i] > short_ma { score += 25.0; } else { score -= 25.0; }

            // Price above/below long MA
            let long_start = i.saturating_sub(self.long_period);
            let long_ma: f64 = close[long_start..=i].iter().sum::<f64>() / self.long_period as f64;
            if close[i] > long_ma { score += 25.0; } else { score -= 25.0; }

            // Short MA above/below long MA
            if short_ma > long_ma { score += 25.0; } else { score -= 25.0; }

            // Price momentum (higher highs/lower lows)
            let mid = (i + long_start) / 2;
            let recent_max = close[mid..=i].iter().cloned().fold(f64::MIN, f64::max);
            let older_max = close[long_start..=mid].iter().cloned().fold(f64::MIN, f64::max);
            if recent_max > older_max { score += 25.0; } else { score -= 25.0; }

            result[i] = score;
        }
        result
    }
}

impl TechnicalIndicator for CompositeTrendScore {
    fn name(&self) -> &str {
        "Composite Trend Score"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Persistence - Measures how long trend has been sustained
#[derive(Debug, Clone)]
pub struct TrendPersistence {
    period: usize,
}

impl TrendPersistence {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate consecutive bars in trend direction
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate MA for trend direction
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let ma: f64 = close[start..=i].iter().sum::<f64>() / self.period as f64;

            // Count consecutive bars above/below MA
            let above = close[i] > ma;
            let mut count = 0;

            for j in (start..=i).rev() {
                let bar_ma: f64 = close[j.saturating_sub(self.period)..=j]
                    .iter()
                    .sum::<f64>() / self.period.min(j + 1) as f64;

                let bar_above = close[j] > bar_ma;
                if bar_above == above {
                    count += 1;
                } else {
                    break;
                }
            }

            result[i] = if above { count as f64 } else { -(count as f64) };
        }
        result
    }
}

impl TechnicalIndicator for TrendPersistence {
    fn name(&self) -> &str {
        "Trend Persistence"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Price Channel Position - Where price sits in channel (0-100)
#[derive(Debug, Clone)]
pub struct PriceChannelPosition {
    period: usize,
}

impl PriceChannelPosition {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate position within price channel
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let channel_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let channel_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);

            let range = channel_high - channel_low;
            if range > 1e-10 {
                result[i] = ((close[i] - channel_low) / range) * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for PriceChannelPosition {
    fn name(&self) -> &str {
        "Price Channel Position"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Trend Exhaustion Index - Measures trend overextension
#[derive(Debug, Clone)]
pub struct TrendExhaustion {
    period: usize,
    threshold: f64,
}

impl TrendExhaustion {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate exhaustion level
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let returns: Vec<f64> = (start + 1..=i)
                .map(|j| close[j] / close[j - 1] - 1.0)
                .collect();

            let total_return: f64 = returns.iter().sum();
            let avg_return = total_return / returns.len() as f64;

            // Std dev of returns
            let var: f64 = returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
                / returns.len() as f64;
            let std = var.sqrt();

            // Z-score of cumulative return
            if std > 1e-10 {
                let z_score = total_return / std;
                // Exhaustion increases as z-score exceeds threshold
                result[i] = if z_score.abs() > self.threshold {
                    z_score.signum() * (z_score.abs() - self.threshold)
                } else {
                    0.0
                };
            }
        }
        result
    }
}

impl TechnicalIndicator for TrendExhaustion {
    fn name(&self) -> &str {
        "Trend Exhaustion"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Directional Movement Quality - Quality of trending movement
#[derive(Debug, Clone)]
pub struct DirectionalMovementQuality {
    period: usize,
}

impl DirectionalMovementQuality {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate quality of directional movement
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate directional movement
            let mut plus_dm_sum = 0.0;
            let mut minus_dm_sum = 0.0;
            let mut tr_sum = 0.0;

            for j in (start + 1)..=i {
                let up_move = high[j] - high[j - 1];
                let down_move = low[j - 1] - low[j];

                if up_move > down_move && up_move > 0.0 {
                    plus_dm_sum += up_move;
                }
                if down_move > up_move && down_move > 0.0 {
                    minus_dm_sum += down_move;
                }

                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                tr_sum += tr;
            }

            if tr_sum > 1e-10 {
                let di_plus = plus_dm_sum / tr_sum * 100.0;
                let di_minus = minus_dm_sum / tr_sum * 100.0;
                let di_diff = (di_plus - di_minus).abs();
                let di_sum = di_plus + di_minus;

                if di_sum > 1e-10 {
                    // DMQ: combination of DX and directional bias
                    let dx = di_diff / di_sum * 100.0;
                    let direction = if di_plus > di_minus { 1.0 } else { -1.0 };
                    result[i] = direction * dx;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for DirectionalMovementQuality {
    fn name(&self) -> &str {
        "Directional Movement Quality"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Multi-Timeframe Trend - Composite across lookbacks
#[derive(Debug, Clone)]
pub struct MultiTimeframeTrend {
    periods: Vec<usize>,
}

impl MultiTimeframeTrend {
    pub fn new(periods: Vec<usize>) -> Result<Self> {
        if periods.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "periods".to_string(),
                reason: "must have at least one period".to_string(),
            });
        }
        if periods.iter().any(|&p| p < 2) {
            return Err(IndicatorError::InvalidParameter {
                name: "periods".to_string(),
                reason: "all periods must be at least 2".to_string(),
            });
        }
        Ok(Self { periods })
    }

    /// Calculate multi-timeframe trend score
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let max_period = *self.periods.iter().max().unwrap_or(&0);

        for i in max_period..n {
            let mut trend_sum = 0.0;

            for &period in &self.periods {
                let start = i.saturating_sub(period);

                // Check if trending up or down
                let ma: f64 = close[start..=i].iter().sum::<f64>() / period as f64;
                let slope = (close[i] - close[start]) / period as f64;

                if close[i] > ma && slope > 0.0 {
                    trend_sum += 1.0;
                } else if close[i] < ma && slope < 0.0 {
                    trend_sum -= 1.0;
                }
            }

            // Normalize to -100 to +100
            result[i] = trend_sum / self.periods.len() as f64 * 100.0;
        }
        result
    }
}

impl TechnicalIndicator for MultiTimeframeTrend {
    fn name(&self) -> &str {
        "Multi-Timeframe Trend"
    }

    fn min_periods(&self) -> usize {
        *self.periods.iter().max().unwrap_or(&0) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                       112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
                       122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0];
        let low = vec![98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
                      108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0,
                      118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0];
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                        110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
                        120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0];
        (high, low, close)
    }

    #[test]
    fn test_composite_trend_score() {
        let (_, _, close) = make_test_data();
        let ts = CompositeTrendScore::new(5, 15).unwrap();
        let result = ts.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Score should be within -100 to +100 range
        assert!(result[20] >= -100.0 && result[20] <= 100.0);
    }

    #[test]
    fn test_trend_persistence() {
        let (_, _, close) = make_test_data();
        let tp = TrendPersistence::new(5).unwrap();
        let result = tp.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Persistence is non-zero in trending market
        assert!(result[20] != 0.0 || result[25] != 0.0);
    }

    #[test]
    fn test_price_channel_position() {
        let (high, low, close) = make_test_data();
        let pcp = PriceChannelPosition::new(10).unwrap();
        let result = pcp.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Should be between 0 and 100
        assert!(result[20] >= 0.0 && result[20] <= 100.0);
    }

    #[test]
    fn test_trend_exhaustion() {
        let (_, _, close) = make_test_data();
        let te = TrendExhaustion::new(10, 1.5).unwrap();
        let result = te.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_directional_movement_quality() {
        let (high, low, close) = make_test_data();
        let dmq = DirectionalMovementQuality::new(10).unwrap();
        let result = dmq.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Positive in uptrend
        assert!(result[20] > 0.0);
    }

    #[test]
    fn test_multi_timeframe_trend() {
        let (_, _, close) = make_test_data();
        let mtt = MultiTimeframeTrend::new(vec![5, 10, 20]).unwrap();
        let result = mtt.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Should be positive in consistent uptrend
        assert!(result[25] > 0.0);
    }
}
