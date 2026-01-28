//! Extended Composite Indicators
//!
//! Additional composite indicators combining multiple analysis methods.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Trend Momentum Score - Combines trend and momentum analysis
#[derive(Debug, Clone)]
pub struct TrendMomentumScore {
    trend_period: usize,
    momentum_period: usize,
}

impl TrendMomentumScore {
    pub fn new(trend_period: usize, momentum_period: usize) -> Result<Self> {
        if trend_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { trend_period, momentum_period })
    }

    /// Calculate combined trend-momentum score (-100 to 100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        let max_period = self.trend_period.max(self.momentum_period);

        for i in max_period..n {
            // Trend component: price vs SMA
            let trend_start = i.saturating_sub(self.trend_period);
            let trend_avg: f64 = close[trend_start..=i].iter().sum::<f64>() / (i - trend_start + 1) as f64;
            let trend_score = ((close[i] / trend_avg - 1.0) * 100.0).clamp(-50.0, 50.0);

            // Momentum component: ROC
            let mom_start = i.saturating_sub(self.momentum_period);
            let momentum = (close[i] / close[mom_start] - 1.0) * 100.0;
            let momentum_score = momentum.clamp(-50.0, 50.0);

            // Combine with equal weights
            result[i] = (trend_score + momentum_score).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for TrendMomentumScore {
    fn name(&self) -> &str {
        "Trend Momentum Score"
    }

    fn min_periods(&self) -> usize {
        self.trend_period.max(self.momentum_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volatility Trend Combo - Combines volatility and trend signals
#[derive(Debug, Clone)]
pub struct VolatilityTrendCombo {
    period: usize,
}

impl VolatilityTrendCombo {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volatility-adjusted trend score
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate ATR for volatility
            let mut atr_sum = 0.0;
            for j in (start + 1)..=i {
                let tr1 = high[j] - low[j];
                let tr2 = (high[j] - close[j - 1]).abs();
                let tr3 = (low[j] - close[j - 1]).abs();
                atr_sum += tr1.max(tr2).max(tr3);
            }
            let atr = atr_sum / self.period as f64;

            // Calculate trend
            let avg = close[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let trend_direction = if close[i] > avg { 1.0 } else { -1.0 };

            // Volatility normalized trend strength
            let price_change = (close[i] - close[start]).abs();
            let vol_adjusted_strength = if atr > 1e-10 {
                (price_change / (atr * self.period as f64)).min(2.0)
            } else {
                0.0
            };

            result[i] = (trend_direction * vol_adjusted_strength * 50.0).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for VolatilityTrendCombo {
    fn name(&self) -> &str {
        "Volatility Trend Combo"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Multi-Period Momentum - Combines momentum across multiple periods
#[derive(Debug, Clone)]
pub struct MultiPeriodMomentum {
    short_period: usize,
    medium_period: usize,
    long_period: usize,
}

impl MultiPeriodMomentum {
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

    /// Calculate multi-period momentum score
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term momentum
            let short_mom = (close[i] / close[i - self.short_period] - 1.0) * 100.0;
            let short_score = short_mom.clamp(-33.33, 33.33);

            // Medium-term momentum
            let med_mom = (close[i] / close[i - self.medium_period] - 1.0) * 100.0;
            let med_score = med_mom.clamp(-33.33, 33.33);

            // Long-term momentum
            let long_mom = (close[i] / close[i - self.long_period] - 1.0) * 100.0;
            let long_score = long_mom.clamp(-33.33, 33.33);

            // Combine with weights: 50% short, 30% medium, 20% long
            result[i] = (short_score * 0.5 + med_score * 0.3 + long_score * 0.2) * 3.0;
        }
        result
    }
}

impl TechnicalIndicator for MultiPeriodMomentum {
    fn name(&self) -> &str {
        "Multi Period Momentum"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Momentum Strength Index - Combined momentum strength
#[derive(Debug, Clone)]
pub struct MomentumStrengthIndex {
    period: usize,
}

impl MomentumStrengthIndex {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate momentum strength index (0 to 100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate gains and losses
            let mut gains = 0.0;
            let mut losses = 0.0;
            for j in (start + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses -= change;
                }
            }

            // RSI-like calculation
            let total_movement = gains + losses;
            let strength = if total_movement > 1e-10 {
                gains / total_movement * 100.0
            } else {
                50.0
            };

            // Momentum direction
            let momentum = close[i] - close[start];
            let direction: f64 = if momentum > 0.0 { 1.0 } else { -1.0 };

            // Combine strength with direction
            result[i] = ((strength - 50.0) * 2.0 * direction).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for MomentumStrengthIndex {
    fn name(&self) -> &str {
        "Momentum Strength Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Market Condition Score - Overall market condition assessment
#[derive(Debug, Clone)]
pub struct MarketConditionScore {
    period: usize,
}

impl MarketConditionScore {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate market condition score (-100 to 100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Trend component
            let avg = close[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let trend_score = ((close[i] / avg - 1.0) * 200.0).clamp(-33.33, 33.33);

            // Volatility component (lower = better)
            let returns: Vec<f64> = (start + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 {
                        Some(close[j] / close[j - 1] - 1.0)
                    } else {
                        None
                    }
                })
                .collect();
            let vol_score = if !returns.is_empty() {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let var: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                let std_dev = var.sqrt();
                // Lower volatility is positive
                (10.0 - std_dev * 1000.0).clamp(-33.33, 33.33)
            } else {
                0.0
            };

            // Volume confirmation
            let avg_vol = volume[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let vol_ratio = if avg_vol > 0.0 { volume[i] / avg_vol } else { 1.0 };
            let volume_score = if trend_score > 0.0 && vol_ratio > 1.0 {
                // Uptrend with increasing volume = bullish
                (vol_ratio - 1.0).min(0.5) * 66.66
            } else if trend_score < 0.0 && vol_ratio > 1.0 {
                // Downtrend with increasing volume = bearish
                -(vol_ratio - 1.0).min(0.5) * 66.66
            } else {
                0.0
            };

            result[i] = (trend_score + vol_score + volume_score).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for MarketConditionScore {
    fn name(&self) -> &str {
        "Market Condition Score"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Price Action Score - Analyzes price action quality
#[derive(Debug, Clone)]
pub struct PriceActionScore {
    period: usize,
}

impl PriceActionScore {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate price action quality score
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Body to range ratio (strong bars have larger bodies)
            let mut body_ratio_sum = 0.0;
            let mut direction_sum = 0.0;

            for j in start..=i {
                let range = high[j] - low[j];
                let body = (close[j] - open[j]).abs();
                if range > 1e-10 {
                    body_ratio_sum += body / range;
                    direction_sum += if close[j] > open[j] { 1.0 } else { -1.0 };
                }
            }

            let avg_body_ratio = body_ratio_sum / (self.period + 1) as f64;
            let direction_bias = direction_sum / (self.period + 1) as f64;

            // Score combines body strength with directional bias
            let strength = avg_body_ratio * 50.0;
            result[i] = (strength * direction_bias).clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for PriceActionScore {
    fn name(&self) -> &str {
        "Price Action Score"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close)))
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
        let close = vec![101.0, 102.0, 103.0, 102.5, 104.0, 105.0, 104.5, 106.0, 107.0, 106.5,
                        108.0, 109.0, 108.5, 110.0, 111.0, 110.5, 112.0, 113.0, 112.5, 114.0,
                        115.0, 114.5, 116.0, 117.0, 116.5, 118.0, 119.0, 118.5, 120.0, 121.0];
        let volume = vec![1000.0; 30];
        (open, high, low, close, volume)
    }

    #[test]
    fn test_trend_momentum_score() {
        let (_, _, _, close, _) = make_test_data();
        let tms = TrendMomentumScore::new(10, 5).unwrap();
        let result = tms.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_volatility_trend_combo() {
        let (_, high, low, close, _) = make_test_data();
        let vtc = VolatilityTrendCombo::new(10).unwrap();
        let result = vtc.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_multi_period_momentum() {
        let (_, _, _, close, _) = make_test_data();
        let mpm = MultiPeriodMomentum::new(5, 10, 20).unwrap();
        let result = mpm.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_momentum_strength_index() {
        let (_, _, _, close, _) = make_test_data();
        let msi = MomentumStrengthIndex::new(10).unwrap();
        let result = msi.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_market_condition_score() {
        let (_, high, low, close, volume) = make_test_data();
        let mcs = MarketConditionScore::new(10).unwrap();
        let result = mcs.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_price_action_score() {
        let (open, high, low, close, _) = make_test_data();
        let pas = PriceActionScore::new(10).unwrap();
        let result = pas.calculate(&open, &high, &low, &close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }
}
