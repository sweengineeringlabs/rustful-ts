//! Multi-Factor Composite Indicators
//!
//! Advanced composite indicators combining multiple technical analysis factors.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Quality Momentum Factor - Combines momentum quality with direction
#[derive(Debug, Clone)]
pub struct QualityMomentumFactor {
    momentum_period: usize,
    quality_period: usize,
}

impl QualityMomentumFactor {
    pub fn new(momentum_period: usize, quality_period: usize) -> Result<Self> {
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if quality_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "quality_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { momentum_period, quality_period })
    }

    /// Calculate quality-adjusted momentum factor
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let max_period = self.momentum_period.max(self.quality_period);

        for i in max_period..n {
            // Raw momentum
            let momentum = if close[i - self.momentum_period] > 1e-10 {
                (close[i] / close[i - self.momentum_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Calculate quality (consistency of returns)
            let start = i.saturating_sub(self.quality_period);
            let mut positive_count = 0;
            let mut negative_count = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    positive_count += 1;
                } else if close[j] < close[j - 1] {
                    negative_count += 1;
                }
            }

            let total = positive_count + negative_count;
            let quality = if total > 0 {
                (positive_count as f64 - negative_count as f64).abs() / total as f64
            } else {
                0.0
            };

            // Quality-adjusted momentum
            result[i] = momentum * (1.0 + quality);
        }

        result
    }
}

impl TechnicalIndicator for QualityMomentumFactor {
    fn name(&self) -> &str {
        "Quality Momentum Factor"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period.max(self.quality_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Value Momentum Composite - Combines value and momentum signals
#[derive(Debug, Clone)]
pub struct ValueMomentumComposite {
    value_period: usize,
    momentum_period: usize,
}

impl ValueMomentumComposite {
    pub fn new(value_period: usize, momentum_period: usize) -> Result<Self> {
        if value_period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "value_period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { value_period, momentum_period })
    }

    /// Calculate composite score (positive = value+momentum alignment)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.value_period..n {
            // Value signal: deviation from long-term average
            let start = i.saturating_sub(self.value_period);
            let long_avg: f64 = close[start..=i].iter().sum::<f64>() / self.value_period as f64;
            let value_signal = (long_avg - close[i]) / close[i] * 100.0; // Positive = undervalued

            // Momentum signal
            let mom_idx = i.saturating_sub(self.momentum_period);
            let momentum = if close[mom_idx] > 1e-10 {
                (close[i] / close[mom_idx] - 1.0) * 100.0
            } else {
                0.0
            };

            // Composite: Value + Momentum (contrarian value with momentum confirmation)
            result[i] = value_signal * 0.5 + momentum * 0.5;
        }

        result
    }
}

impl TechnicalIndicator for ValueMomentumComposite {
    fn name(&self) -> &str {
        "Value Momentum Composite"
    }

    fn min_periods(&self) -> usize {
        self.value_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Risk-Adjusted Trend Score - Trend strength adjusted for volatility
#[derive(Debug, Clone)]
pub struct RiskAdjustedTrend {
    trend_period: usize,
    volatility_period: usize,
}

impl RiskAdjustedTrend {
    pub fn new(trend_period: usize, volatility_period: usize) -> Result<Self> {
        if trend_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if volatility_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { trend_period, volatility_period })
    }

    /// Calculate risk-adjusted trend score
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let max_period = self.trend_period.max(self.volatility_period);

        for i in max_period..n {
            // Trend return
            let trend_return = if close[i - self.trend_period] > 1e-10 {
                (close[i] / close[i - self.trend_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Volatility (standard deviation of returns)
            let vol_start = i.saturating_sub(self.volatility_period);
            let mut returns = Vec::new();
            for j in (vol_start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len().max(1) as f64;
            let variance: f64 = returns.iter()
                .map(|r| (r - mean_ret).powi(2))
                .sum::<f64>() / returns.len().max(1) as f64;
            let volatility = variance.sqrt() * 100.0;

            // Risk-adjusted score (similar to Sharpe)
            if volatility > 1e-10 {
                result[i] = trend_return / volatility;
            }
        }

        result
    }
}

impl TechnicalIndicator for RiskAdjustedTrend {
    fn name(&self) -> &str {
        "Risk Adjusted Trend"
    }

    fn min_periods(&self) -> usize {
        self.trend_period.max(self.volatility_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Breakout Strength Index - Measures breakout strength with volume confirmation
#[derive(Debug, Clone)]
pub struct BreakoutStrengthIndex {
    range_period: usize,
    volume_period: usize,
}

impl BreakoutStrengthIndex {
    pub fn new(range_period: usize, volume_period: usize) -> Result<Self> {
        if range_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "range_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if volume_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { range_period, volume_period })
    }

    /// Calculate breakout strength (positive = bullish breakout, negative = bearish)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let max_period = self.range_period.max(self.volume_period);

        for i in max_period..n {
            // Find range high/low
            let range_start = i.saturating_sub(self.range_period);
            let mut range_high = high[range_start];
            let mut range_low = low[range_start];

            for j in range_start..i {
                if high[j] > range_high { range_high = high[j]; }
                if low[j] < range_low { range_low = low[j]; }
            }

            // Volume confirmation
            let vol_start = i.saturating_sub(self.volume_period);
            let avg_volume: f64 = volume[vol_start..i].iter().sum::<f64>() / self.volume_period as f64;
            let volume_ratio = if avg_volume > 1e-10 { volume[i] / avg_volume } else { 1.0 };

            // Breakout strength
            let range = range_high - range_low;
            if range > 1e-10 {
                if close[i] > range_high {
                    // Bullish breakout
                    result[i] = ((close[i] - range_high) / range * 100.0) * volume_ratio;
                } else if close[i] < range_low {
                    // Bearish breakout
                    result[i] = -((range_low - close[i]) / range * 100.0) * volume_ratio;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for BreakoutStrengthIndex {
    fn name(&self) -> &str {
        "Breakout Strength Index"
    }

    fn min_periods(&self) -> usize {
        self.range_period.max(self.volume_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close, &data.volume)))
    }
}

/// Trend Reversal Probability - Estimates probability of trend reversal
#[derive(Debug, Clone)]
pub struct TrendReversalProbability {
    trend_period: usize,
    exhaustion_threshold: f64,
}

impl TrendReversalProbability {
    pub fn new(trend_period: usize, exhaustion_threshold: f64) -> Result<Self> {
        if trend_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if exhaustion_threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "exhaustion_threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { trend_period, exhaustion_threshold })
    }

    /// Calculate reversal probability (0-100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.trend_period..n {
            let start = i.saturating_sub(self.trend_period);

            // Trend direction and magnitude
            let trend_return = if close[start] > 1e-10 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Count consecutive same-direction bars
            let mut consecutive = 0;
            for j in (start + 1)..=i {
                let bar_direction = close[j] - close[j - 1];
                if (trend_return > 0.0 && bar_direction > 0.0) || (trend_return < 0.0 && bar_direction < 0.0) {
                    consecutive += 1;
                } else {
                    consecutive = 0;
                }
            }

            // Volatility expansion check
            let recent_range = high[i] - low[i];
            let avg_range: f64 = (start..i).map(|j| high[j] - low[j]).sum::<f64>() / self.trend_period as f64;
            let range_expansion = if avg_range > 1e-10 { recent_range / avg_range } else { 1.0 };

            // Reversal probability factors:
            // 1. Extended trend (many consecutive bars)
            let consecutive_factor = (consecutive as f64 / self.trend_period as f64 * 50.0).min(50.0);

            // 2. Large move magnitude
            let magnitude_factor = (trend_return.abs() / self.exhaustion_threshold * 25.0).min(25.0);

            // 3. Range expansion (exhaustion signal)
            let expansion_factor = ((range_expansion - 1.0) * 25.0).max(0.0).min(25.0);

            result[i] = consecutive_factor + magnitude_factor + expansion_factor;
        }

        result
    }
}

impl TechnicalIndicator for TrendReversalProbability {
    fn name(&self) -> &str {
        "Trend Reversal Probability"
    }

    fn min_periods(&self) -> usize {
        self.trend_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Multi-Factor Alpha Score - Combines multiple alpha factors
#[derive(Debug, Clone)]
pub struct MultiFactorAlphaScore {
    short_period: usize,
    long_period: usize,
}

impl MultiFactorAlphaScore {
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

    /// Calculate composite alpha score
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Factor 1: Price momentum
            let short_mom = if close[i - self.short_period] > 1e-10 {
                (close[i] / close[i - self.short_period] - 1.0) * 100.0
            } else {
                0.0
            };

            let long_mom = if close[i - self.long_period] > 1e-10 {
                (close[i] / close[i - self.long_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Factor 2: Volume trend
            let short_start = i.saturating_sub(self.short_period);
            let long_start = i.saturating_sub(self.long_period);
            let short_vol: f64 = volume[short_start..=i].iter().sum::<f64>() / self.short_period as f64;
            let long_vol: f64 = volume[long_start..=i].iter().sum::<f64>() / self.long_period as f64;
            let vol_factor = if long_vol > 1e-10 { (short_vol / long_vol - 1.0) * 50.0 } else { 0.0 };

            // Factor 3: Mean reversion
            let long_ma: f64 = close[long_start..=i].iter().sum::<f64>() / self.long_period as f64;
            let reversion_factor = if long_ma > 1e-10 { (long_ma - close[i]) / long_ma * 100.0 } else { 0.0 };

            // Combined score with momentum emphasis
            result[i] = short_mom * 0.4 + long_mom * 0.3 + vol_factor * 0.15 + reversion_factor * 0.15;
        }

        result
    }
}

impl TechnicalIndicator for MultiFactorAlphaScore {
    fn name(&self) -> &str {
        "Multi Factor Alpha Score"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.3).sin() * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 0.5).sin() * 500.0).collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_quality_momentum_factor() {
        let data = make_test_data();
        let qmf = QualityMomentumFactor::new(10, 14).unwrap();
        let result = qmf.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_value_momentum_composite() {
        let data = make_test_data();
        let vmc = ValueMomentumComposite::new(20, 10).unwrap();
        let result = vmc.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_risk_adjusted_trend() {
        let data = make_test_data();
        let rat = RiskAdjustedTrend::new(14, 14).unwrap();
        let result = rat.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_breakout_strength_index() {
        let data = make_test_data();
        let bsi = BreakoutStrengthIndex::new(14, 10).unwrap();
        let result = bsi.calculate(&data.high, &data.low, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_trend_reversal_probability() {
        let data = make_test_data();
        let trp = TrendReversalProbability::new(14, 10.0).unwrap();
        let result = trp.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        // Probability should be 0-100
        for i in 20..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_multi_factor_alpha_score() {
        let data = make_test_data();
        let mfas = MultiFactorAlphaScore::new(7, 21).unwrap();
        let result = mfas.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_validation() {
        assert!(QualityMomentumFactor::new(2, 10).is_err());
        assert!(ValueMomentumComposite::new(10, 10).is_err()); // value < 20
        assert!(RiskAdjustedTrend::new(5, 10).is_err()); // trend < 10
        assert!(BreakoutStrengthIndex::new(5, 10).is_err()); // range < 10
        assert!(TrendReversalProbability::new(5, 10.0).is_err()); // trend < 10
        assert!(MultiFactorAlphaScore::new(10, 5).is_err()); // long <= short
    }
}
