//! Extended DeMark-Style Indicators
//!
//! Additional technical indicators inspired by Tom DeMark's methodologies.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// TD Camouflage - Detects camouflaged trend reversals
#[derive(Debug, Clone)]
pub struct TDCamouflage {
    lookback: usize,
}

impl TDCamouflage {
    pub fn new(lookback: usize) -> Result<Self> {
        if lookback < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { lookback })
    }

    /// Calculate TD Camouflage signal
    /// Returns: positive for bullish camouflage, negative for bearish, 0 for none
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            // Bullish Camouflage: Close lower than open but higher than previous close
            // after consecutive lower closes
            let mut consecutive_lower = true;
            for j in 1..self.lookback.min(i) {
                if close[i - j] >= close[i - j - 1] {
                    consecutive_lower = false;
                    break;
                }
            }

            // Bearish Camouflage: Close higher than open but lower than previous close
            // after consecutive higher closes
            let mut consecutive_higher = true;
            for j in 1..self.lookback.min(i) {
                if close[i - j] <= close[i - j - 1] {
                    consecutive_higher = false;
                    break;
                }
            }

            // Check for camouflage patterns
            if consecutive_lower && i > 0 {
                // Bullish: Today's low is below yesterday's but close is above open
                let today_open = if i > 0 { close[i - 1] } else { close[i] }; // Use previous close as proxy
                if low[i] < low[i - 1] && close[i] > today_open {
                    result[i] = 1.0;
                }
            }

            if consecutive_higher && i > 0 {
                // Bearish: Today's high is above yesterday's but close is below open
                let today_open = if i > 0 { close[i - 1] } else { close[i] };
                if high[i] > high[i - 1] && close[i] < today_open {
                    result[i] = -1.0;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for TDCamouflage {
    fn name(&self) -> &str {
        "TD Camouflage"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// TD CLOP - Close-Open Pattern Analysis
#[derive(Debug, Clone)]
pub struct TDCLOP {
    period: usize,
}

impl TDCLOP {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate TD CLOP (Close minus Open pattern normalized)
    pub fn calculate(&self, open: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Sum of close-open differences
            let mut clop_sum = 0.0;
            let mut abs_sum = 0.0;

            for j in start..=i {
                let diff = close[j] - open[j];
                clop_sum += diff;
                abs_sum += diff.abs();
            }

            // Normalize to -100 to +100
            if abs_sum > 1e-10 {
                result[i] = (clop_sum / abs_sum) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for TDCLOP {
    fn name(&self) -> &str {
        "TD CLOP"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.close)))
    }
}

/// TD Moving Average Qualifier - Qualifies price position relative to MAs
#[derive(Debug, Clone)]
pub struct TDMovingAverageQualifier {
    fast_period: usize,
    slow_period: usize,
}

impl TDMovingAverageQualifier {
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

    /// Calculate MA qualifier (price position relative to MAs)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.slow_period..n {
            // Calculate fast MA
            let fast_start = i.saturating_sub(self.fast_period - 1);
            let fast_ma: f64 = close[fast_start..=i].iter().sum::<f64>() / self.fast_period as f64;

            // Calculate slow MA
            let slow_start = i.saturating_sub(self.slow_period - 1);
            let slow_ma: f64 = close[slow_start..=i].iter().sum::<f64>() / self.slow_period as f64;

            // Qualifier score based on price vs MAs
            let mut score = 0.0;

            // +1 if price above fast MA
            if close[i] > fast_ma {
                score += 1.0;
            } else {
                score -= 1.0;
            }

            // +1 if price above slow MA
            if close[i] > slow_ma {
                score += 1.0;
            } else {
                score -= 1.0;
            }

            // +1 if fast MA above slow MA (trend alignment)
            if fast_ma > slow_ma {
                score += 1.0;
            } else {
                score -= 1.0;
            }

            result[i] = score;
        }

        result
    }
}

impl TechnicalIndicator for TDMovingAverageQualifier {
    fn name(&self) -> &str {
        "TD MA Qualifier"
    }

    fn min_periods(&self) -> usize {
        self.slow_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// TD Risk Level - Calculates risk level based on recent volatility and position
#[derive(Debug, Clone)]
pub struct TDRiskLevel {
    period: usize,
}

impl TDRiskLevel {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate TD-style risk level (0-100, higher = more risk)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Find high/low range over period
            let mut period_high = high[start];
            let mut period_low = low[start];

            for j in start..=i {
                if high[j] > period_high {
                    period_high = high[j];
                }
                if low[j] < period_low {
                    period_low = low[j];
                }
            }

            let range = period_high - period_low;
            if range < 1e-10 {
                continue;
            }

            // Risk level: how extended is current price within the range
            // Near high = high risk for longs, near low = high risk for shorts
            let position_in_range = (close[i] - period_low) / range;

            // Calculate volatility expansion
            let tr_sum: f64 = (start + 1..=i)
                .map(|j| {
                    let tr = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    tr
                })
                .sum();
            let avg_tr = tr_sum / self.period as f64;
            let current_tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            let volatility_ratio = if avg_tr > 1e-10 { current_tr / avg_tr } else { 1.0 };

            // Risk = position extremity * volatility expansion
            let position_risk = (position_in_range - 0.5).abs() * 2.0; // 0 at center, 1 at extremes
            result[i] = (position_risk * volatility_ratio * 100.0).min(100.0);
        }

        result
    }
}

impl TechnicalIndicator for TDRiskLevel {
    fn name(&self) -> &str {
        "TD Risk Level"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// TD Momentum - DeMark-style momentum oscillator
#[derive(Debug, Clone)]
pub struct TDMomentum {
    period: usize,
    compare_bars: usize,
}

impl TDMomentum {
    pub fn new(period: usize, compare_bars: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if compare_bars < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "compare_bars".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, compare_bars })
    }

    /// Calculate TD Momentum (comparison-based momentum)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        let required = self.period + self.compare_bars;
        if n < required {
            return result;
        }

        for i in required..n {
            // Count closes higher/lower than compare_bars ago
            let mut higher_count = 0;
            let mut lower_count = 0;

            for j in 0..self.period {
                let idx = i - j;
                let compare_idx = idx - self.compare_bars;
                if compare_idx < n {
                    if close[idx] > close[compare_idx] {
                        higher_count += 1;
                    } else if close[idx] < close[compare_idx] {
                        lower_count += 1;
                    }
                }
            }

            // Momentum score: -period to +period
            result[i] = (higher_count as f64 - lower_count as f64) / self.period as f64 * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for TDMomentum {
    fn name(&self) -> &str {
        "TD Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + self.compare_bars + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// TD Differential - Differential analysis of consecutive price changes
#[derive(Debug, Clone)]
pub struct TDDifferential {
    period: usize,
}

impl TDDifferential {
    pub fn new(period: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate TD Differential (acceleration/deceleration of price changes)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 2 {
            return result;
        }

        // First, calculate first differences (changes)
        let mut changes = vec![0.0; n];
        for i in 1..n {
            changes[i] = close[i] - close[i - 1];
        }

        // Calculate second differences (change in changes = acceleration)
        let mut acceleration = vec![0.0; n];
        for i in 2..n {
            acceleration[i] = changes[i] - changes[i - 1];
        }

        // Sum accelerations over period
        for i in (self.period + 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let acc_sum: f64 = acceleration[start..=i].iter().sum();

            // Normalize by average price to make comparable
            let avg_price: f64 = close[start..=i].iter().sum::<f64>() / self.period as f64;
            if avg_price > 1e-10 {
                result[i] = (acc_sum / avg_price) * 1000.0; // Scale for readability
            }
        }

        result
    }
}

impl TechnicalIndicator for TDDifferential {
    fn name(&self) -> &str {
        "TD Differential"
    }

    fn min_periods(&self) -> usize {
        self.period + 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.3).sin() * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.iter().enumerate().map(|(i, c)| if i > 0 { close[i - 1] } else { *c }).collect();
        let volume: Vec<f64> = vec![1000.0; 30];

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_td_camouflage() {
        let data = make_test_data();
        let camo = TDCamouflage::new(3).unwrap();
        let result = camo.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be -1, 0, or 1
        for val in &result {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_td_clop() {
        let data = make_test_data();
        let clop = TDCLOP::new(5).unwrap();
        let result = clop.calculate(&data.open, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be in -100 to +100
        for i in 10..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_td_ma_qualifier() {
        let data = make_test_data();
        let maq = TDMovingAverageQualifier::new(5, 15).unwrap();
        let result = maq.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be -3 to +3
        for i in 20..result.len() {
            assert!(result[i] >= -3.0 && result[i] <= 3.0);
        }
    }

    #[test]
    fn test_td_risk_level() {
        let data = make_test_data();
        let risk = TDRiskLevel::new(10).unwrap();
        let result = risk.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be 0 to 100
        for i in 15..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_td_momentum() {
        let data = make_test_data();
        let mom = TDMomentum::new(5, 4).unwrap();
        let result = mom.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());

        // Values should be -100 to +100
        for i in 15..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_td_differential() {
        let data = make_test_data();
        let diff = TDDifferential::new(5).unwrap();
        let result = diff.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_validation() {
        assert!(TDCamouflage::new(1).is_err());
        assert!(TDCLOP::new(1).is_err());
        assert!(TDMovingAverageQualifier::new(10, 5).is_err()); // slow <= fast
        assert!(TDRiskLevel::new(2).is_err());
        assert!(TDMomentum::new(1, 1).is_err());
        assert!(TDDifferential::new(1).is_err());
    }
}
