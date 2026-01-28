//! Extended Swing Indicators
//!
//! Additional swing trading indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Swing Momentum - Momentum of swing moves
#[derive(Debug, Clone)]
pub struct SwingMomentum {
    period: usize,
}

impl SwingMomentum {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate swing momentum
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Find highest high and lowest low in period
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            if range > 1e-10 {
                // Current position relative to range
                let position = (close[i] - period_low) / range;

                // Momentum based on position change
                let prev_position = if i > self.period {
                    let prev_high = high[(start - 1)..i].iter().cloned().fold(f64::MIN, f64::max);
                    let prev_low = low[(start - 1)..i].iter().cloned().fold(f64::MAX, f64::min);
                    let prev_range = prev_high - prev_low;
                    if prev_range > 1e-10 {
                        (close[i - 1] - prev_low) / prev_range
                    } else {
                        0.5
                    }
                } else {
                    0.5
                };

                result[i] = (position - prev_position) * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for SwingMomentum {
    fn name(&self) -> &str {
        "Swing Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Swing Range - Average swing range indicator
#[derive(Debug, Clone)]
pub struct SwingRange {
    period: usize,
}

impl SwingRange {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate swing range as percentage
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate range as percentage of average price
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            let avg_close = close[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;

            if avg_close > 1e-10 {
                result[i] = range / avg_close * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for SwingRange {
    fn name(&self) -> &str {
        "Swing Range"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Swing Direction - Identifies swing direction changes
#[derive(Debug, Clone)]
pub struct SwingDirection {
    period: usize,
}

impl SwingDirection {
    pub fn new(period: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate swing direction (1 = up, -1 = down, 0 = neutral)
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![0.0; n];
        let mut current_direction = 0.0;
        let mut swing_high = if n > 0 { high[0] } else { 0.0 };
        let mut swing_low = if n > 0 { low[0] } else { 0.0 };

        for i in self.period..n {
            let period_high = high[i - self.period..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[i - self.period..=i].iter().cloned().fold(f64::MAX, f64::min);

            if high[i] == period_high && high[i] > swing_high {
                swing_high = high[i];
                current_direction = 1.0;
            } else if low[i] == period_low && low[i] < swing_low {
                swing_low = low[i];
                current_direction = -1.0;
            }

            result[i] = current_direction;
        }
        result
    }
}

impl TechnicalIndicator for SwingDirection {
    fn name(&self) -> &str {
        "Swing Direction"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low)))
    }
}

/// Swing Velocity - Speed of swing moves
#[derive(Debug, Clone)]
pub struct SwingVelocity {
    period: usize,
}

impl SwingVelocity {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate swing velocity (price change per bar)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            // Price change over period
            let price_change = close[i] - close[i - self.period];

            // Velocity = change / period
            let velocity = price_change / self.period as f64;

            // Normalize by price
            if close[i - self.period] > 1e-10 {
                result[i] = velocity / close[i - self.period] * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for SwingVelocity {
    fn name(&self) -> &str {
        "Swing Velocity"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Swing Strength - Strength of current swing
#[derive(Debug, Clone)]
pub struct SwingStrength {
    period: usize,
}

impl SwingStrength {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate swing strength (0 to 100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate range
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            if range > 1e-10 {
                // Count consecutive moves in same direction
                let mut up_count = 0;
                let mut down_count = 0;
                for j in (start + 1)..=i {
                    if close[j] > close[j - 1] {
                        up_count += 1;
                    } else if close[j] < close[j - 1] {
                        down_count += 1;
                    }
                }

                // Strength based on consistency and position
                let consistency = (up_count as i32 - down_count as i32).abs() as f64 / self.period as f64;
                let position = (close[i] - period_low) / range;

                // Combine: strong swing has consistent moves and extreme position
                let direction = if up_count > down_count { 1.0 } else { -1.0 };
                let strength = consistency * 50.0 + (position - 0.5).abs() * 100.0;

                result[i] = (strength * direction).clamp(-100.0, 100.0);
            }
        }
        result
    }
}

impl TechnicalIndicator for SwingStrength {
    fn name(&self) -> &str {
        "Swing Strength"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Swing Failure Pattern - Detects swing failure patterns
#[derive(Debug, Clone)]
pub struct SwingFailurePattern {
    period: usize,
}

impl SwingFailurePattern {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Detect swing failure patterns (1 = bullish failure, -1 = bearish failure)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in (self.period * 2)..n {
            let mid = i - self.period;
            let start = mid - self.period;

            // Find swing high/low in first period
            let first_high = high[start..=mid].iter().cloned().fold(f64::MIN, f64::max);
            let first_low = low[start..=mid].iter().cloned().fold(f64::MAX, f64::min);

            // Find swing high/low in second period
            let second_high = high[mid..=i].iter().cloned().fold(f64::MIN, f64::max);
            let second_low = low[mid..=i].iter().cloned().fold(f64::MAX, f64::min);

            // Bearish swing failure: higher high but close below first high
            if second_high > first_high && close[i] < first_high {
                result[i] = -1.0;
            }
            // Bullish swing failure: lower low but close above first low
            else if second_low < first_low && close[i] > first_low {
                result[i] = 1.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for SwingFailurePattern {
    fn name(&self) -> &str {
        "Swing Failure Pattern"
    }

    fn min_periods(&self) -> usize {
        self.period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![102.0, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0, 108.0, 107.5,
                       109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0, 114.0, 113.5, 115.0,
                       116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0, 119.5, 121.0, 122.0];
        let low = vec![98.0, 99.0, 100.0, 99.5, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5,
                      105.0, 106.0, 105.5, 107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0,
                      112.0, 111.5, 113.0, 114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0];
        let close = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
                        107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
                        114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0];
        (high, low, close)
    }

    #[test]
    fn test_swing_momentum() {
        let (high, low, close) = make_test_data();
        let sm = SwingMomentum::new(10).unwrap();
        let result = sm.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_swing_range() {
        let (high, low, close) = make_test_data();
        let sr = SwingRange::new(10).unwrap();
        let result = sr.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= 0.0);
    }

    #[test]
    fn test_swing_direction() {
        let (high, low, _) = make_test_data();
        let sd = SwingDirection::new(5).unwrap();
        let result = sd.calculate(&high, &low);

        assert_eq!(result.len(), high.len());
    }

    #[test]
    fn test_swing_velocity() {
        let (_, _, close) = make_test_data();
        let sv = SwingVelocity::new(10).unwrap();
        let result = sv.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_swing_strength() {
        let (high, low, close) = make_test_data();
        let ss = SwingStrength::new(10).unwrap();
        let result = ss.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_swing_failure_pattern() {
        let (high, low, close) = make_test_data();
        let sfp = SwingFailurePattern::new(5).unwrap();
        let result = sfp.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
    }
}
