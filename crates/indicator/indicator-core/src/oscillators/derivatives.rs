//! Momentum Derivative Indicators
//!
//! Second-order and composite momentum indicators for trading systems.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Price Velocity - First derivative of price
#[derive(Debug, Clone)]
pub struct PriceVelocity {
    period: usize,
}

impl PriceVelocity {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate price velocity (rate of change)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            result[i] = (close[i] - close[i - self.period]) / self.period as f64;
        }
        result
    }
}

impl TechnicalIndicator for PriceVelocity {
    fn name(&self) -> &str {
        "Price Velocity"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Price Acceleration - Second derivative of price
#[derive(Debug, Clone)]
pub struct PriceAcceleration {
    period: usize,
}

impl PriceAcceleration {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate price acceleration (change in velocity)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First calculate velocity
        let mut velocity = vec![0.0; n];
        for i in self.period..n {
            velocity[i] = close[i] - close[i - self.period];
        }

        // Then calculate acceleration (change in velocity)
        for i in (2 * self.period)..n {
            result[i] = velocity[i] - velocity[i - self.period];
        }
        result
    }
}

impl TechnicalIndicator for PriceAcceleration {
    fn name(&self) -> &str {
        "Price Acceleration"
    }

    fn min_periods(&self) -> usize {
        2 * self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Momentum Divergence Index - Measures price/momentum divergence
#[derive(Debug, Clone)]
pub struct MomentumDivergence {
    price_period: usize,
    momentum_period: usize,
}

impl MomentumDivergence {
    pub fn new(price_period: usize, momentum_period: usize) -> Result<Self> {
        if price_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "price_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { price_period, momentum_period })
    }

    /// Calculate divergence between price trend and momentum
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let max_period = self.price_period.max(self.momentum_period);

        for i in max_period..n {
            // Price slope
            let price_change = (close[i] - close[i - self.price_period]) / close[i - self.price_period];

            // Momentum (ROC)
            let momentum = (close[i] - close[i - self.momentum_period]) / close[i - self.momentum_period];

            // Divergence: when signs differ or magnitude differs significantly
            if price_change != 0.0 {
                result[i] = (momentum - price_change) / price_change.abs() * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for MomentumDivergence {
    fn name(&self) -> &str {
        "Momentum Divergence"
    }

    fn min_periods(&self) -> usize {
        self.price_period.max(self.momentum_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Smoothed Rate of Change - EMA-smoothed momentum
#[derive(Debug, Clone)]
pub struct SmoothedROC {
    roc_period: usize,
    smooth_period: usize,
}

impl SmoothedROC {
    pub fn new(roc_period: usize, smooth_period: usize) -> Result<Self> {
        if roc_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "roc_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { roc_period, smooth_period })
    }

    /// Calculate EMA-smoothed ROC
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate raw ROC
        let mut roc = vec![0.0; n];
        for i in self.roc_period..n {
            roc[i] = (close[i] - close[i - self.roc_period]) / close[i - self.roc_period] * 100.0;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in self.roc_period..n {
            if i == self.roc_period {
                result[i] = roc[i];
            } else {
                result[i] = alpha * roc[i] + (1.0 - alpha) * result[i - 1];
            }
        }
        result
    }
}

impl TechnicalIndicator for SmoothedROC {
    fn name(&self) -> &str {
        "Smoothed ROC"
    }

    fn min_periods(&self) -> usize {
        self.roc_period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Comparative Momentum Index - Compares multiple timeframe momentum
#[derive(Debug, Clone)]
pub struct ComparativeMomentum {
    short_period: usize,
    long_period: usize,
}

impl ComparativeMomentum {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
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

    /// Calculate ratio of short to long momentum
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let short_mom = close[i] - close[i - self.short_period];
            let long_mom = close[i] - close[i - self.long_period];

            if long_mom.abs() > 1e-10 {
                result[i] = short_mom / long_mom.abs() * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for ComparativeMomentum {
    fn name(&self) -> &str {
        "Comparative Momentum"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Dynamic Momentum Index - Adaptive momentum with volatility adjustment
#[derive(Debug, Clone)]
pub struct DynamicMomentumIndex {
    base_period: usize,
    volatility_period: usize,
}

impl DynamicMomentumIndex {
    pub fn new(base_period: usize, volatility_period: usize) -> Result<Self> {
        if base_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { base_period, volatility_period })
    }

    /// Calculate DMI with adaptive period based on volatility
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate historical volatility
        let mut volatility = vec![0.0; n];
        for i in self.volatility_period..n {
            let start = i.saturating_sub(self.volatility_period);
            let mean: f64 = close[start..=i].iter().sum::<f64>() / self.volatility_period as f64;
            let var: f64 = close[start..=i].iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.volatility_period as f64;
            volatility[i] = var.sqrt();
        }

        // Calculate average volatility for scaling
        let avg_vol: f64 = volatility.iter().filter(|&&v| v > 0.0).sum::<f64>()
            / volatility.iter().filter(|&&v| v > 0.0).count().max(1) as f64;

        for i in (self.base_period + self.volatility_period)..n {
            // Adaptive period based on volatility ratio
            let vol_ratio = if avg_vol > 0.0 { volatility[i] / avg_vol } else { 1.0 };
            let adaptive_period = ((self.base_period as f64 / vol_ratio.max(0.5).min(2.0)) as usize)
                .max(3)
                .min(i);

            let lookback_idx = i.saturating_sub(adaptive_period);
            if close[lookback_idx] != 0.0 {
                result[i] = (close[i] - close[lookback_idx]) / close[lookback_idx] * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for DynamicMomentumIndex {
    fn name(&self) -> &str {
        "Dynamic Momentum Index"
    }

    fn min_periods(&self) -> usize {
        self.base_period + self.volatility_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Momentum Quality - Measures consistency of momentum
#[derive(Debug, Clone)]
pub struct MomentumQuality {
    period: usize,
}

impl MomentumQuality {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate momentum quality (consistency of direction)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut positive_days = 0;
            let mut negative_days = 0;
            let mut sum_positive = 0.0;
            let mut sum_negative = 0.0;

            for j in (start + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    positive_days += 1;
                    sum_positive += change;
                } else if change < 0.0 {
                    negative_days += 1;
                    sum_negative += change.abs();
                }
            }

            let total_days = positive_days + negative_days;
            if total_days > 0 {
                // Quality: how consistent the direction is (0-100)
                let direction_consistency = (positive_days as f64 - negative_days as f64).abs()
                    / total_days as f64 * 100.0;

                // Sign indicates net direction
                let net_direction = if sum_positive > sum_negative { 1.0 } else { -1.0 };
                result[i] = net_direction * direction_consistency;
            }
        }
        result
    }
}

impl TechnicalIndicator for MomentumQuality {
    fn name(&self) -> &str {
        "Momentum Quality"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Normalized Momentum - Z-score normalized momentum
#[derive(Debug, Clone)]
pub struct NormalizedMomentum {
    momentum_period: usize,
    norm_period: usize,
}

impl NormalizedMomentum {
    pub fn new(momentum_period: usize, norm_period: usize) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if norm_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "norm_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { momentum_period, norm_period })
    }

    /// Calculate z-score normalized momentum
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate raw momentum
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            momentum[i] = close[i] - close[i - self.momentum_period];
        }

        // Normalize with z-score
        for i in (self.momentum_period + self.norm_period)..n {
            let start = i.saturating_sub(self.norm_period);
            let slice = &momentum[start..=i];

            let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
            let var: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slice.len() as f64;
            let std = var.sqrt();

            if std > 1e-10 {
                result[i] = (momentum[i] - mean) / std;
            }
        }
        result
    }
}

impl TechnicalIndicator for NormalizedMomentum {
    fn name(&self) -> &str {
        "Normalized Momentum"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.norm_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
             107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
             114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0]
    }

    #[test]
    fn test_price_velocity() {
        let close = make_test_data();
        let pv = PriceVelocity::new(5).unwrap();
        let result = pv.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[10] > 0.0); // Uptrend
    }

    #[test]
    fn test_price_acceleration() {
        let close = make_test_data();
        let pa = PriceAcceleration::new(3).unwrap();
        let result = pa.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_momentum_divergence() {
        let close = make_test_data();
        let md = MomentumDivergence::new(5, 10).unwrap();
        let result = md.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_smoothed_roc() {
        let close = make_test_data();
        let sroc = SmoothedROC::new(5, 3).unwrap();
        let result = sroc.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0); // Uptrend
    }

    #[test]
    fn test_comparative_momentum() {
        let close = make_test_data();
        let cm = ComparativeMomentum::new(5, 15).unwrap();
        let result = cm.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_dynamic_momentum_index() {
        let close = make_test_data();
        let dmi = DynamicMomentumIndex::new(5, 10).unwrap();
        let result = dmi.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[20] > 0.0); // Uptrend
    }

    #[test]
    fn test_momentum_quality() {
        let close = make_test_data();
        let mq = MomentumQuality::new(10).unwrap();
        let result = mq.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[20].abs() <= 100.0);
    }

    #[test]
    fn test_normalized_momentum() {
        let close = make_test_data();
        let nm = NormalizedMomentum::new(5, 15).unwrap();
        let result = nm.calculate(&close);

        assert_eq!(result.len(), close.len());
    }
}
