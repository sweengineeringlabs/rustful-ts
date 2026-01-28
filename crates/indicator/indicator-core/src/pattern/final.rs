//! Final Pattern Indicators
//!
//! Additional pattern recognition indicators to complete the 300-indicator milestone.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Price Momentum Pattern - Detects momentum-based price patterns
#[derive(Debug, Clone)]
pub struct PriceMomentumPattern {
    momentum_period: usize,
    pattern_threshold: f64,
}

impl PriceMomentumPattern {
    pub fn new(momentum_period: usize, pattern_threshold: f64) -> Result<Self> {
        if momentum_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if pattern_threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "pattern_threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { momentum_period, pattern_threshold })
    }

    /// Detect momentum patterns: +1 = bullish, -1 = bearish, 0 = neutral
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in (2 * self.momentum_period)..n {
            // Current momentum
            let current_mom = if close[i - self.momentum_period] > 1e-10 {
                (close[i] / close[i - self.momentum_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Previous momentum
            let prev_mom = if close[i - 2 * self.momentum_period] > 1e-10 {
                (close[i - self.momentum_period] / close[i - 2 * self.momentum_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Pattern detection
            let mom_change = current_mom - prev_mom;

            // Bullish: momentum accelerating upward
            if current_mom > self.pattern_threshold && mom_change > 0.0 {
                result[i] = 1.0;
            }
            // Bearish: momentum accelerating downward
            else if current_mom < -self.pattern_threshold && mom_change < 0.0 {
                result[i] = -1.0;
            }
            // Reversal signals
            else if prev_mom < -self.pattern_threshold && current_mom > 0.0 {
                result[i] = 0.5; // Bullish reversal
            }
            else if prev_mom > self.pattern_threshold && current_mom < 0.0 {
                result[i] = -0.5; // Bearish reversal
            }
        }

        result
    }
}

impl TechnicalIndicator for PriceMomentumPattern {
    fn name(&self) -> &str {
        "Price Momentum Pattern"
    }

    fn min_periods(&self) -> usize {
        2 * self.momentum_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Range Contraction Expansion - Detects volatility contraction/expansion patterns
#[derive(Debug, Clone)]
pub struct RangeContractionExpansion {
    period: usize,
    expansion_mult: f64,
}

impl RangeContractionExpansion {
    pub fn new(period: usize, expansion_mult: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if expansion_mult <= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "expansion_mult".to_string(),
                reason: "must be greater than 1.0".to_string(),
            });
        }
        Ok(Self { period, expansion_mult })
    }

    /// Detect range patterns: >0 = expansion, <0 = contraction, magnitude = intensity
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate average range
            let mut avg_range = 0.0;
            for j in start..i {
                avg_range += high[j] - low[j];
            }
            avg_range /= self.period as f64;

            // Current range
            let current_range = high[i] - low[i];

            if avg_range > 1e-10 {
                let range_ratio = current_range / avg_range;

                // Expansion
                if range_ratio > self.expansion_mult {
                    result[i] = (range_ratio - 1.0) * 100.0;
                }
                // Contraction
                else if range_ratio < (1.0 / self.expansion_mult) {
                    result[i] = -(1.0 - range_ratio) * 100.0;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for RangeContractionExpansion {
    fn name(&self) -> &str {
        "Range Contraction Expansion"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64) * 0.3 + (i as f64 * 0.4).sin() * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; 40];

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_price_momentum_pattern() {
        let data = make_test_data();
        let pmp = PriceMomentumPattern::new(5, 1.0).unwrap();
        let result = pmp.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Pattern values should be in [-1, 1] range
        for val in &result {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_range_contraction_expansion() {
        let data = make_test_data();
        let rce = RangeContractionExpansion::new(10, 1.5).unwrap();
        let result = rce.calculate(&data.high, &data.low);

        assert_eq!(result.len(), data.high.len());
    }

    #[test]
    fn test_validation() {
        assert!(PriceMomentumPattern::new(1, 1.0).is_err());
        assert!(PriceMomentumPattern::new(5, 0.0).is_err());
        assert!(RangeContractionExpansion::new(2, 1.5).is_err());
        assert!(RangeContractionExpansion::new(5, 0.5).is_err()); // mult <= 1.0
    }
}
