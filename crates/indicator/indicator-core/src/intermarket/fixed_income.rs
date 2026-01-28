//! Fixed Income Indicators
//!
//! Indicators for analyzing bond and fixed income markets.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Yield Curve Shape - Measures steepness and curvature
/// Uses price-based slope approximation
#[derive(Debug, Clone)]
pub struct YieldCurveShape {
    short_period: usize,
    long_period: usize,
}

impl YieldCurveShape {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 || long_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "periods must be at least 2".to_string(),
            });
        }
        if short_period >= long_period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than long_period".to_string(),
            });
        }
        Ok(Self { short_period, long_period })
    }

    /// Calculate yield curve shape (steepness proxy)
    /// Positive = Normal curve, Negative = Inverted
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term momentum
            let short_ret = (close[i] / close[i - self.short_period]).ln();
            // Long-term momentum
            let long_ret = (close[i] / close[i - self.long_period]).ln();

            // Slope = long-term vs short-term return difference
            result[i] = (short_ret - long_ret / (self.long_period as f64 / self.short_period as f64))
                * 100.0;
        }
        result
    }
}

impl TechnicalIndicator for YieldCurveShape {
    fn name(&self) -> &str {
        "Yield Curve Shape"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Butterfly Spread (2s5s10s equivalent)
/// Measures curvature of term structure
#[derive(Debug, Clone)]
pub struct ButterflySpread {
    short_period: usize,
    mid_period: usize,
    long_period: usize,
}

impl ButterflySpread {
    pub fn new(short_period: usize, mid_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 || mid_period < 2 || long_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "all periods must be at least 2".to_string(),
            });
        }
        if short_period >= mid_period || mid_period >= long_period {
            return Err(IndicatorError::InvalidParameter {
                name: "periods".to_string(),
                reason: "must be in ascending order".to_string(),
            });
        }
        Ok(Self { short_period, mid_period, long_period })
    }

    /// Calculate butterfly spread
    /// Positive = Curve is convex, Negative = Curve is concave
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            let short_ret = (close[i] / close[i - self.short_period]).ln()
                / self.short_period as f64;
            let mid_ret = (close[i] / close[i - self.mid_period]).ln()
                / self.mid_period as f64;
            let long_ret = (close[i] / close[i - self.long_period]).ln()
                / self.long_period as f64;

            // Butterfly = 2 * mid - short - long (curvature)
            result[i] = (2.0 * mid_ret - short_ret - long_ret) * 10000.0;
        }
        result
    }
}

impl TechnicalIndicator for ButterflySpread {
    fn name(&self) -> &str {
        "Butterfly Spread"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Breakeven Inflation Proxy
/// Uses volatility and momentum as inflation expectations proxy
#[derive(Debug, Clone)]
pub struct BreakevenInflation {
    period: usize,
}

impl BreakevenInflation {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate breakeven inflation proxy
    /// Uses trend + volatility component
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
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
            let vol = variance.sqrt();

            // Breakeven proxy: annualized trend + volatility premium
            let annualized_ret = mean_ret * 252.0;
            result[i] = (annualized_ret + vol * 252.0_f64.sqrt() * 0.5) * 100.0;
        }
        result
    }
}

impl TechnicalIndicator for BreakevenInflation {
    fn name(&self) -> &str {
        "Breakeven Inflation"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Real Rate Proxy
/// Estimated from price dynamics
#[derive(Debug, Clone)]
pub struct RealRate {
    period: usize,
}

impl RealRate {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate real rate proxy
    /// Uses volatility-adjusted returns
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
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
            let vol = variance.sqrt();

            // Real rate proxy: return - volatility component
            let annualized_ret = mean_ret * 252.0;
            let vol_component = vol * 252.0_f64.sqrt() * 0.5;
            result[i] = (annualized_ret - vol_component) * 100.0;
        }
        result
    }
}

impl TechnicalIndicator for RealRate {
    fn name(&self) -> &str {
        "Real Rate"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Effective Duration - Price sensitivity measure
#[derive(Debug, Clone)]
pub struct EffectiveDuration {
    period: usize,
}

impl EffectiveDuration {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate effective duration proxy
    /// Uses price sensitivity to yield changes
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate price changes and simulate yield sensitivity
            let mut sum_sensitivity = 0.0;
            let mut count = 0;

            for j in (start + 1)..=i {
                let price_change = (close[j] - close[j - 1]) / close[j - 1];
                // Approximate yield change from price change (inverse relationship)
                // Duration = -1/P * dP/dy, approximated here
                if price_change != 0.0 {
                    sum_sensitivity += (1.0 / price_change.abs()).min(50.0);
                    count += 1;
                }
            }

            if count > 0 {
                result[i] = sum_sensitivity / count as f64;
            }
        }
        result
    }
}

impl TechnicalIndicator for EffectiveDuration {
    fn name(&self) -> &str {
        "Effective Duration"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Key Rate Duration - Partial duration sensitivities
#[derive(Debug, Clone)]
pub struct KeyRateDuration {
    periods: Vec<usize>,
}

impl KeyRateDuration {
    pub fn new(periods: Vec<usize>) -> Result<Self> {
        if periods.is_empty() || periods.len() > 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "periods".to_string(),
                reason: "must have 1-5 key rates".to_string(),
            });
        }
        for &p in &periods {
            if p < 2 {
                return Err(IndicatorError::InvalidParameter {
                    name: "period".to_string(),
                    reason: "all periods must be at least 2".to_string(),
                });
            }
        }
        Ok(Self { periods })
    }

    /// Calculate key rate durations for each tenor
    /// Returns weighted average of sensitivities
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let max_period = *self.periods.iter().max().unwrap_or(&10);
        let mut result = vec![0.0; n];

        for i in max_period..n {
            let mut total_duration = 0.0;
            let mut weight_sum = 0.0;

            for &period in &self.periods {
                if i >= period {
                    let ret = (close[i] / close[i - period]).ln() / period as f64;
                    let sensitivity = if ret != 0.0 {
                        (1.0 / ret.abs()).min(100.0)
                    } else {
                        0.0
                    };
                    // Weight by period (longer maturities have more impact)
                    let weight = (period as f64).sqrt();
                    total_duration += sensitivity * weight;
                    weight_sum += weight;
                }
            }

            if weight_sum > 0.0 {
                result[i] = total_duration / weight_sum;
            }
        }
        result
    }
}

impl TechnicalIndicator for KeyRateDuration {
    fn name(&self) -> &str {
        "Key Rate Duration"
    }

    fn min_periods(&self) -> usize {
        *self.periods.iter().max().unwrap_or(&10) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        vec![100.0, 100.5, 101.0, 100.8, 101.5, 102.0, 101.5, 102.5, 103.0, 102.5,
             103.5, 104.0, 103.5, 104.5, 105.0, 104.5, 105.5, 106.0, 105.5, 106.5,
             107.0, 106.5, 107.5, 108.0, 107.5, 108.5, 109.0, 108.5, 109.5, 110.0]
    }

    #[test]
    fn test_yield_curve_shape() {
        let close = make_test_data();
        let ycs = YieldCurveShape::new(5, 20).unwrap();
        let result = ycs.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[25].abs() < 100.0);
    }

    #[test]
    fn test_butterfly_spread() {
        let close = make_test_data();
        let bs = ButterflySpread::new(5, 10, 20).unwrap();
        let result = bs.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[25].abs() < 1000.0);
    }

    #[test]
    fn test_breakeven_inflation() {
        let close = make_test_data();
        let bei = BreakevenInflation::new(10).unwrap();
        let result = bei.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Uptrend should show positive breakeven
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_real_rate() {
        let close = make_test_data();
        let rr = RealRate::new(10).unwrap();
        let result = rr.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15].abs() < 100.0);
    }

    #[test]
    fn test_effective_duration() {
        let close = make_test_data();
        let ed = EffectiveDuration::new(10).unwrap();
        let result = ed.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Duration should be positive
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_key_rate_duration() {
        let close = make_test_data();
        let krd = KeyRateDuration::new(vec![5, 10, 20]).unwrap();
        let result = krd.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Duration should be positive
        assert!(result[25] > 0.0);
    }
}
