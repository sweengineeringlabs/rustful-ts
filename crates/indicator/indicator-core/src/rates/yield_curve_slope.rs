//! Yield Curve Slope Indicator (IND-296)
//!
//! Calculates the yield curve slope as the spread between long-term
//! and short-term rates (typically 10Y - 2Y spread).

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Yield Curve Slope - 10Y minus 2Y spread proxy
///
/// This indicator calculates the yield curve slope using a price-based
/// proxy. In practice, this would use actual yield data, but here we
/// simulate it using price momentum differentials as a proxy for
/// different maturity rate changes.
///
/// # Interpretation
/// - Positive slope: Normal yield curve (economy healthy)
/// - Flat (near 0): Potential slowdown warning
/// - Negative (inverted): Recession warning signal
///
/// # Note
/// For actual yield curve analysis, feed in yield data directly
/// where close = long-term yield and use calculate_spread method
/// with short-term yields.
#[derive(Debug, Clone)]
pub struct YieldCurveSlope {
    /// Period for rate of change calculation
    period: usize,
    /// Smoothing period
    smooth_period: usize,
    /// Long-term proxy period (simulates 10Y behavior)
    long_term_period: usize,
    /// Short-term proxy period (simulates 2Y behavior)
    short_term_period: usize,
}

impl YieldCurveSlope {
    /// Create a new YieldCurveSlope indicator
    ///
    /// # Arguments
    /// * `period` - Base calculation period (minimum 5)
    /// * `smooth_period` - EMA smoothing period (minimum 2)
    /// * `long_term_period` - Long-term lookback (default 50, simulates 10Y)
    /// * `short_term_period` - Short-term lookback (default 10, simulates 2Y)
    pub fn new(
        period: usize,
        smooth_period: usize,
        long_term_period: usize,
        short_term_period: usize,
    ) -> Result<Self> {
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
        if long_term_period <= short_term_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_term_period".to_string(),
                reason: "must be greater than short_term_period".to_string(),
            });
        }
        if short_term_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_term_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            period,
            smooth_period,
            long_term_period,
            short_term_period,
        })
    }

    /// Create with default parameters (standard 10Y-2Y proxy)
    pub fn default_params() -> Result<Self> {
        Self::new(10, 5, 50, 10)
    }

    /// Calculate yield curve slope using price momentum proxy
    ///
    /// This simulates yield curve behavior using momentum differentials.
    /// For actual yields, use calculate_spread method.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.long_term_period {
            return result;
        }

        // Calculate long-term and short-term momentum as yield proxies
        for i in self.long_term_period..n {
            // Long-term momentum (10Y proxy)
            let long_mom = if close[i - self.long_term_period] > 1e-10 {
                (close[i] / close[i - self.long_term_period]).ln() * 100.0
            } else {
                0.0
            };

            // Short-term momentum (2Y proxy)
            let short_mom = if close[i - self.short_term_period] > 1e-10 {
                (close[i] / close[i - self.short_term_period]).ln() * 100.0
            } else {
                0.0
            };

            // Slope = long-term - short-term (normalized)
            // Positive = normal curve, Negative = inverted
            let slope = long_mom - short_mom * (self.long_term_period as f64 / self.short_term_period as f64);
            result[i] = slope;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate actual yield curve spread from yield data
    ///
    /// # Arguments
    /// * `long_yields` - Long-term yields (e.g., 10Y Treasury)
    /// * `short_yields` - Short-term yields (e.g., 2Y Treasury)
    ///
    /// # Returns
    /// Spread in basis points (long - short)
    pub fn calculate_spread(&self, long_yields: &[f64], short_yields: &[f64]) -> Vec<f64> {
        let n = long_yields.len().min(short_yields.len());
        let mut result = vec![0.0; n];

        for i in 0..n {
            // Spread in basis points (yields assumed to be in percentage)
            result[i] = (long_yields[i] - short_yields[i]) * 100.0;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Check if yield curve is inverted
    pub fn is_inverted(&self, spread: f64) -> bool {
        spread < 0.0
    }

    /// Check if yield curve is flat
    pub fn is_flat(&self, spread: f64, threshold: f64) -> bool {
        spread.abs() < threshold
    }
}

impl TechnicalIndicator for YieldCurveSlope {
    fn name(&self) -> &str {
        "Yield Curve Slope"
    }

    fn min_periods(&self) -> usize {
        self.long_term_period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let n = 80;
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.2 + (i as f64 * 0.1).sin() * 3.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; n];

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_yield_curve_slope_basic() {
        let data = make_test_data();
        let indicator = YieldCurveSlope::new(10, 5, 50, 10).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_yield_curve_slope_spread() {
        // Simulate yield data
        let long_yields: Vec<f64> = (0..50).map(|i| 3.0 + (i as f64) * 0.01).collect();
        let short_yields: Vec<f64> = (0..50).map(|i| 2.5 + (i as f64) * 0.015).collect();

        let indicator = YieldCurveSlope::new(10, 5, 50, 10).unwrap();
        let spread = indicator.calculate_spread(&long_yields, &short_yields);

        assert_eq!(spread.len(), 50);
        // Initially positive spread (normal curve)
        assert!(spread[10] > 0.0);
    }

    #[test]
    fn test_yield_curve_slope_inverted() {
        let indicator = YieldCurveSlope::default_params().unwrap();

        assert!(indicator.is_inverted(-50.0));
        assert!(!indicator.is_inverted(50.0));
        assert!(!indicator.is_inverted(0.0));
    }

    #[test]
    fn test_yield_curve_slope_flat() {
        let indicator = YieldCurveSlope::default_params().unwrap();

        assert!(indicator.is_flat(5.0, 10.0));
        assert!(indicator.is_flat(-5.0, 10.0));
        assert!(!indicator.is_flat(50.0, 10.0));
    }

    #[test]
    fn test_yield_curve_slope_technical_indicator_trait() {
        let data = make_test_data();
        let indicator = YieldCurveSlope::new(10, 5, 50, 10).unwrap();

        assert_eq!(indicator.name(), "Yield Curve Slope");
        assert_eq!(indicator.min_periods(), 55);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.values.is_empty());
    }

    #[test]
    fn test_yield_curve_slope_parameter_validation() {
        assert!(YieldCurveSlope::new(3, 5, 50, 10).is_err()); // period too small
        assert!(YieldCurveSlope::new(10, 1, 50, 10).is_err()); // smooth_period too small
        assert!(YieldCurveSlope::new(10, 5, 10, 50).is_err()); // long <= short
        assert!(YieldCurveSlope::new(10, 5, 50, 3).is_err()); // short_term too small
    }

    #[test]
    fn test_yield_curve_slope_default_params() {
        let indicator = YieldCurveSlope::default_params().unwrap();
        assert_eq!(indicator.period, 10);
        assert_eq!(indicator.smooth_period, 5);
        assert_eq!(indicator.long_term_period, 50);
        assert_eq!(indicator.short_term_period, 10);
    }
}
