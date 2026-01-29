//! Yield Curve Curvature Indicator (IND-297)
//!
//! Calculates the yield curve curvature (butterfly spread) using the
//! relationship between short, medium, and long-term rates.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Yield Curve Curvature - Butterfly spread proxy
///
/// This indicator measures yield curve curvature using the butterfly spread:
/// Curvature = 2 * Mid-term - Short-term - Long-term
///
/// The butterfly spread measures how much the middle of the yield curve
/// deviates from a straight line between short and long ends.
///
/// # Interpretation
/// - Positive curvature: Humped curve (mid-term rates relatively high)
/// - Zero curvature: Flat/linear curve
/// - Negative curvature: U-shaped curve (mid-term rates relatively low)
///
/// # Note
/// For actual yield analysis, use calculate_butterfly with real yield data.
#[derive(Debug, Clone)]
pub struct YieldCurveCurvature {
    /// Short-term momentum period (2Y proxy)
    short_period: usize,
    /// Medium-term momentum period (5Y proxy)
    mid_period: usize,
    /// Long-term momentum period (10Y proxy)
    long_period: usize,
    /// Smoothing period
    smooth_period: usize,
}

impl YieldCurveCurvature {
    /// Create a new YieldCurveCurvature indicator
    ///
    /// # Arguments
    /// * `short_period` - Short-term lookback (minimum 5)
    /// * `mid_period` - Medium-term lookback (must be > short)
    /// * `long_period` - Long-term lookback (must be > mid)
    /// * `smooth_period` - EMA smoothing period (minimum 2)
    pub fn new(
        short_period: usize,
        mid_period: usize,
        long_period: usize,
        smooth_period: usize,
    ) -> Result<Self> {
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if mid_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "mid_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if long_period <= mid_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than mid_period".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            short_period,
            mid_period,
            long_period,
            smooth_period,
        })
    }

    /// Create with default parameters (2Y-5Y-10Y proxy)
    pub fn default_params() -> Result<Self> {
        Self::new(10, 25, 50, 5)
    }

    /// Calculate yield curve curvature using price momentum proxy
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.long_period {
            return result;
        }

        for i in self.long_period..n {
            // Short-term momentum (2Y proxy)
            let short_mom = if close[i - self.short_period] > 1e-10 {
                (close[i] / close[i - self.short_period]).ln() * 100.0
            } else {
                0.0
            };

            // Medium-term momentum (5Y proxy)
            let mid_mom = if close[i - self.mid_period] > 1e-10 {
                (close[i] / close[i - self.mid_period]).ln() * 100.0
            } else {
                0.0
            };

            // Long-term momentum (10Y proxy)
            let long_mom = if close[i - self.long_period] > 1e-10 {
                (close[i] / close[i - self.long_period]).ln() * 100.0
            } else {
                0.0
            };

            // Normalize by periods for fair comparison
            let short_norm = short_mom / self.short_period as f64;
            let mid_norm = mid_mom / self.mid_period as f64;
            let long_norm = long_mom / self.long_period as f64;

            // Butterfly spread: 2 * mid - short - long
            // Scaled for readability
            result[i] = (2.0 * mid_norm - short_norm - long_norm) * 1000.0;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate actual butterfly spread from yield data
    ///
    /// # Arguments
    /// * `short_yields` - Short-term yields (e.g., 2Y Treasury)
    /// * `mid_yields` - Medium-term yields (e.g., 5Y Treasury)
    /// * `long_yields` - Long-term yields (e.g., 10Y Treasury)
    ///
    /// # Returns
    /// Butterfly spread in basis points: 2*mid - short - long
    pub fn calculate_butterfly(
        &self,
        short_yields: &[f64],
        mid_yields: &[f64],
        long_yields: &[f64],
    ) -> Vec<f64> {
        let n = short_yields.len().min(mid_yields.len()).min(long_yields.len());
        let mut result = vec![0.0; n];

        for i in 0..n {
            // Butterfly spread in basis points
            result[i] = (2.0 * mid_yields[i] - short_yields[i] - long_yields[i]) * 100.0;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Classify the curvature shape
    pub fn classify_shape(&self, curvature: f64, threshold: f64) -> CurveShape {
        if curvature > threshold {
            CurveShape::Humped
        } else if curvature < -threshold {
            CurveShape::UShaped
        } else {
            CurveShape::Flat
        }
    }
}

/// Yield curve shape classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurveShape {
    /// Humped curve (positive curvature)
    Humped,
    /// Flat/linear curve
    Flat,
    /// U-shaped curve (negative curvature)
    UShaped,
}

impl TechnicalIndicator for YieldCurveCurvature {
    fn name(&self) -> &str {
        "Yield Curve Curvature"
    }

    fn min_periods(&self) -> usize {
        self.long_period + self.smooth_period
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
            .map(|i| 100.0 + (i as f64) * 0.2 + (i as f64 * 0.15).sin() * 2.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; n];

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_yield_curve_curvature_basic() {
        let data = make_test_data();
        let indicator = YieldCurveCurvature::new(10, 25, 50, 5).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_yield_curve_curvature_butterfly() {
        // Simulate normal curve with slight hump
        let short_yields: Vec<f64> = vec![2.0; 50];
        let mid_yields: Vec<f64> = vec![2.8; 50]; // Higher than interpolation
        let long_yields: Vec<f64> = vec![3.0; 50];

        let indicator = YieldCurveCurvature::new(10, 25, 50, 5).unwrap();
        let butterfly = indicator.calculate_butterfly(&short_yields, &mid_yields, &long_yields);

        assert_eq!(butterfly.len(), 50);
        // 2 * 2.8 - 2.0 - 3.0 = 0.6 -> positive (humped)
        // With 100x scaling: 60 bps
        assert!(butterfly[30] > 0.0);
    }

    #[test]
    fn test_yield_curve_curvature_u_shaped() {
        // U-shaped curve (mid-term lower than interpolation)
        let short_yields: Vec<f64> = vec![3.0; 50];
        let mid_yields: Vec<f64> = vec![2.5; 50]; // Lower than expected
        let long_yields: Vec<f64> = vec![3.5; 50];

        let indicator = YieldCurveCurvature::new(10, 25, 50, 5).unwrap();
        let butterfly = indicator.calculate_butterfly(&short_yields, &mid_yields, &long_yields);

        // 2 * 2.5 - 3.0 - 3.5 = -1.5 -> negative (U-shaped)
        assert!(butterfly[30] < 0.0);
    }

    #[test]
    fn test_yield_curve_curvature_classify_shape() {
        let indicator = YieldCurveCurvature::default_params().unwrap();

        assert_eq!(indicator.classify_shape(20.0, 10.0), CurveShape::Humped);
        assert_eq!(indicator.classify_shape(-20.0, 10.0), CurveShape::UShaped);
        assert_eq!(indicator.classify_shape(5.0, 10.0), CurveShape::Flat);
        assert_eq!(indicator.classify_shape(-5.0, 10.0), CurveShape::Flat);
    }

    #[test]
    fn test_yield_curve_curvature_technical_indicator_trait() {
        let data = make_test_data();
        let indicator = YieldCurveCurvature::new(10, 25, 50, 5).unwrap();

        assert_eq!(indicator.name(), "Yield Curve Curvature");
        assert_eq!(indicator.min_periods(), 55);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.values.is_empty());
    }

    #[test]
    fn test_yield_curve_curvature_parameter_validation() {
        assert!(YieldCurveCurvature::new(3, 25, 50, 5).is_err()); // short too small
        assert!(YieldCurveCurvature::new(10, 5, 50, 5).is_err()); // mid <= short
        assert!(YieldCurveCurvature::new(10, 25, 20, 5).is_err()); // long <= mid
        assert!(YieldCurveCurvature::new(10, 25, 50, 1).is_err()); // smooth too small
    }

    #[test]
    fn test_yield_curve_curvature_default_params() {
        let indicator = YieldCurveCurvature::default_params().unwrap();
        assert_eq!(indicator.short_period, 10);
        assert_eq!(indicator.mid_period, 25);
        assert_eq!(indicator.long_period, 50);
        assert_eq!(indicator.smooth_period, 5);
    }

    #[test]
    fn test_curve_shape_enum() {
        let humped = CurveShape::Humped;
        let flat = CurveShape::Flat;
        let u_shaped = CurveShape::UShaped;

        assert_ne!(humped, flat);
        assert_ne!(flat, u_shaped);
        assert_ne!(humped, u_shaped);
    }
}
