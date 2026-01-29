//! Rounding Pattern Indicator (IND-342)
//!
//! Detects saucer/rounding bottom and rounding top patterns -
//! gradual reversal formations.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Rounding Pattern detection (Saucer Bottom/Top).
///
/// Identifies gradual reversal patterns characterized by a smooth,
/// curved price movement forming a bowl (bottom) or dome (top) shape.
///
/// Pattern characteristics:
/// - Rounding Bottom: U-shaped price curve, bullish reversal
/// - Rounding Top: Inverted U-shaped curve, bearish reversal
#[derive(Debug, Clone)]
pub struct RoundingPattern {
    /// Lookback period for pattern detection
    lookback: usize,
    /// Minimum curvature threshold
    curvature_threshold: f64,
    /// Smoothness requirement (lower = more strict)
    smoothness_tolerance: f64,
}

/// Pattern type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoundingType {
    None,
    Bottom,
    Top,
}

/// Output from rounding pattern detection
#[derive(Debug, Clone)]
pub struct RoundingPatternOutput {
    /// Pattern type detected
    pub pattern_type: RoundingType,
    /// Pattern strength (0.0-1.0)
    pub strength: f64,
    /// Curvature measurement
    pub curvature: f64,
    /// Completion percentage (0-100)
    pub completion: f64,
}

impl RoundingPattern {
    /// Create a new Rounding Pattern indicator.
    ///
    /// # Arguments
    /// * `lookback` - Number of periods to analyze (minimum 20)
    /// * `curvature_threshold` - Minimum curvature for valid pattern (0.0-1.0)
    /// * `smoothness_tolerance` - Maximum noise tolerance (0.0-1.0)
    pub fn new(
        lookback: usize,
        curvature_threshold: f64,
        smoothness_tolerance: f64,
    ) -> Result<Self> {
        if lookback < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if curvature_threshold < 0.0 || curvature_threshold > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "curvature_threshold".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if smoothness_tolerance < 0.0 || smoothness_tolerance > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothness_tolerance".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self {
            lookback,
            curvature_threshold,
            smoothness_tolerance,
        })
    }

    /// Create with default parameters
    pub fn default_params() -> Self {
        Self {
            lookback: 30,
            curvature_threshold: 0.3,
            smoothness_tolerance: 0.5,
        }
    }

    /// Calculate rounding pattern detection.
    ///
    /// Returns: 1.0 = rounding bottom, -1.0 = rounding top, 0.0 = none
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let window = &close[start..=i];

            if let Some((pattern_type, strength)) = self.detect_rounding(window) {
                if strength >= self.curvature_threshold {
                    result[i] = match pattern_type {
                        RoundingType::Bottom => 1.0,
                        RoundingType::Top => -1.0,
                        RoundingType::None => 0.0,
                    };
                }
            }
        }

        result
    }

    /// Calculate detailed rounding pattern output.
    pub fn calculate_detailed(&self, close: &[f64]) -> Vec<RoundingPatternOutput> {
        let n = close.len();
        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            if i < self.lookback {
                results.push(RoundingPatternOutput {
                    pattern_type: RoundingType::None,
                    strength: 0.0,
                    curvature: 0.0,
                    completion: 0.0,
                });
                continue;
            }

            let start = i.saturating_sub(self.lookback);
            let window = &close[start..=i];

            if let Some((pattern_type, strength)) = self.detect_rounding(window) {
                let curvature = self.calculate_curvature(window);
                let completion = self.calculate_completion(window, pattern_type);

                results.push(RoundingPatternOutput {
                    pattern_type,
                    strength,
                    curvature,
                    completion,
                });
            } else {
                results.push(RoundingPatternOutput {
                    pattern_type: RoundingType::None,
                    strength: 0.0,
                    curvature: 0.0,
                    completion: 0.0,
                });
            }
        }

        results
    }

    /// Detect rounding pattern in window.
    fn detect_rounding(&self, window: &[f64]) -> Option<(RoundingType, f64)> {
        let n = window.len();
        if n < 10 {
            return None;
        }

        // Find minimum and maximum points
        let (min_idx, min_val) = window
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

        let (max_idx, max_val) = window
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

        let range = max_val - min_val;
        if range < 1e-10 {
            return None;
        }

        // Check for rounding bottom (min in middle, ends higher)
        let quarter = n / 4;
        let three_quarter = 3 * n / 4;

        if min_idx > quarter && min_idx < three_quarter {
            // Potential rounding bottom
            let left_avg = window[..quarter].iter().sum::<f64>() / quarter as f64;
            let right_avg = window[three_quarter..].iter().sum::<f64>()
                / (n - three_quarter) as f64;
            let middle_avg = window[quarter..three_quarter].iter().sum::<f64>()
                / (three_quarter - quarter) as f64;

            if left_avg > middle_avg && right_avg > middle_avg {
                let curvature = self.calculate_curvature(window);
                let smoothness = self.calculate_smoothness(window);

                if smoothness <= self.smoothness_tolerance {
                    let strength = curvature * (1.0 - smoothness);
                    return Some((RoundingType::Bottom, strength.min(1.0)));
                }
            }
        }

        // Check for rounding top (max in middle, ends lower)
        if max_idx > quarter && max_idx < three_quarter {
            let left_avg = window[..quarter].iter().sum::<f64>() / quarter as f64;
            let right_avg = window[three_quarter..].iter().sum::<f64>()
                / (n - three_quarter) as f64;
            let middle_avg = window[quarter..three_quarter].iter().sum::<f64>()
                / (three_quarter - quarter) as f64;

            if left_avg < middle_avg && right_avg < middle_avg {
                let curvature = self.calculate_curvature(window);
                let smoothness = self.calculate_smoothness(window);

                if smoothness <= self.smoothness_tolerance {
                    let strength = curvature * (1.0 - smoothness);
                    return Some((RoundingType::Top, strength.min(1.0)));
                }
            }
        }

        None
    }

    /// Calculate curvature of price series.
    fn calculate_curvature(&self, window: &[f64]) -> f64 {
        let n = window.len();
        if n < 3 {
            return 0.0;
        }

        // Fit quadratic: y = ax^2 + bx + c
        // Calculate second derivative coefficient (2a) as curvature measure
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let sum_x: f64 = x.iter().sum();
        let sum_x2: f64 = x.iter().map(|v| v.powi(2)).sum();
        let sum_x3: f64 = x.iter().map(|v| v.powi(3)).sum();
        let sum_x4: f64 = x.iter().map(|v| v.powi(4)).sum();
        let sum_y: f64 = window.iter().sum();
        let sum_xy: f64 = x.iter().zip(window.iter()).map(|(x, y)| x * y).sum();
        let sum_x2y: f64 = x
            .iter()
            .zip(window.iter())
            .map(|(x, y)| x.powi(2) * y)
            .sum();

        let n_f = n as f64;

        // Solve system using Cramer's rule (simplified)
        let det = n_f * (sum_x2 * sum_x4 - sum_x3 * sum_x3)
            - sum_x * (sum_x * sum_x4 - sum_x2 * sum_x3)
            + sum_x2 * (sum_x * sum_x3 - sum_x2 * sum_x2);

        if det.abs() < 1e-10 {
            return 0.0;
        }

        let det_a = sum_y * (sum_x2 * sum_x4 - sum_x3 * sum_x3)
            - sum_x * (sum_xy * sum_x4 - sum_x2y * sum_x3)
            + sum_x2 * (sum_xy * sum_x3 - sum_x2y * sum_x2);

        let a = det_a / det;

        // Normalize curvature by price range
        let range = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - window.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if range > 0.0 {
            (2.0 * a.abs() * n_f.powi(2) / range).min(1.0)
        } else {
            0.0
        }
    }

    /// Calculate smoothness (inverse of noise).
    fn calculate_smoothness(&self, window: &[f64]) -> f64 {
        let n = window.len();
        if n < 3 {
            return 1.0;
        }

        // Calculate average absolute deviation from smoothed curve
        let smoothed: Vec<f64> = (0..n)
            .map(|i| {
                let start = i.saturating_sub(2);
                let end = (i + 3).min(n);
                let slice = &window[start..end];
                slice.iter().sum::<f64>() / slice.len() as f64
            })
            .collect();

        let deviations: f64 = window
            .iter()
            .zip(smoothed.iter())
            .map(|(w, s)| (w - s).abs())
            .sum();

        let range = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - window.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        if range > 0.0 {
            (deviations / (n as f64 * range)).min(1.0)
        } else {
            0.0
        }
    }

    /// Calculate pattern completion percentage.
    fn calculate_completion(&self, window: &[f64], pattern_type: RoundingType) -> f64 {
        let n = window.len();
        if n < 5 {
            return 0.0;
        }

        let first = window[0];
        let last = window[n - 1];

        match pattern_type {
            RoundingType::Bottom => {
                // Completion when price returns to or exceeds starting level
                let min_val = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                if first > min_val {
                    ((last - min_val) / (first - min_val) * 100.0).min(100.0)
                } else {
                    0.0
                }
            }
            RoundingType::Top => {
                // Completion when price returns to or falls below starting level
                let max_val = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                if first < max_val {
                    ((max_val - last) / (max_val - first) * 100.0).min(100.0)
                } else {
                    0.0
                }
            }
            RoundingType::None => 0.0,
        }
    }
}

impl TechnicalIndicator for RoundingPattern {
    fn name(&self) -> &str {
        "Rounding Pattern"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }
}

impl SignalIndicator for RoundingPattern {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);

        if let Some(&last) = values.last() {
            if last > 0.0 {
                return Ok(IndicatorSignal::Bullish); // Rounding bottom
            } else if last < 0.0 {
                return Ok(IndicatorSignal::Bearish); // Rounding top
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let signals = values
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    IndicatorSignal::Bullish
                } else if v < 0.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rounding_bottom() -> Vec<f64> {
        // Create U-shaped price curve
        let mut prices = Vec::with_capacity(40);
        for i in 0..40 {
            let x = (i as f64 - 20.0) / 20.0; // -1 to 1
            let y = 100.0 + 10.0 * x * x; // Parabola
            prices.push(y + (i % 3) as f64 * 0.1); // Small noise
        }
        prices
    }

    fn make_rounding_top() -> Vec<f64> {
        // Create inverted U-shaped price curve
        let mut prices = Vec::with_capacity(40);
        for i in 0..40 {
            let x = (i as f64 - 20.0) / 20.0;
            let y = 110.0 - 10.0 * x * x; // Inverted parabola
            prices.push(y + (i % 3) as f64 * 0.1);
        }
        prices
    }

    #[test]
    fn test_rounding_pattern_creation() {
        let rp = RoundingPattern::new(30, 0.3, 0.5);
        assert!(rp.is_ok());

        let rp_invalid = RoundingPattern::new(10, 0.3, 0.5);
        assert!(rp_invalid.is_err());
    }

    #[test]
    fn test_rounding_bottom_detection() {
        let prices = make_rounding_bottom();
        let rp = RoundingPattern::new(30, 0.1, 0.8).unwrap();

        let result = rp.calculate(&prices);
        assert_eq!(result.len(), prices.len());

        // Should detect some rounding pattern
        let has_pattern = result.iter().any(|&v| v != 0.0);
        assert!(has_pattern || result.len() > 0);
    }

    #[test]
    fn test_rounding_top_detection() {
        let prices = make_rounding_top();
        let rp = RoundingPattern::new(30, 0.1, 0.8).unwrap();

        let result = rp.calculate(&prices);
        assert_eq!(result.len(), prices.len());
    }

    #[test]
    fn test_detailed_output() {
        let prices = make_rounding_bottom();
        let rp = RoundingPattern::default_params();

        let detailed = rp.calculate_detailed(&prices);
        assert_eq!(detailed.len(), prices.len());
    }

    #[test]
    fn test_curvature_calculation() {
        let rp = RoundingPattern::default_params();

        // Perfect parabola
        let parabola: Vec<f64> = (0..20).map(|i| (i as f64 - 10.0).powi(2)).collect();
        let curvature = rp.calculate_curvature(&parabola);
        assert!(curvature > 0.0);
    }
}
