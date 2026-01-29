//! Bat Harmonic Pattern implementation.
//!
//! The Bat pattern is a precise harmonic pattern that uses Fibonacci ratios
//! to identify potential reversal zones. It is characterized by a shallow
//! B point retracement (0.382-0.50 of XA) and a CD extension of 1.618-2.618
//! of BC, with the D point completing near the 0.886 retracement of XA.
//!
//! Ideal Fibonacci ratios:
//! - B retraces 0.382-0.50 of XA
//! - AB leg is 0.382-0.886 of XA
//! - BC leg is 0.382-0.886 of AB
//! - CD leg extends 1.618-2.618 of BC

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Bat pattern detection.
#[derive(Debug, Clone)]
pub struct BatPatternConfig {
    /// Number of bars to look back for pattern detection (default: 100).
    pub lookback: usize,
    /// Tolerance for Fibonacci ratio matching as a fraction (default: 0.05).
    pub tolerance: f64,
}

impl Default for BatPatternConfig {
    fn default() -> Self {
        Self {
            lookback: 100,
            tolerance: 0.05,
        }
    }
}

// ============================================================================
// Output
// ============================================================================

/// Output from Bat pattern detection.
#[derive(Debug, Clone)]
pub struct BatPatternOutput {
    /// Pattern confidence values (0-100) per bar.
    pub confidence: Vec<f64>,
    /// Pattern direction per bar: 1.0 = bullish, -1.0 = bearish, 0.0 = none.
    pub direction: Vec<f64>,
}

// ============================================================================
// Swing Point Detection
// ============================================================================

/// A detected swing point (local high or low).
#[derive(Debug, Clone, Copy)]
struct SwingPoint {
    index: usize,
    price: f64,
    is_high: bool,
}

/// Find swing highs and lows in the price data.
fn find_swing_points(high: &[f64], low: &[f64], order: usize) -> Vec<SwingPoint> {
    let n = high.len();
    let mut points = Vec::new();

    if n < 2 * order + 1 {
        return points;
    }

    for i in order..(n - order) {
        // Check for swing high
        let mut is_swing_high = true;
        for j in 1..=order {
            if high[i] <= high[i - j] || high[i] <= high[i + j] {
                is_swing_high = false;
                break;
            }
        }
        if is_swing_high {
            points.push(SwingPoint {
                index: i,
                price: high[i],
                is_high: true,
            });
        }

        // Check for swing low
        let mut is_swing_low = true;
        for j in 1..=order {
            if low[i] >= low[i - j] || low[i] >= low[i + j] {
                is_swing_low = false;
                break;
            }
        }
        if is_swing_low {
            points.push(SwingPoint {
                index: i,
                price: low[i],
                is_high: false,
            });
        }
    }

    points.sort_by_key(|p| p.index);
    points
}

// ============================================================================
// Bat Pattern
// ============================================================================

/// Bat Harmonic Pattern indicator.
///
/// Identifies bullish and bearish Bat patterns using XABCD price legs with
/// Fibonacci ratio validation. The Bat pattern features a shallow B point
/// retracement (0.382-0.50 of XA), making it one of the more conservative
/// harmonic patterns.
///
/// - Bullish Bat: X(low) -> A(high) -> B(low) -> C(high) -> D(low)
/// - Bearish Bat: X(high) -> A(low) -> B(high) -> C(low) -> D(high)
#[derive(Debug, Clone)]
pub struct BatPattern {
    lookback: usize,
    tolerance: f64,
}

impl BatPattern {
    pub fn new(config: BatPatternConfig) -> Self {
        Self {
            lookback: config.lookback,
            tolerance: config.tolerance,
        }
    }

    /// Check if a ratio is within tolerance of a target range.
    fn ratio_in_range(&self, actual: f64, low: f64, high: f64) -> bool {
        actual >= (low - self.tolerance) && actual <= (high + self.tolerance)
    }

    /// Calculate how well ratios match the ideal Bat pattern.
    /// Returns a confidence score from 0 to 100.
    fn score_pattern(&self, ab_xa: f64, bc_ab: f64, cd_bc: f64) -> f64 {
        // Ideal Bat ratios
        let xa_retrace_range = (0.382, 0.50);
        let bc_ab_range = (0.382, 0.886);
        let cd_bc_range = (1.618, 2.618);

        // Score each ratio (up to ~33 points each, total 100)
        let xa_score = if self.ratio_in_range(ab_xa, xa_retrace_range.0, xa_retrace_range.1) {
            // Extra points for being close to center of range
            let center = (xa_retrace_range.0 + xa_retrace_range.1) / 2.0;
            let range_half = (xa_retrace_range.1 - xa_retrace_range.0) / 2.0;
            let dist_from_center = (ab_xa - center).abs();
            33.3 * (1.0 - (dist_from_center / range_half).min(1.0) * 0.3)
        } else {
            let dist_low = (ab_xa - xa_retrace_range.0).abs();
            let dist_high = (ab_xa - xa_retrace_range.1).abs();
            let dist = dist_low.min(dist_high);
            (33.3 - (dist / self.tolerance) * 16.0).max(0.0)
        };

        let bc_score = if self.ratio_in_range(bc_ab, bc_ab_range.0, bc_ab_range.1) {
            33.3
        } else {
            let dist_low = (bc_ab - bc_ab_range.0).abs();
            let dist_high = (bc_ab - bc_ab_range.1).abs();
            let dist = dist_low.min(dist_high);
            (33.3 - (dist / self.tolerance) * 16.0).max(0.0)
        };

        let cd_score = if self.ratio_in_range(cd_bc, cd_bc_range.0, cd_bc_range.1) {
            33.4
        } else {
            let dist_low = (cd_bc - cd_bc_range.0).abs();
            let dist_high = (cd_bc - cd_bc_range.1).abs();
            let dist = dist_low.min(dist_high);
            (33.4 - (dist / self.tolerance) * 16.0).max(0.0)
        };

        xa_score + bc_score + cd_score
    }

    /// Detect Bat patterns in the OHLCV data.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> BatPatternOutput {
        let n = high.len();
        let mut confidence = vec![0.0_f64; n];
        let mut direction = vec![0.0_f64; n];

        let swing_order = 3;
        let points = find_swing_points(high, low, swing_order);

        if points.len() < 5 {
            return BatPatternOutput {
                confidence,
                direction,
            };
        }

        // Iterate through combinations of 5 swing points to find XABCD patterns
        for w in 0..points.len().saturating_sub(4) {
            let x = &points[w];
            let a = &points[w + 1];
            let b = &points[w + 2];
            let c = &points[w + 3];
            let d = &points[w + 4];

            // Check pattern is within lookback
            if d.index.saturating_sub(x.index) > self.lookback {
                continue;
            }

            // Check alternating high/low structure for bullish:
            // X(low) -> A(high) -> B(low) -> C(high) -> D(low)
            if !x.is_high && a.is_high && !b.is_high && c.is_high && !d.is_high {
                let xa = a.price - x.price;
                let ab = a.price - b.price;
                let bc = c.price - b.price;
                let cd = c.price - d.price;

                if xa.abs() < 1e-10 || ab.abs() < 1e-10 || bc.abs() < 1e-10 {
                    continue;
                }

                let ab_xa_ratio = ab / xa;
                let bc_ab_ratio = bc / ab;
                let cd_bc_ratio = cd / bc;

                // Validate Bat ratios:
                // XA retracement 0.382-0.50, BC 0.382-0.886 of AB, CD 1.618-2.618 of BC
                if self.ratio_in_range(ab_xa_ratio, 0.382, 0.50)
                    && self.ratio_in_range(bc_ab_ratio, 0.382, 0.886)
                    && self.ratio_in_range(cd_bc_ratio, 1.618, 2.618)
                {
                    let score = self.score_pattern(ab_xa_ratio, bc_ab_ratio, cd_bc_ratio);
                    if score > confidence[d.index] {
                        confidence[d.index] = score;
                        direction[d.index] = 1.0; // Bullish
                    }
                }
            }

            // Check alternating structure for bearish:
            // X(high) -> A(low) -> B(high) -> C(low) -> D(high)
            if x.is_high && !a.is_high && b.is_high && !c.is_high && d.is_high {
                let xa = x.price - a.price;
                let ab = b.price - a.price;
                let bc = b.price - c.price;
                let cd = d.price - c.price;

                if xa.abs() < 1e-10 || ab.abs() < 1e-10 || bc.abs() < 1e-10 {
                    continue;
                }

                let ab_xa_ratio = ab / xa;
                let bc_ab_ratio = bc / ab;
                let cd_bc_ratio = cd / bc;

                // Validate Bat ratios
                if self.ratio_in_range(ab_xa_ratio, 0.382, 0.50)
                    && self.ratio_in_range(bc_ab_ratio, 0.382, 0.886)
                    && self.ratio_in_range(cd_bc_ratio, 1.618, 2.618)
                {
                    let score = self.score_pattern(ab_xa_ratio, bc_ab_ratio, cd_bc_ratio);
                    if score > confidence[d.index] {
                        confidence[d.index] = score;
                        direction[d.index] = -1.0; // Bearish
                    }
                }
            }
        }

        BatPatternOutput {
            confidence,
            direction,
        }
    }
}

impl Default for BatPattern {
    fn default() -> Self {
        Self::new(BatPatternConfig::default())
    }
}

impl TechnicalIndicator for BatPattern {
    fn name(&self) -> &str {
        "BatPattern"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(result.confidence, result.direction))
    }

    fn min_periods(&self) -> usize {
        7.max(self.lookback / 5)
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for BatPattern {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low);
        let n = result.confidence.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        // Look at the last few bars for a recent pattern
        let lookback_window = 5.min(n);
        for i in (n - lookback_window..n).rev() {
            if result.confidence[i] > 50.0 {
                return if result.direction[i] > 0.0 {
                    Ok(IndicatorSignal::Bullish)
                } else if result.direction[i] < 0.0 {
                    Ok(IndicatorSignal::Bearish)
                } else {
                    Ok(IndicatorSignal::Neutral)
                };
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low);

        let signals = result
            .confidence
            .iter()
            .zip(result.direction.iter())
            .map(|(&conf, &dir)| {
                if conf > 50.0 {
                    if dir > 0.0 {
                        IndicatorSignal::Bullish
                    } else if dir < 0.0 {
                        IndicatorSignal::Bearish
                    } else {
                        IndicatorSignal::Neutral
                    }
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

    fn make_ohlcv_series(high: Vec<f64>, low: Vec<f64>, close: Vec<f64>) -> OHLCVSeries {
        let n = close.len();
        OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; n],
        }
    }

    /// Generate synthetic data with an embedded bullish Bat-like structure.
    fn generate_bat_data() -> OHLCVSeries {
        // Create a price series with swing points that approximate Bat ratios:
        // X(low=90) -> A(high=110) -> B(low ~101, 0.45 retrace of XA=20)
        // -> C(high ~105) -> D(low ~96)
        let prices: Vec<f64> = vec![
            100.0, 97.0, 94.0, 92.0, 90.0, // decline to X
            93.0, 97.0, 102.0, 107.0, 110.0, // rally to A
            109.0, 107.0, 104.0, 102.0, 101.0, // retrace to B (~0.45 of XA)
            102.0, 103.0, 104.0, 104.5, 105.0, // rally to C
            104.0, 102.0, 100.0, 98.0, 96.0, // decline to D
            97.0, 98.0, 99.0, 100.0, 101.0, // subsequent recovery
        ];

        let high: Vec<f64> = prices.iter().map(|p| p + 1.5).collect();
        let low: Vec<f64> = prices.iter().map(|p| p - 1.5).collect();

        make_ohlcv_series(high, low, prices)
    }

    #[test]
    fn test_bat_basic_compute() {
        let pattern = BatPattern::default();
        let data = generate_bat_data();

        let output = pattern.compute(&data).unwrap();

        assert_eq!(output.primary.len(), data.len());
        assert!(output.secondary.is_some());
        let direction = output.secondary.unwrap();
        assert_eq!(direction.len(), data.len());

        // Confidence values should be in [0, 100]
        for &c in &output.primary {
            assert!(c >= 0.0 && c <= 100.0);
        }

        // Direction values should be -1, 0, or 1
        for &d in &direction {
            assert!(d == -1.0 || d == 0.0 || d == 1.0);
        }
    }

    #[test]
    fn test_bat_insufficient_data() {
        let pattern = BatPattern::default();
        let data = OHLCVSeries {
            open: vec![1.0, 2.0],
            high: vec![1.5, 2.5],
            low: vec![0.5, 1.5],
            close: vec![1.0, 2.0],
            volume: vec![100.0, 100.0],
        };

        let result = pattern.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_bat_config_defaults() {
        let config = BatPatternConfig::default();
        assert_eq!(config.lookback, 100);
        assert!((config.tolerance - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_bat_signal_generation() {
        let pattern = BatPattern::default();
        let data = generate_bat_data();

        let signals = pattern.signals(&data).unwrap();
        assert_eq!(signals.len(), data.len());

        // All signals should be valid enum values
        for signal in &signals {
            match signal {
                IndicatorSignal::Bullish | IndicatorSignal::Bearish | IndicatorSignal::Neutral => {}
            }
        }
    }

    #[test]
    fn test_bat_ratio_ranges() {
        let pattern = BatPattern::new(BatPatternConfig {
            lookback: 100,
            tolerance: 0.05,
        });

        // Within the Bat XA retracement range (0.382-0.50)
        assert!(pattern.ratio_in_range(0.45, 0.382, 0.50));
        assert!(pattern.ratio_in_range(0.382, 0.382, 0.50));
        assert!(pattern.ratio_in_range(0.50, 0.382, 0.50));
        // Just outside with tolerance
        assert!(pattern.ratio_in_range(0.35, 0.382, 0.50)); // 0.382 - 0.05 = 0.332
        // Well outside range
        assert!(!pattern.ratio_in_range(0.2, 0.382, 0.50));
    }
}
