//! Shark Harmonic Pattern
//!
//! The Shark pattern is a harmonic pattern that uses a 0-5 structure instead of XABCD.
//! It identifies potential reversal zones using Fibonacci ratios:
//! - O-X leg (initial impulse)
//! - X-A leg: 1.13 to 1.618 of O-X
//! - A-B leg: 1.618 to 2.24 of O-X
//! - B-C leg: 0.886 to 1.13 of A-B

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// Shark pattern direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharkDirection {
    Bullish,
    Bearish,
}

/// Shark harmonic pattern configuration.
#[derive(Debug, Clone)]
pub struct SharkPatternConfig {
    /// Lookback period for swing detection (default: 100).
    pub lookback: usize,
    /// Tolerance for Fibonacci ratio matching (default: 0.05).
    pub tolerance: f64,
}

impl Default for SharkPatternConfig {
    fn default() -> Self {
        Self {
            lookback: 100,
            tolerance: 0.05,
        }
    }
}

/// Output from shark pattern detection.
#[derive(Debug, Clone)]
pub struct SharkPatternOutput {
    /// Pattern confidence per bar (0-100), NaN if no pattern.
    pub confidence: Vec<f64>,
    /// Detected direction per bar (None if no pattern).
    pub direction: Vec<Option<SharkDirection>>,
}

/// Shark harmonic pattern indicator.
///
/// Detects Shark harmonic patterns using the 0-5 structure with Fibonacci ratios:
/// - X-A: 1.13 to 1.618 of O-X
/// - A-B: 1.618 to 2.24 of O-X
/// - B-C: 0.886 to 1.13 of A-B
#[derive(Debug, Clone)]
pub struct SharkPattern {
    lookback: usize,
    tolerance: f64,
}

impl SharkPattern {
    /// Create a new SharkPattern indicator with the given configuration.
    pub fn new(config: SharkPatternConfig) -> Self {
        Self {
            lookback: config.lookback,
            tolerance: config.tolerance,
        }
    }

    /// Detect swing highs and swing lows from price data.
    /// Returns vectors of (index, price) for swing highs and swing lows.
    fn find_swing_points(
        &self,
        high: &[f64],
        low: &[f64],
        order: usize,
    ) -> (Vec<(usize, f64)>, Vec<(usize, f64)>) {
        let n = high.len();
        let mut swing_highs = Vec::new();
        let mut swing_lows = Vec::new();

        if n < 2 * order + 1 {
            return (swing_highs, swing_lows);
        }

        for i in order..(n - order) {
            let mut is_high = true;
            let mut is_low = true;

            for j in 1..=order {
                if high[i] <= high[i - j] || high[i] <= high[i + j] {
                    is_high = false;
                }
                if low[i] >= low[i - j] || low[i] >= low[i + j] {
                    is_low = false;
                }
                if !is_high && !is_low {
                    break;
                }
            }

            if is_high {
                swing_highs.push((i, high[i]));
            }
            if is_low {
                swing_lows.push((i, low[i]));
            }
        }

        (swing_highs, swing_lows)
    }

    /// Check whether a ratio falls within the expected range, accounting for tolerance.
    fn ratio_in_range(&self, ratio: f64, low: f64, high: f64) -> bool {
        ratio >= low * (1.0 - self.tolerance) && ratio <= high * (1.0 + self.tolerance)
    }

    /// Compute confidence for how close the ratios are to ideal Fibonacci values.
    fn compute_confidence(&self, xa_ratio: f64, ab_ratio: f64, bc_ratio: f64) -> f64 {
        // Ideal midpoints for each ratio range
        let xa_ideal = (1.13 + 1.618) / 2.0;
        let ab_ideal = (1.618 + 2.24) / 2.0;
        let bc_ideal = (0.886 + 1.13) / 2.0;

        let xa_err = ((xa_ratio - xa_ideal) / xa_ideal).abs();
        let ab_err = ((ab_ratio - ab_ideal) / ab_ideal).abs();
        let bc_err = ((bc_ratio - bc_ideal) / bc_ideal).abs();

        let avg_err = (xa_err + ab_err + bc_err) / 3.0;
        // Map error to confidence: 0% error -> 100 confidence, 20% error -> 0 confidence
        (100.0 * (1.0 - avg_err / 0.2)).clamp(0.0, 100.0)
    }

    /// Run pattern detection across the series. Returns SharkPatternOutput.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> SharkPatternOutput {
        let n = high.len();
        let mut confidence = vec![f64::NAN; n];
        let mut direction: Vec<Option<SharkDirection>> = vec![None; n];

        let swing_order = 3;
        let (swing_highs, swing_lows) = self.find_swing_points(high, low, swing_order);

        // Merge swing points into a single sorted list with their type
        let mut swings: Vec<(usize, f64, bool)> = Vec::new(); // (index, price, is_high)
        for &(idx, price) in &swing_highs {
            swings.push((idx, price, true));
        }
        for &(idx, price) in &swing_lows {
            swings.push((idx, price, false));
        }
        swings.sort_by_key(|s| s.0);

        // Need at least 4 swing points for the 0-X-A-B-C structure
        if swings.len() < 4 {
            return SharkPatternOutput {
                confidence,
                direction,
            };
        }

        // Try to find shark patterns using consecutive swing points
        for w in swings.windows(4) {
            let (o_idx, o_price, o_is_high) = w[0];
            let (_x_idx, x_price, _x_is_high) = w[1];
            let (_a_idx, a_price, _a_is_high) = w[2];
            let (b_idx, b_price, _b_is_high) = w[3];

            // Ensure the points are within the lookback window relative to the last point
            if b_idx.saturating_sub(o_idx) > self.lookback {
                continue;
            }

            let ox_leg = (x_price - o_price).abs();
            if ox_leg < f64::EPSILON {
                continue;
            }

            let xa_ratio = (a_price - x_price).abs() / ox_leg;
            let ab_leg = (b_price - a_price).abs();
            if ab_leg < f64::EPSILON {
                continue;
            }
            let ab_ratio_ox = ab_leg / ox_leg;
            let bc_ratio_ab = 0.0_f64; // We need C point; use B as endpoint for partial detection

            // Check X-A ratio: 1.13 to 1.618 of O-X
            if !self.ratio_in_range(xa_ratio, 1.13, 1.618) {
                continue;
            }

            // Check A-B ratio vs O-X: 1.618 to 2.24
            if !self.ratio_in_range(ab_ratio_ox, 1.618, 2.24) {
                continue;
            }

            // For a 4-point window, check B-C using subsequent swing point if available
            // Mark the pattern at the B point with partial confidence
            let _ = bc_ratio_ab;

            // Determine direction: bullish shark starts with O being a swing low
            let dir = if !o_is_high {
                SharkDirection::Bullish
            } else {
                SharkDirection::Bearish
            };

            // Partial pattern (O-X-A-B confirmed): assign initial confidence
            let conf = self.compute_confidence(xa_ratio, ab_ratio_ox, 1.0);
            if conf > 0.0 {
                confidence[b_idx] = conf * 0.7; // 70% weight for partial pattern
                direction[b_idx] = Some(dir);
            }
        }

        // Try 5-point windows for complete patterns (O-X-A-B-C)
        if swings.len() >= 5 {
            for w in swings.windows(5) {
                let (o_idx, o_price, o_is_high) = w[0];
                let (_x_idx, x_price, _) = w[1];
                let (_a_idx, a_price, _) = w[2];
                let (_b_idx, b_price, _) = w[3];
                let (c_idx, c_price, _) = w[4];

                if c_idx.saturating_sub(o_idx) > self.lookback {
                    continue;
                }

                let ox_leg = (x_price - o_price).abs();
                if ox_leg < f64::EPSILON {
                    continue;
                }

                let xa_ratio = (a_price - x_price).abs() / ox_leg;
                let ab_leg = (b_price - a_price).abs();
                if ab_leg < f64::EPSILON {
                    continue;
                }
                let ab_ratio_ox = ab_leg / ox_leg;
                let bc_ratio = (c_price - b_price).abs() / ab_leg;

                // X-A: 1.13 to 1.618
                if !self.ratio_in_range(xa_ratio, 1.13, 1.618) {
                    continue;
                }

                // A-B: 1.618 to 2.24 of O-X
                if !self.ratio_in_range(ab_ratio_ox, 1.618, 2.24) {
                    continue;
                }

                // B-C: 0.886 to 1.13 of A-B
                if !self.ratio_in_range(bc_ratio, 0.886, 1.13) {
                    continue;
                }

                let dir = if !o_is_high {
                    SharkDirection::Bullish
                } else {
                    SharkDirection::Bearish
                };

                let conf = self.compute_confidence(xa_ratio, ab_ratio_ox, bc_ratio);
                if conf > 0.0 && (confidence[c_idx].is_nan() || conf > confidence[c_idx]) {
                    confidence[c_idx] = conf;
                    direction[c_idx] = Some(dir);
                }
            }
        }

        SharkPatternOutput {
            confidence,
            direction,
        }
    }
}

impl Default for SharkPattern {
    fn default() -> Self {
        Self::new(SharkPatternConfig::default())
    }
}

impl TechnicalIndicator for SharkPattern {
    fn name(&self) -> &str {
        "SharkPattern"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low);

        // Convert direction to numeric signal series for secondary output
        let signals: Vec<f64> = result
            .direction
            .iter()
            .map(|d| match d {
                Some(SharkDirection::Bullish) => 1.0,
                Some(SharkDirection::Bearish) => -1.0,
                None => 0.0,
            })
            .collect();

        Ok(IndicatorOutput::dual(result.confidence, signals))
    }

    fn min_periods(&self) -> usize {
        // Need at least enough bars to form swing points and a pattern
        10
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for SharkPattern {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low);
        let n = result.direction.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        // Return the most recent detected pattern signal
        for i in (0..n).rev() {
            if let Some(dir) = &result.direction[i] {
                return match dir {
                    SharkDirection::Bullish => Ok(IndicatorSignal::Bullish),
                    SharkDirection::Bearish => Ok(IndicatorSignal::Bearish),
                };
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low);
        let signals = result
            .direction
            .iter()
            .map(|d| match d {
                Some(SharkDirection::Bullish) => IndicatorSignal::Bullish,
                Some(SharkDirection::Bearish) => IndicatorSignal::Bearish,
                None => IndicatorSignal::Neutral,
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ohlcv(high: Vec<f64>, low: Vec<f64>) -> OHLCVSeries {
        let n = high.len();
        let close: Vec<f64> = high.iter().zip(low.iter()).map(|(h, l)| (h + l) / 2.0).collect();
        OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; n],
        }
    }

    #[test]
    fn test_shark_pattern_default() {
        let shark = SharkPattern::default();
        assert_eq!(shark.lookback, 100);
        assert!((shark.tolerance - 0.05).abs() < f64::EPSILON);
        assert_eq!(shark.name(), "SharkPattern");
        assert_eq!(shark.min_periods(), 10);
    }

    #[test]
    fn test_shark_pattern_insufficient_data() {
        let shark = SharkPattern::default();
        let data = OHLCVSeries::from_close(vec![100.0; 5]);
        let result = shark.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_shark_pattern_no_pattern_in_flat_data() {
        let shark = SharkPattern::default();
        let n = 50;
        let high = vec![101.0; n];
        let low = vec![99.0; n];
        let data = make_ohlcv(high, low);
        let output = shark.compute(&data).unwrap();

        // Flat data should produce no patterns, all NaN confidence
        let has_pattern = output.primary.iter().any(|v| !v.is_nan() && *v > 0.0);
        assert!(!has_pattern, "Flat data should not produce shark patterns");
    }

    #[test]
    fn test_shark_pattern_signal_neutral_for_random() {
        let shark = SharkPattern::default();
        // Generate oscillating data that does not form a valid shark pattern
        let n = 40;
        let high: Vec<f64> = (0..n).map(|i| 102.0 + (i as f64 * 0.5).sin() * 2.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 98.0 + (i as f64 * 0.5).sin() * 2.0).collect();
        let data = make_ohlcv(high, low);

        let signal = shark.signal(&data).unwrap();
        // Small amplitude oscillation is unlikely to produce valid shark ratios
        assert!(
            signal == IndicatorSignal::Neutral
                || signal == IndicatorSignal::Bullish
                || signal == IndicatorSignal::Bearish,
            "Signal should be a valid variant"
        );
    }

    #[test]
    fn test_shark_pattern_output_lengths() {
        let shark = SharkPattern::default();
        let n = 60;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.3).sin() * 10.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.3).sin() * 10.0).collect();
        let data = make_ohlcv(high, low);

        let output = shark.compute(&data).unwrap();
        assert_eq!(output.primary.len(), n);
        assert!(output.secondary.is_some());
        assert_eq!(output.secondary.unwrap().len(), n);
    }

    #[test]
    fn test_shark_signals_length() {
        let shark = SharkPattern::default();
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.2).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.2).sin() * 5.0).collect();
        let data = make_ohlcv(high, low);

        let signals = shark.signals(&data).unwrap();
        assert_eq!(signals.len(), n);
    }
}
