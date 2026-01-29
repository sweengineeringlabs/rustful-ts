//! ABCD Harmonic Pattern
//!
//! The ABCD pattern is the simplest harmonic pattern, consisting of two
//! equivalent price legs (AB and CD) connected by a retracement (BC).
//! Fibonacci ratios:
//! - AB leg (initial impulse)
//! - BC: 0.382 to 0.886 retracement of AB
//! - CD: equals AB (1:1) or 1.27 to 1.618 extension of BC

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// ABCD pattern direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ABCDDirection {
    Bullish,
    Bearish,
}

/// ABCD harmonic pattern configuration.
#[derive(Debug, Clone)]
pub struct ABCDPatternConfig {
    /// Lookback period for swing detection (default: 100).
    pub lookback: usize,
    /// Tolerance for Fibonacci ratio matching (default: 0.05).
    pub tolerance: f64,
}

impl Default for ABCDPatternConfig {
    fn default() -> Self {
        Self {
            lookback: 100,
            tolerance: 0.05,
        }
    }
}

/// Output from ABCD pattern detection.
#[derive(Debug, Clone)]
pub struct ABCDPatternOutput {
    /// Pattern confidence per bar (0-100), NaN if no pattern.
    pub confidence: Vec<f64>,
    /// Detected direction per bar (None if no pattern).
    pub direction: Vec<Option<ABCDDirection>>,
}

/// ABCD harmonic pattern indicator.
///
/// Detects ABCD harmonic patterns - the simplest harmonic formation:
/// - BC: 0.382 to 0.886 retracement of AB
/// - CD: equals AB (1:1 ratio) or 1.27 to 1.618 extension of BC
#[derive(Debug, Clone)]
pub struct ABCDPattern {
    lookback: usize,
    tolerance: f64,
}

impl ABCDPattern {
    /// Create a new ABCDPattern indicator with the given configuration.
    pub fn new(config: ABCDPatternConfig) -> Self {
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

    /// Check whether a ratio is close to a target value within tolerance.
    fn ratio_near(&self, ratio: f64, target: f64) -> bool {
        (ratio - target).abs() <= target * self.tolerance
    }

    /// Compute confidence for how close the ratios match ideal Fibonacci values.
    /// ab_cd_ratio: CD/AB ratio (ideal = 1.0)
    /// bc_retrace: BC retracement of AB
    /// cd_bc_ext: CD extension of BC (ideal = 1.27 to 1.618 range)
    fn compute_confidence(
        &self,
        bc_retrace: f64,
        ab_cd_ratio: f64,
        cd_bc_ext: f64,
    ) -> f64 {
        let bc_ideal = (0.382 + 0.886) / 2.0;
        let bc_err = ((bc_retrace - bc_ideal) / bc_ideal).abs();

        // CD can match either: 1:1 with AB or extension of BC
        let cd_ab_err = ((ab_cd_ratio - 1.0) / 1.0).abs();
        let cd_bc_ideal = (1.27 + 1.618) / 2.0;
        let cd_bc_err = ((cd_bc_ext - cd_bc_ideal) / cd_bc_ideal).abs();

        // Use the better CD match (either 1:1 or BC extension)
        let cd_err = cd_ab_err.min(cd_bc_err);

        let avg_err = (bc_err + cd_err) / 2.0;
        (100.0 * (1.0 - avg_err / 0.2)).clamp(0.0, 100.0)
    }

    /// Run pattern detection across the series. Returns ABCDPatternOutput.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> ABCDPatternOutput {
        let n = high.len();
        let mut confidence = vec![f64::NAN; n];
        let mut direction: Vec<Option<ABCDDirection>> = vec![None; n];

        let swing_order = 3;
        let (swing_highs, swing_lows) = self.find_swing_points(high, low, swing_order);

        // Merge swing points into a single sorted list with type info
        let mut swings: Vec<(usize, f64, bool)> = Vec::new();
        for &(idx, price) in &swing_highs {
            swings.push((idx, price, true));
        }
        for &(idx, price) in &swing_lows {
            swings.push((idx, price, false));
        }
        swings.sort_by_key(|s| s.0);

        // Need at least 4 swing points for A-B-C-D
        if swings.len() < 4 {
            return ABCDPatternOutput {
                confidence,
                direction,
            };
        }

        for w in swings.windows(4) {
            let (a_idx, a_price, a_is_high) = w[0];
            let (_b_idx, b_price, _) = w[1];
            let (_c_idx, c_price, _) = w[2];
            let (d_idx, d_price, _) = w[3];

            // Ensure all points fall within the lookback window
            if d_idx.saturating_sub(a_idx) > self.lookback {
                continue;
            }

            let ab_leg = (b_price - a_price).abs();
            if ab_leg < f64::EPSILON {
                continue;
            }

            // BC: 0.382 to 0.886 retracement of AB
            let bc_retrace = (c_price - b_price).abs() / ab_leg;
            if !self.ratio_in_range(bc_retrace, 0.382, 0.886) {
                continue;
            }

            let cd_leg = (d_price - c_price).abs();
            let bc_leg = (c_price - b_price).abs();

            // CD should equal AB (1:1) or be 1.27-1.618 extension of BC
            let ab_cd_ratio = cd_leg / ab_leg;
            let cd_bc_ext = if bc_leg > f64::EPSILON {
                cd_leg / bc_leg
            } else {
                f64::NAN
            };

            let cd_is_equal = self.ratio_near(ab_cd_ratio, 1.0);
            let cd_is_ext = !cd_bc_ext.is_nan() && self.ratio_in_range(cd_bc_ext, 1.27, 1.618);

            if !cd_is_equal && !cd_is_ext {
                continue;
            }

            // Determine direction:
            // Bullish ABCD: A is high, B is low, C is high (lower than A), D is low
            // Bearish ABCD: A is low, B is high, C is low (higher than A), D is high
            let dir = if a_is_high {
                ABCDDirection::Bullish // descending A-B, the D point is a potential buy
            } else {
                ABCDDirection::Bearish // ascending A-B, the D point is a potential sell
            };

            let cd_bc_for_conf = if cd_bc_ext.is_nan() { 1.27 } else { cd_bc_ext };
            let conf = self.compute_confidence(bc_retrace, ab_cd_ratio, cd_bc_for_conf);
            if conf > 0.0 && (confidence[d_idx].is_nan() || conf > confidence[d_idx]) {
                confidence[d_idx] = conf;
                direction[d_idx] = Some(dir);
            }
        }

        ABCDPatternOutput {
            confidence,
            direction,
        }
    }
}

impl Default for ABCDPattern {
    fn default() -> Self {
        Self::new(ABCDPatternConfig::default())
    }
}

impl TechnicalIndicator for ABCDPattern {
    fn name(&self) -> &str {
        "ABCDPattern"
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

        let signals: Vec<f64> = result
            .direction
            .iter()
            .map(|d| match d {
                Some(ABCDDirection::Bullish) => 1.0,
                Some(ABCDDirection::Bearish) => -1.0,
                None => 0.0,
            })
            .collect();

        Ok(IndicatorOutput::dual(result.confidence, signals))
    }

    fn min_periods(&self) -> usize {
        10
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for ABCDPattern {
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
                    ABCDDirection::Bullish => Ok(IndicatorSignal::Bullish),
                    ABCDDirection::Bearish => Ok(IndicatorSignal::Bearish),
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
                Some(ABCDDirection::Bullish) => IndicatorSignal::Bullish,
                Some(ABCDDirection::Bearish) => IndicatorSignal::Bearish,
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
    fn test_abcd_pattern_default() {
        let abcd = ABCDPattern::default();
        assert_eq!(abcd.lookback, 100);
        assert!((abcd.tolerance - 0.05).abs() < f64::EPSILON);
        assert_eq!(abcd.name(), "ABCDPattern");
        assert_eq!(abcd.min_periods(), 10);
    }

    #[test]
    fn test_abcd_pattern_insufficient_data() {
        let abcd = ABCDPattern::default();
        let data = OHLCVSeries::from_close(vec![100.0; 5]);
        let result = abcd.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_abcd_pattern_no_pattern_in_flat_data() {
        let abcd = ABCDPattern::default();
        let n = 50;
        let high = vec![101.0; n];
        let low = vec![99.0; n];
        let data = make_ohlcv(high, low);
        let output = abcd.compute(&data).unwrap();

        // Flat data should produce no patterns
        let has_pattern = output.primary.iter().any(|v| !v.is_nan() && *v > 0.0);
        assert!(!has_pattern, "Flat data should not produce ABCD patterns");
    }

    #[test]
    fn test_abcd_pattern_signal_variant() {
        let abcd = ABCDPattern::default();
        let n = 40;
        let high: Vec<f64> = (0..n).map(|i| 102.0 + (i as f64 * 0.5).sin() * 2.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 98.0 + (i as f64 * 0.5).sin() * 2.0).collect();
        let data = make_ohlcv(high, low);

        let signal = abcd.signal(&data).unwrap();
        assert!(
            signal == IndicatorSignal::Neutral
                || signal == IndicatorSignal::Bullish
                || signal == IndicatorSignal::Bearish,
            "Signal should be a valid variant"
        );
    }

    #[test]
    fn test_abcd_pattern_output_lengths() {
        let abcd = ABCDPattern::default();
        let n = 60;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.3).sin() * 10.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.3).sin() * 10.0).collect();
        let data = make_ohlcv(high, low);

        let output = abcd.compute(&data).unwrap();
        assert_eq!(output.primary.len(), n);
        assert!(output.secondary.is_some());
        assert_eq!(output.secondary.unwrap().len(), n);
    }

    #[test]
    fn test_abcd_signals_length() {
        let abcd = ABCDPattern::default();
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.2).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.2).sin() * 5.0).collect();
        let data = make_ohlcv(high, low);

        let signals = abcd.signals(&data).unwrap();
        assert_eq!(signals.len(), n);
    }

    #[test]
    fn test_abcd_config_custom() {
        let config = ABCDPatternConfig {
            lookback: 50,
            tolerance: 0.1,
        };
        let abcd = ABCDPattern::new(config);
        assert_eq!(abcd.lookback, 50);
        assert!((abcd.tolerance - 0.1).abs() < f64::EPSILON);
    }
}
