//! Cypher Harmonic Pattern
//!
//! The Cypher pattern is a harmonic pattern using the XABCD structure with
//! specific Fibonacci ratios:
//! - XA leg (initial impulse)
//! - AB: 0.382 to 0.618 retracement of XA
//! - BC: 1.13 to 1.414 extension of AB
//! - CD: 0.786 retracement of XC

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// Cypher pattern direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CypherDirection {
    Bullish,
    Bearish,
}

/// Cypher harmonic pattern configuration.
#[derive(Debug, Clone)]
pub struct CypherPatternConfig {
    /// Lookback period for swing detection (default: 100).
    pub lookback: usize,
    /// Tolerance for Fibonacci ratio matching (default: 0.05).
    pub tolerance: f64,
}

impl Default for CypherPatternConfig {
    fn default() -> Self {
        Self {
            lookback: 100,
            tolerance: 0.05,
        }
    }
}

/// Output from cypher pattern detection.
#[derive(Debug, Clone)]
pub struct CypherPatternOutput {
    /// Pattern confidence per bar (0-100), NaN if no pattern.
    pub confidence: Vec<f64>,
    /// Detected direction per bar (None if no pattern).
    pub direction: Vec<Option<CypherDirection>>,
}

/// Cypher harmonic pattern indicator.
///
/// Detects Cypher harmonic patterns using XABCD structure with Fibonacci ratios:
/// - AB: 0.382 to 0.618 of XA
/// - BC: 1.13 to 1.414 of AB
/// - CD: 0.786 of XC
#[derive(Debug, Clone)]
pub struct CypherPattern {
    lookback: usize,
    tolerance: f64,
}

impl CypherPattern {
    /// Create a new CypherPattern indicator with the given configuration.
    pub fn new(config: CypherPatternConfig) -> Self {
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
    fn compute_confidence(&self, ab_ratio: f64, bc_ratio: f64, cd_ratio: f64) -> f64 {
        let ab_ideal = (0.382 + 0.618) / 2.0;
        let bc_ideal = (1.13 + 1.414) / 2.0;
        let cd_ideal = 0.786;

        let ab_err = ((ab_ratio - ab_ideal) / ab_ideal).abs();
        let bc_err = ((bc_ratio - bc_ideal) / bc_ideal).abs();
        let cd_err = ((cd_ratio - cd_ideal) / cd_ideal).abs();

        let avg_err = (ab_err + bc_err + cd_err) / 3.0;
        (100.0 * (1.0 - avg_err / 0.2)).clamp(0.0, 100.0)
    }

    /// Run pattern detection across the series. Returns CypherPatternOutput.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> CypherPatternOutput {
        let n = high.len();
        let mut confidence = vec![f64::NAN; n];
        let mut direction: Vec<Option<CypherDirection>> = vec![None; n];

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

        // Need at least 5 swing points for X-A-B-C-D
        if swings.len() < 5 {
            return CypherPatternOutput {
                confidence,
                direction,
            };
        }

        for w in swings.windows(5) {
            let (x_idx, x_price, x_is_high) = w[0];
            let (_a_idx, a_price, _) = w[1];
            let (_b_idx, b_price, _) = w[2];
            let (_c_idx, c_price, _) = w[3];
            let (d_idx, d_price, _) = w[4];

            // Ensure all points fall within the lookback window
            if d_idx.saturating_sub(x_idx) > self.lookback {
                continue;
            }

            let xa_leg = (a_price - x_price).abs();
            if xa_leg < f64::EPSILON {
                continue;
            }

            // AB: 0.382 to 0.618 retracement of XA
            let ab_retrace = (b_price - a_price).abs() / xa_leg;
            if !self.ratio_in_range(ab_retrace, 0.382, 0.618) {
                continue;
            }

            // BC: 1.13 to 1.414 extension of AB
            let ab_leg = (b_price - a_price).abs();
            if ab_leg < f64::EPSILON {
                continue;
            }
            let bc_ext = (c_price - b_price).abs() / ab_leg;
            if !self.ratio_in_range(bc_ext, 1.13, 1.414) {
                continue;
            }

            // CD: 0.786 retracement of XC
            let xc_leg = (c_price - x_price).abs();
            if xc_leg < f64::EPSILON {
                continue;
            }
            let cd_retrace = (d_price - c_price).abs() / xc_leg;
            if !self.ratio_near(cd_retrace, 0.786) {
                continue;
            }

            // Determine direction: bullish cypher starts with X as a swing low
            let dir = if !x_is_high {
                CypherDirection::Bullish
            } else {
                CypherDirection::Bearish
            };

            let conf = self.compute_confidence(ab_retrace, bc_ext, cd_retrace);
            if conf > 0.0 && (confidence[d_idx].is_nan() || conf > confidence[d_idx]) {
                confidence[d_idx] = conf;
                direction[d_idx] = Some(dir);
            }
        }

        CypherPatternOutput {
            confidence,
            direction,
        }
    }
}

impl Default for CypherPattern {
    fn default() -> Self {
        Self::new(CypherPatternConfig::default())
    }
}

impl TechnicalIndicator for CypherPattern {
    fn name(&self) -> &str {
        "CypherPattern"
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
                Some(CypherDirection::Bullish) => 1.0,
                Some(CypherDirection::Bearish) => -1.0,
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

impl SignalIndicator for CypherPattern {
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
                    CypherDirection::Bullish => Ok(IndicatorSignal::Bullish),
                    CypherDirection::Bearish => Ok(IndicatorSignal::Bearish),
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
                Some(CypherDirection::Bullish) => IndicatorSignal::Bullish,
                Some(CypherDirection::Bearish) => IndicatorSignal::Bearish,
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
    fn test_cypher_pattern_default() {
        let cypher = CypherPattern::default();
        assert_eq!(cypher.lookback, 100);
        assert!((cypher.tolerance - 0.05).abs() < f64::EPSILON);
        assert_eq!(cypher.name(), "CypherPattern");
        assert_eq!(cypher.min_periods(), 10);
    }

    #[test]
    fn test_cypher_pattern_insufficient_data() {
        let cypher = CypherPattern::default();
        let data = OHLCVSeries::from_close(vec![100.0; 5]);
        let result = cypher.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_cypher_pattern_no_pattern_in_flat_data() {
        let cypher = CypherPattern::default();
        let n = 50;
        let high = vec![101.0; n];
        let low = vec![99.0; n];
        let data = make_ohlcv(high, low);
        let output = cypher.compute(&data).unwrap();

        // Flat data should produce no patterns
        let has_pattern = output.primary.iter().any(|v| !v.is_nan() && *v > 0.0);
        assert!(!has_pattern, "Flat data should not produce cypher patterns");
    }

    #[test]
    fn test_cypher_pattern_signal_variant() {
        let cypher = CypherPattern::default();
        let n = 40;
        let high: Vec<f64> = (0..n).map(|i| 102.0 + (i as f64 * 0.5).sin() * 2.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 98.0 + (i as f64 * 0.5).sin() * 2.0).collect();
        let data = make_ohlcv(high, low);

        let signal = cypher.signal(&data).unwrap();
        assert!(
            signal == IndicatorSignal::Neutral
                || signal == IndicatorSignal::Bullish
                || signal == IndicatorSignal::Bearish,
            "Signal should be a valid variant"
        );
    }

    #[test]
    fn test_cypher_pattern_output_lengths() {
        let cypher = CypherPattern::default();
        let n = 60;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.3).sin() * 10.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.3).sin() * 10.0).collect();
        let data = make_ohlcv(high, low);

        let output = cypher.compute(&data).unwrap();
        assert_eq!(output.primary.len(), n);
        assert!(output.secondary.is_some());
        assert_eq!(output.secondary.unwrap().len(), n);
    }

    #[test]
    fn test_cypher_signals_length() {
        let cypher = CypherPattern::default();
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.2).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.2).sin() * 5.0).collect();
        let data = make_ohlcv(high, low);

        let signals = cypher.signals(&data).unwrap();
        assert_eq!(signals.len(), n);
    }
}
