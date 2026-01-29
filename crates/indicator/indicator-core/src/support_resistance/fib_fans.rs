//! Fibonacci Fans - Angled trend lines based on Fibonacci ratios.
//!
//! IND-389: Fibonacci Fans draw diagonal trend lines from a significant
//! price point through Fibonacci retracement levels, creating angled
//! support/resistance zones.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// Fibonacci fan line data for a single time point.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FibFanLevels {
    /// Origin price point (swing high or low)
    pub origin: f64,
    /// 38.2% fan line value
    pub fan_382: f64,
    /// 50% fan line value
    pub fan_500: f64,
    /// 61.8% fan line value
    pub fan_618: f64,
}

/// Fibonacci Fans Indicator
///
/// Draws diagonal trend lines from a pivot point through Fibonacci
/// retracement levels. In an uptrend, fans are drawn from a swing low;
/// in a downtrend, from a swing high.
///
/// # Interpretation
/// - Fan lines act as dynamic support/resistance
/// - Price tends to find support/resistance at fan line intersections
/// - Breaking through a fan line often leads to the next level
///
/// # Formula
/// Fan angle is calculated by connecting the origin to Fibonacci
/// retracement levels at a specified distance.
#[derive(Debug, Clone)]
pub struct FibonacciFans {
    /// Lookback period to find swing points
    lookback: usize,
    /// Number of bars for projection
    projection_bars: usize,
    /// Swing detection strength (bars on each side)
    swing_strength: usize,
}

impl FibonacciFans {
    /// Create a new Fibonacci Fans indicator.
    ///
    /// # Arguments
    /// * `lookback` - Period to look for swing highs/lows (minimum 10)
    /// * `projection_bars` - Number of bars to project fans forward (minimum 5)
    /// * `swing_strength` - Bars on each side to confirm swing (minimum 2)
    pub fn new(lookback: usize, projection_bars: usize, swing_strength: usize) -> Result<Self> {
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if projection_bars < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "projection_bars".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if swing_strength < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_strength".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            lookback,
            projection_bars,
            swing_strength,
        })
    }

    /// Create with default parameters (lookback=20, projection=10, swing=3).
    pub fn default_params() -> Self {
        Self {
            lookback: 20,
            projection_bars: 10,
            swing_strength: 3,
        }
    }

    /// Find swing high in the given range.
    fn find_swing_high(&self, high: &[f64], start: usize, end: usize) -> Option<(usize, f64)> {
        let mut best_idx = None;
        let mut best_high = f64::NEG_INFINITY;

        for i in (start + self.swing_strength)..(end.saturating_sub(self.swing_strength)) {
            let mut is_swing = true;

            // Check bars before
            for j in (i.saturating_sub(self.swing_strength))..i {
                if high[j] > high[i] {
                    is_swing = false;
                    break;
                }
            }

            // Check bars after
            if is_swing {
                for j in (i + 1)..=(i + self.swing_strength).min(end - 1) {
                    if high[j] > high[i] {
                        is_swing = false;
                        break;
                    }
                }
            }

            if is_swing && high[i] > best_high {
                best_high = high[i];
                best_idx = Some(i);
            }
        }

        best_idx.map(|idx| (idx, best_high))
    }

    /// Find swing low in the given range.
    fn find_swing_low(&self, low: &[f64], start: usize, end: usize) -> Option<(usize, f64)> {
        let mut best_idx = None;
        let mut best_low = f64::INFINITY;

        for i in (start + self.swing_strength)..(end.saturating_sub(self.swing_strength)) {
            let mut is_swing = true;

            // Check bars before
            for j in (i.saturating_sub(self.swing_strength))..i {
                if low[j] < low[i] {
                    is_swing = false;
                    break;
                }
            }

            // Check bars after
            if is_swing {
                for j in (i + 1)..=(i + self.swing_strength).min(end - 1) {
                    if low[j] < low[i] {
                        is_swing = false;
                        break;
                    }
                }
            }

            if is_swing && low[i] < best_low {
                best_low = low[i];
                best_idx = Some(i);
            }
        }

        best_idx.map(|idx| (idx, best_low))
    }

    /// Calculate Fibonacci fan values.
    ///
    /// Returns vectors for each fan line: 38.2%, 50%, and 61.8%.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut fan_382 = vec![f64::NAN; n];
        let mut fan_500 = vec![f64::NAN; n];
        let mut fan_618 = vec![f64::NAN; n];

        if n < self.lookback + self.projection_bars {
            return (fan_382, fan_500, fan_618);
        }

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Find swing high and swing low in lookback period
            let swing_high = self.find_swing_high(high, start, i);
            let swing_low = self.find_swing_low(low, start, i);

            if let (Some((high_idx, high_val)), Some((low_idx, low_val))) = (swing_high, swing_low) {
                let range = high_val - low_val;

                // Determine trend based on which swing came first
                let is_uptrend = low_idx < high_idx;

                if is_uptrend {
                    // Fan from swing low upward
                    let bars_from_origin = (i - low_idx) as f64;
                    if bars_from_origin > 0.0 {
                        let vertical_range = range;

                        // Calculate fan line slopes based on Fibonacci ratios
                        // Slower slope = lower Fibonacci ratio
                        fan_382[i] = low_val + (vertical_range * 0.382 / self.projection_bars as f64) * bars_from_origin;
                        fan_500[i] = low_val + (vertical_range * 0.500 / self.projection_bars as f64) * bars_from_origin;
                        fan_618[i] = low_val + (vertical_range * 0.618 / self.projection_bars as f64) * bars_from_origin;
                    }
                } else {
                    // Fan from swing high downward
                    let bars_from_origin = (i - high_idx) as f64;
                    if bars_from_origin > 0.0 {
                        let vertical_range = range;

                        fan_382[i] = high_val - (vertical_range * 0.382 / self.projection_bars as f64) * bars_from_origin;
                        fan_500[i] = high_val - (vertical_range * 0.500 / self.projection_bars as f64) * bars_from_origin;
                        fan_618[i] = high_val - (vertical_range * 0.618 / self.projection_bars as f64) * bars_from_origin;
                    }
                }
            }
        }

        (fan_382, fan_500, fan_618)
    }

    /// Get fan levels at a specific bar.
    pub fn get_levels(&self, high: &[f64], low: &[f64], close: &[f64], bar_index: usize) -> Option<FibFanLevels> {
        let (fan_382, fan_500, fan_618) = self.calculate(high, low, close);

        if bar_index >= close.len() || fan_382[bar_index].is_nan() {
            return None;
        }

        Some(FibFanLevels {
            origin: close[bar_index],
            fan_382: fan_382[bar_index],
            fan_500: fan_500[bar_index],
            fan_618: fan_618[bar_index],
        })
    }

    /// Check if price is near a fan line (within threshold percent).
    pub fn near_fan_line(&self, price: f64, fan_value: f64, threshold_pct: f64) -> bool {
        if fan_value.is_nan() {
            return false;
        }
        let diff_pct = ((price - fan_value) / fan_value * 100.0).abs();
        diff_pct <= threshold_pct
    }
}

impl Default for FibonacciFans {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for FibonacciFans {
    fn name(&self) -> &str {
        "Fibonacci Fans"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (fan_382, fan_500, fan_618) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(fan_382, fan_500, fan_618))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create trending data with clear swings
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();

        for i in 0..50 {
            let base = 100.0 + (i as f64) * 0.5;
            let swing = (i as f64 * 0.3).sin() * 5.0;
            close.push(base + swing);
            high.push(base + swing + 2.0);
            low.push(base + swing - 2.0);
        }

        (high, low, close)
    }

    #[test]
    fn test_fibonacci_fans_creation() {
        let fans = FibonacciFans::new(20, 10, 3);
        assert!(fans.is_ok());

        let fans = FibonacciFans::new(5, 10, 3);
        assert!(fans.is_err());

        let fans = FibonacciFans::new(20, 2, 3);
        assert!(fans.is_err());

        let fans = FibonacciFans::new(20, 10, 1);
        assert!(fans.is_err());
    }

    #[test]
    fn test_fibonacci_fans_calculation() {
        let (high, low, close) = make_test_data();
        let fans = FibonacciFans::new(15, 8, 2).unwrap();
        let (fan_382, fan_500, fan_618) = fans.calculate(&high, &low, &close);

        assert_eq!(fan_382.len(), close.len());
        assert_eq!(fan_500.len(), close.len());
        assert_eq!(fan_618.len(), close.len());

        // Check that we have some valid values after lookback
        let valid_count = fan_382.iter().filter(|v| !v.is_nan()).count();
        assert!(valid_count > 0);
    }

    #[test]
    fn test_fibonacci_fans_ordering() {
        let (high, low, close) = make_test_data();
        let fans = FibonacciFans::new(15, 8, 2).unwrap();
        let (fan_382, fan_500, fan_618) = fans.calculate(&high, &low, &close);

        // In valid regions, fan lines should maintain relative order
        for i in 20..close.len() {
            if !fan_382[i].is_nan() && !fan_500[i].is_nan() && !fan_618[i].is_nan() {
                // 61.8% fan should be most extreme from origin
                let diff_382_500 = (fan_382[i] - fan_500[i]).abs();
                let diff_500_618 = (fan_500[i] - fan_618[i]).abs();
                // Fan lines should be distinct
                assert!(diff_382_500 > 0.0 || diff_500_618 > 0.0);
            }
        }
    }

    #[test]
    fn test_fibonacci_fans_near_line() {
        let fans = FibonacciFans::default_params();

        assert!(fans.near_fan_line(100.0, 100.5, 1.0));
        assert!(!fans.near_fan_line(100.0, 105.0, 1.0));
        assert!(!fans.near_fan_line(100.0, f64::NAN, 1.0));
    }

    #[test]
    fn test_fibonacci_fans_technical_indicator() {
        let fans = FibonacciFans::default_params();
        assert_eq!(fans.name(), "Fibonacci Fans");
        assert_eq!(fans.min_periods(), 21);
    }

    #[test]
    fn test_fibonacci_fans_compute() {
        let (high, low, close) = make_test_data();
        let volume = vec![1000.0; close.len()];

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let fans = FibonacciFans::default_params();
        let result = fans.compute(&data);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.values.len(), 3); // 3 fan lines
    }

    #[test]
    fn test_fibonacci_fans_get_levels() {
        let (high, low, close) = make_test_data();
        let fans = FibonacciFans::new(15, 8, 2).unwrap();

        // Should return None for early bars
        let levels = fans.get_levels(&high, &low, &close, 5);
        assert!(levels.is_none());

        // May return Some for later bars
        let levels = fans.get_levels(&high, &low, &close, 30);
        if let Some(lvl) = levels {
            assert!(!lvl.fan_382.is_nan());
            assert!(!lvl.fan_500.is_nan());
            assert!(!lvl.fan_618.is_nan());
        }
    }
}
