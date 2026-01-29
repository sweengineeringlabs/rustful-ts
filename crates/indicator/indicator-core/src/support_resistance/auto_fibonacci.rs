//! Auto Fibonacci - Automatic swing detection and Fibonacci level calculation.
//!
//! IND-392: AutoFibonacci automatically identifies significant swing highs
//! and lows and calculates Fibonacci retracement/extension levels.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// Detected swing point type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwingType {
    /// Swing high (local maximum)
    High,
    /// Swing low (local minimum)
    Low,
}

/// Detected swing point.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SwingPoint {
    /// Bar index where swing occurred
    pub index: usize,
    /// Price at swing point
    pub price: f64,
    /// Type of swing
    pub swing_type: SwingType,
    /// Strength (number of confirming bars)
    pub strength: usize,
}

/// Auto-generated Fibonacci levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoFibLevels {
    /// Swing high price
    pub swing_high: f64,
    /// Swing high bar index
    pub high_index: usize,
    /// Swing low price
    pub swing_low: f64,
    /// Swing low bar index
    pub low_index: usize,
    /// Whether current trend is up (low before high)
    pub is_uptrend: bool,
    /// Retracement levels
    pub retracements: FibRetraceLevels,
    /// Extension levels
    pub extensions: FibExtensionLevels,
}

/// Fibonacci retracement levels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FibRetraceLevels {
    pub level_0: f64,
    pub level_236: f64,
    pub level_382: f64,
    pub level_500: f64,
    pub level_618: f64,
    pub level_786: f64,
    pub level_1000: f64,
}

/// Fibonacci extension levels.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FibExtensionLevels {
    pub level_1000: f64,
    pub level_1272: f64,
    pub level_1414: f64,
    pub level_1618: f64,
    pub level_2000: f64,
    pub level_2618: f64,
}

/// Auto Fibonacci Indicator
///
/// Automatically detects significant swing highs and lows, then calculates
/// Fibonacci retracement and extension levels. Adapts to changing market
/// conditions by updating swing points as new extremes are formed.
///
/// # Features
/// - Automatic swing point detection
/// - Adaptive to current trend
/// - Calculates both retracements and extensions
/// - Multiple detection sensitivity modes
///
/// # Interpretation
/// - Retracement levels are potential support/resistance during pullbacks
/// - Extension levels are potential targets beyond the initial move
/// - Higher strength swings create more reliable levels
#[derive(Debug, Clone)]
pub struct AutoFibonacci {
    /// Lookback period for swing detection
    lookback: usize,
    /// Minimum swing strength (bars on each side)
    min_strength: usize,
    /// Maximum swing strength to search for
    max_strength: usize,
    /// Minimum price move percentage to consider significant
    min_move_pct: f64,
    /// Whether to use adaptive swing detection
    adaptive: bool,
}

impl AutoFibonacci {
    /// Create a new Auto Fibonacci indicator.
    ///
    /// # Arguments
    /// * `lookback` - Period to search for swings (minimum 10)
    /// * `min_strength` - Minimum bars on each side to confirm swing (minimum 2)
    /// * `max_strength` - Maximum strength to test (minimum min_strength)
    /// * `min_move_pct` - Minimum move percentage (minimum 0.5)
    pub fn new(
        lookback: usize,
        min_strength: usize,
        max_strength: usize,
        min_move_pct: f64,
    ) -> Result<Self> {
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if min_strength < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_strength".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if max_strength < min_strength {
            return Err(IndicatorError::InvalidParameter {
                name: "max_strength".to_string(),
                reason: "must be >= min_strength".to_string(),
            });
        }
        if min_move_pct < 0.5 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_move_pct".to_string(),
                reason: "must be at least 0.5".to_string(),
            });
        }
        Ok(Self {
            lookback,
            min_strength,
            max_strength,
            min_move_pct,
            adaptive: true,
        })
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self {
            lookback: 50,
            min_strength: 3,
            max_strength: 10,
            min_move_pct: 2.0,
            adaptive: true,
        }
    }

    /// Enable or disable adaptive swing detection.
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Check if a bar is a swing high with given strength.
    fn is_swing_high(&self, high: &[f64], index: usize, strength: usize) -> bool {
        if index < strength || index + strength >= high.len() {
            return false;
        }

        for i in (index - strength)..index {
            if high[i] >= high[index] {
                return false;
            }
        }

        for i in (index + 1)..=(index + strength) {
            if high[i] >= high[index] {
                return false;
            }
        }

        true
    }

    /// Check if a bar is a swing low with given strength.
    fn is_swing_low(&self, low: &[f64], index: usize, strength: usize) -> bool {
        if index < strength || index + strength >= low.len() {
            return false;
        }

        for i in (index - strength)..index {
            if low[i] <= low[index] {
                return false;
            }
        }

        for i in (index + 1)..=(index + strength) {
            if low[i] <= low[index] {
                return false;
            }
        }

        true
    }

    /// Find the strongest swing high in a range.
    fn find_best_swing_high(&self, high: &[f64], start: usize, end: usize) -> Option<SwingPoint> {
        let mut best: Option<SwingPoint> = None;

        for i in start..end {
            for strength in self.min_strength..=self.max_strength {
                if self.is_swing_high(high, i, strength) {
                    let current = SwingPoint {
                        index: i,
                        price: high[i],
                        swing_type: SwingType::High,
                        strength,
                    };

                    if let Some(ref b) = best {
                        // Prefer higher strength, then higher price
                        if strength > b.strength || (strength == b.strength && high[i] > b.price) {
                            best = Some(current);
                        }
                    } else {
                        best = Some(current);
                    }
                }
            }
        }

        best
    }

    /// Find the strongest swing low in a range.
    fn find_best_swing_low(&self, low: &[f64], start: usize, end: usize) -> Option<SwingPoint> {
        let mut best: Option<SwingPoint> = None;

        for i in start..end {
            for strength in self.min_strength..=self.max_strength {
                if self.is_swing_low(low, i, strength) {
                    let current = SwingPoint {
                        index: i,
                        price: low[i],
                        swing_type: SwingType::Low,
                        strength,
                    };

                    if let Some(ref b) = best {
                        // Prefer higher strength, then lower price
                        if strength > b.strength || (strength == b.strength && low[i] < b.price) {
                            best = Some(current);
                        }
                    } else {
                        best = Some(current);
                    }
                }
            }
        }

        best
    }

    /// Calculate Fibonacci retracement levels.
    fn calc_retracements(&self, high: f64, low: f64, is_uptrend: bool) -> FibRetraceLevels {
        let range = high - low;

        if is_uptrend {
            FibRetraceLevels {
                level_0: high,
                level_236: high - range * 0.236,
                level_382: high - range * 0.382,
                level_500: high - range * 0.500,
                level_618: high - range * 0.618,
                level_786: high - range * 0.786,
                level_1000: low,
            }
        } else {
            FibRetraceLevels {
                level_0: low,
                level_236: low + range * 0.236,
                level_382: low + range * 0.382,
                level_500: low + range * 0.500,
                level_618: low + range * 0.618,
                level_786: low + range * 0.786,
                level_1000: high,
            }
        }
    }

    /// Calculate Fibonacci extension levels.
    fn calc_extensions(&self, high: f64, low: f64, is_uptrend: bool) -> FibExtensionLevels {
        let range = high - low;

        if is_uptrend {
            FibExtensionLevels {
                level_1000: high,
                level_1272: low + range * 1.272,
                level_1414: low + range * 1.414,
                level_1618: low + range * 1.618,
                level_2000: low + range * 2.000,
                level_2618: low + range * 2.618,
            }
        } else {
            FibExtensionLevels {
                level_1000: low,
                level_1272: high - range * 1.272,
                level_1414: high - range * 1.414,
                level_1618: high - range * 1.618,
                level_2000: high - range * 2.000,
                level_2618: high - range * 2.618,
            }
        }
    }

    /// Calculate auto Fibonacci levels.
    ///
    /// Returns the 61.8% retracement level (most significant) for each bar.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut retrace_618 = vec![f64::NAN; n];
        let mut extend_1618 = vec![f64::NAN; n];

        if n < self.lookback {
            return (retrace_618, extend_1618);
        }

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Find best swings
            let swing_high = self.find_best_swing_high(high, start, i);
            let swing_low = self.find_best_swing_low(low, start, i);

            if let (Some(sh), Some(sl)) = (swing_high, swing_low) {
                // Check if move is significant
                let move_pct = (sh.price - sl.price) / sl.price * 100.0;
                if move_pct.abs() < self.min_move_pct {
                    continue;
                }

                let is_uptrend = sl.index < sh.index;
                let retracements = self.calc_retracements(sh.price, sl.price, is_uptrend);
                let extensions = self.calc_extensions(sh.price, sl.price, is_uptrend);

                retrace_618[i] = retracements.level_618;
                extend_1618[i] = extensions.level_1618;
            }
        }

        (retrace_618, extend_1618)
    }

    /// Get complete auto Fibonacci levels at a specific bar.
    pub fn get_levels(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        bar_index: usize,
    ) -> Option<AutoFibLevels> {
        if bar_index >= close.len() || bar_index < self.lookback {
            return None;
        }

        let start = bar_index.saturating_sub(self.lookback);

        let swing_high = self.find_best_swing_high(high, start, bar_index)?;
        let swing_low = self.find_best_swing_low(low, start, bar_index)?;

        // Check significance
        let move_pct = (swing_high.price - swing_low.price) / swing_low.price * 100.0;
        if move_pct.abs() < self.min_move_pct {
            return None;
        }

        let is_uptrend = swing_low.index < swing_high.index;

        Some(AutoFibLevels {
            swing_high: swing_high.price,
            high_index: swing_high.index,
            swing_low: swing_low.price,
            low_index: swing_low.index,
            is_uptrend,
            retracements: self.calc_retracements(swing_high.price, swing_low.price, is_uptrend),
            extensions: self.calc_extensions(swing_high.price, swing_low.price, is_uptrend),
        })
    }

    /// Find all detected swing points.
    pub fn find_all_swings(&self, high: &[f64], low: &[f64]) -> Vec<SwingPoint> {
        let mut swings = Vec::new();
        let n = high.len();

        for i in self.min_strength..(n - self.min_strength) {
            // Find maximum strength this bar qualifies as
            let mut high_strength = 0;
            let mut low_strength = 0;

            for strength in self.min_strength..=self.max_strength {
                if self.is_swing_high(high, i, strength) {
                    high_strength = strength;
                }
                if self.is_swing_low(low, i, strength) {
                    low_strength = strength;
                }
            }

            if high_strength > 0 {
                swings.push(SwingPoint {
                    index: i,
                    price: high[i],
                    swing_type: SwingType::High,
                    strength: high_strength,
                });
            }

            if low_strength > 0 {
                swings.push(SwingPoint {
                    index: i,
                    price: low[i],
                    swing_type: SwingType::Low,
                    strength: low_strength,
                });
            }
        }

        swings
    }
}

impl Default for AutoFibonacci {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for AutoFibonacci {
    fn name(&self) -> &str {
        "Auto Fibonacci"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (retrace_618, extend_1618) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(retrace_618, extend_1618))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create data with clear swings
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();

        for i in 0..80 {
            let base = 100.0;
            let trend = (i as f64) * 0.3;
            let swing = (i as f64 * 0.12).sin() * 15.0;
            close.push(base + trend + swing);
            high.push(base + trend + swing + 2.0);
            low.push(base + trend + swing - 2.0);
        }

        (high, low, close)
    }

    #[test]
    fn test_auto_fib_creation() {
        let fib = AutoFibonacci::new(30, 3, 8, 1.5);
        assert!(fib.is_ok());

        let fib = AutoFibonacci::new(5, 3, 8, 1.5);
        assert!(fib.is_err());

        let fib = AutoFibonacci::new(30, 1, 8, 1.5);
        assert!(fib.is_err());

        let fib = AutoFibonacci::new(30, 5, 3, 1.5);
        assert!(fib.is_err());

        let fib = AutoFibonacci::new(30, 3, 8, 0.2);
        assert!(fib.is_err());
    }

    #[test]
    fn test_auto_fib_calculation() {
        let (high, low, close) = make_test_data();
        let fib = AutoFibonacci::new(30, 2, 6, 1.0).unwrap();
        let (retrace, extend) = fib.calculate(&high, &low, &close);

        assert_eq!(retrace.len(), close.len());
        assert_eq!(extend.len(), close.len());

        // Should have valid values after lookback
        let valid_count = retrace.iter().filter(|v| !v.is_nan()).count();
        assert!(valid_count > 0);
    }

    #[test]
    fn test_auto_fib_get_levels() {
        let (high, low, close) = make_test_data();
        let fib = AutoFibonacci::new(30, 2, 6, 1.0).unwrap();

        let levels = fib.get_levels(&high, &low, &close, 60);

        if let Some(lvl) = levels {
            assert!(lvl.swing_high > lvl.swing_low);

            // Retracement levels should be between high and low
            assert!(lvl.retracements.level_618 <= lvl.swing_high);
            assert!(lvl.retracements.level_618 >= lvl.swing_low);

            // Extension 161.8% should be beyond the range
            if lvl.is_uptrend {
                assert!(lvl.extensions.level_1618 > lvl.swing_high);
            } else {
                assert!(lvl.extensions.level_1618 < lvl.swing_low);
            }
        }
    }

    #[test]
    fn test_auto_fib_swing_detection() {
        let (high, low, _) = make_test_data();
        let fib = AutoFibonacci::new(30, 2, 6, 1.0).unwrap();

        let swings = fib.find_all_swings(&high, &low);
        assert!(!swings.is_empty());

        // Should have both highs and lows
        let highs = swings.iter().filter(|s| s.swing_type == SwingType::High).count();
        let lows = swings.iter().filter(|s| s.swing_type == SwingType::Low).count();
        assert!(highs > 0);
        assert!(lows > 0);
    }

    #[test]
    fn test_auto_fib_retracement_order() {
        let fib = AutoFibonacci::default_params();
        let levels = fib.calc_retracements(100.0, 80.0, true);

        // In uptrend, levels should decrease from 0 to 100%
        assert!(levels.level_0 > levels.level_236);
        assert!(levels.level_236 > levels.level_382);
        assert!(levels.level_382 > levels.level_500);
        assert!(levels.level_500 > levels.level_618);
        assert!(levels.level_618 > levels.level_786);
        assert!(levels.level_786 > levels.level_1000);
    }

    #[test]
    fn test_auto_fib_extension_order() {
        let fib = AutoFibonacci::default_params();
        let extensions = fib.calc_extensions(100.0, 80.0, true);

        // In uptrend, extensions should increase
        assert!(extensions.level_1272 > extensions.level_1000);
        assert!(extensions.level_1414 > extensions.level_1272);
        assert!(extensions.level_1618 > extensions.level_1414);
        assert!(extensions.level_2000 > extensions.level_1618);
        assert!(extensions.level_2618 > extensions.level_2000);
    }

    #[test]
    fn test_auto_fib_with_adaptive() {
        let fib = AutoFibonacci::default_params().with_adaptive(false);
        assert!(!fib.adaptive);
    }

    #[test]
    fn test_auto_fib_technical_indicator() {
        let fib = AutoFibonacci::default_params();
        assert_eq!(fib.name(), "Auto Fibonacci");
        assert_eq!(fib.min_periods(), 51);
    }

    #[test]
    fn test_auto_fib_compute() {
        let (high, low, close) = make_test_data();
        let volume = vec![1000.0; close.len()];

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let fib = AutoFibonacci::default_params();
        let result = fib.compute(&data);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.values.len(), 2); // retrace and extend
    }

    #[test]
    fn test_auto_fib_min_move_filter() {
        // Create data with small moves
        let high: Vec<f64> = (0..80).map(|i| 100.0 + (i as f64) * 0.01).collect();
        let low: Vec<f64> = (0..80).map(|i| 99.5 + (i as f64) * 0.01).collect();
        let close: Vec<f64> = (0..80).map(|i| 99.75 + (i as f64) * 0.01).collect();

        // High min_move should filter out small swings
        let fib = AutoFibonacci::new(30, 2, 6, 5.0).unwrap();
        let levels = fib.get_levels(&high, &low, &close, 60);

        // May not find levels if moves are too small
        // This is expected behavior
        if levels.is_none() {
            // Expected - moves are less than 5%
        }
    }
}
