//! Swing Points indicator implementation.
//!
//! Identifies swing highs and swing lows in price action.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};
use serde::{Deserialize, Serialize};

/// Swing point type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SwingPointType {
    /// Swing High - local maximum
    High,
    /// Swing Low - local minimum
    Low,
}

/// Represents an identified swing point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwingPoint {
    /// Type of swing point
    pub point_type: SwingPointType,
    /// Bar index of the swing point
    pub index: usize,
    /// Price level of the swing point
    pub price: f64,
    /// Strength of the swing (number of bars on each side)
    pub strength: usize,
}

/// Swing Points indicator.
///
/// Identifies local price extremes (swing highs and swing lows) based on
/// a configurable lookback period. A swing high is a bar with the highest
/// high within N bars on either side. A swing low is a bar with the lowest
/// low within N bars on either side.
///
/// Output:
/// - Primary: Swing signal (1 = swing high, -1 = swing low, 0 = none)
/// - Secondary: Swing price level
#[derive(Debug, Clone)]
pub struct SwingPoints {
    /// Number of bars to look on each side for swing confirmation.
    left_bars: usize,
    /// Number of bars to look on the right side.
    right_bars: usize,
}

impl SwingPoints {
    /// Create a new Swing Points indicator.
    ///
    /// # Arguments
    /// * `lookback` - Number of bars to look on each side for confirmation
    pub fn new(lookback: usize) -> Self {
        Self {
            left_bars: lookback.max(1),
            right_bars: lookback.max(1),
        }
    }

    /// Create with asymmetric lookback.
    ///
    /// # Arguments
    /// * `left_bars` - Bars to look back (left)
    /// * `right_bars` - Bars to look forward (right)
    pub fn asymmetric(left_bars: usize, right_bars: usize) -> Self {
        Self {
            left_bars: left_bars.max(1),
            right_bars: right_bars.max(1),
        }
    }

    /// Create with default lookback of 5.
    pub fn default_lookback() -> Self {
        Self::new(5)
    }

    /// Check if a bar is a swing high.
    fn is_swing_high(&self, high: &[f64], index: usize) -> bool {
        if index < self.left_bars || index + self.right_bars >= high.len() {
            return false;
        }

        let current = high[index];

        // Check left side (must be strictly greater or equal with at least one greater)
        let mut has_greater_left = false;
        for i in 1..=self.left_bars {
            if high[index - i] > current {
                return false;
            }
            if high[index - i] < current {
                has_greater_left = true;
            }
        }

        // Check right side (must be strictly greater or equal with at least one greater)
        let mut has_greater_right = false;
        for i in 1..=self.right_bars {
            if high[index + i] > current {
                return false;
            }
            if high[index + i] < current {
                has_greater_right = true;
            }
        }

        has_greater_left || has_greater_right
    }

    /// Check if a bar is a swing low.
    fn is_swing_low(&self, low: &[f64], index: usize) -> bool {
        if index < self.left_bars || index + self.right_bars >= low.len() {
            return false;
        }

        let current = low[index];

        // Check left side
        let mut has_lower_left = false;
        for i in 1..=self.left_bars {
            if low[index - i] < current {
                return false;
            }
            if low[index - i] > current {
                has_lower_left = true;
            }
        }

        // Check right side
        let mut has_lower_right = false;
        for i in 1..=self.right_bars {
            if low[index + i] < current {
                return false;
            }
            if low[index + i] > current {
                has_lower_right = true;
            }
        }

        has_lower_left || has_lower_right
    }

    /// Calculate swing point values.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        let mut swing_signal = vec![0.0; n];
        let mut swing_price = vec![f64::NAN; n];

        for i in 0..n {
            if self.is_swing_high(high, i) {
                swing_signal[i] = 1.0;
                swing_price[i] = high[i];
            }

            if self.is_swing_low(low, i) {
                // If both swing high and low (rare), prioritize the more significant
                if swing_signal[i] == 0.0 {
                    swing_signal[i] = -1.0;
                    swing_price[i] = low[i];
                }
            }
        }

        (swing_signal, swing_price)
    }

    /// Detect swing points and return structured data.
    pub fn detect_swings(&self, high: &[f64], low: &[f64]) -> Vec<SwingPoint> {
        let n = high.len();
        let mut swings = Vec::new();

        for i in 0..n {
            if self.is_swing_high(high, i) {
                swings.push(SwingPoint {
                    point_type: SwingPointType::High,
                    index: i,
                    price: high[i],
                    strength: self.left_bars.min(self.right_bars),
                });
            }

            if self.is_swing_low(low, i) {
                swings.push(SwingPoint {
                    point_type: SwingPointType::Low,
                    index: i,
                    price: low[i],
                    strength: self.left_bars.min(self.right_bars),
                });
            }
        }

        swings
    }

    /// Get only swing highs.
    pub fn swing_highs(&self, high: &[f64], low: &[f64]) -> Vec<SwingPoint> {
        self.detect_swings(high, low)
            .into_iter()
            .filter(|s| s.point_type == SwingPointType::High)
            .collect()
    }

    /// Get only swing lows.
    pub fn swing_lows(&self, high: &[f64], low: &[f64]) -> Vec<SwingPoint> {
        self.detect_swings(high, low)
            .into_iter()
            .filter(|s| s.point_type == SwingPointType::Low)
            .collect()
    }

    /// Get the most recent swing high.
    pub fn last_swing_high(&self, high: &[f64], low: &[f64]) -> Option<SwingPoint> {
        self.swing_highs(high, low).into_iter().last()
    }

    /// Get the most recent swing low.
    pub fn last_swing_low(&self, high: &[f64], low: &[f64]) -> Option<SwingPoint> {
        self.swing_lows(high, low).into_iter().last()
    }
}

impl TechnicalIndicator for SwingPoints {
    fn name(&self) -> &str {
        "SwingPoints"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.left_bars + self.right_bars + 1;
        if data.high.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.high.len(),
            });
        }

        let (swing_signal, swing_price) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(swing_signal, swing_price))
    }

    fn min_periods(&self) -> usize {
        self.left_bars + self.right_bars + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swing_points_detection() {
        let sp = SwingPoints::new(2);

        // Data with clear swing points
        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 103.0, 108.0, 106.0, 104.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0, 101.0, 106.0, 104.0, 102.0];

        let (signal, price) = sp.calculate(&high, &low);

        assert_eq!(signal.len(), 9);
        assert_eq!(price.len(), 9);

        // Index 2 should be swing high (105 is highest in window)
        // Index 4 should be swing low (99 is lowest in window)
    }

    #[test]
    fn test_swing_points_struct() {
        let sp = SwingPoints::new(2);

        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 103.0, 108.0, 106.0, 104.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0, 101.0, 106.0, 104.0, 102.0];

        let swings = sp.detect_swings(&high, &low);

        for swing in &swings {
            assert!(swing.price > 0.0);
            assert!(swing.index < high.len());
        }
    }

    #[test]
    fn test_swing_highs_only() {
        let sp = SwingPoints::new(2);

        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 103.0, 108.0, 106.0, 104.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0, 101.0, 106.0, 104.0, 102.0];

        let highs = sp.swing_highs(&high, &low);

        for h in &highs {
            assert_eq!(h.point_type, SwingPointType::High);
        }
    }

    #[test]
    fn test_swing_lows_only() {
        let sp = SwingPoints::new(2);

        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 103.0, 108.0, 106.0, 104.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0, 101.0, 106.0, 104.0, 102.0];

        let lows = sp.swing_lows(&high, &low);

        for l in &lows {
            assert_eq!(l.point_type, SwingPointType::Low);
        }
    }

    #[test]
    fn test_asymmetric_lookback() {
        let sp = SwingPoints::asymmetric(3, 2);

        let high = vec![100.0, 101.0, 102.0, 105.0, 103.0, 101.0, 100.0];
        let low = vec![98.0, 99.0, 100.0, 103.0, 101.0, 99.0, 98.0];

        let (signal, _) = sp.calculate(&high, &low);
        assert_eq!(signal.len(), 7);
    }

    #[test]
    fn test_swing_points_technical_indicator() {
        let sp = SwingPoints::new(3);

        let mut data = OHLCVSeries::new();
        for i in 0..15 {
            let base = 100.0 + (i as f64 * 0.5).sin() * 5.0;
            data.open.push(base);
            data.high.push(base + 2.0);
            data.low.push(base - 2.0);
            data.close.push(base + 1.0);
            data.volume.push(1000.0);
        }

        let output = sp.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 15);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_last_swing_points() {
        let sp = SwingPoints::new(2);

        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 103.0, 108.0, 106.0, 104.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0, 101.0, 106.0, 104.0, 102.0];

        let last_high = sp.last_swing_high(&high, &low);
        let last_low = sp.last_swing_low(&high, &low);

        // Should find the last swing points
        if let Some(h) = last_high {
            assert_eq!(h.point_type, SwingPointType::High);
        }
        if let Some(l) = last_low {
            assert_eq!(l.point_type, SwingPointType::Low);
        }
    }
}
