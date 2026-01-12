//! Pivot Highs/Lows indicator implementation.
//!
//! Identifies pivot points with configurable lookback periods.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};
use serde::{Deserialize, Serialize};

/// Pivot point type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PivotType {
    /// Pivot High
    High,
    /// Pivot Low
    Low,
}

/// Represents an identified pivot point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PivotPoint {
    /// Type of pivot
    pub pivot_type: PivotType,
    /// Bar index of the pivot
    pub index: usize,
    /// Price level of the pivot
    pub price: f64,
    /// Left lookback that confirmed this pivot
    pub left_bars: usize,
    /// Right lookback that confirmed this pivot
    pub right_bars: usize,
    /// Whether this pivot is confirmed (right bars have passed)
    pub confirmed: bool,
}

/// Pivot Highs/Lows indicator.
///
/// A more flexible version of swing point detection that allows different
/// lookback periods for left and right confirmation. This is the standard
/// pivot point detection used in many trading platforms.
///
/// Pivot High: A bar's high is higher than the highs of N bars on either side
/// Pivot Low: A bar's low is lower than the lows of N bars on either side
///
/// The pivot is confirmed only after the right-side bars have formed.
///
/// Output:
/// - Primary: Pivot signal (1 = pivot high, -1 = pivot low, 0 = none)
/// - Secondary: Pivot price level
/// - Tertiary: Confirmation status (1 = confirmed, 0 = unconfirmed)
#[derive(Debug, Clone)]
pub struct PivotHighsLows {
    /// Number of bars to look left for confirmation.
    left_bars: usize,
    /// Number of bars to look right for confirmation.
    right_bars: usize,
    /// Whether to include unconfirmed (real-time) pivots.
    include_unconfirmed: bool,
}

impl PivotHighsLows {
    /// Create a new Pivot Highs/Lows indicator.
    ///
    /// # Arguments
    /// * `left_bars` - Bars to look back (left)
    /// * `right_bars` - Bars to look forward (right) for confirmation
    pub fn new(left_bars: usize, right_bars: usize) -> Self {
        Self {
            left_bars: left_bars.max(1),
            right_bars: right_bars.max(1),
            include_unconfirmed: false,
        }
    }

    /// Create with symmetric lookback.
    pub fn symmetric(bars: usize) -> Self {
        Self::new(bars, bars)
    }

    /// Create with default lookback of 5 bars each side.
    pub fn default_lookback() -> Self {
        Self::symmetric(5)
    }

    /// Include unconfirmed pivots (useful for real-time analysis).
    pub fn with_unconfirmed(mut self) -> Self {
        self.include_unconfirmed = true;
        self
    }

    /// Check if a bar is a pivot high.
    fn is_pivot_high(&self, high: &[f64], index: usize, confirmed: bool) -> bool {
        if index < self.left_bars {
            return false;
        }

        let right_check = if confirmed {
            self.right_bars
        } else {
            0
        };

        if confirmed && index + self.right_bars >= high.len() {
            return false;
        }

        let current = high[index];

        // Check left side - must be strictly higher
        for i in 1..=self.left_bars {
            if high[index - i] >= current {
                return false;
            }
        }

        // Check right side if confirming
        if confirmed {
            for i in 1..=right_check {
                if high[index + i] >= current {
                    return false;
                }
            }
        }

        true
    }

    /// Check if a bar is a pivot low.
    fn is_pivot_low(&self, low: &[f64], index: usize, confirmed: bool) -> bool {
        if index < self.left_bars {
            return false;
        }

        let right_check = if confirmed {
            self.right_bars
        } else {
            0
        };

        if confirmed && index + self.right_bars >= low.len() {
            return false;
        }

        let current = low[index];

        // Check left side - must be strictly lower
        for i in 1..=self.left_bars {
            if low[index - i] <= current {
                return false;
            }
        }

        // Check right side if confirming
        if confirmed {
            for i in 1..=right_check {
                if low[index + i] <= current {
                    return false;
                }
            }
        }

        true
    }

    /// Calculate pivot point values.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        let mut pivot_signal = vec![0.0; n];
        let mut pivot_price = vec![f64::NAN; n];
        let mut pivot_confirmed = vec![0.0; n];

        for i in 0..n {
            // Check confirmed pivots first
            if self.is_pivot_high(high, i, true) {
                pivot_signal[i] = 1.0;
                pivot_price[i] = high[i];
                pivot_confirmed[i] = 1.0;
            } else if self.include_unconfirmed && self.is_pivot_high(high, i, false) {
                pivot_signal[i] = 1.0;
                pivot_price[i] = high[i];
                pivot_confirmed[i] = 0.0;
            }

            if self.is_pivot_low(low, i, true) {
                if pivot_signal[i] == 0.0 {
                    pivot_signal[i] = -1.0;
                    pivot_price[i] = low[i];
                    pivot_confirmed[i] = 1.0;
                }
            } else if self.include_unconfirmed && self.is_pivot_low(low, i, false) {
                if pivot_signal[i] == 0.0 {
                    pivot_signal[i] = -1.0;
                    pivot_price[i] = low[i];
                    pivot_confirmed[i] = 0.0;
                }
            }
        }

        (pivot_signal, pivot_price, pivot_confirmed)
    }

    /// Detect pivot points and return structured data.
    pub fn detect_pivots(&self, high: &[f64], low: &[f64]) -> Vec<PivotPoint> {
        let n = high.len();
        let mut pivots = Vec::new();

        for i in 0..n {
            // Confirmed pivot high
            if self.is_pivot_high(high, i, true) {
                pivots.push(PivotPoint {
                    pivot_type: PivotType::High,
                    index: i,
                    price: high[i],
                    left_bars: self.left_bars,
                    right_bars: self.right_bars,
                    confirmed: true,
                });
            } else if self.include_unconfirmed && self.is_pivot_high(high, i, false) {
                pivots.push(PivotPoint {
                    pivot_type: PivotType::High,
                    index: i,
                    price: high[i],
                    left_bars: self.left_bars,
                    right_bars: self.right_bars,
                    confirmed: false,
                });
            }

            // Confirmed pivot low
            if self.is_pivot_low(low, i, true) {
                pivots.push(PivotPoint {
                    pivot_type: PivotType::Low,
                    index: i,
                    price: low[i],
                    left_bars: self.left_bars,
                    right_bars: self.right_bars,
                    confirmed: true,
                });
            } else if self.include_unconfirmed && self.is_pivot_low(low, i, false) {
                pivots.push(PivotPoint {
                    pivot_type: PivotType::Low,
                    index: i,
                    price: low[i],
                    left_bars: self.left_bars,
                    right_bars: self.right_bars,
                    confirmed: false,
                });
            }
        }

        pivots
    }

    /// Get only confirmed pivot highs.
    pub fn pivot_highs(&self, high: &[f64], low: &[f64]) -> Vec<PivotPoint> {
        self.detect_pivots(high, low)
            .into_iter()
            .filter(|p| p.pivot_type == PivotType::High && p.confirmed)
            .collect()
    }

    /// Get only confirmed pivot lows.
    pub fn pivot_lows(&self, high: &[f64], low: &[f64]) -> Vec<PivotPoint> {
        self.detect_pivots(high, low)
            .into_iter()
            .filter(|p| p.pivot_type == PivotType::Low && p.confirmed)
            .collect()
    }

    /// Get the N most recent pivot points.
    pub fn recent_pivots(&self, high: &[f64], low: &[f64], count: usize) -> Vec<PivotPoint> {
        let pivots = self.detect_pivots(high, low);
        let confirmed: Vec<_> = pivots.into_iter().filter(|p| p.confirmed).collect();

        confirmed.into_iter().rev().take(count).collect()
    }

    /// Calculate support/resistance levels from recent pivots.
    pub fn support_resistance(
        &self,
        high: &[f64],
        low: &[f64],
        num_levels: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let pivots = self.detect_pivots(high, low);

        let resistance: Vec<f64> = pivots
            .iter()
            .filter(|p| p.pivot_type == PivotType::High && p.confirmed)
            .rev()
            .take(num_levels)
            .map(|p| p.price)
            .collect();

        let support: Vec<f64> = pivots
            .iter()
            .filter(|p| p.pivot_type == PivotType::Low && p.confirmed)
            .rev()
            .take(num_levels)
            .map(|p| p.price)
            .collect();

        (resistance, support)
    }
}

impl TechnicalIndicator for PivotHighsLows {
    fn name(&self) -> &str {
        "PivotHighsLows"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.left_bars + self.right_bars + 1;
        if data.high.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.high.len(),
            });
        }

        let (pivot_signal, pivot_price, pivot_confirmed) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::triple(pivot_signal, pivot_price, pivot_confirmed))
    }

    fn min_periods(&self) -> usize {
        self.left_bars + self.right_bars + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pivot_highs_lows_detection() {
        let pivot = PivotHighsLows::new(2, 2);

        // Clear pivot points
        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 103.0, 108.0, 106.0, 104.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0, 101.0, 106.0, 104.0, 102.0];

        let (signal, price, confirmed) = pivot.calculate(&high, &low);

        assert_eq!(signal.len(), 9);
        assert_eq!(price.len(), 9);
        assert_eq!(confirmed.len(), 9);
    }

    #[test]
    fn test_pivot_struct() {
        let pivot = PivotHighsLows::new(2, 2);

        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 103.0, 108.0, 106.0, 104.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0, 101.0, 106.0, 104.0, 102.0];

        let pivots = pivot.detect_pivots(&high, &low);

        for p in &pivots {
            assert!(p.price > 0.0);
            assert!(p.index < high.len());
        }
    }

    #[test]
    fn test_pivot_highs_only() {
        let pivot = PivotHighsLows::new(2, 2);

        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 103.0, 108.0, 106.0, 104.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0, 101.0, 106.0, 104.0, 102.0];

        let highs = pivot.pivot_highs(&high, &low);

        for h in &highs {
            assert_eq!(h.pivot_type, PivotType::High);
            assert!(h.confirmed);
        }
    }

    #[test]
    fn test_pivot_lows_only() {
        let pivot = PivotHighsLows::new(2, 2);

        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 103.0, 108.0, 106.0, 104.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0, 101.0, 106.0, 104.0, 102.0];

        let lows = pivot.pivot_lows(&high, &low);

        for l in &lows {
            assert_eq!(l.pivot_type, PivotType::Low);
            assert!(l.confirmed);
        }
    }

    #[test]
    fn test_support_resistance() {
        let pivot = PivotHighsLows::new(2, 2);

        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 103.0, 108.0, 106.0, 104.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0, 101.0, 106.0, 104.0, 102.0];

        let (resistance, support) = pivot.support_resistance(&high, &low, 3);

        // All resistance levels should be positive
        for r in &resistance {
            assert!(*r > 0.0);
        }

        // All support levels should be positive
        for s in &support {
            assert!(*s > 0.0);
        }
    }

    #[test]
    fn test_symmetric_constructor() {
        let pivot = PivotHighsLows::symmetric(3);
        assert_eq!(pivot.left_bars, 3);
        assert_eq!(pivot.right_bars, 3);
    }

    #[test]
    fn test_unconfirmed_pivots() {
        let pivot = PivotHighsLows::new(2, 2).with_unconfirmed();

        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0];

        let pivots = pivot.detect_pivots(&high, &low);

        // May include unconfirmed pivots at the end
        assert!(!pivots.is_empty() || high.len() < pivot.left_bars + pivot.right_bars + 1);
    }

    #[test]
    fn test_recent_pivots() {
        let pivot = PivotHighsLows::new(2, 2);

        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 103.0, 108.0, 106.0, 104.0];
        let low = vec![98.0, 100.0, 103.0, 101.0, 99.0, 101.0, 106.0, 104.0, 102.0];

        let recent = pivot.recent_pivots(&high, &low, 2);

        // Should return at most 2 pivots
        assert!(recent.len() <= 2);
    }

    #[test]
    fn test_pivot_technical_indicator() {
        let pivot = PivotHighsLows::new(3, 3);

        let mut data = OHLCVSeries::new();
        for i in 0..15 {
            let base = 100.0 + (i as f64 * 0.5).sin() * 5.0;
            data.open.push(base);
            data.high.push(base + 2.0);
            data.low.push(base - 2.0);
            data.close.push(base + 1.0);
            data.volume.push(1000.0);
        }

        let output = pivot.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 15);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }
}
