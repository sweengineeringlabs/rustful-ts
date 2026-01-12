//! Advance/Decline Line (A/D Line) indicator.

use crate::{BreadthIndicator, BreadthSeries};
use indicator_spi::{IndicatorError, IndicatorOutput, Result};

/// Advance/Decline Line
///
/// A cumulative breadth indicator that tracks the running total of
/// net advances (advances minus declines) over time. It measures
/// the breadth of market participation and can diverge from price
/// indexes to signal potential trend changes.
///
/// # Formula
/// A/D Line = Previous A/D Line + (Advances - Declines)
///
/// Or: A/D Line = Cumulative Sum of Net Advances
///
/// # Interpretation
/// - Rising A/D Line: Broad market participation in rally
/// - Falling A/D Line: Broad market participation in decline
/// - Divergence from price index: Potential trend reversal signal
#[derive(Debug, Clone)]
pub struct AdvanceDeclineLine {
    /// Starting value for the cumulative line (default: 0)
    start_value: f64,
}

impl Default for AdvanceDeclineLine {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvanceDeclineLine {
    pub fn new() -> Self {
        Self { start_value: 0.0 }
    }

    pub fn with_start_value(start_value: f64) -> Self {
        Self { start_value }
    }

    /// Calculate A/D Line from advance/decline arrays.
    pub fn calculate(&self, advances: &[f64], declines: &[f64]) -> Vec<f64> {
        if advances.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(advances.len());
        let mut cumulative = self.start_value;

        for (adv, dec) in advances.iter().zip(declines.iter()) {
            cumulative += adv - dec;
            result.push(cumulative);
        }

        result
    }

    /// Calculate from BreadthSeries
    pub fn calculate_series(&self, data: &BreadthSeries) -> Vec<f64> {
        self.calculate(&data.advances, &data.declines)
    }
}

impl BreadthIndicator for AdvanceDeclineLine {
    fn name(&self) -> &str {
        "Advance/Decline Line"
    }

    fn compute_breadth(&self, data: &BreadthSeries) -> Result<IndicatorOutput> {
        if data.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let values = self.calculate_series(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BreadthData;

    fn create_test_series() -> BreadthSeries {
        let mut series = BreadthSeries::new();
        // Day 1: More advances than declines (+200)
        series.push(BreadthData::from_ad(1500.0, 1300.0));
        // Day 2: More declines than advances (-100)
        series.push(BreadthData::from_ad(1400.0, 1500.0));
        // Day 3: Even more advances (+400)
        series.push(BreadthData::from_ad(1800.0, 1400.0));
        // Day 4: Slight decline (-50)
        series.push(BreadthData::from_ad(1475.0, 1525.0));
        // Day 5: Strong advance (+300)
        series.push(BreadthData::from_ad(1650.0, 1350.0));
        series
    }

    #[test]
    fn test_ad_line_basic() {
        let ad = AdvanceDeclineLine::new();
        let series = create_test_series();
        let result = ad.calculate_series(&series);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 200.0).abs() < 1e-10); // +200
        assert!((result[1] - 100.0).abs() < 1e-10); // 200 - 100 = 100
        assert!((result[2] - 500.0).abs() < 1e-10); // 100 + 400 = 500
        assert!((result[3] - 450.0).abs() < 1e-10); // 500 - 50 = 450
        assert!((result[4] - 750.0).abs() < 1e-10); // 450 + 300 = 750
    }

    #[test]
    fn test_ad_line_with_start_value() {
        let ad = AdvanceDeclineLine::with_start_value(10000.0);
        let series = create_test_series();
        let result = ad.calculate_series(&series);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 10200.0).abs() < 1e-10);
        assert!((result[4] - 10750.0).abs() < 1e-10);
    }

    #[test]
    fn test_ad_line_empty() {
        let ad = AdvanceDeclineLine::new();
        let series = BreadthSeries::new();
        let result = ad.compute_breadth(&series);

        assert!(result.is_err());
    }

    #[test]
    fn test_ad_line_direct_arrays() {
        let ad = AdvanceDeclineLine::new();
        let advances = vec![100.0, 150.0, 120.0];
        let declines = vec![80.0, 100.0, 130.0];
        let result = ad.calculate(&advances, &declines);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 20.0).abs() < 1e-10); // 100 - 80
        assert!((result[1] - 70.0).abs() < 1e-10); // 20 + 50
        assert!((result[2] - 60.0).abs() < 1e-10); // 70 - 10
    }
}
