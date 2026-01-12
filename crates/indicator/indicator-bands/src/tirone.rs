//! Tirone Levels implementation.
//!
//! Support and resistance levels developed by John Tirone.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Tirone Levels.
///
/// Tirone Levels are a series of horizontal lines that identify support
/// and resistance levels. Developed by John Tirone, they divide the price
/// range into thirds:
/// - Top Line: Highest high
/// - Adjusted Mean: (Highest high + Lowest low + Close) / 3 + 1/3 of range
/// - Mean: (Highest high + Lowest low + Close) / 3
/// - Adjusted Mean: (Highest high + Lowest low + Close) / 3 - 1/3 of range
/// - Bottom Line: Lowest low
///
/// These levels help identify potential support and resistance zones.
#[derive(Debug, Clone)]
pub struct TironeLevels {
    /// Period for the highest high/lowest low lookback.
    period: usize,
}

impl TironeLevels {
    /// Create a new Tirone Levels indicator.
    ///
    /// # Arguments
    /// * `period` - Period for high/low lookback (typically 20)
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Create with default parameters (20-period).
    pub fn default_params() -> Self {
        Self::new(20)
    }

    /// Calculate Tirone Levels.
    ///
    /// Returns five series:
    /// - Top: Highest high
    /// - Upper: Mean + 1/3 of range
    /// - Middle: (High + Low + Close) / 3
    /// - Lower: Mean - 1/3 of range
    /// - Bottom: Lowest low
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> TironeLevelsOutput {
        let n = close.len();

        if n < self.period || self.period == 0 {
            return TironeLevelsOutput {
                top: vec![f64::NAN; n],
                upper: vec![f64::NAN; n],
                middle: vec![f64::NAN; n],
                lower: vec![f64::NAN; n],
                bottom: vec![f64::NAN; n],
            };
        }

        let mut top = vec![f64::NAN; self.period - 1];
        let mut upper = vec![f64::NAN; self.period - 1];
        let mut middle = vec![f64::NAN; self.period - 1];
        let mut lower = vec![f64::NAN; self.period - 1];
        let mut bottom = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;

            // Find highest high and lowest low in the period
            let highest_high = high[start..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lowest_low = low[start..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);

            let range = highest_high - lowest_low;
            let mean = (highest_high + lowest_low + close[i]) / 3.0;

            top.push(highest_high);
            upper.push(mean + range / 3.0);
            middle.push(mean);
            lower.push(mean - range / 3.0);
            bottom.push(lowest_low);
        }

        TironeLevelsOutput {
            top,
            upper,
            middle,
            lower,
            bottom,
        }
    }

    /// Calculate basic levels (top, middle, bottom) for simplified output.
    pub fn calculate_basic(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let output = self.calculate(high, low, close);
        (output.top, output.middle, output.bottom)
    }
}

/// Output structure for Tirone Levels.
#[derive(Debug, Clone)]
pub struct TironeLevelsOutput {
    /// Top line (highest high).
    pub top: Vec<f64>,
    /// Upper adjusted mean (mean + 1/3 range).
    pub upper: Vec<f64>,
    /// Middle line (mean).
    pub middle: Vec<f64>,
    /// Lower adjusted mean (mean - 1/3 range).
    pub lower: Vec<f64>,
    /// Bottom line (lowest low).
    pub bottom: Vec<f64>,
}

impl Default for TironeLevels {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for TironeLevels {
    fn name(&self) -> &str {
        "TironeLevels"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        // Return middle, upper, lower for standard 3-output format
        let (top, middle, bottom) = self.calculate_basic(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, top, bottom))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tirone_levels() {
        let tl = TironeLevels::new(5);
        let high = vec![105.0, 106.0, 107.0, 106.0, 108.0, 109.0, 108.0, 110.0];
        let low = vec![95.0, 96.0, 97.0, 96.0, 98.0, 99.0, 98.0, 100.0];
        let close = vec![100.0, 101.0, 102.0, 101.0, 103.0, 104.0, 103.0, 105.0];

        let output = tl.calculate(&high, &low, &close);

        // Check first valid values (at index 4)
        assert!((output.top[4] - 108.0).abs() < 1e-10);
        assert!((output.bottom[4] - 95.0).abs() < 1e-10);

        // Mean should be (108 + 95 + 103) / 3 = 102
        assert!((output.middle[4] - 102.0).abs() < 1e-10);

        // Range is 108 - 95 = 13, so 1/3 of range is ~4.33
        let range = 108.0 - 95.0;
        assert!((output.upper[4] - (102.0 + range / 3.0)).abs() < 1e-10);
        assert!((output.lower[4] - (102.0 - range / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_tirone_levels_ordering() {
        let tl = TironeLevels::new(10);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let output = tl.calculate(&high, &low, &close);

        // Verify ordering: top > upper > middle > lower > bottom
        for i in 10..n {
            if !output.top[i].is_nan() {
                assert!(output.top[i] >= output.upper[i], "Top should be >= upper");
                assert!(output.upper[i] >= output.middle[i], "Upper should be >= middle");
                assert!(output.middle[i] >= output.lower[i], "Middle should be >= lower");
                assert!(output.lower[i] >= output.bottom[i], "Lower should be >= bottom");
            }
        }
    }

    #[test]
    fn test_tirone_default() {
        let tl = TironeLevels::default();
        assert_eq!(tl.period, 20);
    }
}
