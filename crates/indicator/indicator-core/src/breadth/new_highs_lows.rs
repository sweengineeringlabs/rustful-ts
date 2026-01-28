//! New Highs/New Lows indicator.

use super::{BreadthIndicator, BreadthSeries};
use crate::{IndicatorError, IndicatorOutput, Result, OHLCVSeries, TechnicalIndicator};

/// Output mode for New Highs/New Lows indicator.
///
/// Determines how the indicator value is calculated from the
/// new highs and new lows counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NewHighsLowsMode {
    /// New Highs - New Lows (raw difference)
    #[default]
    Difference,
    /// New Highs / New Lows ratio
    Ratio,
    /// (New Highs - New Lows) / (New Highs + New Lows) * 100
    Percent,
}

/// New Highs/New Lows Indicator
///
/// Tracks the difference between stocks making new highs versus new lows
/// over a configurable lookback period. This is a market breadth indicator
/// that helps gauge overall market strength or weakness.
///
/// # Algorithm
/// 1. For each stock, check if current high > highest high over lookback period (new high)
/// 2. Check if current low < lowest low over lookback period (new low)
/// 3. Calculate output based on mode: difference, ratio, or percent
///
/// # Interpretation
/// - Positive values: More stocks making new highs than new lows (bullish)
/// - Negative values: More stocks making new lows than new highs (bearish)
/// - Divergence from price: Potential trend reversal warning
/// - Extremes: May indicate overbought/oversold conditions
///
/// # Default Parameters
/// - Period: 252 (52 weeks of trading days)
/// - Output Mode: Difference
#[derive(Debug, Clone)]
pub struct NewHighsLows {
    /// Lookback period for determining new highs/lows (default: 252 for 52-week)
    period: usize,
    /// Output calculation mode
    output_mode: NewHighsLowsMode,
}

impl Default for NewHighsLows {
    fn default() -> Self {
        Self::new()
    }
}

impl NewHighsLows {
    /// Create a new NewHighsLows indicator with default settings.
    pub fn new() -> Self {
        Self {
            period: 252,
            output_mode: NewHighsLowsMode::Difference,
        }
    }

    /// Create a new NewHighsLows indicator with a custom period.
    pub fn with_period(period: usize) -> Self {
        Self {
            period,
            output_mode: NewHighsLowsMode::Difference,
        }
    }

    /// Set the output mode.
    pub fn with_mode(mut self, mode: NewHighsLowsMode) -> Self {
        self.output_mode = mode;
        self
    }

    /// Get the lookback period.
    pub fn period(&self) -> usize {
        self.period
    }

    /// Get the output mode.
    pub fn output_mode(&self) -> NewHighsLowsMode {
        self.output_mode
    }

    /// Calculate the indicator value from new highs and new lows counts.
    fn calculate_value(&self, new_highs: f64, new_lows: f64) -> f64 {
        match self.output_mode {
            NewHighsLowsMode::Difference => new_highs - new_lows,
            NewHighsLowsMode::Ratio => {
                if new_lows == 0.0 {
                    if new_highs > 0.0 {
                        f64::INFINITY
                    } else {
                        1.0 // Both zero = neutral
                    }
                } else {
                    new_highs / new_lows
                }
            }
            NewHighsLowsMode::Percent => {
                let total = new_highs + new_lows;
                if total == 0.0 {
                    0.0
                } else {
                    (new_highs - new_lows) / total * 100.0
                }
            }
        }
    }

    /// Check if current high is a new high (greater than highest high over lookback period).
    fn is_new_high(&self, highs: &[f64], current_idx: usize) -> bool {
        if current_idx < self.period {
            return false;
        }

        let current_high = highs[current_idx];
        let start_idx = current_idx - self.period;

        // Check if current high exceeds all highs in the lookback window
        for i in start_idx..current_idx {
            if highs[i] >= current_high {
                return false;
            }
        }
        true
    }

    /// Check if current low is a new low (less than lowest low over lookback period).
    fn is_new_low(&self, lows: &[f64], current_idx: usize) -> bool {
        if current_idx < self.period {
            return false;
        }

        let current_low = lows[current_idx];
        let start_idx = current_idx - self.period;

        // Check if current low is below all lows in the lookback window
        for i in start_idx..current_idx {
            if lows[i] <= current_low {
                return false;
            }
        }
        true
    }

    /// Calculate new highs/lows for a single OHLCV series (single stock).
    ///
    /// Returns (new_highs_flags, new_lows_flags) where each is 1.0 or 0.0.
    pub fn calculate_single(&self, data: &OHLCVSeries) -> (Vec<f64>, Vec<f64>) {
        let len = data.high.len();
        let mut new_highs = vec![0.0; len];
        let mut new_lows = vec![0.0; len];

        for i in self.period..len {
            if self.is_new_high(&data.high, i) {
                new_highs[i] = 1.0;
            }
            if self.is_new_low(&data.low, i) {
                new_lows[i] = 1.0;
            }
        }

        (new_highs, new_lows)
    }

    /// Calculate the indicator from pre-computed new highs and new lows counts.
    ///
    /// This is useful when you have aggregated breadth data from multiple stocks.
    pub fn calculate_from_counts(&self, new_highs: &[f64], new_lows: &[f64]) -> Vec<f64> {
        new_highs
            .iter()
            .zip(new_lows.iter())
            .map(|(h, l)| self.calculate_value(*h, *l))
            .collect()
    }

    /// Calculate from BreadthSeries (using advances/declines as proxy for new highs/lows).
    ///
    /// This is useful when the breadth data represents aggregated new high/low counts.
    pub fn calculate_from_breadth(&self, data: &BreadthSeries) -> Vec<f64> {
        self.calculate_from_counts(&data.advances, &data.declines)
    }
}

impl TechnicalIndicator for NewHighsLows {
    fn name(&self) -> &str {
        "New Highs/New Lows"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.len() <= self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.len(),
            });
        }

        let (new_highs, new_lows) = self.calculate_single(data);
        let values = self.calculate_from_counts(&new_highs, &new_lows);

        // Return multiple outputs: indicator value, new highs count, new lows count
        Ok(IndicatorOutput::triple(values, new_highs, new_lows))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }
}

impl BreadthIndicator for NewHighsLows {
    fn name(&self) -> &str {
        "New Highs/New Lows"
    }

    fn compute_breadth(&self, data: &BreadthSeries) -> Result<IndicatorOutput> {
        if data.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let values = self.calculate_from_breadth(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::breadth::BreadthData;

    fn create_test_ohlcv() -> OHLCVSeries {
        // Create 20 bars of data with some new highs and lows
        OHLCVSeries {
            open: vec![
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
            ],
            high: vec![
                105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
                115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0,
            ],
            low: vec![
                95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
                105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0,
            ],
            close: vec![
                102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
            ],
            volume: vec![
                1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
                2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0,
            ],
        }
    }

    fn create_test_breadth_series() -> BreadthSeries {
        let mut series = BreadthSeries::new();
        // Various new highs/lows scenarios
        series.push(BreadthData::from_ad(100.0, 50.0));  // More highs
        series.push(BreadthData::from_ad(80.0, 80.0));   // Equal
        series.push(BreadthData::from_ad(50.0, 100.0));  // More lows
        series.push(BreadthData::from_ad(120.0, 30.0));  // Strong highs
        series.push(BreadthData::from_ad(20.0, 150.0));  // Strong lows
        series
    }

    #[test]
    fn test_new_highs_lows_difference_mode() {
        let nhl = NewHighsLows::new().with_mode(NewHighsLowsMode::Difference);
        let series = create_test_breadth_series();
        let result = nhl.calculate_from_breadth(&series);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 50.0).abs() < 1e-10);   // 100 - 50
        assert!((result[1] - 0.0).abs() < 1e-10);    // 80 - 80
        assert!((result[2] - (-50.0)).abs() < 1e-10); // 50 - 100
        assert!((result[3] - 90.0).abs() < 1e-10);   // 120 - 30
        assert!((result[4] - (-130.0)).abs() < 1e-10); // 20 - 150
    }

    #[test]
    fn test_new_highs_lows_ratio_mode() {
        let nhl = NewHighsLows::new().with_mode(NewHighsLowsMode::Ratio);
        let series = create_test_breadth_series();
        let result = nhl.calculate_from_breadth(&series);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 2.0).abs() < 1e-10);    // 100/50 = 2.0
        assert!((result[1] - 1.0).abs() < 1e-10);    // 80/80 = 1.0
        assert!((result[2] - 0.5).abs() < 1e-10);    // 50/100 = 0.5
        assert!((result[3] - 4.0).abs() < 1e-10);    // 120/30 = 4.0
    }

    #[test]
    fn test_new_highs_lows_percent_mode() {
        let nhl = NewHighsLows::new().with_mode(NewHighsLowsMode::Percent);
        let series = create_test_breadth_series();
        let result = nhl.calculate_from_breadth(&series);

        assert_eq!(result.len(), 5);
        // (100-50)/(100+50)*100 = 33.33%
        assert!((result[0] - 33.333333333333336).abs() < 1e-10);
        // (80-80)/(80+80)*100 = 0%
        assert!((result[1] - 0.0).abs() < 1e-10);
        // (50-100)/(50+100)*100 = -33.33%
        assert!((result[2] - (-33.333333333333336)).abs() < 1e-10);
    }

    #[test]
    fn test_new_highs_lows_ratio_zero_lows() {
        let nhl = NewHighsLows::new().with_mode(NewHighsLowsMode::Ratio);

        let mut series = BreadthSeries::new();
        series.push(BreadthData::from_ad(100.0, 0.0)); // No lows
        series.push(BreadthData::from_ad(0.0, 0.0));   // No highs or lows

        let result = nhl.calculate_from_breadth(&series);

        assert!(result[0].is_infinite()); // 100/0 = infinity
        assert!((result[1] - 1.0).abs() < 1e-10); // 0/0 = 1 (neutral)
    }

    #[test]
    fn test_new_highs_lows_single_stock() {
        let nhl = NewHighsLows::with_period(5);
        let data = create_test_ohlcv();
        let (new_highs, new_lows) = nhl.calculate_single(&data);

        assert_eq!(new_highs.len(), 20);
        assert_eq!(new_lows.len(), 20);

        // First 5 values should be 0 (not enough lookback)
        for i in 0..5 {
            assert!((new_highs[i] - 0.0).abs() < 1e-10);
            assert!((new_lows[i] - 0.0).abs() < 1e-10);
        }

        // In an uptrend, we should see new highs but no new lows
        // Since highs keep increasing, most should be new highs
        let total_new_highs: f64 = new_highs[5..].iter().sum();
        assert!(total_new_highs > 0.0, "Expected some new highs in uptrend");

        // Since lows keep increasing, none should be new lows
        let total_new_lows: f64 = new_lows[5..].iter().sum();
        assert!((total_new_lows - 0.0).abs() < 1e-10, "Expected no new lows in uptrend");
    }

    #[test]
    fn test_new_highs_lows_downtrend() {
        let nhl = NewHighsLows::with_period(5);

        // Create downtrend data
        let data = OHLCVSeries {
            open: vec![
                120.0, 119.0, 118.0, 117.0, 116.0, 115.0, 114.0, 113.0, 112.0, 111.0,
                110.0, 109.0, 108.0, 107.0, 106.0,
            ],
            high: vec![
                125.0, 124.0, 123.0, 122.0, 121.0, 120.0, 119.0, 118.0, 117.0, 116.0,
                115.0, 114.0, 113.0, 112.0, 111.0,
            ],
            low: vec![
                115.0, 114.0, 113.0, 112.0, 111.0, 110.0, 109.0, 108.0, 107.0, 106.0,
                105.0, 104.0, 103.0, 102.0, 101.0,
            ],
            close: vec![
                118.0, 117.0, 116.0, 115.0, 114.0, 113.0, 112.0, 111.0, 110.0, 109.0,
                108.0, 107.0, 106.0, 105.0, 104.0,
            ],
            volume: vec![
                1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
                2000.0, 2100.0, 2200.0, 2300.0, 2400.0,
            ],
        };

        let (new_highs, new_lows) = nhl.calculate_single(&data);

        // In a downtrend, we should see new lows but no new highs
        let total_new_highs: f64 = new_highs[5..].iter().sum();
        let total_new_lows: f64 = new_lows[5..].iter().sum();

        assert!((total_new_highs - 0.0).abs() < 1e-10, "Expected no new highs in downtrend");
        assert!(total_new_lows > 0.0, "Expected some new lows in downtrend");
    }

    #[test]
    fn test_new_highs_lows_technical_indicator_trait() {
        let nhl = NewHighsLows::with_period(5);
        let data = create_test_ohlcv();

        let result = nhl.compute(&data);
        assert!(result.is_ok());

        let output = result.unwrap();
        // primary = indicator values, secondary = new_highs, tertiary = new_lows
        assert_eq!(output.primary.len(), 20);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
        assert_eq!(output.secondary.unwrap().len(), 20);
    }

    #[test]
    fn test_new_highs_lows_breadth_indicator_trait() {
        let nhl = NewHighsLows::new();
        let series = create_test_breadth_series();

        let result = nhl.compute_breadth(&series);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.primary.len(), 5);
    }

    #[test]
    fn test_new_highs_lows_insufficient_data() {
        let nhl = NewHighsLows::with_period(10);

        let data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![105.0; 5],
            low: vec![95.0; 5],
            close: vec![102.0; 5],
            volume: vec![1000.0; 5],
        };

        let result = nhl.compute(&data);
        assert!(result.is_err());

        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 11);
            assert_eq!(got, 5);
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    #[test]
    fn test_new_highs_lows_default() {
        let nhl = NewHighsLows::default();
        assert_eq!(nhl.period(), 252);
        assert_eq!(nhl.output_mode(), NewHighsLowsMode::Difference);
    }

    #[test]
    fn test_new_highs_lows_builder_pattern() {
        let nhl = NewHighsLows::with_period(20).with_mode(NewHighsLowsMode::Percent);
        assert_eq!(nhl.period(), 20);
        assert_eq!(nhl.output_mode(), NewHighsLowsMode::Percent);
    }
}
