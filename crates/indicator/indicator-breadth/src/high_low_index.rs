//! New Highs vs New Lows Index indicator.

use crate::{BreadthIndicator, BreadthSeries};
use indicator_spi::{IndicatorError, IndicatorOutput, Result};

/// New Highs vs New Lows Index
///
/// Measures market breadth by tracking the number of stocks making
/// new 52-week highs versus new 52-week lows. The indicator can be
/// expressed as a ratio, difference, or percentage.
///
/// # Variants
/// 1. High-Low Ratio = New Highs / New Lows
/// 2. High-Low Difference = New Highs - New Lows
/// 3. High-Low Percent = (New Highs - New Lows) / (New Highs + New Lows) * 100
/// 4. Record High Percent = New Highs / Total Issues * 100
///
/// # Interpretation
/// - Expanding new highs: Bullish breadth confirmation
/// - Expanding new lows: Bearish breadth confirmation
/// - Divergence from price: Potential trend reversal
/// - Smoothed with MA for trend identification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighLowMethod {
    /// New Highs / New Lows ratio
    Ratio,
    /// New Highs - New Lows difference
    Difference,
    /// (New Highs - New Lows) / (New Highs + New Lows) * 100
    Percent,
    /// New Highs / Total Issues * 100
    RecordHighPercent,
}

/// High/Low data for a single period
#[derive(Debug, Clone)]
pub struct HighLowData {
    /// Number of new 52-week highs
    pub new_highs: f64,
    /// Number of new 52-week lows
    pub new_lows: f64,
    /// Total issues (optional, used for RecordHighPercent)
    pub total_issues: Option<f64>,
}

impl HighLowData {
    pub fn new(new_highs: f64, new_lows: f64) -> Self {
        Self {
            new_highs,
            new_lows,
            total_issues: None,
        }
    }

    pub fn with_total(new_highs: f64, new_lows: f64, total_issues: f64) -> Self {
        Self {
            new_highs,
            new_lows,
            total_issues: Some(total_issues),
        }
    }
}

/// Series of high/low data
#[derive(Debug, Clone, Default)]
pub struct HighLowSeries {
    pub new_highs: Vec<f64>,
    pub new_lows: Vec<f64>,
    pub total_issues: Vec<f64>,
}

impl HighLowSeries {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, data: HighLowData) {
        self.new_highs.push(data.new_highs);
        self.new_lows.push(data.new_lows);
        self.total_issues
            .push(data.total_issues.unwrap_or(data.new_highs + data.new_lows));
    }

    pub fn len(&self) -> usize {
        self.new_highs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.new_highs.is_empty()
    }
}

/// High-Low Index
#[derive(Debug, Clone)]
pub struct HighLowIndex {
    /// Calculation method
    method: HighLowMethod,
    /// Optional smoothing period (0 = no smoothing)
    smoothing_period: usize,
}

impl Default for HighLowIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl HighLowIndex {
    pub fn new() -> Self {
        Self {
            method: HighLowMethod::Percent,
            smoothing_period: 0,
        }
    }

    pub fn with_method(method: HighLowMethod) -> Self {
        Self {
            method,
            smoothing_period: 0,
        }
    }

    pub fn with_smoothing(mut self, period: usize) -> Self {
        self.smoothing_period = period;
        self
    }

    /// Calculate raw high-low index values
    fn calculate_raw(&self, data: &HighLowSeries) -> Vec<f64> {
        match self.method {
            HighLowMethod::Ratio => data
                .new_highs
                .iter()
                .zip(data.new_lows.iter())
                .map(|(h, l)| {
                    if *l == 0.0 {
                        if *h > 0.0 {
                            f64::INFINITY
                        } else {
                            1.0 // Both zero = neutral
                        }
                    } else {
                        h / l
                    }
                })
                .collect(),

            HighLowMethod::Difference => data
                .new_highs
                .iter()
                .zip(data.new_lows.iter())
                .map(|(h, l)| h - l)
                .collect(),

            HighLowMethod::Percent => data
                .new_highs
                .iter()
                .zip(data.new_lows.iter())
                .map(|(h, l)| {
                    let total = h + l;
                    if total == 0.0 {
                        0.0
                    } else {
                        (h - l) / total * 100.0
                    }
                })
                .collect(),

            HighLowMethod::RecordHighPercent => data
                .new_highs
                .iter()
                .zip(data.total_issues.iter())
                .map(|(h, t)| {
                    if *t == 0.0 {
                        0.0
                    } else {
                        h / t * 100.0
                    }
                })
                .collect(),
        }
    }

    /// Calculate SMA for smoothing
    fn calculate_sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().sum();
        result.push(sum / period as f64);

        for i in period..data.len() {
            sum = sum - data[i - period] + data[i];
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate high-low index from HighLowSeries
    pub fn calculate(&self, data: &HighLowSeries) -> Vec<f64> {
        let raw = self.calculate_raw(data);

        if self.smoothing_period > 0 {
            self.calculate_sma(&raw, self.smoothing_period)
        } else {
            raw
        }
    }

    /// Calculate from BreadthSeries (using advances/declines as proxy for highs/lows)
    /// This is useful when high/low data isn't available
    pub fn calculate_from_breadth(&self, data: &BreadthSeries) -> Vec<f64> {
        let hl_series = HighLowSeries {
            new_highs: data.advances.clone(),
            new_lows: data.declines.clone(),
            total_issues: data.total_issues(),
        };
        self.calculate(&hl_series)
    }
}

impl BreadthIndicator for HighLowIndex {
    fn name(&self) -> &str {
        "High-Low Index"
    }

    fn compute_breadth(&self, data: &BreadthSeries) -> Result<IndicatorOutput> {
        let min_required = if self.smoothing_period > 0 {
            self.smoothing_period
        } else {
            1
        };

        if data.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.len(),
            });
        }

        let values = self.calculate_from_breadth(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        if self.smoothing_period > 0 {
            self.smoothing_period
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_series() -> HighLowSeries {
        let mut series = HighLowSeries::new();
        // Various high/low scenarios
        series.push(HighLowData::with_total(100.0, 50.0, 3000.0)); // More highs
        series.push(HighLowData::with_total(80.0, 80.0, 3000.0)); // Equal
        series.push(HighLowData::with_total(50.0, 100.0, 3000.0)); // More lows
        series.push(HighLowData::with_total(120.0, 30.0, 3000.0)); // Strong highs
        series.push(HighLowData::with_total(20.0, 150.0, 3000.0)); // Strong lows
        series
    }

    #[test]
    fn test_high_low_ratio() {
        let hli = HighLowIndex::with_method(HighLowMethod::Ratio);
        let series = create_test_series();
        let result = hli.calculate(&series);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 2.0).abs() < 1e-10); // 100/50 = 2.0
        assert!((result[1] - 1.0).abs() < 1e-10); // 80/80 = 1.0
        assert!((result[2] - 0.5).abs() < 1e-10); // 50/100 = 0.5
        assert!((result[3] - 4.0).abs() < 1e-10); // 120/30 = 4.0
    }

    #[test]
    fn test_high_low_difference() {
        let hli = HighLowIndex::with_method(HighLowMethod::Difference);
        let series = create_test_series();
        let result = hli.calculate(&series);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 50.0).abs() < 1e-10); // 100-50
        assert!((result[1] - 0.0).abs() < 1e-10); // 80-80
        assert!((result[2] - (-50.0)).abs() < 1e-10); // 50-100
        assert!((result[3] - 90.0).abs() < 1e-10); // 120-30
    }

    #[test]
    fn test_high_low_percent() {
        let hli = HighLowIndex::with_method(HighLowMethod::Percent);
        let series = create_test_series();
        let result = hli.calculate(&series);

        assert_eq!(result.len(), 5);
        // (100-50)/(100+50)*100 = 33.33%
        assert!((result[0] - 33.333333333333336).abs() < 1e-10);
        // (80-80)/(80+80)*100 = 0%
        assert!((result[1] - 0.0).abs() < 1e-10);
        // (50-100)/(50+100)*100 = -33.33%
        assert!((result[2] - (-33.333333333333336)).abs() < 1e-10);
    }

    #[test]
    fn test_record_high_percent() {
        let hli = HighLowIndex::with_method(HighLowMethod::RecordHighPercent);
        let series = create_test_series();
        let result = hli.calculate(&series);

        assert_eq!(result.len(), 5);
        // 100/3000*100 = 3.33%
        assert!((result[0] - 3.333333333333333).abs() < 1e-10);
        // 120/3000*100 = 4%
        assert!((result[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_high_low_with_smoothing() {
        let hli = HighLowIndex::with_method(HighLowMethod::Difference).with_smoothing(3);
        let series = create_test_series();
        let result = hli.calculate(&series);

        assert_eq!(result.len(), 5);
        // First 2 values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Third value is SMA of first 3: (50 + 0 + -50) / 3 = 0
        assert!((result[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_high_low_zero_lows() {
        let mut series = HighLowSeries::new();
        series.push(HighLowData::new(100.0, 0.0)); // No lows
        series.push(HighLowData::new(0.0, 0.0)); // No highs or lows

        let hli = HighLowIndex::with_method(HighLowMethod::Ratio);
        let result = hli.calculate(&series);

        assert!(result[0].is_infinite()); // 100/0 = infinity
        assert!((result[1] - 1.0).abs() < 1e-10); // 0/0 = 1 (neutral)
    }

    #[test]
    fn test_from_breadth_series() {
        use crate::BreadthData;

        let mut breadth = BreadthSeries::new();
        breadth.push(BreadthData::from_ad(100.0, 50.0));
        breadth.push(BreadthData::from_ad(80.0, 80.0));

        let hli = HighLowIndex::with_method(HighLowMethod::Percent);
        let result = hli.calculate_from_breadth(&breadth);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 33.333333333333336).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
    }
}
