//! Percent of Stocks Above Moving Average indicator.

use crate::{BreadthIndicator, BreadthSeries};
use indicator_spi::{IndicatorError, IndicatorOutput, Result};

/// Percent of Stocks Above Moving Average
///
/// Measures market breadth by tracking what percentage of stocks in an
/// index are trading above their X-day moving average. This provides
/// insight into overall market participation and momentum.
///
/// # Common Variants
/// - % Above 200-day MA: Long-term trend health
/// - % Above 50-day MA: Intermediate-term trend
/// - % Above 20-day MA: Short-term momentum
///
/// # Interpretation
/// - Above 70-80%: Market is broadly strong (potentially overbought)
/// - Below 20-30%: Market is broadly weak (potentially oversold)
/// - Divergence from price: Narrowing participation, potential reversal
///
/// Note: This indicator requires pre-calculated data showing how many
/// stocks are above their MA, which is typically provided by data vendors.
#[derive(Debug, Clone)]
pub struct PercentAboveMA {
    /// MA period being tracked (e.g., 200, 50, 20)
    ma_period: usize,
    /// Optional smoothing period (0 = no smoothing)
    smoothing_period: usize,
    /// Overbought threshold (default: 70%)
    overbought_threshold: f64,
    /// Oversold threshold (default: 30%)
    oversold_threshold: f64,
}

impl Default for PercentAboveMA {
    fn default() -> Self {
        Self::new(200)
    }
}

impl PercentAboveMA {
    pub fn new(ma_period: usize) -> Self {
        Self {
            ma_period,
            smoothing_period: 0,
            overbought_threshold: 70.0,
            oversold_threshold: 30.0,
        }
    }

    /// Create for 200-day MA (long-term)
    pub fn ma_200() -> Self {
        Self::new(200)
    }

    /// Create for 50-day MA (intermediate)
    pub fn ma_50() -> Self {
        Self::new(50)
    }

    /// Create for 20-day MA (short-term)
    pub fn ma_20() -> Self {
        Self::new(20)
    }

    pub fn with_smoothing(mut self, period: usize) -> Self {
        self.smoothing_period = period;
        self
    }

    pub fn with_thresholds(mut self, overbought: f64, oversold: f64) -> Self {
        self.overbought_threshold = overbought;
        self.oversold_threshold = oversold;
        self
    }

    /// Get the MA period this indicator tracks
    pub fn ma_period(&self) -> usize {
        self.ma_period
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

    /// Calculate percent above MA from counts
    ///
    /// # Arguments
    /// * `above_count` - Number of stocks above MA each period
    /// * `total_count` - Total number of stocks each period
    pub fn calculate(&self, above_count: &[f64], total_count: &[f64]) -> Vec<f64> {
        let raw: Vec<f64> = above_count
            .iter()
            .zip(total_count.iter())
            .map(|(above, total)| {
                if *total == 0.0 {
                    0.0
                } else {
                    (above / total) * 100.0
                }
            })
            .collect();

        if self.smoothing_period > 0 {
            self.calculate_sma(&raw, self.smoothing_period)
        } else {
            raw
        }
    }

    /// Calculate from pre-computed percentage values (0-100)
    pub fn calculate_from_percent(&self, percent_values: &[f64]) -> Vec<f64> {
        if self.smoothing_period > 0 {
            self.calculate_sma(percent_values, self.smoothing_period)
        } else {
            percent_values.to_vec()
        }
    }

    /// Calculate from BreadthSeries using advances as proxy for "above MA"
    /// This approximation uses: above = advances, total = advances + declines + unchanged
    pub fn calculate_from_breadth(&self, data: &BreadthSeries) -> Vec<f64> {
        let totals = data.total_issues();
        self.calculate(&data.advances, &totals)
    }

    /// Check if current value indicates overbought condition
    pub fn is_overbought(&self, value: f64) -> bool {
        !value.is_nan() && value >= self.overbought_threshold
    }

    /// Check if current value indicates oversold condition
    pub fn is_oversold(&self, value: f64) -> bool {
        !value.is_nan() && value <= self.oversold_threshold
    }

    /// Get market condition based on current value
    pub fn market_condition(&self, value: f64) -> MarketCondition {
        if value.is_nan() {
            MarketCondition::Unknown
        } else if value >= self.overbought_threshold {
            MarketCondition::Overbought
        } else if value <= self.oversold_threshold {
            MarketCondition::Oversold
        } else {
            MarketCondition::Neutral
        }
    }
}

/// Market condition based on percent above MA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketCondition {
    Overbought,
    Oversold,
    Neutral,
    Unknown,
}

impl BreadthIndicator for PercentAboveMA {
    fn name(&self) -> &str {
        "Percent Above MA"
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

/// Data structure for pre-calculated percent above MA values
#[derive(Debug, Clone, Default)]
pub struct PercentAboveMASeries {
    /// Percent of stocks above MA each period (0-100)
    pub values: Vec<f64>,
}

impl PercentAboveMASeries {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, percent: f64) {
        self.values.push(percent);
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BreadthData;

    #[test]
    fn test_percent_above_ma_basic() {
        let pama = PercentAboveMA::new(200);

        let above = vec![1500.0, 1600.0, 1400.0, 1800.0, 1200.0];
        let total = vec![2000.0, 2000.0, 2000.0, 2000.0, 2000.0];

        let result = pama.calculate(&above, &total);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 75.0).abs() < 1e-10);
        assert!((result[1] - 80.0).abs() < 1e-10);
        assert!((result[2] - 70.0).abs() < 1e-10);
        assert!((result[3] - 90.0).abs() < 1e-10);
        assert!((result[4] - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_percent_above_ma_with_smoothing() {
        let pama = PercentAboveMA::new(200).with_smoothing(3);

        let above = vec![1500.0, 1600.0, 1400.0, 1800.0, 1200.0];
        let total = vec![2000.0, 2000.0, 2000.0, 2000.0, 2000.0];

        let result = pama.calculate(&above, &total);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // SMA of 75, 80, 70 = 75
        assert!((result[2] - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_percent_above_ma_thresholds() {
        let pama = PercentAboveMA::new(200).with_thresholds(80.0, 20.0);

        assert!(pama.is_overbought(85.0));
        assert!(!pama.is_overbought(75.0));
        assert!(pama.is_oversold(15.0));
        assert!(!pama.is_oversold(25.0));
    }

    #[test]
    fn test_market_condition() {
        let pama = PercentAboveMA::new(200);

        assert_eq!(pama.market_condition(75.0), MarketCondition::Overbought);
        assert_eq!(pama.market_condition(25.0), MarketCondition::Oversold);
        assert_eq!(pama.market_condition(50.0), MarketCondition::Neutral);
        assert_eq!(pama.market_condition(f64::NAN), MarketCondition::Unknown);
    }

    #[test]
    fn test_from_percent_values() {
        let pama = PercentAboveMA::new(50);
        let percent = vec![65.0, 70.0, 72.0, 68.0, 75.0];

        let result = pama.calculate_from_percent(&percent);

        assert_eq!(result, percent);
    }

    #[test]
    fn test_from_breadth_series() {
        let pama = PercentAboveMA::new(200);

        let mut series = BreadthSeries::new();
        series.push(BreadthData::from_ad(1500.0, 500.0)); // 1500/2000 = 75%
        series.push(BreadthData::from_ad(1600.0, 400.0)); // 1600/2000 = 80%

        let result = pama.calculate_from_breadth(&series);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 75.0).abs() < 1e-10);
        assert!((result[1] - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_preset_periods() {
        let ma200 = PercentAboveMA::ma_200();
        let ma50 = PercentAboveMA::ma_50();
        let ma20 = PercentAboveMA::ma_20();

        assert_eq!(ma200.ma_period(), 200);
        assert_eq!(ma50.ma_period(), 50);
        assert_eq!(ma20.ma_period(), 20);
    }

    #[test]
    fn test_zero_total() {
        let pama = PercentAboveMA::new(200);

        let above = vec![100.0, 0.0];
        let total = vec![0.0, 0.0];

        let result = pama.calculate(&above, &total);

        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
    }
}
