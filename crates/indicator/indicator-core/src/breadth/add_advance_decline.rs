//! ADD Advance/Decline - Real-Time A/D Indicator (IND-399)

use super::{BreadthIndicator, BreadthSeries};
use crate::{IndicatorError, IndicatorOutput, Result};

/// ADD Advance/Decline Configuration
#[derive(Debug, Clone)]
pub struct ADDAdvanceDeclineConfig {
    /// Smoothing period for the A/D line (0 = no smoothing)
    pub smoothing_period: usize,
    /// Use EMA instead of SMA for smoothing
    pub use_ema: bool,
    /// Strong bullish threshold
    pub strong_bullish_threshold: f64,
    /// Bullish threshold
    pub bullish_threshold: f64,
    /// Bearish threshold
    pub bearish_threshold: f64,
    /// Strong bearish threshold
    pub strong_bearish_threshold: f64,
}

impl Default for ADDAdvanceDeclineConfig {
    fn default() -> Self {
        Self {
            smoothing_period: 0,
            use_ema: false,
            strong_bullish_threshold: 1500.0,
            bullish_threshold: 500.0,
            bearish_threshold: -500.0,
            strong_bearish_threshold: -1500.0,
        }
    }
}

/// ADD Advance/Decline Indicator
///
/// Real-time advance/decline indicator that measures the net difference
/// between advancing and declining issues. This is a raw market breadth
/// measurement used for intraday trading decisions.
///
/// # Formula
/// ADD = Advancing Issues - Declining Issues
///
/// # Interpretation
/// - ADD > 1500: Strong bullish breadth
/// - ADD > 500: Bullish breadth
/// - ADD between -500 and 500: Neutral
/// - ADD < -500: Bearish breadth
/// - ADD < -1500: Strong bearish breadth
///
/// # Use Cases
/// - Real-time market direction
/// - Confirming index moves
/// - Identifying broad vs narrow rallies
/// - Divergence analysis
#[derive(Debug, Clone)]
pub struct ADDAdvanceDecline {
    config: ADDAdvanceDeclineConfig,
}

impl Default for ADDAdvanceDecline {
    fn default() -> Self {
        Self::new()
    }
}

impl ADDAdvanceDecline {
    pub fn new() -> Self {
        Self {
            config: ADDAdvanceDeclineConfig::default(),
        }
    }

    pub fn with_config(config: ADDAdvanceDeclineConfig) -> Self {
        Self { config }
    }

    pub fn with_smoothing(mut self, period: usize) -> Self {
        self.config.smoothing_period = period;
        self
    }

    pub fn with_ema(mut self) -> Self {
        self.config.use_ema = true;
        self
    }

    /// Calculate SMA
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

    /// Calculate EMA
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let multiplier = 2.0 / (period as f64 + 1.0);

        let sum: f64 = data[..period].iter().sum();
        let mut ema = sum / period as f64;
        result.push(ema);

        for i in period..data.len() {
            ema = (data[i] - ema) * multiplier + ema;
            result.push(ema);
        }

        result
    }

    /// Calculate raw ADD values
    pub fn calculate_raw(&self, advances: &[f64], declines: &[f64]) -> Vec<f64> {
        advances
            .iter()
            .zip(declines.iter())
            .map(|(a, d)| a - d)
            .collect()
    }

    /// Calculate ADD from arrays
    pub fn calculate(&self, advances: &[f64], declines: &[f64]) -> Vec<f64> {
        let raw = self.calculate_raw(advances, declines);

        if self.config.smoothing_period > 0 {
            if self.config.use_ema {
                self.calculate_ema(&raw, self.config.smoothing_period)
            } else {
                self.calculate_sma(&raw, self.config.smoothing_period)
            }
        } else {
            raw
        }
    }

    /// Calculate ADD from BreadthSeries
    pub fn calculate_series(&self, data: &BreadthSeries) -> Vec<f64> {
        self.calculate(&data.advances, &data.declines)
    }

    /// Interpret ADD value
    pub fn interpret(&self, value: f64) -> ADDSignal {
        if value.is_nan() {
            ADDSignal::Unknown
        } else if value >= self.config.strong_bullish_threshold {
            ADDSignal::StrongBullish
        } else if value >= self.config.bullish_threshold {
            ADDSignal::Bullish
        } else if value <= self.config.strong_bearish_threshold {
            ADDSignal::StrongBearish
        } else if value <= self.config.bearish_threshold {
            ADDSignal::Bearish
        } else {
            ADDSignal::Neutral
        }
    }

    /// Calculate cumulative ADD (running total)
    pub fn calculate_cumulative(&self, advances: &[f64], declines: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(advances.len());
        let mut cumulative = 0.0;

        for (a, d) in advances.iter().zip(declines.iter()) {
            cumulative += a - d;
            result.push(cumulative);
        }

        result
    }

    /// Calculate ADD ratio (advances / total)
    pub fn calculate_ratio(&self, advances: &[f64], declines: &[f64]) -> Vec<f64> {
        advances
            .iter()
            .zip(declines.iter())
            .map(|(a, d)| {
                let total = a + d;
                if total == 0.0 {
                    0.5
                } else {
                    a / total
                }
            })
            .collect()
    }

    /// Calculate ADD percentage
    pub fn calculate_percentage(&self, advances: &[f64], declines: &[f64]) -> Vec<f64> {
        advances
            .iter()
            .zip(declines.iter())
            .map(|(a, d)| {
                let total = a + d;
                if total == 0.0 {
                    0.0
                } else {
                    (a - d) / total * 100.0
                }
            })
            .collect()
    }

    /// Analyze session breadth
    pub fn session_analysis(
        &self,
        advances: &[f64],
        declines: &[f64],
    ) -> ADDSessionAnalysis {
        if advances.is_empty() {
            return ADDSessionAnalysis::default();
        }

        let add_values = self.calculate_raw(advances, declines);
        let valid: Vec<f64> = add_values.iter().filter(|v| !v.is_nan()).copied().collect();

        if valid.is_empty() {
            return ADDSessionAnalysis::default();
        }

        let high = valid.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let low = valid.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let close = *valid.last().unwrap();
        let sum: f64 = valid.iter().sum();
        let average = sum / valid.len() as f64;

        // Count bullish vs bearish readings
        let bullish_count = valid
            .iter()
            .filter(|&&v| v >= self.config.bullish_threshold)
            .count();
        let bearish_count = valid
            .iter()
            .filter(|&&v| v <= self.config.bearish_threshold)
            .count();

        let bias = if bullish_count > bearish_count * 2 {
            ADDBias::StronglyBullish
        } else if bullish_count > bearish_count {
            ADDBias::Bullish
        } else if bearish_count > bullish_count * 2 {
            ADDBias::StronglyBearish
        } else if bearish_count > bullish_count {
            ADDBias::Bearish
        } else {
            ADDBias::Neutral
        };

        ADDSessionAnalysis {
            high,
            low,
            close,
            average,
            range: high - low,
            bullish_count,
            bearish_count,
            bias,
        }
    }
}

/// ADD signal interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ADDSignal {
    /// Very positive ADD: Strong buying
    StrongBullish,
    /// Positive ADD: Buying pressure
    Bullish,
    /// Near zero: Balanced
    Neutral,
    /// Negative ADD: Selling pressure
    Bearish,
    /// Very negative ADD: Strong selling
    StrongBearish,
    /// Invalid data
    Unknown,
}

/// Session analysis results
#[derive(Debug, Clone, Default)]
pub struct ADDSessionAnalysis {
    /// Highest ADD reading
    pub high: f64,
    /// Lowest ADD reading
    pub low: f64,
    /// Closing ADD reading
    pub close: f64,
    /// Average ADD reading
    pub average: f64,
    /// Range (high - low)
    pub range: f64,
    /// Count of bullish readings
    pub bullish_count: usize,
    /// Count of bearish readings
    pub bearish_count: usize,
    /// Overall bias
    pub bias: ADDBias,
}

/// Session bias from ADD readings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ADDBias {
    StronglyBullish,
    Bullish,
    #[default]
    Neutral,
    Bearish,
    StronglyBearish,
}

impl BreadthIndicator for ADDAdvanceDecline {
    fn name(&self) -> &str {
        "ADD Advance/Decline"
    }

    fn compute_breadth(&self, data: &BreadthSeries) -> Result<IndicatorOutput> {
        let min_required = if self.config.smoothing_period > 0 {
            self.config.smoothing_period
        } else {
            1
        };

        if data.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.len(),
            });
        }

        let values = self.calculate_series(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        if self.config.smoothing_period > 0 {
            self.config.smoothing_period
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::breadth::BreadthData;

    fn create_test_series() -> BreadthSeries {
        let mut series = BreadthSeries::new();
        series.push(BreadthData::from_ad(2000.0, 1000.0)); // +1000
        series.push(BreadthData::from_ad(1800.0, 1200.0)); // +600
        series.push(BreadthData::from_ad(1500.0, 1500.0)); // 0
        series.push(BreadthData::from_ad(1200.0, 1800.0)); // -600
        series.push(BreadthData::from_ad(1000.0, 2000.0)); // -1000
        series
    }

    #[test]
    fn test_add_basic() {
        let add = ADDAdvanceDecline::new();
        let series = create_test_series();
        let result = add.calculate_series(&series);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 1000.0).abs() < 1e-10);
        assert!((result[1] - 600.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
        assert!((result[3] - (-600.0)).abs() < 1e-10);
        assert!((result[4] - (-1000.0)).abs() < 1e-10);
    }

    #[test]
    fn test_add_smoothed() {
        let add = ADDAdvanceDecline::new().with_smoothing(3);
        let series = create_test_series();
        let result = add.calculate_series(&series);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // SMA(3) of 1000, 600, 0 = 533.33...
        assert!((result[2] - 533.3333333333333).abs() < 1e-10);
    }

    #[test]
    fn test_add_ema() {
        let add = ADDAdvanceDecline::new().with_smoothing(3).with_ema();
        let series = create_test_series();
        let result = add.calculate_series(&series);

        assert_eq!(result.len(), 5);
        assert!(!result[2].is_nan());
    }

    #[test]
    fn test_add_interpretation() {
        let add = ADDAdvanceDecline::new();

        assert_eq!(add.interpret(2000.0), ADDSignal::StrongBullish);
        assert_eq!(add.interpret(800.0), ADDSignal::Bullish);
        assert_eq!(add.interpret(0.0), ADDSignal::Neutral);
        assert_eq!(add.interpret(-800.0), ADDSignal::Bearish);
        assert_eq!(add.interpret(-2000.0), ADDSignal::StrongBearish);
        assert_eq!(add.interpret(f64::NAN), ADDSignal::Unknown);
    }

    #[test]
    fn test_cumulative_add() {
        let add = ADDAdvanceDecline::new();
        let advances = vec![2000.0, 1800.0, 1500.0];
        let declines = vec![1000.0, 1200.0, 1500.0];

        let result = add.calculate_cumulative(&advances, &declines);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1000.0).abs() < 1e-10);
        assert!((result[1] - 1600.0).abs() < 1e-10);
        assert!((result[2] - 1600.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_ratio() {
        let add = ADDAdvanceDecline::new();
        let advances = vec![2000.0, 1500.0, 1000.0];
        let declines = vec![1000.0, 1500.0, 2000.0];

        let result = add.calculate_ratio(&advances, &declines);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.6666666666666666).abs() < 1e-10);
        assert!((result[1] - 0.5).abs() < 1e-10);
        assert!((result[2] - 0.3333333333333333).abs() < 1e-10);
    }

    #[test]
    fn test_add_percentage() {
        let add = ADDAdvanceDecline::new();
        let advances = vec![2000.0, 1500.0, 1000.0];
        let declines = vec![1000.0, 1500.0, 2000.0];

        let result = add.calculate_percentage(&advances, &declines);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 33.333333333333336).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - (-33.333333333333336)).abs() < 1e-10);
    }

    #[test]
    fn test_session_analysis() {
        let add = ADDAdvanceDecline::new();
        let advances = vec![2000.0, 1800.0, 1500.0, 1200.0, 1000.0];
        let declines = vec![1000.0, 1200.0, 1500.0, 1800.0, 2000.0];

        let analysis = add.session_analysis(&advances, &declines);

        assert!((analysis.high - 1000.0).abs() < 1e-10);
        assert!((analysis.low - (-1000.0)).abs() < 1e-10);
        assert!((analysis.close - (-1000.0)).abs() < 1e-10);
        assert!((analysis.range - 2000.0).abs() < 1e-10);
    }

    #[test]
    fn test_breadth_indicator_trait() {
        let add = ADDAdvanceDecline::new();
        let series = create_test_series();

        let result = add.compute_breadth(&series);
        assert!(result.is_ok());

        assert_eq!(add.min_periods(), 1);
        assert_eq!(add.name(), "ADD Advance/Decline");
    }

    #[test]
    fn test_empty_series() {
        let add = ADDAdvanceDecline::new();
        let series = BreadthSeries::new();

        let result = add.compute_breadth(&series);
        assert!(result.is_err());
    }
}
