//! Cumulative Tick - Running Tick Total (IND-397)

use super::BreadthIndicator;
use crate::{IndicatorError, IndicatorOutput, Result};

/// Cumulative Tick Configuration
#[derive(Debug, Clone)]
pub struct CumulativeTickConfig {
    /// Reset cumulative at specific threshold (0 = no reset)
    pub reset_threshold: f64,
    /// Smoothing period for smoothed cumulative (0 = no smoothing)
    pub smoothing_period: usize,
    /// Bullish threshold for interpretation
    pub bullish_threshold: f64,
    /// Bearish threshold for interpretation
    pub bearish_threshold: f64,
}

impl Default for CumulativeTickConfig {
    fn default() -> Self {
        Self {
            reset_threshold: 0.0,
            smoothing_period: 0,
            bullish_threshold: 10000.0,
            bearish_threshold: -10000.0,
        }
    }
}

/// Cumulative Tick Indicator
///
/// Tracks the running total of tick readings throughout a trading session.
/// The cumulative tick provides insight into intraday buying and selling pressure
/// and overall market direction.
///
/// # Formula
/// Cumulative Tick = Sum of all (Upticks - Downticks) readings
///
/// # Interpretation
/// - Rising Cumulative Tick: Persistent buying pressure
/// - Falling Cumulative Tick: Persistent selling pressure
/// - Positive values: Net buying bias for the session
/// - Negative values: Net selling bias for the session
/// - Extreme readings can indicate capitulation or exhaustion
///
/// # Use Cases
/// - Intraday trend confirmation
/// - Identifying institutional order flow
/// - Measuring session buying/selling pressure
/// - Detecting capitulation in selloffs
#[derive(Debug, Clone)]
pub struct CumulativeTick {
    config: CumulativeTickConfig,
}

impl Default for CumulativeTick {
    fn default() -> Self {
        Self::new()
    }
}

impl CumulativeTick {
    pub fn new() -> Self {
        Self {
            config: CumulativeTickConfig::default(),
        }
    }

    pub fn with_config(config: CumulativeTickConfig) -> Self {
        Self { config }
    }

    pub fn with_smoothing(mut self, period: usize) -> Self {
        self.config.smoothing_period = period;
        self
    }

    pub fn with_reset_threshold(mut self, threshold: f64) -> Self {
        self.config.reset_threshold = threshold;
        self
    }

    /// Calculate cumulative tick from upticks and downticks arrays
    pub fn calculate(&self, upticks: &[f64], downticks: &[f64]) -> Vec<f64> {
        if upticks.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(upticks.len());
        let mut cumulative = 0.0;

        for (up, down) in upticks.iter().zip(downticks.iter()) {
            let tick = up - down;
            cumulative += tick;

            // Optional reset at threshold
            if self.config.reset_threshold != 0.0 && cumulative.abs() >= self.config.reset_threshold
            {
                cumulative = 0.0;
            }

            result.push(cumulative);
        }

        if self.config.smoothing_period > 0 {
            self.smooth(&result)
        } else {
            result
        }
    }

    /// Calculate cumulative tick from raw tick values
    pub fn calculate_from_ticks(&self, ticks: &[f64]) -> Vec<f64> {
        if ticks.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(ticks.len());
        let mut cumulative = 0.0;

        for tick in ticks {
            cumulative += tick;

            if self.config.reset_threshold != 0.0 && cumulative.abs() >= self.config.reset_threshold
            {
                cumulative = 0.0;
            }

            result.push(cumulative);
        }

        if self.config.smoothing_period > 0 {
            self.smooth(&result)
        } else {
            result
        }
    }

    /// Apply SMA smoothing
    fn smooth(&self, data: &[f64]) -> Vec<f64> {
        let period = self.config.smoothing_period;
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

    /// Interpret cumulative tick value
    pub fn interpret(&self, value: f64) -> CumulativeTickSignal {
        if value.is_nan() {
            CumulativeTickSignal::Unknown
        } else if value >= self.config.bullish_threshold {
            CumulativeTickSignal::StrongBullish
        } else if value > 0.0 {
            CumulativeTickSignal::Bullish
        } else if value <= self.config.bearish_threshold {
            CumulativeTickSignal::StrongBearish
        } else {
            CumulativeTickSignal::Bearish
        }
    }

    /// Calculate session statistics
    pub fn session_stats(&self, cumulative_ticks: &[f64]) -> CumulativeTickStats {
        if cumulative_ticks.is_empty() {
            return CumulativeTickStats::default();
        }

        let valid: Vec<f64> = cumulative_ticks
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();

        if valid.is_empty() {
            return CumulativeTickStats::default();
        }

        let high = valid.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let low = valid.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let close = *valid.last().unwrap();
        let sum: f64 = valid.iter().sum();
        let average = sum / valid.len() as f64;

        // Calculate trend direction changes
        let mut direction_changes = 0;
        for i in 2..valid.len() {
            let prev_delta = valid[i - 1] - valid[i - 2];
            let curr_delta = valid[i] - valid[i - 1];
            if prev_delta.signum() != curr_delta.signum() && prev_delta != 0.0 && curr_delta != 0.0
            {
                direction_changes += 1;
            }
        }

        CumulativeTickStats {
            high,
            low,
            close,
            average,
            range: high - low,
            direction_changes,
        }
    }
}

/// Cumulative Tick signal interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CumulativeTickSignal {
    /// Strong bullish cumulative tick
    StrongBullish,
    /// Positive cumulative tick
    Bullish,
    /// Negative cumulative tick
    Bearish,
    /// Strong bearish cumulative tick
    StrongBearish,
    /// Invalid data
    Unknown,
}

/// Cumulative Tick session statistics
#[derive(Debug, Clone, Default)]
pub struct CumulativeTickStats {
    /// Highest cumulative tick reading
    pub high: f64,
    /// Lowest cumulative tick reading
    pub low: f64,
    /// Closing cumulative tick
    pub close: f64,
    /// Average cumulative tick
    pub average: f64,
    /// Range (high - low)
    pub range: f64,
    /// Number of direction changes
    pub direction_changes: usize,
}

/// Series of tick data for cumulative analysis
#[derive(Debug, Clone, Default)]
pub struct CumulativeTickSeries {
    pub upticks: Vec<f64>,
    pub downticks: Vec<f64>,
}

impl CumulativeTickSeries {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, upticks: f64, downticks: f64) {
        self.upticks.push(upticks);
        self.downticks.push(downticks);
    }

    pub fn len(&self) -> usize {
        self.upticks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.upticks.is_empty()
    }
}

impl BreadthIndicator for CumulativeTick {
    fn name(&self) -> &str {
        "Cumulative Tick"
    }

    fn compute_breadth(&self, data: &super::BreadthSeries) -> Result<IndicatorOutput> {
        if data.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        // Use advances/declines as proxy for upticks/downticks
        let values = self.calculate(&data.advances, &data.declines);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cumulative_tick_basic() {
        let cum_tick = CumulativeTick::new();
        let upticks = vec![1500.0, 1600.0, 1400.0, 1700.0, 1300.0];
        let downticks = vec![1000.0, 1400.0, 1600.0, 1200.0, 1700.0];

        let result = cum_tick.calculate(&upticks, &downticks);

        assert_eq!(result.len(), 5);
        // Day 1: 1500 - 1000 = 500
        assert!((result[0] - 500.0).abs() < 1e-10);
        // Day 2: 500 + (1600 - 1400) = 700
        assert!((result[1] - 700.0).abs() < 1e-10);
        // Day 3: 700 + (1400 - 1600) = 500
        assert!((result[2] - 500.0).abs() < 1e-10);
        // Day 4: 500 + (1700 - 1200) = 1000
        assert!((result[3] - 1000.0).abs() < 1e-10);
        // Day 5: 1000 + (1300 - 1700) = 600
        assert!((result[4] - 600.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_tick_from_ticks() {
        let cum_tick = CumulativeTick::new();
        let ticks = vec![500.0, 200.0, -200.0, 500.0, -400.0];

        let result = cum_tick.calculate_from_ticks(&ticks);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 500.0).abs() < 1e-10);
        assert!((result[1] - 700.0).abs() < 1e-10);
        assert!((result[2] - 500.0).abs() < 1e-10);
        assert!((result[3] - 1000.0).abs() < 1e-10);
        assert!((result[4] - 600.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_tick_with_smoothing() {
        let cum_tick = CumulativeTick::new().with_smoothing(3);
        let ticks = vec![100.0, 200.0, 300.0, 400.0, 500.0];

        let result = cum_tick.calculate_from_ticks(&ticks);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Cumulative: 100, 300, 600, 1000, 1500
        // SMA(3) of cumulative: (100+300+600)/3 = 333.33...
        assert!((result[2] - 333.3333333333333).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_tick_with_reset() {
        let cum_tick = CumulativeTick::new().with_reset_threshold(500.0);
        let ticks = vec![200.0, 200.0, 200.0, 100.0, 100.0];

        let result = cum_tick.calculate_from_ticks(&ticks);

        assert_eq!(result.len(), 5);
        // 200, 400, then reset (600 >= 500), then 100, 200
        assert!((result[0] - 200.0).abs() < 1e-10);
        assert!((result[1] - 400.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10); // Reset
        assert!((result[3] - 100.0).abs() < 1e-10);
        assert!((result[4] - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_tick_interpretation() {
        let cum_tick = CumulativeTick::new();

        assert_eq!(
            cum_tick.interpret(15000.0),
            CumulativeTickSignal::StrongBullish
        );
        assert_eq!(cum_tick.interpret(5000.0), CumulativeTickSignal::Bullish);
        assert_eq!(cum_tick.interpret(-5000.0), CumulativeTickSignal::Bearish);
        assert_eq!(
            cum_tick.interpret(-15000.0),
            CumulativeTickSignal::StrongBearish
        );
        assert_eq!(cum_tick.interpret(f64::NAN), CumulativeTickSignal::Unknown);
    }

    #[test]
    fn test_session_stats() {
        let cum_tick = CumulativeTick::new();
        let cumulative = vec![100.0, 200.0, 150.0, 300.0, 250.0];

        let stats = cum_tick.session_stats(&cumulative);

        assert!((stats.high - 300.0).abs() < 1e-10);
        assert!((stats.low - 100.0).abs() < 1e-10);
        assert!((stats.close - 250.0).abs() < 1e-10);
        assert!((stats.range - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_tick_series() {
        let mut series = CumulativeTickSeries::new();
        series.push(1500.0, 1000.0);
        series.push(1600.0, 1400.0);

        assert_eq!(series.len(), 2);
        assert!(!series.is_empty());
    }

    #[test]
    fn test_breadth_indicator_trait() {
        use crate::breadth::{BreadthData, BreadthSeries};

        let cum_tick = CumulativeTick::new();
        let mut series = BreadthSeries::new();
        series.push(BreadthData::from_ad(1500.0, 1000.0));
        series.push(BreadthData::from_ad(1600.0, 1400.0));

        let result = cum_tick.compute_breadth(&series);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.values.len(), 2);
    }

    #[test]
    fn test_empty_data() {
        let cum_tick = CumulativeTick::new();
        let result = cum_tick.calculate(&[], &[]);
        assert!(result.is_empty());
    }
}
