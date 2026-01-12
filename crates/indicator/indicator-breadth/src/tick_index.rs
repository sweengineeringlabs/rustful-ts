//! NYSE Tick Index indicator.

use crate::BreadthIndicator;
use indicator_spi::{IndicatorError, IndicatorOutput, Result};

/// NYSE Tick Index
///
/// The Tick Index measures the number of stocks trading on an uptick
/// minus the number trading on a downtick at any given moment. It's
/// a real-time market sentiment indicator widely used by day traders.
///
/// # Formula
/// TICK = Number of Upticks - Number of Downticks
///
/// Where:
/// - Uptick: Stock trading at a price higher than previous trade
/// - Downtick: Stock trading at a price lower than previous trade
///
/// # Interpretation
/// - TICK > +1000: Extreme buying pressure (potentially overbought)
/// - TICK > +500: Strong buying pressure
/// - TICK between -500 and +500: Normal trading
/// - TICK < -500: Strong selling pressure
/// - TICK < -1000: Extreme selling pressure (potentially oversold)
///
/// # Common Uses
/// - Intraday sentiment gauge
/// - Scalping entry/exit timing
/// - Confirmation of breakouts/breakdowns
/// - Identifying capitulation (extreme readings)
#[derive(Debug, Clone)]
pub struct TickIndex {
    /// Extreme high threshold (default: 1000)
    extreme_high: f64,
    /// High threshold (default: 500)
    high_threshold: f64,
    /// Low threshold (default: -500)
    low_threshold: f64,
    /// Extreme low threshold (default: -1000)
    extreme_low: f64,
    /// Smoothing period for cumulative tick (0 = no smoothing)
    smoothing_period: usize,
}

impl Default for TickIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl TickIndex {
    pub fn new() -> Self {
        Self {
            extreme_high: 1000.0,
            high_threshold: 500.0,
            low_threshold: -500.0,
            extreme_low: -1000.0,
            smoothing_period: 0,
        }
    }

    pub fn with_thresholds(
        mut self,
        extreme_high: f64,
        high: f64,
        low: f64,
        extreme_low: f64,
    ) -> Self {
        self.extreme_high = extreme_high;
        self.high_threshold = high;
        self.low_threshold = low;
        self.extreme_low = extreme_low;
        self
    }

    pub fn with_smoothing(mut self, period: usize) -> Self {
        self.smoothing_period = period;
        self
    }

    /// Calculate tick value from upticks and downticks
    pub fn calculate(&self, upticks: f64, downticks: f64) -> f64 {
        upticks - downticks
    }

    /// Calculate tick series from arrays
    pub fn calculate_series(&self, upticks: &[f64], downticks: &[f64]) -> Vec<f64> {
        upticks
            .iter()
            .zip(downticks.iter())
            .map(|(up, down)| up - down)
            .collect()
    }

    /// Calculate cumulative tick (running total of tick readings)
    pub fn calculate_cumulative(&self, upticks: &[f64], downticks: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(upticks.len());
        let mut cumulative = 0.0;

        for (up, down) in upticks.iter().zip(downticks.iter()) {
            cumulative += up - down;
            result.push(cumulative);
        }

        result
    }

    /// Calculate smoothed tick using SMA
    pub fn calculate_smoothed(&self, upticks: &[f64], downticks: &[f64]) -> Vec<f64> {
        let raw = self.calculate_series(upticks, downticks);

        if self.smoothing_period == 0 || raw.len() < self.smoothing_period {
            return raw;
        }

        let mut result = vec![f64::NAN; self.smoothing_period - 1];
        let mut sum: f64 = raw[..self.smoothing_period].iter().sum();
        result.push(sum / self.smoothing_period as f64);

        for i in self.smoothing_period..raw.len() {
            sum = sum - raw[i - self.smoothing_period] + raw[i];
            result.push(sum / self.smoothing_period as f64);
        }

        result
    }

    /// Interpret tick value
    pub fn interpret(&self, tick: f64) -> TickSignal {
        if tick.is_nan() {
            TickSignal::Unknown
        } else if tick >= self.extreme_high {
            TickSignal::ExtremeBuying
        } else if tick >= self.high_threshold {
            TickSignal::StrongBuying
        } else if tick <= self.extreme_low {
            TickSignal::ExtremeSelling
        } else if tick <= self.low_threshold {
            TickSignal::StrongSelling
        } else {
            TickSignal::Neutral
        }
    }

    /// Calculate tick statistics for a trading day
    pub fn daily_stats(&self, tick_readings: &[f64]) -> TickStats {
        if tick_readings.is_empty() {
            return TickStats::default();
        }

        let mut high = f64::MIN;
        let mut low = f64::MAX;
        let mut sum = 0.0;
        let mut extreme_high_count = 0;
        let mut extreme_low_count = 0;

        for &tick in tick_readings {
            if tick.is_nan() {
                continue;
            }
            high = high.max(tick);
            low = low.min(tick);
            sum += tick;

            if tick >= self.extreme_high {
                extreme_high_count += 1;
            } else if tick <= self.extreme_low {
                extreme_low_count += 1;
            }
        }

        let count = tick_readings.iter().filter(|t| !t.is_nan()).count();

        TickStats {
            high,
            low,
            average: if count > 0 { sum / count as f64 } else { 0.0 },
            close: *tick_readings.last().unwrap_or(&0.0),
            extreme_high_count,
            extreme_low_count,
        }
    }
}

/// Tick signal interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickSignal {
    /// TICK >= extreme_high: Extreme buying, potentially overbought
    ExtremeBuying,
    /// TICK >= high_threshold: Strong buying pressure
    StrongBuying,
    /// Normal trading range
    Neutral,
    /// TICK <= low_threshold: Strong selling pressure
    StrongSelling,
    /// TICK <= extreme_low: Extreme selling, potentially oversold
    ExtremeSelling,
    /// Invalid data
    Unknown,
}

/// Daily tick statistics
#[derive(Debug, Clone, Default)]
pub struct TickStats {
    /// Highest tick reading of the day
    pub high: f64,
    /// Lowest tick reading of the day
    pub low: f64,
    /// Average tick reading
    pub average: f64,
    /// Closing tick reading
    pub close: f64,
    /// Count of extreme high readings
    pub extreme_high_count: usize,
    /// Count of extreme low readings
    pub extreme_low_count: usize,
}

impl TickStats {
    /// Calculate tick range (high - low)
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Determine bias based on average
    pub fn bias(&self) -> TickBias {
        if self.average > 200.0 {
            TickBias::Bullish
        } else if self.average < -200.0 {
            TickBias::Bearish
        } else {
            TickBias::Neutral
        }
    }
}

/// Daily tick bias
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickBias {
    Bullish,
    Bearish,
    Neutral,
}

/// Series of tick data for analysis
#[derive(Debug, Clone, Default)]
pub struct TickSeries {
    pub upticks: Vec<f64>,
    pub downticks: Vec<f64>,
}

impl TickSeries {
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

    /// Get tick values (upticks - downticks)
    pub fn tick_values(&self) -> Vec<f64> {
        self.upticks
            .iter()
            .zip(self.downticks.iter())
            .map(|(u, d)| u - d)
            .collect()
    }
}

impl BreadthIndicator for TickIndex {
    fn name(&self) -> &str {
        "Tick Index"
    }

    fn compute_breadth(&self, data: &crate::BreadthSeries) -> Result<IndicatorOutput> {
        if data.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        // Use advances as proxy for upticks when full tick data isn't available
        let values = self.calculate_series(&data.advances, &data.declines);
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
    fn test_tick_basic() {
        let tick = TickIndex::new();

        assert!((tick.calculate(2000.0, 1000.0) - 1000.0).abs() < 1e-10);
        assert!((tick.calculate(1000.0, 2000.0) - (-1000.0)).abs() < 1e-10);
        assert!((tick.calculate(1500.0, 1500.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tick_series() {
        let tick = TickIndex::new();

        let upticks = vec![2000.0, 1500.0, 1000.0, 1800.0];
        let downticks = vec![1000.0, 1500.0, 2000.0, 1200.0];

        let result = tick.calculate_series(&upticks, &downticks);

        assert_eq!(result.len(), 4);
        assert!((result[0] - 1000.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - (-1000.0)).abs() < 1e-10);
        assert!((result[3] - 600.0).abs() < 1e-10);
    }

    #[test]
    fn test_tick_cumulative() {
        let tick = TickIndex::new();

        let upticks = vec![2000.0, 1500.0, 1800.0];
        let downticks = vec![1000.0, 1500.0, 1200.0];

        let result = tick.calculate_cumulative(&upticks, &downticks);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1000.0).abs() < 1e-10); // 1000
        assert!((result[1] - 1000.0).abs() < 1e-10); // 1000 + 0
        assert!((result[2] - 1600.0).abs() < 1e-10); // 1000 + 0 + 600
    }

    #[test]
    fn test_tick_interpretation() {
        let tick = TickIndex::new();

        assert_eq!(tick.interpret(1200.0), TickSignal::ExtremeBuying);
        assert_eq!(tick.interpret(700.0), TickSignal::StrongBuying);
        assert_eq!(tick.interpret(200.0), TickSignal::Neutral);
        assert_eq!(tick.interpret(-700.0), TickSignal::StrongSelling);
        assert_eq!(tick.interpret(-1200.0), TickSignal::ExtremeSelling);
    }

    #[test]
    fn test_tick_smoothed() {
        let tick = TickIndex::new().with_smoothing(3);

        let upticks = vec![2000.0, 1500.0, 1000.0, 1800.0, 1600.0];
        let downticks = vec![1000.0, 1500.0, 2000.0, 1200.0, 1400.0];

        let result = tick.calculate_smoothed(&upticks, &downticks);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // SMA of 1000, 0, -1000 = 0
        assert!((result[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tick_stats() {
        let tick = TickIndex::new();

        let readings = vec![500.0, -200.0, 1100.0, -1100.0, 300.0];
        let stats = tick.daily_stats(&readings);

        assert!((stats.high - 1100.0).abs() < 1e-10);
        assert!((stats.low - (-1100.0)).abs() < 1e-10);
        assert!((stats.close - 300.0).abs() < 1e-10);
        assert_eq!(stats.extreme_high_count, 1);
        assert_eq!(stats.extreme_low_count, 1);
    }

    #[test]
    fn test_tick_stats_bias() {
        let stats = TickStats {
            high: 1000.0,
            low: -500.0,
            average: 300.0,
            close: 200.0,
            extreme_high_count: 2,
            extreme_low_count: 0,
        };

        assert_eq!(stats.bias(), TickBias::Bullish);
        assert!((stats.range() - 1500.0).abs() < 1e-10);
    }

    #[test]
    fn test_tick_series_struct() {
        let mut series = TickSeries::new();
        series.push(2000.0, 1000.0);
        series.push(1500.0, 1500.0);

        let values = series.tick_values();
        assert_eq!(values.len(), 2);
        assert!((values[0] - 1000.0).abs() < 1e-10);
        assert!((values[1] - 0.0).abs() < 1e-10);
    }
}
