//! NYSE Tick - Uptick minus downtick stocks indicator.
//!
//! IND-394: The NYSE Tick measures the number of stocks trading on an
//! uptick minus those on a downtick, providing real-time market sentiment.

use super::{BreadthIndicator, BreadthSeries};
use crate::{IndicatorError, IndicatorOutput, Result};
use serde::{Deserialize, Serialize};

/// NYSE Tick reading with analysis.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TickReading {
    /// Raw tick value (upticks - downticks)
    pub value: f64,
    /// Signal classification
    pub signal: NYSETickSignal,
    /// Cumulative tick for the session
    pub cumulative: f64,
}

/// NYSE Tick signal classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NYSETickSignal {
    /// Extreme buying pressure (>= +1000)
    ExtremeBuying,
    /// Strong buying pressure (>= +500)
    StrongBuying,
    /// Moderate buying pressure (>= +200)
    ModerateBuying,
    /// Neutral trading range
    Neutral,
    /// Moderate selling pressure (<= -200)
    ModerateSelling,
    /// Strong selling pressure (<= -500)
    StrongSelling,
    /// Extreme selling pressure (<= -1000)
    ExtremeSelling,
    /// Invalid or missing data
    Unknown,
}

/// NYSE Tick daily statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NYSETickStats {
    /// Highest tick reading of the session
    pub high: f64,
    /// Lowest tick reading of the session
    pub low: f64,
    /// Average tick reading
    pub average: f64,
    /// Closing/latest tick reading
    pub close: f64,
    /// Number of extreme high readings
    pub extreme_highs: usize,
    /// Number of extreme low readings
    pub extreme_lows: usize,
    /// Cumulative tick (sum of all readings)
    pub cumulative: f64,
    /// Total readings analyzed
    pub reading_count: usize,
}

impl NYSETickStats {
    /// Calculate tick range (high - low).
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Determine session bias based on statistics.
    pub fn session_bias(&self) -> SessionBias {
        if self.average > 200.0 && self.cumulative > 0.0 {
            SessionBias::Bullish
        } else if self.average < -200.0 && self.cumulative < 0.0 {
            SessionBias::Bearish
        } else if self.extreme_highs > self.extreme_lows * 2 {
            SessionBias::Bullish
        } else if self.extreme_lows > self.extreme_highs * 2 {
            SessionBias::Bearish
        } else {
            SessionBias::Neutral
        }
    }
}

/// Session bias based on tick analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionBias {
    /// More buying pressure overall
    Bullish,
    /// More selling pressure overall
    Bearish,
    /// Balanced or unclear
    Neutral,
}

/// NYSE Tick Indicator
///
/// Measures real-time market sentiment by calculating the difference between
/// stocks trading on an uptick vs. downtick. This is a key intraday indicator
/// for day traders and scalpers.
///
/// # Formula
/// NYSE TICK = Number of Upticks - Number of Downticks
///
/// Where:
/// - Uptick: Stock's last trade was higher than previous trade
/// - Downtick: Stock's last trade was lower than previous trade
///
/// # Interpretation
/// | Tick Reading | Interpretation |
/// |--------------|----------------|
/// | >= +1000 | Extreme buying, potential reversal |
/// | +500 to +1000 | Strong buying pressure |
/// | +200 to +500 | Moderate buying |
/// | -200 to +200 | Neutral/balanced |
/// | -500 to -200 | Moderate selling |
/// | -1000 to -500 | Strong selling pressure |
/// | <= -1000 | Extreme selling, potential reversal |
///
/// # Trading Applications
/// - Scalping: Enter longs on extreme negative readings, shorts on extreme positive
/// - Trend confirmation: Strong trend has tick readings in same direction
/// - Divergence: Price making highs while tick readings decline = warning
/// - Cumulative tick: Rising cumulative indicates sustained buying
#[derive(Debug, Clone)]
pub struct NYSETick {
    /// Extreme high threshold
    extreme_high: f64,
    /// Strong buying threshold
    strong_high: f64,
    /// Moderate buying threshold
    moderate_high: f64,
    /// Moderate selling threshold
    moderate_low: f64,
    /// Strong selling threshold
    strong_low: f64,
    /// Extreme low threshold
    extreme_low: f64,
    /// Smoothing period (0 = no smoothing)
    smoothing_period: usize,
    /// EMA smoothing factor (for cumulative tick)
    ema_alpha: f64,
}

impl Default for NYSETick {
    fn default() -> Self {
        Self::new()
    }
}

impl NYSETick {
    /// Create a new NYSE Tick indicator with default thresholds.
    pub fn new() -> Self {
        Self {
            extreme_high: 1000.0,
            strong_high: 500.0,
            moderate_high: 200.0,
            moderate_low: -200.0,
            strong_low: -500.0,
            extreme_low: -1000.0,
            smoothing_period: 0,
            ema_alpha: 0.0,
        }
    }

    /// Create with custom thresholds.
    ///
    /// # Arguments
    /// * `extreme_high` - Threshold for extreme buying (default: 1000)
    /// * `strong_high` - Threshold for strong buying (default: 500)
    /// * `extreme_low` - Threshold for extreme selling (default: -1000)
    /// * `strong_low` - Threshold for strong selling (default: -500)
    pub fn with_thresholds(
        mut self,
        extreme_high: f64,
        strong_high: f64,
        strong_low: f64,
        extreme_low: f64,
    ) -> Self {
        self.extreme_high = extreme_high;
        self.strong_high = strong_high;
        self.strong_low = strong_low;
        self.extreme_low = extreme_low;
        self.moderate_high = strong_high / 2.5;
        self.moderate_low = strong_low / 2.5;
        self
    }

    /// Enable SMA smoothing.
    pub fn with_smoothing(mut self, period: usize) -> Self {
        self.smoothing_period = period;
        self
    }

    /// Enable EMA smoothing for cumulative tick.
    pub fn with_ema_smoothing(mut self, period: usize) -> Self {
        if period > 0 {
            self.ema_alpha = 2.0 / (period as f64 + 1.0);
        }
        self
    }

    /// Calculate raw tick value from uptick and downtick counts.
    pub fn calculate(&self, upticks: f64, downticks: f64) -> f64 {
        upticks - downticks
    }

    /// Calculate tick series from arrays.
    pub fn calculate_series(&self, upticks: &[f64], downticks: &[f64]) -> Vec<f64> {
        upticks
            .iter()
            .zip(downticks.iter())
            .map(|(up, down)| up - down)
            .collect()
    }

    /// Calculate smoothed tick series using SMA.
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

    /// Calculate cumulative tick (running sum).
    pub fn calculate_cumulative(&self, upticks: &[f64], downticks: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(upticks.len());
        let mut cumulative = 0.0;

        for (up, down) in upticks.iter().zip(downticks.iter()) {
            cumulative += up - down;
            result.push(cumulative);
        }

        result
    }

    /// Calculate EMA-smoothed cumulative tick.
    pub fn calculate_cumulative_ema(&self, upticks: &[f64], downticks: &[f64]) -> Vec<f64> {
        let cumulative = self.calculate_cumulative(upticks, downticks);

        if self.ema_alpha == 0.0 || cumulative.is_empty() {
            return cumulative;
        }

        let mut result = Vec::with_capacity(cumulative.len());
        let mut ema = cumulative[0];
        result.push(ema);

        for &val in cumulative.iter().skip(1) {
            ema = self.ema_alpha * val + (1.0 - self.ema_alpha) * ema;
            result.push(ema);
        }

        result
    }

    /// Interpret a single tick reading.
    pub fn interpret(&self, tick: f64) -> NYSETickSignal {
        if tick.is_nan() {
            NYSETickSignal::Unknown
        } else if tick >= self.extreme_high {
            NYSETickSignal::ExtremeBuying
        } else if tick >= self.strong_high {
            NYSETickSignal::StrongBuying
        } else if tick >= self.moderate_high {
            NYSETickSignal::ModerateBuying
        } else if tick <= self.extreme_low {
            NYSETickSignal::ExtremeSelling
        } else if tick <= self.strong_low {
            NYSETickSignal::StrongSelling
        } else if tick <= self.moderate_low {
            NYSETickSignal::ModerateSelling
        } else {
            NYSETickSignal::Neutral
        }
    }

    /// Get full tick reading with cumulative value.
    pub fn get_reading(&self, tick: f64, cumulative: f64) -> TickReading {
        TickReading {
            value: tick,
            signal: self.interpret(tick),
            cumulative,
        }
    }

    /// Calculate session statistics from tick readings.
    pub fn session_stats(&self, tick_readings: &[f64]) -> NYSETickStats {
        if tick_readings.is_empty() {
            return NYSETickStats::default();
        }

        let mut high = f64::MIN;
        let mut low = f64::MAX;
        let mut sum = 0.0;
        let mut cumulative = 0.0;
        let mut extreme_highs = 0;
        let mut extreme_lows = 0;
        let mut count = 0;

        for &tick in tick_readings {
            if tick.is_nan() {
                continue;
            }

            high = high.max(tick);
            low = low.min(tick);
            sum += tick;
            cumulative += tick;
            count += 1;

            if tick >= self.extreme_high {
                extreme_highs += 1;
            } else if tick <= self.extreme_low {
                extreme_lows += 1;
            }
        }

        NYSETickStats {
            high: if high == f64::MIN { 0.0 } else { high },
            low: if low == f64::MAX { 0.0 } else { low },
            average: if count > 0 { sum / count as f64 } else { 0.0 },
            close: *tick_readings.last().unwrap_or(&0.0),
            extreme_highs,
            extreme_lows,
            cumulative,
            reading_count: count,
        }
    }

    /// Detect tick divergence from price.
    ///
    /// Returns positive if bullish divergence (price falling, tick rising)
    /// Returns negative if bearish divergence (price rising, tick falling)
    pub fn detect_divergence(&self, prices: &[f64], ticks: &[f64], lookback: usize) -> Vec<f64> {
        let n = prices.len().min(ticks.len());
        let mut result = vec![0.0; n];

        if n < lookback + 1 {
            return result;
        }

        for i in lookback..n {
            let price_change = prices[i] - prices[i - lookback];
            let tick_change = ticks[i] - ticks[i - lookback];

            // Bullish divergence: price down, tick up
            if price_change < 0.0 && tick_change > 0.0 {
                result[i] = (tick_change / lookback as f64).abs();
            }
            // Bearish divergence: price up, tick down
            else if price_change > 0.0 && tick_change < 0.0 {
                result[i] = -(tick_change / lookback as f64).abs();
            }
        }

        result
    }

    /// Find extreme tick reversals (potential entry points).
    ///
    /// Returns +1 for bullish reversal (extreme low followed by uptick)
    /// Returns -1 for bearish reversal (extreme high followed by downtick)
    pub fn find_reversals(&self, ticks: &[f64]) -> Vec<f64> {
        let n = ticks.len();
        let mut result = vec![0.0; n];

        if n < 2 {
            return result;
        }

        for i in 1..n {
            // Bullish reversal: extreme low followed by improvement
            if ticks[i - 1] <= self.extreme_low && ticks[i] > ticks[i - 1] {
                result[i] = 1.0;
            }
            // Bearish reversal: extreme high followed by decline
            else if ticks[i - 1] >= self.extreme_high && ticks[i] < ticks[i - 1] {
                result[i] = -1.0;
            }
        }

        result
    }
}

impl BreadthIndicator for NYSETick {
    fn name(&self) -> &str {
        "NYSE Tick"
    }

    fn compute_breadth(&self, data: &BreadthSeries) -> Result<IndicatorOutput> {
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

    fn output_features(&self) -> usize {
        1
    }
}

/// NYSE Tick data series for analysis.
#[derive(Debug, Clone, Default)]
pub struct NYSETickSeries {
    /// Uptick counts per period
    pub upticks: Vec<f64>,
    /// Downtick counts per period
    pub downticks: Vec<f64>,
    /// Timestamps or bar indices
    pub timestamps: Vec<u64>,
}

impl NYSETickSeries {
    /// Create new empty series.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            upticks: Vec::with_capacity(capacity),
            downticks: Vec::with_capacity(capacity),
            timestamps: Vec::with_capacity(capacity),
        }
    }

    /// Push a new tick reading.
    pub fn push(&mut self, upticks: f64, downticks: f64, timestamp: u64) {
        self.upticks.push(upticks);
        self.downticks.push(downticks);
        self.timestamps.push(timestamp);
    }

    /// Get length of series.
    pub fn len(&self) -> usize {
        self.upticks.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.upticks.is_empty()
    }

    /// Get tick values (upticks - downticks).
    pub fn tick_values(&self) -> Vec<f64> {
        self.upticks
            .iter()
            .zip(self.downticks.iter())
            .map(|(u, d)| u - d)
            .collect()
    }

    /// Get cumulative tick values.
    pub fn cumulative_values(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.len());
        let mut cum = 0.0;

        for (up, down) in self.upticks.iter().zip(self.downticks.iter()) {
            cum += up - down;
            result.push(cum);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nyse_tick_basic() {
        let tick = NYSETick::new();

        assert!((tick.calculate(2000.0, 1000.0) - 1000.0).abs() < 1e-10);
        assert!((tick.calculate(1000.0, 2000.0) - (-1000.0)).abs() < 1e-10);
        assert!((tick.calculate(1500.0, 1500.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_nyse_tick_series() {
        let tick = NYSETick::new();

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
    fn test_nyse_tick_cumulative() {
        let tick = NYSETick::new();

        let upticks = vec![2000.0, 1500.0, 1800.0];
        let downticks = vec![1000.0, 1500.0, 1200.0];

        let result = tick.calculate_cumulative(&upticks, &downticks);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1000.0).abs() < 1e-10);
        assert!((result[1] - 1000.0).abs() < 1e-10); // 1000 + 0
        assert!((result[2] - 1600.0).abs() < 1e-10); // 1000 + 0 + 600
    }

    #[test]
    fn test_nyse_tick_interpretation() {
        let tick = NYSETick::new();

        assert_eq!(tick.interpret(1200.0), NYSETickSignal::ExtremeBuying);
        assert_eq!(tick.interpret(700.0), NYSETickSignal::StrongBuying);
        assert_eq!(tick.interpret(300.0), NYSETickSignal::ModerateBuying);
        assert_eq!(tick.interpret(50.0), NYSETickSignal::Neutral);
        assert_eq!(tick.interpret(-300.0), NYSETickSignal::ModerateSelling);
        assert_eq!(tick.interpret(-700.0), NYSETickSignal::StrongSelling);
        assert_eq!(tick.interpret(-1200.0), NYSETickSignal::ExtremeSelling);
        assert_eq!(tick.interpret(f64::NAN), NYSETickSignal::Unknown);
    }

    #[test]
    fn test_nyse_tick_smoothed() {
        let tick = NYSETick::new().with_smoothing(3);

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
    fn test_nyse_tick_session_stats() {
        let tick = NYSETick::new();

        let readings = vec![500.0, -200.0, 1100.0, -1100.0, 300.0];
        let stats = tick.session_stats(&readings);

        assert!((stats.high - 1100.0).abs() < 1e-10);
        assert!((stats.low - (-1100.0)).abs() < 1e-10);
        assert!((stats.close - 300.0).abs() < 1e-10);
        assert_eq!(stats.extreme_highs, 1);
        assert_eq!(stats.extreme_lows, 1);
        assert_eq!(stats.reading_count, 5);
    }

    #[test]
    fn test_nyse_tick_stats_bias() {
        let stats = NYSETickStats {
            high: 1000.0,
            low: -500.0,
            average: 300.0,
            close: 200.0,
            extreme_highs: 5,
            extreme_lows: 1,
            cumulative: 5000.0,
            reading_count: 20,
        };

        assert_eq!(stats.session_bias(), SessionBias::Bullish);
        assert!((stats.range() - 1500.0).abs() < 1e-10);
    }

    #[test]
    fn test_nyse_tick_divergence() {
        let tick = NYSETick::new();

        let prices = vec![100.0, 101.0, 102.0, 101.0, 100.0];
        let ticks = vec![500.0, 400.0, 300.0, 400.0, 500.0];

        let divergence = tick.detect_divergence(&prices, &ticks, 2);

        assert_eq!(divergence.len(), 5);
        // At index 4: price went from 102 to 100 (down), tick went from 300 to 500 (up)
        // This is bullish divergence
        assert!(divergence[4] > 0.0);
    }

    #[test]
    fn test_nyse_tick_reversals() {
        let tick = NYSETick::new();

        let ticks = vec![-1200.0, -1000.0, -800.0, 1200.0, 1000.0];

        let reversals = tick.find_reversals(&ticks);

        assert_eq!(reversals.len(), 5);
        // Bullish reversal at index 1: extreme low followed by improvement
        assert_eq!(reversals[1], 1.0);
        // Bearish reversal at index 4: extreme high followed by decline
        assert_eq!(reversals[4], -1.0);
    }

    #[test]
    fn test_nyse_tick_custom_thresholds() {
        let tick = NYSETick::new().with_thresholds(800.0, 400.0, -400.0, -800.0);

        assert_eq!(tick.interpret(850.0), NYSETickSignal::ExtremeBuying);
        assert_eq!(tick.interpret(500.0), NYSETickSignal::StrongBuying);
        assert_eq!(tick.interpret(-500.0), NYSETickSignal::StrongSelling);
        assert_eq!(tick.interpret(-850.0), NYSETickSignal::ExtremeSelling);
    }

    #[test]
    fn test_nyse_tick_ema_cumulative() {
        let tick = NYSETick::new().with_ema_smoothing(5);

        let upticks = vec![2000.0, 1800.0, 1600.0, 1400.0, 1500.0, 1700.0];
        let downticks = vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0];

        let ema_cum = tick.calculate_cumulative_ema(&upticks, &downticks);

        assert_eq!(ema_cum.len(), 6);
        // EMA should be smoother than raw cumulative
        let raw_cum = tick.calculate_cumulative(&upticks, &downticks);
        // Final values should be different due to smoothing
        assert!((ema_cum[5] - raw_cum[5]).abs() > 0.0);
    }

    #[test]
    fn test_nyse_tick_series_struct() {
        let mut series = NYSETickSeries::new();
        series.push(2000.0, 1000.0, 1000);
        series.push(1500.0, 1500.0, 1001);
        series.push(1800.0, 1200.0, 1002);

        assert_eq!(series.len(), 3);

        let values = series.tick_values();
        assert_eq!(values.len(), 3);
        assert!((values[0] - 1000.0).abs() < 1e-10);
        assert!((values[1] - 0.0).abs() < 1e-10);
        assert!((values[2] - 600.0).abs() < 1e-10);

        let cumulative = series.cumulative_values();
        assert!((cumulative[2] - 1600.0).abs() < 1e-10);
    }

    #[test]
    fn test_nyse_tick_get_reading() {
        let tick = NYSETick::new();
        let reading = tick.get_reading(750.0, 5000.0);

        assert!((reading.value - 750.0).abs() < 1e-10);
        assert!((reading.cumulative - 5000.0).abs() < 1e-10);
        assert_eq!(reading.signal, NYSETickSignal::StrongBuying);
    }

    #[test]
    fn test_nyse_tick_breadth_indicator() {
        let tick = NYSETick::new();
        assert_eq!(tick.name(), "NYSE Tick");
        assert_eq!(tick.min_periods(), 1);
        assert_eq!(tick.output_features(), 1);
    }

    #[test]
    fn test_nyse_tick_compute_breadth() {
        let tick = NYSETick::new();

        let mut series = BreadthSeries::new();
        series.advances = vec![2000.0, 1500.0, 1800.0];
        series.declines = vec![1000.0, 1500.0, 1200.0];
        series.unchanged = vec![500.0, 500.0, 500.0];
        series.advance_volume = vec![0.0; 3];
        series.decline_volume = vec![0.0; 3];
        series.unchanged_volume = vec![0.0; 3];

        let result = tick.compute_breadth(&series);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.values.len(), 1);
        assert_eq!(output.values[0].len(), 3);
        assert!((output.values[0][0] - 1000.0).abs() < 1e-10);
    }
}
