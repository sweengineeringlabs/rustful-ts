//! Extended Donchian Channel Indicators
//!
//! Additional indicators based on Donchian's work and the Turtle Trading system.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use crate::bands::DonchianChannels;

/// Donchian Channel Width - Measures channel width as a percentage.
///
/// Width = (Upper - Lower) / Middle * 100
#[derive(Debug, Clone)]
pub struct DonchianWidth {
    period: usize,
}

impl DonchianWidth {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate channel width as percentage.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let dc = DonchianChannels::new(self.period);
        let (upper, middle, lower) = dc.calculate(high, low);

        upper.iter()
            .zip(middle.iter())
            .zip(lower.iter())
            .map(|((&u, &m), &l)| {
                if u.is_nan() || m.is_nan() || l.is_nan() || m.abs() < 1e-10 {
                    f64::NAN
                } else {
                    (u - l) / m * 100.0
                }
            })
            .collect()
    }
}

impl TechnicalIndicator for DonchianWidth {
    fn name(&self) -> &str {
        "Donchian Width"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Four-Week Rule - Donchian's classic breakout system.
///
/// Buy when price exceeds 4-week (20-day) high
/// Sell when price falls below 4-week (20-day) low
#[derive(Debug, Clone)]
pub struct FourWeekRule {
    period: usize,
}

impl FourWeekRule {
    pub fn new() -> Self {
        Self { period: 20 }
    }

    pub fn with_period(period: usize) -> Self {
        Self { period }
    }

    /// Calculate breakout signals.
    /// Returns 1.0 for buy, -1.0 for sell, 0.0 for neutral.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let dc = DonchianChannels::new(self.period);
        let (upper, _, lower) = dc.calculate(high, low);
        let n = close.len();

        let mut signals = vec![0.0; n];

        for i in self.period..n {
            // Use previous bar's channel (not including current bar)
            let prev_upper = upper[i - 1];
            let prev_lower = lower[i - 1];

            if prev_upper.is_nan() || prev_lower.is_nan() {
                continue;
            }

            // Buy signal: close breaks above previous upper channel
            if close[i] > prev_upper {
                signals[i] = 1.0;
            }
            // Sell signal: close breaks below previous lower channel
            else if close[i] < prev_lower {
                signals[i] = -1.0;
            }
        }

        signals
    }
}

impl Default for FourWeekRule {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for FourWeekRule {
    fn name(&self) -> &str {
        "Four-Week Rule"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for FourWeekRule {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signals = self.calculate(&data.high, &data.low, &data.close);

        if signals.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let last = *signals.last().unwrap();
        if last > 0.5 {
            Ok(IndicatorSignal::Bullish)
        } else if last < -0.5 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let signals = self.calculate(&data.high, &data.low, &data.close);

        Ok(signals.iter().map(|&s| {
            if s > 0.5 {
                IndicatorSignal::Bullish
            } else if s < -0.5 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

/// Turtle Trading Entry - The famous Turtle Trading entry system.
///
/// System 1: 20-day breakout for entry, 10-day for exit
/// System 2: 55-day breakout for entry, 20-day for exit
#[derive(Debug, Clone)]
pub struct TurtleEntry {
    entry_period: usize,
    exit_period: usize,
}

/// Output for Turtle Entry system.
#[derive(Debug, Clone)]
pub struct TurtleEntryOutput {
    /// Entry channel upper (for long entries)
    pub entry_upper: Vec<f64>,
    /// Entry channel lower (for short entries)
    pub entry_lower: Vec<f64>,
    /// Exit channel upper (for short exits)
    pub exit_upper: Vec<f64>,
    /// Exit channel lower (for long exits)
    pub exit_lower: Vec<f64>,
    /// Entry signals: 1.0 = long entry, -1.0 = short entry
    pub entry_signals: Vec<f64>,
}

impl TurtleEntry {
    /// System 1: 20-day entry, 10-day exit
    pub fn system1() -> Self {
        Self { entry_period: 20, exit_period: 10 }
    }

    /// System 2: 55-day entry, 20-day exit
    pub fn system2() -> Self {
        Self { entry_period: 55, exit_period: 20 }
    }

    pub fn with_periods(entry: usize, exit: usize) -> Self {
        Self { entry_period: entry, exit_period: exit }
    }

    /// Calculate Turtle entry signals.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> TurtleEntryOutput {
        let n = close.len();

        let entry_dc = DonchianChannels::new(self.entry_period);
        let exit_dc = DonchianChannels::new(self.exit_period);

        let (entry_upper, _, entry_lower) = entry_dc.calculate(high, low);
        let (exit_upper, _, exit_lower) = exit_dc.calculate(high, low);

        let mut entry_signals = vec![0.0; n];

        let start = self.entry_period.max(self.exit_period);
        for i in start..n {
            let prev_entry_upper = entry_upper[i - 1];
            let prev_entry_lower = entry_lower[i - 1];

            if prev_entry_upper.is_nan() || prev_entry_lower.is_nan() {
                continue;
            }

            // Long entry: close breaks above entry upper
            if close[i] > prev_entry_upper {
                entry_signals[i] = 1.0;
            }
            // Short entry: close breaks below entry lower
            else if close[i] < prev_entry_lower {
                entry_signals[i] = -1.0;
            }
        }

        TurtleEntryOutput {
            entry_upper,
            entry_lower,
            exit_upper,
            exit_lower,
            entry_signals,
        }
    }
}

impl TechnicalIndicator for TurtleEntry {
    fn name(&self) -> &str {
        "Turtle Entry"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.entry_period.max(self.exit_period);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let output = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(output.entry_upper, output.entry_lower, output.entry_signals))
    }

    fn min_periods(&self) -> usize {
        self.entry_period.max(self.exit_period)
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for TurtleEntry {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate(&data.high, &data.low, &data.close);

        if output.entry_signals.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let last = *output.entry_signals.last().unwrap();
        if last > 0.5 {
            Ok(IndicatorSignal::Bullish)
        } else if last < -0.5 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate(&data.high, &data.low, &data.close);

        Ok(output.entry_signals.iter().map(|&s| {
            if s > 0.5 {
                IndicatorSignal::Bullish
            } else if s < -0.5 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

/// Turtle Trading Exit - Exit signals for the Turtle system.
#[derive(Debug, Clone)]
pub struct TurtleExit {
    exit_period: usize,
}

impl TurtleExit {
    pub fn system1() -> Self {
        Self { exit_period: 10 }
    }

    pub fn system2() -> Self {
        Self { exit_period: 20 }
    }

    pub fn with_period(period: usize) -> Self {
        Self { exit_period: period }
    }

    /// Calculate exit signals.
    /// Returns: (exit_upper, exit_lower, exit_signals)
    /// Exit signal: 1.0 = exit short (hit upper), -1.0 = exit long (hit lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let dc = DonchianChannels::new(self.exit_period);
        let (upper, _, lower) = dc.calculate(high, low);

        let mut exit_signals = vec![0.0; n];

        for i in self.exit_period..n {
            let prev_upper = upper[i - 1];
            let prev_lower = lower[i - 1];

            if prev_upper.is_nan() || prev_lower.is_nan() {
                continue;
            }

            // Exit short: close breaks above exit upper
            if close[i] > prev_upper {
                exit_signals[i] = 1.0;
            }
            // Exit long: close breaks below exit lower
            else if close[i] < prev_lower {
                exit_signals[i] = -1.0;
            }
        }

        (upper, lower, exit_signals)
    }
}

impl TechnicalIndicator for TurtleExit {
    fn name(&self) -> &str {
        "Turtle Exit"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.exit_period {
            return Err(IndicatorError::InsufficientData {
                required: self.exit_period,
                got: data.close.len(),
            });
        }

        let (upper, lower, signals) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(upper, lower, signals))
    }

    fn min_periods(&self) -> usize {
        self.exit_period
    }

    fn output_features(&self) -> usize {
        3
    }
}

/// 5/20 Day Breakout - Dual timeframe breakout system.
///
/// Uses 5-day channel for timing, 20-day for trend confirmation.
#[derive(Debug, Clone)]
pub struct DualBreakout {
    short_period: usize,
    long_period: usize,
}

impl DualBreakout {
    pub fn new() -> Self {
        Self { short_period: 5, long_period: 20 }
    }

    pub fn with_periods(short: usize, long: usize) -> Self {
        Self { short_period: short, long_period: long }
    }

    /// Calculate dual breakout signals.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let short_dc = DonchianChannels::new(self.short_period);
        let long_dc = DonchianChannels::new(self.long_period);

        let (short_upper, _, short_lower) = short_dc.calculate(high, low);
        let (long_upper, _, long_lower) = long_dc.calculate(high, low);

        let mut signals = vec![0.0; n];
        let start = self.long_period;

        for i in start..n {
            let s_upper = short_upper[i - 1];
            let s_lower = short_lower[i - 1];
            let l_upper = long_upper[i - 1];
            let l_lower = long_lower[i - 1];

            if s_upper.is_nan() || l_upper.is_nan() {
                continue;
            }

            // Strong buy: breaks both short AND long term resistance
            if close[i] > s_upper && close[i] > l_upper {
                signals[i] = 2.0;
            }
            // Moderate buy: breaks short-term resistance
            else if close[i] > s_upper {
                signals[i] = 1.0;
            }
            // Strong sell: breaks both short AND long term support
            else if close[i] < s_lower && close[i] < l_lower {
                signals[i] = -2.0;
            }
            // Moderate sell: breaks short-term support
            else if close[i] < s_lower {
                signals[i] = -1.0;
            }
        }

        signals
    }
}

impl Default for DualBreakout {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for DualBreakout {
    fn name(&self) -> &str {
        "5/20 Day Breakout"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.long_period {
            return Err(IndicatorError::InsufficientData {
                required: self.long_period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.long_period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for DualBreakout {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signals = self.calculate(&data.high, &data.low, &data.close);

        if signals.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let last = *signals.last().unwrap();
        if last >= 1.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last <= -1.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let signals = self.calculate(&data.high, &data.low, &data.close);

        Ok(signals.iter().map(|&s| {
            if s >= 1.0 {
                IndicatorSignal::Bullish
            } else if s <= -1.0 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

/// Donchian Middle Crossover - Signals based on price crossing the middle band.
#[derive(Debug, Clone)]
pub struct DonchianMiddleCross {
    period: usize,
}

impl DonchianMiddleCross {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate crossover signals.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let dc = DonchianChannels::new(self.period);
        let (_, middle, _) = dc.calculate(high, low);
        let n = close.len();

        let mut signals = vec![0.0; n];

        for i in 1..n {
            if middle[i].is_nan() || middle[i - 1].is_nan() {
                continue;
            }

            // Bullish cross: close crosses above middle
            if close[i - 1] <= middle[i - 1] && close[i] > middle[i] {
                signals[i] = 1.0;
            }
            // Bearish cross: close crosses below middle
            else if close[i - 1] >= middle[i - 1] && close[i] < middle[i] {
                signals[i] = -1.0;
            }
        }

        signals
    }
}

impl TechnicalIndicator for DonchianMiddleCross {
    fn name(&self) -> &str {
        "Donchian Middle Cross"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for DonchianMiddleCross {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signals = self.calculate(&data.high, &data.low, &data.close);

        if signals.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let last = *signals.last().unwrap();
        if last > 0.5 {
            Ok(IndicatorSignal::Bullish)
        } else if last < -0.5 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let signals = self.calculate(&data.high, &data.low, &data.close);

        Ok(signals.iter().map(|&s| {
            if s > 0.5 {
                IndicatorSignal::Bullish
            } else if s < -0.5 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..100).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0 + i as f64 * 0.05).collect();
        let low: Vec<f64> = (0..100).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0 + i as f64 * 0.05).collect();
        let close: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + i as f64 * 0.05).collect();
        (high, low, close)
    }

    #[test]
    fn test_donchian_width() {
        let (high, low, _) = make_test_data();
        let dw = DonchianWidth::new(20);
        let result = dw.calculate(&high, &low);

        assert_eq!(result.len(), 100);
        // Width should be positive
        for i in 19..100 {
            if !result[i].is_nan() {
                assert!(result[i] > 0.0);
            }
        }
    }

    #[test]
    fn test_four_week_rule() {
        let (high, low, close) = make_test_data();
        let fwr = FourWeekRule::new();
        let signals = fwr.calculate(&high, &low, &close);

        assert_eq!(signals.len(), 100);
    }

    #[test]
    fn test_turtle_entry() {
        let (high, low, close) = make_test_data();
        let turtle = TurtleEntry::system1();
        let output = turtle.calculate(&high, &low, &close);

        assert_eq!(output.entry_upper.len(), 100);
        assert_eq!(output.entry_signals.len(), 100);
    }

    #[test]
    fn test_turtle_exit() {
        let (high, low, close) = make_test_data();
        let turtle = TurtleExit::system1();
        let (upper, lower, signals) = turtle.calculate(&high, &low, &close);

        assert_eq!(upper.len(), 100);
        assert_eq!(lower.len(), 100);
        assert_eq!(signals.len(), 100);
    }

    #[test]
    fn test_dual_breakout() {
        let (high, low, close) = make_test_data();
        let db = DualBreakout::new();
        let signals = db.calculate(&high, &low, &close);

        assert_eq!(signals.len(), 100);
    }

    #[test]
    fn test_donchian_middle_cross() {
        let (high, low, close) = make_test_data();
        let dmc = DonchianMiddleCross::new(20);
        let signals = dmc.calculate(&high, &low, &close);

        assert_eq!(signals.len(), 100);
    }
}
