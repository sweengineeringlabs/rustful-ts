//! Lunar Cycle Indicator (IND-242)
//!
//! Analyzes moon phase correlation with market behavior.
//! While controversial, some traders believe lunar cycles affect sentiment.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Moon phase classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoonPhase {
    /// New Moon (day 0-3)
    NewMoon,
    /// Waxing Crescent (day 4-6)
    WaxingCrescent,
    /// First Quarter (day 7-10)
    FirstQuarter,
    /// Waxing Gibbous (day 11-13)
    WaxingGibbous,
    /// Full Moon (day 14-17)
    FullMoon,
    /// Waning Gibbous (day 18-20)
    WaningGibbous,
    /// Last Quarter (day 21-24)
    LastQuarter,
    /// Waning Crescent (day 25-28)
    WaningCrescent,
}

/// Configuration for the Lunar Cycle indicator.
#[derive(Debug, Clone)]
pub struct LunarCycleConfig {
    /// Lookback for calculating phase returns
    pub lookback_cycles: usize,
    /// Smoothing period for the indicator
    pub smoothing: usize,
    /// Whether to use trading days or calendar days
    pub use_trading_days: bool,
}

impl Default for LunarCycleConfig {
    fn default() -> Self {
        Self {
            lookback_cycles: 6,
            smoothing: 3,
            use_trading_days: true,
        }
    }
}

/// Lunar Cycle indicator for moon phase correlation analysis.
///
/// The lunar cycle is approximately 29.5 days. This indicator:
/// - Tracks current moon phase
/// - Calculates historical returns by phase
/// - Provides a lunar-based sentiment indicator
#[derive(Debug, Clone)]
pub struct LunarCycle {
    config: LunarCycleConfig,
}

impl LunarCycle {
    /// Synodic month length in days (average lunar cycle).
    const SYNODIC_MONTH: f64 = 29.53059;

    /// Create a new Lunar Cycle indicator.
    pub fn new() -> Self {
        Self {
            config: LunarCycleConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: LunarCycleConfig) -> Self {
        Self { config }
    }

    /// Calculate lunar day (0-29) based on a reference point.
    /// In practice, this would use actual astronomical data.
    fn calculate_lunar_day(&self, day_index: usize) -> f64 {
        // Using a fixed reference: Jan 1, 2000 was ~lunar day 24
        // This is a simplified model - real implementation would use ephemeris
        let reference_lunar_day = 24.0;

        // Adjust for trading days vs calendar days
        let adjustment = if self.config.use_trading_days {
            day_index as f64 * (365.0 / 252.0) // Convert trading to calendar days
        } else {
            day_index as f64
        };

        let lunar_day = (reference_lunar_day + adjustment) % Self::SYNODIC_MONTH;
        lunar_day
    }

    /// Get moon phase from lunar day.
    fn get_moon_phase(&self, lunar_day: f64) -> MoonPhase {
        let day = lunar_day.floor() as i32;
        match day {
            0..=3 => MoonPhase::NewMoon,
            4..=6 => MoonPhase::WaxingCrescent,
            7..=10 => MoonPhase::FirstQuarter,
            11..=13 => MoonPhase::WaxingGibbous,
            14..=17 => MoonPhase::FullMoon,
            18..=20 => MoonPhase::WaningGibbous,
            21..=24 => MoonPhase::LastQuarter,
            _ => MoonPhase::WaningCrescent,
        }
    }

    /// Calculate lunar phase position as 0 to 1 (full cycle).
    fn calculate_phase_position(&self, lunar_day: f64) -> f64 {
        lunar_day / Self::SYNODIC_MONTH
    }

    /// Calculate sine wave oscillator based on lunar cycle.
    /// Peaks at full moon, troughs at new moon.
    fn calculate_lunar_oscillator(&self, lunar_day: f64) -> f64 {
        // Use cosine so it's -1 at new moon, +1 at full moon
        let phase_angle = (lunar_day / Self::SYNODIC_MONTH) * 2.0 * std::f64::consts::PI;
        -phase_angle.cos() // Inverted so full moon is +1
    }

    /// Calculate historical returns by moon phase.
    fn calculate_phase_returns(&self, close: &[f64], idx: usize) -> [f64; 8] {
        let mut phase_returns = [0.0; 8];
        let mut phase_counts = [0usize; 8];

        // Look back through previous lunar cycles
        let lookback = (Self::SYNODIC_MONTH as usize) * self.config.lookback_cycles;

        for i in 1..lookback.min(idx) {
            let prev_idx = idx - i;
            if prev_idx == 0 || close[prev_idx - 1] <= 0.0 {
                continue;
            }

            let lunar_day = self.calculate_lunar_day(prev_idx);
            let phase = self.get_moon_phase(lunar_day);
            let phase_idx = phase as usize;

            let ret = (close[prev_idx] - close[prev_idx - 1]) / close[prev_idx - 1];
            phase_returns[phase_idx] += ret;
            phase_counts[phase_idx] += 1;
        }

        // Average returns per phase
        for i in 0..8 {
            if phase_counts[i] > 0 {
                phase_returns[i] /= phase_counts[i] as f64;
            }
        }

        phase_returns
    }

    /// Calculate the lunar cycle indicators.
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<MoonPhase>) {
        let n = close.len();
        let mut lunar_oscillator = vec![f64::NAN; n];
        let mut phase_bias = vec![f64::NAN; n];
        let mut moon_phase = vec![MoonPhase::NewMoon; n];

        for i in 0..n {
            let lunar_day = self.calculate_lunar_day(i);
            moon_phase[i] = self.get_moon_phase(lunar_day);
            lunar_oscillator[i] = self.calculate_lunar_oscillator(lunar_day);

            if i >= (Self::SYNODIC_MONTH as usize) * self.config.lookback_cycles {
                let phase_returns = self.calculate_phase_returns(close, i);
                let current_phase_idx = moon_phase[i] as usize;
                phase_bias[i] = phase_returns[current_phase_idx] * 100.0; // Convert to percentage
            }
        }

        // Apply smoothing to phase bias
        if self.config.smoothing > 1 {
            let mut smoothed = vec![f64::NAN; n];
            for i in self.config.smoothing..n {
                let sum: f64 = (0..self.config.smoothing)
                    .map(|j| phase_bias[i - j])
                    .filter(|v| !v.is_nan())
                    .sum();
                let count = (0..self.config.smoothing)
                    .filter(|&j| !phase_bias[i - j].is_nan())
                    .count();
                if count > 0 {
                    smoothed[i] = sum / count as f64;
                }
            }
            phase_bias = smoothed;
        }

        (lunar_oscillator, phase_bias, moon_phase)
    }

    /// Get the primary lunar signal (oscillator).
    pub fn calculate_signal(&self, close: &[f64]) -> Vec<f64> {
        self.calculate(close).0
    }
}

impl Default for LunarCycle {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for LunarCycle {
    fn name(&self) -> &str {
        "LunarCycle"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_periods = (Self::SYNODIC_MONTH as usize) * self.config.lookback_cycles;
        if data.close.len() < min_periods {
            return Err(IndicatorError::InsufficientData {
                required: min_periods,
                got: data.close.len(),
            });
        }

        let (lunar_oscillator, phase_bias, _) = self.calculate(&data.close);

        Ok(IndicatorOutput::dual(
            lunar_oscillator,
            phase_bias,
        ))
    }

    fn min_periods(&self) -> usize {
        (Self::SYNODIC_MONTH as usize) * self.config.lookback_cycles
    }
}

impl SignalIndicator for LunarCycle {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let min_periods = (Self::SYNODIC_MONTH as usize) * self.config.lookback_cycles;
        if data.close.len() < min_periods {
            return Ok(IndicatorSignal::Neutral);
        }

        let (_, phase_bias, moon_phase) = self.calculate(&data.close);
        let n = phase_bias.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let bias = phase_bias[n - 1];
        let phase = moon_phase[n - 1];

        // Signal based on historical phase performance
        if !bias.is_nan() {
            // Strong historical performance in current phase
            if bias > 0.1 {
                return Ok(IndicatorSignal::Bullish);
            } else if bias < -0.1 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        // Some traders prefer full moon periods
        if matches!(phase, MoonPhase::FullMoon) {
            return Ok(IndicatorSignal::Bullish);
        } else if matches!(phase, MoonPhase::NewMoon) {
            return Ok(IndicatorSignal::Bearish);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (_, phase_bias, moon_phase) = self.calculate(&data.close);

        let signals = phase_bias
            .iter()
            .enumerate()
            .map(|(i, &bias)| {
                if !bias.is_nan() {
                    if bias > 0.1 {
                        IndicatorSignal::Bullish
                    } else if bias < -0.1 {
                        IndicatorSignal::Bearish
                    } else {
                        IndicatorSignal::Neutral
                    }
                } else {
                    match moon_phase[i] {
                        MoonPhase::FullMoon => IndicatorSignal::Bullish,
                        MoonPhase::NewMoon => IndicatorSignal::Bearish,
                        _ => IndicatorSignal::Neutral,
                    }
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + (i as f64 * 0.1) + (i as f64 * 0.05).sin()).collect()
    }

    #[test]
    fn test_lunar_cycle_basic() {
        let indicator = LunarCycle::new();
        let close = create_test_data(200);

        let (oscillator, phase_bias, moon_phase) = indicator.calculate(&close);

        assert_eq!(oscillator.len(), 200);
        assert_eq!(phase_bias.len(), 200);
        assert_eq!(moon_phase.len(), 200);
    }

    #[test]
    fn test_lunar_oscillator_range() {
        let indicator = LunarCycle::new();
        let close = create_test_data(200);

        let (oscillator, _, _) = indicator.calculate(&close);

        // Oscillator should be in -1 to 1 range (cosine)
        for &osc in oscillator.iter().filter(|v| !v.is_nan()) {
            assert!(osc >= -1.0 && osc <= 1.0);
        }
    }

    #[test]
    fn test_moon_phases() {
        let indicator = LunarCycle::new();
        let close = create_test_data(200);

        let (_, _, moon_phase) = indicator.calculate(&close);

        // Should have multiple different phases over 200 days (~6-7 cycles)
        let unique_phases: std::collections::HashSet<_> = moon_phase.iter().collect();
        assert!(unique_phases.len() > 4);
    }

    #[test]
    fn test_lunar_day_calculation() {
        let indicator = LunarCycle::new();

        // Check that lunar day wraps around synodic month
        let day1 = indicator.calculate_lunar_day(0);
        let day2 = indicator.calculate_lunar_day(30);

        // Both should be in valid range
        assert!(day1 >= 0.0 && day1 < LunarCycle::SYNODIC_MONTH);
        assert!(day2 >= 0.0 && day2 < LunarCycle::SYNODIC_MONTH);
    }

    #[test]
    fn test_phase_from_lunar_day() {
        let indicator = LunarCycle::new();

        // Test specific phases
        assert_eq!(indicator.get_moon_phase(0.0), MoonPhase::NewMoon);
        assert_eq!(indicator.get_moon_phase(15.0), MoonPhase::FullMoon);
        assert_eq!(indicator.get_moon_phase(7.0), MoonPhase::FirstQuarter);
        assert_eq!(indicator.get_moon_phase(22.0), MoonPhase::LastQuarter);
    }

    #[test]
    fn test_lunar_insufficient_data() {
        let indicator = LunarCycle::new();
        let data = OHLCVSeries {
            open: vec![100.0; 50],
            high: vec![101.0; 50],
            low: vec![99.0; 50],
            close: vec![100.0; 50],
            volume: vec![1000.0; 50],
        };

        let result = indicator.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_lunar_technical_indicator() {
        let indicator = LunarCycle::new();
        let close: Vec<f64> = (0..250).map(|i| 100.0 + i as f64 * 0.1).collect();
        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|c| c + 1.0).collect(),
            low: close.iter().map(|c| c - 1.0).collect(),
            close,
            volume: vec![1000000.0; 250],
        };

        let result = indicator.compute(&data);
        assert!(result.is_ok());

        assert_eq!(indicator.name(), "LunarCycle");
    }
}
