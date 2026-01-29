//! Planetary Cycles Indicator (IND-326)
//!
//! This indicator creates a proxy for astronomical/planetary cycle correlations
//! in market data. It uses mathematical wave functions to simulate cyclical
//! patterns similar to those proposed in financial astrology theories.
//!
//! Note: This is a mathematical proxy and does not use actual astronomical data.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Cycle phase type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CyclePhaseType {
    /// New cycle beginning (0-45 degrees)
    NewPhase,
    /// First quarter (45-90 degrees)
    FirstQuarter,
    /// Waxing phase (90-135 degrees)
    Waxing,
    /// Second quarter (135-180 degrees)
    SecondQuarter,
    /// Full phase (180-225 degrees)
    FullPhase,
    /// Third quarter (225-270 degrees)
    ThirdQuarter,
    /// Waning phase (270-315 degrees)
    Waning,
    /// Fourth quarter (315-360 degrees)
    FourthQuarter,
}

impl CyclePhaseType {
    /// Get phase from angle in degrees
    pub fn from_angle(angle: f64) -> Self {
        let normalized = angle % 360.0;
        match normalized {
            a if a < 45.0 => CyclePhaseType::NewPhase,
            a if a < 90.0 => CyclePhaseType::FirstQuarter,
            a if a < 135.0 => CyclePhaseType::Waxing,
            a if a < 180.0 => CyclePhaseType::SecondQuarter,
            a if a < 225.0 => CyclePhaseType::FullPhase,
            a if a < 270.0 => CyclePhaseType::ThirdQuarter,
            a if a < 315.0 => CyclePhaseType::Waning,
            _ => CyclePhaseType::FourthQuarter,
        }
    }
}

/// Planetary Cycles output structure
#[derive(Debug, Clone)]
pub struct PlanetaryCyclesOutput {
    /// Short cycle value (approx. 28-day lunar proxy)
    pub lunar_cycle: Vec<f64>,
    /// Medium cycle value (approx. 365-day solar proxy)
    pub solar_cycle: Vec<f64>,
    /// Long cycle value (approx. 12-year Jupiter proxy)
    pub jupiter_cycle: Vec<f64>,
    /// Very long cycle (approx. 29-year Saturn proxy)
    pub saturn_cycle: Vec<f64>,
    /// Composite cycle value (weighted combination)
    pub composite: Vec<f64>,
    /// Current lunar phase
    pub lunar_phase: Vec<CyclePhaseType>,
    /// Cycle confluence (when multiple cycles align)
    pub confluence: Vec<f64>,
    /// Turning point probability
    pub turning_point: Vec<f64>,
}

/// Planetary Cycles configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanetaryCyclesConfig {
    /// Lunar cycle period (default: 28 bars)
    pub lunar_period: usize,
    /// Solar cycle period (default: 365 bars)
    pub solar_period: usize,
    /// Jupiter cycle period (default: 4380 bars ~12 years daily)
    pub jupiter_period: usize,
    /// Saturn cycle period (default: 10585 bars ~29 years daily)
    pub saturn_period: usize,
    /// Weight for lunar cycle
    pub lunar_weight: f64,
    /// Weight for solar cycle
    pub solar_weight: f64,
    /// Weight for Jupiter cycle
    pub jupiter_weight: f64,
    /// Weight for Saturn cycle
    pub saturn_weight: f64,
    /// Confluence threshold (0-1)
    pub confluence_threshold: f64,
}

impl Default for PlanetaryCyclesConfig {
    fn default() -> Self {
        Self {
            lunar_period: 28,
            solar_period: 365,
            jupiter_period: 4380,
            saturn_period: 10585,
            lunar_weight: 0.4,
            solar_weight: 0.3,
            jupiter_weight: 0.2,
            saturn_weight: 0.1,
            confluence_threshold: 0.8,
        }
    }
}

/// Planetary Cycles Indicator
///
/// Mathematical proxy for planetary/astronomical cycle correlations.
/// Uses sine waves with different periods to simulate cyclical patterns.
///
/// # Cycle Periods (Default)
/// - Lunar: 28 bars (monthly cycle)
/// - Solar: 365 bars (yearly cycle)
/// - Jupiter: 4380 bars (~12 years)
/// - Saturn: 10585 bars (~29 years)
///
/// # Trading Rules
/// - High confluence suggests potential turning points
/// - Lunar phase can indicate short-term reversals
/// - Composite cycle indicates overall cyclical position
/// - Use in conjunction with other technical indicators
#[derive(Debug, Clone)]
pub struct PlanetaryCycles {
    config: PlanetaryCyclesConfig,
}

impl PlanetaryCycles {
    pub fn new() -> Self {
        Self {
            config: PlanetaryCyclesConfig::default(),
        }
    }

    pub fn with_config(config: PlanetaryCyclesConfig) -> Self {
        Self { config }
    }

    pub fn with_lunar_period(mut self, period: usize) -> Self {
        self.config.lunar_period = period;
        self
    }

    pub fn with_solar_period(mut self, period: usize) -> Self {
        self.config.solar_period = period;
        self
    }

    /// Calculate cycle value at a given bar
    fn calc_cycle(&self, bar: usize, period: usize) -> f64 {
        let angle = 2.0 * PI * (bar as f64 / period as f64);
        angle.sin()
    }

    /// Calculate cycle angle in degrees
    fn calc_angle(&self, bar: usize, period: usize) -> f64 {
        ((bar as f64 / period as f64) * 360.0) % 360.0
    }

    /// Calculate Planetary Cycles from OHLCV data
    pub fn calculate(&self, data: &OHLCVSeries) -> PlanetaryCyclesOutput {
        let n = data.close.len();

        let mut lunar_cycle = vec![f64::NAN; n];
        let mut solar_cycle = vec![f64::NAN; n];
        let mut jupiter_cycle = vec![f64::NAN; n];
        let mut saturn_cycle = vec![f64::NAN; n];
        let mut composite = vec![f64::NAN; n];
        let mut lunar_phase = vec![CyclePhaseType::NewPhase; n];
        let mut confluence = vec![f64::NAN; n];
        let mut turning_point = vec![f64::NAN; n];

        let total_weight = self.config.lunar_weight
            + self.config.solar_weight
            + self.config.jupiter_weight
            + self.config.saturn_weight;

        for i in 0..n {
            // Calculate individual cycle values
            lunar_cycle[i] = self.calc_cycle(i, self.config.lunar_period);
            solar_cycle[i] = self.calc_cycle(i, self.config.solar_period);
            jupiter_cycle[i] = self.calc_cycle(i, self.config.jupiter_period);
            saturn_cycle[i] = self.calc_cycle(i, self.config.saturn_period);

            // Calculate composite (weighted average)
            composite[i] = (self.config.lunar_weight * lunar_cycle[i]
                + self.config.solar_weight * solar_cycle[i]
                + self.config.jupiter_weight * jupiter_cycle[i]
                + self.config.saturn_weight * saturn_cycle[i])
                / total_weight;

            // Determine lunar phase
            let lunar_angle = self.calc_angle(i, self.config.lunar_period);
            lunar_phase[i] = CyclePhaseType::from_angle(lunar_angle);

            // Calculate confluence (how aligned are the cycles)
            // High confluence when all cycles have similar sign and magnitude
            let signs = [
                lunar_cycle[i].signum(),
                solar_cycle[i].signum(),
                jupiter_cycle[i].signum(),
                saturn_cycle[i].signum(),
            ];
            let same_sign_count = signs.iter().filter(|&&s| s == signs[0]).count();
            let sign_confluence = same_sign_count as f64 / 4.0;

            // Magnitude confluence (are they all near extremes?)
            let magnitudes = [
                lunar_cycle[i].abs(),
                solar_cycle[i].abs(),
                jupiter_cycle[i].abs(),
                saturn_cycle[i].abs(),
            ];
            let avg_magnitude: f64 = magnitudes.iter().sum::<f64>() / 4.0;

            confluence[i] = sign_confluence * avg_magnitude;

            // Turning point probability (high when cycles are at extremes and aligned)
            let at_extreme = |v: f64| -> f64 {
                let abs_v = v.abs();
                if abs_v > 0.9 {
                    1.0
                } else if abs_v > 0.7 {
                    0.7
                } else if abs_v > 0.5 {
                    0.4
                } else {
                    0.1
                }
            };

            let extreme_lunar = at_extreme(lunar_cycle[i]);
            let extreme_solar = at_extreme(solar_cycle[i]);

            turning_point[i] = (extreme_lunar * self.config.lunar_weight
                + extreme_solar * self.config.solar_weight
                + confluence[i] * 0.3)
                / (self.config.lunar_weight + self.config.solar_weight + 0.3);
        }

        PlanetaryCyclesOutput {
            lunar_cycle,
            solar_cycle,
            jupiter_cycle,
            saturn_cycle,
            composite,
            lunar_phase,
            confluence,
            turning_point,
        }
    }
}

impl Default for PlanetaryCycles {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for PlanetaryCycles {
    fn name(&self) -> &str {
        "Planetary Cycles"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 2 {
            return Err(IndicatorError::InsufficientData {
                required: 2,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);

        // Primary: composite, Secondary: confluence, Tertiary: turning_point
        Ok(IndicatorOutput::triple(
            result.composite,
            result.confluence,
            result.turning_point,
        ))
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for PlanetaryCycles {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.composite.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let composite = result.composite[n - 1];
        let prev_composite = result.composite[n - 2];
        let turning = result.turning_point[n - 1];

        if composite.is_nan() || prev_composite.is_nan() || turning.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal on turning points with cycle direction change
        if turning > 0.7 {
            if composite > prev_composite && composite < 0.0 {
                // Rising from bottom
                Ok(IndicatorSignal::Bullish)
            } else if composite < prev_composite && composite > 0.0 {
                // Falling from top
                Ok(IndicatorSignal::Bearish)
            } else {
                Ok(IndicatorSignal::Neutral)
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);
        let n = result.composite.len();

        let mut signals = vec![IndicatorSignal::Neutral; n];

        for i in 1..n {
            let composite = result.composite[i];
            let prev_composite = result.composite[i - 1];
            let turning = result.turning_point[i];

            if composite.is_nan() || prev_composite.is_nan() || turning.is_nan() {
                continue;
            }

            if turning > 0.7 {
                if composite > prev_composite && composite < 0.0 {
                    signals[i] = IndicatorSignal::Bullish;
                } else if composite < prev_composite && composite > 0.0 {
                    signals[i] = IndicatorSignal::Bearish;
                }
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(bars: usize) -> OHLCVSeries {
        let mut closes = Vec::with_capacity(bars);
        for i in 0..bars {
            closes.push(100.0 + (i as f64 * 0.1).sin() * 10.0);
        }

        OHLCVSeries {
            open: closes.iter().map(|c| c - 0.5).collect(),
            high: closes.iter().map(|c| c + 2.0).collect(),
            low: closes.iter().map(|c| c - 2.0).collect(),
            close: closes,
            volume: vec![1000.0; bars],
        }
    }

    #[test]
    fn test_planetary_cycles_initialization() {
        let pc = PlanetaryCycles::new();
        assert_eq!(pc.name(), "Planetary Cycles");
        assert_eq!(pc.min_periods(), 2);
        assert_eq!(pc.output_features(), 3);
    }

    #[test]
    fn test_planetary_cycles_calculation() {
        let data = create_test_data(100);
        let pc = PlanetaryCycles::new();
        let result = pc.calculate(&data);

        assert_eq!(result.lunar_cycle.len(), 100);
        assert_eq!(result.composite.len(), 100);
        assert_eq!(result.confluence.len(), 100);

        // Verify cycle values are bounded
        for i in 0..100 {
            assert!(result.lunar_cycle[i] >= -1.0 && result.lunar_cycle[i] <= 1.0);
            assert!(result.solar_cycle[i] >= -1.0 && result.solar_cycle[i] <= 1.0);
        }
    }

    #[test]
    fn test_lunar_phase() {
        let phase = CyclePhaseType::from_angle(0.0);
        assert_eq!(phase, CyclePhaseType::NewPhase);

        let phase = CyclePhaseType::from_angle(90.0);
        assert_eq!(phase, CyclePhaseType::Waxing);

        let phase = CyclePhaseType::from_angle(180.0);
        assert_eq!(phase, CyclePhaseType::FullPhase);

        let phase = CyclePhaseType::from_angle(270.0);
        assert_eq!(phase, CyclePhaseType::Waning);
    }

    #[test]
    fn test_cycle_periodicity() {
        let pc = PlanetaryCycles::new().with_lunar_period(28);

        // Value at bar 0 and bar 28 should be similar (one full cycle)
        let v0 = pc.calc_cycle(0, 28);
        let v28 = pc.calc_cycle(28, 28);
        assert!((v0 - v28).abs() < 0.001);

        // Value at bar 14 (half cycle) should be opposite
        let v14 = pc.calc_cycle(14, 28);
        assert!((v0 + v14).abs() < 0.001);
    }

    #[test]
    fn test_planetary_cycles_compute() {
        let data = create_test_data(100);
        let pc = PlanetaryCycles::new();
        let output = pc.compute(&data).unwrap();

        assert_eq!(output.primary.len(), 100);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_planetary_cycles_signals() {
        let data = create_test_data(100);
        let pc = PlanetaryCycles::new();
        let signals = pc.signals(&data).unwrap();

        assert_eq!(signals.len(), 100);
    }

    #[test]
    fn test_insufficient_data() {
        let data = OHLCVSeries {
            open: vec![100.0],
            high: vec![102.0],
            low: vec![98.0],
            close: vec![100.0],
            volume: vec![1000.0],
        };

        let pc = PlanetaryCycles::new();
        let result = pc.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_confluence_calculation() {
        let data = create_test_data(100);
        let pc = PlanetaryCycles::new();
        let result = pc.calculate(&data);

        // Confluence should be between 0 and 1
        for i in 0..100 {
            if !result.confluence[i].is_nan() {
                assert!(result.confluence[i] >= 0.0 && result.confluence[i] <= 1.0);
            }
        }
    }
}
