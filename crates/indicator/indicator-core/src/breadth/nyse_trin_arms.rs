//! NYSE TRIN (Arms Index) - Market Internals Indicator (IND-395)

use super::{BreadthIndicator, BreadthSeries};
use crate::{IndicatorError, IndicatorOutput, Result};

/// NYSE TRIN Arms Index Configuration
#[derive(Debug, Clone)]
pub struct NYSETRINArmsConfig {
    /// Smoothing period for the TRIN (0 = no smoothing)
    pub smoothing_period: usize,
    /// Use EMA instead of SMA for smoothing
    pub use_ema: bool,
    /// Overbought threshold (TRIN < this is overbought)
    pub overbought_threshold: f64,
    /// Oversold threshold (TRIN > this is oversold)
    pub oversold_threshold: f64,
}

impl Default for NYSETRINArmsConfig {
    fn default() -> Self {
        Self {
            smoothing_period: 0,
            use_ema: false,
            overbought_threshold: 0.75,
            oversold_threshold: 1.25,
        }
    }
}

/// NYSE TRIN (Arms Index)
///
/// The TRIN (Trading Index), also known as the Arms Index, measures the
/// relationship between advance/decline ratio and advance/decline volume ratio.
/// It was developed by Richard Arms to identify market buying or selling pressure.
///
/// # Formula
/// TRIN = (A/D Ratio) / (Volume Ratio)
/// Where:
/// - A/D Ratio = Advancing Issues / Declining Issues
/// - Volume Ratio = Advancing Volume / Declining Volume
///
/// # Interpretation
/// - TRIN < 1.0: Bullish (volume flowing into advancing stocks)
/// - TRIN > 1.0: Bearish (volume flowing into declining stocks)
/// - TRIN = 1.0: Neutral (balanced)
/// - TRIN < 0.75: Extremely overbought (potential reversal down)
/// - TRIN > 1.25: Extremely oversold (potential reversal up)
///
/// Note: TRIN is "inverted" - low values are bullish, high values are bearish.
#[derive(Debug, Clone)]
pub struct NYSETRINArms {
    config: NYSETRINArmsConfig,
}

impl Default for NYSETRINArms {
    fn default() -> Self {
        Self::new()
    }
}

impl NYSETRINArms {
    pub fn new() -> Self {
        Self {
            config: NYSETRINArmsConfig::default(),
        }
    }

    pub fn with_config(config: NYSETRINArmsConfig) -> Self {
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

    /// Calculate SMA for smoothing
    fn calculate_sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().filter(|v| !v.is_nan()).sum();
        let count = data[..period].iter().filter(|v| !v.is_nan()).count();
        if count > 0 {
            result.push(sum / count as f64);
        } else {
            result.push(f64::NAN);
        }

        for i in period..data.len() {
            if !data[i - period].is_nan() {
                sum -= data[i - period];
            }
            if !data[i].is_nan() {
                sum += data[i];
            }
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate EMA for smoothing
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let multiplier = 2.0 / (period as f64 + 1.0);

        let valid_count = data[..period].iter().filter(|v| !v.is_nan()).count();
        let sum: f64 = data[..period].iter().filter(|v| !v.is_nan()).sum();
        let mut ema = if valid_count > 0 {
            sum / valid_count as f64
        } else {
            f64::NAN
        };
        result.push(ema);

        for i in period..data.len() {
            if !data[i].is_nan() && !ema.is_nan() {
                ema = (data[i] - ema) * multiplier + ema;
            }
            result.push(ema);
        }

        result
    }

    /// Calculate raw TRIN values
    fn calculate_raw(&self, data: &BreadthSeries) -> Vec<f64> {
        data.advances
            .iter()
            .zip(data.declines.iter())
            .zip(data.advance_volume.iter())
            .zip(data.decline_volume.iter())
            .map(|(((adv, dec), av), dv)| {
                // Handle edge cases
                if *dec == 0.0 || *av == 0.0 || *dv == 0.0 {
                    return f64::NAN;
                }

                let ad_ratio = adv / dec;
                let volume_ratio = av / dv;

                ad_ratio / volume_ratio
            })
            .collect()
    }

    /// Calculate TRIN from BreadthSeries
    pub fn calculate(&self, data: &BreadthSeries) -> Vec<f64> {
        let raw = self.calculate_raw(data);

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

    /// Interpret TRIN value
    pub fn interpret(&self, value: f64) -> NYSETRINSignal {
        if value.is_nan() {
            NYSETRINSignal::Unknown
        } else if value < self.config.overbought_threshold {
            NYSETRINSignal::ExtremelyOverbought
        } else if value < 1.0 {
            NYSETRINSignal::Bullish
        } else if value <= self.config.oversold_threshold {
            NYSETRINSignal::Bearish
        } else {
            NYSETRINSignal::ExtremelyOversold
        }
    }

    /// Calculate Open TRIN (cumulative intraday)
    pub fn calculate_open_trin(
        &self,
        cum_advances: f64,
        cum_declines: f64,
        cum_advance_volume: f64,
        cum_decline_volume: f64,
    ) -> f64 {
        if cum_declines == 0.0 || cum_advance_volume == 0.0 || cum_decline_volume == 0.0 {
            return f64::NAN;
        }

        let ad_ratio = cum_advances / cum_declines;
        let volume_ratio = cum_advance_volume / cum_decline_volume;

        ad_ratio / volume_ratio
    }
}

/// NYSE TRIN signal interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NYSETRINSignal {
    /// TRIN very low: Market may be overbought
    ExtremelyOverbought,
    /// TRIN < 1.0: Volume favoring advances
    Bullish,
    /// TRIN > 1.0: Volume favoring declines
    Bearish,
    /// TRIN very high: Market may be oversold
    ExtremelyOversold,
    /// Invalid data
    Unknown,
}

impl BreadthIndicator for NYSETRINArms {
    fn name(&self) -> &str {
        "NYSE TRIN Arms"
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

        let values = self.calculate(data);
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
        // Bullish: More advances, more advance volume -> TRIN < 1
        series.push(BreadthData::from_ad_volume(
            2000.0, 1000.0, 2_000_000.0, 1_000_000.0,
        ));
        // Neutral: Proportional
        series.push(BreadthData::from_ad_volume(
            1500.0, 1500.0, 1_000_000.0, 1_000_000.0,
        ));
        // Bearish: TRIN > 1
        series.push(BreadthData::from_ad_volume(
            1000.0, 2000.0, 500_000.0, 2_000_000.0,
        ));
        // Very bullish
        series.push(BreadthData::from_ad_volume(
            1800.0, 1200.0, 3_000_000.0, 500_000.0,
        ));
        // Very bearish
        series.push(BreadthData::from_ad_volume(
            1200.0, 1800.0, 400_000.0, 2_400_000.0,
        ));
        series
    }

    #[test]
    fn test_nyse_trin_basic() {
        let trin = NYSETRINArms::new();
        let series = create_test_series();
        let result = trin.calculate(&series);

        assert_eq!(result.len(), 5);
        // Day 1: (2000/1000) / (2000000/1000000) = 2/2 = 1.0
        assert!((result[0] - 1.0).abs() < 1e-10);
        // Day 2: (1500/1500) / (1000000/1000000) = 1/1 = 1.0
        assert!((result[1] - 1.0).abs() < 1e-10);
        // Day 3: (1000/2000) / (500000/2000000) = 0.5/0.25 = 2.0
        assert!((result[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_nyse_trin_interpretation() {
        let trin = NYSETRINArms::new();

        assert_eq!(trin.interpret(0.5), NYSETRINSignal::ExtremelyOverbought);
        assert_eq!(trin.interpret(0.9), NYSETRINSignal::Bullish);
        assert_eq!(trin.interpret(1.1), NYSETRINSignal::Bearish);
        assert_eq!(trin.interpret(1.5), NYSETRINSignal::ExtremelyOversold);
        assert_eq!(trin.interpret(f64::NAN), NYSETRINSignal::Unknown);
    }

    #[test]
    fn test_nyse_trin_smoothed() {
        let trin = NYSETRINArms::new().with_smoothing(3);
        let series = create_test_series();
        let result = trin.calculate(&series);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // SMA of first 3 values: (1.0 + 1.0 + 2.0) / 3 = 1.333...
        assert!((result[2] - 1.3333333333333333).abs() < 1e-10);
    }

    #[test]
    fn test_nyse_trin_ema() {
        let trin = NYSETRINArms::new().with_smoothing(3).with_ema();
        let series = create_test_series();
        let result = trin.calculate(&series);

        assert_eq!(result.len(), 5);
        assert!(!result[2].is_nan());
    }

    #[test]
    fn test_open_trin() {
        let trin = NYSETRINArms::new();

        let result = trin.calculate_open_trin(2000.0, 1000.0, 2_000_000.0, 1_000_000.0);
        assert!((result - 1.0).abs() < 1e-10);

        let result = trin.calculate_open_trin(1000.0, 2000.0, 500_000.0, 2_000_000.0);
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_breadth_indicator_trait() {
        let trin = NYSETRINArms::new();
        let series = create_test_series();
        let result = trin.compute_breadth(&series);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.values.len(), 5);
    }
}
