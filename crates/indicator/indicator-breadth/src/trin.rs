//! TRIN (Arms Index) indicator.

use crate::{BreadthIndicator, BreadthSeries};
use indicator_spi::{IndicatorError, IndicatorOutput, Result};

/// TRIN (Trading Index / Arms Index)
///
/// Developed by Richard Arms, TRIN measures the relationship between
/// advancing/declining issues and advancing/declining volume. It identifies
/// whether volume is flowing into advancing or declining stocks.
///
/// # Formula
/// TRIN = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)
///
/// Or equivalently:
/// TRIN = (Advances / Declines) * (Decline Volume / Advance Volume)
///
/// # Interpretation
/// - TRIN = 1.0: Neutral (volume proportionally distributed)
/// - TRIN < 1.0: Bullish (more volume in advancing stocks)
/// - TRIN > 1.0: Bearish (more volume in declining stocks)
/// - TRIN < 0.5: Extremely overbought (potential reversal)
/// - TRIN > 2.0: Extremely oversold (potential reversal)
///
/// Note: TRIN is inverted from typical indicators (low = bullish, high = bearish)
#[derive(Debug, Clone)]
pub struct TRIN {
    /// Smoothing period (0 = no smoothing)
    smoothing_period: usize,
    /// Use EMA instead of SMA for smoothing
    use_ema: bool,
}

impl Default for TRIN {
    fn default() -> Self {
        Self::new()
    }
}

impl TRIN {
    pub fn new() -> Self {
        Self {
            smoothing_period: 0,
            use_ema: false,
        }
    }

    /// Create TRIN with 10-day moving average (common setting)
    pub fn smoothed_10() -> Self {
        Self {
            smoothing_period: 10,
            use_ema: false,
        }
    }

    pub fn with_smoothing(mut self, period: usize) -> Self {
        self.smoothing_period = period;
        self
    }

    pub fn with_ema(mut self) -> Self {
        self.use_ema = true;
        self
    }

    /// Calculate SMA
    fn calculate_sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().filter(|v| !v.is_nan()).sum();
        result.push(sum / period as f64);

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

    /// Calculate EMA
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let multiplier = 2.0 / (period as f64 + 1.0);

        // Initial SMA as first EMA
        let valid_count = data[..period].iter().filter(|v| !v.is_nan()).count();
        let sum: f64 = data[..period].iter().filter(|v| !v.is_nan()).sum();
        let mut ema = sum / valid_count as f64;
        result.push(ema);

        // Calculate EMA
        for i in period..data.len() {
            if !data[i].is_nan() {
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
                if *dec == 0.0 || *av == 0.0 {
                    return f64::NAN;
                }

                let ad_ratio = adv / dec;
                let volume_ratio = av / dv;

                if volume_ratio == 0.0 {
                    f64::NAN
                } else {
                    ad_ratio / volume_ratio
                }
            })
            .collect()
    }

    /// Calculate TRIN from BreadthSeries
    pub fn calculate(&self, data: &BreadthSeries) -> Vec<f64> {
        let raw = self.calculate_raw(data);

        if self.smoothing_period > 0 {
            if self.use_ema {
                self.calculate_ema(&raw, self.smoothing_period)
            } else {
                self.calculate_sma(&raw, self.smoothing_period)
            }
        } else {
            raw
        }
    }

    /// Interpret TRIN value
    pub fn interpret(&self, value: f64) -> TRINSignal {
        if value.is_nan() {
            TRINSignal::Unknown
        } else if value < 0.5 {
            TRINSignal::ExtremelyOverbought
        } else if value < 0.8 {
            TRINSignal::Overbought
        } else if value <= 1.2 {
            TRINSignal::Neutral
        } else if value <= 2.0 {
            TRINSignal::Oversold
        } else {
            TRINSignal::ExtremelyOversold
        }
    }

    /// Calculate Open TRIN (cumulative intraday TRIN)
    /// This uses cumulative advances/declines/volumes from market open
    pub fn calculate_open_trin(
        &self,
        cum_advances: f64,
        cum_declines: f64,
        cum_advance_volume: f64,
        cum_decline_volume: f64,
    ) -> f64 {
        if cum_declines == 0.0 || cum_advance_volume == 0.0 {
            return f64::NAN;
        }

        let ad_ratio = cum_advances / cum_declines;
        let volume_ratio = cum_advance_volume / cum_decline_volume;

        if volume_ratio == 0.0 {
            f64::NAN
        } else {
            ad_ratio / volume_ratio
        }
    }
}

/// TRIN signal interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TRINSignal {
    /// TRIN < 0.5: Very bullish but potential reversal
    ExtremelyOverbought,
    /// TRIN 0.5-0.8: Bullish, volume favoring advances
    Overbought,
    /// TRIN 0.8-1.2: Neutral, balanced market
    Neutral,
    /// TRIN 1.2-2.0: Bearish, volume favoring declines
    Oversold,
    /// TRIN > 2.0: Very bearish but potential reversal
    ExtremelyOversold,
    /// Invalid or insufficient data
    Unknown,
}

impl BreadthIndicator for TRIN {
    fn name(&self) -> &str {
        "TRIN"
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

        let values = self.calculate(data);
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
    use crate::BreadthData;

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
        // Bearish: More declines, but even more decline volume -> TRIN > 1
        series.push(BreadthData::from_ad_volume(
            1000.0, 2000.0, 500_000.0, 2_000_000.0,
        ));
        // Very bullish: Volume heavily in advances
        series.push(BreadthData::from_ad_volume(
            1800.0, 1200.0, 3_000_000.0, 500_000.0,
        ));
        // Very bearish: Volume heavily in declines
        series.push(BreadthData::from_ad_volume(
            1200.0, 1800.0, 400_000.0, 2_400_000.0,
        ));
        series
    }

    #[test]
    fn test_trin_basic() {
        let trin = TRIN::new();
        let series = create_test_series();
        let result = trin.calculate(&series);

        assert_eq!(result.len(), 5);

        // Day 1: (2000/1000) / (2000000/1000000) = 2/2 = 1.0
        assert!((result[0] - 1.0).abs() < 1e-10);

        // Day 2: (1500/1500) / (1000000/1000000) = 1/1 = 1.0
        assert!((result[1] - 1.0).abs() < 1e-10);

        // Day 3: (1000/2000) / (500000/2000000) = 0.5/0.25 = 2.0
        assert!((result[2] - 2.0).abs() < 1e-10);

        // Day 4: (1800/1200) / (3000000/500000) = 1.5/6 = 0.25
        assert!((result[3] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_trin_interpretation() {
        let trin = TRIN::new();

        assert_eq!(trin.interpret(0.3), TRINSignal::ExtremelyOverbought);
        assert_eq!(trin.interpret(0.6), TRINSignal::Overbought);
        assert_eq!(trin.interpret(1.0), TRINSignal::Neutral);
        assert_eq!(trin.interpret(1.5), TRINSignal::Oversold);
        assert_eq!(trin.interpret(2.5), TRINSignal::ExtremelyOversold);
        assert_eq!(trin.interpret(f64::NAN), TRINSignal::Unknown);
    }

    #[test]
    fn test_trin_with_smoothing() {
        let trin = TRIN::new().with_smoothing(3);
        let series = create_test_series();
        let result = trin.calculate(&series);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // SMA of first 3 values: (1.0 + 1.0 + 2.0) / 3 = 1.333...
        assert!((result[2] - 1.3333333333333333).abs() < 1e-10);
    }

    #[test]
    fn test_trin_with_ema() {
        let trin = TRIN::new().with_smoothing(3).with_ema();
        let series = create_test_series();
        let result = trin.calculate(&series);

        assert_eq!(result.len(), 5);
        // EMA calculation should differ from SMA
        assert!(!result[2].is_nan());
    }

    #[test]
    fn test_trin_edge_cases() {
        let trin = TRIN::new();
        let mut series = BreadthSeries::new();

        // Zero declines
        series.push(BreadthData::from_ad_volume(
            1000.0, 0.0, 1_000_000.0, 500_000.0,
        ));

        let result = trin.calculate(&series);
        assert!(result[0].is_nan());
    }

    #[test]
    fn test_open_trin() {
        let trin = TRIN::new();

        // Bullish cumulative reading
        let result = trin.calculate_open_trin(2000.0, 1000.0, 2_000_000.0, 1_000_000.0);
        assert!((result - 1.0).abs() < 1e-10);

        // Bearish cumulative reading
        let result = trin.calculate_open_trin(1000.0, 2000.0, 500_000.0, 2_000_000.0);
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothed_10() {
        let trin = TRIN::smoothed_10();
        assert_eq!(trin.smoothing_period, 10);
    }
}
