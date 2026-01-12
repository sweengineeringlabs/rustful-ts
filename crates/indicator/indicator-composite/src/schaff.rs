//! Schaff Trend Cycle implementation.
//!
//! Combines MACD with Stochastic smoothing for faster trend detection.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_core::EMA;

/// Schaff Trend Cycle output.
#[derive(Debug, Clone)]
pub struct SchaffOutput {
    /// STC values (0-100 scale).
    pub stc: Vec<f64>,
    /// MACD values used in calculation.
    pub macd: Vec<f64>,
}

/// Schaff Trend Cycle configuration.
#[derive(Debug, Clone)]
pub struct SchaffConfig {
    /// MACD fast period (default: 23).
    pub macd_fast: usize,
    /// MACD slow period (default: 50).
    pub macd_slow: usize,
    /// Cycle period for stochastic (default: 10).
    pub cycle_period: usize,
    /// Smoothing factor (default: 0.5).
    pub factor: f64,
}

impl Default for SchaffConfig {
    fn default() -> Self {
        Self {
            macd_fast: 23,
            macd_slow: 50,
            cycle_period: 10,
            factor: 0.5,
        }
    }
}

/// Schaff Trend Cycle.
///
/// A faster cycle indicator that combines:
/// - MACD for trend direction
/// - Double stochastic smoothing for faster signals
///
/// Interpretation:
/// - STC > 75: Overbought, potential bearish reversal
/// - STC < 25: Oversold, potential bullish reversal
/// - Crossing 25 from below: Bullish signal
/// - Crossing 75 from above: Bearish signal
#[derive(Debug, Clone)]
pub struct SchaffTrendCycle {
    macd_fast: usize,
    macd_slow: usize,
    cycle_period: usize,
    factor: f64,
}

impl SchaffTrendCycle {
    pub fn new(config: SchaffConfig) -> Self {
        Self {
            macd_fast: config.macd_fast,
            macd_slow: config.macd_slow,
            cycle_period: config.cycle_period,
            factor: config.factor,
        }
    }

    /// Calculate Schaff Trend Cycle values.
    pub fn calculate(&self, close: &[f64]) -> SchaffOutput {
        // Calculate MACD line (fast EMA - slow EMA)
        let fast_ema = EMA::new(self.macd_fast).calculate(close);
        let slow_ema = EMA::new(self.macd_slow).calculate(close);

        let macd: Vec<f64> = fast_ema.iter()
            .zip(slow_ema.iter())
            .map(|(f, s)| {
                if f.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    f - s
                }
            })
            .collect();

        // First stochastic of MACD
        let stoch1 = self.stochastic_smooth(&macd);

        // Second stochastic of first stochastic (double smoothing)
        let stc = self.stochastic_smooth(&stoch1);

        SchaffOutput { stc, macd }
    }

    /// Apply stochastic formula with exponential smoothing.
    fn stochastic_smooth(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < self.cycle_period {
            return result;
        }

        let mut prev_val = 50.0; // Start at midpoint

        for i in (self.cycle_period - 1)..n {
            let start = i + 1 - self.cycle_period;
            let window = &data[start..=i];

            // Skip if any NaN in window
            if window.iter().any(|x| x.is_nan()) {
                result[i] = prev_val;
                continue;
            }

            let highest = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = window.iter().cloned().fold(f64::INFINITY, f64::min);
            let range = highest - lowest;

            let stoch = if range.abs() < 1e-10 {
                prev_val // No change if no range
            } else {
                ((data[i] - lowest) / range) * 100.0
            };

            // Exponential smoothing
            let smoothed = prev_val + self.factor * (stoch - prev_val);
            result[i] = smoothed.clamp(0.0, 100.0);
            prev_val = result[i];
        }

        result
    }
}

impl Default for SchaffTrendCycle {
    fn default() -> Self {
        Self::new(SchaffConfig::default())
    }
}

impl TechnicalIndicator for SchaffTrendCycle {
    fn name(&self) -> &str {
        "SchaffTrendCycle"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.macd_slow + self.cycle_period;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(result.stc, result.macd))
    }

    fn min_periods(&self) -> usize {
        self.macd_slow + self.cycle_period
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for SchaffTrendCycle {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.close);
        let n = result.stc.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let curr = result.stc[n - 1];
        let prev = result.stc[n - 2];

        if curr.is_nan() || prev.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: crossing above 25 from below
        if prev <= 25.0 && curr > 25.0 {
            return Ok(IndicatorSignal::Bullish);
        }

        // Bearish: crossing below 75 from above
        if prev >= 75.0 && curr < 75.0 {
            return Ok(IndicatorSignal::Bearish);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.close);
        let n = result.stc.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let curr = result.stc[i];
            let prev = result.stc[i - 1];

            if curr.is_nan() || prev.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if prev <= 25.0 && curr > 25.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if prev >= 75.0 && curr < 75.0 {
                signals.push(IndicatorSignal::Bearish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect()
    }

    #[test]
    fn test_schaff_basic() {
        let stc = SchaffTrendCycle::default();
        let close = generate_test_data(80);

        let result = stc.calculate(&close);

        assert_eq!(result.stc.len(), 80);
        assert_eq!(result.macd.len(), 80);
    }

    #[test]
    fn test_schaff_range() {
        let stc = SchaffTrendCycle::default();
        let close = generate_test_data(80);

        let result = stc.calculate(&close);

        // STC values should be between 0 and 100
        for i in 60..80 {
            if !result.stc[i].is_nan() {
                assert!(result.stc[i] >= 0.0 && result.stc[i] <= 100.0,
                    "STC value {} out of range at index {}", result.stc[i], i);
            }
        }
    }

    #[test]
    fn test_schaff_compute() {
        let stc = SchaffTrendCycle::default();
        let close = generate_test_data(80);
        let series = OHLCVSeries::from_close(close);

        let output = stc.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 80);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_schaff_config() {
        let config = SchaffConfig {
            macd_fast: 12,
            macd_slow: 26,
            cycle_period: 8,
            factor: 0.6,
        };

        let stc = SchaffTrendCycle::new(config);
        assert_eq!(stc.min_periods(), 26 + 8);
    }
}
