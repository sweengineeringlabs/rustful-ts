//! Elder's Bull/Bear Power (Enhanced) implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_api::ElderPowerConfig;
use crate::EMA;

/// Elder Power output containing all computed values.
#[derive(Debug, Clone)]
pub struct ElderPowerOutput {
    /// Bull Power = High - EMA (buyers pushing above average)
    pub bull_power: Vec<f64>,
    /// Bear Power = Low - EMA (sellers pushing below average)
    pub bear_power: Vec<f64>,
    /// Combined Power = Bull Power + Bear Power
    pub combined_power: Vec<f64>,
}

/// Elder's Bull/Bear Power (Enhanced).
///
/// Enhanced version of Elder Ray that combines bull and bear power
/// into a single signal and generates signals based on divergences
/// and zero crossings.
///
/// # Algorithm
/// 1. Calculate EMA of close prices
/// 2. Bull Power = High - EMA (buyers pushing above average)
/// 3. Bear Power = Low - EMA (sellers pushing below average)
/// 4. Combined Power = Bull Power + Bear Power
/// 5. Generate signals based on divergences and zero crossings
///
/// # Interpretation
/// - Bull Power > 0: Buyers pushing prices above average
/// - Bear Power < 0: Sellers pushing prices below average
/// - Combined Power crossing zero indicates trend changes
/// - Divergences between price and power can signal reversals
#[derive(Debug, Clone)]
pub struct ElderPower {
    period: usize,
}

impl ElderPower {
    /// Create a new Elder Power indicator with specified EMA period.
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Create from configuration.
    pub fn from_config(config: ElderPowerConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate Elder Power values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> ElderPowerOutput {
        let n = close.len();

        // Calculate EMA of close
        let ema = EMA::new(self.period);
        let ema_values = ema.calculate(close);

        let mut bull_power = Vec::with_capacity(n);
        let mut bear_power = Vec::with_capacity(n);
        let mut combined_power = Vec::with_capacity(n);

        for i in 0..n {
            if ema_values[i].is_nan() {
                bull_power.push(f64::NAN);
                bear_power.push(f64::NAN);
                combined_power.push(f64::NAN);
            } else {
                let bp = high[i] - ema_values[i];
                let brp = low[i] - ema_values[i];
                bull_power.push(bp);
                bear_power.push(brp);
                combined_power.push(bp + brp);
            }
        }

        ElderPowerOutput {
            bull_power,
            bear_power,
            combined_power,
        }
    }

    /// Generate signal based on combined power zero crossing and divergence.
    fn generate_signal(&self, output: &ElderPowerOutput, close: &[f64]) -> IndicatorSignal {
        let n = output.combined_power.len();
        if n < 2 {
            return IndicatorSignal::Neutral;
        }

        let curr_power = output.combined_power[n - 1];
        let prev_power = output.combined_power[n - 2];

        if curr_power.is_nan() || prev_power.is_nan() {
            return IndicatorSignal::Neutral;
        }

        // Zero crossing signals
        // Bullish: Combined power crosses above zero
        if prev_power <= 0.0 && curr_power > 0.0 {
            return IndicatorSignal::Bullish;
        }

        // Bearish: Combined power crosses below zero
        if prev_power >= 0.0 && curr_power < 0.0 {
            return IndicatorSignal::Bearish;
        }

        // Check for divergence (need more history)
        if n >= 3 {
            let curr_close = close[n - 1];
            let prev_close = close[n - 2];

            // Bullish divergence: price making lower lows but power making higher lows
            if curr_close < prev_close && curr_power > prev_power && curr_power < 0.0 {
                return IndicatorSignal::Bullish;
            }

            // Bearish divergence: price making higher highs but power making lower highs
            if curr_close > prev_close && curr_power < prev_power && curr_power > 0.0 {
                return IndicatorSignal::Bearish;
            }
        }

        IndicatorSignal::Neutral
    }
}

impl Default for ElderPower {
    fn default() -> Self {
        Self::from_config(ElderPowerConfig::default())
    }
}

impl TechnicalIndicator for ElderPower {
    fn name(&self) -> &str {
        "ElderPower"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        // Return combined power as primary, bull power as secondary, bear power as tertiary
        Ok(IndicatorOutput::triple(
            result.combined_power,
            result.bull_power,
            result.bear_power,
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3 // combined_power, bull_power, bear_power
    }
}

impl SignalIndicator for ElderPower {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate(&data.high, &data.low, &data.close);
        Ok(self.generate_signal(&output, &data.close))
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate(&data.high, &data.low, &data.close);
        let n = output.combined_power.len();
        let mut signals = Vec::with_capacity(n);

        for i in 0..n {
            if i < 1 || output.combined_power[i].is_nan() || output.combined_power[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            let curr_power = output.combined_power[i];
            let prev_power = output.combined_power[i - 1];

            // Zero crossing signals
            if prev_power <= 0.0 && curr_power > 0.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if prev_power >= 0.0 && curr_power < 0.0 {
                signals.push(IndicatorSignal::Bearish);
            } else if i >= 2 {
                // Divergence check
                let curr_close = data.close[i];
                let prev_close = data.close[i - 1];

                if curr_close < prev_close && curr_power > prev_power && curr_power < 0.0 {
                    signals.push(IndicatorSignal::Bullish);
                } else if curr_close > prev_close && curr_power < prev_power && curr_power > 0.0 {
                    signals.push(IndicatorSignal::Bearish);
                } else {
                    signals.push(IndicatorSignal::Neutral);
                }
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

    #[test]
    fn test_elder_power_basic() {
        let ep = ElderPower::new(13);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64 * 0.5).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64 * 0.5).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect();

        let result = ep.calculate(&high, &low, &close);

        // Bull power should be positive (high > ema in uptrend)
        // Bear power should be negative (low < ema)
        for i in 13..n {
            if !result.bull_power[i].is_nan() {
                assert!(result.bull_power[i] > 0.0, "Bull power should be positive at index {}", i);
                assert!(result.bear_power[i] < 0.0, "Bear power should be negative at index {}", i);
            }
        }
    }

    #[test]
    fn test_elder_power_combined() {
        let ep = ElderPower::new(5);
        let n = 20;
        // Create data where high-low range is symmetric around EMA
        let high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let result = ep.calculate(&high, &low, &close);

        // Combined power exists after warmup
        for i in 5..n {
            assert!(!result.combined_power[i].is_nan(), "Combined power should be valid at index {}", i);
        }
    }

    #[test]
    fn test_elder_power_default() {
        let ep = ElderPower::default();
        assert_eq!(ep.period, 13);
    }

    #[test]
    fn test_elder_power_output_features() {
        let ep = ElderPower::new(13);
        assert_eq!(ep.output_features(), 3);
    }

    #[test]
    fn test_elder_power_zero_crossing_signal() {
        let ep = ElderPower::new(3);

        // Create data that will produce a zero crossing in combined power
        // Start with downtrend then transition to uptrend
        let high = vec![100.0, 99.0, 98.0, 97.0, 96.0, 100.0, 105.0, 110.0, 115.0, 120.0];
        let low = vec![95.0, 94.0, 93.0, 92.0, 91.0, 95.0, 100.0, 105.0, 110.0, 115.0];
        let close = vec![97.0, 96.0, 95.0, 94.0, 93.0, 97.0, 102.0, 107.0, 112.0, 117.0];

        let result = ep.calculate(&high, &low, &close);

        // Combined power should exist after warmup
        for i in 3..close.len() {
            assert!(!result.combined_power[i].is_nan());
        }
    }

    #[test]
    fn test_elder_power_technical_indicator() {
        use indicator_spi::TechnicalIndicator;

        let ep = ElderPower::new(5);
        let data = OHLCVSeries {
            open: vec![100.0; 20],
            high: (0..20).map(|i| 105.0 + i as f64).collect(),
            low: (0..20).map(|i| 95.0 + i as f64).collect(),
            close: (0..20).map(|i| 100.0 + i as f64).collect(),
            volume: vec![1000.0; 20],
        };

        let output = ep.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 20); // combined_power
        assert!(output.secondary.is_some());
        assert_eq!(output.secondary.as_ref().unwrap().len(), 20); // bull_power
        assert!(output.tertiary.is_some());
        assert_eq!(output.tertiary.as_ref().unwrap().len(), 20); // bear_power
    }

    #[test]
    fn test_elder_power_insufficient_data() {
        use indicator_spi::TechnicalIndicator;

        let ep = ElderPower::new(13);
        let data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![105.0; 5],
            low: vec![95.0; 5],
            close: vec![100.0; 5],
            volume: vec![1000.0; 5],
        };

        let result = ep.compute(&data);
        assert!(result.is_err());
    }
}
