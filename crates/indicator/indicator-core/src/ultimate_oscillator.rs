//! Ultimate Oscillator implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};
use indicator_api::UltimateOscillatorConfig;

/// Ultimate Oscillator.
///
/// Combines short, medium, and long-term price momentum.
/// UO = 100 * (4*Avg7 + 2*Avg14 + Avg28) / 7
/// - Above 70: Overbought
/// - Below 30: Oversold
#[derive(Debug, Clone)]
pub struct UltimateOscillator {
    period1: usize,  // Short
    period2: usize,  // Medium
    period3: usize,  // Long
}

impl UltimateOscillator {
    pub fn new(short: usize, medium: usize, long: usize) -> Self {
        Self {
            period1: short,
            period2: medium,
            period3: long,
        }
    }

    pub fn from_config(config: UltimateOscillatorConfig) -> Self {
        Self {
            period1: config.period1,
            period2: config.period2,
            period3: config.period3,
        }
    }

    /// Calculate Ultimate Oscillator values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let max_period = self.period1.max(self.period2).max(self.period3);
        let mut result = vec![f64::NAN; n];

        if n < max_period + 1 {
            return result;
        }

        // Calculate Buying Pressure (BP) and True Range (TR) for each bar
        let mut bp: Vec<f64> = Vec::with_capacity(n);
        let mut tr: Vec<f64> = Vec::with_capacity(n);

        bp.push(0.0);
        tr.push(high[0] - low[0]);

        for i in 1..n {
            let prev_close = close[i - 1];
            let true_low = low[i].min(prev_close);
            let true_high = high[i].max(prev_close);

            bp.push(close[i] - true_low);
            tr.push(true_high - true_low);
        }

        // Calculate Ultimate Oscillator
        for i in max_period..n {
            // Calculate averages for each period
            let bp1: f64 = bp[(i + 1 - self.period1)..=i].iter().sum();
            let tr1: f64 = tr[(i + 1 - self.period1)..=i].iter().sum();
            let avg1 = if tr1 > 0.0 { bp1 / tr1 } else { 0.0 };

            let bp2: f64 = bp[(i + 1 - self.period2)..=i].iter().sum();
            let tr2: f64 = tr[(i + 1 - self.period2)..=i].iter().sum();
            let avg2 = if tr2 > 0.0 { bp2 / tr2 } else { 0.0 };

            let bp3: f64 = bp[(i + 1 - self.period3)..=i].iter().sum();
            let tr3: f64 = tr[(i + 1 - self.period3)..=i].iter().sum();
            let avg3 = if tr3 > 0.0 { bp3 / tr3 } else { 0.0 };

            // Weighted average: 4 * short + 2 * medium + 1 * long, divided by 7
            result[i] = 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0;
        }

        result
    }
}

impl Default for UltimateOscillator {
    fn default() -> Self {
        Self::from_config(UltimateOscillatorConfig::default())
    }
}

impl TechnicalIndicator for UltimateOscillator {
    fn name(&self) -> &str {
        "UltimateOscillator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let max_period = self.period1.max(self.period2).max(self.period3);
        if data.close.len() < max_period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: max_period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period1.max(self.period2).max(self.period3) + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for UltimateOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last > 70.0 {
                    return Ok(IndicatorSignal::Bearish);  // Overbought
                } else if last < 30.0 {
                    return Ok(IndicatorSignal::Bullish);  // Oversold
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(values.iter().map(|&v| {
            if v.is_nan() {
                IndicatorSignal::Neutral
            } else if v > 70.0 {
                IndicatorSignal::Bearish
            } else if v < 30.0 {
                IndicatorSignal::Bullish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ultimate_oscillator() {
        let uo = UltimateOscillator::default();
        let n = 50;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let result = uo.calculate(&high, &low, &close);

        // Should have NaN initially
        assert!(result[0].is_nan());
        // After max period, should have values between 0 and 100
        assert!(!result[49].is_nan());
        assert!(result[49] >= 0.0 && result[49] <= 100.0);
    }
}
