//! Accelerator Oscillator (AC) - Bill Williams.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Accelerator Oscillator - IND-008
///
/// Bill Williams indicator: AO - 5-period SMA of AO.
/// Measures the acceleration/deceleration of market momentum.
#[derive(Debug, Clone)]
pub struct AcceleratorOscillator {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
}

impl AcceleratorOscillator {
    pub fn new() -> Self {
        Self {
            fast_period: 5,
            slow_period: 34,
            signal_period: 5,
        }
    }

    pub fn with_periods(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast_period: fast,
            slow_period: slow,
            signal_period: signal,
        }
    }

    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().filter(|x| !x.is_nan()).sum();
        result.push(sum / period as f64);

        for i in period..n {
            let old = if data[i - period].is_nan() { 0.0 } else { data[i - period] };
            let new = if data[i].is_nan() { 0.0 } else { data[i] };
            sum = sum - old + new;
            result.push(sum / period as f64);
        }

        result
    }

    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < self.slow_period + self.signal_period {
            return vec![f64::NAN; n];
        }

        // Calculate median prices
        let median: Vec<f64> = high.iter()
            .zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        // Calculate AO
        let fast_sma = Self::sma(&median, self.fast_period);
        let slow_sma = Self::sma(&median, self.slow_period);

        let ao: Vec<f64> = fast_sma.iter()
            .zip(slow_sma.iter())
            .map(|(f, s)| {
                if f.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    f - s
                }
            })
            .collect();

        // AC = AO - SMA(AO, signal_period)
        let ao_sma = Self::sma(&ao, self.signal_period);

        ao.iter()
            .zip(ao_sma.iter())
            .map(|(a, s)| {
                if a.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    a - s
                }
            })
            .collect()
    }
}

impl Default for AcceleratorOscillator {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for AcceleratorOscillator {
    fn name(&self) -> &str {
        "AC"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.slow_period + self.signal_period {
            return Err(IndicatorError::InsufficientData {
                required: self.slow_period + self.signal_period,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.slow_period + self.signal_period
    }
}

impl SignalIndicator for AcceleratorOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low);
        if values.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let last = values[values.len() - 1];
        let prev = values[values.len() - 2];

        if last.is_nan() || prev.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if last > prev && last > 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < prev && last < 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..values.len() {
            let curr = values[i];
            let prev = values[i - 1];

            if curr.is_nan() || prev.is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if curr > prev && curr > 0.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if curr < prev && curr < 0.0 {
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

    #[test]
    fn test_ac_basic() {
        let ac = AcceleratorOscillator::new();
        let high: Vec<f64> = (0..50).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..50).map(|i| 90.0 + i as f64).collect();
        let result = ac.calculate(&high, &low);

        assert_eq!(result.len(), 50);
        // Last value should be valid
        let last = result.last().unwrap();
        assert!(!last.is_nan());
    }
}
