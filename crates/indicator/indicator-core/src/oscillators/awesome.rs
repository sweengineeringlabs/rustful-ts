//! Awesome Oscillator (AO) - Bill Williams.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Awesome Oscillator - IND-007
///
/// Bill Williams indicator: 5-period SMA of median - 34-period SMA of median.
/// Median = (High + Low) / 2
#[derive(Debug, Clone)]
pub struct AwesomeOscillator {
    fast_period: usize,
    slow_period: usize,
}

impl AwesomeOscillator {
    pub fn new() -> Self {
        Self {
            fast_period: 5,
            slow_period: 34,
        }
    }

    pub fn with_periods(fast: usize, slow: usize) -> Self {
        Self {
            fast_period: fast,
            slow_period: slow,
        }
    }

    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().sum();
        result.push(sum / period as f64);

        for i in period..n {
            sum = sum - data[i - period] + data[i];
            result.push(sum / period as f64);
        }

        result
    }

    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < self.slow_period {
            return vec![f64::NAN; n];
        }

        // Calculate median prices
        let median: Vec<f64> = high.iter()
            .zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        let fast_sma = Self::sma(&median, self.fast_period);
        let slow_sma = Self::sma(&median, self.slow_period);

        fast_sma.iter()
            .zip(slow_sma.iter())
            .map(|(f, s)| {
                if f.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    f - s
                }
            })
            .collect()
    }
}

impl Default for AwesomeOscillator {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for AwesomeOscillator {
    fn name(&self) -> &str {
        "AO"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.slow_period {
            return Err(IndicatorError::InsufficientData {
                required: self.slow_period,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.slow_period
    }
}

impl SignalIndicator for AwesomeOscillator {
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

        // Bullish when AO crosses above zero or increases
        if last > 0.0 && prev <= 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < 0.0 && prev >= 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else if last > prev {
            Ok(IndicatorSignal::Bullish)
        } else if last < prev {
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
            } else if curr > 0.0 && prev <= 0.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if curr < 0.0 && prev >= 0.0 {
                signals.push(IndicatorSignal::Bearish);
            } else if curr > prev {
                signals.push(IndicatorSignal::Bullish);
            } else if curr < prev {
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
    fn test_ao_basic() {
        let ao = AwesomeOscillator::new();
        let high: Vec<f64> = (0..50).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..50).map(|i| 90.0 + i as f64).collect();
        let result = ao.calculate(&high, &low);

        // In an uptrend, AO should be positive after warmup
        let last = result.last().unwrap();
        assert!(!last.is_nan());
        assert!(*last > 0.0);
    }
}
