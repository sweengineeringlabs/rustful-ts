//! Absolute Price Oscillator (APO).

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Absolute Price Oscillator (APO) - IND-153
///
/// Fast MA - Slow MA (absolute difference, not percentage).
#[derive(Debug, Clone)]
pub struct APO {
    fast_period: usize,
    slow_period: usize,
}

impl APO {
    pub fn new(fast: usize, slow: usize) -> Self {
        Self {
            fast_period: fast,
            slow_period: slow,
        }
    }

    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut result = Vec::with_capacity(n);

        // Initial SMA
        let initial: f64 = data[..period].iter().sum::<f64>() / period as f64;
        for _ in 0..period - 1 {
            result.push(f64::NAN);
        }
        result.push(initial);

        let mut prev = initial;
        for i in period..n {
            let ema = alpha * data[i] + (1.0 - alpha) * prev;
            result.push(ema);
            prev = ema;
        }

        result
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.slow_period {
            return vec![f64::NAN; n];
        }

        let fast_ema = Self::ema(data, self.fast_period);
        let slow_ema = Self::ema(data, self.slow_period);

        fast_ema.iter()
            .zip(slow_ema.iter())
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

impl Default for APO {
    fn default() -> Self {
        Self::new(12, 26)
    }
}

impl TechnicalIndicator for APO {
    fn name(&self) -> &str {
        "APO"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.slow_period {
            return Err(IndicatorError::InsufficientData {
                required: self.slow_period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.slow_period
    }
}

impl SignalIndicator for APO {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);

        if values.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let last = values[values.len() - 1];
        let prev = values[values.len() - 2];

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Zero-line crossover
        if last > 0.0 && prev <= 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < 0.0 && prev >= 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..values.len() {
            if values[i].is_nan() || values[i-1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if values[i] > 0.0 && values[i-1] <= 0.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if values[i] < 0.0 && values[i-1] >= 0.0 {
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
    fn test_apo_basic() {
        let apo = APO::default();
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let result = apo.calculate(&data);

        assert_eq!(result.len(), 50);
        // In uptrend, APO should be positive
        let last = result.last().unwrap();
        assert!(!last.is_nan());
        assert!(*last > 0.0);
    }
}
