//! Detrended Price Oscillator (DPO).

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Detrended Price Oscillator (DPO) - IND-029
///
/// Removes trend to show cycles.
/// DPO = Close - SMA(Close, Period) shifted back by (Period/2 + 1)
#[derive(Debug, Clone)]
pub struct DPO {
    period: usize,
}

impl DPO {
    pub fn new(period: usize) -> Self {
        Self { period }
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

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let shift = self.period / 2 + 1;

        if n < self.period + shift {
            return vec![f64::NAN; n];
        }

        let sma = Self::sma(data, self.period);

        // DPO = Price(shift bars ago) - SMA
        let mut result = vec![f64::NAN; shift + self.period - 1];

        for i in (shift + self.period - 1)..n {
            let price = data[i - shift];
            let ma = sma[i];

            if ma.is_nan() {
                result.push(f64::NAN);
            } else {
                result.push(price - ma);
            }
        }

        result
    }
}

impl Default for DPO {
    fn default() -> Self {
        Self::new(20)
    }
}

impl TechnicalIndicator for DPO {
    fn name(&self) -> &str {
        "DPO"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let shift = self.period / 2 + 1;

        if data.close.len() < self.period + shift {
            return Err(IndicatorError::InsufficientData {
                required: self.period + shift,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period / 2 + 1 + self.period
    }
}

impl SignalIndicator for DPO {
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
    fn test_dpo_basic() {
        let dpo = DPO::new(20);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.2).sin() * 10.0).collect();
        let result = dpo.calculate(&data);

        assert_eq!(result.len(), 50);
    }
}
