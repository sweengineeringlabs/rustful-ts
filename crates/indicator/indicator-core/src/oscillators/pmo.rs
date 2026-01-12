//! Price Momentum Oscillator (PMO).

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Price Momentum Oscillator (PMO) - IND-149
///
/// DecisionPoint PMO: Double smoothed ROC.
#[derive(Debug, Clone)]
pub struct PMO {
    smooth1: usize,
    smooth2: usize,
    signal_period: usize,
}

impl PMO {
    pub fn new(smooth1: usize, smooth2: usize, signal_period: usize) -> Self {
        Self { smooth1, smooth2, signal_period }
    }

    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut result = Vec::with_capacity(n);

        let first_valid = data.iter().position(|x| !x.is_nan());
        if first_valid.is_none() {
            return vec![f64::NAN; n];
        }

        let start = first_valid.unwrap();
        for _ in 0..start {
            result.push(f64::NAN);
        }

        let mut prev = data[start];
        result.push(prev);

        for i in (start + 1)..n {
            if data[i].is_nan() {
                result.push(prev);
            } else {
                let ema = alpha * data[i] + (1.0 - alpha) * prev;
                result.push(ema);
                prev = ema;
            }
        }

        result
    }

    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < 2 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate 1-period ROC
        let mut roc = vec![f64::NAN; 1];
        for i in 1..n {
            if data[i - 1] != 0.0 {
                roc.push(((data[i] / data[i - 1]) - 1.0) * 100.0);
            } else {
                roc.push(f64::NAN);
            }
        }

        // First smoothing (35-period EMA of ROC * 10)
        let scaled_roc: Vec<f64> = roc.iter().map(|&x| if x.is_nan() { f64::NAN } else { x * 10.0 }).collect();
        let smooth1_ema = Self::ema(&scaled_roc, self.smooth1);

        // Second smoothing (20-period EMA)
        let pmo = Self::ema(&smooth1_ema, self.smooth2);

        // Signal line
        let signal = Self::ema(&pmo, self.signal_period);

        (pmo, signal)
    }
}

impl Default for PMO {
    fn default() -> Self {
        Self::new(35, 20, 10)
    }
}

impl TechnicalIndicator for PMO {
    fn name(&self) -> &str {
        "PMO"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.smooth1 + self.smooth2 {
            return Err(IndicatorError::InsufficientData {
                required: self.smooth1 + self.smooth2,
                got: data.close.len(),
            });
        }

        let (pmo, signal) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(pmo, signal))
    }

    fn min_periods(&self) -> usize {
        self.smooth1 + self.smooth2
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for PMO {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (pmo, signal) = self.calculate(&data.close);

        if pmo.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let pmo_last = pmo[pmo.len() - 1];
        let sig_last = signal[signal.len() - 1];
        let pmo_prev = pmo[pmo.len() - 2];
        let sig_prev = signal[signal.len() - 2];

        if pmo_last.is_nan() || sig_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if pmo_last > sig_last && pmo_prev <= sig_prev {
            Ok(IndicatorSignal::Bullish)
        } else if pmo_last < sig_last && pmo_prev >= sig_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (pmo, signal) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..pmo.len() {
            if pmo[i].is_nan() || signal[i].is_nan() || pmo[i-1].is_nan() || signal[i-1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if pmo[i] > signal[i] && pmo[i-1] <= signal[i-1] {
                signals.push(IndicatorSignal::Bullish);
            } else if pmo[i] < signal[i] && pmo[i-1] >= signal[i-1] {
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
    fn test_pmo_basic() {
        let pmo = PMO::default();
        let data: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();
        let (pmo_line, signal_line) = pmo.calculate(&data);

        assert_eq!(pmo_line.len(), 100);
        assert_eq!(signal_line.len(), 100);
    }
}
