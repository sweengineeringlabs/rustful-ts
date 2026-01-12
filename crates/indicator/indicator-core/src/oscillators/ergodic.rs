//! Ergodic Oscillator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Ergodic Oscillator - IND-154
///
/// True Strength Index with signal line (TSI signal line).
#[derive(Debug, Clone)]
pub struct ErgodicOscillator {
    long_period: usize,
    short_period: usize,
    signal_period: usize,
}

impl ErgodicOscillator {
    pub fn new(long_period: usize, short_period: usize, signal_period: usize) -> Self {
        Self { long_period, short_period, signal_period }
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

    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < self.long_period + self.short_period + 1 {
            return (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Price change
        let mut pc = vec![f64::NAN; 1];
        for i in 1..n {
            pc.push(data[i] - data[i - 1]);
        }

        // Absolute price change
        let abs_pc: Vec<f64> = pc.iter().map(|&x| if x.is_nan() { f64::NAN } else { x.abs() }).collect();

        // Double smooth price change
        let pc_ema1 = Self::ema(&pc, self.long_period);
        let pc_ema2 = Self::ema(&pc_ema1, self.short_period);

        // Double smooth absolute price change
        let abs_pc_ema1 = Self::ema(&abs_pc, self.long_period);
        let abs_pc_ema2 = Self::ema(&abs_pc_ema1, self.short_period);

        // TSI line
        let tsi: Vec<f64> = pc_ema2.iter()
            .zip(abs_pc_ema2.iter())
            .map(|(pc, abs_pc)| {
                if pc.is_nan() || abs_pc.is_nan() || *abs_pc == 0.0 {
                    f64::NAN
                } else {
                    (pc / abs_pc) * 100.0
                }
            })
            .collect();

        // Signal line
        let signal = Self::ema(&tsi, self.signal_period);

        // Histogram
        let histogram: Vec<f64> = tsi.iter()
            .zip(signal.iter())
            .map(|(t, s)| {
                if t.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    t - s
                }
            })
            .collect();

        (tsi, signal, histogram)
    }
}

impl Default for ErgodicOscillator {
    fn default() -> Self {
        Self::new(25, 13, 5)
    }
}

impl TechnicalIndicator for ErgodicOscillator {
    fn name(&self) -> &str {
        "Ergodic"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.long_period + self.short_period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.long_period + self.short_period + 1,
                got: data.close.len(),
            });
        }

        let (tsi, signal, histogram) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(tsi, signal, histogram))
    }

    fn min_periods(&self) -> usize {
        self.long_period + self.short_period + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for ErgodicOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (tsi, signal, _) = self.calculate(&data.close);

        if tsi.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let tsi_last = tsi[tsi.len() - 1];
        let sig_last = signal[signal.len() - 1];
        let tsi_prev = tsi[tsi.len() - 2];
        let sig_prev = signal[signal.len() - 2];

        if tsi_last.is_nan() || sig_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if tsi_last > sig_last && tsi_prev <= sig_prev {
            Ok(IndicatorSignal::Bullish)
        } else if tsi_last < sig_last && tsi_prev >= sig_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (tsi, signal, _) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..tsi.len() {
            if tsi[i].is_nan() || signal[i].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if tsi[i] > signal[i] && tsi[i-1] <= signal[i-1] {
                signals.push(IndicatorSignal::Bullish);
            } else if tsi[i] < signal[i] && tsi[i-1] >= signal[i-1] {
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
    fn test_ergodic_basic() {
        let ergodic = ErgodicOscillator::default();
        let data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let (tsi, signal, histogram) = ergodic.calculate(&data);

        assert_eq!(tsi.len(), 100);
        assert_eq!(signal.len(), 100);
        assert_eq!(histogram.len(), 100);
    }
}
