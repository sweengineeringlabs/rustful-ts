//! True Strength Index (TSI).

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// True Strength Index (TSI) - IND-039
///
/// Double-smoothed momentum oscillator.
/// TSI = 100 * (Double Smoothed PC / Double Smoothed Absolute PC)
#[derive(Debug, Clone)]
pub struct TSI {
    long_period: usize,
    short_period: usize,
    signal_period: usize,
    overbought: f64,
    oversold: f64,
}

impl TSI {
    pub fn new(long_period: usize, short_period: usize, signal_period: usize) -> Self {
        Self {
            long_period,
            short_period,
            signal_period,
            overbought: 25.0,
            oversold: -25.0,
        }
    }

    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n == 0 {
            return vec![];
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut result = Vec::with_capacity(n);

        // Find first valid value
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
        if n < self.long_period + self.short_period + 1 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
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

        // Calculate TSI
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

        (tsi, signal)
    }
}

impl Default for TSI {
    fn default() -> Self {
        Self::new(25, 13, 7)
    }
}

impl TechnicalIndicator for TSI {
    fn name(&self) -> &str {
        "TSI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.long_period + self.short_period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.long_period + self.short_period + 1,
                got: data.close.len(),
            });
        }

        let (tsi, signal) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(tsi, signal))
    }

    fn min_periods(&self) -> usize {
        self.long_period + self.short_period + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for TSI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (tsi, signal) = self.calculate(&data.close);

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

        // Crossover signals
        if tsi_last > sig_last && tsi_prev <= sig_prev {
            Ok(IndicatorSignal::Bullish)
        } else if tsi_last < sig_last && tsi_prev >= sig_prev {
            Ok(IndicatorSignal::Bearish)
        } else if tsi_last >= self.overbought {
            Ok(IndicatorSignal::Bearish)
        } else if tsi_last <= self.oversold {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (tsi, signal) = self.calculate(&data.close);
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
    fn test_tsi_range() {
        let tsi = TSI::default();
        let data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let (tsi_line, signal_line) = tsi.calculate(&data);

        assert_eq!(tsi_line.len(), 100);
        assert_eq!(signal_line.len(), 100);

        // TSI should be in range [-100, 100]
        for val in tsi_line.iter() {
            if !val.is_nan() {
                assert!(*val >= -100.0 && *val <= 100.0, "TSI value {} out of range", val);
            }
        }
    }
}
