//! Know Sure Thing (KST) indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Know Sure Thing (KST) - IND-027
///
/// Weighted sum of four smoothed ROC values.
/// KST = (ROCMA1 × 1) + (ROCMA2 × 2) + (ROCMA3 × 3) + (ROCMA4 × 4)
#[derive(Debug, Clone)]
pub struct KST {
    roc_periods: [usize; 4],
    sma_periods: [usize; 4],
    signal_period: usize,
}

impl KST {
    pub fn new() -> Self {
        Self {
            roc_periods: [10, 15, 20, 30],
            sma_periods: [10, 10, 10, 15],
            signal_period: 9,
        }
    }

    pub fn with_params(roc_periods: [usize; 4], sma_periods: [usize; 4], signal_period: usize) -> Self {
        Self { roc_periods, sma_periods, signal_period }
    }

    fn roc(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n <= period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period];
        for i in period..n {
            let prev = data[i - period];
            if prev != 0.0 {
                result.push(((data[i] - prev) / prev) * 100.0);
            } else {
                result.push(f64::NAN);
            }
        }
        result
    }

    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period || period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..n {
            let start_idx = i + 1 - period;
            let sum: f64 = data[start_idx..=i]
                .iter()
                .filter(|x| !x.is_nan())
                .sum();
            let count = data[start_idx..=i]
                .iter()
                .filter(|x| !x.is_nan())
                .count();
            if count > 0 {
                result.push(sum / count as f64);
            } else {
                result.push(f64::NAN);
            }
        }
        result
    }

    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        let min_req = self.roc_periods[3] + self.sma_periods[3] + self.signal_period;

        if n < min_req {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate ROC for each period
        let roc1 = Self::roc(data, self.roc_periods[0]);
        let roc2 = Self::roc(data, self.roc_periods[1]);
        let roc3 = Self::roc(data, self.roc_periods[2]);
        let roc4 = Self::roc(data, self.roc_periods[3]);

        // Smooth each ROC
        let rocma1 = Self::sma(&roc1, self.sma_periods[0]);
        let rocma2 = Self::sma(&roc2, self.sma_periods[1]);
        let rocma3 = Self::sma(&roc3, self.sma_periods[2]);
        let rocma4 = Self::sma(&roc4, self.sma_periods[3]);

        // Calculate KST
        let kst: Vec<f64> = (0..n)
            .map(|i| {
                let r1 = rocma1[i];
                let r2 = rocma2[i];
                let r3 = rocma3[i];
                let r4 = rocma4[i];

                if r1.is_nan() || r2.is_nan() || r3.is_nan() || r4.is_nan() {
                    f64::NAN
                } else {
                    r1 * 1.0 + r2 * 2.0 + r3 * 3.0 + r4 * 4.0
                }
            })
            .collect();

        // Signal line
        let signal = Self::sma(&kst, self.signal_period);

        (kst, signal)
    }
}

impl Default for KST {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for KST {
    fn name(&self) -> &str {
        "KST"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_req = self.roc_periods[3] + self.sma_periods[3] + self.signal_period;

        if data.close.len() < min_req {
            return Err(IndicatorError::InsufficientData {
                required: min_req,
                got: data.close.len(),
            });
        }

        let (kst, signal) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(kst, signal))
    }

    fn min_periods(&self) -> usize {
        self.roc_periods[3] + self.sma_periods[3] + self.signal_period
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for KST {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (kst, signal) = self.calculate(&data.close);

        if kst.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let kst_last = kst[kst.len() - 1];
        let sig_last = signal[signal.len() - 1];
        let kst_prev = kst[kst.len() - 2];
        let sig_prev = signal[signal.len() - 2];

        if kst_last.is_nan() || sig_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish crossover
        if kst_last > sig_last && kst_prev <= sig_prev {
            Ok(IndicatorSignal::Bullish)
        } else if kst_last < sig_last && kst_prev >= sig_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (kst, signal) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..kst.len() {
            let kst_curr = kst[i];
            let sig_curr = signal[i];
            let kst_prev = kst[i - 1];
            let sig_prev = signal[i - 1];

            if kst_curr.is_nan() || sig_curr.is_nan() || kst_prev.is_nan() || sig_prev.is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if kst_curr > sig_curr && kst_prev <= sig_prev {
                signals.push(IndicatorSignal::Bullish);
            } else if kst_curr < sig_curr && kst_prev >= sig_prev {
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
    fn test_kst_basic() {
        let kst = KST::new();
        let data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        let (kst_line, signal_line) = kst.calculate(&data);

        assert_eq!(kst_line.len(), 100);
        assert_eq!(signal_line.len(), 100);

        // In uptrend, KST should be positive
        let last = kst_line.last().unwrap();
        assert!(!last.is_nan());
        assert!(*last > 0.0);
    }
}
