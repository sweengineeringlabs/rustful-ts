//! Special K (Pring's Special K).

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Special K - IND-150
///
/// Pring's summed ROC oscillator combining multiple timeframes.
#[derive(Debug, Clone)]
pub struct SpecialK {
    signal_period: usize,
}

impl SpecialK {
    pub fn new(signal_period: usize) -> Self {
        Self { signal_period }
    }

    fn roc(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n <= period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period];
        for i in period..n {
            if data[i - period] != 0.0 {
                result.push(((data[i] - data[i - period]) / data[i - period]) * 100.0);
            } else {
                result.push(f64::NAN);
            }
        }
        result
    }

    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            if i < period - 1 {
                result.push(f64::NAN);
            } else {
                let sum: f64 = data[(i - period + 1)..=i]
                    .iter()
                    .filter(|x| !x.is_nan())
                    .sum();
                let count = data[(i - period + 1)..=i]
                    .iter()
                    .filter(|x| !x.is_nan())
                    .count();
                if count > 0 {
                    result.push(sum / count as f64);
                } else {
                    result.push(f64::NAN);
                }
            }
        }
        result
    }

    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < 78 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Standard Special K components with weights
        let roc10 = Self::roc(data, 10);
        let roc15 = Self::roc(data, 15);
        let roc20 = Self::roc(data, 20);
        let roc30 = Self::roc(data, 30);
        let roc40 = Self::roc(data, 40);
        let roc65 = Self::roc(data, 65);
        let roc75 = Self::roc(data, 75);

        let sma10_10 = Self::sma(&roc10, 10);
        let sma10_15 = Self::sma(&roc15, 10);
        let sma10_20 = Self::sma(&roc20, 10);
        let sma15_30 = Self::sma(&roc30, 15);
        let sma50_40 = Self::sma(&roc40, 50);
        let sma65_65 = Self::sma(&roc65, 65);
        let sma75_75 = Self::sma(&roc75, 75);

        // Combine with weights: 1, 2, 3, 4, 1, 2, 3
        let special_k: Vec<f64> = (0..n)
            .map(|i| {
                let v1 = sma10_10[i];
                let v2 = sma10_15[i];
                let v3 = sma10_20[i];
                let v4 = sma15_30[i];
                let v5 = sma50_40[i];
                let v6 = sma65_65[i];
                let v7 = sma75_75[i];

                if v1.is_nan() || v2.is_nan() || v3.is_nan() || v4.is_nan()
                    || v5.is_nan() || v6.is_nan() || v7.is_nan() {
                    f64::NAN
                } else {
                    v1 * 1.0 + v2 * 2.0 + v3 * 3.0 + v4 * 4.0 + v5 * 1.0 + v6 * 2.0 + v7 * 3.0
                }
            })
            .collect();

        // Signal line
        let signal = Self::sma(&special_k, self.signal_period);

        (special_k, signal)
    }
}

impl Default for SpecialK {
    fn default() -> Self {
        Self::new(9)
    }
}

impl TechnicalIndicator for SpecialK {
    fn name(&self) -> &str {
        "SpecialK"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 150 {
            return Err(IndicatorError::InsufficientData {
                required: 150,
                got: data.close.len(),
            });
        }

        let (sk, signal) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(sk, signal))
    }

    fn min_periods(&self) -> usize {
        150
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for SpecialK {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (sk, signal) = self.calculate(&data.close);

        if sk.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let sk_last = sk[sk.len() - 1];
        let sig_last = signal[signal.len() - 1];
        let sk_prev = sk[sk.len() - 2];
        let sig_prev = signal[signal.len() - 2];

        if sk_last.is_nan() || sig_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if sk_last > sig_last && sk_prev <= sig_prev {
            Ok(IndicatorSignal::Bullish)
        } else if sk_last < sig_last && sk_prev >= sig_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (sk, signal) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..sk.len() {
            if sk[i].is_nan() || signal[i].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if sk[i] > signal[i] && sk[i-1] <= signal[i-1] {
                signals.push(IndicatorSignal::Bullish);
            } else if sk[i] < signal[i] && sk[i-1] >= signal[i-1] {
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
    fn test_special_k_basic() {
        let sk = SpecialK::default();
        let data: Vec<f64> = (0..200).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let (sk_line, signal_line) = sk.calculate(&data);

        assert_eq!(sk_line.len(), 200);
        assert_eq!(signal_line.len(), 200);
    }
}
