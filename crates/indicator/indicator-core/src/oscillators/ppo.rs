//! Percentage Price Oscillator (PPO).

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Percentage Price Oscillator (PPO) - IND-028
///
/// Similar to MACD but expressed as percentage.
/// PPO = ((Fast EMA - Slow EMA) / Slow EMA) * 100
#[derive(Debug, Clone)]
pub struct PPO {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
}

impl PPO {
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast_period: fast,
            slow_period: slow,
            signal_period: signal,
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

        // EMA calculation
        let mut prev = initial;
        for i in period..n {
            let ema = alpha * data[i] + (1.0 - alpha) * prev;
            result.push(ema);
            prev = ema;
        }

        result
    }

    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < self.slow_period {
            return (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let fast_ema = Self::ema(data, self.fast_period);
        let slow_ema = Self::ema(data, self.slow_period);

        // PPO line
        let ppo: Vec<f64> = fast_ema.iter()
            .zip(slow_ema.iter())
            .map(|(f, s)| {
                if f.is_nan() || s.is_nan() || *s == 0.0 {
                    f64::NAN
                } else {
                    ((f - s) / s) * 100.0
                }
            })
            .collect();

        // Signal line (EMA of PPO)
        let signal = Self::ema(&ppo, self.signal_period);

        // Histogram
        let histogram: Vec<f64> = ppo.iter()
            .zip(signal.iter())
            .map(|(p, s)| {
                if p.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    p - s
                }
            })
            .collect();

        (ppo, signal, histogram)
    }
}

impl Default for PPO {
    fn default() -> Self {
        Self::new(12, 26, 9)
    }
}

impl TechnicalIndicator for PPO {
    fn name(&self) -> &str {
        "PPO"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.slow_period {
            return Err(IndicatorError::InsufficientData {
                required: self.slow_period,
                got: data.close.len(),
            });
        }

        let (ppo, signal, histogram) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(ppo, signal, histogram))
    }

    fn min_periods(&self) -> usize {
        self.slow_period
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for PPO {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (ppo, signal, _) = self.calculate(&data.close);

        if ppo.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let ppo_last = ppo[ppo.len() - 1];
        let sig_last = signal[signal.len() - 1];
        let ppo_prev = ppo[ppo.len() - 2];
        let sig_prev = signal[signal.len() - 2];

        if ppo_last.is_nan() || sig_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if ppo_last > sig_last && ppo_prev <= sig_prev {
            Ok(IndicatorSignal::Bullish)
        } else if ppo_last < sig_last && ppo_prev >= sig_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (ppo, signal, _) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..ppo.len() {
            let ppo_curr = ppo[i];
            let sig_curr = signal[i];
            let ppo_prev = ppo[i - 1];
            let sig_prev = signal[i - 1];

            if ppo_curr.is_nan() || sig_curr.is_nan() || ppo_prev.is_nan() || sig_prev.is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if ppo_curr > sig_curr && ppo_prev <= sig_prev {
                signals.push(IndicatorSignal::Bullish);
            } else if ppo_curr < sig_curr && ppo_prev >= sig_prev {
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
    fn test_ppo_basic() {
        let ppo = PPO::default();
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let (ppo_line, signal_line, histogram) = ppo.calculate(&data);

        assert_eq!(ppo_line.len(), 50);
        assert_eq!(signal_line.len(), 50);
        assert_eq!(histogram.len(), 50);

        let last = ppo_line.last().unwrap();
        assert!(!last.is_nan());
    }
}
