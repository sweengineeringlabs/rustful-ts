//! Klinger Oscillator implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Klinger Oscillator (Klinger Volume Oscillator - KVO).
///
/// The Klinger Oscillator uses high-low-close and volume to measure
/// the long-term trend of money flow.
///
/// Trend = Current HLC > Previous HLC (volume multiplier +1/-1)
/// dm = high - low
/// cm = dm + previous cm (if same trend) or dm + previous dm (if trend reversal)
/// Volume Force (VF) = Volume * |2*(dm/cm) - 1| * Trend * 100
/// KVO = EMA(34) of VF - EMA(55) of VF
/// Signal = EMA(13) of KVO
#[derive(Debug, Clone)]
pub struct KlingerOscillator {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
}

impl KlingerOscillator {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            signal_period,
        }
    }

    /// Calculate EMA
    fn ema(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let alpha = 2.0 / (period as f64 + 1.0);

        // Find first non-NaN and calculate initial SMA
        let mut first_valid = None;
        for i in 0..n {
            if !values[i].is_nan() {
                first_valid = Some(i);
                break;
            }
        }

        if first_valid.is_none() {
            return result;
        }

        let start = first_valid.unwrap();
        if start + period > n {
            return result;
        }

        let mut sum = 0.0;
        for i in start..(start + period) {
            sum += values[i];
        }
        result[start + period - 1] = sum / period as f64;

        for i in (start + period)..n {
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate Klinger Oscillator values.
    /// Returns (KVO, Signal Line)
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();

        if n < 2 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate HLC (typical price proxy)
        let hlc: Vec<f64> = (0..n).map(|i| high[i] + low[i] + close[i]).collect();

        // Calculate trend direction
        let mut trend = vec![0.0_f64; n];
        for i in 1..n {
            trend[i] = if hlc[i] > hlc[i - 1] { 1.0 } else { -1.0 };
        }

        // Calculate dm and cm
        let dm: Vec<f64> = (0..n).map(|i| high[i] - low[i]).collect();
        let mut cm = vec![0.0_f64; n];
        cm[0] = dm[0];
        for i in 1..n {
            if trend[i] == trend[i - 1] || trend[i - 1] == 0.0 {
                cm[i] = cm[i - 1] + dm[i];
            } else {
                cm[i] = dm[i - 1] + dm[i];
            }
        }

        // Calculate Volume Force
        let mut vf = vec![0.0_f64; n];
        for i in 0..n {
            if cm[i] != 0.0 {
                let temp = (2.0 * dm[i] / cm[i] - 1.0).abs();
                vf[i] = volume[i] * temp * trend[i] * 100.0;
            }
        }

        // Calculate KVO = Fast EMA - Slow EMA of VF
        let fast_ema = self.ema(&vf, self.fast_period);
        let slow_ema = self.ema(&vf, self.slow_period);

        let mut kvo = vec![f64::NAN; n];
        for i in 0..n {
            if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
                kvo[i] = fast_ema[i] - slow_ema[i];
            }
        }

        // Calculate signal line
        let signal = self.ema(&kvo, self.signal_period);

        (kvo, signal)
    }
}

impl Default for KlingerOscillator {
    fn default() -> Self {
        Self {
            fast_period: 34,
            slow_period: 55,
            signal_period: 13,
        }
    }
}

impl TechnicalIndicator for KlingerOscillator {
    fn name(&self) -> &str {
        "Klinger Oscillator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.slow_period {
            return Err(IndicatorError::InsufficientData {
                required: self.slow_period,
                got: data.close.len(),
            });
        }

        let (kvo, signal) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::dual(kvo, signal))
    }

    fn min_periods(&self) -> usize {
        self.slow_period
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for KlingerOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (kvo, signal) = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        if kvo.len() < 2 || signal.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = kvo.len();
        let curr_kvo = kvo[n - 1];
        let prev_kvo = kvo[n - 2];
        let curr_sig = signal[n - 1];
        let prev_sig = signal[n - 2];

        if curr_kvo.is_nan() || curr_sig.is_nan() || prev_kvo.is_nan() || prev_sig.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: KVO crosses above signal
        if prev_kvo <= prev_sig && curr_kvo > curr_sig {
            return Ok(IndicatorSignal::Bullish);
        }
        // Bearish: KVO crosses below signal
        if prev_kvo >= prev_sig && curr_kvo < curr_sig {
            return Ok(IndicatorSignal::Bearish);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (kvo, signal) = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..kvo.len() {
            if kvo[i].is_nan() || signal[i].is_nan() || kvo[i - 1].is_nan() || signal[i - 1].is_nan()
            {
                signals.push(IndicatorSignal::Neutral);
            } else if kvo[i - 1] <= signal[i - 1] && kvo[i] > signal[i] {
                signals.push(IndicatorSignal::Bullish);
            } else if kvo[i - 1] >= signal[i - 1] && kvo[i] < signal[i] {
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
    fn test_klinger_oscillator() {
        let kvo = KlingerOscillator::new(5, 10, 3);
        let n = 20;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let volume: Vec<f64> = (0..n).map(|_| 1000.0).collect();

        let (kvo_values, signal_values) = kvo.calculate(&high, &low, &close, &volume);

        assert_eq!(kvo_values.len(), n);
        assert_eq!(signal_values.len(), n);
    }
}
