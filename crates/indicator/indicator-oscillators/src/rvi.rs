//! Relative Vigor Index (RVI).

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Relative Vigor Index (RVI) - IND-036
///
/// Compares closing price to trading range, weighted by recent data.
#[derive(Debug, Clone)]
pub struct RVI {
    period: usize,
}

impl RVI {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        if n < self.period + 3 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate numerator (close - open) and denominator (high - low) with symmetric weighting
        let mut num = vec![0.0; n];
        let mut den = vec![0.0; n];

        for i in 3..n {
            num[i] = (close[i] - open[i]) + 2.0 * (close[i-1] - open[i-1])
                     + 2.0 * (close[i-2] - open[i-2]) + (close[i-3] - open[i-3]);
            num[i] /= 6.0;

            den[i] = (high[i] - low[i]) + 2.0 * (high[i-1] - low[i-1])
                     + 2.0 * (high[i-2] - low[i-2]) + (high[i-3] - low[i-3]);
            den[i] /= 6.0;
        }

        // Calculate RVI as SMA(num) / SMA(den)
        let mut rvi = vec![f64::NAN; self.period + 2];
        let mut signal = vec![f64::NAN; self.period + 5];

        for i in (self.period + 2)..n {
            let num_sum: f64 = num[(i - self.period + 1)..=i].iter().sum();
            let den_sum: f64 = den[(i - self.period + 1)..=i].iter().sum();

            if den_sum != 0.0 {
                rvi.push(num_sum / den_sum);
            } else {
                rvi.push(0.0);
            }
        }

        // Signal line: symmetric weighted average
        for i in (self.period + 5)..n {
            let idx = i - (self.period + 2);
            if idx >= 3 && idx < rvi.len() {
                let sig = (rvi[idx] + 2.0 * rvi[idx-1] + 2.0 * rvi[idx-2] + rvi[idx-3]) / 6.0;
                signal.push(sig);
            } else {
                signal.push(f64::NAN);
            }
        }

        // Pad signal to match length
        while signal.len() < n {
            signal.push(f64::NAN);
        }

        (rvi, signal)
    }
}

impl Default for RVI {
    fn default() -> Self {
        Self::new(10)
    }
}

impl TechnicalIndicator for RVI {
    fn name(&self) -> &str {
        "RVI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 6 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 6,
                got: data.close.len(),
            });
        }

        let (rvi, signal) = self.calculate(&data.open, &data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(rvi, signal))
    }

    fn min_periods(&self) -> usize {
        self.period + 6
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for RVI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (rvi, signal) = self.calculate(&data.open, &data.high, &data.low, &data.close);

        if rvi.len() < 2 || signal.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let rvi_last = rvi[rvi.len() - 1];
        let sig_last = signal[signal.len() - 1];
        let rvi_prev = rvi[rvi.len() - 2];
        let sig_prev = signal[signal.len() - 2];

        if rvi_last.is_nan() || sig_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if rvi_last > sig_last && rvi_prev <= sig_prev {
            Ok(IndicatorSignal::Bullish)
        } else if rvi_last < sig_last && rvi_prev >= sig_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (rvi, signal) = self.calculate(&data.open, &data.high, &data.low, &data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..rvi.len().min(signal.len()) {
            if rvi[i].is_nan() || signal[i].is_nan() || rvi[i-1].is_nan() || signal[i-1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if rvi[i] > signal[i] && rvi[i-1] <= signal[i-1] {
                signals.push(IndicatorSignal::Bullish);
            } else if rvi[i] < signal[i] && rvi[i-1] >= signal[i-1] {
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
    fn test_rvi_basic() {
        let rvi = RVI::new(10);
        let n = 50;
        let open: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 103.0 + i as f64).collect();

        let (rvi_line, signal_line) = rvi.calculate(&open, &high, &low, &close);

        assert_eq!(rvi_line.len(), n);
        assert_eq!(signal_line.len(), n);
    }
}
