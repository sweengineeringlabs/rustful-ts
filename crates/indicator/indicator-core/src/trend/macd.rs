//! MACD (Moving Average Convergence Divergence) implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_api::MACDConfig;
use crate::EMA;

/// MACD (Moving Average Convergence Divergence).
///
/// Trend-following momentum indicator showing relationship between two EMAs.
/// Outputs: MACD line, Signal line, Histogram.
#[derive(Debug, Clone)]
pub struct MACD {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
}

impl MACD {
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast_period: fast,
            slow_period: slow,
            signal_period: signal,
        }
    }

    pub fn from_config(config: MACDConfig) -> Self {
        Self {
            fast_period: config.fast_period,
            slow_period: config.slow_period,
            signal_period: config.signal_period,
        }
    }

    /// Calculate MACD values (macd_line, signal_line, histogram).
    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = data.len();
        let min_required = self.slow_period + self.signal_period - 1;

        if n < min_required {
            return (
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
            );
        }

        // Calculate EMAs
        let fast_ema = EMA::new(self.fast_period).calculate(data);
        let slow_ema = EMA::new(self.slow_period).calculate(data);

        // MACD line = Fast EMA - Slow EMA
        let macd_line: Vec<f64> = fast_ema.iter()
            .zip(slow_ema.iter())
            .map(|(f, s)| {
                if f.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    f - s
                }
            })
            .collect();

        // Signal line = EMA of MACD line
        let signal_ema = EMA::new(self.signal_period);
        let signal_line = signal_ema.calculate(&macd_line);

        // Histogram = MACD - Signal
        let histogram: Vec<f64> = macd_line.iter()
            .zip(signal_line.iter())
            .map(|(m, s)| {
                if m.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    m - s
                }
            })
            .collect();

        (macd_line, signal_line, histogram)
    }
}

impl TechnicalIndicator for MACD {
    fn name(&self) -> &str {
        "MACD"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.slow_period + self.signal_period - 1;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (macd, signal, histogram) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(macd, signal, histogram))
    }

    fn min_periods(&self) -> usize {
        self.slow_period + self.signal_period - 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for MACD {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (macd, signal, _) = self.calculate(&data.close);

        if macd.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = macd.len();
        let curr_macd = macd[n - 1];
        let prev_macd = macd[n - 2];
        let curr_signal = signal[n - 1];
        let prev_signal = signal[n - 2];

        if curr_macd.is_nan() || curr_signal.is_nan() ||
           prev_macd.is_nan() || prev_signal.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish crossover: MACD crosses above signal
        if prev_macd <= prev_signal && curr_macd > curr_signal {
            Ok(IndicatorSignal::Bullish)
        }
        // Bearish crossover: MACD crosses below signal
        else if prev_macd >= prev_signal && curr_macd < curr_signal {
            Ok(IndicatorSignal::Bearish)
        }
        else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (macd, signal, _) = self.calculate(&data.close);
        let n = macd.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let curr_macd = macd[i];
            let prev_macd = macd[i - 1];
            let curr_signal = signal[i];
            let prev_signal = signal[i - 1];

            if curr_macd.is_nan() || curr_signal.is_nan() ||
               prev_macd.is_nan() || prev_signal.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if prev_macd <= prev_signal && curr_macd > curr_signal {
                signals.push(IndicatorSignal::Bullish);
            } else if prev_macd >= prev_signal && curr_macd < curr_signal {
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
    fn test_macd_basic() {
        let macd = MACD::new(12, 26, 9);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let (macd_line, signal_line, histogram) = macd.calculate(&data);

        assert_eq!(macd_line.len(), 50);
        assert_eq!(signal_line.len(), 50);
        assert_eq!(histogram.len(), 50);

        // Check that valid values exist after warmup
        let valid_macd: Vec<_> = macd_line.iter().filter(|x| !x.is_nan()).collect();
        assert!(!valid_macd.is_empty());
    }
}
