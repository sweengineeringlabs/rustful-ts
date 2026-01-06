//! Stochastic Oscillator implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_api::StochasticConfig;
use crate::SMA;

/// Stochastic Oscillator.
///
/// Momentum indicator comparing closing price to price range.
/// Outputs %K (fast) and %D (slow) lines.
#[derive(Debug, Clone)]
pub struct Stochastic {
    k_period: usize,
    d_period: usize,
    overbought: f64,
    oversold: f64,
}

impl Stochastic {
    pub fn new(k_period: usize, d_period: usize) -> Self {
        Self {
            k_period,
            d_period,
            overbought: 80.0,
            oversold: 20.0,
        }
    }

    pub fn from_config(config: StochasticConfig) -> Self {
        Self {
            k_period: config.k_period,
            d_period: config.d_period,
            overbought: config.overbought,
            oversold: config.oversold,
        }
    }

    /// Calculate Stochastic (%K, %D).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        if n < self.k_period || self.k_period == 0 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate %K
        let mut k_line = vec![f64::NAN; self.k_period - 1];

        for i in (self.k_period - 1)..n {
            let start = i + 1 - self.k_period;
            let window_high = &high[start..=i];
            let window_low = &low[start..=i];

            let highest = window_high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = window_low.iter().cloned().fold(f64::INFINITY, f64::min);

            let range = highest - lowest;
            if range.abs() < 1e-10 {
                k_line.push(50.0); // Neutral when no range
            } else {
                let k = ((close[i] - lowest) / range) * 100.0;
                k_line.push(k);
            }
        }

        // Calculate %D (SMA of %K)
        let sma = SMA::new(self.d_period);
        let d_line = sma.calculate(&k_line);

        (k_line, d_line)
    }
}

impl TechnicalIndicator for Stochastic {
    fn name(&self) -> &str {
        "Stochastic"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.k_period + self.d_period - 1;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(k_line, d_line))
    }

    fn min_periods(&self) -> usize {
        self.k_period + self.d_period - 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for Stochastic {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);

        if k_line.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = k_line.len();
        let k = k_line[n - 1];
        let d = d_line[n - 1];
        let prev_k = k_line[n - 2];
        let prev_d = d_line[n - 2];

        if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: %K crosses above %D in oversold zone
        if prev_k <= prev_d && k > d && k < self.oversold + 10.0 {
            Ok(IndicatorSignal::Bullish)
        }
        // Bearish: %K crosses below %D in overbought zone
        else if prev_k >= prev_d && k < d && k > self.overbought - 10.0 {
            Ok(IndicatorSignal::Bearish)
        }
        else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);
        let n = k_line.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let k = k_line[i];
            let d = d_line[i];
            let prev_k = k_line[i - 1];
            let prev_d = d_line[i - 1];

            if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if prev_k <= prev_d && k > d && k < self.oversold + 10.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if prev_k >= prev_d && k < d && k > self.overbought - 10.0 {
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
    fn test_stochastic() {
        let stoch = Stochastic::new(14, 3);

        // Create sample OHLC data
        let high: Vec<f64> = (0..30).map(|i| 105.0 + (i as f64).sin() * 5.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 95.0 + (i as f64).sin() * 5.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();

        let (k_line, d_line) = stoch.calculate(&high, &low, &close);

        assert_eq!(k_line.len(), 30);
        assert_eq!(d_line.len(), 30);

        // %K values should be between 0 and 100
        for i in 13..30 {
            if !k_line[i].is_nan() {
                assert!(k_line[i] >= 0.0 && k_line[i] <= 100.0);
            }
        }
    }
}
