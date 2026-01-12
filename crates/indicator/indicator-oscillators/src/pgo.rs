//! Pretty Good Oscillator (PGO).

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Pretty Good Oscillator (PGO) - IND-152
///
/// ATR-normalized distance from SMA.
/// PGO = (Close - SMA) / (EMA of ATR)
#[derive(Debug, Clone)]
pub struct PrettyGoodOscillator {
    period: usize,
}

impl PrettyGoodOscillator {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().sum();
        result.push(sum / period as f64);

        for i in period..n {
            sum = sum - data[i - period] + data[i];
            result.push(sum / period as f64);
        }

        result
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

    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        // Calculate True Range
        let mut tr = vec![high[0] - low[0]];
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        // EMA of TR
        let atr_ema = Self::ema(&tr, self.period);

        // SMA of close
        let sma = Self::sma(close, self.period);

        // PGO = (Close - SMA) / EMA(ATR)
        (0..n)
            .map(|i| {
                let avg = sma[i];
                let atr = atr_ema[i];
                if avg.is_nan() || atr.is_nan() || atr == 0.0 {
                    f64::NAN
                } else {
                    (close[i] - avg) / atr
                }
            })
            .collect()
    }
}

impl Default for PrettyGoodOscillator {
    fn default() -> Self {
        Self::new(14)
    }
}

impl TechnicalIndicator for PrettyGoodOscillator {
    fn name(&self) -> &str {
        "PGO"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }
}

impl SignalIndicator for PrettyGoodOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Standard thresholds: +3/-3
        if last > 3.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < -3.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        let signals = values.iter().map(|&val| {
            if val.is_nan() {
                IndicatorSignal::Neutral
            } else if val > 3.0 {
                IndicatorSignal::Bullish
            } else if val < -3.0 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect();
        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pgo_basic() {
        let pgo = PrettyGoodOscillator::new(14);
        let n = 50;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();

        let result = pgo.calculate(&high, &low, &close);
        assert_eq!(result.len(), n);
    }
}
