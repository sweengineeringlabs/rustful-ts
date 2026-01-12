//! Stochastic Momentum Index (SMI).

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Stochastic Momentum Index (SMI) - IND-040
///
/// Shows where the close is relative to the midpoint of the high/low range.
/// SMI = ((Close - Midpoint) / (Range / 2)) * 100
#[derive(Debug, Clone)]
pub struct SMI {
    k_period: usize,
    d_period: usize,
    smooth_period: usize,
    overbought: f64,
    oversold: f64,
}

impl SMI {
    pub fn new(k_period: usize, d_period: usize, smooth_period: usize) -> Self {
        Self {
            k_period,
            d_period,
            smooth_period,
            overbought: 40.0,
            oversold: -40.0,
        }
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

    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        if n < self.k_period {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate highest high and lowest low over k_period
        let mut hh = vec![f64::NAN; self.k_period - 1];
        let mut ll = vec![f64::NAN; self.k_period - 1];

        for i in (self.k_period - 1)..n {
            let start_idx = i + 1 - self.k_period;
            let window_high = &high[start_idx..=i];
            let window_low = &low[start_idx..=i];

            hh.push(window_high.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
            ll.push(window_low.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        }

        // Calculate raw SMI components
        let mut rel_close = vec![f64::NAN; n];
        let mut range_val = vec![f64::NAN; n];

        for i in (self.k_period - 1)..n {
            let midpoint = (hh[i] + ll[i]) / 2.0;
            rel_close[i] = close[i] - midpoint;
            range_val[i] = hh[i] - ll[i];
        }

        // Double smooth
        let rel_smooth1 = Self::ema(&rel_close, self.smooth_period);
        let rel_smooth2 = Self::ema(&rel_smooth1, self.d_period);

        let range_smooth1 = Self::ema(&range_val, self.smooth_period);
        let range_smooth2 = Self::ema(&range_smooth1, self.d_period);

        // Calculate SMI
        let smi: Vec<f64> = rel_smooth2.iter()
            .zip(range_smooth2.iter())
            .map(|(rel, rng)| {
                if rel.is_nan() || rng.is_nan() || *rng == 0.0 {
                    f64::NAN
                } else {
                    // Clamp to [-100, 100] to handle floating point precision issues
                    ((rel / (rng / 2.0)) * 100.0).clamp(-100.0, 100.0)
                }
            })
            .collect();

        // Signal line
        let signal = Self::ema(&smi, self.smooth_period);

        (smi, signal)
    }
}

impl Default for SMI {
    fn default() -> Self {
        Self::new(13, 25, 2)
    }
}

impl TechnicalIndicator for SMI {
    fn name(&self) -> &str {
        "SMI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.k_period {
            return Err(IndicatorError::InsufficientData {
                required: self.k_period,
                got: data.close.len(),
            });
        }

        let (smi, signal) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(smi, signal))
    }

    fn min_periods(&self) -> usize {
        self.k_period
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for SMI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (smi, _) = self.calculate(&data.high, &data.low, &data.close);
        let last = smi.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if last >= self.overbought {
            Ok(IndicatorSignal::Bearish)
        } else if last <= self.oversold {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (smi, _) = self.calculate(&data.high, &data.low, &data.close);
        let signals = smi.iter().map(|&val| {
            if val.is_nan() {
                IndicatorSignal::Neutral
            } else if val >= self.overbought {
                IndicatorSignal::Bearish
            } else if val <= self.oversold {
                IndicatorSignal::Bullish
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
    fn test_smi_basic() {
        let smi = SMI::default();
        let n = 50;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 105.0 + i as f64).collect();

        let (smi_line, signal_line) = smi.calculate(&high, &low, &close);

        assert_eq!(smi_line.len(), n);
        assert_eq!(signal_line.len(), n);
    }
}
