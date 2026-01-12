//! Random Walk Index (RWI).

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Random Walk Index (RWI) - IND-186
///
/// Compares price movement to random walk.
/// High values indicate trending behavior.
#[derive(Debug, Clone)]
pub struct RandomWalkIndex {
    period: usize,
}

impl RandomWalkIndex {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n < self.period + 1 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate ATR for normalization
        let mut tr = vec![0.0; n];
        tr[0] = high[0] - low[0];

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        let mut rwi_high = vec![f64::NAN; self.period];
        let mut rwi_low = vec![f64::NAN; self.period];

        for i in self.period..n {
            let mut max_rwi_high = 0.0_f64;
            let mut max_rwi_low = 0.0_f64;

            for j in 2..=self.period {
                // Sum of ATR over j periods
                let atr_sum: f64 = tr[(i - j + 1)..=i].iter().sum();
                let avg_atr = atr_sum / j as f64;
                let denominator = avg_atr * (j as f64).sqrt();

                if denominator > 0.0 {
                    // RWI High: (High - Low[j periods ago]) / denominator
                    let rwi_h = (high[i] - low[i - j + 1]) / denominator;
                    max_rwi_high = max_rwi_high.max(rwi_h);

                    // RWI Low: (High[j periods ago] - Low) / denominator
                    let rwi_l = (high[i - j + 1] - low[i]) / denominator;
                    max_rwi_low = max_rwi_low.max(rwi_l);
                }
            }

            rwi_high.push(max_rwi_high);
            rwi_low.push(max_rwi_low);
        }

        (rwi_high, rwi_low)
    }
}

impl Default for RandomWalkIndex {
    fn default() -> Self {
        Self::new(14)
    }
}

impl TechnicalIndicator for RandomWalkIndex {
    fn name(&self) -> &str {
        "RWI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.high.len(),
            });
        }

        let (rwi_high, rwi_low) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(rwi_high, rwi_low))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for RandomWalkIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (rwi_high, rwi_low) = self.calculate(&data.high, &data.low, &data.close);

        let high_last = rwi_high.last().copied().unwrap_or(f64::NAN);
        let low_last = rwi_low.last().copied().unwrap_or(f64::NAN);

        if high_last.is_nan() || low_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Values > 1.0 suggest trending behavior
        if high_last > 1.0 && high_last > low_last {
            Ok(IndicatorSignal::Bullish)
        } else if low_last > 1.0 && low_last > high_last {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (rwi_high, rwi_low) = self.calculate(&data.high, &data.low, &data.close);
        let n = rwi_high.len().min(rwi_low.len());
        let mut signals = Vec::with_capacity(n);

        for i in 0..n {
            if rwi_high[i].is_nan() || rwi_low[i].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if rwi_high[i] > 1.0 && rwi_high[i] > rwi_low[i] {
                signals.push(IndicatorSignal::Bullish);
            } else if rwi_low[i] > 1.0 && rwi_low[i] > rwi_high[i] {
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
    fn test_rwi_basic() {
        let rwi = RandomWalkIndex::new(14);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let (rwi_high, rwi_low) = rwi.calculate(&high, &low, &close);

        assert_eq!(rwi_high.len(), n);
        assert_eq!(rwi_low.len(), n);
    }
}
