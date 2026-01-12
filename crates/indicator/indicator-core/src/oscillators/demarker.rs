//! DeMarker indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// DeMarker - IND-014
///
/// Exhaustion detection indicator comparing high/low ranges.
/// Oscillates between 0 and 1.
#[derive(Debug, Clone)]
pub struct DeMarker {
    period: usize,
    overbought: f64,
    oversold: f64,
}

impl DeMarker {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: 0.7,
            oversold: 0.3,
        }
    }

    pub fn with_thresholds(period: usize, overbought: f64, oversold: f64) -> Self {
        Self { period, overbought, oversold }
    }

    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        // Calculate DeMax and DeMin
        let mut de_max = vec![0.0; n];
        let mut de_min = vec![0.0; n];

        for i in 1..n {
            let high_diff = high[i] - high[i - 1];
            let low_diff = low[i - 1] - low[i];

            de_max[i] = if high_diff > 0.0 { high_diff } else { 0.0 };
            de_min[i] = if low_diff > 0.0 { low_diff } else { 0.0 };
        }

        let mut result = vec![f64::NAN; self.period];

        for i in self.period..n {
            let sum_max: f64 = de_max[(i - self.period + 1)..=i].iter().sum();
            let sum_min: f64 = de_min[(i - self.period + 1)..=i].iter().sum();
            let total = sum_max + sum_min;

            if total == 0.0 {
                result.push(0.5);
            } else {
                result.push(sum_max / total);
            }
        }

        result
    }
}

impl TechnicalIndicator for DeMarker {
    fn name(&self) -> &str {
        "DeMarker"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }
}

impl SignalIndicator for DeMarker {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low);
        let last = values.last().copied().unwrap_or(f64::NAN);

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
        let values = self.calculate(&data.high, &data.low);
        let signals = values.iter().map(|&dm| {
            if dm.is_nan() {
                IndicatorSignal::Neutral
            } else if dm >= self.overbought {
                IndicatorSignal::Bearish
            } else if dm <= self.oversold {
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
    fn test_demarker_range() {
        let dm = DeMarker::new(14);
        let high: Vec<f64> = (0..50).map(|i| 110.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..50).map(|i| 90.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let result = dm.calculate(&high, &low);

        // DeMarker should be in range [0, 1]
        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 1.0);
            }
        }
    }
}
