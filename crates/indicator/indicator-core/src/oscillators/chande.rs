//! Chande Momentum Oscillator (CMO).

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Chande Momentum Oscillator (CMO) - IND-013
///
/// Measures momentum on a scale of -100 to +100.
/// CMO = ((Sum of up changes - Sum of down changes) / (Sum of up + Sum of down)) * 100
#[derive(Debug, Clone)]
pub struct ChandeMomentum {
    period: usize,
    overbought: f64,
    oversold: f64,
}

impl ChandeMomentum {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: 50.0,
            oversold: -50.0,
        }
    }

    pub fn with_thresholds(period: usize, overbought: f64, oversold: f64) -> Self {
        Self { period, overbought, oversold }
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period];

        for i in self.period..n {
            let mut sum_up = 0.0;
            let mut sum_down = 0.0;

            for j in (i - self.period + 1)..=i {
                let change = data[j] - data[j - 1];
                if change > 0.0 {
                    sum_up += change;
                } else if change < 0.0 {
                    sum_down += -change;
                }
            }

            let total = sum_up + sum_down;
            if total == 0.0 {
                result.push(0.0);
            } else {
                result.push(((sum_up - sum_down) / total) * 100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for ChandeMomentum {
    fn name(&self) -> &str {
        "CMO"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }
}

impl SignalIndicator for ChandeMomentum {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
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
        let values = self.calculate(&data.close);
        let signals = values.iter().map(|&cmo| {
            if cmo.is_nan() {
                IndicatorSignal::Neutral
            } else if cmo >= self.overbought {
                IndicatorSignal::Bearish
            } else if cmo <= self.oversold {
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
    fn test_cmo_basic() {
        let cmo = ChandeMomentum::new(5);
        // Strong uptrend
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();
        let result = cmo.calculate(&data);

        // Should be close to +100 in strong uptrend
        let last = result.last().unwrap();
        assert!(*last > 90.0);
    }

    #[test]
    fn test_cmo_range() {
        let cmo = ChandeMomentum::new(14);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let result = cmo.calculate(&data);

        // CMO should be in range [-100, 100]
        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= -100.0 && *val <= 100.0);
            }
        }
    }
}
