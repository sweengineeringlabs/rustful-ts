//! Qstick indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Qstick - IND-066
///
/// Moving average of (Close - Open).
/// Measures buying vs selling pressure.
#[derive(Debug, Clone)]
pub struct Qstick {
    period: usize,
}

impl Qstick {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn calculate(&self, open: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period {
            return vec![f64::NAN; n];
        }

        // Calculate (Close - Open)
        let co: Vec<f64> = close.iter()
            .zip(open.iter())
            .map(|(c, o)| c - o)
            .collect();

        // SMA of (Close - Open)
        let mut result = vec![f64::NAN; self.period - 1];
        let mut sum: f64 = co[..self.period].iter().sum();
        result.push(sum / self.period as f64);

        for i in self.period..n {
            sum = sum - co[i - self.period] + co[i];
            result.push(sum / self.period as f64);
        }

        result
    }
}

impl Default for Qstick {
    fn default() -> Self {
        Self::new(8)
    }
}

impl TechnicalIndicator for Qstick {
    fn name(&self) -> &str {
        "Qstick"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.open, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

impl SignalIndicator for Qstick {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.open, &data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if last > 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.open, &data.close);
        let signals = values.iter().map(|&val| {
            if val.is_nan() {
                IndicatorSignal::Neutral
            } else if val > 0.0 {
                IndicatorSignal::Bullish
            } else if val < 0.0 {
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
    fn test_qstick_bullish() {
        let qstick = Qstick::new(5);
        // Close always higher than open = bullish
        let open = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let close = vec![102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let result = qstick.calculate(&open, &close);

        let last = result.last().unwrap();
        assert!(*last > 0.0);
    }
}
