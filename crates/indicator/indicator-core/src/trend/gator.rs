//! Gator Oscillator - Bill Williams.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Gator Oscillator - IND-009
///
/// Bill Williams Gator Oscillator from Alligator.
/// Upper histogram: |Jaw - Teeth|
/// Lower histogram: -|Teeth - Lips|
#[derive(Debug, Clone)]
pub struct GatorOscillator {
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
}

impl GatorOscillator {
    pub fn new() -> Self {
        Self {
            jaw_period: 13,
            jaw_offset: 8,
            teeth_period: 8,
            teeth_offset: 5,
            lips_period: 5,
            lips_offset: 3,
        }
    }

    fn smma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];
        let initial: f64 = data[..period].iter().sum::<f64>() / period as f64;
        result.push(initial);

        let mut prev = initial;
        for i in period..n {
            let smma = (prev * (period - 1) as f64 + data[i]) / period as f64;
            result.push(smma);
            prev = smma;
        }

        result
    }

    fn offset_forward(data: &[f64], offset: usize) -> Vec<f64> {
        let mut result = vec![f64::NAN; offset];
        result.extend(&data[..data.len().saturating_sub(offset)]);
        while result.len() < data.len() {
            result.push(f64::NAN);
        }
        result.truncate(data.len());
        result
    }

    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n < self.jaw_period + self.jaw_offset {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let median: Vec<f64> = high.iter()
            .zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        let jaw_raw = Self::smma(&median, self.jaw_period);
        let teeth_raw = Self::smma(&median, self.teeth_period);
        let lips_raw = Self::smma(&median, self.lips_period);

        let jaw = Self::offset_forward(&jaw_raw, self.jaw_offset);
        let teeth = Self::offset_forward(&teeth_raw, self.teeth_offset);
        let lips = Self::offset_forward(&lips_raw, self.lips_offset);

        // Calculate Gator histograms
        let upper: Vec<f64> = jaw.iter()
            .zip(teeth.iter())
            .map(|(j, t)| {
                if j.is_nan() || t.is_nan() {
                    f64::NAN
                } else {
                    (j - t).abs()
                }
            })
            .collect();

        let lower: Vec<f64> = teeth.iter()
            .zip(lips.iter())
            .map(|(t, l)| {
                if t.is_nan() || l.is_nan() {
                    f64::NAN
                } else {
                    -(t - l).abs()
                }
            })
            .collect();

        (upper, lower)
    }
}

impl Default for GatorOscillator {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for GatorOscillator {
    fn name(&self) -> &str {
        "Gator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.jaw_period + self.jaw_offset {
            return Err(IndicatorError::InsufficientData {
                required: self.jaw_period + self.jaw_offset,
                got: data.high.len(),
            });
        }

        let (upper, lower) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.jaw_period + self.jaw_offset
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for GatorOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (upper, lower) = self.calculate(&data.high, &data.low);

        if upper.len() < 2 || lower.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let upper_last = upper[upper.len() - 1];
        let upper_prev = upper[upper.len() - 2];
        let lower_last = lower[lower.len() - 1];
        let lower_prev = lower[lower.len() - 2];

        if upper_last.is_nan() || lower_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Expanding bars = awakening, converging = sleeping
        let upper_expanding = upper_last > upper_prev;
        let lower_expanding = lower_last.abs() > lower_prev.abs();

        if upper_expanding && lower_expanding {
            // Alligator eating (trending)
            if upper_last > 0.0 {
                Ok(IndicatorSignal::Bullish)
            } else {
                Ok(IndicatorSignal::Bearish)
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (upper, lower) = self.calculate(&data.high, &data.low);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..upper.len().min(lower.len()) {
            if upper[i].is_nan() || lower[i].is_nan() || upper[i-1].is_nan() || lower[i-1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else {
                let upper_expanding = upper[i] > upper[i-1];
                let lower_expanding = lower[i].abs() > lower[i-1].abs();

                if upper_expanding && lower_expanding {
                    signals.push(IndicatorSignal::Bullish);
                } else {
                    signals.push(IndicatorSignal::Neutral);
                }
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gator_basic() {
        let gator = GatorOscillator::new();
        let n = 50;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();

        let (upper, lower) = gator.calculate(&high, &low);

        assert_eq!(upper.len(), n);
        assert_eq!(lower.len(), n);

        // Upper should be positive
        for val in upper.iter() {
            if !val.is_nan() {
                assert!(*val >= 0.0);
            }
        }

        // Lower should be negative
        for val in lower.iter() {
            if !val.is_nan() {
                assert!(*val <= 0.0);
            }
        }
    }
}
