//! Alligator indicator - Bill Williams.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Alligator Output containing Jaw, Teeth, and Lips.
#[derive(Debug, Clone)]
pub struct AlligatorOutput {
    pub jaw: Vec<f64>,
    pub teeth: Vec<f64>,
    pub lips: Vec<f64>,
}

/// Alligator - IND-010
///
/// Bill Williams indicator using 3 smoothed moving averages.
/// Jaw (Blue): 13-period SMMA, offset 8
/// Teeth (Red): 8-period SMMA, offset 5
/// Lips (Green): 5-period SMMA, offset 3
#[derive(Debug, Clone)]
pub struct Alligator {
    jaw_period: usize,
    jaw_offset: usize,
    teeth_period: usize,
    teeth_offset: usize,
    lips_period: usize,
    lips_offset: usize,
}

impl Alligator {
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

    pub fn with_params(
        jaw_period: usize, jaw_offset: usize,
        teeth_period: usize, teeth_offset: usize,
        lips_period: usize, lips_offset: usize
    ) -> Self {
        Self {
            jaw_period, jaw_offset,
            teeth_period, teeth_offset,
            lips_period, lips_offset,
        }
    }

    fn smma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];

        // First value is SMA
        let initial: f64 = data[..period].iter().sum::<f64>() / period as f64;
        result.push(initial);

        // SMMA = (Previous SMMA * (Period - 1) + Current) / Period
        let mut prev = initial;
        for i in period..n {
            let smma = (prev * (period - 1) as f64 + data[i]) / period as f64;
            result.push(smma);
            prev = smma;
        }

        result
    }

    fn offset_forward(data: &[f64], offset: usize) -> Vec<f64> {
        // Shift data forward by offset (insert NaNs at beginning)
        let mut result = vec![f64::NAN; offset];
        result.extend(&data[..data.len().saturating_sub(offset)]);
        while result.len() < data.len() {
            result.push(f64::NAN);
        }
        result.truncate(data.len());
        result
    }

    pub fn calculate(&self, high: &[f64], low: &[f64]) -> AlligatorOutput {
        let n = high.len();
        if n < self.jaw_period + self.jaw_offset {
            return AlligatorOutput {
                jaw: vec![f64::NAN; n],
                teeth: vec![f64::NAN; n],
                lips: vec![f64::NAN; n],
            };
        }

        // Calculate median price (HL2)
        let median: Vec<f64> = high.iter()
            .zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        // Calculate SMAs
        let jaw_raw = Self::smma(&median, self.jaw_period);
        let teeth_raw = Self::smma(&median, self.teeth_period);
        let lips_raw = Self::smma(&median, self.lips_period);

        // Apply offsets
        let jaw = Self::offset_forward(&jaw_raw, self.jaw_offset);
        let teeth = Self::offset_forward(&teeth_raw, self.teeth_offset);
        let lips = Self::offset_forward(&lips_raw, self.lips_offset);

        AlligatorOutput { jaw, teeth, lips }
    }
}

impl Default for Alligator {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for Alligator {
    fn name(&self) -> &str {
        "Alligator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.jaw_period + self.jaw_offset {
            return Err(IndicatorError::InsufficientData {
                required: self.jaw_period + self.jaw_offset,
                got: data.high.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::triple(result.jaw, result.teeth, result.lips))
    }

    fn min_periods(&self) -> usize {
        self.jaw_period + self.jaw_offset
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for Alligator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low);

        let jaw = result.jaw.last().copied().unwrap_or(f64::NAN);
        let teeth = result.teeth.last().copied().unwrap_or(f64::NAN);
        let lips = result.lips.last().copied().unwrap_or(f64::NAN);

        if jaw.is_nan() || teeth.is_nan() || lips.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish: Lips > Teeth > Jaw
        if lips > teeth && teeth > jaw {
            Ok(IndicatorSignal::Bullish)
        } else if lips < teeth && teeth < jaw {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low);
        let n = result.jaw.len();
        let mut signals = Vec::with_capacity(n);

        for i in 0..n {
            let jaw = result.jaw[i];
            let teeth = result.teeth[i];
            let lips = result.lips[i];

            if jaw.is_nan() || teeth.is_nan() || lips.is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if lips > teeth && teeth > jaw {
                signals.push(IndicatorSignal::Bullish);
            } else if lips < teeth && teeth < jaw {
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
    fn test_alligator_basic() {
        let alligator = Alligator::new();
        let n = 50;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();

        let result = alligator.calculate(&high, &low);

        assert_eq!(result.jaw.len(), n);
        assert_eq!(result.teeth.len(), n);
        assert_eq!(result.lips.len(), n);
    }
}
