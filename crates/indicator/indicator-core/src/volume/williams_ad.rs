//! Williams Accumulation/Distribution (Williams AD) implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Williams Accumulation/Distribution.
///
/// Developed by Larry Williams, this indicator measures accumulation
/// when price closes higher than the previous close and distribution
/// when it closes lower.
///
/// True Range High (TRH) = max(Previous Close, Current High)
/// True Range Low (TRL) = min(Previous Close, Current Low)
///
/// If Close > Previous Close: AD = Close - TRL
/// If Close < Previous Close: AD = Close - TRH
/// If Close = Previous Close: AD = 0
///
/// Williams AD = Cumulative sum of AD values
#[derive(Debug, Clone)]
pub struct WilliamsAD;

impl WilliamsAD {
    pub fn new() -> Self {
        Self
    }

    /// Calculate Williams AD values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);
        result.push(0.0); // First bar has no previous close

        for i in 1..n {
            let prev_close = close[i - 1];
            let trh = high[i].max(prev_close);
            let trl = low[i].min(prev_close);

            let ad = if close[i] > prev_close {
                close[i] - trl
            } else if close[i] < prev_close {
                close[i] - trh
            } else {
                0.0
            };

            result.push(result[i - 1] + ad);
        }

        result
    }
}

impl Default for WilliamsAD {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for WilliamsAD {
    fn name(&self) -> &str {
        "Williams A/D"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for WilliamsAD {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close);

        if values.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let current = values[values.len() - 1];
        let previous = values[values.len() - 2];

        if current > previous {
            Ok(IndicatorSignal::Bullish)
        } else if current < previous {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close);

        let mut signals = vec![IndicatorSignal::Neutral];
        for i in 1..values.len() {
            if values[i] > values[i - 1] {
                signals.push(IndicatorSignal::Bullish);
            } else if values[i] < values[i - 1] {
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
    fn test_williams_ad_uptrend() {
        let wad = WilliamsAD::new();
        // Uptrend
        let high = vec![105.0, 107.0, 109.0, 111.0, 113.0];
        let low = vec![95.0, 97.0, 99.0, 101.0, 103.0];
        let close = vec![100.0, 104.0, 106.0, 108.0, 110.0];

        let result = wad.calculate(&high, &low, &close);

        assert_eq!(result.len(), 5);
        // Should be accumulating
        assert!(result[4] > result[1]);
    }

    #[test]
    fn test_williams_ad_downtrend() {
        let wad = WilliamsAD::new();
        // Downtrend
        let high = vec![113.0, 111.0, 109.0, 107.0, 105.0];
        let low = vec![103.0, 101.0, 99.0, 97.0, 95.0];
        let close = vec![110.0, 106.0, 104.0, 102.0, 100.0];

        let result = wad.calculate(&high, &low, &close);

        // Should be distributing (negative)
        assert!(result[4] < result[1]);
    }

    #[test]
    fn test_williams_ad_calculation() {
        let wad = WilliamsAD::new();
        let high = vec![105.0, 107.0];
        let low = vec![95.0, 97.0];
        let close = vec![100.0, 104.0]; // Close up

        let result = wad.calculate(&high, &low, &close);

        // TRL = min(100, 97) = 97
        // AD = 104 - 97 = 7
        assert!((result[1] - 7.0).abs() < 1e-10);
    }
}
