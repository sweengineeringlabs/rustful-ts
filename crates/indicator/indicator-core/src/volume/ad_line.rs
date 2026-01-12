//! Accumulation/Distribution Line (A/D Line) implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Accumulation/Distribution Line.
///
/// The A/D Line measures cumulative money flow by combining price and volume.
/// - Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
/// - Money Flow Volume = MFM * Volume
/// - A/D Line = Previous A/D + Current Money Flow Volume
///
/// Rising A/D with rising price confirms uptrend.
/// Falling A/D with falling price confirms downtrend.
#[derive(Debug, Clone)]
pub struct ADLine;

impl ADLine {
    pub fn new() -> Self {
        Self
    }

    /// Calculate A/D Line values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);

        // Calculate first Money Flow Volume
        let mfv_0 = self.money_flow_volume(high[0], low[0], close[0], volume[0]);
        result.push(mfv_0);

        // Cumulative sum
        for i in 1..n {
            let mfv = self.money_flow_volume(high[i], low[i], close[i], volume[i]);
            result.push(result[i - 1] + mfv);
        }

        result
    }

    /// Calculate Money Flow Volume for a single bar.
    fn money_flow_volume(&self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let hl_range = high - low;
        if hl_range > 0.0 {
            let mfm = ((close - low) - (high - close)) / hl_range;
            mfm * volume
        } else {
            0.0
        }
    }
}

impl Default for ADLine {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for ADLine {
    fn name(&self) -> &str {
        "A/D Line"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for ADLine {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);

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
        let values = self.calculate(&data.high, &data.low, &data.close, &data.volume);

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
    fn test_ad_line() {
        let ad = ADLine::new();
        // Bullish scenario: closes near highs
        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![95.0, 96.0, 97.0, 98.0, 99.0];
        let close = vec![104.0, 105.0, 106.0, 107.0, 108.0]; // Closes near highs
        let volume = vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0];

        let result = ad.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 5);
        // A/D should be rising (positive MFV with closes near highs)
        assert!(result[4] > result[0]);
    }

    #[test]
    fn test_ad_line_bearish() {
        let ad = ADLine::new();
        // Bearish scenario: closes near lows
        let high = vec![105.0, 104.0, 103.0, 102.0, 101.0];
        let low = vec![95.0, 94.0, 93.0, 92.0, 91.0];
        let close = vec![96.0, 95.0, 94.0, 93.0, 92.0]; // Closes near lows
        let volume = vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0];

        let result = ad.calculate(&high, &low, &close, &volume);

        // A/D should be falling (negative MFV with closes near lows)
        assert!(result[4] < result[0]);
    }
}
