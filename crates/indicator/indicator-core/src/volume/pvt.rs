//! Price Volume Trend (PVT) implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Price Volume Trend.
///
/// PVT is a cumulative indicator that relates volume to price change.
/// Similar to OBV but accounts for the magnitude of price change.
///
/// PVT = Previous PVT + (Volume * (Close - Previous Close) / Previous Close)
///
/// - Rising PVT: Positive volume flow (accumulation)
/// - Falling PVT: Negative volume flow (distribution)
#[derive(Debug, Clone)]
pub struct PVT;

impl PVT {
    pub fn new() -> Self {
        Self
    }

    /// Calculate PVT values.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);
        result.push(0.0); // Start at 0

        for i in 1..n {
            if close[i - 1] > 0.0 {
                let price_change_pct = (close[i] - close[i - 1]) / close[i - 1];
                result.push(result[i - 1] + volume[i] * price_change_pct);
            } else {
                result.push(result[i - 1]);
            }
        }

        result
    }
}

impl Default for PVT {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for PVT {
    fn name(&self) -> &str {
        "PVT"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let values = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for PVT {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close, &data.volume);

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
        let values = self.calculate(&data.close, &data.volume);

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
    fn test_pvt_uptrend() {
        let pvt = PVT::new();
        // Uptrend with consistent price increases
        let close = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let volume = vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0];

        let result = pvt.calculate(&close, &volume);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-10);
        // PVT should be rising in uptrend
        assert!(result[4] > result[0]);
    }

    #[test]
    fn test_pvt_downtrend() {
        let pvt = PVT::new();
        // Downtrend
        let close = vec![100.0, 98.0, 96.0, 94.0, 92.0];
        let volume = vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0];

        let result = pvt.calculate(&close, &volume);

        // PVT should be falling in downtrend
        assert!(result[4] < result[0]);
    }

    #[test]
    fn test_pvt_calculation() {
        let pvt = PVT::new();
        let close = vec![100.0, 102.0];
        let volume = vec![1000.0, 1500.0];

        let result = pvt.calculate(&close, &volume);

        // PVT[1] = 0 + 1500 * (102-100)/100 = 30
        assert!((result[1] - 30.0).abs() < 1e-10);
    }
}
