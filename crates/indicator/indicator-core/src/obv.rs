//! On-Balance Volume (OBV) implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::OBVConfig;

/// On-Balance Volume.
///
/// OBV uses volume flow to predict changes in stock price.
/// - If close > previous close: OBV = previous OBV + volume
/// - If close < previous close: OBV = previous OBV - volume
/// - If close = previous close: OBV = previous OBV
#[derive(Debug, Clone)]
pub struct OBV {
    #[allow(dead_code)]
    signal_period: Option<usize>,
}

impl OBV {
    pub fn new() -> Self {
        Self { signal_period: None }
    }

    pub fn from_config(config: OBVConfig) -> Self {
        Self {
            signal_period: config.signal_period,
        }
    }

    /// Calculate OBV values.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);
        result.push(volume[0]);

        for i in 1..n {
            if close[i] > close[i - 1] {
                result.push(result[i - 1] + volume[i]);
            } else if close[i] < close[i - 1] {
                result.push(result[i - 1] - volume[i]);
            } else {
                result.push(result[i - 1]);
            }
        }

        result
    }
}

impl Default for OBV {
    fn default() -> Self {
        Self::from_config(OBVConfig::default())
    }
}

impl TechnicalIndicator for OBV {
    fn name(&self) -> &str {
        "OBV"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obv() {
        let obv = OBV::new();
        // Uptrend: each close higher than previous
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 1400.0];

        let result = obv.calculate(&close, &volume);

        assert_eq!(result.len(), 5);
        // First OBV equals first volume
        assert!((result[0] - 1000.0).abs() < 1e-10);
        // OBV should increase in uptrend
        assert!(result[4] > result[0]);
        // Expected: 1000 + 1500 + 1200 + 1800 + 1400 = 6900
        assert!((result[4] - 6900.0).abs() < 1e-10);
    }

    #[test]
    fn test_obv_mixed() {
        let obv = OBV::new();
        let close = vec![100.0, 101.0, 100.0, 99.0, 100.0];
        let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 1400.0];

        let result = obv.calculate(&close, &volume);

        // 1000 + 1500 (up) - 1200 (down) - 1800 (down) + 1400 (up) = 900
        assert!((result[4] - 900.0).abs() < 1e-10);
    }
}
