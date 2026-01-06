//! Volume Weighted Average Price (VWAP) implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::VWAPConfig;

/// Volume Weighted Average Price.
///
/// VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
/// Typical Price = (High + Low + Close) / 3
#[derive(Debug, Clone)]
pub struct VWAP {
    #[allow(dead_code)]
    reset_daily: bool,
}

impl VWAP {
    pub fn new() -> Self {
        Self { reset_daily: false }
    }

    pub fn from_config(config: VWAPConfig) -> Self {
        Self {
            reset_daily: config.reset_daily,
        }
    }

    /// Calculate VWAP values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = Vec::with_capacity(n);

        let mut cum_tp_vol = 0.0;
        let mut cum_vol = 0.0;

        for i in 0..n {
            let typical_price = (high[i] + low[i] + close[i]) / 3.0;
            cum_tp_vol += typical_price * volume[i];
            cum_vol += volume[i];

            result.push(if cum_vol > 0.0 {
                cum_tp_vol / cum_vol
            } else {
                typical_price
            });
        }

        result
    }
}

impl Default for VWAP {
    fn default() -> Self {
        Self::from_config(VWAPConfig::default())
    }
}

impl TechnicalIndicator for VWAP {
    fn name(&self) -> &str {
        "VWAP"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap() {
        let vwap = VWAP::new();
        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![95.0, 96.0, 97.0, 98.0, 99.0];
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let volume = vec![1000.0, 1500.0, 1200.0, 1800.0, 1400.0];

        let result = vwap.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 5);
        // First VWAP equals first typical price
        let tp0 = (105.0 + 95.0 + 100.0) / 3.0;
        assert!((result[0] - tp0).abs() < 1e-10);
        // VWAP should be around the typical price level
        assert!(result[4] > 95.0 && result[4] < 110.0);
    }
}
