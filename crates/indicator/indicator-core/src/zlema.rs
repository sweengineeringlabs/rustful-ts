//! Zero-Lag Exponential Moving Average (ZLEMA) implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::ZLEMAConfig;

/// Zero-Lag Exponential Moving Average.
///
/// ZLEMA reduces lag by adding a momentum term.
/// Adjusted Price = 2 * Price - Price(lag), where lag = (period - 1) / 2
/// ZLEMA = EMA of Adjusted Price
#[derive(Debug, Clone)]
pub struct ZLEMA {
    period: usize,
}

impl ZLEMA {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: ZLEMAConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate ZLEMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < self.period {
            return result;
        }

        let lag = (self.period - 1) / 2;
        let alpha = 2.0 / (self.period as f64 + 1.0);

        // Initialize with adjusted price
        let start_idx = self.period.saturating_sub(1);
        if start_idx < n {
            let lag_idx = if start_idx >= lag { start_idx - lag } else { 0 };
            let adjusted = 2.0 * data[start_idx] - data[lag_idx];
            result[start_idx] = adjusted;
        }

        // Calculate ZLEMA
        for i in self.period..n {
            let lag_idx = if i >= lag { i - lag } else { 0 };
            let adjusted = 2.0 * data[i] - data[lag_idx];
            result[i] = alpha * adjusted + (1.0 - alpha) * result[i - 1];
        }

        result
    }
}

impl Default for ZLEMA {
    fn default() -> Self {
        Self::from_config(ZLEMAConfig::default())
    }
}

impl TechnicalIndicator for ZLEMA {
    fn name(&self) -> &str {
        "ZLEMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zlema() {
        let zlema = ZLEMA::new(10);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = zlema.calculate(&data);

        // Should have NaN initially
        assert!(result[0].is_nan());
        // After period, should have values
        assert!(!result[29].is_nan());
        // ZLEMA should closely track the uptrend
        assert!(result[29] > 120.0);
    }
}
