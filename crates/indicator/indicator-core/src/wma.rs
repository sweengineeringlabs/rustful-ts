//! Weighted Moving Average (WMA) implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::WMAConfig;

/// Weighted Moving Average.
///
/// Assigns linearly increasing weights to more recent data points.
/// Weight for position i (0-indexed from oldest): i + 1
/// WMA = Σ(price_i * weight_i) / Σ(weight_i)
#[derive(Debug, Clone)]
pub struct WMA {
    period: usize,
}

impl WMA {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: WMAConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate WMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < self.period {
            return result;
        }

        // Weight sum: 1 + 2 + ... + period = period * (period + 1) / 2
        let weight_sum = (self.period * (self.period + 1) / 2) as f64;

        for i in (self.period - 1)..n {
            let start_idx = i + 1 - self.period;
            let mut weighted_sum = 0.0;
            for j in 0..self.period {
                let weight = (j + 1) as f64;
                weighted_sum += data[start_idx + j] * weight;
            }
            result[i] = weighted_sum / weight_sum;
        }

        result
    }
}

impl Default for WMA {
    fn default() -> Self {
        Self::from_config(WMAConfig::default())
    }
}

impl TechnicalIndicator for WMA {
    fn name(&self) -> &str {
        "WMA"
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
    fn test_wma() {
        let wma = WMA::new(3);
        // Data: [1, 2, 3, 4, 5]
        // WMA at i=2: (1*1 + 2*2 + 3*3) / 6 = 14/6 = 2.333...
        // WMA at i=3: (2*1 + 3*2 + 4*3) / 6 = 20/6 = 3.333...
        // WMA at i=4: (3*1 + 4*2 + 5*3) / 6 = 26/6 = 4.333...
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = wma.calculate(&data);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 14.0/6.0).abs() < 1e-10);
        assert!((result[3] - 20.0/6.0).abs() < 1e-10);
        assert!((result[4] - 26.0/6.0).abs() < 1e-10);
    }
}
