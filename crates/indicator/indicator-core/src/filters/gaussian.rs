//! Gaussian Filter implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::GaussianConfig;

/// Gaussian Smoothing Filter.
///
/// Applies a Gaussian kernel for smooth filtering.
/// Weight at distance x: exp(-x² / (2σ²))
#[derive(Debug, Clone)]
pub struct GaussianFilter {
    period: usize,
    sigma: f64,
}

impl GaussianFilter {
    pub fn new(period: usize, sigma: f64) -> Self {
        Self { period, sigma }
    }

    pub fn from_config(config: GaussianConfig) -> Self {
        Self {
            period: config.period,
            sigma: config.sigma,
        }
    }

    /// Calculate Gaussian filtered values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < self.period {
            return result;
        }

        // Pre-calculate Gaussian weights
        let half = self.period / 2;
        let mut weights: Vec<f64> = Vec::with_capacity(self.period);
        let mut weight_sum = 0.0;

        for i in 0..self.period {
            let x = i as f64 - half as f64;
            let w = (-x * x / (2.0 * self.sigma * self.sigma)).exp();
            weights.push(w);
            weight_sum += w;
        }

        // Normalize weights
        for w in weights.iter_mut() {
            *w /= weight_sum;
        }

        // Apply Gaussian filter
        for i in half..(n.saturating_sub(half)) {
            let mut sum = 0.0;
            for (j, &w) in weights.iter().enumerate() {
                let idx = i + j - half;
                if idx < n {
                    sum += data[idx] * w;
                }
            }
            result[i] = sum;
        }

        result
    }
}

impl Default for GaussianFilter {
    fn default() -> Self {
        Self::from_config(GaussianConfig::default())
    }
}

impl TechnicalIndicator for GaussianFilter {
    fn name(&self) -> &str {
        "Gaussian"
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
    fn test_gaussian() {
        let gaussian = GaussianFilter::new(5, 1.0);
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let result = gaussian.calculate(&data);

        // Should have NaN at edges
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Middle values should be smoothed
        assert!(!result[10].is_nan());
        // Should roughly track the linear trend
        assert!(result[10] > 105.0 && result[10] < 115.0);
    }
}
