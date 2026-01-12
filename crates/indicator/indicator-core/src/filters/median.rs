//! Median Filter implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::MedianConfig;

/// Median Filter.
///
/// Removes spikes while preserving trends.
/// For each window, outputs the median value.
#[derive(Debug, Clone)]
pub struct MedianFilter {
    period: usize,
}

impl MedianFilter {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: MedianConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate median filtered values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < self.period {
            return result;
        }

        let mut window: Vec<f64> = Vec::with_capacity(self.period);

        for i in (self.period - 1)..n {
            window.clear();
            window.extend_from_slice(&data[(i + 1 - self.period)..=i]);
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            result[i] = if self.period % 2 == 1 {
                window[self.period / 2]
            } else {
                (window[self.period / 2 - 1] + window[self.period / 2]) / 2.0
            };
        }

        result
    }
}

impl Default for MedianFilter {
    fn default() -> Self {
        Self::from_config(MedianConfig::default())
    }
}

impl TechnicalIndicator for MedianFilter {
    fn name(&self) -> &str {
        "Median"
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
    fn test_median() {
        let median = MedianFilter::new(3);
        // Data with a spike: [1, 2, 100, 4, 5]
        let data = vec![1.0, 2.0, 100.0, 4.0, 5.0];
        let result = median.calculate(&data);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Median of [1, 2, 100] = 2
        assert!((result[2] - 2.0).abs() < 1e-10);
        // Median of [2, 100, 4] = 4
        assert!((result[3] - 4.0).abs() < 1e-10);
        // Median of [100, 4, 5] = 5
        assert!((result[4] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even_period() {
        let median = MedianFilter::new(4);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = median.calculate(&data);

        // Median of [1, 2, 3, 4] = (2 + 3) / 2 = 2.5
        assert!((result[3] - 2.5).abs() < 1e-10);
    }
}
