//! Rolling Median implementation.
//!
//! Calculates the rolling median (middle value) over a specified period.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Rolling Median - Middle value over a period.
///
/// The median is the middle value when data is sorted. For even-length
/// windows, it's the average of the two middle values.
///
/// More robust to outliers than the mean.
#[derive(Debug, Clone)]
pub struct Median {
    period: usize,
}

impl Median {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling median.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let mut window: Vec<f64> = data[start..=i].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = if self.period % 2 == 1 {
                window[self.period / 2]
            } else {
                (window[self.period / 2 - 1] + window[self.period / 2]) / 2.0
            };

            result.push(median);
        }

        result
    }
}

impl TechnicalIndicator for Median {
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

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
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
    fn test_median_odd_period() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        let median = Median::new(3);
        let result = median.calculate(&data);

        assert_eq!(result.len(), 9);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 3.0).abs() < 0.001); // sorted [1,3,4] -> 3
        assert!((result[3] - 1.0).abs() < 0.001); // sorted [1,1,4] -> 1
        assert!((result[4] - 4.0).abs() < 0.001); // sorted [1,4,5] -> 4
    }

    #[test]
    fn test_median_even_period() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let median = Median::new(4);
        let result = median.calculate(&data);

        assert_eq!(result.len(), 6);
        assert!(result[2].is_nan());
        assert!((result[3] - 2.5).abs() < 0.001); // sorted [1,2,3,4] -> (2+3)/2 = 2.5
        assert!((result[4] - 3.5).abs() < 0.001); // sorted [2,3,4,5] -> (3+4)/2 = 3.5
    }

    #[test]
    fn test_median_single_period() {
        let data = vec![5.0, 10.0, 15.0];
        let median = Median::new(1);
        let result = median.calculate(&data);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0).abs() < 0.001);
        assert!((result[1] - 10.0).abs() < 0.001);
        assert!((result[2] - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_median_with_outlier() {
        // Median is robust to outliers
        let data = vec![1.0, 2.0, 100.0, 4.0, 5.0]; // 100 is an outlier
        let median = Median::new(3);
        let result = median.calculate(&data);

        // Median of [1,2,100] = 2 (not affected by outlier as much as mean)
        assert!((result[2] - 2.0).abs() < 0.001);
        // Median of [2,100,4] = 4
        assert!((result[3] - 4.0).abs() < 0.001);
    }
}
