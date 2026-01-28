//! Final Filter Indicators
//!
//! Additional filtering indicators to complete the 350-indicator milestone.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Recursive Filter - Applies recursive filtering to price data
#[derive(Debug, Clone)]
pub struct RecursiveFilter {
    alpha: f64,
    beta: f64,
}

impl RecursiveFilter {
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be between 0 and 1 exclusive".to_string(),
            });
        }
        if beta < 0.0 || beta >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "beta".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { alpha, beta })
    }

    /// Apply recursive filter: y[i] = alpha * x[i] + beta * y[i-1] + (1-alpha-beta) * y[i-2]
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n == 0 {
            return result;
        }

        result[0] = close[0];
        if n > 1 {
            result[1] = self.alpha * close[1] + (1.0 - self.alpha) * result[0];
        }

        let gamma = 1.0 - self.alpha - self.beta;
        for i in 2..n {
            result[i] = self.alpha * close[i] + self.beta * result[i - 1] + gamma * result[i - 2];
        }

        result
    }
}

impl TechnicalIndicator for RecursiveFilter {
    fn name(&self) -> &str {
        "Recursive Filter"
    }

    fn min_periods(&self) -> usize {
        3
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Normalized Price Filter - Filters and normalizes price to 0-100 range
#[derive(Debug, Clone)]
pub struct NormalizedPriceFilter {
    period: usize,
    smooth_period: usize,
}

impl NormalizedPriceFilter {
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, smooth_period })
    }

    /// Calculate normalized and filtered price (0-100)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut normalized = vec![0.0; n];
        let mut result = vec![0.0; n];

        // First normalize
        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let mut highest = high[start];
            let mut lowest = low[start];
            for j in start..=i {
                if high[j] > highest { highest = high[j]; }
                if low[j] < lowest { lowest = low[j]; }
            }

            let range = highest - lowest;
            if range > 1e-10 {
                normalized[i] = ((close[i] - lowest) / range) * 100.0;
            } else {
                normalized[i] = 50.0;
            }
        }

        // Then smooth
        if self.smooth_period == 1 {
            return normalized;
        }

        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in self.period..n {
            if i == self.period {
                result[i] = normalized[i];
            } else {
                result[i] = alpha * normalized[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for NormalizedPriceFilter {
    fn name(&self) -> &str {
        "Normalized Price Filter"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64) * 0.3 + (i as f64 * 0.4).sin() * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; 40];

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_recursive_filter() {
        let data = make_test_data();
        let rf = RecursiveFilter::new(0.3, 0.5).unwrap();
        let result = rf.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_normalized_price_filter() {
        let data = make_test_data();
        let npf = NormalizedPriceFilter::new(14, 3).unwrap();
        let result = npf.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        // Should be 0-100
        for i in 20..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_validation() {
        assert!(RecursiveFilter::new(0.0, 0.5).is_err());
        assert!(RecursiveFilter::new(1.0, 0.5).is_err());
        assert!(RecursiveFilter::new(0.5, 1.0).is_err());
        assert!(NormalizedPriceFilter::new(2, 3).is_err());
        assert!(NormalizedPriceFilter::new(5, 0).is_err());
    }
}
