//! Triangular Moving Average (TRIMA) implementation.
//!
//! A double-smoothed SMA that applies triangular weighting.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::TRIMAConfig;

/// Triangular Moving Average (TRIMA).
///
/// TRIMA applies double smoothing by calculating the SMA of an SMA.
/// This creates a triangular weighting pattern where middle values
/// have the highest weight and weights decrease towards both ends.
///
/// For odd periods: TRIMA = SMA(SMA(price, (n+1)/2), (n+1)/2)
/// For even periods: TRIMA = SMA(SMA(price, n/2), n/2+1)
#[derive(Debug, Clone)]
pub struct TRIMA {
    period: usize,
}

impl TRIMA {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: TRIMAConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate TRIMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.period || self.period == 0 {
            return vec![f64::NAN; data.len()];
        }

        // Calculate sub-periods for double SMA
        let (first_period, second_period) = if self.period % 2 == 1 {
            let half = (self.period + 1) / 2;
            (half, half)
        } else {
            let half = self.period / 2;
            (half, half + 1)
        };

        // First SMA
        let first_sma = self.sma(data, first_period);

        // Second SMA on first SMA
        self.sma(&first_sma, second_period)
    }

    /// Calculate simple moving average.
    fn sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return vec![f64::NAN; data.len()];
        }

        let mut result = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            if i < period - 1 {
                result.push(f64::NAN);
            } else {
                let window = &data[(i + 1 - period)..=i];
                if window.iter().all(|x| !x.is_nan()) {
                    let sum: f64 = window.iter().sum();
                    result.push(sum / period as f64);
                } else {
                    result.push(f64::NAN);
                }
            }
        }

        result
    }
}

impl Default for TRIMA {
    fn default() -> Self {
        Self::from_config(TRIMAConfig::default())
    }
}

impl TechnicalIndicator for TRIMA {
    fn name(&self) -> &str {
        "TRIMA"
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trima() {
        let trima = TRIMA::new(5);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = trima.calculate(&data);

        // First values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(result[2].is_nan());
        assert!(result[3].is_nan());
        // At index 4 (5th value), we have enough data
        assert!(!result[4].is_nan());
    }

    #[test]
    fn test_trima_odd_period() {
        let trima = TRIMA::new(5);
        // Period 5: first SMA uses 3, second SMA uses 3
        // For ascending data, TRIMA should be close to middle value
        let data: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let result = trima.calculate(&data);

        // TRIMA should smooth the data
        assert!(!result[9].is_nan());
    }

    #[test]
    fn test_trima_even_period() {
        let trima = TRIMA::new(6);
        // Period 6: first SMA uses 3, second SMA uses 4
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let result = trima.calculate(&data);

        assert!(!result[11].is_nan());
    }

    #[test]
    fn test_trima_insufficient_data() {
        let trima = TRIMA::new(10);
        let data = vec![1.0, 2.0, 3.0];
        let result = trima.calculate(&data);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_trima_default() {
        let trima = TRIMA::default();
        assert_eq!(trima.period, 20);
    }

    #[test]
    fn test_trima_technical_indicator_trait() {
        let trima = TRIMA::new(10);
        assert_eq!(trima.name(), "TRIMA");
        assert_eq!(trima.min_periods(), 10);
    }

    #[test]
    fn test_trima_smoothness() {
        let trima = TRIMA::new(5);
        // Noisy data
        let data = vec![10.0, 12.0, 8.0, 14.0, 6.0, 16.0, 4.0, 18.0, 2.0, 20.0];
        let result = trima.calculate(&data);

        // TRIMA should reduce noise variance
        let valid_results: Vec<f64> = result.iter().filter(|x| !x.is_nan()).cloned().collect();
        if valid_results.len() >= 2 {
            let variance: f64 = {
                let mean = valid_results.iter().sum::<f64>() / valid_results.len() as f64;
                valid_results.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / valid_results.len() as f64
            };
            let raw_variance = {
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
            };
            assert!(variance < raw_variance);
        }
    }
}
