//! Smoothed Moving Average (SMMA) implementation.
//!
//! Also known as Wilder's Smoothing Method or Running Moving Average (RMA).

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::SMMAConfig;

/// Smoothed Moving Average (SMMA).
///
/// Also known as Wilder's Smoothing Method, this is an exponentially weighted
/// moving average where the smoothing factor is 1/period (instead of 2/(period+1)
/// like in standard EMA).
///
/// Formula: SMMA(i) = (SMMA(i-1) * (period - 1) + price(i)) / period
#[derive(Debug, Clone)]
pub struct SMMA {
    period: usize,
}

impl SMMA {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: SMMAConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate SMMA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        if data.len() < self.period || self.period == 0 {
            return vec![f64::NAN; data.len()];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        // Initial SMA as seed for first SMMA value
        let initial_sma: f64 = data[0..self.period].iter().sum::<f64>() / self.period as f64;
        result.push(initial_sma);

        // SMMA calculation: SMMA(i) = (SMMA(i-1) * (period - 1) + price(i)) / period
        let mut smma = initial_sma;
        for i in self.period..data.len() {
            if data[i].is_nan() {
                result.push(f64::NAN);
            } else {
                smma = (smma * (self.period - 1) as f64 + data[i]) / self.period as f64;
                result.push(smma);
            }
        }

        result
    }
}

impl Default for SMMA {
    fn default() -> Self {
        Self::from_config(SMMAConfig::default())
    }
}

impl TechnicalIndicator for SMMA {
    fn name(&self) -> &str {
        "SMMA"
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
    fn test_smma() {
        let smma = SMMA::new(3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = smma.calculate(&data);

        // First two values should be NaN
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // First SMMA is SMA = (1+2+3)/3 = 2.0
        assert!((result[2] - 2.0).abs() < 1e-10);
        // SMMA(3) = (2.0 * 2 + 4) / 3 = 8/3 = 2.666...
        assert!((result[3] - 8.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_smma_insufficient_data() {
        let smma = SMMA::new(5);
        let data = vec![1.0, 2.0, 3.0];
        let result = smma.calculate(&data);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_smma_default() {
        let smma = SMMA::default();
        assert_eq!(smma.period, 14);
    }

    #[test]
    fn test_smma_technical_indicator_trait() {
        let smma = SMMA::new(3);
        assert_eq!(smma.name(), "SMMA");
        assert_eq!(smma.min_periods(), 3);
    }
}
