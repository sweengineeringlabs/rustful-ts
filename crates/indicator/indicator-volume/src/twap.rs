//! Time Weighted Average Price (TWAP) implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};

/// Time Weighted Average Price.
///
/// TWAP calculates the average price over time, giving equal weight to
/// each time period regardless of volume.
///
/// TWAP = Sum(Typical Price) / Number of Periods
///
/// Where Typical Price = (High + Low + Close) / 3
///
/// Can be calculated as:
/// - Rolling: Over a fixed lookback window
/// - Cumulative: From the start of the data
#[derive(Debug, Clone)]
pub struct TWAP {
    period: Option<usize>,
}

impl TWAP {
    /// Create a cumulative TWAP.
    pub fn new() -> Self {
        Self { period: None }
    }

    /// Create a rolling TWAP with specified period.
    pub fn rolling(period: usize) -> Self {
        Self {
            period: Some(period),
        }
    }

    /// Calculate TWAP values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n == 0 {
            return vec![];
        }

        // Calculate typical prices
        let typical: Vec<f64> = (0..n)
            .map(|i| (high[i] + low[i] + close[i]) / 3.0)
            .collect();

        match self.period {
            Some(period) => self.calculate_rolling(&typical, period),
            None => self.calculate_cumulative(&typical),
        }
    }

    /// Calculate cumulative TWAP.
    fn calculate_cumulative(&self, typical: &[f64]) -> Vec<f64> {
        let n = typical.len();
        let mut result = Vec::with_capacity(n);
        let mut sum = 0.0;

        for i in 0..n {
            sum += typical[i];
            result.push(sum / (i + 1) as f64);
        }

        result
    }

    /// Calculate rolling TWAP.
    fn calculate_rolling(&self, typical: &[f64], period: usize) -> Vec<f64> {
        let n = typical.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let mut sum: f64 = typical[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        for i in period..n {
            sum = sum - typical[i - period] + typical[i];
            result[i] = sum / period as f64;
        }

        result
    }
}

impl Default for TWAP {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TWAP {
    fn name(&self) -> &str {
        "TWAP"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.period.unwrap_or(1);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period.unwrap_or(1)
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_cumulative() {
        let twap = TWAP::new();
        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![95.0, 96.0, 97.0, 98.0, 99.0];
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0];

        let result = twap.calculate(&high, &low, &close);

        assert_eq!(result.len(), 5);
        // First TWAP equals first typical price
        let tp0 = (105.0 + 95.0 + 100.0) / 3.0;
        assert!((result[0] - tp0).abs() < 1e-10);
        // TWAP should converge to average typical price
        assert!(result[4] > 95.0 && result[4] < 110.0);
    }

    #[test]
    fn test_twap_rolling() {
        let twap = TWAP::rolling(3);
        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![95.0, 96.0, 97.0, 98.0, 99.0];
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0];

        let result = twap.calculate(&high, &low, &close);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(!result[2].is_nan());

        // Verify rolling calculation at index 2
        let tp0 = (105.0 + 95.0 + 100.0) / 3.0;
        let tp1 = (106.0 + 96.0 + 101.0) / 3.0;
        let tp2 = (107.0 + 97.0 + 102.0) / 3.0;
        let expected = (tp0 + tp1 + tp2) / 3.0;
        assert!((result[2] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_twap_trait() {
        let twap_cum = TWAP::new();
        let twap_roll = TWAP::rolling(20);

        assert_eq!(twap_cum.name(), "TWAP");
        assert_eq!(twap_cum.min_periods(), 1);
        assert_eq!(twap_roll.min_periods(), 20);
    }
}
