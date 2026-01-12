//! Chaikin Volatility implementation.
//!
//! Measures volatility based on the spread between high and low prices.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Chaikin Volatility.
///
/// Measures volatility by calculating the rate of change in the difference
/// between high and low prices, smoothed by an EMA.
///
/// Formula:
/// 1. Calculate H-L spread: high - low
/// 2. Calculate EMA of H-L spread
/// 3. Chaikin Vol = ((EMA today - EMA n days ago) / EMA n days ago) * 100
#[derive(Debug, Clone)]
pub struct ChaikinVolatility {
    /// EMA period for smoothing H-L spread.
    ema_period: usize,
    /// Rate of change period.
    roc_period: usize,
}

impl ChaikinVolatility {
    /// Create a new Chaikin Volatility indicator.
    ///
    /// # Arguments
    /// * `ema_period` - Period for EMA smoothing (default: 10)
    /// * `roc_period` - Period for rate of change (default: 10)
    pub fn new(ema_period: usize, roc_period: usize) -> Self {
        Self { ema_period, roc_period }
    }

    /// Create with default parameters (10, 10).
    pub fn default_params() -> Self {
        Self::new(10, 10)
    }

    /// Calculate EMA of a series.
    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period || period == 0 {
            return vec![f64::NAN; n];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = vec![f64::NAN; period - 1];

        // Initial EMA is SMA of first period
        let initial_sma: f64 = data[0..period].iter().sum::<f64>() / period as f64;
        result.push(initial_sma);

        let mut prev_ema = initial_sma;
        for i in period..n {
            let ema = (data[i] - prev_ema) * multiplier + prev_ema;
            result.push(ema);
            prev_ema = ema;
        }

        result
    }

    /// Calculate Chaikin Volatility values.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let min_len = self.ema_period + self.roc_period;

        if n < min_len || self.ema_period == 0 || self.roc_period == 0 {
            return vec![f64::NAN; n];
        }

        // Calculate H-L spread
        let hl_spread: Vec<f64> = high.iter()
            .zip(low.iter())
            .map(|(&h, &l)| h - l)
            .collect();

        // Calculate EMA of H-L spread
        let ema_hl = Self::ema(&hl_spread, self.ema_period);

        // Calculate rate of change
        let mut result = vec![f64::NAN; self.ema_period + self.roc_period - 1];

        for i in (self.ema_period + self.roc_period - 1)..n {
            let current = ema_hl[i];
            let past = ema_hl[i - self.roc_period];

            if past.is_nan() || current.is_nan() || past.abs() < 1e-10 {
                result.push(f64::NAN);
            } else {
                let chaikin_vol = ((current - past) / past) * 100.0;
                result.push(chaikin_vol);
            }
        }

        result
    }
}

impl TechnicalIndicator for ChaikinVolatility {
    fn name(&self) -> &str {
        "ChaikinVolatility"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_len = self.ema_period + self.roc_period;
        if data.high.len() < min_len {
            return Err(IndicatorError::InsufficientData {
                required: min_len,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.ema_period + self.roc_period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chaikin_volatility() {
        let cv = ChaikinVolatility::new(10, 10);

        // Generate sample OHLC data
        let high: Vec<f64> = (0..50)
            .map(|i| 105.0 + (i as f64 * 0.2).sin() * 3.0)
            .collect();
        let low: Vec<f64> = (0..50)
            .map(|i| 95.0 + (i as f64 * 0.2).sin() * 3.0)
            .collect();

        let result = cv.calculate(&high, &low);

        assert_eq!(result.len(), 50);

        // First 19 values should be NaN
        for i in 0..19 {
            assert!(result[i].is_nan());
        }

        // Should have valid values after warmup
        for i in 19..50 {
            assert!(!result[i].is_nan(), "Expected valid value at index {}", i);
        }
    }

    #[test]
    fn test_ema_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ema = ChaikinVolatility::ema(&data, 5);

        assert_eq!(ema.len(), 10);
        // First 4 values should be NaN
        for i in 0..4 {
            assert!(ema[i].is_nan());
        }
        // First EMA should be SMA of first 5 values = 3.0
        assert!((ema[4] - 3.0).abs() < 1e-10);
    }
}
