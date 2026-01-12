//! Calmar Ratio implementation.
//!
//! Measures risk-adjusted returns relative to maximum drawdown.

use crate::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Calmar Ratio indicator.
///
/// Calculates the ratio of annualized return to maximum drawdown.
/// Formula: Annualized Return / |Maximum Drawdown|
///
/// Higher values indicate better risk-adjusted performance relative to drawdown risk.
#[derive(Debug, Clone)]
pub struct CalmarRatio {
    /// Rolling window period for calculation.
    period: usize,
    /// Annualization factor (252 for daily, 52 for weekly, 12 for monthly).
    annualization_factor: f64,
}

impl CalmarRatio {
    /// Create a new Calmar Ratio indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Self {
        Self {
            period,
            annualization_factor: 252.0,
        }
    }

    /// Create a new Calmar Ratio with custom annualization factor.
    pub fn with_annualization(period: usize, annualization_factor: f64) -> Self {
        Self {
            period,
            annualization_factor,
        }
    }

    /// Calculate maximum drawdown for a price window.
    fn max_drawdown(prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let mut peak = prices[0];
        let mut max_dd = 0.0;

        for &price in prices.iter() {
            if price > peak {
                peak = price;
            }
            let drawdown = (peak - price) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        max_dd
    }

    /// Calculate annualized return.
    fn annualized_return(prices: &[f64], annualization_factor: f64) -> f64 {
        if prices.len() < 2 {
            return f64::NAN;
        }

        let start = prices[0];
        let end = prices[prices.len() - 1];

        if start == 0.0 {
            return f64::NAN;
        }

        let total_return = (end - start) / start;
        let periods = prices.len() as f64;

        // Annualize the return
        // (1 + total_return)^(annualization_factor / periods) - 1
        (1.0 + total_return).powf(annualization_factor / periods) - 1.0
    }

    /// Calculate Calmar Ratio values.
    pub fn calculate(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let window = &prices[(i + 1 - self.period)..=i];
            let max_dd = Self::max_drawdown(window);
            let ann_return = Self::annualized_return(window, self.annualization_factor);

            if max_dd == 0.0 {
                // No drawdown
                if ann_return > 0.0 {
                    result.push(f64::INFINITY);
                } else if ann_return == 0.0 {
                    result.push(0.0);
                } else {
                    result.push(f64::NEG_INFINITY);
                }
            } else if ann_return.is_nan() {
                result.push(f64::NAN);
            } else {
                let calmar = ann_return / max_dd;
                result.push(calmar);
            }
        }

        result
    }
}

impl TechnicalIndicator for CalmarRatio {
    fn name(&self) -> &str {
        "CalmarRatio"
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
    fn test_calmar_basic() {
        let calmar = CalmarRatio::new(20);
        // Generate increasing prices with some volatility
        let prices: Vec<f64> = (0..50)
            .map(|i| {
                let base = 100.0 + (i as f64) * 0.5;
                if i % 5 == 0 { base - 1.0 } else { base }
            })
            .collect();
        let result = calmar.calculate(&prices);

        // Should have valid values after warm-up period
        assert!(!result[25].is_nan());
    }

    #[test]
    fn test_calmar_no_drawdown() {
        let calmar = CalmarRatio::new(10);
        // Strictly increasing prices - no drawdown
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64)).collect();
        let result = calmar.calculate(&prices);

        // With no drawdown and positive returns, Calmar should be infinity
        assert!(result[15].is_infinite());
    }

    #[test]
    fn test_calmar_drawdown() {
        let calmar = CalmarRatio::new(10);
        // Prices with clear drawdown
        let mut prices = vec![100.0; 5];
        prices.extend(vec![95.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0]);
        let result = calmar.calculate(&prices);

        // Should have finite positive value
        let last = result.last().unwrap();
        assert!(!last.is_nan());
        assert!(last.is_finite());
    }
}
