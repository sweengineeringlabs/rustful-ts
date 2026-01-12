//! Maximum Drawdown implementation.
//!
//! Measures the largest peak-to-trough decline in portfolio value.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Maximum Drawdown indicator.
///
/// Calculates the maximum observed loss from a peak to a trough before a new peak.
/// Values are expressed as positive percentages (e.g., 0.20 = 20% drawdown).
///
/// Lower values indicate more stable performance with smaller declines.
#[derive(Debug, Clone)]
pub struct MaxDrawdown {
    /// Rolling window period for calculation (0 for cumulative).
    period: usize,
}

impl MaxDrawdown {
    /// Create a new Maximum Drawdown indicator with rolling window.
    ///
    /// # Arguments
    /// * `period` - Rolling window period (use 0 for cumulative from start)
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Create a cumulative Maximum Drawdown (from beginning of series).
    pub fn cumulative() -> Self {
        Self { period: 0 }
    }

    /// Calculate maximum drawdown for a single price window.
    fn window_max_drawdown(prices: &[f64]) -> f64 {
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

    /// Calculate Maximum Drawdown values.
    pub fn calculate(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < 2 {
            return vec![0.0; n];
        }

        if self.period == 0 {
            // Cumulative max drawdown
            let mut result = Vec::with_capacity(n);
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
                result.push(max_dd);
            }
            result
        } else {
            // Rolling max drawdown
            if n < self.period {
                return vec![f64::NAN; n];
            }

            let mut result = vec![f64::NAN; self.period - 1];

            for i in (self.period - 1)..n {
                let window = &prices[(i + 1 - self.period)..=i];
                let max_dd = Self::window_max_drawdown(window);
                result.push(max_dd);
            }

            result
        }
    }

    /// Calculate current drawdown from peak (not max, just current).
    pub fn calculate_current_drawdown(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < 2 {
            return vec![0.0; n];
        }

        let mut result = Vec::with_capacity(n);
        let mut peak = prices[0];

        for &price in prices.iter() {
            if price > peak {
                peak = price;
            }
            let drawdown = (peak - price) / peak;
            result.push(drawdown);
        }

        result
    }
}

impl TechnicalIndicator for MaxDrawdown {
    fn name(&self) -> &str {
        "MaxDrawdown"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = if self.period == 0 { 2 } else { self.period };

        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        if self.period == 0 { 2 } else { self.period }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_drawdown_basic() {
        let mdd = MaxDrawdown::new(20);
        // Prices with volatility
        let mut prices: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64) * 0.5).collect();
        // Insert a significant drop
        prices[15] = 90.0;
        prices[16] = 88.0;
        prices[17] = 92.0;

        let result = mdd.calculate(&prices);

        // Should have valid values after warm-up
        assert!(!result[25].is_nan());
        // Max drawdown should be positive
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_max_drawdown_cumulative() {
        let mdd = MaxDrawdown::cumulative();
        let prices = vec![100.0, 110.0, 105.0, 115.0, 100.0, 120.0];
        let result = mdd.calculate(&prices);

        // At index 4, drawdown from peak 115 to 100 = 13.04%
        let expected_dd = (115.0 - 100.0) / 115.0;
        assert!((result[4] - expected_dd).abs() < 0.001);
    }

    #[test]
    fn test_max_drawdown_no_decline() {
        let mdd = MaxDrawdown::new(10);
        // Strictly increasing prices
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64)).collect();
        let result = mdd.calculate(&prices);

        // No drawdown expected
        assert!(result[15] == 0.0);
    }

    #[test]
    fn test_current_drawdown() {
        let mdd = MaxDrawdown::cumulative();
        let prices = vec![100.0, 110.0, 105.0, 120.0, 100.0];
        let current_dd = mdd.calculate_current_drawdown(&prices);

        // At peak 110, then 105 = 4.5% drawdown
        let dd_at_2 = (110.0 - 105.0) / 110.0;
        assert!((current_dd[2] - dd_at_2).abs() < 0.001);

        // At new peak 120, current drawdown = 0
        assert!(current_dd[3] == 0.0);

        // At 100 from peak 120 = 16.67%
        let dd_at_4 = (120.0 - 100.0) / 120.0;
        assert!((current_dd[4] - dd_at_4).abs() < 0.001);
    }
}
