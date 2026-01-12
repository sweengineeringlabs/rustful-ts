//! Ulcer Index implementation.
//!
//! Measures downside volatility and drawdown risk.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Ulcer Index.
///
/// The Ulcer Index measures downside volatility, specifically the depth and
/// duration of price declines. It focuses only on drawdowns, making it
/// particularly useful for risk-averse investors.
///
/// Formula:
/// 1. Calculate percentage drawdown: R = ((close - max(close, n)) / max(close, n)) * 100
/// 2. Ulcer Index = sqrt(sum(R^2, n) / n)
///
/// Lower values indicate less risk/drawdown, higher values indicate more stress.
#[derive(Debug, Clone)]
pub struct UlcerIndex {
    /// Lookback period (commonly 14).
    period: usize,
}

impl UlcerIndex {
    /// Create a new Ulcer Index indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period (commonly 14)
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate Ulcer Index values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Find the highest high in the window for each point
            // and calculate squared percentage drawdowns
            let mut sum_sq = 0.0;
            let mut max_so_far = f64::NEG_INFINITY;

            for j in 0..window.len() {
                max_so_far = max_so_far.max(window[j]);

                if max_so_far > 0.0 {
                    let pct_drawdown = ((window[j] - max_so_far) / max_so_far) * 100.0;
                    sum_sq += pct_drawdown.powi(2);
                }
            }

            // Ulcer Index = sqrt(mean of squared drawdowns)
            let ulcer = (sum_sq / self.period as f64).sqrt();
            result.push(ulcer);
        }

        result
    }

    /// Calculate using running max (alternative implementation).
    /// This looks at drawdown from the running maximum over the entire history.
    pub fn calculate_running(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        // Calculate percentage drawdown from running max
        let mut drawdowns = Vec::with_capacity(n);
        let mut max_so_far = data[0];

        for &price in data.iter() {
            max_so_far = max_so_far.max(price);
            if max_so_far > 0.0 {
                let pct_dd = ((price - max_so_far) / max_so_far) * 100.0;
                drawdowns.push(pct_dd.powi(2));
            } else {
                drawdowns.push(0.0);
            }
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &drawdowns[start..=i];

            let sum_sq: f64 = window.iter().sum();
            let ulcer = (sum_sq / self.period as f64).sqrt();
            result.push(ulcer);
        }

        result
    }
}

impl TechnicalIndicator for UlcerIndex {
    fn name(&self) -> &str {
        "UlcerIndex"
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
    fn test_ulcer_index() {
        let ui = UlcerIndex::new(14);

        // Generate sample data with some drawdowns
        let data: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.2).sin() * 10.0)
            .collect();

        let result = ui.calculate(&data);

        assert_eq!(result.len(), 50);

        // First 13 values should be NaN
        for i in 0..13 {
            assert!(result[i].is_nan());
        }

        // Ulcer Index should be non-negative
        for i in 13..50 {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_ulcer_uptrend() {
        let ui = UlcerIndex::new(10);

        // Pure uptrend - no drawdowns
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = ui.calculate(&data);

        // In pure uptrend, Ulcer Index should be 0
        for i in 9..30 {
            assert!((result[i] - 0.0).abs() < 1e-10, "Ulcer should be 0 in uptrend");
        }
    }

    #[test]
    fn test_ulcer_drawdown() {
        let ui = UlcerIndex::new(10);

        // Data with significant drawdown
        let mut data: Vec<f64> = (0..15).map(|i| 100.0 + i as f64 * 2.0).collect();
        // Add drawdown
        data.extend((0..15).map(|i| 128.0 - i as f64 * 2.0));

        let result = ui.calculate(&data);

        // After drawdown begins, Ulcer Index should increase
        let early = result[14]; // Before drawdown
        let late = result[25]; // During drawdown

        assert!(late > early, "Ulcer should increase during drawdown");
    }

    #[test]
    fn test_ulcer_running() {
        let ui = UlcerIndex::new(10);

        let data: Vec<f64> = (0..30)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 10.0)
            .collect();

        let result1 = ui.calculate(&data);
        let result2 = ui.calculate_running(&data);

        // Both methods should produce valid results
        for i in 9..30 {
            assert!(!result1[i].is_nan());
            assert!(!result2[i].is_nan());
            assert!(result1[i] >= 0.0);
            assert!(result2[i] >= 0.0);
        }
    }
}
