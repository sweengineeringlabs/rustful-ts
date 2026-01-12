//! Gain/Loss Ratio implementation.
//!
//! Measures the ratio of average gains to average losses.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Gain/Loss Ratio indicator.
///
/// Calculates the ratio of average winning returns to average losing returns.
/// Also known as the Win/Loss Ratio or Profit Factor.
///
/// Formula: Average Gain / Average Loss
///
/// A ratio > 1 means average wins are larger than average losses.
/// Combined with win rate, this determines overall profitability.
///
/// This indicator also provides the win rate (percentage of positive returns).
#[derive(Debug, Clone)]
pub struct GainLossRatio {
    /// Rolling window period for calculation.
    period: usize,
}

impl GainLossRatio {
    /// Create a new Gain/Loss Ratio indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate returns from price series.
    fn calculate_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }
        prices
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate gain/loss ratio for a window of returns.
    fn calculate_ratio(&self, returns: &[f64]) -> (f64, f64, f64) {
        if returns.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }

        let mut gains: Vec<f64> = Vec::new();
        let mut losses: Vec<f64> = Vec::new();

        for &r in returns {
            if r > 0.0 {
                gains.push(r);
            } else if r < 0.0 {
                losses.push(r.abs());
            }
        }

        let win_rate = gains.len() as f64 / returns.len() as f64;

        let avg_gain = if gains.is_empty() {
            0.0
        } else {
            gains.iter().sum::<f64>() / gains.len() as f64
        };

        let avg_loss = if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f64>() / losses.len() as f64
        };

        let ratio = if avg_loss == 0.0 {
            if avg_gain > 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        } else {
            avg_gain / avg_loss
        };

        (ratio, win_rate, avg_gain)
    }

    /// Calculate Gain/Loss Ratio values.
    pub fn calculate(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let returns = Self::calculate_returns(prices);
        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..returns.len() {
            let start = i + 1 - self.period;
            let window = &returns[start..=i];
            let (ratio, _, _) = self.calculate_ratio(window);
            result.push(ratio);
        }

        result
    }

    /// Calculate Gain/Loss Ratio with win rate (returns dual output).
    pub fn calculate_with_win_rate(&self, prices: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = prices.len();
        if n < self.period + 1 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let returns = Self::calculate_returns(prices);
        let mut ratio_result = vec![f64::NAN; self.period];
        let mut win_rate_result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..returns.len() {
            let start = i + 1 - self.period;
            let window = &returns[start..=i];
            let (ratio, win_rate, _) = self.calculate_ratio(window);
            ratio_result.push(ratio);
            win_rate_result.push(win_rate);
        }

        (ratio_result, win_rate_result)
    }

    /// Calculate the Profit Factor (total gains / total losses).
    pub fn calculate_profit_factor(&self, prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let returns = Self::calculate_returns(prices);
        let mut result = vec![f64::NAN; self.period];

        for i in (self.period - 1)..returns.len() {
            let start = i + 1 - self.period;
            let window = &returns[start..=i];

            let total_gains: f64 = window.iter().filter(|&&r| r > 0.0).sum();
            let total_losses: f64 = window.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

            let profit_factor = if total_losses == 0.0 {
                if total_gains > 0.0 {
                    f64::INFINITY
                } else {
                    f64::NAN
                }
            } else {
                total_gains / total_losses
            };

            result.push(profit_factor);
        }

        result
    }
}

impl TechnicalIndicator for GainLossRatio {
    fn name(&self) -> &str {
        "GainLossRatio"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let (ratio, win_rate) = self.calculate_with_win_rate(&data.close);
        Ok(IndicatorOutput::dual(ratio, win_rate))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        2 // Gain/Loss Ratio and Win Rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gain_loss_basic() {
        let gl = GainLossRatio::new(20);
        // Generate prices with some volatility
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.5).sin() * 2.0)
            .collect();
        let result = gl.calculate(&prices);

        // Should have valid values after warm-up period
        assert!(!result[30].is_nan());
    }

    #[test]
    fn test_gain_loss_uptrend() {
        let gl = GainLossRatio::new(10);
        // Strong uptrend
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64)).collect();
        let result = gl.calculate(&prices);

        // All gains, no losses - should be infinity
        let last = result.last().unwrap();
        assert!(last.is_infinite());
    }

    #[test]
    fn test_gain_loss_with_win_rate() {
        let gl = GainLossRatio::new(20);
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.5).sin() * 2.0)
            .collect();

        let (ratio, win_rate) = gl.calculate_with_win_rate(&prices);

        // Both should have valid values
        assert!(!ratio[30].is_nan());
        assert!(!win_rate[30].is_nan());

        // Win rate should be between 0 and 1
        assert!(win_rate[30] >= 0.0 && win_rate[30] <= 1.0);
    }

    #[test]
    fn test_profit_factor() {
        let gl = GainLossRatio::new(20);
        let prices: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.5).sin() * 2.0)
            .collect();

        let result = gl.calculate_profit_factor(&prices);

        // Should have valid values
        assert!(!result[30].is_nan());
    }
}
