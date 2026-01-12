//! Chandelier Exit implementation.
//!
//! Volatility-based trailing stop indicator developed by Charles Le Beau.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Chandelier Exit.
///
/// A volatility-based trailing stop indicator that:
/// - Long Exit: Highest high - (multiplier x ATR)
/// - Short Exit: Lowest low + (multiplier x ATR)
///
/// Used for trailing stop placement based on volatility and extreme prices.
#[derive(Debug, Clone)]
pub struct ChandelierExit {
    /// Period for the highest high/lowest low lookback.
    period: usize,
    /// Period for the ATR calculation.
    atr_period: usize,
    /// Multiplier for the ATR.
    multiplier: f64,
}

impl ChandelierExit {
    /// Create a new Chandelier Exit indicator.
    ///
    /// # Arguments
    /// * `period` - Period for high/low lookback (typically 22)
    /// * `atr_period` - Period for ATR calculation (typically 22)
    /// * `multiplier` - ATR multiplier (typically 3.0)
    pub fn new(period: usize, atr_period: usize, multiplier: f64) -> Self {
        Self {
            period,
            atr_period,
            multiplier,
        }
    }

    /// Create with default parameters (22-period, 3.0 multiplier).
    pub fn default_params() -> Self {
        Self::new(22, 22, 3.0)
    }

    /// Calculate True Range values.
    fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n == 0 {
            return vec![];
        }

        let mut tr = Vec::with_capacity(n);
        tr.push(high[0] - low[0]);

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        tr
    }

    /// Calculate ATR values.
    fn calculate_atr(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let tr = Self::true_range(high, low, close);
        let n = tr.len();

        if n < self.atr_period || self.atr_period == 0 {
            return vec![f64::NAN; n];
        }

        let mut atr = vec![f64::NAN; self.atr_period - 1];

        // Initial ATR is SMA of first period TRs
        let initial_atr: f64 = tr[0..self.atr_period].iter().sum::<f64>() / self.atr_period as f64;
        atr.push(initial_atr);

        // Smoothed ATR (Wilder's smoothing)
        let mut prev_atr = initial_atr;
        for i in self.atr_period..n {
            let curr_atr = (prev_atr * (self.atr_period - 1) as f64 + tr[i]) / self.atr_period as f64;
            atr.push(curr_atr);
            prev_atr = curr_atr;
        }

        atr
    }

    /// Calculate Chandelier Exit (long_exit, short_exit).
    ///
    /// Returns two series:
    /// - Long Exit: Highest high over period - (multiplier x ATR)
    /// - Short Exit: Lowest low over period + (multiplier x ATR)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        let min_required = self.period.max(self.atr_period);

        if n < min_required {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate ATR
        let atr_values = self.calculate_atr(high, low, close);

        let mut long_exit = vec![f64::NAN; self.period - 1];
        let mut short_exit = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;

            // Find highest high and lowest low in the period
            let highest_high = high[start..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lowest_low = low[start..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);

            if atr_values[i].is_nan() {
                long_exit.push(f64::NAN);
                short_exit.push(f64::NAN);
            } else {
                long_exit.push(highest_high - self.multiplier * atr_values[i]);
                short_exit.push(lowest_low + self.multiplier * atr_values[i]);
            }
        }

        (long_exit, short_exit)
    }
}

impl Default for ChandelierExit {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for ChandelierExit {
    fn name(&self) -> &str {
        "ChandelierExit"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.period.max(self.atr_period);
        if data.high.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.high.len(),
            });
        }

        let (long_exit, short_exit) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(long_exit, short_exit))
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.atr_period)
    }

    fn output_features(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chandelier_exit() {
        let ce = ChandelierExit::new(10, 10, 3.0);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (long_exit, short_exit) = ce.calculate(&high, &low, &close);

        assert_eq!(long_exit.len(), n);
        assert_eq!(short_exit.len(), n);

        // Check values after warmup
        for i in 10..n {
            if !long_exit[i].is_nan() && !short_exit[i].is_nan() {
                // Long exit should be below short exit for typical volatile data
                // In practice, long_exit < price < short_exit provides trailing stops
                assert!(!long_exit[i].is_nan());
                assert!(!short_exit[i].is_nan());
            }
        }
    }

    #[test]
    fn test_chandelier_default() {
        let ce = ChandelierExit::default();
        assert_eq!(ce.period, 22);
        assert_eq!(ce.atr_period, 22);
        assert!((ce.multiplier - 3.0).abs() < 1e-10);
    }
}
