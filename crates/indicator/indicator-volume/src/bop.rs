//! Balance of Power (BOP) implementation.

use indicator_spi::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Balance of Power.
///
/// BOP measures the strength of buyers vs sellers by comparing
/// close-open to high-low range.
///
/// BOP = (Close - Open) / (High - Low)
///
/// - Values near +1: Buyers in control (bullish)
/// - Values near -1: Sellers in control (bearish)
/// - Values near 0: Balanced
///
/// A smoothed version (SMA) is often used for cleaner signals.
#[derive(Debug, Clone)]
pub struct BalanceOfPower {
    smooth_period: Option<usize>,
}

impl BalanceOfPower {
    pub fn new() -> Self {
        Self { smooth_period: None }
    }

    pub fn with_smoothing(period: usize) -> Self {
        Self {
            smooth_period: Some(period),
        }
    }

    /// Calculate raw BOP values.
    pub fn calculate_raw(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let hl_range = high[i] - low[i];
            let bop = if hl_range > 0.0 {
                (close[i] - open[i]) / hl_range
            } else {
                0.0
            };
            result.push(bop);
        }

        result
    }

    /// Calculate BOP values (optionally smoothed).
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let raw = self.calculate_raw(open, high, low, close);

        match self.smooth_period {
            Some(period) => self.sma(&raw, period),
            None => raw,
        }
    }

    /// Simple moving average for smoothing.
    fn sma(&self, values: &[f64], period: usize) -> Vec<f64> {
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let mut sum: f64 = values[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        for i in period..n {
            sum = sum - values[i - period] + values[i];
            result[i] = sum / period as f64;
        }

        result
    }
}

impl Default for BalanceOfPower {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for BalanceOfPower {
    fn name(&self) -> &str {
        "Balance of Power"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.smooth_period.unwrap_or(1);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.smooth_period.unwrap_or(1)
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for BalanceOfPower {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last > 0.3 {
                    return Ok(IndicatorSignal::Bullish);
                } else if last < -0.3 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        Ok(values
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    IndicatorSignal::Neutral
                } else if v > 0.3 {
                    IndicatorSignal::Bullish
                } else if v < -0.3 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bop_bullish() {
        let bop = BalanceOfPower::new();
        // Bullish: close near high
        let open = vec![100.0];
        let high = vec![110.0];
        let low = vec![95.0];
        let close = vec![109.0];

        let result = bop.calculate_raw(&open, &high, &low, &close);

        // BOP = (109-100)/(110-95) = 9/15 = 0.6
        assert!((result[0] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_bop_bearish() {
        let bop = BalanceOfPower::new();
        // Bearish: close near low
        let open = vec![105.0];
        let high = vec![110.0];
        let low = vec![95.0];
        let close = vec![96.0];

        let result = bop.calculate_raw(&open, &high, &low, &close);

        // BOP = (96-105)/(110-95) = -9/15 = -0.6
        assert!((result[0] - (-0.6)).abs() < 1e-10);
    }

    #[test]
    fn test_bop_smoothed() {
        let bop = BalanceOfPower::with_smoothing(3);
        let open = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![95.0, 96.0, 97.0, 98.0, 99.0];
        let close = vec![104.0, 105.0, 106.0, 107.0, 108.0];

        let result = bop.calculate(&open, &high, &low, &close);

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!(!result[2].is_nan());
    }
}
