//! Average True Range implementation.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};
use indicator_api::ATRConfig;

/// Average True Range (ATR).
///
/// Volatility indicator showing degree of price movement.
#[derive(Debug, Clone)]
pub struct ATR {
    period: usize,
}

impl ATR {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: ATRConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate True Range for each bar.
    pub fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n == 0 {
            return vec![];
        }

        let mut tr = Vec::with_capacity(n);
        tr.push(high[0] - low[0]); // First TR is just high-low

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        tr
    }

    /// Calculate ATR values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let tr = Self::true_range(high, low, close);
        let n = tr.len();

        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut atr = vec![f64::NAN; self.period - 1];

        // Initial ATR is SMA of first period TRs
        let initial_atr: f64 = tr[0..self.period].iter().sum::<f64>() / self.period as f64;
        atr.push(initial_atr);

        // Smoothed ATR (Wilder's smoothing)
        let mut prev_atr = initial_atr;
        for i in self.period..n {
            let curr_atr = (prev_atr * (self.period - 1) as f64 + tr[i]) / self.period as f64;
            atr.push(curr_atr);
            prev_atr = curr_atr;
        }

        atr
    }
}

impl TechnicalIndicator for ATR {
    fn name(&self) -> &str {
        "ATR"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low, &data.close);
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
    fn test_atr() {
        let atr = ATR::new(14);

        // Create sample OHLC data
        let high: Vec<f64> = (0..30).map(|i| 102.0 + (i as f64).sin() * 2.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 98.0 + (i as f64).sin() * 2.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64).sin() * 2.0).collect();

        let result = atr.calculate(&high, &low, &close);

        assert_eq!(result.len(), 30);
        // First 13 values should be NaN
        for i in 0..13 {
            assert!(result[i].is_nan());
        }
        // ATR should be positive
        for i in 13..30 {
            assert!(result[i] > 0.0);
        }
    }
}
