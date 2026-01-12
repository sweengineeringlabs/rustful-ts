//! Normalized Average True Range (NATR) implementation.
//!
//! ATR expressed as a percentage of price.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Normalized Average True Range (NATR).
///
/// The Normalized ATR expresses the Average True Range as a percentage of the
/// closing price, making it comparable across different price levels and securities.
///
/// Formula:
/// NATR = (ATR / Close) * 100
///
/// Where ATR is calculated using Wilder's smoothing method.
#[derive(Debug, Clone)]
pub struct NormalizedATR {
    /// ATR period.
    period: usize,
}

impl NormalizedATR {
    /// Create a new Normalized ATR indicator.
    ///
    /// # Arguments
    /// * `period` - ATR period (commonly 14)
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate True Range for each bar.
    fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
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

    /// Calculate ATR values using Wilder's smoothing.
    fn atr(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
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

    /// Calculate Normalized ATR values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let atr_values = self.atr(high, low, close);

        atr_values.iter()
            .zip(close.iter())
            .map(|(&atr, &c)| {
                if atr.is_nan() || c.abs() < 1e-10 {
                    f64::NAN
                } else {
                    (atr / c) * 100.0
                }
            })
            .collect()
    }
}

impl TechnicalIndicator for NormalizedATR {
    fn name(&self) -> &str {
        "NormalizedATR"
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
    fn test_normalized_atr() {
        let natr = NormalizedATR::new(14);

        // Generate sample OHLC data
        let high: Vec<f64> = (0..50)
            .map(|i| 102.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let low: Vec<f64> = (0..50)
            .map(|i| 98.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();

        let result = natr.calculate(&high, &low, &close);

        assert_eq!(result.len(), 50);

        // First 13 values should be NaN
        for i in 0..13 {
            assert!(result[i].is_nan());
        }

        // NATR should be positive and reasonable (typically 0-10% for most stocks)
        for i in 13..50 {
            assert!(result[i] > 0.0);
            assert!(result[i] < 50.0); // Sanity check
        }
    }

    #[test]
    fn test_natr_price_independence() {
        let natr = NormalizedATR::new(10);

        // High-priced stock
        let high1: Vec<f64> = (0..30).map(|_| 1020.0).collect();
        let low1: Vec<f64> = (0..30).map(|_| 980.0).collect();
        let close1: Vec<f64> = (0..30).map(|_| 1000.0).collect();

        // Low-priced stock with same relative range
        let high2: Vec<f64> = (0..30).map(|_| 10.2).collect();
        let low2: Vec<f64> = (0..30).map(|_| 9.8).collect();
        let close2: Vec<f64> = (0..30).map(|_| 10.0).collect();

        let result1 = natr.calculate(&high1, &low1, &close1);
        let result2 = natr.calculate(&high2, &low2, &close2);

        // NATR should be similar for same relative range
        for i in 10..30 {
            if !result1[i].is_nan() && !result2[i].is_nan() {
                let diff = (result1[i] - result2[i]).abs();
                assert!(diff < 0.01, "NATR should be similar for same relative range");
            }
        }
    }
}
