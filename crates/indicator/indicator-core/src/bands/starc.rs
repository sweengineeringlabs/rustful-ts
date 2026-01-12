//! STARC Bands (Stoller Average Range Channels) implementation.
//!
//! STARC Bands are volatility bands based on ATR around an SMA.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// STARC Bands (Stoller Average Range Channels).
///
/// Volatility-based envelope indicator consisting of:
/// - Middle Band: SMA of close
/// - Upper Band: SMA + (multiplier x ATR)
/// - Lower Band: SMA - (multiplier x ATR)
///
/// STARC Bands help identify overbought/oversold conditions and potential
/// reversal points based on volatility.
#[derive(Debug, Clone)]
pub struct STARCBands {
    /// Period for the SMA calculation.
    sma_period: usize,
    /// Period for the ATR calculation.
    atr_period: usize,
    /// Multiplier for the ATR to determine band width.
    multiplier: f64,
}

impl STARCBands {
    /// Create a new STARC Bands indicator.
    ///
    /// # Arguments
    /// * `sma_period` - Period for the SMA (typically 5-6)
    /// * `atr_period` - Period for the ATR (typically 15)
    /// * `multiplier` - ATR multiplier for band width (typically 2.0)
    pub fn new(sma_period: usize, atr_period: usize, multiplier: f64) -> Self {
        Self {
            sma_period,
            atr_period,
            multiplier,
        }
    }

    /// Create with default parameters (6-period SMA, 15-period ATR, 2.0 multiplier).
    pub fn default_params() -> Self {
        Self::new(6, 15, 2.0)
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

    /// Calculate SMA values.
    fn calculate_sma(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.sma_period || self.sma_period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.sma_period - 1];
        let mut sum: f64 = data[0..self.sma_period].iter().sum();
        result.push(sum / self.sma_period as f64);

        for i in self.sma_period..n {
            sum = sum - data[i - self.sma_period] + data[i];
            result.push(sum / self.sma_period as f64);
        }

        result
    }

    /// Calculate STARC Bands (middle, upper, lower).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        // Calculate SMA of close (middle band)
        let middle = self.calculate_sma(close);

        // Calculate ATR
        let atr_values = self.calculate_atr(high, low, close);

        // Calculate upper and lower bands
        let mut upper = Vec::with_capacity(n);
        let mut lower = Vec::with_capacity(n);

        for i in 0..n {
            if middle[i].is_nan() || atr_values[i].is_nan() {
                upper.push(f64::NAN);
                lower.push(f64::NAN);
            } else {
                upper.push(middle[i] + self.multiplier * atr_values[i]);
                lower.push(middle[i] - self.multiplier * atr_values[i]);
            }
        }

        (middle, upper, lower)
    }
}

impl Default for STARCBands {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for STARCBands {
    fn name(&self) -> &str {
        "STARCBands"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.sma_period.max(self.atr_period);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.sma_period.max(self.atr_period)
    }

    fn output_features(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starc_bands() {
        let starc = STARCBands::new(5, 10, 2.0);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (middle, upper, lower) = starc.calculate(&high, &low, &close);

        assert_eq!(middle.len(), n);
        assert_eq!(upper.len(), n);
        assert_eq!(lower.len(), n);

        // Check bands exist after warmup
        for i in 10..n {
            if !middle[i].is_nan() && !upper[i].is_nan() {
                assert!(upper[i] > middle[i], "Upper band should be above middle");
                assert!(lower[i] < middle[i], "Lower band should be below middle");
            }
        }
    }

    #[test]
    fn test_starc_default() {
        let starc = STARCBands::default();
        assert_eq!(starc.sma_period, 6);
        assert_eq!(starc.atr_period, 15);
        assert!((starc.multiplier - 2.0).abs() < 1e-10);
    }
}
