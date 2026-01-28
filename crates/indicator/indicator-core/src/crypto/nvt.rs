//! NVT Ratio (Network Value to Transactions) - IND-085
//!
//! A valuation metric for cryptocurrencies, similar to P/E ratio for stocks.
//! NVT = Market Cap / Transaction Volume (in USD)
//!
//! Interpretation:
//! - High NVT (>95): Network may be overvalued relative to transaction activity
//! - Low NVT (<65): Network may be undervalued relative to transaction activity
//! - NVT Signal uses smoothed transaction volume for less noise

use indicator_spi::IndicatorSignal;

/// NVT Ratio output containing raw and signal values.
#[derive(Debug, Clone)]
pub struct NVTRatioOutput {
    /// Raw NVT ratio values.
    pub nvt: Vec<f64>,
    /// NVT Signal (smoothed version).
    pub nvt_signal: Vec<f64>,
}

/// NVT signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NVTSignal {
    /// NVT below lower threshold - potentially undervalued.
    Undervalued,
    /// NVT above upper threshold - potentially overvalued.
    Overvalued,
    /// NVT in normal range.
    Normal,
}

/// NVT Ratio (Network Value to Transactions) - IND-085
///
/// Measures cryptocurrency valuation relative to its transaction volume.
/// Similar to P/E ratio for stocks.
///
/// # Formula
/// ```text
/// NVT = Market Cap / Transaction Volume
/// NVT Signal = Market Cap / SMA(Transaction Volume, signal_period)
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::NVTRatio;
///
/// let nvt = NVTRatio::new(14, 65.0, 95.0);
/// let market_caps = vec![1e9, 1.1e9, 1.05e9, 1.2e9];
/// let tx_volumes = vec![1e7, 1.2e7, 1.1e7, 1.3e7];
/// let output = nvt.calculate(&market_caps, &tx_volumes);
/// ```
#[derive(Debug, Clone)]
pub struct NVTRatio {
    /// Period for smoothing transaction volume (NVT Signal).
    signal_period: usize,
    /// Lower threshold for undervalued signal.
    lower_threshold: f64,
    /// Upper threshold for overvalued signal.
    upper_threshold: f64,
}

impl NVTRatio {
    /// Create a new NVT Ratio indicator.
    ///
    /// # Arguments
    /// * `signal_period` - Period for SMA smoothing of transaction volume (default: 14)
    /// * `lower_threshold` - Below this NVT value suggests undervaluation (default: 65)
    /// * `upper_threshold` - Above this NVT value suggests overvaluation (default: 95)
    pub fn new(signal_period: usize, lower_threshold: f64, upper_threshold: f64) -> Self {
        Self {
            signal_period,
            lower_threshold,
            upper_threshold,
        }
    }

    /// Create with default thresholds (65/95).
    pub fn with_period(signal_period: usize) -> Self {
        Self::new(signal_period, 65.0, 95.0)
    }

    /// Calculate NVT ratio from market cap and transaction volume series.
    ///
    /// # Arguments
    /// * `market_caps` - Series of market capitalization values
    /// * `tx_volumes` - Series of transaction volumes (in same currency as market cap)
    ///
    /// # Returns
    /// NVTRatioOutput containing raw NVT and smoothed NVT Signal
    pub fn calculate(&self, market_caps: &[f64], tx_volumes: &[f64]) -> NVTRatioOutput {
        let n = market_caps.len().min(tx_volumes.len());

        if n == 0 {
            return NVTRatioOutput {
                nvt: vec![],
                nvt_signal: vec![],
            };
        }

        // Calculate raw NVT
        let nvt: Vec<f64> = (0..n)
            .map(|i| {
                if tx_volumes[i] > 0.0 {
                    market_caps[i] / tx_volumes[i]
                } else {
                    f64::NAN
                }
            })
            .collect();

        // Calculate NVT Signal using SMA of transaction volume
        let nvt_signal = self.calculate_nvt_signal(market_caps, tx_volumes, n);

        NVTRatioOutput { nvt, nvt_signal }
    }

    /// Calculate NVT Signal (smoothed version).
    fn calculate_nvt_signal(
        &self,
        market_caps: &[f64],
        tx_volumes: &[f64],
        n: usize,
    ) -> Vec<f64> {
        let mut result = vec![f64::NAN; n];

        if n < self.signal_period {
            return result;
        }

        for i in (self.signal_period - 1)..n {
            // Calculate SMA of transaction volume
            let start = i + 1 - self.signal_period;
            let sum: f64 = tx_volumes[start..=i].iter().sum();
            let avg_tx_volume = sum / self.signal_period as f64;

            if avg_tx_volume > 0.0 {
                result[i] = market_caps[i] / avg_tx_volume;
            }
        }

        result
    }

    /// Get signal interpretation for a single NVT value.
    pub fn interpret(&self, nvt_value: f64) -> NVTSignal {
        if nvt_value.is_nan() {
            NVTSignal::Normal
        } else if nvt_value < self.lower_threshold {
            NVTSignal::Undervalued
        } else if nvt_value > self.upper_threshold {
            NVTSignal::Overvalued
        } else {
            NVTSignal::Normal
        }
    }

    /// Get signals for all NVT values.
    pub fn signals(&self, output: &NVTRatioOutput) -> Vec<NVTSignal> {
        output.nvt_signal.iter().map(|&v| self.interpret(v)).collect()
    }

    /// Convert NVT signal to trading signal.
    pub fn to_indicator_signal(&self, nvt_signal: NVTSignal) -> IndicatorSignal {
        match nvt_signal {
            NVTSignal::Undervalued => IndicatorSignal::Bullish,
            NVTSignal::Overvalued => IndicatorSignal::Bearish,
            NVTSignal::Normal => IndicatorSignal::Neutral,
        }
    }

    /// Get the signal period.
    pub fn signal_period(&self) -> usize {
        self.signal_period
    }

    /// Get the lower threshold.
    pub fn lower_threshold(&self) -> f64 {
        self.lower_threshold
    }

    /// Get the upper threshold.
    pub fn upper_threshold(&self) -> f64 {
        self.upper_threshold
    }
}

impl Default for NVTRatio {
    fn default() -> Self {
        Self::new(14, 65.0, 95.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvt_basic() {
        let nvt = NVTRatio::default();
        let market_caps = vec![1_000_000_000.0; 20]; // $1B market cap
        let tx_volumes = vec![10_000_000.0; 20]; // $10M daily tx volume

        let output = nvt.calculate(&market_caps, &tx_volumes);

        // NVT should be 100 (1B / 10M)
        assert!(!output.nvt.is_empty());
        assert!((output.nvt[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_nvt_signal_smoothing() {
        let nvt = NVTRatio::with_period(5);

        // Varying transaction volumes
        let market_caps = vec![1e9; 10];
        let tx_volumes = vec![10e6, 12e6, 8e6, 15e6, 10e6, 11e6, 9e6, 14e6, 10e6, 12e6];

        let output = nvt.calculate(&market_caps, &tx_volumes);

        // First 4 values should be NAN (period - 1)
        assert!(output.nvt_signal[0].is_nan());
        assert!(output.nvt_signal[3].is_nan());

        // 5th value should be valid
        assert!(!output.nvt_signal[4].is_nan());

        // NVT Signal should be smoothed (less volatile than raw NVT)
        let nvt_variance = variance(&output.nvt);
        let signal_variance = variance(&output.nvt_signal[4..].to_vec());
        assert!(signal_variance < nvt_variance);
    }

    #[test]
    fn test_nvt_interpretation() {
        let nvt = NVTRatio::new(14, 65.0, 95.0);

        assert_eq!(nvt.interpret(50.0), NVTSignal::Undervalued);
        assert_eq!(nvt.interpret(80.0), NVTSignal::Normal);
        assert_eq!(nvt.interpret(100.0), NVTSignal::Overvalued);
        assert_eq!(nvt.interpret(f64::NAN), NVTSignal::Normal);
    }

    #[test]
    fn test_nvt_zero_volume() {
        let nvt = NVTRatio::default();
        let market_caps = vec![1e9, 1e9, 1e9];
        let tx_volumes = vec![10e6, 0.0, 10e6];

        let output = nvt.calculate(&market_caps, &tx_volumes);

        assert!(!output.nvt[0].is_nan());
        assert!(output.nvt[1].is_nan()); // Zero volume should produce NaN
        assert!(!output.nvt[2].is_nan());
    }

    #[test]
    fn test_nvt_empty_input() {
        let nvt = NVTRatio::default();
        let output = nvt.calculate(&[], &[]);

        assert!(output.nvt.is_empty());
        assert!(output.nvt_signal.is_empty());
    }

    #[test]
    fn test_nvt_mismatched_lengths() {
        let nvt = NVTRatio::default();
        let market_caps = vec![1e9; 10];
        let tx_volumes = vec![10e6; 5];

        let output = nvt.calculate(&market_caps, &tx_volumes);

        // Should use minimum length
        assert_eq!(output.nvt.len(), 5);
        assert_eq!(output.nvt_signal.len(), 5);
    }

    #[test]
    fn test_nvt_indicator_signal_conversion() {
        let nvt = NVTRatio::default();

        assert_eq!(
            nvt.to_indicator_signal(NVTSignal::Undervalued),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            nvt.to_indicator_signal(NVTSignal::Overvalued),
            IndicatorSignal::Bearish
        );
        assert_eq!(
            nvt.to_indicator_signal(NVTSignal::Normal),
            IndicatorSignal::Neutral
        );
    }

    #[test]
    fn test_nvt_default() {
        let nvt = NVTRatio::default();
        assert_eq!(nvt.signal_period(), 14);
        assert_eq!(nvt.lower_threshold(), 65.0);
        assert_eq!(nvt.upper_threshold(), 95.0);
    }

    // Helper function for variance calculation
    fn variance(data: &[f64]) -> f64 {
        let valid: Vec<f64> = data.iter().filter(|x| !x.is_nan()).copied().collect();
        if valid.is_empty() {
            return 0.0;
        }
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64
    }
}
