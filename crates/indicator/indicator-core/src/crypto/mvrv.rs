//! MVRV Ratio (Market Value to Realized Value) - IND-086
//!
//! Compares market capitalization to realized capitalization for crypto valuation.
//!
//! Realized Cap = Sum of all UTXOs valued at the price when they last moved.
//! This represents the aggregate cost basis of all holders.
//!
//! Interpretation:
//! - MVRV > 3.7: Market top zone, overvalued
//! - MVRV 1.0-3.7: Normal range
//! - MVRV < 1.0: Market bottom zone, undervalued (holders in loss)

use indicator_spi::IndicatorSignal;

/// MVRV Ratio output.
#[derive(Debug, Clone)]
pub struct MVRVOutput {
    /// Raw MVRV ratio values.
    pub mvrv: Vec<f64>,
    /// Z-Score of MVRV (for trend analysis).
    pub mvrv_zscore: Vec<f64>,
}

/// MVRV signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MVRVSignal {
    /// MVRV below 1.0 - market bottom zone.
    ExtremeUndervalued,
    /// MVRV between 1.0 and 1.5 - undervalued.
    Undervalued,
    /// MVRV between 1.5 and 3.0 - fair value.
    FairValue,
    /// MVRV between 3.0 and 3.7 - overvalued.
    Overvalued,
    /// MVRV above 3.7 - market top zone.
    ExtremeOvervalued,
}

/// MVRV Ratio (Market Value to Realized Value) - IND-086
///
/// A fundamental on-chain valuation metric for cryptocurrencies.
///
/// # Formula
/// ```text
/// MVRV = Market Cap / Realized Cap
/// ```
///
/// Where Realized Cap is the sum of all coins valued at their last movement price.
///
/// # Example
/// ```
/// use indicator_core::crypto::MVRVRatio;
///
/// let mvrv = MVRVRatio::new(30);
/// let market_caps = vec![100e9, 110e9, 105e9];
/// let realized_caps = vec![50e9, 52e9, 51e9];
/// let output = mvrv.calculate(&market_caps, &realized_caps);
/// ```
#[derive(Debug, Clone)]
pub struct MVRVRatio {
    /// Period for Z-Score calculation.
    zscore_period: usize,
    /// Lower threshold (default: 1.0).
    lower_threshold: f64,
    /// Upper threshold (default: 3.7).
    upper_threshold: f64,
}

impl MVRVRatio {
    /// Create a new MVRV Ratio indicator.
    pub fn new(zscore_period: usize) -> Self {
        Self {
            zscore_period,
            lower_threshold: 1.0,
            upper_threshold: 3.7,
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(zscore_period: usize, lower: f64, upper: f64) -> Self {
        Self {
            zscore_period,
            lower_threshold: lower,
            upper_threshold: upper,
        }
    }

    /// Calculate MVRV ratio from market cap and realized cap series.
    pub fn calculate(&self, market_caps: &[f64], realized_caps: &[f64]) -> MVRVOutput {
        let n = market_caps.len().min(realized_caps.len());

        if n == 0 {
            return MVRVOutput {
                mvrv: vec![],
                mvrv_zscore: vec![],
            };
        }

        // Calculate raw MVRV
        let mvrv: Vec<f64> = (0..n)
            .map(|i| {
                if realized_caps[i] > 0.0 {
                    market_caps[i] / realized_caps[i]
                } else {
                    f64::NAN
                }
            })
            .collect();

        // Calculate Z-Score of MVRV
        let mvrv_zscore = self.calculate_zscore(&mvrv);

        MVRVOutput { mvrv, mvrv_zscore }
    }

    /// Calculate rolling Z-Score.
    fn calculate_zscore(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < self.zscore_period {
            return result;
        }

        for i in (self.zscore_period - 1)..n {
            let start = i + 1 - self.zscore_period;
            let window: Vec<f64> = data[start..=i]
                .iter()
                .filter(|x| !x.is_nan())
                .copied()
                .collect();

            if window.len() < 2 {
                continue;
            }

            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / window.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 && !data[i].is_nan() {
                result[i] = (data[i] - mean) / std_dev;
            }
        }

        result
    }

    /// Get signal interpretation for a single MVRV value.
    pub fn interpret(&self, mvrv_value: f64) -> MVRVSignal {
        if mvrv_value.is_nan() {
            MVRVSignal::FairValue
        } else if mvrv_value < self.lower_threshold {
            MVRVSignal::ExtremeUndervalued
        } else if mvrv_value < 1.5 {
            MVRVSignal::Undervalued
        } else if mvrv_value < 3.0 {
            MVRVSignal::FairValue
        } else if mvrv_value < self.upper_threshold {
            MVRVSignal::Overvalued
        } else {
            MVRVSignal::ExtremeOvervalued
        }
    }

    /// Convert MVRV signal to trading signal.
    pub fn to_indicator_signal(&self, mvrv_signal: MVRVSignal) -> IndicatorSignal {
        match mvrv_signal {
            MVRVSignal::ExtremeUndervalued => IndicatorSignal::Bullish,
            MVRVSignal::Undervalued => IndicatorSignal::Bullish,
            MVRVSignal::FairValue => IndicatorSignal::Neutral,
            MVRVSignal::Overvalued => IndicatorSignal::Bearish,
            MVRVSignal::ExtremeOvervalued => IndicatorSignal::Bearish,
        }
    }
}

impl Default for MVRVRatio {
    fn default() -> Self {
        Self::new(365) // Yearly Z-Score by default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mvrv_basic() {
        let mvrv = MVRVRatio::new(10);
        let market_caps = vec![100e9; 20];
        let realized_caps = vec![50e9; 20]; // MVRV = 2.0

        let output = mvrv.calculate(&market_caps, &realized_caps);

        assert_eq!(output.mvrv.len(), 20);
        assert!((output.mvrv[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mvrv_interpretation() {
        let mvrv = MVRVRatio::default();

        assert_eq!(mvrv.interpret(0.8), MVRVSignal::ExtremeUndervalued);
        assert_eq!(mvrv.interpret(1.2), MVRVSignal::Undervalued);
        assert_eq!(mvrv.interpret(2.0), MVRVSignal::FairValue);
        assert_eq!(mvrv.interpret(3.5), MVRVSignal::Overvalued);
        assert_eq!(mvrv.interpret(4.0), MVRVSignal::ExtremeOvervalued);
    }

    #[test]
    fn test_mvrv_zscore() {
        let mvrv = MVRVRatio::new(5);

        // Create varying MVRV data
        let market_caps: Vec<f64> = (0..20).map(|i| 100e9 + (i as f64 * 5e9)).collect();
        let realized_caps = vec![50e9; 20];

        let output = mvrv.calculate(&market_caps, &realized_caps);

        // Z-Score should be valid after warmup
        assert!(!output.mvrv_zscore[10].is_nan());
    }

    #[test]
    fn test_mvrv_zero_realized_cap() {
        let mvrv = MVRVRatio::new(5);
        let market_caps = vec![100e9, 100e9, 100e9];
        let realized_caps = vec![50e9, 0.0, 50e9];

        let output = mvrv.calculate(&market_caps, &realized_caps);

        assert!(!output.mvrv[0].is_nan());
        assert!(output.mvrv[1].is_nan());
        assert!(!output.mvrv[2].is_nan());
    }

    #[test]
    fn test_mvrv_empty_input() {
        let mvrv = MVRVRatio::default();
        let output = mvrv.calculate(&[], &[]);

        assert!(output.mvrv.is_empty());
        assert!(output.mvrv_zscore.is_empty());
    }

    #[test]
    fn test_mvrv_signal_conversion() {
        let mvrv = MVRVRatio::default();

        assert_eq!(
            mvrv.to_indicator_signal(MVRVSignal::ExtremeUndervalued),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            mvrv.to_indicator_signal(MVRVSignal::ExtremeOvervalued),
            IndicatorSignal::Bearish
        );
        assert_eq!(
            mvrv.to_indicator_signal(MVRVSignal::FairValue),
            IndicatorSignal::Neutral
        );
    }
}
