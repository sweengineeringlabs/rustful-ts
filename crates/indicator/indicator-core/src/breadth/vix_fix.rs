//! VIX Fix - Synthetic VIX from Price (IND-396)

use crate::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// VIX Fix Configuration
#[derive(Debug, Clone)]
pub struct VIXFixConfig {
    /// Lookback period for highest close (default: 22)
    pub period: usize,
    /// Bollinger Band period for bands (default: 20)
    pub bb_period: usize,
    /// Bollinger Band standard deviation multiplier (default: 2.0)
    pub bb_mult: f64,
    /// High threshold for extreme readings (default: 0.5)
    pub high_threshold: f64,
    /// Low threshold (default: 0.15)
    pub low_threshold: f64,
}

impl Default for VIXFixConfig {
    fn default() -> Self {
        Self {
            period: 22,
            bb_period: 20,
            bb_mult: 2.0,
            high_threshold: 0.5,
            low_threshold: 0.15,
        }
    }
}

/// VIX Fix (Williams VIX Fix)
///
/// A synthetic volatility indicator created by Larry Williams that mimics
/// the behavior of the VIX using only price data. It identifies market
/// bottoms by measuring the distance from recent highs.
///
/// # Formula
/// VIX Fix = (Highest Close over N periods - Low) / Highest Close over N periods * 100
///
/// # Interpretation
/// - High values (> 0.5): Potential market bottom, extreme fear
/// - Low values (< 0.15): Complacency, potential top
/// - Spikes indicate panic selling and potential reversal points
///
/// # Use Cases
/// - Identifying market bottoms
/// - Measuring implied volatility from price
/// - Contrarian buy signals when VIX Fix spikes
#[derive(Debug, Clone)]
pub struct VIXFix {
    config: VIXFixConfig,
}

impl Default for VIXFix {
    fn default() -> Self {
        Self::new()
    }
}

impl VIXFix {
    pub fn new() -> Self {
        Self {
            config: VIXFixConfig::default(),
        }
    }

    pub fn with_config(config: VIXFixConfig) -> Self {
        Self { config }
    }

    pub fn with_period(mut self, period: usize) -> Self {
        self.config.period = period;
        self
    }

    pub fn with_bb_settings(mut self, period: usize, mult: f64) -> Self {
        self.config.bb_period = period;
        self.config.bb_mult = mult;
        self
    }

    /// Calculate highest value over a lookback period
    fn highest(&self, data: &[f64], period: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            if i < period - 1 {
                result.push(f64::NAN);
            } else {
                let start = i + 1 - period;
                let max = data[start..=i]
                    .iter()
                    .filter(|v| !v.is_nan())
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                result.push(if max == f64::NEG_INFINITY { f64::NAN } else { max });
            }
        }

        result
    }

    /// Calculate SMA
    fn sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            if i < period - 1 {
                result.push(f64::NAN);
            } else {
                let start = i + 1 - period;
                let valid: Vec<f64> = data[start..=i]
                    .iter()
                    .filter(|v| !v.is_nan())
                    .copied()
                    .collect();
                if valid.is_empty() {
                    result.push(f64::NAN);
                } else {
                    result.push(valid.iter().sum::<f64>() / valid.len() as f64);
                }
            }
        }

        result
    }

    /// Calculate standard deviation
    fn std_dev(&self, data: &[f64], sma: &[f64], period: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            if i < period - 1 || sma[i].is_nan() {
                result.push(f64::NAN);
            } else {
                let start = i + 1 - period;
                let mean = sma[i];
                let valid: Vec<f64> = data[start..=i]
                    .iter()
                    .filter(|v| !v.is_nan())
                    .copied()
                    .collect();
                if valid.is_empty() {
                    result.push(f64::NAN);
                } else {
                    let variance: f64 = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / valid.len() as f64;
                    result.push(variance.sqrt());
                }
            }
        }

        result
    }

    /// Calculate VIX Fix values
    pub fn calculate(&self, data: &OHLCVSeries) -> Vec<f64> {
        if data.close.len() < self.config.period {
            return vec![f64::NAN; data.close.len()];
        }

        let highest_close = self.highest(&data.close, self.config.period);

        data.low
            .iter()
            .zip(highest_close.iter())
            .map(|(low, hc)| {
                if hc.is_nan() || *hc == 0.0 {
                    f64::NAN
                } else {
                    (hc - low) / hc * 100.0
                }
            })
            .collect()
    }

    /// Calculate VIX Fix with Bollinger Bands
    pub fn calculate_with_bands(&self, data: &OHLCVSeries) -> VIXFixOutput {
        let vix_fix = self.calculate(data);
        let sma = self.sma(&vix_fix, self.config.bb_period);
        let std = self.std_dev(&vix_fix, &sma, self.config.bb_period);

        let upper_band: Vec<f64> = sma
            .iter()
            .zip(std.iter())
            .map(|(s, sd)| {
                if s.is_nan() || sd.is_nan() {
                    f64::NAN
                } else {
                    s + self.config.bb_mult * sd
                }
            })
            .collect();

        let lower_band: Vec<f64> = sma
            .iter()
            .zip(std.iter())
            .map(|(s, sd)| {
                if s.is_nan() || sd.is_nan() {
                    f64::NAN
                } else {
                    s - self.config.bb_mult * sd
                }
            })
            .collect();

        VIXFixOutput {
            vix_fix,
            middle_band: sma,
            upper_band,
            lower_band,
        }
    }

    /// Interpret VIX Fix value
    pub fn interpret(&self, value: f64) -> VIXFixSignal {
        if value.is_nan() {
            VIXFixSignal::Unknown
        } else if value >= self.config.high_threshold * 100.0 {
            VIXFixSignal::ExtremeFear
        } else if value >= 0.3 * 100.0 {
            VIXFixSignal::HighFear
        } else if value <= self.config.low_threshold * 100.0 {
            VIXFixSignal::Complacency
        } else {
            VIXFixSignal::Neutral
        }
    }
}

/// VIX Fix output with bands
#[derive(Debug, Clone)]
pub struct VIXFixOutput {
    /// VIX Fix values
    pub vix_fix: Vec<f64>,
    /// Middle band (SMA)
    pub middle_band: Vec<f64>,
    /// Upper Bollinger Band
    pub upper_band: Vec<f64>,
    /// Lower Bollinger Band
    pub lower_band: Vec<f64>,
}

/// VIX Fix signal interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VIXFixSignal {
    /// Very high VIX Fix: Extreme fear, potential bottom
    ExtremeFear,
    /// High VIX Fix: Elevated fear
    HighFear,
    /// Normal range
    Neutral,
    /// Very low VIX Fix: Complacency, potential top
    Complacency,
    /// Invalid data
    Unknown,
}

impl TechnicalIndicator for VIXFix {
    fn name(&self) -> &str {
        "VIX Fix"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.len() < self.config.period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period,
                got: data.len(),
            });
        }

        let values = self.calculate(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.config.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> OHLCVSeries {
        OHLCVSeries {
            open: vec![
                100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 104.0, 106.0,
                105.0, 107.0, 106.0, 108.0, 107.0, 109.0, 108.0, 110.0, 109.0, 111.0,
                110.0, 108.0, 105.0, 100.0, 95.0,
            ],
            high: vec![
                101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 104.0, 106.0, 105.0, 107.0,
                106.0, 108.0, 107.0, 109.0, 108.0, 110.0, 109.0, 111.0, 110.0, 112.0,
                111.0, 109.0, 106.0, 101.0, 96.0,
            ],
            low: vec![
                99.0, 101.0, 100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0,
                104.0, 106.0, 105.0, 107.0, 106.0, 108.0, 107.0, 109.0, 108.0, 110.0,
                109.0, 105.0, 100.0, 90.0, 85.0,
            ],
            close: vec![
                100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 104.0, 106.0,
                105.0, 107.0, 106.0, 108.0, 107.0, 109.0, 108.0, 110.0, 109.0, 111.0,
                110.0, 106.0, 102.0, 92.0, 88.0,
            ],
            volume: vec![1000.0; 25],
        }
    }

    #[test]
    fn test_vix_fix_basic() {
        let vix_fix = VIXFix::new();
        let data = create_test_data();
        let result = vix_fix.calculate(&data);

        assert_eq!(result.len(), 25);
        // First 21 values should be NaN (period - 1)
        for i in 0..21 {
            assert!(result[i].is_nan());
        }
        // Values after should be valid
        assert!(!result[21].is_nan());
        // Last values should show higher VIX Fix due to selloff
        assert!(result[24] > result[21]);
    }

    #[test]
    fn test_vix_fix_formula() {
        // Create simple test case
        let mut config = VIXFixConfig::default();
        config.period = 3;
        let vix_fix = VIXFix::with_config(config);

        let data = OHLCVSeries {
            open: vec![100.0, 105.0, 110.0, 105.0],
            high: vec![102.0, 108.0, 112.0, 108.0],
            low: vec![98.0, 103.0, 108.0, 95.0],
            close: vec![100.0, 105.0, 110.0, 100.0],
            volume: vec![1000.0; 4],
        };

        let result = vix_fix.calculate(&data);
        assert_eq!(result.len(), 4);

        // At index 3: highest close over 3 periods = 110, low = 95
        // VIX Fix = (110 - 95) / 110 * 100 = 15/110 * 100 = 13.636...
        assert!(!result[3].is_nan());
        let expected = (110.0 - 95.0) / 110.0 * 100.0;
        assert!((result[3] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_vix_fix_interpretation() {
        let vix_fix = VIXFix::new();

        assert_eq!(vix_fix.interpret(60.0), VIXFixSignal::ExtremeFear);
        assert_eq!(vix_fix.interpret(35.0), VIXFixSignal::HighFear);
        assert_eq!(vix_fix.interpret(20.0), VIXFixSignal::Neutral);
        assert_eq!(vix_fix.interpret(10.0), VIXFixSignal::Complacency);
        assert_eq!(vix_fix.interpret(f64::NAN), VIXFixSignal::Unknown);
    }

    #[test]
    fn test_vix_fix_with_bands() {
        let mut config = VIXFixConfig::default();
        config.period = 5;
        config.bb_period = 3;
        let vix_fix = VIXFix::with_config(config);

        let data = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![105.0; 10],
            low: vec![95.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0, 87.0, 86.0],
            close: vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            volume: vec![1000.0; 10],
        };

        let output = vix_fix.calculate_with_bands(&data);
        assert_eq!(output.vix_fix.len(), 10);
        assert_eq!(output.middle_band.len(), 10);
        assert_eq!(output.upper_band.len(), 10);
        assert_eq!(output.lower_band.len(), 10);
    }

    #[test]
    fn test_technical_indicator_trait() {
        let mut config = VIXFixConfig::default();
        config.period = 5;
        let vix_fix = VIXFix::with_config(config);

        let data = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![105.0; 10],
            low: vec![95.0; 10],
            close: vec![100.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = vix_fix.compute(&data);
        assert!(result.is_ok());

        assert_eq!(vix_fix.min_periods(), 5);
        assert_eq!(vix_fix.name(), "VIX Fix");
    }

    #[test]
    fn test_insufficient_data() {
        let vix_fix = VIXFix::new(); // default period 22
        let data = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![105.0; 10],
            low: vec![95.0; 10],
            close: vec![100.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = vix_fix.compute(&data);
        assert!(result.is_err());
    }
}
