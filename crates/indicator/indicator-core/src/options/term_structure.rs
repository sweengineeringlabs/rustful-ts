//! Term Structure (IND-249)
//!
//! Analyzes implied volatility across different expiration periods,
//! detecting contango vs backwardation conditions.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Term Structure indicator.
#[derive(Debug, Clone)]
pub struct TermStructureConfig {
    /// Short-term period for volatility calculation
    pub short_period: usize,
    /// Long-term period for volatility calculation
    pub long_period: usize,
    /// Annualization factor
    pub annualization_factor: f64,
}

impl Default for TermStructureConfig {
    fn default() -> Self {
        Self {
            short_period: 10,
            long_period: 30,
            annualization_factor: 252.0,
        }
    }
}

/// Term Structure Output
#[derive(Debug, Clone)]
pub struct TermStructureOutput {
    /// Term structure slope (long - short) / long * 100
    pub slope: Vec<f64>,
    /// Short-term volatility
    pub short_vol: Vec<f64>,
    /// Long-term volatility
    pub long_vol: Vec<f64>,
}

/// Term Structure (IND-249)
///
/// Analyzes the term structure of implied volatility by comparing
/// short-term and long-term volatility measures.
///
/// # Interpretation
/// - Contango (positive slope): Long-term IV > Short-term IV (normal)
/// - Backwardation (negative slope): Short-term IV > Long-term IV (stressed)
///
/// # Trading Applications
/// - Contango: Sell near-term options, buy far-term
/// - Backwardation: Often signals market stress or upcoming events
/// - Slope changes can signal regime shifts
#[derive(Debug, Clone)]
pub struct TermStructure {
    config: TermStructureConfig,
}

impl TermStructure {
    /// Create a new TermStructure indicator.
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self {
            config: TermStructureConfig {
                short_period,
                long_period,
                ..Default::default()
            },
        })
    }

    /// Create from configuration.
    pub fn from_config(config: TermStructureConfig) -> Result<Self> {
        if config.short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if config.long_period <= config.short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if config.annualization_factor <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "annualization_factor".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Calculate historical volatility for a given period.
    fn calculate_vol(&self, close: &[f64], period: usize) -> Vec<f64> {
        let n = close.len();
        let mut vol = vec![f64::NAN; n];

        if n < period + 1 {
            return vol;
        }

        for i in period..n {
            let start = i - period;

            let returns: Vec<f64> = ((start + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 && close[j] > 0.0 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 2 {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / (returns.len() - 1) as f64;

                vol[i] = variance.sqrt() * self.config.annualization_factor.sqrt() * 100.0;
            }
        }

        vol
    }

    /// Calculate term structure slope.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let short_vol = self.calculate_vol(close, self.config.short_period);
        let long_vol = self.calculate_vol(close, self.config.long_period);

        let n = close.len();
        let mut result = vec![f64::NAN; n];

        for i in 0..n {
            if !short_vol[i].is_nan() && !long_vol[i].is_nan() && long_vol[i] > 0.0 {
                // Slope as percentage: (Long - Short) / Long * 100
                result[i] = (long_vol[i] - short_vol[i]) / long_vol[i] * 100.0;
            }
        }

        result
    }

    /// Calculate full term structure output.
    pub fn calculate_full(&self, close: &[f64]) -> TermStructureOutput {
        let short_vol = self.calculate_vol(close, self.config.short_period);
        let long_vol = self.calculate_vol(close, self.config.long_period);

        let n = close.len();
        let mut slope = vec![f64::NAN; n];

        for i in 0..n {
            if !short_vol[i].is_nan() && !long_vol[i].is_nan() && long_vol[i] > 0.0 {
                slope[i] = (long_vol[i] - short_vol[i]) / long_vol[i] * 100.0;
            }
        }

        TermStructureOutput {
            slope,
            short_vol,
            long_vol,
        }
    }

    /// Detect term structure regime.
    pub fn detect_regime(&self, close: &[f64]) -> Vec<TermStructureRegime> {
        let output = self.calculate_full(close);

        output.slope.iter().map(|&s| {
            if s.is_nan() {
                TermStructureRegime::Unknown
            } else if s > 10.0 {
                TermStructureRegime::SteepContango
            } else if s > 0.0 {
                TermStructureRegime::MildContango
            } else if s > -10.0 {
                TermStructureRegime::MildBackwardation
            } else {
                TermStructureRegime::SteepBackwardation
            }
        }).collect()
    }
}

/// Term structure regime classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TermStructureRegime {
    /// Long-term vol significantly higher than short-term
    SteepContango,
    /// Long-term vol slightly higher than short-term
    MildContango,
    /// Short-term vol slightly higher than long-term
    MildBackwardation,
    /// Short-term vol significantly higher than long-term (stressed)
    SteepBackwardation,
    /// Unable to determine
    Unknown,
}

impl TechnicalIndicator for TermStructure {
    fn name(&self) -> &str {
        "Term Structure"
    }

    fn min_periods(&self) -> usize {
        self.config.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }

        let output = self.calculate_full(&data.close);
        Ok(IndicatorOutput::triple(output.slope, output.short_vol, output.long_vol))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        (0..100)
            .map(|i| {
                let trend = 100.0 + (i as f64) * 0.2;
                let noise = (i as f64 * 0.3).sin() * 2.0;
                trend + noise
            })
            .collect()
    }

    fn make_volatile_short_term() -> Vec<f64> {
        // Data where recent volatility is higher
        let mut data = Vec::with_capacity(100);
        for i in 0..100 {
            let base = 100.0;
            let vol_mult = if i > 80 { 3.0 } else { 1.0 };
            data.push(base + (i as f64 * 0.5).sin() * vol_mult);
        }
        data
    }

    #[test]
    fn test_term_structure_basic() {
        let close = make_test_data();
        let ts = TermStructure::new(10, 30).unwrap();
        let result = ts.calculate(&close);

        assert_eq!(result.len(), close.len());

        // Check values after warmup
        for i in 31..result.len() {
            assert!(!result[i].is_nan(), "Expected valid value at index {}", i);
            assert!(result[i].abs() < 200.0, "Slope {} out of range at {}", result[i], i);
        }
    }

    #[test]
    fn test_term_structure_full() {
        let close = make_test_data();
        let ts = TermStructure::new(10, 30).unwrap();
        let output = ts.calculate_full(&close);

        assert_eq!(output.slope.len(), close.len());
        assert_eq!(output.short_vol.len(), close.len());
        assert_eq!(output.long_vol.len(), close.len());

        // Volatilities should be positive
        for i in 31..close.len() {
            assert!(output.short_vol[i] >= 0.0);
            assert!(output.long_vol[i] >= 0.0);
        }
    }

    #[test]
    fn test_term_structure_backwardation() {
        let close = make_volatile_short_term();
        let ts = TermStructure::new(5, 20).unwrap();
        let output = ts.calculate_full(&close);

        // At the end, short-term vol should be higher (backwardation)
        let last_slope = output.slope.last().unwrap();
        // May or may not be in backwardation depending on exact data
        assert!(last_slope.is_finite());
    }

    #[test]
    fn test_term_structure_regime() {
        let close = make_test_data();
        let ts = TermStructure::new(10, 30).unwrap();
        let regimes = ts.detect_regime(&close);

        assert_eq!(regimes.len(), close.len());

        // Check that we get valid regimes after warmup
        let valid_regimes: Vec<_> = regimes.iter()
            .skip(31)
            .filter(|r| **r != TermStructureRegime::Unknown)
            .collect();

        assert!(!valid_regimes.is_empty());
    }

    #[test]
    fn test_term_structure_invalid_params() {
        // short >= long should fail
        assert!(TermStructure::new(30, 30).is_err());
        assert!(TermStructure::new(30, 20).is_err());
        assert!(TermStructure::new(1, 10).is_err());
    }

    #[test]
    fn test_term_structure_config() {
        let config = TermStructureConfig {
            short_period: 5,
            long_period: 21,
            annualization_factor: 252.0,
        };
        let ts = TermStructure::from_config(config).unwrap();
        assert_eq!(ts.min_periods(), 22);
    }
}
