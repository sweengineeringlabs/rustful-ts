//! VIX Term Structure Indicators
//!
//! Indicators for analyzing volatility term structure (contango/backwardation).

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// VIX Term Structure (IND-285) - Contango/Backwardation analysis
///
/// This indicator creates a proxy for VIX term structure using
/// realized volatility at different timeframes to simulate the
/// relationship between near-term and longer-term volatility.
#[derive(Debug, Clone)]
pub struct VIXTermStructure {
    short_period: usize,
    long_period: usize,
    smoothing: usize,
}

/// Configuration for VIXTermStructure
#[derive(Debug, Clone)]
pub struct VIXTermStructureConfig {
    pub short_period: usize,
    pub long_period: usize,
    pub smoothing: usize,
}

impl Default for VIXTermStructureConfig {
    fn default() -> Self {
        Self {
            short_period: 10,   // Near-term volatility (like VIX)
            long_period: 30,    // Longer-term volatility (like VIX3M)
            smoothing: 5,
        }
    }
}

impl VIXTermStructure {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        Self::with_config(VIXTermStructureConfig {
            short_period,
            long_period,
            ..Default::default()
        })
    }

    pub fn with_config(config: VIXTermStructureConfig) -> Result<Self> {
        if config.short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if config.long_period <= config.short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if config.smoothing < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            short_period: config.short_period,
            long_period: config.long_period,
            smoothing: config.smoothing,
        })
    }

    /// Calculate realized volatility for a given period (annualized)
    fn realized_volatility(close: &[f64], start: usize, end: usize) -> f64 {
        if end <= start + 1 || end > close.len() {
            return 0.0;
        }

        let returns: Vec<f64> = ((start + 1)..=end)
            .filter_map(|j| {
                if close[j - 1] > 0.0 {
                    Some((close[j] / close[j - 1]).ln())
                } else {
                    None
                }
            })
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt() * (252.0_f64).sqrt() * 100.0
    }

    /// Calculate term structure ratio
    ///
    /// Ratio > 1.0 = Contango (normal, complacent market)
    /// Ratio < 1.0 = Backwardation (stressed, fearful market)
    pub fn calculate_ratio(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![1.0; n];

        for i in self.long_period..n {
            let short_vol = Self::realized_volatility(close, i - self.short_period, i);
            let long_vol = Self::realized_volatility(close, i - self.long_period, i);

            if short_vol > 0.0 {
                // Ratio of long/short vol (simulates VIX3M/VIX)
                result[i] = (long_vol / short_vol).clamp(0.5, 2.0);
            }
        }
        result
    }

    /// Calculate contango percentage
    ///
    /// Positive = Contango, Negative = Backwardation
    pub fn calculate_contango(&self, close: &[f64]) -> Vec<f64> {
        let ratio = self.calculate_ratio(close);
        let n = ratio.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            result[i] = (ratio[i] - 1.0) * 100.0;
        }
        result
    }

    /// Calculate term structure sentiment (-100 to 100)
    ///
    /// Positive = Contango (bullish bias), Negative = Backwardation (bearish/fearful)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let contango = self.calculate_contango(close);
        let n = contango.len();
        let mut raw = vec![0.0; n];
        let mut result = vec![0.0; n];

        // Convert contango to sentiment scale
        for i in 0..n {
            // Map contango range to sentiment
            // Typical contango: 5-15%, backwardation: -5 to -30%
            raw[i] = (contango[i] * 5.0).clamp(-100.0, 100.0);
        }

        // Apply smoothing
        for i in self.smoothing..n {
            let sum: f64 = raw[(i - self.smoothing + 1)..=i].iter().sum();
            result[i] = (sum / self.smoothing as f64).clamp(-100.0, 100.0);
        }

        result
    }

    /// Detect term structure regime
    ///
    /// Returns: 1 = Strong Contango, 0 = Normal, -1 = Backwardation
    pub fn calculate_regime(&self, close: &[f64]) -> Vec<f64> {
        let ratio = self.calculate_ratio(close);
        let n = ratio.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            if ratio[i] > 1.1 {
                result[i] = 1.0;  // Strong contango
            } else if ratio[i] < 0.95 {
                result[i] = -1.0; // Backwardation
            } else {
                result[i] = 0.0;  // Normal
            }
        }
        result
    }
}

impl TechnicalIndicator for VIXTermStructure {
    fn name(&self) -> &str {
        "VIX Term Structure"
    }

    fn min_periods(&self) -> usize {
        self.long_period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let ratio = self.calculate_ratio(&data.close);
        let contango = self.calculate_contango(&data.close);
        let sentiment = self.calculate(&data.close);
        let regime = self.calculate_regime(&data.close);

        Ok(IndicatorOutput::triple(ratio, contango, sentiment))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        // Create volatile data with some trends
        let mut close = Vec::with_capacity(60);
        for i in 0..60 {
            let trend = 100.0 + (i as f64 * 0.5);
            let noise = ((i as f64 * 0.5).sin() * 2.0);
            close.push(trend + noise);
        }
        close
    }

    fn make_stressed_data() -> Vec<f64> {
        // Create data with increasing volatility (stress scenario)
        let mut close = Vec::with_capacity(60);
        for i in 0..60 {
            let base = 100.0;
            let volatility_factor = 1.0 + (i as f64 / 30.0);
            let noise = ((i as f64 * 0.7).sin() * 3.0 * volatility_factor);
            close.push(base + noise);
        }
        close
    }

    #[test]
    fn test_vix_term_structure_creation() {
        let indicator = VIXTermStructure::new(10, 30);
        assert!(indicator.is_ok());

        let indicator = VIXTermStructure::new(3, 30);
        assert!(indicator.is_err());

        let indicator = VIXTermStructure::new(30, 10);
        assert!(indicator.is_err());
    }

    #[test]
    fn test_vix_term_structure_ratio() {
        let close = make_test_data();
        let indicator = VIXTermStructure::new(10, 30).unwrap();
        let result = indicator.calculate_ratio(&close);

        assert_eq!(result.len(), close.len());
        for i in 31..result.len() {
            assert!(result[i] >= 0.5 && result[i] <= 2.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_vix_term_structure_contango() {
        let close = make_test_data();
        let indicator = VIXTermStructure::new(10, 30).unwrap();
        let result = indicator.calculate_contango(&close);

        assert_eq!(result.len(), close.len());
        // Contango can be positive or negative
    }

    #[test]
    fn test_vix_term_structure_sentiment() {
        let close = make_test_data();
        let indicator = VIXTermStructure::new(10, 30).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), close.len());
        for i in 36..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_vix_term_structure_regime() {
        let close = make_test_data();
        let indicator = VIXTermStructure::new(10, 30).unwrap();
        let result = indicator.calculate_regime(&close);

        assert_eq!(result.len(), close.len());
        for i in 31..result.len() {
            assert!(result[i] == -1.0 || result[i] == 0.0 || result[i] == 1.0,
                    "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_vix_term_structure_with_config() {
        let config = VIXTermStructureConfig {
            short_period: 10,
            long_period: 30,
            smoothing: 3,
        };
        let indicator = VIXTermStructure::with_config(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_vix_term_structure_min_periods() {
        let indicator = VIXTermStructure::new(10, 30).unwrap();
        assert_eq!(indicator.min_periods(), 35);
    }
}
