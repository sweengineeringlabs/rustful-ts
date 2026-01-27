//! Kaufman Efficiency Ratio - IND-188
//!
//! The Efficiency Ratio (ER) measures price movement efficiency.
//! Developed by Perry Kaufman as part of his adaptive trading systems.
//!
//! Formula:
//! ER = |Price Change over N periods| / Sum of |Individual Price Changes|
//!
//! Range: 0 to 1
//! - ER close to 1: Strong trend (efficient price movement)
//! - ER close to 0: Choppy/ranging market (inefficient movement)
//!
//! Used as a component in KAMA and for trend detection.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Kaufman Efficiency Ratio - IND-188
///
/// Measures how efficiently price moves from point A to point B.
/// A high ER indicates trending market, low ER indicates ranging/choppy.
#[derive(Debug, Clone)]
pub struct EfficiencyRatio {
    period: usize,
}

impl EfficiencyRatio {
    /// Create new Efficiency Ratio indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period (typically 10)
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate Efficiency Ratio values.
    ///
    /// # Arguments
    /// * `data` - Close prices
    ///
    /// # Returns
    /// Vector of ER values (0 to 1)
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n <= self.period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period];

        for i in self.period..n {
            // Net price change over the period (direction)
            let change = (data[i] - data[i - self.period]).abs();

            // Sum of individual price changes (volatility)
            let mut volatility = 0.0;
            for j in (i - self.period + 1)..=i {
                volatility += (data[j] - data[j - 1]).abs();
            }

            // Efficiency Ratio
            let er = if volatility != 0.0 {
                change / volatility
            } else {
                0.0
            };

            result.push(er);
        }

        result
    }

    /// Classify trend strength based on ER value.
    ///
    /// - Strong trend: ER > 0.6
    /// - Moderate trend: ER > 0.3
    /// - Weak/ranging: ER <= 0.3
    pub fn trend_strength(er: f64) -> &'static str {
        if er.is_nan() {
            "unknown"
        } else if er > 0.6 {
            "strong"
        } else if er > 0.3 {
            "moderate"
        } else {
            "weak"
        }
    }
}

impl Default for EfficiencyRatio {
    fn default() -> Self {
        Self::new(10)
    }
}

impl TechnicalIndicator for EfficiencyRatio {
    fn name(&self) -> &str {
        "EfficiencyRatio"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() <= self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for EfficiencyRatio {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);

        if values.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let er_last = values[values.len() - 1];
        let er_prev = values[values.len() - 2];

        if er_last.is_nan() || er_prev.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal based on ER threshold and direction
        // Rising ER with strong value = bullish trend confirmed
        // Falling ER = trend weakening
        if er_last > 0.5 && er_last > er_prev {
            // Determine direction from price
            let n = data.close.len();
            if n >= 2 && data.close[n - 1] > data.close[n - 2] {
                Ok(IndicatorSignal::Bullish)
            } else {
                Ok(IndicatorSignal::Bearish)
            }
        } else if er_last < 0.3 {
            Ok(IndicatorSignal::Neutral) // Ranging market
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..values.len() {
            if values[i].is_nan() || values[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if values[i] > 0.5 && values[i] > values[i - 1] {
                // Determine direction from price
                if i < data.close.len() && data.close[i] > data.close[i - 1] {
                    signals.push(IndicatorSignal::Bullish);
                } else {
                    signals.push(IndicatorSignal::Bearish);
                }
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efficiency_ratio_perfect_trend() {
        let er = EfficiencyRatio::new(10);
        // Perfect uptrend: price increases by 1 each bar
        let data: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let result = er.calculate(&data);

        // After period, ER should be 1.0 (perfect trend)
        let last = result.last().unwrap();
        assert!((*last - 1.0).abs() < 1e-10, "Perfect trend should have ER = 1.0, got {}", last);
    }

    #[test]
    fn test_efficiency_ratio_ranging() {
        let er = EfficiencyRatio::new(10);
        // Ranging: oscillating around 100
        let data: Vec<f64> = (0..20)
            .map(|i| 100.0 + if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let result = er.calculate(&data);

        // ER should be low for ranging market
        let last = result.last().unwrap();
        assert!(*last < 0.5, "Ranging market should have low ER, got {}", last);
    }

    #[test]
    fn test_efficiency_ratio_bounds() {
        let er = EfficiencyRatio::new(10);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.5).sin() * 10.0).collect();
        let result = er.calculate(&data);

        // All valid ER values should be in [0, 1]
        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 1.0, "ER out of bounds: {}", val);
            }
        }
    }

    #[test]
    fn test_efficiency_ratio_nan_prefix() {
        let er = EfficiencyRatio::new(10);
        let data: Vec<f64> = (0..15).map(|i| 100.0 + i as f64).collect();
        let result = er.calculate(&data);

        // First `period` values should be NaN
        for i in 0..10 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }

        // Remaining should be valid
        for i in 10..15 {
            assert!(!result[i].is_nan(), "Expected valid value at index {}", i);
        }
    }

    #[test]
    fn test_efficiency_ratio_default() {
        let er = EfficiencyRatio::default();
        assert_eq!(er.period, 10);
    }

    #[test]
    fn test_trend_strength() {
        assert_eq!(EfficiencyRatio::trend_strength(0.7), "strong");
        assert_eq!(EfficiencyRatio::trend_strength(0.45), "moderate");
        assert_eq!(EfficiencyRatio::trend_strength(0.2), "weak");
        assert_eq!(EfficiencyRatio::trend_strength(f64::NAN), "unknown");
    }

    #[test]
    fn test_technical_indicator_trait() {
        let er = EfficiencyRatio::new(10);
        assert_eq!(er.name(), "EfficiencyRatio");
        assert_eq!(er.min_periods(), 11);
        assert_eq!(er.output_features(), 1);
    }
}
