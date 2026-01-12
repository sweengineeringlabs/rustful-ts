//! McClellan Summation Index indicator.

use super::{BreadthIndicator, BreadthSeries, McClellanOscillator};
use crate::{IndicatorError, IndicatorOutput, Result};

/// McClellan Summation Index
///
/// A cumulative version of the McClellan Oscillator that provides a
/// longer-term view of market breadth momentum. It's the running total
/// of McClellan Oscillator values.
///
/// # Formula
/// McClellan Summation Index = Cumulative Sum of McClellan Oscillator
///
/// Or equivalently:
/// MSI = Previous MSI + Current McClellan Oscillator
///
/// # Interpretation
/// - Above +1000: Strongly overbought
/// - Below -1000: Strongly oversold
/// - Zero line crossovers: Major trend changes
/// - Divergence from price index: Long-term trend reversal signal
/// - Rising from deeply oversold: Potential bullish reversal
#[derive(Debug, Clone)]
pub struct McClellanSummationIndex {
    /// Underlying McClellan Oscillator
    oscillator: McClellanOscillator,
    /// Starting value for the summation (default: 0)
    start_value: f64,
}

impl Default for McClellanSummationIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl McClellanSummationIndex {
    pub fn new() -> Self {
        Self {
            oscillator: McClellanOscillator::new(),
            start_value: 0.0,
        }
    }

    pub fn with_oscillator(oscillator: McClellanOscillator) -> Self {
        Self {
            oscillator,
            start_value: 0.0,
        }
    }

    pub fn with_start_value(mut self, start_value: f64) -> Self {
        self.start_value = start_value;
        self
    }

    /// Calculate McClellan Summation Index from BreadthSeries
    pub fn calculate(&self, data: &BreadthSeries) -> Vec<f64> {
        let oscillator_values = self.oscillator.calculate(data);

        let mut result = Vec::with_capacity(oscillator_values.len());
        let mut cumulative = self.start_value;

        for value in oscillator_values {
            if value.is_nan() {
                result.push(f64::NAN);
            } else {
                cumulative += value;
                result.push(cumulative);
            }
        }

        result
    }

    /// Calculate and return both summation index and oscillator values
    pub fn calculate_with_oscillator(&self, data: &BreadthSeries) -> (Vec<f64>, Vec<f64>) {
        let oscillator_values = self.oscillator.calculate(data);

        let mut summation = Vec::with_capacity(oscillator_values.len());
        let mut cumulative = self.start_value;

        for &value in &oscillator_values {
            if value.is_nan() {
                summation.push(f64::NAN);
            } else {
                cumulative += value;
                summation.push(cumulative);
            }
        }

        (summation, oscillator_values)
    }

    /// Get the minimum periods required (from underlying oscillator)
    pub fn required_periods(&self) -> usize {
        self.oscillator.min_periods()
    }
}

impl BreadthIndicator for McClellanSummationIndex {
    fn name(&self) -> &str {
        "McClellan Summation Index"
    }

    fn compute_breadth(&self, data: &BreadthSeries) -> Result<IndicatorOutput> {
        let required = self.oscillator.min_periods();
        if data.len() < required {
            return Err(IndicatorError::InsufficientData {
                required,
                got: data.len(),
            });
        }

        let values = self.calculate(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.oscillator.min_periods()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::breadth::BreadthData;

    fn create_test_series(len: usize) -> BreadthSeries {
        let mut series = BreadthSeries::new();
        for i in 0..len {
            // Create data that trends up then down
            let phase = i as f64 / 10.0;
            let advances = 1500.0 + phase.sin() * 300.0;
            let declines = 1400.0 - phase.sin() * 200.0;
            series.push(BreadthData::from_ad(advances, declines));
        }
        series
    }

    #[test]
    fn test_mcclellan_sum_basic() {
        let msi = McClellanSummationIndex::new();
        let series = create_test_series(50);
        let result = msi.calculate(&series);

        assert_eq!(result.len(), 50);

        // First values should be NaN until oscillator has enough data
        for i in 0..38 {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }

        // After enough data, values should be valid
        assert!(!result[39].is_nan());
    }

    #[test]
    fn test_mcclellan_sum_with_start_value() {
        let msi = McClellanSummationIndex::new().with_start_value(1000.0);
        let series = create_test_series(50);
        let result = msi.calculate(&series);

        // First valid value should include the start value
        let first_valid = result.iter().find(|v| !v.is_nan()).unwrap();
        assert!(*first_valid != 0.0); // Should be non-zero due to start value
    }

    #[test]
    fn test_mcclellan_sum_with_oscillator() {
        let msi = McClellanSummationIndex::new();
        let series = create_test_series(50);
        let (summation, oscillator) = msi.calculate_with_oscillator(&series);

        assert_eq!(summation.len(), 50);
        assert_eq!(oscillator.len(), 50);

        // Verify summation is cumulative sum of oscillator
        let mut running_sum = 0.0;
        for i in 0..50 {
            if !oscillator[i].is_nan() {
                running_sum += oscillator[i];
                assert!(
                    (summation[i] - running_sum).abs() < 1e-10,
                    "Mismatch at index {}: {} vs {}",
                    i,
                    summation[i],
                    running_sum
                );
            }
        }
    }

    #[test]
    fn test_mcclellan_sum_cumulative_nature() {
        let msi = McClellanSummationIndex::new();
        let series = create_test_series(50);
        let result = msi.calculate(&series);

        // Values should show cumulative behavior
        // Later values should be larger in absolute terms (generally)
        let first_valid_idx = result.iter().position(|v| !v.is_nan()).unwrap();

        // The summation should generally grow or shrink over time
        // (not stay constant like a simple oscillator)
        let first_valid = result[first_valid_idx];
        let last_valid = result.last().unwrap();

        // They should be different (cumulative effect)
        assert!((first_valid - last_valid).abs() > 0.1);
    }

    #[test]
    fn test_mcclellan_sum_insufficient_data() {
        let msi = McClellanSummationIndex::new();
        let series = create_test_series(30);
        let result = msi.compute_breadth(&series);

        assert!(result.is_err());
    }
}
