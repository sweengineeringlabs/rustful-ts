//! Roofing Filter indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use std::f64::consts::PI;

/// Ehlers Roofing Filter
///
/// The Roofing Filter combines a high-pass filter with a supersmoother filter
/// to create a bandpass filter. It removes both the low-frequency trend component
/// and the high-frequency noise, leaving only the cycle component.
///
/// The output oscillates around zero and can be used to identify cycles.
#[derive(Debug, Clone)]
pub struct RoofingFilter {
    /// High-pass filter period (default: 48)
    pub hp_period: usize,
    /// Supersmoother period (default: 10)
    pub smooth_period: usize,
}

impl RoofingFilter {
    pub fn new(hp_period: usize, smooth_period: usize) -> Self {
        Self { hp_period, smooth_period }
    }

    /// Calculate Roofing Filter
    /// Returns (roofing, trigger)
    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < 3 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut roofing = vec![f64::NAN; n];
        let mut trigger = vec![f64::NAN; n];
        let mut highpass = vec![0.0; n];

        // High-pass filter coefficients
        let hp_alpha = (0.707 * 2.0 * PI / self.hp_period as f64).cos();
        let hp_coef = (1.0 + (2.0 * PI / self.hp_period as f64).sin()) / hp_alpha;
        let alpha1 = hp_coef - (hp_coef * hp_coef - 1.0).sqrt();

        // Supersmoother coefficients (2-pole Butterworth)
        let a1 = (-1.414 * PI / self.smooth_period as f64).exp();
        let b1 = 2.0 * a1 * (1.414 * PI / self.smooth_period as f64).cos();
        let c2 = b1;
        let c3 = -a1 * a1;
        let c1 = 1.0 - c2 - c3;

        // Apply high-pass filter first
        for i in 2..n {
            highpass[i] = (1.0 - alpha1 / 2.0) * (1.0 - alpha1 / 2.0)
                * (data[i] - 2.0 * data[i - 1] + data[i - 2])
                + 2.0 * (1.0 - alpha1) * highpass[i - 1]
                - (1.0 - alpha1) * (1.0 - alpha1) * highpass[i - 2];
        }

        // Apply supersmoother (2-pole Butterworth low-pass)
        for i in 2..n {
            roofing[i] = c1 * (highpass[i] + highpass[i - 1]) / 2.0
                + c2 * roofing[i - 1].max(0.0)
                + c3 * roofing[i - 2].max(0.0);

            // Handle NaN initialization
            if roofing[i - 1].is_nan() {
                roofing[i] = c1 * (highpass[i] + highpass[i - 1]) / 2.0;
            } else if roofing[i - 2].is_nan() {
                roofing[i] = c1 * (highpass[i] + highpass[i - 1]) / 2.0
                    + c2 * roofing[i - 1];
            }
        }

        // Trigger is the previous roofing value
        for i in 3..n {
            trigger[i] = roofing[i - 1];
        }

        (roofing, trigger)
    }

    /// Calculate just the high-pass component
    pub fn calculate_highpass(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 3 {
            return vec![f64::NAN; n];
        }

        let mut highpass = vec![f64::NAN; n];

        // High-pass filter coefficients
        let hp_alpha = (0.707 * 2.0 * PI / self.hp_period as f64).cos();
        let hp_coef = (1.0 + (2.0 * PI / self.hp_period as f64).sin()) / hp_alpha;
        let alpha1 = hp_coef - (hp_coef * hp_coef - 1.0).sqrt();

        highpass[0] = 0.0;
        highpass[1] = 0.0;

        for i in 2..n {
            highpass[i] = (1.0 - alpha1 / 2.0) * (1.0 - alpha1 / 2.0)
                * (data[i] - 2.0 * data[i - 1] + data[i - 2])
                + 2.0 * (1.0 - alpha1) * highpass[i - 1]
                - (1.0 - alpha1) * (1.0 - alpha1) * highpass[i - 2];
        }

        highpass
    }
}

impl Default for RoofingFilter {
    fn default() -> Self {
        Self::new(48, 10)
    }
}

impl TechnicalIndicator for RoofingFilter {
    fn name(&self) -> &str {
        "RoofingFilter"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 3 {
            return Err(IndicatorError::InsufficientData {
                required: 3,
                got: data.close.len(),
            });
        }

        let (roofing, trigger) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(roofing, trigger))
    }

    fn min_periods(&self) -> usize {
        3
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for RoofingFilter {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (roofing, trigger) = self.calculate(&data.close);
        let n = roofing.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let roofing_last = roofing[n - 1];
        let trigger_last = trigger[n - 1];
        let roofing_prev = roofing[n - 2];
        let trigger_prev = trigger[n - 2];

        if roofing_last.is_nan() || trigger_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Crossover signals
        if roofing_last > trigger_last && roofing_prev <= trigger_prev {
            Ok(IndicatorSignal::Bullish)
        } else if roofing_last < trigger_last && roofing_prev >= trigger_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (roofing, trigger) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..roofing.len().min(trigger.len()) {
            if roofing[i].is_nan() || trigger[i].is_nan() || roofing[i - 1].is_nan() || trigger[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if roofing[i] > trigger[i] && roofing[i - 1] <= trigger[i - 1] {
                signals.push(IndicatorSignal::Bullish);
            } else if roofing[i] < trigger[i] && roofing[i - 1] >= trigger[i - 1] {
                signals.push(IndicatorSignal::Bearish);
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
    fn test_roofing_filter_basic() {
        let rf = RoofingFilter::default();
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.2).sin() * 10.0).collect();

        let (roofing, trigger) = rf.calculate(&data);

        assert_eq!(roofing.len(), n);
        assert_eq!(trigger.len(), n);

        // First 2 values should be NAN
        assert!(roofing[0].is_nan());
        assert!(roofing[1].is_nan());

        // Values after warmup should be valid
        assert!(!roofing[50].is_nan());
        assert!(!trigger[50].is_nan());
    }

    #[test]
    fn test_roofing_filter_removes_trend() {
        let rf = RoofingFilter::new(20, 5);

        // Linear trend (should be removed by high-pass)
        let n = 100;
        let trend: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 2.0).collect();

        let (roofing, _) = rf.calculate(&trend);

        // After convergence, the roofing filter should produce near-zero values
        // for a pure trend (no cycle component)
        let avg: f64 = roofing[50..n].iter()
            .filter(|x| !x.is_nan())
            .sum::<f64>() / 50.0;

        assert!(avg.abs() < 10.0,
            "Roofing filter should remove trend, got avg = {}", avg);
    }

    #[test]
    fn test_roofing_filter_oscillation() {
        let rf = RoofingFilter::default();
        let n = 200;

        // Generate cyclic signal with trend
        let data: Vec<f64> = (0..n).map(|i| {
            100.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin() * 20.0
        }).collect();

        let (roofing, _) = rf.calculate(&data);

        // Roofing filter should oscillate around zero
        let mut positive = 0;
        let mut negative = 0;

        for i in 50..n {
            if !roofing[i].is_nan() {
                if roofing[i] > 0.0 {
                    positive += 1;
                } else if roofing[i] < 0.0 {
                    negative += 1;
                }
            }
        }

        // Should have both positive and negative values
        assert!(positive > 10 && negative > 10,
            "Roofing should oscillate: positive={}, negative={}", positive, negative);
    }

    #[test]
    fn test_roofing_filter_highpass() {
        let rf = RoofingFilter::new(30, 10);
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let highpass = rf.calculate_highpass(&data);

        assert_eq!(highpass.len(), n);

        // Highpass on linear trend should converge to near-zero
        let last_values: Vec<f64> = highpass[80..100].to_vec();
        let avg: f64 = last_values.iter()
            .filter(|x| !x.is_nan())
            .sum::<f64>() / 20.0;

        assert!(avg.abs() < 5.0, "Highpass on trend should be small, got {}", avg);
    }

    #[test]
    fn test_roofing_filter_custom_periods() {
        let rf = RoofingFilter::new(24, 8);
        assert_eq!(rf.hp_period, 24);
        assert_eq!(rf.smooth_period, 8);
    }

    #[test]
    fn test_roofing_filter_trait_impl() {
        let rf = RoofingFilter::default();
        assert_eq!(rf.name(), "RoofingFilter");
        assert_eq!(rf.min_periods(), 3);
        assert_eq!(rf.output_features(), 2);
    }
}
