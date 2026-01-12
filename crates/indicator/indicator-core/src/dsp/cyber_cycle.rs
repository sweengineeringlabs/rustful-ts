//! Ehlers Cyber Cycle indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use std::f64::consts::PI;

/// Ehlers Cyber Cycle Indicator
///
/// The Cyber Cycle is a leading indicator that extracts the cycle component
/// from price data using a 2-pole high-pass filter followed by a smooth.
/// It helps identify cycle turning points before they occur in price.
///
/// The indicator outputs the Cyber Cycle and its trigger (previous value).
#[derive(Debug, Clone)]
pub struct CyberCycle {
    /// Alpha smoothing factor (default: 0.07)
    pub alpha: f64,
}

impl CyberCycle {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Create from period (converts period to alpha)
    pub fn from_period(period: usize) -> Self {
        // alpha = 2 / (period + 1) for EMA-like smoothing
        let alpha = (2.0 * PI / period as f64).cos();
        let alpha = (1.0 - alpha).max(0.01).min(0.99);
        Self { alpha }
    }

    /// Calculate Cyber Cycle
    /// Returns (cycle, trigger)
    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < 5 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut cycle = vec![f64::NAN; n];
        let mut trigger = vec![f64::NAN; n];
        let mut smooth = vec![0.0; n];

        // Coefficients
        let a = self.alpha;
        let a2 = a * a;

        // First smooth the price data
        for i in 3..n {
            smooth[i] = (data[i] + 2.0 * data[i - 1] + 2.0 * data[i - 2] + data[i - 3]) / 6.0;
        }

        // Initialize cycle values - need to initialize enough history for recursive formula
        if n >= 4 {
            cycle[3] = (smooth[3] - 2.0 * smooth[2] + smooth[1]) / 4.0;
        }
        if n >= 5 {
            cycle[4] = (smooth[4] - 2.0 * smooth[3] + smooth[2]) / 4.0;
        }

        // Calculate cyber cycle using 2-pole high-pass filter
        for i in 5..n {
            // Cyber Cycle formula from Ehlers
            cycle[i] = (1.0 - 0.5 * a) * (1.0 - 0.5 * a) * (smooth[i] - 2.0 * smooth[i - 1] + smooth[i - 2])
                + 2.0 * (1.0 - a) * cycle[i - 1]
                - (1.0 - a) * (1.0 - a) * cycle[i - 2];

            // Alternative simpler formula if alpha is very small
            if a2 < 0.001 {
                cycle[i] = smooth[i] - 2.0 * smooth[i - 1] + smooth[i - 2];
            }
        }

        // Trigger is the previous cycle value
        for i in 5..n {
            trigger[i] = cycle[i - 1];
        }

        (cycle, trigger)
    }
}

impl Default for CyberCycle {
    fn default() -> Self {
        Self::new(0.07)
    }
}

impl TechnicalIndicator for CyberCycle {
    fn name(&self) -> &str {
        "CyberCycle"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 5 {
            return Err(IndicatorError::InsufficientData {
                required: 5,
                got: data.close.len(),
            });
        }

        let (cycle, trigger) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(cycle, trigger))
    }

    fn min_periods(&self) -> usize {
        5
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for CyberCycle {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (cycle, trigger) = self.calculate(&data.close);
        let n = cycle.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let cycle_last = cycle[n - 1];
        let trigger_last = trigger[n - 1];
        let cycle_prev = cycle[n - 2];
        let trigger_prev = trigger[n - 2];

        if cycle_last.is_nan() || trigger_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Crossover signals
        if cycle_last > trigger_last && cycle_prev <= trigger_prev {
            Ok(IndicatorSignal::Bullish)
        } else if cycle_last < trigger_last && cycle_prev >= trigger_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (cycle, trigger) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..cycle.len().min(trigger.len()) {
            if cycle[i].is_nan() || trigger[i].is_nan() || cycle[i - 1].is_nan() || trigger[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if cycle[i] > trigger[i] && cycle[i - 1] <= trigger[i - 1] {
                signals.push(IndicatorSignal::Bullish);
            } else if cycle[i] < trigger[i] && cycle[i - 1] >= trigger[i - 1] {
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
    fn test_cyber_cycle_basic() {
        let cc = CyberCycle::default();
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.2).sin() * 10.0).collect();

        let (cycle, trigger) = cc.calculate(&data);

        assert_eq!(cycle.len(), n);
        assert_eq!(trigger.len(), n);

        // First 3 values should be NAN (indices 0, 1, 2)
        // Index 3+ have values for the recursive filter to work
        for i in 0..3 {
            assert!(cycle[i].is_nan());
        }

        // Values after warmup should be valid
        assert!(!cycle[50].is_nan());
        assert!(!trigger[50].is_nan());
    }

    #[test]
    fn test_cyber_cycle_from_period() {
        let cc = CyberCycle::from_period(20);
        // Alpha should be derived from period
        assert!(cc.alpha > 0.0 && cc.alpha < 1.0);
    }

    #[test]
    fn test_cyber_cycle_oscillation() {
        let cc = CyberCycle::default();
        let n = 100;
        // Generate a clean sinusoidal signal
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.3).sin() * 20.0).collect();

        let (cycle, _) = cc.calculate(&data);

        // Cyber cycle should oscillate around zero
        let mut positive_count = 0;
        let mut negative_count = 0;

        for i in 20..n {
            if !cycle[i].is_nan() {
                if cycle[i] > 0.0 {
                    positive_count += 1;
                } else if cycle[i] < 0.0 {
                    negative_count += 1;
                }
            }
        }

        // Should have both positive and negative values (oscillating)
        assert!(positive_count > 0);
        assert!(negative_count > 0);
    }

    #[test]
    fn test_cyber_cycle_trigger_lag() {
        let cc = CyberCycle::default();
        let n = 50;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let (cycle, trigger) = cc.calculate(&data);

        // Trigger should lag cycle by 1 bar
        for i in 6..n {
            if !cycle[i - 1].is_nan() && !trigger[i].is_nan() {
                assert!((trigger[i] - cycle[i - 1]).abs() < 1e-10,
                    "Trigger[{}] should equal Cycle[{}]", i, i - 1);
            }
        }
    }

    #[test]
    fn test_cyber_cycle_trait_impl() {
        let cc = CyberCycle::default();
        assert_eq!(cc.name(), "CyberCycle");
        assert_eq!(cc.min_periods(), 5);
        assert_eq!(cc.output_features(), 2);
    }
}
