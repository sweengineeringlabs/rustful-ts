//! Center of Gravity Oscillator indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Center of Gravity (CG) Oscillator
///
/// John Ehlers' Center of Gravity oscillator computes the center of gravity
/// of prices over a lookback period. The CG acts like a leading indicator,
/// identifying potential turning points before they occur.
///
/// The CG is calculated as a weighted sum where recent prices have higher weights.
/// The indicator outputs the CG line and a trigger (signal) line.
#[derive(Debug, Clone)]
pub struct CGOscillator {
    /// Lookback period (default: 10)
    pub period: usize,
}

impl CGOscillator {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate Center of Gravity Oscillator
    /// Returns (cg, trigger)
    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < self.period {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut cg = vec![f64::NAN; n];
        let mut trigger = vec![f64::NAN; n];

        for i in (self.period - 1)..n {
            // Calculate weighted numerator and denominator
            let mut num = 0.0;
            let mut denom = 0.0;

            for j in 0..self.period {
                let idx = i - j;
                let weight = (j + 1) as f64;
                num += weight * data[idx];
                denom += data[idx];
            }

            // CG = -Num / Denom (negative for display purposes)
            if denom != 0.0 {
                cg[i] = -num / denom;
            } else {
                cg[i] = 0.0;
            }
        }

        // Trigger is the previous CG value
        for i in self.period..n {
            trigger[i] = cg[i - 1];
        }

        (cg, trigger)
    }
}

impl Default for CGOscillator {
    fn default() -> Self {
        Self::new(10)
    }
}

impl TechnicalIndicator for CGOscillator {
    fn name(&self) -> &str {
        "CGOscillator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let (cg, trigger) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(cg, trigger))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for CGOscillator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (cg, trigger) = self.calculate(&data.close);
        let n = cg.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let cg_last = cg[n - 1];
        let trigger_last = trigger[n - 1];
        let cg_prev = cg[n - 2];
        let trigger_prev = trigger[n - 2];

        if cg_last.is_nan() || trigger_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // CG crossing above trigger = bullish (anticipating upturn)
        if cg_last > trigger_last && cg_prev <= trigger_prev {
            Ok(IndicatorSignal::Bullish)
        } else if cg_last < trigger_last && cg_prev >= trigger_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (cg, trigger) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..cg.len().min(trigger.len()) {
            if cg[i].is_nan() || trigger[i].is_nan() || cg[i - 1].is_nan() || trigger[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if cg[i] > trigger[i] && cg[i - 1] <= trigger[i - 1] {
                signals.push(IndicatorSignal::Bullish);
            } else if cg[i] < trigger[i] && cg[i - 1] >= trigger[i - 1] {
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
    fn test_cg_oscillator_basic() {
        let cg = CGOscillator::default();
        let n = 50;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.2).sin() * 10.0).collect();

        let (cg_line, trigger_line) = cg.calculate(&data);

        assert_eq!(cg_line.len(), n);
        assert_eq!(trigger_line.len(), n);

        // First (period-1) values should be NAN
        for i in 0..(cg.period - 1) {
            assert!(cg_line[i].is_nan());
        }

        // Values after warmup should be valid
        assert!(!cg_line[20].is_nan());
        assert!(!trigger_line[20].is_nan());
    }

    #[test]
    fn test_cg_oscillator_range() {
        let cg = CGOscillator::new(10);
        let n = 50;
        let data: Vec<f64> = vec![100.0; n]; // Constant prices

        let (cg_line, _) = cg.calculate(&data);

        // For constant prices, CG should be at the center of the period
        // CG = -sum(i * price) / sum(price) = -sum(i) / n
        // For period 10: sum(1..10) = 55, so CG = -55/10 = -5.5
        for i in 10..n {
            if !cg_line[i].is_nan() {
                assert!((cg_line[i] - (-5.5)).abs() < 1e-10,
                    "CG[{}] = {} should be -5.5", i, cg_line[i]);
            }
        }
    }

    #[test]
    fn test_cg_oscillator_custom_period() {
        let cg = CGOscillator::new(20);
        assert_eq!(cg.period, 20);
        assert_eq!(cg.min_periods(), 20);
    }

    #[test]
    fn test_cg_oscillator_trigger_lag() {
        let cg = CGOscillator::default();
        let n = 50;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let (cg_line, trigger_line) = cg.calculate(&data);

        // Trigger should lag CG by 1 bar
        for i in (cg.period + 1)..n {
            if !cg_line[i - 1].is_nan() && !trigger_line[i].is_nan() {
                assert!((trigger_line[i] - cg_line[i - 1]).abs() < 1e-10,
                    "Trigger[{}] should equal CG[{}]", i, i - 1);
            }
        }
    }

    #[test]
    fn test_cg_oscillator_trait_impl() {
        let cg = CGOscillator::default();
        assert_eq!(cg.name(), "CGOscillator");
        assert_eq!(cg.min_periods(), 10);
        assert_eq!(cg.output_features(), 2);
    }
}
