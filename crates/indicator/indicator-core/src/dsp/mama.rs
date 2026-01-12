//! MAMA (Mother of Adaptive Moving Averages) with FAMA indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use std::f64::consts::PI;

/// MAMA - Mother of Adaptive Moving Averages with FAMA (Following Adaptive MA)
///
/// John Ehlers' MAMA indicator uses the Hilbert Transform to calculate the
/// rate of change of phase (frequency) and adapts the moving average speed.
/// FAMA is a slower following version of MAMA for crossover signals.
///
/// The indicator outputs MAMA and FAMA lines.
#[derive(Debug, Clone)]
pub struct MAMA {
    /// Fast limit for alpha adaptation (default: 0.5)
    pub fast_limit: f64,
    /// Slow limit for alpha adaptation (default: 0.05)
    pub slow_limit: f64,
}

impl MAMA {
    pub fn new(fast_limit: f64, slow_limit: f64) -> Self {
        Self { fast_limit, slow_limit }
    }

    /// Calculate MAMA and FAMA
    /// Returns (mama_line, fama_line)
    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < 33 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut mama = vec![f64::NAN; n];
        let mut fama = vec![f64::NAN; n];

        // Initialize working arrays
        let mut smooth = vec![0.0; n];
        let mut detrender = vec![0.0; n];
        let mut period = vec![0.0; n];
        let mut smooth_period = vec![0.0; n];
        let mut phase = vec![0.0; n];
        let mut i1 = vec![0.0; n];
        let mut q1 = vec![0.0; n];
        let mut i2 = vec![0.0; n];
        let mut q2 = vec![0.0; n];
        let mut re = vec![0.0; n];
        let mut im = vec![0.0; n];

        // Initialize first values
        mama[32] = data[32];
        fama[32] = data[32];

        for i in 6..n {
            // Compute smooth price (4-bar weighted average)
            smooth[i] = (4.0 * data[i] + 3.0 * data[i - 1] + 2.0 * data[i - 2] + data[i - 3]) / 10.0;

            // Hilbert Transform coefficients
            let c1 = 0.0962;
            let c2 = 0.5769;

            // Compute detrended price
            if i >= 6 {
                detrender[i] = c1 * (smooth[i] + smooth[i - 6]) / 2.0
                    - c2 * (c1 * (smooth[i - 2] + smooth[i - 4]) / 2.0)
                    + c2 * detrender[i - 1];
            }

            // Compute InPhase and Quadrature components
            if i >= 9 {
                q1[i] = c1 * (detrender[i] + detrender[i - 6]) / 2.0
                    - c2 * (c1 * (detrender[i - 2] + detrender[i - 4]) / 2.0)
                    + c2 * q1[i - 1];

                i1[i] = detrender[i - 3];
            }

            // Advance the phase of I1 and Q1 by 90 degrees
            if i >= 12 {
                let ji = c1 * (i1[i] + i1[i - 6]) / 2.0
                    - c2 * (c1 * (i1[i - 2] + i1[i - 4]) / 2.0);

                let jq = c1 * (q1[i] + q1[i - 6]) / 2.0
                    - c2 * (c1 * (q1[i - 2] + q1[i - 4]) / 2.0);

                // Phasor addition for 3-bar averaging
                i2[i] = i1[i] - jq;
                q2[i] = q1[i] + ji;

                // Smooth the I and Q components
                i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1];
                q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1];
            }

            // Homodyne Discriminator
            if i >= 13 {
                re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1];
                im[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1];

                re[i] = 0.2 * re[i] + 0.8 * re[i - 1];
                im[i] = 0.2 * im[i] + 0.8 * im[i - 1];

                // Calculate period
                if im[i] != 0.0 && re[i] != 0.0 {
                    period[i] = 2.0 * PI / im[i].atan2(re[i]);
                }

                // Constrain period between 6 and 50
                if period[i] > 1.5 * period[i - 1] {
                    period[i] = 1.5 * period[i - 1];
                }
                if period[i] < 0.67 * period[i - 1] {
                    period[i] = 0.67 * period[i - 1];
                }
                if period[i] < 6.0 {
                    period[i] = 6.0;
                }
                if period[i] > 50.0 {
                    period[i] = 50.0;
                }

                period[i] = 0.2 * period[i] + 0.8 * period[i - 1];
                smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1];
            }

            // Compute phase
            if i >= 6 && i1[i] != 0.0 {
                phase[i] = (q1[i] / i1[i]).atan() * 180.0 / PI;
            }

            // Compute MAMA and FAMA
            if i >= 33 {
                // Calculate delta phase
                let mut delta_phase = phase[i - 1] - phase[i];
                if delta_phase < 1.0 {
                    delta_phase = 1.0;
                }

                // Compute alpha
                let alpha = self.fast_limit / delta_phase;
                let alpha = if alpha < self.slow_limit {
                    self.slow_limit
                } else if alpha > self.fast_limit {
                    self.fast_limit
                } else {
                    alpha
                };

                mama[i] = alpha * data[i] + (1.0 - alpha) * mama[i - 1];
                fama[i] = 0.5 * alpha * mama[i] + (1.0 - 0.5 * alpha) * fama[i - 1];
            }
        }

        (mama, fama)
    }
}

impl Default for MAMA {
    fn default() -> Self {
        Self::new(0.5, 0.05)
    }
}

impl TechnicalIndicator for MAMA {
    fn name(&self) -> &str {
        "MAMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 33 {
            return Err(IndicatorError::InsufficientData {
                required: 33,
                got: data.close.len(),
            });
        }

        let (mama, fama) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(mama, fama))
    }

    fn min_periods(&self) -> usize {
        33
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for MAMA {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (mama, fama) = self.calculate(&data.close);
        let n = mama.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let mama_last = mama[n - 1];
        let fama_last = fama[n - 1];
        let mama_prev = mama[n - 2];
        let fama_prev = fama[n - 2];

        if mama_last.is_nan() || fama_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Crossover signals
        if mama_last > fama_last && mama_prev <= fama_prev {
            Ok(IndicatorSignal::Bullish)
        } else if mama_last < fama_last && mama_prev >= fama_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (mama, fama) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..mama.len().min(fama.len()) {
            if mama[i].is_nan() || fama[i].is_nan() || mama[i - 1].is_nan() || fama[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if mama[i] > fama[i] && mama[i - 1] <= fama[i - 1] {
                signals.push(IndicatorSignal::Bullish);
            } else if mama[i] < fama[i] && mama[i - 1] >= fama[i - 1] {
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
    fn test_mama_basic() {
        let mama = MAMA::default();
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();

        let (mama_line, fama_line) = mama.calculate(&data);

        assert_eq!(mama_line.len(), n);
        assert_eq!(fama_line.len(), n);

        // Values before warmup should be NAN
        for i in 0..32 {
            assert!(mama_line[i].is_nan());
        }

        // Values after warmup should be valid
        assert!(!mama_line[50].is_nan());
        assert!(!fama_line[50].is_nan());
    }

    #[test]
    fn test_mama_insufficient_data() {
        let mama = MAMA::default();
        let data: Vec<f64> = vec![100.0; 20];
        let series = OHLCVSeries::from_close(data);

        let result = mama.compute(&series);
        assert!(result.is_err());
    }

    #[test]
    fn test_mama_custom_params() {
        let mama = MAMA::new(0.8, 0.01);
        assert_eq!(mama.fast_limit, 0.8);
        assert_eq!(mama.slow_limit, 0.01);
    }

    #[test]
    fn test_mama_fama_relationship() {
        let mama = MAMA::default();
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.15).sin() * 15.0).collect();

        let (mama_line, fama_line) = mama.calculate(&data);

        // After warmup, FAMA should be smoother than MAMA (follow with lag)
        // This is a basic sanity check - FAMA uses half the alpha
        let mut mama_changes = 0.0;
        let mut fama_changes = 0.0;

        for i in 34..n {
            mama_changes += (mama_line[i] - mama_line[i - 1]).abs();
            fama_changes += (fama_line[i] - fama_line[i - 1]).abs();
        }

        // FAMA should have less total change (smoother)
        assert!(fama_changes < mama_changes);
    }
}
