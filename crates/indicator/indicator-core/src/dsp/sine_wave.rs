//! Ehlers Sine Wave indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use std::f64::consts::PI;

/// Ehlers Sine Wave Indicator
///
/// Uses the Hilbert Transform to extract the dominant cycle and generate
/// sine and lead sine waves. Crossovers between the sine and lead sine
/// indicate cycle turning points.
///
/// The indicator outputs the Sine and LeadSine lines.
#[derive(Debug, Clone)]
pub struct SineWave {
    /// Minimum period for cycle detection (default: 6)
    pub min_period: usize,
    /// Maximum period for cycle detection (default: 50)
    pub max_period: usize,
}

impl SineWave {
    pub fn new(min_period: usize, max_period: usize) -> Self {
        Self { min_period, max_period }
    }

    /// Calculate Sine Wave indicator
    /// Returns (sine, lead_sine)
    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < 8 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut sine = vec![f64::NAN; n];
        let mut lead_sine = vec![f64::NAN; n];

        // Working arrays
        let mut smooth = vec![0.0; n];
        let mut detrender = vec![0.0; n];
        let mut i1 = vec![0.0; n];
        let mut q1 = vec![0.0; n];
        let mut i2 = vec![0.0; n];
        let mut q2 = vec![0.0; n];
        let mut re = vec![0.0; n];
        let mut im = vec![0.0; n];
        let mut period = vec![0.0; n];
        let mut smooth_period = vec![0.0; n];
        let mut dc_phase = vec![0.0; n];

        for i in 7..n {
            // Compute smooth price
            smooth[i] = (4.0 * data[i] + 3.0 * data[i - 1] + 2.0 * data[i - 2] + data[i - 3]) / 10.0;

            // Hilbert Transform coefficients
            let c1 = 0.0962;
            let c2 = 0.5769;

            // Compute detrended price
            detrender[i] = c1 * smooth[i] + c2 * (c1 * smooth[i - 2])
                - c2 * (c1 * smooth[i - 4])
                - c1 * smooth[i - 6];
            detrender[i] += c2 * detrender[i - 1];

            // Compute InPhase and Quadrature components
            q1[i] = c1 * detrender[i] + c2 * (c1 * detrender[i - 2])
                - c2 * (c1 * detrender[i - 4])
                - c1 * detrender[i - 6];
            q1[i] += c2 * q1[i - 1];

            i1[i] = detrender[i - 3];

            // Advance the phase of I1 and Q1 by 90 degrees
            let ji = c1 * i1[i] + c2 * (c1 * i1[i - 2])
                - c2 * (c1 * i1[i - 4])
                - c1 * i1[i - 6];

            let jq = c1 * q1[i] + c2 * (c1 * q1[i - 2])
                - c2 * (c1 * q1[i - 4])
                - c1 * q1[i - 6];

            // Phasor addition
            i2[i] = i1[i] - jq;
            q2[i] = q1[i] + ji;

            // Smooth the I and Q components
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1];
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1];

            // Homodyne Discriminator
            re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1];
            im[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1];

            re[i] = 0.2 * re[i] + 0.8 * re[i - 1];
            im[i] = 0.2 * im[i] + 0.8 * im[i - 1];

            // Calculate period
            if im[i] != 0.0 && re[i] != 0.0 {
                period[i] = 2.0 * PI / im[i].atan2(re[i]);
            }

            // Constrain period
            if period[i] > 1.5 * period[i - 1] {
                period[i] = 1.5 * period[i - 1];
            }
            if period[i] < 0.67 * period[i - 1] {
                period[i] = 0.67 * period[i - 1];
            }
            if period[i] < self.min_period as f64 {
                period[i] = self.min_period as f64;
            }
            if period[i] > self.max_period as f64 {
                period[i] = self.max_period as f64;
            }

            period[i] = 0.2 * period[i] + 0.8 * period[i - 1];
            smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1];

            // Compute DC Phase
            let dc_period = smooth_period[i] as usize;
            if dc_period > 0 && i >= dc_period {
                let mut real_part = 0.0;
                let mut imag_part = 0.0;

                for j in 0..dc_period {
                    if i >= j {
                        let angle = 2.0 * PI * j as f64 / dc_period as f64;
                        real_part += smooth[i - j] * angle.cos();
                        imag_part += smooth[i - j] * angle.sin();
                    }
                }

                if real_part.abs() > 0.0 {
                    dc_phase[i] = imag_part.atan2(real_part) * 180.0 / PI;
                }

                // Adjust phase for negative values
                if dc_phase[i] < 0.0 {
                    dc_phase[i] += 360.0;
                }

                // Add 90 degrees for phase adjustment
                dc_phase[i] += 90.0;
                if dc_phase[i] > 315.0 {
                    dc_phase[i] -= 360.0;
                }

                // Compute Sine and LeadSine
                sine[i] = (dc_phase[i] * PI / 180.0).sin();
                lead_sine[i] = ((dc_phase[i] + 45.0) * PI / 180.0).sin();
            }
        }

        (sine, lead_sine)
    }
}

impl Default for SineWave {
    fn default() -> Self {
        Self::new(6, 50)
    }
}

impl TechnicalIndicator for SineWave {
    fn name(&self) -> &str {
        "SineWave"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 8 {
            return Err(IndicatorError::InsufficientData {
                required: 8,
                got: data.close.len(),
            });
        }

        let (sine, lead_sine) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(sine, lead_sine))
    }

    fn min_periods(&self) -> usize {
        8
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for SineWave {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (sine, lead_sine) = self.calculate(&data.close);
        let n = sine.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let sine_last = sine[n - 1];
        let lead_last = lead_sine[n - 1];
        let sine_prev = sine[n - 2];
        let lead_prev = lead_sine[n - 2];

        if sine_last.is_nan() || lead_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Crossover signals - sine crossing above lead_sine = bullish
        if sine_last > lead_last && sine_prev <= lead_prev {
            Ok(IndicatorSignal::Bullish)
        } else if sine_last < lead_last && sine_prev >= lead_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (sine, lead_sine) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..sine.len().min(lead_sine.len()) {
            if sine[i].is_nan() || lead_sine[i].is_nan() || sine[i - 1].is_nan() || lead_sine[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if sine[i] > lead_sine[i] && sine[i - 1] <= lead_sine[i - 1] {
                signals.push(IndicatorSignal::Bullish);
            } else if sine[i] < lead_sine[i] && sine[i - 1] >= lead_sine[i - 1] {
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
    fn test_sine_wave_basic() {
        let sw = SineWave::default();
        let n = 100;
        // Generate a cyclic signal
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.2).sin() * 10.0).collect();

        let (sine, lead_sine) = sw.calculate(&data);

        assert_eq!(sine.len(), n);
        assert_eq!(lead_sine.len(), n);

        // First 7 values should be NAN
        for i in 0..7 {
            assert!(sine[i].is_nan());
        }
    }

    #[test]
    fn test_sine_wave_bounds() {
        let sw = SineWave::default();
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.3).sin() * 15.0).collect();

        let (sine, lead_sine) = sw.calculate(&data);

        // Sine values should be bounded between -1 and 1
        for i in 20..n {
            if !sine[i].is_nan() {
                assert!(sine[i] >= -1.0 && sine[i] <= 1.0,
                    "Sine[{}] = {} out of bounds", i, sine[i]);
            }
            if !lead_sine[i].is_nan() {
                assert!(lead_sine[i] >= -1.0 && lead_sine[i] <= 1.0,
                    "LeadSine[{}] = {} out of bounds", i, lead_sine[i]);
            }
        }
    }

    #[test]
    fn test_sine_wave_custom_period() {
        let sw = SineWave::new(8, 40);
        assert_eq!(sw.min_period, 8);
        assert_eq!(sw.max_period, 40);
    }
}
