//! Hilbert Transform (Dominant Cycle) indicator.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use std::f64::consts::PI;

/// Hilbert Transform - Dominant Cycle Period
///
/// John Ehlers' Hilbert Transform indicator measures the dominant cycle
/// period of price data. It uses the Hilbert Transform to compute the
/// instantaneous phase and then derives the period from the phase change.
///
/// The indicator outputs the dominant cycle period and its smoothed version.
#[derive(Debug, Clone)]
pub struct HilbertTransform {
    /// Minimum period constraint (default: 6)
    pub min_period: usize,
    /// Maximum period constraint (default: 50)
    pub max_period: usize,
}

impl HilbertTransform {
    pub fn new(min_period: usize, max_period: usize) -> Self {
        Self { min_period, max_period }
    }

    /// Calculate Hilbert Transform Dominant Cycle
    /// Returns (period, smooth_period)
    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < 8 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut period = vec![f64::NAN; n];
        let mut smooth_period = vec![f64::NAN; n];

        // Working arrays
        let mut smooth = vec![0.0; n];
        let mut detrender = vec![0.0; n];
        let mut i1 = vec![0.0; n];
        let mut q1 = vec![0.0; n];
        let mut i2 = vec![0.0; n];
        let mut q2 = vec![0.0; n];
        let mut re = vec![0.0; n];
        let mut im = vec![0.0; n];
        let mut inst_period = vec![self.min_period as f64; n];
        let mut inst_smooth_period = vec![self.min_period as f64; n];

        for i in 7..n {
            // Compute smooth price (4-bar weighted average)
            smooth[i] = (4.0 * data[i] + 3.0 * data[i - 1] + 2.0 * data[i - 2] + data[i - 3]) / 10.0;

            // Hilbert Transform coefficients
            let c1 = 0.0962;
            let c2 = 0.5769;

            // Compute detrended price using Hilbert Transform
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

            // Phasor addition for 3-bar averaging
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

            // Calculate instantaneous period
            if im[i] != 0.0 && re[i] != 0.0 {
                inst_period[i] = 2.0 * PI / im[i].atan2(re[i]);
            } else {
                inst_period[i] = inst_period[i - 1];
            }

            // Constrain period with rate of change limits
            if inst_period[i] > 1.5 * inst_period[i - 1] {
                inst_period[i] = 1.5 * inst_period[i - 1];
            }
            if inst_period[i] < 0.67 * inst_period[i - 1] {
                inst_period[i] = 0.67 * inst_period[i - 1];
            }

            // Constrain period within bounds
            if inst_period[i] < self.min_period as f64 {
                inst_period[i] = self.min_period as f64;
            }
            if inst_period[i] > self.max_period as f64 {
                inst_period[i] = self.max_period as f64;
            }

            // Smooth the period
            inst_period[i] = 0.2 * inst_period[i] + 0.8 * inst_period[i - 1];
            inst_smooth_period[i] = 0.33 * inst_period[i] + 0.67 * inst_smooth_period[i - 1];

            period[i] = inst_period[i];
            smooth_period[i] = inst_smooth_period[i];
        }

        (period, smooth_period)
    }

    /// Calculate instantaneous phase
    pub fn calculate_phase(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 8 {
            return vec![f64::NAN; n];
        }

        let mut phase = vec![f64::NAN; n];
        let mut smooth = vec![0.0; n];
        let mut detrender = vec![0.0; n];
        let mut i1 = vec![0.0; n];
        let mut q1 = vec![0.0; n];

        for i in 7..n {
            smooth[i] = (4.0 * data[i] + 3.0 * data[i - 1] + 2.0 * data[i - 2] + data[i - 3]) / 10.0;

            let c1 = 0.0962;
            let c2 = 0.5769;

            detrender[i] = c1 * smooth[i] + c2 * (c1 * smooth[i - 2])
                - c2 * (c1 * smooth[i - 4])
                - c1 * smooth[i - 6];
            detrender[i] += c2 * detrender[i - 1];

            q1[i] = c1 * detrender[i] + c2 * (c1 * detrender[i - 2])
                - c2 * (c1 * detrender[i - 4])
                - c1 * detrender[i - 6];
            q1[i] += c2 * q1[i - 1];

            i1[i] = detrender[i - 3];

            if i1[i] != 0.0 {
                phase[i] = (q1[i] / i1[i]).atan() * 180.0 / PI;
            }
        }

        phase
    }
}

impl Default for HilbertTransform {
    fn default() -> Self {
        Self::new(6, 50)
    }
}

impl TechnicalIndicator for HilbertTransform {
    fn name(&self) -> &str {
        "HilbertTransform"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 8 {
            return Err(IndicatorError::InsufficientData {
                required: 8,
                got: data.close.len(),
            });
        }

        let (period, smooth_period) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(period, smooth_period))
    }

    fn min_periods(&self) -> usize {
        8
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for HilbertTransform {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (period, smooth_period) = self.calculate(&data.close);
        let n = period.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let period_last = period[n - 1];
        let smooth_last = smooth_period[n - 1];
        let period_prev = period[n - 2];
        let smooth_prev = smooth_period[n - 2];

        if period_last.is_nan() || smooth_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Shorter periods indicate faster cycles (potential bullish momentum)
        // Longer periods indicate slower cycles (potential consolidation)
        if period_last < smooth_last && period_prev >= smooth_prev {
            Ok(IndicatorSignal::Bullish)
        } else if period_last > smooth_last && period_prev <= smooth_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (period, smooth_period) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..period.len().min(smooth_period.len()) {
            if period[i].is_nan() || smooth_period[i].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if period[i] < smooth_period[i] && period[i - 1] >= smooth_period[i - 1] {
                signals.push(IndicatorSignal::Bullish);
            } else if period[i] > smooth_period[i] && period[i - 1] <= smooth_period[i - 1] {
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
    fn test_hilbert_basic() {
        let ht = HilbertTransform::default();
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.2).sin() * 10.0).collect();

        let (period, smooth_period) = ht.calculate(&data);

        assert_eq!(period.len(), n);
        assert_eq!(smooth_period.len(), n);

        // First 7 values should be NAN
        for i in 0..7 {
            assert!(period[i].is_nan());
        }
    }

    #[test]
    fn test_hilbert_period_bounds() {
        let ht = HilbertTransform::new(8, 40);
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.3).sin() * 15.0).collect();

        let (period, _) = ht.calculate(&data);

        // Period values should be within bounds
        for i in 20..n {
            if !period[i].is_nan() {
                assert!(period[i] >= 8.0 && period[i] <= 40.0,
                    "Period[{}] = {} out of bounds", i, period[i]);
            }
        }
    }

    #[test]
    fn test_hilbert_phase() {
        let ht = HilbertTransform::default();
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.25).sin() * 12.0).collect();

        let phase = ht.calculate_phase(&data);

        assert_eq!(phase.len(), n);

        // Phase should be bounded (-90 to 90 degrees typically)
        for i in 20..n {
            if !phase[i].is_nan() {
                assert!(phase[i] >= -90.0 && phase[i] <= 90.0,
                    "Phase[{}] = {} out of expected bounds", i, phase[i]);
            }
        }
    }

    #[test]
    fn test_hilbert_trait_impl() {
        let ht = HilbertTransform::default();
        assert_eq!(ht.name(), "HilbertTransform");
        assert_eq!(ht.min_periods(), 8);
        assert_eq!(ht.output_features(), 2);
    }
}
