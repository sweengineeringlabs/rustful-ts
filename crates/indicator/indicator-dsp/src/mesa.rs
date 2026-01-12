//! MESA Adaptive Moving Average indicator.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use std::f64::consts::PI;

/// MESA Adaptive Moving Average
///
/// John Ehlers' MESA (Maximum Entropy Spectral Analysis) Adaptive Moving Average.
/// Uses the Hilbert Transform to measure the dominant cycle period and adapts
/// the moving average accordingly.
///
/// The indicator outputs the MESA line and a smoothed signal line.
#[derive(Debug, Clone)]
pub struct MESA {
    /// Fast limit for alpha adaptation (default: 0.5)
    pub fast_limit: f64,
    /// Slow limit for alpha adaptation (default: 0.05)
    pub slow_limit: f64,
}

impl MESA {
    pub fn new(fast_limit: f64, slow_limit: f64) -> Self {
        Self { fast_limit, slow_limit }
    }

    /// Calculate MESA Adaptive Moving Average
    /// Returns (mesa_line, signal_line)
    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < 7 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut mesa = vec![f64::NAN; n];
        let mut signal = vec![f64::NAN; n];

        // Smoothed price
        let mut smooth = vec![0.0; n];
        // Detrended price
        let mut detrender = vec![0.0; n];
        // In-phase and quadrature components
        let mut i1 = vec![0.0; n];
        let mut q1 = vec![0.0; n];
        // Previous values for smoothing
        let mut ji = vec![0.0; n];
        let mut jq = vec![0.0; n];
        let mut i2 = vec![0.0; n];
        let mut q2 = vec![0.0; n];
        let mut re = vec![0.0; n];
        let mut im = vec![0.0; n];
        let mut period = vec![0.0; n];
        let mut smooth_period = vec![0.0; n];
        let mut phase = vec![0.0; n];

        for i in 6..n {
            // Compute smooth price
            smooth[i] = (4.0 * data[i] + 3.0 * data[i - 1] + 2.0 * data[i - 2] + data[i - 3]) / 10.0;

            // Compute detrended price using Hilbert Transform coefficients
            let hilbert_coef = 0.0962;
            let hilbert_factor = 0.5769;
            detrender[i] = hilbert_coef * smooth[i] + hilbert_factor * (hilbert_coef * smooth[i - 2])
                - hilbert_factor * (hilbert_coef * smooth[i - 4])
                - hilbert_coef * smooth[i - 6];
            detrender[i] += hilbert_factor * detrender[i - 1];

            // Compute in-phase and quadrature components
            q1[i] = hilbert_coef * detrender[i] + hilbert_factor * (hilbert_coef * detrender[i - 2])
                - hilbert_factor * (hilbert_coef * detrender[i - 4])
                - hilbert_coef * detrender[i - 6];
            q1[i] += hilbert_factor * q1[i - 1];

            i1[i] = detrender[i - 3];

            // Advance the phase by 90 degrees
            ji[i] = hilbert_coef * i1[i] + hilbert_factor * (hilbert_coef * i1[i - 2])
                - hilbert_factor * (hilbert_coef * i1[i - 4])
                - hilbert_coef * i1[i - 6];
            ji[i] += hilbert_factor * ji[i - 1];

            jq[i] = hilbert_coef * q1[i] + hilbert_factor * (hilbert_coef * q1[i - 2])
                - hilbert_factor * (hilbert_coef * q1[i - 4])
                - hilbert_coef * q1[i - 6];
            jq[i] += hilbert_factor * jq[i - 1];

            // Phasor addition for 3-bar averaging
            i2[i] = i1[i] - jq[i];
            q2[i] = q1[i] + ji[i];

            // Smooth the I and Q components
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1];
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1];

            // Homodyne discriminator
            re[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1];
            im[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1];

            re[i] = 0.2 * re[i] + 0.8 * re[i - 1];
            im[i] = 0.2 * im[i] + 0.8 * im[i - 1];

            // Calculate period
            if im[i] != 0.0 && re[i] != 0.0 {
                period[i] = 2.0 * PI / (im[i] / re[i]).atan();
            }

            // Constrain period
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

            // Smooth the period
            period[i] = 0.2 * period[i] + 0.8 * period[i - 1];
            smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1];

            // Compute phase
            if i1[i] != 0.0 {
                phase[i] = (q1[i] / i1[i]).atan() * 180.0 / PI;
            }

            // Calculate delta phase
            let delta_phase = (phase[i - 1] - phase[i]).max(1.0);

            // Calculate alpha
            let alpha = (self.fast_limit / delta_phase).max(self.slow_limit);

            // Calculate MESA
            mesa[i] = alpha * data[i] + (1.0 - alpha) * mesa[i - 1].max(data[i - 1]);
            signal[i] = 0.5 * alpha * mesa[i] + (1.0 - 0.5 * alpha) * signal[i - 1].max(mesa[i - 1]);
        }

        (mesa, signal)
    }
}

impl Default for MESA {
    fn default() -> Self {
        Self::new(0.5, 0.05)
    }
}

impl TechnicalIndicator for MESA {
    fn name(&self) -> &str {
        "MESA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 7 {
            return Err(IndicatorError::InsufficientData {
                required: 7,
                got: data.close.len(),
            });
        }

        let (mesa, signal) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(mesa, signal))
    }

    fn min_periods(&self) -> usize {
        7
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for MESA {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (mesa, signal) = self.calculate(&data.close);
        let n = mesa.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let mesa_last = mesa[n - 1];
        let signal_last = signal[n - 1];
        let mesa_prev = mesa[n - 2];
        let signal_prev = signal[n - 2];

        if mesa_last.is_nan() || signal_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Crossover signals
        if mesa_last > signal_last && mesa_prev <= signal_prev {
            Ok(IndicatorSignal::Bullish)
        } else if mesa_last < signal_last && mesa_prev >= signal_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (mesa, signal) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..mesa.len().min(signal.len()) {
            if mesa[i].is_nan() || signal[i].is_nan() || mesa[i - 1].is_nan() || signal[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if mesa[i] > signal[i] && mesa[i - 1] <= signal[i - 1] {
                signals.push(IndicatorSignal::Bullish);
            } else if mesa[i] < signal[i] && mesa[i - 1] >= signal[i - 1] {
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
    fn test_mesa_basic() {
        let mesa = MESA::default();
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();

        let (mesa_line, signal_line) = mesa.calculate(&data);

        assert_eq!(mesa_line.len(), n);
        assert_eq!(signal_line.len(), n);

        // First 6 values should be NAN
        for i in 0..6 {
            assert!(mesa_line[i].is_nan());
        }

        // Values after warmup should be valid
        assert!(!mesa_line[50].is_nan());
        assert!(!signal_line[50].is_nan());
    }

    #[test]
    fn test_mesa_insufficient_data() {
        let mesa = MESA::default();
        let data: Vec<f64> = vec![100.0, 101.0, 102.0];
        let series = OHLCVSeries::from_close(data);

        let result = mesa.compute(&series);
        assert!(result.is_err());
    }

    #[test]
    fn test_mesa_custom_params() {
        let mesa = MESA::new(0.8, 0.01);
        assert_eq!(mesa.fast_limit, 0.8);
        assert_eq!(mesa.slow_limit, 0.01);
    }
}
