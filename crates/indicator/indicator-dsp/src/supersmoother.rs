//! Ehlers Supersmoother filter.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use std::f64::consts::PI;

/// Ehlers Supersmoother (2-pole Butterworth Filter)
///
/// The Supersmoother is a 2-pole Butterworth low-pass filter that provides
/// superior smoothing with minimal lag compared to traditional moving averages.
/// It effectively removes high-frequency noise while preserving the underlying
/// trend and cycle components.
///
/// The output is a smoothed version of the input price data.
#[derive(Debug, Clone)]
pub struct Supersmoother {
    /// Cutoff period for the filter (default: 10)
    pub period: usize,
}

impl Supersmoother {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate Supersmoother filter
    /// Returns smoothed values
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 3 {
            return vec![f64::NAN; n];
        }

        let mut smooth = vec![f64::NAN; n];

        // 2-pole Butterworth coefficients
        let a1 = (-1.414 * PI / self.period as f64).exp();
        let b1 = 2.0 * a1 * (1.414 * PI / self.period as f64).cos();
        let c2 = b1;
        let c3 = -a1 * a1;
        let c1 = 1.0 - c2 - c3;

        // Initialize with simple values
        smooth[0] = data[0];
        smooth[1] = data[1];

        // Apply filter
        for i in 2..n {
            smooth[i] = c1 * (data[i] + data[i - 1]) / 2.0
                + c2 * smooth[i - 1]
                + c3 * smooth[i - 2];
        }

        smooth
    }

    /// Calculate with custom coefficients (for advanced use)
    pub fn calculate_with_coefficients(&self, data: &[f64], c1: f64, c2: f64, c3: f64) -> Vec<f64> {
        let n = data.len();
        if n < 3 {
            return vec![f64::NAN; n];
        }

        let mut smooth = vec![f64::NAN; n];

        smooth[0] = data[0];
        smooth[1] = data[1];

        for i in 2..n {
            smooth[i] = c1 * (data[i] + data[i - 1]) / 2.0
                + c2 * smooth[i - 1]
                + c3 * smooth[i - 2];
        }

        smooth
    }

    /// Get the filter coefficients for the current period
    pub fn coefficients(&self) -> (f64, f64, f64) {
        let a1 = (-1.414 * PI / self.period as f64).exp();
        let b1 = 2.0 * a1 * (1.414 * PI / self.period as f64).cos();
        let c2 = b1;
        let c3 = -a1 * a1;
        let c1 = 1.0 - c2 - c3;
        (c1, c2, c3)
    }
}

impl Default for Supersmoother {
    fn default() -> Self {
        Self::new(10)
    }
}

impl TechnicalIndicator for Supersmoother {
    fn name(&self) -> &str {
        "Supersmoother"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 3 {
            return Err(IndicatorError::InsufficientData {
                required: 3,
                got: data.close.len(),
            });
        }

        let smooth = self.calculate(&data.close);
        Ok(IndicatorOutput::single(smooth))
    }

    fn min_periods(&self) -> usize {
        3
    }
}

impl SignalIndicator for Supersmoother {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let smooth = self.calculate(&data.close);
        let n = smooth.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let smooth_last = smooth[n - 1];
        let smooth_prev = smooth[n - 2];
        let price_last = data.close[n - 1];

        if smooth_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Price crossing above smoothed line = bullish
        if price_last > smooth_last && data.close[n - 2] <= smooth_prev {
            Ok(IndicatorSignal::Bullish)
        } else if price_last < smooth_last && data.close[n - 2] >= smooth_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let smooth = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..smooth.len() {
            if smooth[i].is_nan() || smooth[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if data.close[i] > smooth[i] && data.close[i - 1] <= smooth[i - 1] {
                signals.push(IndicatorSignal::Bullish);
            } else if data.close[i] < smooth[i] && data.close[i - 1] >= smooth[i - 1] {
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
    fn test_supersmoother_basic() {
        let ss = Supersmoother::default();
        let n = 50;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.3).sin() * 10.0).collect();

        let smooth = ss.calculate(&data);

        assert_eq!(smooth.len(), n);

        // All values should be valid
        for i in 0..n {
            assert!(!smooth[i].is_nan(), "smooth[{}] is NaN", i);
        }
    }

    #[test]
    fn test_supersmoother_removes_noise() {
        let ss = Supersmoother::new(20);

        // Signal with high-frequency noise
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| {
            100.0 + (i as f64 * 0.1).sin() * 10.0  // Low freq
                + (i as f64 * 2.0).sin() * 2.0    // High freq noise
        }).collect();

        let smooth = ss.calculate(&data);

        // Calculate variance of differences (smoothness measure)
        let mut raw_diff_sq = 0.0;
        let mut smooth_diff_sq = 0.0;

        for i in 1..n {
            raw_diff_sq += (data[i] - data[i - 1]).powi(2);
            if !smooth[i].is_nan() && !smooth[i - 1].is_nan() {
                smooth_diff_sq += (smooth[i] - smooth[i - 1]).powi(2);
            }
        }

        // Smoothed signal should have less variance in differences
        assert!(smooth_diff_sq < raw_diff_sq,
            "Smoothed variance ({}) should be less than raw ({})", smooth_diff_sq, raw_diff_sq);
    }

    #[test]
    fn test_supersmoother_trend_following() {
        let ss = Supersmoother::new(10);

        // Linear uptrend
        let n = 50;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 2.0).collect();

        let smooth = ss.calculate(&data);

        // Smoothed should follow the trend closely
        for i in 20..n {
            // Smoothed value should be below current price in uptrend (lag)
            assert!(smooth[i] < data[i],
                "In uptrend, smooth[{}]={} should be < data[{}]={}",
                i, smooth[i], i, data[i]);
        }
    }

    #[test]
    fn test_supersmoother_coefficients() {
        let ss = Supersmoother::new(10);
        let (c1, c2, c3) = ss.coefficients();

        // Coefficients should satisfy stability conditions
        assert!(c1 > 0.0 && c1 < 1.0, "c1 should be in (0,1)");
        assert!(c2 > 0.0, "c2 should be positive");
        assert!(c3 < 0.0, "c3 should be negative");

        // Sum should be approximately 1 for unity gain at DC
        let sum = c1 + c2 + c3;
        assert!((sum - 1.0).abs() < 0.1, "Coefficients should sum to ~1, got {}", sum);
    }

    #[test]
    fn test_supersmoother_period_effect() {
        let ss_fast = Supersmoother::new(5);
        let ss_slow = Supersmoother::new(30);

        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.4).sin() * 15.0).collect();

        let smooth_fast = ss_fast.calculate(&data);
        let smooth_slow = ss_slow.calculate(&data);

        // Calculate lag by finding peak difference
        let mut fast_lag = 0.0;
        let mut slow_lag = 0.0;

        for i in 50..n {
            fast_lag += (data[i] - smooth_fast[i]).abs();
            slow_lag += (data[i] - smooth_slow[i]).abs();
        }

        // Shorter period should have less lag (closer to price)
        assert!(fast_lag < slow_lag,
            "Fast lag ({}) should be < slow lag ({})", fast_lag, slow_lag);
    }

    #[test]
    fn test_supersmoother_custom_coefficients() {
        let ss = Supersmoother::new(10);
        let data: Vec<f64> = vec![100.0, 102.0, 104.0, 103.0, 105.0];

        let (c1, c2, c3) = ss.coefficients();
        let smooth1 = ss.calculate(&data);
        let smooth2 = ss.calculate_with_coefficients(&data, c1, c2, c3);

        // Should produce identical results
        for i in 0..data.len() {
            if !smooth1[i].is_nan() && !smooth2[i].is_nan() {
                assert!((smooth1[i] - smooth2[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_supersmoother_trait_impl() {
        let ss = Supersmoother::default();
        assert_eq!(ss.name(), "Supersmoother");
        assert_eq!(ss.min_periods(), 3);
        assert_eq!(ss.output_features(), 1);
    }
}
