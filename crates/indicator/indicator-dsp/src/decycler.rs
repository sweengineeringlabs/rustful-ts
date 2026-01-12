//! Ehlers Decycler indicator.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use std::f64::consts::PI;

/// Ehlers Decycler
///
/// The Decycler is a high-pass filter that removes the cycle component from
/// price data, leaving only the trend. It's the opposite of the Roofing Filter
/// which removes the trend and keeps the cycle.
///
/// The output represents the smoothed trend with cycle noise removed.
#[derive(Debug, Clone)]
pub struct Decycler {
    /// High-pass cutoff period (default: 125)
    /// Cycles shorter than this period are removed
    pub period: usize,
}

impl Decycler {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate Decycler
    /// Returns (decycler, trigger) where trigger is a smoothed signal line
    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < 3 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut decycler = vec![f64::NAN; n];
        let mut trigger = vec![f64::NAN; n];
        let mut highpass = vec![0.0; n];

        // High-pass filter coefficient
        // alpha = (cosine(360/HP_period) + sine(360/HP_period) - 1) / cosine(360/HP_period)
        let angle = 2.0 * PI / self.period as f64;
        let alpha = ((1.0 - angle.sin()) / angle.cos()).max(0.0).min(1.0);

        // Apply 2-pole high-pass filter
        for i in 2..n {
            highpass[i] = (1.0 - alpha / 2.0) * (1.0 - alpha / 2.0)
                * (data[i] - 2.0 * data[i - 1] + data[i - 2])
                + 2.0 * (1.0 - alpha) * highpass[i - 1]
                - (1.0 - alpha) * (1.0 - alpha) * highpass[i - 2];
        }

        // Decycler = Price - HighPass (removing cycles leaves trend)
        for i in 0..n {
            if i < 2 {
                decycler[i] = data[i];
            } else {
                decycler[i] = data[i] - highpass[i];
            }
        }

        // Create trigger using simple smoothing
        for i in 1..n {
            if decycler[i - 1].is_nan() {
                trigger[i] = decycler[i];
            } else {
                trigger[i] = decycler[i - 1];
            }
        }

        (decycler, trigger)
    }

    /// Calculate just the cycle component (inverse of decycler)
    pub fn calculate_cycle(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 3 {
            return vec![f64::NAN; n];
        }

        let (decycler, _) = self.calculate(data);

        // Cycle = Price - Decycler (what was removed)
        let cycle: Vec<f64> = data.iter()
            .zip(decycler.iter())
            .map(|(price, dec)| {
                if dec.is_nan() {
                    f64::NAN
                } else {
                    price - dec
                }
            })
            .collect();

        cycle
    }
}

impl Default for Decycler {
    fn default() -> Self {
        Self::new(125)
    }
}

impl TechnicalIndicator for Decycler {
    fn name(&self) -> &str {
        "Decycler"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 3 {
            return Err(IndicatorError::InsufficientData {
                required: 3,
                got: data.close.len(),
            });
        }

        let (decycler, trigger) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(decycler, trigger))
    }

    fn min_periods(&self) -> usize {
        3
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for Decycler {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (decycler, trigger) = self.calculate(&data.close);
        let n = decycler.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let dec_last = decycler[n - 1];
        let trigger_last = trigger[n - 1];
        let dec_prev = decycler[n - 2];
        let trigger_prev = trigger[n - 2];

        if dec_last.is_nan() || trigger_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Decycler crossing above trigger = bullish trend
        if dec_last > trigger_last && dec_prev <= trigger_prev {
            Ok(IndicatorSignal::Bullish)
        } else if dec_last < trigger_last && dec_prev >= trigger_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (decycler, trigger) = self.calculate(&data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..decycler.len().min(trigger.len()) {
            if decycler[i].is_nan() || trigger[i].is_nan() || decycler[i - 1].is_nan() || trigger[i - 1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if decycler[i] > trigger[i] && decycler[i - 1] <= trigger[i - 1] {
                signals.push(IndicatorSignal::Bullish);
            } else if decycler[i] < trigger[i] && decycler[i - 1] >= trigger[i - 1] {
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
    fn test_decycler_basic() {
        let dc = Decycler::default();
        let n = 200;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.2).sin() * 10.0).collect();

        let (decycler, trigger) = dc.calculate(&data);

        assert_eq!(decycler.len(), n);
        assert_eq!(trigger.len(), n);

        // Values should be valid
        assert!(!decycler[50].is_nan());
        assert!(!trigger[50].is_nan());
    }

    #[test]
    fn test_decycler_preserves_trend() {
        let dc = Decycler::new(50);

        // Linear trend with cycles
        let n = 200;
        let data: Vec<f64> = (0..n).map(|i| {
            100.0 + i as f64 * 0.5  // Trend
            + (i as f64 * 0.3).sin() * 10.0  // Cycle
        }).collect();

        let (decycler, _) = dc.calculate(&data);

        // Decycler should approximately follow the trend
        // Check slope is similar to original trend
        let trend_slope = 0.5;  // Our artificial trend slope
        let mut decycler_slope_sum = 0.0;
        let mut count = 0;

        for i in 100..(n - 1) {
            if !decycler[i].is_nan() && !decycler[i + 1].is_nan() {
                decycler_slope_sum += decycler[i + 1] - decycler[i];
                count += 1;
            }
        }

        let avg_slope = decycler_slope_sum / count as f64;
        assert!((avg_slope - trend_slope).abs() < 0.2,
            "Decycler avg slope {} should be close to trend slope {}", avg_slope, trend_slope);
    }

    #[test]
    fn test_decycler_removes_cycles() {
        let dc = Decycler::new(30);

        // Pure cycle (no trend)
        let n = 200;
        let data: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.4).sin() * 15.0).collect();

        let (decycler, _) = dc.calculate(&data);

        // For pure cycle, decycler should be relatively flat
        // Calculate standard deviation of decycler (should be much less than input)
        let dec_values: Vec<f64> = decycler[100..n].iter()
            .filter(|x| !x.is_nan())
            .cloned()
            .collect();

        let dec_mean: f64 = dec_values.iter().sum::<f64>() / dec_values.len() as f64;
        let dec_std: f64 = (dec_values.iter()
            .map(|x| (x - dec_mean).powi(2))
            .sum::<f64>() / dec_values.len() as f64).sqrt();

        // Input std dev is approximately 15 * 0.707 ≈ 10.6
        // Decycler should have much lower std dev
        assert!(dec_std < 5.0,
            "Decycler std dev {} should be much less than cycle amplitude", dec_std);
    }

    #[test]
    fn test_decycler_cycle_extraction() {
        let dc = Decycler::new(50);

        let n = 200;
        let data: Vec<f64> = (0..n).map(|i| {
            100.0 + i as f64 * 0.3  // Trend
            + (i as f64 * 0.25).sin() * 12.0  // Cycle
        }).collect();

        let (decycler, _) = dc.calculate(&data);
        let cycle = dc.calculate_cycle(&data);

        // Price = Decycler + Cycle should hold
        for i in 50..n {
            if !decycler[i].is_nan() && !cycle[i].is_nan() {
                let reconstructed = decycler[i] + cycle[i];
                assert!((reconstructed - data[i]).abs() < 1e-10,
                    "Price[{}]={} should equal Decycler[{}]+Cycle[{}]={}",
                    i, data[i], i, i, reconstructed);
            }
        }
    }

    #[test]
    fn test_decycler_period_effect() {
        // Shorter period = more aggressive cycle removal
        let dc_short = Decycler::new(30);
        let dc_long = Decycler::new(200);

        let n = 300;
        let data: Vec<f64> = (0..n).map(|i| {
            100.0 + (i as f64 * 0.15).sin() * 20.0
        }).collect();

        let (dec_short, _) = dc_short.calculate(&data);
        let (dec_long, _) = dc_long.calculate(&data);

        // Calculate variance of each
        let short_var: f64 = dec_short[150..n].iter()
            .filter(|x| !x.is_nan())
            .map(|x| (x - 100.0).powi(2))
            .sum::<f64>() / 150.0;

        let long_var: f64 = dec_long[150..n].iter()
            .filter(|x| !x.is_nan())
            .map(|x| (x - 100.0).powi(2))
            .sum::<f64>() / 150.0;

        // Shorter period should remove more cycles (lower variance)
        assert!(short_var < long_var,
            "Short period variance ({}) should be < long period variance ({})",
            short_var, long_var);
    }

    #[test]
    fn test_decycler_trait_impl() {
        let dc = Decycler::default();
        assert_eq!(dc.name(), "Decycler");
        assert_eq!(dc.min_periods(), 3);
        assert_eq!(dc.output_features(), 2);
        assert_eq!(dc.period, 125);
    }
}
