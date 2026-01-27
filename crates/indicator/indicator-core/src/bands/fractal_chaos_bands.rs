//! Fractal Chaos Bands implementation.
//!
//! Bands based on Williams Fractals for support and resistance.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Fractal Chaos Bands indicator.
///
/// Creates dynamic support and resistance bands based on Williams Fractals.
/// The bands track the most recent fractal highs and lows, creating a channel
/// that adapts to market structure.
///
/// - Upper Band: Most recent fractal high (resistance)
/// - Lower Band: Most recent fractal low (support)
/// - Middle Band: Midpoint between upper and lower
///
/// Fractal Chaos Bands are useful for identifying:
/// - Key support and resistance levels
/// - Potential breakout points
/// - Trend direction based on band slope
#[derive(Debug, Clone)]
pub struct FractalChaosBands {
    /// Number of bars on each side for fractal detection.
    period: usize,
}

impl FractalChaosBands {
    /// Create a new FractalChaosBands indicator.
    ///
    /// # Arguments
    /// * `period` - Number of bars on each side for fractal detection (default: 2)
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Create with classic 5-bar fractals (2 bars on each side).
    pub fn classic() -> Self {
        Self { period: 2 }
    }

    /// Detect up fractals (local highs).
    fn detect_up_fractals(&self, high: &[f64]) -> Vec<Option<f64>> {
        let n = high.len();
        let mut fractals = vec![None; n];

        if n < self.period * 2 + 1 {
            return fractals;
        }

        for i in self.period..(n - self.period) {
            let mut is_fractal = true;
            for j in 1..=self.period {
                if high[i - j] >= high[i] || high[i + j] >= high[i] {
                    is_fractal = false;
                    break;
                }
            }
            if is_fractal {
                fractals[i] = Some(high[i]);
            }
        }

        fractals
    }

    /// Detect down fractals (local lows).
    fn detect_down_fractals(&self, low: &[f64]) -> Vec<Option<f64>> {
        let n = low.len();
        let mut fractals = vec![None; n];

        if n < self.period * 2 + 1 {
            return fractals;
        }

        for i in self.period..(n - self.period) {
            let mut is_fractal = true;
            for j in 1..=self.period {
                if low[i - j] <= low[i] || low[i + j] <= low[i] {
                    is_fractal = false;
                    break;
                }
            }
            if is_fractal {
                fractals[i] = Some(low[i]);
            }
        }

        fractals
    }

    /// Calculate Fractal Chaos Bands (upper, middle, lower).
    ///
    /// The bands are forward-filled from the most recent fractal values.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n < self.period * 2 + 1 {
            return (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let up_fractals = self.detect_up_fractals(high);
        let down_fractals = self.detect_down_fractals(low);

        let mut upper = Vec::with_capacity(n);
        let mut lower = Vec::with_capacity(n);
        let mut middle = Vec::with_capacity(n);

        let mut last_upper = f64::NAN;
        let mut last_lower = f64::NAN;

        for i in 0..n {
            // Update upper band from up fractal
            if let Some(val) = up_fractals[i] {
                last_upper = val;
            }

            // Update lower band from down fractal
            if let Some(val) = down_fractals[i] {
                last_lower = val;
            }

            upper.push(last_upper);
            lower.push(last_lower);

            // Middle is the midpoint
            if last_upper.is_nan() || last_lower.is_nan() {
                middle.push(f64::NAN);
            } else {
                middle.push((last_upper + last_lower) / 2.0);
            }
        }

        (upper, middle, lower)
    }

    /// Calculate raw fractal points (for visualization).
    /// Returns (up_fractals, down_fractals) with NaN for non-fractal bars.
    pub fn fractal_points(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let up_fractals = self.detect_up_fractals(high);
        let down_fractals = self.detect_down_fractals(low);

        let up: Vec<f64> = up_fractals.iter()
            .map(|f| f.unwrap_or(f64::NAN))
            .collect();
        let down: Vec<f64> = down_fractals.iter()
            .map(|f| f.unwrap_or(f64::NAN))
            .collect();

        (up, down)
    }

    /// Calculate band width (distance between upper and lower).
    pub fn band_width(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let (upper, _, lower) = self.calculate(high, low);
        upper.iter()
            .zip(lower.iter())
            .map(|(&u, &l)| {
                if u.is_nan() || l.is_nan() {
                    f64::NAN
                } else {
                    u - l
                }
            })
            .collect()
    }

    /// Calculate position within bands (0 = at lower, 1 = at upper).
    pub fn position(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let (upper, _, lower) = self.calculate(high, low);
        close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() || (u - l).abs() < 1e-10 {
                    f64::NAN
                } else {
                    (price - l) / (u - l)
                }
            })
            .collect()
    }
}

impl Default for FractalChaosBands {
    fn default() -> Self {
        Self::classic()
    }
}

impl TechnicalIndicator for FractalChaosBands {
    fn name(&self) -> &str {
        "FractalChaosBands"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.period * 2 + 1;
        if data.high.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.high.len(),
            });
        }

        let (upper, middle, lower) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::triple(upper, middle, lower))
    }

    fn min_periods(&self) -> usize {
        self.period * 2 + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for FractalChaosBands {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (upper, _, lower) = self.calculate(&data.high, &data.low);

        let n = data.close.len();
        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let price = data.close[n - 1];
        let u = upper[n - 1];
        let l = lower[n - 1];

        if u.is_nan() || l.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Price breaking above upper band = bullish breakout
        if price > u {
            Ok(IndicatorSignal::Bullish)
        }
        // Price breaking below lower band = bearish breakdown
        else if price < l {
            Ok(IndicatorSignal::Bearish)
        }
        // Price within bands
        else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (upper, _, lower) = self.calculate(&data.high, &data.low);

        let signals: Vec<_> = data.close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() {
                    IndicatorSignal::Neutral
                } else if price > u {
                    IndicatorSignal::Bullish
                } else if price < l {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractal_chaos_bands_basic() {
        let fcb = FractalChaosBands::classic();

        // Create data with clear fractal patterns
        let high = vec![
            100.0, 102.0, 105.0, 103.0, 101.0,  // Up fractal at index 2
            99.0, 98.0, 96.0, 97.0, 99.0,       // Down fractal at index 7
            101.0, 103.0, 106.0, 104.0, 102.0,  // Up fractal at index 12
        ];
        let low = vec![
            90.0, 92.0, 95.0, 93.0, 91.0,
            89.0, 88.0, 86.0, 87.0, 89.0,       // Down fractal at index 7
            91.0, 93.0, 96.0, 94.0, 92.0,
        ];

        let (upper, middle, lower) = fcb.calculate(&high, &low);

        assert_eq!(upper.len(), 15);
        assert_eq!(middle.len(), 15);
        assert_eq!(lower.len(), 15);

        // Check that up fractal at index 2 sets upper band
        assert!((upper[2] - 105.0).abs() < 1e-10);

        // Check that down fractal at index 7 sets lower band
        assert!((lower[7] - 86.0).abs() < 1e-10);

        // Upper band should persist
        assert!((upper[5] - 105.0).abs() < 1e-10);
    }

    #[test]
    fn test_fractal_chaos_bands_forward_fill() {
        let fcb = FractalChaosBands::classic();

        // Single up fractal at index 2
        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 100.0, 99.0, 98.0];
        let low = vec![90.0, 92.0, 95.0, 93.0, 91.0, 90.0, 89.0, 88.0];

        let (upper, _, _) = fcb.calculate(&high, &low);

        // Upper band should be forward-filled from index 2
        for i in 2..8 {
            assert!((upper[i] - 105.0).abs() < 1e-10, "Upper should persist at index {}", i);
        }
    }

    #[test]
    fn test_fractal_chaos_bands_middle() {
        let fcb = FractalChaosBands::classic();

        let high = vec![100.0, 102.0, 110.0, 103.0, 101.0, 99.0, 98.0, 96.0, 97.0, 99.0];
        let low = vec![90.0, 92.0, 100.0, 93.0, 91.0, 89.0, 88.0, 86.0, 87.0, 89.0];

        let (upper, middle, lower) = fcb.calculate(&high, &low);

        // After both fractals are set, middle should be the average
        for i in 7..10 {
            if !upper[i].is_nan() && !lower[i].is_nan() {
                let expected_middle = (upper[i] + lower[i]) / 2.0;
                assert!((middle[i] - expected_middle).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_fractal_chaos_bands_default() {
        let fcb = FractalChaosBands::default();
        assert_eq!(fcb.period, 2);
    }

    #[test]
    fn test_fractal_chaos_bands_period() {
        let fcb = FractalChaosBands::new(3);
        assert_eq!(fcb.period, 3);
        assert_eq!(fcb.min_periods(), 7);
    }

    #[test]
    fn test_fractal_points() {
        let fcb = FractalChaosBands::classic();

        // Inverted V for up fractal
        let high = vec![100.0, 103.0, 105.0, 103.0, 100.0];
        let low = vec![95.0, 98.0, 100.0, 98.0, 95.0];

        let (up, down) = fcb.fractal_points(&high, &low);

        assert_eq!(up.len(), 5);
        assert_eq!(down.len(), 5);

        // Up fractal at index 2
        assert!(!up[2].is_nan());
        assert!((up[2] - 105.0).abs() < 1e-10);

        // No down fractal (inverted V pattern)
        for i in 0..5 {
            assert!(down[i].is_nan());
        }
    }

    #[test]
    fn test_fractal_chaos_bands_band_width() {
        let fcb = FractalChaosBands::classic();

        let high = vec![100.0, 102.0, 110.0, 103.0, 101.0, 99.0, 98.0, 90.0, 97.0, 99.0];
        let low = vec![90.0, 92.0, 100.0, 93.0, 91.0, 89.0, 88.0, 80.0, 87.0, 89.0];

        let width = fcb.band_width(&high, &low);

        assert_eq!(width.len(), 10);

        // After both fractals, width should be upper - lower
        for i in 7..10 {
            if !width[i].is_nan() {
                let (upper, _, lower) = fcb.calculate(&high, &low);
                let expected = upper[i] - lower[i];
                assert!((width[i] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_fractal_chaos_bands_position() {
        let fcb = FractalChaosBands::classic();

        let high = vec![100.0, 102.0, 110.0, 103.0, 101.0, 99.0, 98.0, 90.0, 97.0, 99.0];
        let low = vec![90.0, 92.0, 100.0, 93.0, 91.0, 89.0, 88.0, 80.0, 87.0, 89.0];
        let close = vec![95.0, 97.0, 105.0, 98.0, 96.0, 94.0, 93.0, 85.0, 92.0, 94.0];

        let position = fcb.position(&high, &low, &close);

        assert_eq!(position.len(), 10);

        // Position should be between 0 and 1 when price is within bands
        for i in 7..10 {
            if !position[i].is_nan() {
                // Position can be outside 0-1 if price breaks bands
                assert!(position[i].is_finite());
            }
        }
    }

    #[test]
    fn test_fractal_chaos_bands_signal_breakout() {
        let fcb = FractalChaosBands::classic();

        // Create pattern with breakout above resistance
        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 102.0, 108.0]; // Breaks above 105
        let low = vec![90.0, 92.0, 95.0, 93.0, 91.0, 92.0, 100.0];
        let close = vec![95.0, 97.0, 100.0, 98.0, 96.0, 97.0, 107.0]; // Last close > 105

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 7],
        };

        let signal = fcb.signal(&data).unwrap();

        // Price is above upper band (105), should be bullish
        assert_eq!(signal, IndicatorSignal::Bullish);
    }

    #[test]
    fn test_fractal_chaos_bands_signal_breakdown() {
        let fcb = FractalChaosBands::classic();

        // Create pattern with breakdown below support
        let high = vec![100.0, 98.0, 95.0, 97.0, 99.0, 98.0, 90.0];
        let low = vec![90.0, 88.0, 85.0, 87.0, 89.0, 88.0, 80.0]; // Breaks below 85
        let close = vec![95.0, 93.0, 90.0, 92.0, 94.0, 93.0, 82.0]; // Last close < 85

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 7],
        };

        let signal = fcb.signal(&data).unwrap();

        // Price is below lower band (85), should be bearish
        assert_eq!(signal, IndicatorSignal::Bearish);
    }

    #[test]
    fn test_fractal_chaos_bands_compute() {
        let fcb = FractalChaosBands::classic();

        let high: Vec<f64> = (0..30).map(|i| 105.0 + (i as f64 * 0.3).sin() * 5.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 95.0 + (i as f64 * 0.3).sin() * 5.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0).collect();

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 30],
        };

        let output = fcb.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_fractal_chaos_bands_insufficient_data() {
        let fcb = FractalChaosBands::classic();

        let data = OHLCVSeries {
            open: vec![100.0; 4],
            high: vec![101.0; 4],
            low: vec![99.0; 4],
            close: vec![100.0; 4],
            volume: vec![1000.0; 4],
        };

        let result = fcb.compute(&data);
        assert!(result.is_err());

        match result {
            Err(IndicatorError::InsufficientData { required, got }) => {
                assert_eq!(required, 5);
                assert_eq!(got, 4);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_fractal_chaos_bands_signals() {
        let fcb = FractalChaosBands::classic();

        let high = vec![100.0, 102.0, 105.0, 103.0, 101.0, 102.0, 103.0];
        let low = vec![90.0, 92.0, 95.0, 93.0, 91.0, 92.0, 93.0];
        let close = vec![95.0, 97.0, 100.0, 98.0, 96.0, 97.0, 98.0];

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 7],
        };

        let signals = fcb.signals(&data).unwrap();
        assert_eq!(signals.len(), 7);
    }
}
