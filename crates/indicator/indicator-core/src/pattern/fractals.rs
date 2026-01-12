//! Williams Fractals Indicator
//!
//! Identifies reversal points using Bill Williams' fractal pattern.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Williams Fractals indicator for identifying potential reversal points.
///
/// A fractal is a reversal point in the market:
/// - Bullish fractal (down fractal): A low surrounded by higher lows
/// - Bearish fractal (up fractal): A high surrounded by lower highs
///
/// The classic fractal requires 5 bars (2 on each side of the pivot).
#[derive(Debug, Clone)]
pub struct Fractals {
    /// Number of bars on each side of the pivot (default: 2)
    period: usize,
}

impl Fractals {
    /// Create a new Fractals indicator.
    ///
    /// # Arguments
    /// * `period` - Number of bars on each side (2 = classic 5-bar fractal)
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Create classic 5-bar fractals (2 bars on each side).
    pub fn classic() -> Self {
        Self { period: 2 }
    }

    /// Calculate fractal signals.
    ///
    /// Returns (up_fractals, down_fractals) where:
    /// - up_fractals: Contains high price at bearish fractal, NaN otherwise
    /// - down_fractals: Contains low price at bullish fractal, NaN otherwise
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        let min_len = self.period * 2 + 1;

        if n < min_len {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut up_fractals = vec![f64::NAN; n];
        let mut down_fractals = vec![f64::NAN; n];

        for i in self.period..(n - self.period) {
            // Check for up fractal (bearish - high surrounded by lower highs)
            let mut is_up_fractal = true;
            for j in 1..=self.period {
                if high[i - j] >= high[i] || high[i + j] >= high[i] {
                    is_up_fractal = false;
                    break;
                }
            }
            if is_up_fractal {
                up_fractals[i] = high[i];
            }

            // Check for down fractal (bullish - low surrounded by higher lows)
            let mut is_down_fractal = true;
            for j in 1..=self.period {
                if low[i - j] <= low[i] || low[i + j] <= low[i] {
                    is_down_fractal = false;
                    break;
                }
            }
            if is_down_fractal {
                down_fractals[i] = low[i];
            }
        }

        (up_fractals, down_fractals)
    }

    /// Get fractal points as (index, price, is_up) tuples.
    pub fn fractal_points(&self, high: &[f64], low: &[f64]) -> Vec<(usize, f64, bool)> {
        let (up, down) = self.calculate(high, low);
        let mut points = Vec::new();

        for (i, (&u, &d)) in up.iter().zip(down.iter()).enumerate() {
            if !u.is_nan() {
                points.push((i, u, true));
            }
            if !d.is_nan() {
                points.push((i, d, false));
            }
        }

        points
    }

    /// Calculate combined signal: 1.0 for down fractal, -1.0 for up fractal.
    pub fn signal_values(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let (up, down) = self.calculate(high, low);
        up.iter()
            .zip(down.iter())
            .map(|(&u, &d)| {
                if !d.is_nan() {
                    1.0 // Bullish (down fractal)
                } else if !u.is_nan() {
                    -1.0 // Bearish (up fractal)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

impl TechnicalIndicator for Fractals {
    fn name(&self) -> &str {
        "Fractals"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.period * 2 + 1;
        if data.high.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.high.len(),
            });
        }

        let (up_fractals, down_fractals) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(up_fractals, down_fractals))
    }

    fn min_periods(&self) -> usize {
        self.period * 2 + 1
    }

    fn output_features(&self) -> usize {
        2 // up_fractals, down_fractals
    }
}

impl SignalIndicator for Fractals {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (up, down) = self.calculate(&data.high, &data.low);

        // Find the most recent fractal
        for i in (0..up.len()).rev() {
            if !down[i].is_nan() {
                return Ok(IndicatorSignal::Bullish);
            }
            if !up[i].is_nan() {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let signal_vals = self.signal_values(&data.high, &data.low);
        let signals = signal_vals.iter().map(|&s| {
            if s > 0.0 {
                IndicatorSignal::Bullish
            } else if s < 0.0 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractals_basic() {
        let fractals = Fractals::classic();
        // Classic V-pattern for down fractal
        let high = vec![105.0, 103.0, 100.0, 103.0, 105.0];
        let low = vec![100.0, 98.0, 95.0, 98.0, 100.0];

        let (up, down) = fractals.calculate(&high, &low);

        // Middle bar (index 2) should have a down fractal (lowest low)
        assert!(!down[2].is_nan());
        assert_eq!(down[2], 95.0);
    }

    #[test]
    fn test_fractals_up() {
        let fractals = Fractals::classic();
        // Inverted V-pattern for up fractal
        let high = vec![100.0, 103.0, 105.0, 103.0, 100.0];
        let low = vec![95.0, 98.0, 100.0, 98.0, 95.0];

        let (up, down) = fractals.calculate(&high, &low);

        // Middle bar (index 2) should have an up fractal (highest high)
        assert!(!up[2].is_nan());
        assert_eq!(up[2], 105.0);
    }

    #[test]
    fn test_fractals_min_periods() {
        let fractals = Fractals::new(3);
        assert_eq!(fractals.min_periods(), 7);

        let fractals = Fractals::classic();
        assert_eq!(fractals.min_periods(), 5);
    }
}
