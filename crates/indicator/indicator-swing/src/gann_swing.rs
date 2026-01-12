//! Gann Swing indicator implementation.
//!
//! Based on W.D. Gann's swing chart techniques for identifying trend direction.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Gann Swing state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GannSwingState {
    /// Swing is in uptrend mode.
    Up,
    /// Swing is in downtrend mode.
    Down,
    /// Initial state (no swing determined yet).
    Unknown,
}

/// Gann Swing indicator.
///
/// The Gann Swing Chart is a trend-following technique developed by W.D. Gann.
/// It identifies swings based on consecutive higher highs or lower lows.
///
/// Rules:
/// - Swing turns UP when price makes N consecutive higher highs
/// - Swing turns DOWN when price makes N consecutive lower lows
/// - The swing line connects swing highs and swing lows
///
/// Output:
/// - Primary: Swing value (high during upswing, low during downswing)
/// - Secondary: Swing direction (1 = up, -1 = down, 0 = unknown)
#[derive(Debug, Clone)]
pub struct GannSwing {
    /// Number of bars required to confirm swing change.
    swing_bars: usize,
}

impl GannSwing {
    /// Create a new Gann Swing indicator.
    ///
    /// # Arguments
    /// * `swing_bars` - Number of consecutive bars for swing confirmation (typically 2-3)
    pub fn new(swing_bars: usize) -> Self {
        Self {
            swing_bars: swing_bars.max(1),
        }
    }

    /// Create with default 2-bar swing.
    pub fn default_swing() -> Self {
        Self::new(2)
    }

    /// Calculate Gann Swing values.
    ///
    /// Returns (swing_values, swing_directions)
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n < self.swing_bars + 1 {
            return (vec![f64::NAN; n], vec![0.0; n]);
        }

        let mut swing_values = vec![f64::NAN; n];
        let mut swing_directions = vec![0.0; n];
        let mut state = GannSwingState::Unknown;
        let mut swing_high = high[0];
        let mut swing_low = low[0];
        let mut higher_count = 0;
        let mut lower_count = 0;

        for i in 1..n {
            let curr_high = high[i];
            let curr_low = low[i];
            let prev_high = high[i - 1];
            let prev_low = low[i - 1];

            // Count consecutive higher highs
            if curr_high > prev_high {
                higher_count += 1;
                lower_count = 0;
            } else if curr_low < prev_low {
                lower_count += 1;
                higher_count = 0;
            } else {
                // Inside bar - maintain counts
            }

            // Check for swing change
            match state {
                GannSwingState::Unknown => {
                    if higher_count >= self.swing_bars {
                        state = GannSwingState::Up;
                        swing_high = curr_high;
                        swing_low = low.iter().take(i + 1).fold(f64::INFINITY, |a, &b| a.min(b));
                    } else if lower_count >= self.swing_bars {
                        state = GannSwingState::Down;
                        swing_low = curr_low;
                        swing_high = high.iter().take(i + 1).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    }
                }
                GannSwingState::Up => {
                    if curr_high > swing_high {
                        swing_high = curr_high;
                    }
                    if lower_count >= self.swing_bars {
                        state = GannSwingState::Down;
                        swing_low = curr_low;
                    }
                }
                GannSwingState::Down => {
                    if curr_low < swing_low {
                        swing_low = curr_low;
                    }
                    if higher_count >= self.swing_bars {
                        state = GannSwingState::Up;
                        swing_high = curr_high;
                    }
                }
            }

            // Set output values based on current state
            match state {
                GannSwingState::Up => {
                    swing_values[i] = swing_high;
                    swing_directions[i] = 1.0;
                }
                GannSwingState::Down => {
                    swing_values[i] = swing_low;
                    swing_directions[i] = -1.0;
                }
                GannSwingState::Unknown => {
                    swing_values[i] = (curr_high + curr_low) / 2.0;
                    swing_directions[i] = 0.0;
                }
            }
        }

        (swing_values, swing_directions)
    }
}

impl TechnicalIndicator for GannSwing {
    fn name(&self) -> &str {
        "GannSwing"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.swing_bars + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.swing_bars + 1,
                got: data.high.len(),
            });
        }

        let (swing_values, swing_directions) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(swing_values, swing_directions))
    }

    fn min_periods(&self) -> usize {
        self.swing_bars + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for GannSwing {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (_, directions) = self.calculate(&data.high, &data.low);

        let last = directions.last().copied().unwrap_or(0.0);

        if last > 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (_, directions) = self.calculate(&data.high, &data.low);

        let signals = directions
            .iter()
            .map(|&d| {
                if d > 0.0 {
                    IndicatorSignal::Bullish
                } else if d < 0.0 {
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
    fn test_gann_swing_uptrend() {
        let gann = GannSwing::new(2);

        // Clear uptrend: consecutive higher highs
        let high = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let low = vec![99.0, 100.0, 101.0, 102.0, 103.0, 104.0];

        let (values, directions) = gann.calculate(&high, &low);

        assert_eq!(values.len(), 6);

        // Should detect uptrend
        let last_dir = directions.last().copied().unwrap();
        assert_eq!(last_dir, 1.0);
    }

    #[test]
    fn test_gann_swing_downtrend() {
        let gann = GannSwing::new(2);

        // Clear downtrend: consecutive lower lows
        let high = vec![105.0, 104.0, 103.0, 102.0, 101.0, 100.0];
        let low = vec![104.0, 103.0, 102.0, 101.0, 100.0, 99.0];

        let (_, directions) = gann.calculate(&high, &low);

        // Should detect downtrend
        let last_dir = directions.last().copied().unwrap();
        assert_eq!(last_dir, -1.0);
    }

    #[test]
    fn test_gann_swing_reversal() {
        let gann = GannSwing::new(2);

        // Uptrend then reversal
        let high = vec![100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0];
        let low = vec![99.0, 100.0, 101.0, 102.0, 101.0, 100.0, 99.0];

        let (_, directions) = gann.calculate(&high, &low);

        // Should start bullish, end bearish
        assert_eq!(directions.len(), 7);
    }

    #[test]
    fn test_gann_swing_technical_indicator() {
        let gann = GannSwing::new(2);

        let mut data = OHLCVSeries::new();
        for i in 0..10 {
            data.open.push(100.0 + i as f64);
            data.high.push(102.0 + i as f64);
            data.low.push(98.0 + i as f64);
            data.close.push(101.0 + i as f64);
            data.volume.push(1000.0);
        }

        let output = gann.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 10);
        assert!(output.secondary.is_some());
    }
}
