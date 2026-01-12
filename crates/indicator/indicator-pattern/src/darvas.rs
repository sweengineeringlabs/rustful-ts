//! Darvas Box Theory Indicator
//!
//! Identifies trading boxes based on Nicolas Darvas's box theory.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Darvas Box indicator for trend-following breakout strategies.
///
/// The Darvas Box theory identifies consolidation boxes and signals breakouts.
/// A box is formed when price makes a new high and then consolidates for
/// a specified number of periods before breaking out.
#[derive(Debug, Clone)]
pub struct DarvasBox {
    /// Number of periods to confirm box formation.
    lookback: usize,
}

impl DarvasBox {
    /// Create a new Darvas Box indicator.
    ///
    /// # Arguments
    /// * `lookback` - Number of periods to confirm the box (typically 3-5)
    pub fn new(lookback: usize) -> Self {
        Self { lookback }
    }

    /// Calculate Darvas Box levels.
    ///
    /// Returns (box_top, box_bottom) vectors.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n < self.lookback + 1 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut box_top = vec![f64::NAN; n];
        let mut box_bottom = vec![f64::NAN; n];

        let mut current_top = high[0];
        let mut current_bottom = low[0];
        let mut top_confirmed = false;
        let mut bottom_confirmed = false;
        let mut top_count = 0;
        let mut bottom_count = 0;

        for i in 1..n {
            // Check for new high
            if high[i] > current_top {
                current_top = high[i];
                top_confirmed = false;
                top_count = 0;
                // Reset bottom search
                current_bottom = low[i];
                bottom_confirmed = false;
                bottom_count = 0;
            } else {
                top_count += 1;
                if top_count >= self.lookback {
                    top_confirmed = true;
                }
            }

            // If top is confirmed, look for bottom
            if top_confirmed {
                if !bottom_confirmed {
                    if low[i] < current_bottom {
                        current_bottom = low[i];
                        bottom_count = 0;
                    } else {
                        bottom_count += 1;
                        if bottom_count >= self.lookback {
                            bottom_confirmed = true;
                        }
                    }
                }

                // Check for breakout above box
                if bottom_confirmed && close[i] > current_top {
                    // Breakout - reset for new box
                    current_top = high[i];
                    top_confirmed = false;
                    top_count = 0;
                    current_bottom = low[i];
                    bottom_confirmed = false;
                    bottom_count = 0;
                }
            }

            // Record current box levels if both are confirmed
            if top_confirmed {
                box_top[i] = current_top;
            }
            if bottom_confirmed {
                box_bottom[i] = current_bottom;
            }
        }

        (box_top, box_bottom)
    }

    /// Detect breakout signals.
    /// Returns 1.0 for bullish breakout, -1.0 for bearish breakdown, 0.0 otherwise.
    pub fn breakout_signals(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let (box_top, box_bottom) = self.calculate(high, low, close);
        let mut signals = vec![0.0; n];

        for i in 1..n {
            if !box_top[i].is_nan() && !box_bottom[i].is_nan() {
                // Bullish breakout above box top
                if close[i] > box_top[i] && close[i - 1] <= box_top[i - 1].max(box_top[i]) {
                    signals[i] = 1.0;
                }
                // Bearish breakdown below box bottom
                else if close[i] < box_bottom[i] && close[i - 1] >= box_bottom[i - 1].min(box_bottom[i]) {
                    signals[i] = -1.0;
                }
            }
        }

        signals
    }
}

impl TechnicalIndicator for DarvasBox {
    fn name(&self) -> &str {
        "DarvasBox"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.lookback + 1;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (box_top, box_bottom) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(box_top, box_bottom))
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn output_features(&self) -> usize {
        2 // box_top, box_bottom
    }
}

impl SignalIndicator for DarvasBox {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signals = self.breakout_signals(&data.high, &data.low, &data.close);

        if let Some(&last) = signals.last() {
            if last > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if last < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let breakouts = self.breakout_signals(&data.high, &data.low, &data.close);
        let signals = breakouts.iter().map(|&s| {
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
    fn test_darvas_box_basic() {
        let darvas = DarvasBox::new(3);
        let high = vec![100.0, 105.0, 104.0, 103.0, 102.0, 106.0, 107.0, 106.0, 105.0, 104.0];
        let low = vec![95.0, 100.0, 99.0, 98.0, 97.0, 101.0, 102.0, 101.0, 100.0, 99.0];
        let close = vec![98.0, 103.0, 101.0, 100.0, 99.0, 104.0, 105.0, 103.0, 102.0, 101.0];

        let (box_top, box_bottom) = darvas.calculate(&high, &low, &close);
        assert_eq!(box_top.len(), 10);
        assert_eq!(box_bottom.len(), 10);
    }

    #[test]
    fn test_darvas_breakout() {
        let darvas = DarvasBox::new(2);
        // Create a clear box formation and breakout
        let high = vec![100.0, 105.0, 104.0, 103.0, 102.0, 101.0, 110.0];
        let low = vec![95.0, 100.0, 99.0, 98.0, 97.0, 96.0, 105.0];
        let close = vec![98.0, 103.0, 101.0, 100.0, 99.0, 98.0, 108.0];

        let signals = darvas.breakout_signals(&high, &low, &close);
        assert_eq!(signals.len(), 7);
    }
}
