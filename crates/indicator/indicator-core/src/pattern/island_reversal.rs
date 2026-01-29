//! Island Reversal Pattern Indicator (IND-343)
//!
//! Detects island reversal patterns - gapped isolation patterns
//! that signal strong reversals.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Island Reversal Pattern detection.
///
/// An island reversal occurs when price gaps away from a trend,
/// trades in a range (the "island"), and then gaps back in the
/// opposite direction, creating an isolated price cluster.
///
/// Pattern types:
/// - Bullish Island Reversal: Gap down, island, gap up
/// - Bearish Island Reversal: Gap up, island, gap down
#[derive(Debug, Clone)]
pub struct IslandReversal {
    /// Minimum gap size as percentage of price
    min_gap_percent: f64,
    /// Maximum island duration in bars
    max_island_bars: usize,
    /// Minimum island duration in bars
    min_island_bars: usize,
}

/// Output from island reversal detection
#[derive(Debug, Clone)]
pub struct IslandReversalOutput {
    /// Pattern signal: 1.0 = bullish, -1.0 = bearish, 0.0 = none
    pub signal: f64,
    /// Gap-in size (first gap)
    pub gap_in: f64,
    /// Gap-out size (second gap)
    pub gap_out: f64,
    /// Island duration in bars
    pub island_duration: usize,
    /// Pattern strength (based on gap sizes and symmetry)
    pub strength: f64,
}

impl IslandReversal {
    /// Create a new Island Reversal indicator.
    ///
    /// # Arguments
    /// * `min_gap_percent` - Minimum gap size as percentage (e.g., 1.0 = 1%)
    /// * `max_island_bars` - Maximum bars in island (typically 5-15)
    /// * `min_island_bars` - Minimum bars in island (typically 1-3)
    pub fn new(
        min_gap_percent: f64,
        max_island_bars: usize,
        min_island_bars: usize,
    ) -> Result<Self> {
        if min_gap_percent <= 0.0 || min_gap_percent > 10.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_gap_percent".to_string(),
                reason: "must be between 0 and 10 percent".to_string(),
            });
        }
        if max_island_bars < min_island_bars {
            return Err(IndicatorError::InvalidParameter {
                name: "max_island_bars".to_string(),
                reason: "must be >= min_island_bars".to_string(),
            });
        }
        if min_island_bars < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_island_bars".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            min_gap_percent,
            max_island_bars,
            min_island_bars,
        })
    }

    /// Create with default parameters
    pub fn default_params() -> Self {
        Self {
            min_gap_percent: 0.5,
            max_island_bars: 10,
            min_island_bars: 1,
        }
    }

    /// Calculate island reversal pattern detection.
    ///
    /// Returns: 1.0 = bullish reversal, -1.0 = bearish reversal, 0.0 = none
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = open.len().min(high.len()).min(low.len()).min(close.len());
        let mut result = vec![0.0; n];

        let min_lookback = self.min_island_bars + 2;
        if n < min_lookback {
            return result;
        }

        // Look for island reversals
        for i in (self.min_island_bars + 1)..n {
            // Check for bearish island reversal (gap up, then gap down)
            if let Some(bearish) = self.check_bearish_island(open, high, low, close, i) {
                result[i] = bearish;
            }
            // Check for bullish island reversal (gap down, then gap up)
            else if let Some(bullish) = self.check_bullish_island(open, high, low, close, i) {
                result[i] = bullish;
            }
        }

        result
    }

    /// Check for bearish island reversal at index i.
    fn check_bearish_island(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        _close: &[f64],
        i: usize,
    ) -> Option<f64> {
        // Current bar gaps down from previous
        let gap_out = low[i - 1] - high[i];
        let gap_out_percent = (gap_out / high[i]) * 100.0;

        if gap_out_percent < self.min_gap_percent {
            return None;
        }

        // Look backwards for gap-in (gap up)
        for island_len in self.min_island_bars..=self.max_island_bars.min(i - 1) {
            let gap_in_idx = i - island_len - 1;
            if gap_in_idx == 0 {
                continue;
            }

            let gap_in = low[gap_in_idx] - high[gap_in_idx - 1];
            let gap_in_percent = (gap_in / high[gap_in_idx - 1]) * 100.0;

            if gap_in_percent >= self.min_gap_percent {
                // Verify island is isolated (no overlap with pre-gap or post-gap)
                let island_high = high[gap_in_idx..i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let island_low = low[gap_in_idx..i].iter().fold(f64::INFINITY, |a, &b| a.min(b));

                // Check gap integrity
                let pre_gap_high = high[gap_in_idx - 1];
                let post_gap_low = low[i];

                if island_low > pre_gap_high && island_high > post_gap_low {
                    // Valid bearish island
                    let strength = ((gap_in_percent + gap_out_percent) / 2.0).min(1.0);
                    return Some(-strength);
                }
            }
        }

        None
    }

    /// Check for bullish island reversal at index i.
    fn check_bullish_island(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        _close: &[f64],
        i: usize,
    ) -> Option<f64> {
        // Current bar gaps up from previous
        let gap_out = low[i] - high[i - 1];
        let gap_out_percent = (gap_out / high[i - 1]) * 100.0;

        if gap_out_percent < self.min_gap_percent {
            return None;
        }

        // Look backwards for gap-in (gap down)
        for island_len in self.min_island_bars..=self.max_island_bars.min(i - 1) {
            let gap_in_idx = i - island_len - 1;
            if gap_in_idx == 0 {
                continue;
            }

            let gap_in = low[gap_in_idx - 1] - high[gap_in_idx];
            let gap_in_percent = (gap_in / low[gap_in_idx - 1]) * 100.0;

            if gap_in_percent >= self.min_gap_percent {
                // Verify island is isolated
                let island_high = high[gap_in_idx..i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let island_low = low[gap_in_idx..i].iter().fold(f64::INFINITY, |a, &b| a.min(b));

                let pre_gap_low = low[gap_in_idx - 1];
                let post_gap_high = high[i];

                if island_high < pre_gap_low && island_low < post_gap_high {
                    // Valid bullish island
                    let strength = ((gap_in_percent + gap_out_percent) / 2.0).min(1.0);
                    return Some(strength);
                }
            }
        }

        None
    }

    /// Calculate detailed island reversal output.
    pub fn calculate_detailed(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<IslandReversalOutput> {
        let n = open.len().min(high.len()).min(low.len()).min(close.len());
        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            if i < self.min_island_bars + 2 {
                results.push(IslandReversalOutput {
                    signal: 0.0,
                    gap_in: 0.0,
                    gap_out: 0.0,
                    island_duration: 0,
                    strength: 0.0,
                });
                continue;
            }

            let mut found = false;

            // Check bearish
            for island_len in self.min_island_bars..=self.max_island_bars.min(i - 1) {
                let gap_in_idx = i - island_len - 1;
                if gap_in_idx == 0 {
                    continue;
                }

                let gap_out = low[i - 1] - high[i];
                let gap_out_percent = (gap_out / high[i]) * 100.0;

                if gap_out_percent < self.min_gap_percent {
                    continue;
                }

                let gap_in = low[gap_in_idx] - high[gap_in_idx - 1];
                let gap_in_percent = (gap_in / high[gap_in_idx - 1]) * 100.0;

                if gap_in_percent >= self.min_gap_percent {
                    let strength = ((gap_in_percent + gap_out_percent) / 2.0).min(1.0);
                    results.push(IslandReversalOutput {
                        signal: -1.0,
                        gap_in,
                        gap_out,
                        island_duration: island_len,
                        strength,
                    });
                    found = true;
                    break;
                }
            }

            // Check bullish
            if !found {
                for island_len in self.min_island_bars..=self.max_island_bars.min(i - 1) {
                    let gap_in_idx = i - island_len - 1;
                    if gap_in_idx == 0 {
                        continue;
                    }

                    let gap_out = low[i] - high[i - 1];
                    let gap_out_percent = (gap_out / high[i - 1]) * 100.0;

                    if gap_out_percent < self.min_gap_percent {
                        continue;
                    }

                    let gap_in = low[gap_in_idx - 1] - high[gap_in_idx];
                    let gap_in_percent = (gap_in / low[gap_in_idx - 1]) * 100.0;

                    if gap_in_percent >= self.min_gap_percent {
                        let strength = ((gap_in_percent + gap_out_percent) / 2.0).min(1.0);
                        results.push(IslandReversalOutput {
                            signal: 1.0,
                            gap_in,
                            gap_out,
                            island_duration: island_len,
                            strength,
                        });
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                results.push(IslandReversalOutput {
                    signal: 0.0,
                    gap_in: 0.0,
                    gap_out: 0.0,
                    island_duration: 0,
                    strength: 0.0,
                });
            }
        }

        results
    }

    /// Detect gaps in price data.
    /// Returns (gap_size, gap_direction) for each bar.
    pub fn detect_gaps(&self, high: &[f64], low: &[f64]) -> Vec<(f64, f64)> {
        let n = high.len().min(low.len());
        let mut gaps = vec![(0.0, 0.0); n];

        for i in 1..n {
            // Gap up: current low > previous high
            if low[i] > high[i - 1] {
                let gap = low[i] - high[i - 1];
                let gap_percent = (gap / high[i - 1]) * 100.0;
                if gap_percent >= self.min_gap_percent {
                    gaps[i] = (gap, 1.0);
                }
            }
            // Gap down: current high < previous low
            else if high[i] < low[i - 1] {
                let gap = low[i - 1] - high[i];
                let gap_percent = (gap / low[i - 1]) * 100.0;
                if gap_percent >= self.min_gap_percent {
                    gaps[i] = (gap, -1.0);
                }
            }
        }

        gaps
    }
}

impl TechnicalIndicator for IslandReversal {
    fn name(&self) -> &str {
        "Island Reversal"
    }

    fn min_periods(&self) -> usize {
        self.max_island_bars + 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }
}

impl SignalIndicator for IslandReversal {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);

        if let Some(&last) = values.last() {
            if last > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if last < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        let signals = values
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    IndicatorSignal::Bullish
                } else if v < 0.0 {
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

    fn make_bullish_island() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Prices: uptrend, gap down, island, gap up
        let open = vec![
            100.0, 102.0, 104.0, // Uptrend
            95.0, 96.0, 95.5,    // Gap down + island
            102.0, 103.0, 104.0, // Gap up + continuation
        ];
        let high = vec![
            101.0, 103.0, 105.0,
            97.0, 97.5, 97.0,
            104.0, 105.0, 106.0,
        ];
        let low = vec![
            99.0, 101.0, 103.0,
            94.0, 95.0, 94.5,
            101.0, 102.0, 103.0,
        ];
        let close = vec![
            100.5, 102.5, 104.5,
            96.0, 96.5, 96.0,
            103.0, 104.0, 105.0,
        ];
        (open, high, low, close)
    }

    #[test]
    fn test_island_reversal_creation() {
        let ir = IslandReversal::new(0.5, 10, 1);
        assert!(ir.is_ok());

        let ir_invalid = IslandReversal::new(0.5, 5, 10);
        assert!(ir_invalid.is_err());
    }

    #[test]
    fn test_gap_detection() {
        let ir = IslandReversal::default_params();

        let high = vec![100.0, 95.0, 97.0]; // Gap down between 0 and 1
        let low = vec![98.0, 93.0, 95.0];

        let gaps = ir.detect_gaps(&high, &low);
        assert_eq!(gaps.len(), 3);
    }

    #[test]
    fn test_island_pattern_detection() {
        let (open, high, low, close) = make_bullish_island();
        let ir = IslandReversal::new(0.5, 5, 1).unwrap();

        let result = ir.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), open.len());
    }

    #[test]
    fn test_detailed_output() {
        let (open, high, low, close) = make_bullish_island();
        let ir = IslandReversal::default_params();

        let detailed = ir.calculate_detailed(&open, &high, &low, &close);
        assert_eq!(detailed.len(), open.len());
    }
}
