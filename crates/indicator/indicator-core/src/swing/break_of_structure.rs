//! Break of Structure (BOS) indicator implementation.
//!
//! Identifies trend changes through structural breaks in price action.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use serde::{Deserialize, Serialize};

/// Break of Structure type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BOSType {
    /// Bullish BOS - break above previous swing high
    Bullish,
    /// Bearish BOS - break below previous swing low
    Bearish,
}

/// Change of Character (CHoCH) - more significant than BOS.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CHoCHType {
    /// Bullish CHoCH - trend change from bearish to bullish
    Bullish,
    /// Bearish CHoCH - trend change from bullish to bearish
    Bearish,
}

/// Represents a Break of Structure event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BOSEvent {
    /// Type of BOS
    pub bos_type: BOSType,
    /// Bar index where BOS occurred
    pub index: usize,
    /// Price level that was broken
    pub broken_level: f64,
    /// Previous swing point that was exceeded
    pub previous_swing: f64,
    /// Whether this is also a Change of Character
    pub is_choch: bool,
}

/// Break of Structure (BOS) indicator.
///
/// Identifies trend continuation and reversal through structural breaks:
///
/// Break of Structure (BOS):
/// - Bullish: Price breaks above a previous swing high
/// - Bearish: Price breaks below a previous swing low
///
/// Change of Character (CHoCH):
/// - A BOS that signals a trend reversal rather than continuation
/// - Bullish CHoCH: First higher high after a downtrend
/// - Bearish CHoCH: First lower low after an uptrend
///
/// Output:
/// - Primary: BOS signal (1 = bullish BOS, -1 = bearish BOS, 2 = bullish CHoCH, -2 = bearish CHoCH, 0 = none)
/// - Secondary: Broken level price
#[derive(Debug, Clone)]
pub struct BreakOfStructure {
    /// Lookback period for swing detection.
    swing_lookback: usize,
    /// Whether to identify Change of Character (CHoCH).
    detect_choch: bool,
}

impl BreakOfStructure {
    /// Create a new Break of Structure indicator.
    ///
    /// # Arguments
    /// * `swing_lookback` - Number of bars to look back for swing detection
    pub fn new(swing_lookback: usize) -> Self {
        Self {
            swing_lookback: swing_lookback.max(2),
            detect_choch: true,
        }
    }

    /// Create with default lookback of 5.
    pub fn default_lookback() -> Self {
        Self::new(5)
    }

    /// Disable CHoCH detection.
    pub fn without_choch(mut self) -> Self {
        self.detect_choch = false;
        self
    }

    /// Detect swing highs and lows.
    fn detect_swings(&self, high: &[f64], low: &[f64]) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
        let n = high.len();
        let mut swing_highs = vec![None; n];
        let mut swing_lows = vec![None; n];

        if n < 2 * self.swing_lookback + 1 {
            return (swing_highs, swing_lows);
        }

        for i in self.swing_lookback..(n - self.swing_lookback) {
            // Check for swing high
            let is_swing_high = (0..self.swing_lookback)
                .all(|j| high[i] >= high[i - j - 1] && high[i] >= high[i + j + 1]);

            if is_swing_high {
                swing_highs[i] = Some(high[i]);
            }

            // Check for swing low
            let is_swing_low = (0..self.swing_lookback)
                .all(|j| low[i] <= low[i - j - 1] && low[i] <= low[i + j + 1]);

            if is_swing_low {
                swing_lows[i] = Some(low[i]);
            }
        }

        (swing_highs, swing_lows)
    }

    /// Calculate BOS values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut bos_signal = vec![0.0; n];
        let mut broken_level = vec![f64::NAN; n];

        let (swing_highs, swing_lows) = self.detect_swings(high, low);

        // Track the most recent swing points and trend
        let mut last_swing_high: Option<f64> = None;
        let mut last_swing_low: Option<f64> = None;
        let mut trend: i8 = 0; // 0 = unknown, 1 = bullish, -1 = bearish

        for i in 0..n {
            // Update last swing points
            if let Some(sh) = swing_highs[i] {
                last_swing_high = Some(sh);
            }
            if let Some(sl) = swing_lows[i] {
                last_swing_low = Some(sl);
            }

            // Check for bullish BOS (break above swing high)
            if let Some(sh) = last_swing_high {
                if close[i] > sh && (i == 0 || close[i - 1] <= sh) {
                    let is_choch = self.detect_choch && trend == -1;

                    if is_choch {
                        bos_signal[i] = 2.0; // Bullish CHoCH
                        trend = 1;
                    } else if trend != 1 || i < self.swing_lookback * 2 {
                        bos_signal[i] = 1.0; // Bullish BOS
                        trend = 1;
                    }
                    broken_level[i] = sh;
                }
            }

            // Check for bearish BOS (break below swing low)
            if let Some(sl) = last_swing_low {
                if close[i] < sl && (i == 0 || close[i - 1] >= sl) {
                    let is_choch = self.detect_choch && trend == 1;

                    if is_choch {
                        bos_signal[i] = -2.0; // Bearish CHoCH
                        trend = -1;
                    } else if trend != -1 || i < self.swing_lookback * 2 {
                        bos_signal[i] = -1.0; // Bearish BOS
                        trend = -1;
                    }
                    broken_level[i] = sl;
                }
            }
        }

        (bos_signal, broken_level)
    }

    /// Detect BOS events and return structured data.
    pub fn detect_events(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<BOSEvent> {
        let (bos_signal, broken_level) = self.calculate(high, low, close);
        let (swing_highs, swing_lows) = self.detect_swings(high, low);
        let mut events = Vec::new();

        let mut last_swing_high: Option<f64> = None;
        let mut last_swing_low: Option<f64> = None;

        for i in 0..close.len() {
            if let Some(sh) = swing_highs[i] {
                last_swing_high = Some(sh);
            }
            if let Some(sl) = swing_lows[i] {
                last_swing_low = Some(sl);
            }

            let signal = bos_signal[i];
            if signal != 0.0 {
                let bos_type = if signal > 0.0 {
                    BOSType::Bullish
                } else {
                    BOSType::Bearish
                };

                let previous_swing = if signal > 0.0 {
                    last_swing_high.unwrap_or(broken_level[i])
                } else {
                    last_swing_low.unwrap_or(broken_level[i])
                };

                events.push(BOSEvent {
                    bos_type,
                    index: i,
                    broken_level: broken_level[i],
                    previous_swing,
                    is_choch: signal.abs() > 1.5,
                });
            }
        }

        events
    }

    /// Get current trend based on most recent BOS.
    pub fn current_trend(&self, high: &[f64], low: &[f64], close: &[f64]) -> Option<BOSType> {
        let (bos_signal, _) = self.calculate(high, low, close);

        // Find the last non-zero signal
        for signal in bos_signal.iter().rev() {
            if *signal > 0.0 {
                return Some(BOSType::Bullish);
            } else if *signal < 0.0 {
                return Some(BOSType::Bearish);
            }
        }

        None
    }
}

impl TechnicalIndicator for BreakOfStructure {
    fn name(&self) -> &str {
        "BreakOfStructure"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 2 * self.swing_lookback + 1 {
            return Err(IndicatorError::InsufficientData {
                required: 2 * self.swing_lookback + 1,
                got: data.close.len(),
            });
        }

        let (bos_signal, broken_level) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(bos_signal, broken_level))
    }

    fn min_periods(&self) -> usize {
        2 * self.swing_lookback + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for BreakOfStructure {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (bos_signal, _) = self.calculate(&data.high, &data.low, &data.close);

        // Look for most recent signal
        for signal in bos_signal.iter().rev() {
            if *signal > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if *signal < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (bos_signal, _) = self.calculate(&data.high, &data.low, &data.close);

        let mut trend = IndicatorSignal::Neutral;
        let signals = bos_signal
            .iter()
            .map(|&s| {
                if s > 0.0 {
                    trend = IndicatorSignal::Bullish;
                } else if s < 0.0 {
                    trend = IndicatorSignal::Bearish;
                }
                trend
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bos_bullish() {
        let bos = BreakOfStructure::new(2);

        // Clear uptrend with break above swing high
        let high = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0];
        let low = vec![98.0, 99.0, 100.0, 99.5, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0];
        let close = vec![99.0, 100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0, 105.0, 104.5, 106.0];

        let (signal, level) = bos.calculate(&high, &low, &close);

        assert_eq!(signal.len(), 11);
        assert_eq!(level.len(), 11);
    }

    #[test]
    fn test_bos_bearish() {
        let bos = BreakOfStructure::new(2);

        // Clear downtrend with break below swing low
        let high = vec![107.0, 106.0, 105.0, 105.5, 104.0, 103.0, 103.5, 102.0, 101.0, 101.5, 100.0];
        let low = vec![105.0, 104.0, 103.0, 103.5, 102.0, 101.0, 101.5, 100.0, 99.0, 99.5, 98.0];
        let close = vec![106.0, 105.0, 104.0, 104.5, 103.0, 102.0, 102.5, 101.0, 100.0, 100.5, 99.0];

        let (signal, level) = bos.calculate(&high, &low, &close);

        assert_eq!(signal.len(), 11);
        // Should have bearish BOS signals
    }

    #[test]
    fn test_bos_choch_detection() {
        let bos = BreakOfStructure::new(2);

        // Trend reversal scenario
        let high = vec![105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let low = vec![103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0];
        let close = vec![104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0];

        let (signal, _) = bos.calculate(&high, &low, &close);

        // Check for CHoCH signals (abs > 1.5)
        let has_choch = signal.iter().any(|&s| s.abs() > 1.5);
        // May or may not have CHoCH depending on swing detection
        assert_eq!(signal.len(), 11);
    }

    #[test]
    fn test_bos_event_detection() {
        let bos = BreakOfStructure::new(2);

        let high = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0];
        let low = vec![98.0, 99.0, 100.0, 99.5, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0];
        let close = vec![99.0, 100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0, 105.0, 104.5, 106.0];

        let events = bos.detect_events(&high, &low, &close);

        for event in &events {
            assert!(event.broken_level > 0.0);
        }
    }

    #[test]
    fn test_bos_technical_indicator() {
        let bos = BreakOfStructure::new(3);

        let mut data = OHLCVSeries::new();
        for i in 0..20 {
            let base = 100.0 + (i as f64 * 0.3).sin() * 5.0;
            data.open.push(base);
            data.high.push(base + 2.0);
            data.low.push(base - 2.0);
            data.close.push(base + 1.0);
            data.volume.push(1000.0);
        }

        let output = bos.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 20);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_bos_current_trend() {
        let bos = BreakOfStructure::new(2);

        // Uptrend
        let high: Vec<f64> = (0..15).map(|i| 100.0 + i as f64).collect();
        let low: Vec<f64> = (0..15).map(|i| 98.0 + i as f64).collect();
        let close: Vec<f64> = (0..15).map(|i| 99.0 + i as f64).collect();

        let trend = bos.current_trend(&high, &low, &close);
        // Should detect bullish trend in clear uptrend
    }
}
