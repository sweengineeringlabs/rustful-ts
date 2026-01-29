//! New York Session Range Indicator (IND-378)
//!
//! Tracks New York session high and low price levels.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator, IndicatorSignal,
};

/// Configuration for New York Session Range.
#[derive(Debug, Clone)]
pub struct NYSessionConfig {
    /// Session start hour (UTC), default 13 (1:00 PM UTC = 8:00 AM EST).
    pub start_hour: u8,
    /// Session end hour (UTC), default 21 (9:00 PM UTC = 4:00 PM EST).
    pub end_hour: u8,
    /// Whether to extend range into next session.
    pub extend_range: bool,
    /// Include pre-market (8:00 AM - 9:30 AM EST).
    pub include_premarket: bool,
    /// Include after-hours (4:00 PM - 8:00 PM EST).
    pub include_afterhours: bool,
}

impl Default for NYSessionConfig {
    fn default() -> Self {
        Self {
            start_hour: 13,  // 8:00 AM EST in UTC
            end_hour: 21,    // 4:00 PM EST in UTC
            extend_range: true,
            include_premarket: false,
            include_afterhours: false,
        }
    }
}

/// New York Session Range (IND-378).
///
/// Identifies the high and low of the New York trading session (typically 13:00-21:00 UTC).
/// The NY session is considered the most liquid session for USD pairs and US equities.
///
/// Key features:
/// - High overlap with London session (13:00-16:00 UTC)
/// - Major economic data releases
/// - Options expiration effects
///
/// # Output
/// - Primary: Session high values
/// - Secondary: Session low values
/// - Tertiary: Session mid-point values
#[derive(Debug, Clone)]
pub struct NYSessionRange {
    config: NYSessionConfig,
}

impl NYSessionRange {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: NYSessionConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: NYSessionConfig) -> Self {
        Self { config }
    }

    /// Create with custom session hours.
    pub fn with_hours(start_hour: u8, end_hour: u8) -> Self {
        Self {
            config: NYSessionConfig {
                start_hour,
                end_hour,
                ..Default::default()
            },
        }
    }

    /// Get effective session duration based on config.
    fn effective_duration(&self) -> usize {
        let mut duration = (self.config.end_hour - self.config.start_hour) as usize;
        if self.config.include_premarket {
            duration += 2; // Pre-market hours
        }
        if self.config.include_afterhours {
            duration += 4; // After-hours
        }
        duration.max(1)
    }

    /// Calculate session range from OHLCV data.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n == 0 {
            return (vec![], vec![], vec![]);
        }

        let session_bars = self.effective_duration();

        let mut session_high = vec![f64::NAN; n];
        let mut session_low = vec![f64::NAN; n];
        let mut session_mid = vec![f64::NAN; n];

        let mut current_high = f64::NEG_INFINITY;
        let mut current_low = f64::INFINITY;
        let mut bars_in_session = 0;

        for i in 0..n {
            // Update running high/low
            if high[i] > current_high {
                current_high = high[i];
            }
            if low[i] < current_low {
                current_low = low[i];
            }

            bars_in_session += 1;

            // After session_bars, we have a complete session
            if bars_in_session >= session_bars {
                session_high[i] = current_high;
                session_low[i] = current_low;
                session_mid[i] = (current_high + current_low) / 2.0;

                // Reset for next session
                current_high = f64::NEG_INFINITY;
                current_low = f64::INFINITY;
                bars_in_session = 0;
            } else if i > 0 && !session_high[i - 1].is_nan() && self.config.extend_range {
                // Extend previous session range
                session_high[i] = session_high[i - 1];
                session_low[i] = session_low[i - 1];
                session_mid[i] = session_mid[i - 1];
            }
        }

        // Fill remaining bars with last known session range if extending
        if self.config.extend_range {
            let mut last_high = f64::NAN;
            let mut last_low = f64::NAN;
            let mut last_mid = f64::NAN;

            for i in 0..n {
                if !session_high[i].is_nan() {
                    last_high = session_high[i];
                    last_low = session_low[i];
                    last_mid = session_mid[i];
                } else if !last_high.is_nan() {
                    session_high[i] = last_high;
                    session_low[i] = last_low;
                    session_mid[i] = last_mid;
                }
            }
        }

        (session_high, session_low, session_mid)
    }

    /// Calculate session range (width).
    pub fn session_range_width(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let (session_high, session_low, _) = self.calculate(high, low);
        session_high
            .iter()
            .zip(session_low.iter())
            .map(|(&h, &l)| {
                if h.is_nan() || l.is_nan() {
                    f64::NAN
                } else {
                    h - l
                }
            })
            .collect()
    }

    /// Check if price broke above session high.
    pub fn breakout_up(&self, close: &[f64], high: &[f64], low: &[f64]) -> Vec<bool> {
        let (session_high, _, _) = self.calculate(high, low);
        close
            .iter()
            .zip(session_high.iter())
            .map(|(&c, &sh)| !sh.is_nan() && c > sh)
            .collect()
    }

    /// Check if price broke below session low.
    pub fn breakout_down(&self, close: &[f64], high: &[f64], low: &[f64]) -> Vec<bool> {
        let (_, session_low, _) = self.calculate(high, low);
        close
            .iter()
            .zip(session_low.iter())
            .map(|(&c, &sl)| !sl.is_nan() && c < sl)
            .collect()
    }

    /// Calculate position within session range (0.0 = low, 1.0 = high).
    pub fn range_position(&self, close: &[f64], high: &[f64], low: &[f64]) -> Vec<f64> {
        let (session_high, session_low, _) = self.calculate(high, low);
        close
            .iter()
            .zip(session_high.iter().zip(session_low.iter()))
            .map(|(&c, (&sh, &sl))| {
                if sh.is_nan() || sl.is_nan() || (sh - sl).abs() < 1e-10 {
                    f64::NAN
                } else {
                    (c - sl) / (sh - sl)
                }
            })
            .collect()
    }
}

impl Default for NYSessionRange {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for NYSessionRange {
    fn name(&self) -> &str {
        "NYSessionRange"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let (session_high, session_low, session_mid) =
            self.calculate(&data.high, &data.low);

        Ok(IndicatorOutput::triple(session_high, session_low, session_mid))
    }

    fn min_periods(&self) -> usize {
        self.effective_duration()
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for NYSessionRange {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let n = data.close.len();
        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let breakout_up = self.breakout_up(&data.close, &data.high, &data.low);
        let breakout_down = self.breakout_down(&data.close, &data.high, &data.low);

        let last_up = breakout_up.last().copied().unwrap_or(false);
        let last_down = breakout_down.last().copied().unwrap_or(false);

        if last_up {
            Ok(IndicatorSignal::Bullish)
        } else if last_down {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let breakout_up = self.breakout_up(&data.close, &data.high, &data.low);
        let breakout_down = self.breakout_down(&data.close, &data.high, &data.low);

        let signals = breakout_up
            .iter()
            .zip(breakout_down.iter())
            .map(|(&up, &down)| {
                if up {
                    IndicatorSignal::Bullish
                } else if down {
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

    fn create_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // 24 bars simulating hourly data
        let high: Vec<f64> = (0..24)
            .map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.3).sin() * 2.0)
            .collect();
        let low: Vec<f64> = high.iter().map(|h| h - 1.0).collect();
        let close: Vec<f64> = high.iter().zip(low.iter()).map(|(h, l)| (h + l) / 2.0).collect();
        (high, low, close)
    }

    #[test]
    fn test_ny_session_range_basic() {
        let indicator = NYSessionRange::new();
        let (high, low, _) = create_test_data();

        let (session_high, session_low, session_mid) = indicator.calculate(&high, &low);

        assert_eq!(session_high.len(), 24);
        assert_eq!(session_low.len(), 24);
        assert_eq!(session_mid.len(), 24);

        // After 8 bars (default session length), should have valid values
        for i in 8..24 {
            assert!(!session_high[i].is_nan(), "session_high[{}] should not be NaN", i);
            assert!(session_high[i] >= session_low[i]);
        }
    }

    #[test]
    fn test_ny_session_range_width() {
        let indicator = NYSessionRange::new();
        let (high, low, _) = create_test_data();

        let range_width = indicator.session_range_width(&high, &low);

        assert_eq!(range_width.len(), 24);
        for i in 8..24 {
            if !range_width[i].is_nan() {
                assert!(range_width[i] >= 0.0);
            }
        }
    }

    #[test]
    fn test_ny_session_range_position() {
        let indicator = NYSessionRange::new();
        let (high, low, close) = create_test_data();

        let position = indicator.range_position(&close, &high, &low);

        assert_eq!(position.len(), 24);
        for pos in position.iter().skip(8) {
            if !pos.is_nan() {
                // Position should be between -infinity and +infinity, but typically near 0-1
                assert!(pos.is_finite());
            }
        }
    }

    #[test]
    fn test_ny_session_breakout_detection() {
        let indicator = NYSessionRange::new();
        let (high, low, close) = create_test_data();

        let breakout_up = indicator.breakout_up(&close, &high, &low);
        let breakout_down = indicator.breakout_down(&close, &high, &low);

        assert_eq!(breakout_up.len(), 24);
        assert_eq!(breakout_down.len(), 24);

        // Breakout up and down should be mutually exclusive (mostly)
        for i in 0..24 {
            assert!(!(breakout_up[i] && breakout_down[i]));
        }
    }

    #[test]
    fn test_ny_session_with_extended_hours() {
        let config = NYSessionConfig {
            start_hour: 13,
            end_hour: 21,
            extend_range: true,
            include_premarket: true,
            include_afterhours: true,
        };
        let indicator = NYSessionRange::with_config(config);

        let (high, low, _) = create_test_data();
        let (session_high, _, _) = indicator.calculate(&high, &low);

        assert_eq!(session_high.len(), 24);
    }

    #[test]
    fn test_technical_indicator_impl() {
        let indicator = NYSessionRange::new();
        let (high, low, close) = create_test_data();

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 24],
        };

        let output = indicator.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 24);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let indicator = NYSessionRange::new();
        let (high, low, close) = create_test_data();

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 24],
        };

        let signal = indicator.signal(&data).unwrap();
        assert!(matches!(
            signal,
            IndicatorSignal::Bullish | IndicatorSignal::Bearish | IndicatorSignal::Neutral
        ));

        let signals = indicator.signals(&data).unwrap();
        assert_eq!(signals.len(), 24);
    }

    #[test]
    fn test_empty_data() {
        let indicator = NYSessionRange::new();
        let (empty_high, empty_low, empty_mid) = indicator.calculate(&[], &[]);

        assert!(empty_high.is_empty());
        assert!(empty_low.is_empty());
        assert!(empty_mid.is_empty());
    }

    #[test]
    fn test_min_periods() {
        let indicator = NYSessionRange::new();
        assert_eq!(indicator.min_periods(), 8); // Default 8-hour session

        let config = NYSessionConfig {
            include_premarket: true,
            include_afterhours: true,
            ..Default::default()
        };
        let extended = NYSessionRange::with_config(config);
        assert_eq!(extended.min_periods(), 14); // 8 + 2 + 4
    }
}
