//! London Session Range Indicator (IND-377)
//!
//! Tracks London session high and low price levels.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator, IndicatorSignal,
};

/// Configuration for London Session Range.
#[derive(Debug, Clone)]
pub struct LondonSessionConfig {
    /// Session start hour (UTC), default 8 (8:00 AM).
    pub start_hour: u8,
    /// Session end hour (UTC), default 16 (4:00 PM).
    pub end_hour: u8,
    /// Whether to extend range into next session.
    pub extend_range: bool,
}

impl Default for LondonSessionConfig {
    fn default() -> Self {
        Self {
            start_hour: 8,
            end_hour: 16,
            extend_range: true,
        }
    }
}

/// London Session Range (IND-377).
///
/// Identifies the high and low of the London trading session (typically 8:00-16:00 UTC).
/// Useful for:
/// - Identifying key support/resistance levels
/// - Breakout trading strategies
/// - Understanding institutional activity periods
///
/// # Output
/// - Primary: Session high values
/// - Secondary: Session low values
/// - Tertiary: Session mid-point values
#[derive(Debug, Clone)]
pub struct LondonSessionRange {
    config: LondonSessionConfig,
}

impl LondonSessionRange {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: LondonSessionConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: LondonSessionConfig) -> Self {
        Self { config }
    }

    /// Create with custom session hours.
    pub fn with_hours(start_hour: u8, end_hour: u8) -> Self {
        Self {
            config: LondonSessionConfig {
                start_hour,
                end_hour,
                ..Default::default()
            },
        }
    }

    /// Calculate session range from OHLCV data.
    ///
    /// Note: In production, this would use timestamp data to identify session bars.
    /// For this implementation, we simulate by treating each `session_bars` bars as a session.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n == 0 {
            return (vec![], vec![], vec![]);
        }

        // Approximate session as 8 hours = 8 bars for hourly data
        let session_bars = (self.config.end_hour - self.config.start_hour) as usize;
        let session_bars = session_bars.max(1);

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
                // Record session range
                session_high[i] = current_high;
                session_low[i] = current_low;
                session_mid[i] = (current_high + current_low) / 2.0;

                // If extending range, propagate forward
                if self.config.extend_range {
                    // Previous values will be filled in next iteration
                }

                // Reset for next session
                current_high = f64::NEG_INFINITY;
                current_low = f64::INFINITY;
                bars_in_session = 0;
            } else if i > 0 && !session_high[i - 1].is_nan() && self.config.extend_range {
                // Extend previous session range until new session completes
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
}

impl Default for LondonSessionRange {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for LondonSessionRange {
    fn name(&self) -> &str {
        "LondonSessionRange"
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
        (self.config.end_hour - self.config.start_hour) as usize
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for LondonSessionRange {
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
    fn test_london_session_range_basic() {
        let indicator = LondonSessionRange::new();
        let (high, low, _) = create_test_data();

        let (session_high, session_low, session_mid) = indicator.calculate(&high, &low);

        assert_eq!(session_high.len(), 24);
        assert_eq!(session_low.len(), 24);
        assert_eq!(session_mid.len(), 24);

        // After 8 bars (default session length), should have valid values
        for i in 8..24 {
            assert!(!session_high[i].is_nan(), "session_high[{}] should not be NaN", i);
            assert!(!session_low[i].is_nan(), "session_low[{}] should not be NaN", i);
            assert!(session_high[i] >= session_low[i]);
        }
    }

    #[test]
    fn test_london_session_mid_calculation() {
        let indicator = LondonSessionRange::new();
        let (high, low, _) = create_test_data();

        let (session_high, session_low, session_mid) = indicator.calculate(&high, &low);

        for i in 8..24 {
            if !session_high[i].is_nan() && !session_low[i].is_nan() {
                let expected_mid = (session_high[i] + session_low[i]) / 2.0;
                assert!(
                    (session_mid[i] - expected_mid).abs() < 1e-10,
                    "Mid-point should be average of high and low"
                );
            }
        }
    }

    #[test]
    fn test_london_session_breakout() {
        let indicator = LondonSessionRange::new();
        let (high, low, close) = create_test_data();

        let breakout_up = indicator.breakout_up(&close, &high, &low);
        let breakout_down = indicator.breakout_down(&close, &high, &low);

        assert_eq!(breakout_up.len(), 24);
        assert_eq!(breakout_down.len(), 24);
    }

    #[test]
    fn test_london_session_custom_hours() {
        let indicator = LondonSessionRange::with_hours(7, 15);
        let (high, low, _) = create_test_data();

        let (session_high, session_low, _) = indicator.calculate(&high, &low);

        // Custom 8-hour session
        assert_eq!(session_high.len(), 24);
        assert_eq!(session_low.len(), 24);
    }

    #[test]
    fn test_technical_indicator_impl() {
        let indicator = LondonSessionRange::new();
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
        let indicator = LondonSessionRange::new();
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
        let indicator = LondonSessionRange::new();
        let (empty_high, empty_low, empty_mid) = indicator.calculate(&[], &[]);

        assert!(empty_high.is_empty());
        assert!(empty_low.is_empty());
        assert!(empty_mid.is_empty());
    }
}
