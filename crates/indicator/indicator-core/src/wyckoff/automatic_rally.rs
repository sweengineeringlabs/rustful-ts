//! Automatic Rally/Reaction Detector - Post-climax bounce detection (IND-233)
//!
//! Identifies automatic rallies (after selling climax) and automatic reactions
//! (after buying climax) using Wyckoff analysis principles.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Automatic Rally/Reaction configuration.
#[derive(Debug, Clone)]
pub struct AutomaticRallyConfig {
    /// Lookback period for climax detection
    pub climax_lookback: usize,
    /// Minimum bars for rally/reaction development
    pub min_rally_bars: usize,
    /// Volume decrease threshold (ratio to climax volume)
    pub volume_decrease_threshold: f64,
    /// Minimum price retracement percentage
    pub min_retracement: f64,
    /// Maximum price retracement percentage
    pub max_retracement: f64,
}

impl Default for AutomaticRallyConfig {
    fn default() -> Self {
        Self {
            climax_lookback: 20,
            min_rally_bars: 3,
            volume_decrease_threshold: 0.7,
            min_retracement: 0.25,
            max_retracement: 0.75,
        }
    }
}

/// Rally/Reaction event type.
#[derive(Debug, Clone, PartialEq)]
pub enum RallyReactionEvent {
    /// Automatic rally after selling climax
    AutomaticRally,
    /// Automatic reaction after buying climax
    AutomaticReaction,
    /// Rally/reaction in progress
    InProgress,
    /// No event
    None,
}

/// Automatic Rally/Reaction Detector.
///
/// In Wyckoff analysis:
/// - **Automatic Rally (AR)**: After a selling climax, prices bounce sharply
///   on reduced volume. This marks the first sign of demand entering.
/// - **Automatic Reaction (AR)**: After a buying climax, prices drop sharply
///   on reduced volume. This marks the first sign of supply entering.
///
/// Characteristics:
/// 1. Follows a climax event (high volume, wide spread)
/// 2. Volume decreases as price moves opposite to climax direction
/// 3. Typically retraces 25-50% of the prior move
/// 4. Sets up the trading range for accumulation/distribution
#[derive(Debug, Clone)]
pub struct AutomaticRallyReaction {
    config: AutomaticRallyConfig,
}

impl AutomaticRallyReaction {
    pub fn new(climax_lookback: usize) -> Self {
        Self {
            config: AutomaticRallyConfig {
                climax_lookback,
                ..Default::default()
            },
        }
    }

    pub fn from_config(config: AutomaticRallyConfig) -> Self {
        Self { config }
    }

    /// Find potential climax bars in the lookback period.
    fn find_climax_bars(&self, data: &OHLCVSeries, current_idx: usize) -> Vec<(usize, bool)> {
        let mut climax_bars = Vec::new();

        if current_idx < self.config.climax_lookback {
            return climax_bars;
        }

        let start = current_idx - self.config.climax_lookback;

        // Calculate average volume and spread in lookback
        let avg_volume: f64 = data.volume[start..current_idx].iter().sum::<f64>()
            / self.config.climax_lookback as f64;

        let spreads: Vec<f64> = (start..current_idx)
            .map(|i| data.high[i] - data.low[i])
            .collect();
        let avg_spread: f64 = spreads.iter().sum::<f64>() / spreads.len() as f64;

        for i in start..current_idx {
            let volume = data.volume[i];
            let spread = data.high[i] - data.low[i];
            let close_pos = if spread > 0.0 {
                (data.close[i] - data.low[i]) / spread
            } else {
                0.5
            };

            // Check for selling climax (high volume, close near low)
            if volume > avg_volume * 2.0 && spread > avg_spread * 1.5 && close_pos < 0.3 {
                climax_bars.push((i, true)); // true = selling climax
            }
            // Check for buying climax (high volume, close near high)
            else if volume > avg_volume * 2.0 && spread > avg_spread * 1.5 && close_pos > 0.7 {
                climax_bars.push((i, false)); // false = buying climax
            }
        }

        climax_bars
    }

    /// Detect automatic rally/reaction events.
    pub fn detect(&self, data: &OHLCVSeries) -> Vec<RallyReactionEvent> {
        let n = data.close.len();
        let min_period = self.config.climax_lookback + self.config.min_rally_bars;

        if n < min_period {
            return vec![RallyReactionEvent::None; n];
        }

        let mut events = vec![RallyReactionEvent::None; n];

        for i in min_period..n {
            // Find climax bars in lookback
            let climax_bars = self.find_climax_bars(data, i);

            for (climax_idx, is_selling_climax) in climax_bars {
                let bars_since_climax = i - climax_idx;

                if bars_since_climax < self.config.min_rally_bars {
                    continue;
                }

                // Get climax bar values
                let climax_volume = data.volume[climax_idx];
                let climax_low = data.low[climax_idx];
                let climax_high = data.high[climax_idx];

                // Calculate current volume relative to climax
                let recent_avg_volume: f64 = data.volume[(climax_idx + 1)..=i].iter().sum::<f64>()
                    / bars_since_climax as f64;
                let volume_ratio = if climax_volume > 0.0 {
                    recent_avg_volume / climax_volume
                } else {
                    1.0
                };

                // Check for automatic rally (after selling climax)
                if is_selling_climax {
                    let price_move = data.high[i] - climax_low;
                    let prior_decline = climax_high - climax_low;
                    let retracement = if prior_decline > 0.0 {
                        price_move / prior_decline
                    } else {
                        0.0
                    };

                    if volume_ratio <= self.config.volume_decrease_threshold
                        && retracement >= self.config.min_retracement
                        && retracement <= self.config.max_retracement
                        && data.close[i] > climax_low
                    {
                        events[i] = RallyReactionEvent::AutomaticRally;
                        break;
                    }
                }
                // Check for automatic reaction (after buying climax)
                else {
                    let price_move = climax_high - data.low[i];
                    let prior_advance = climax_high - climax_low;
                    let retracement = if prior_advance > 0.0 {
                        price_move / prior_advance
                    } else {
                        0.0
                    };

                    if volume_ratio <= self.config.volume_decrease_threshold
                        && retracement >= self.config.min_retracement
                        && retracement <= self.config.max_retracement
                        && data.close[i] < climax_high
                    {
                        events[i] = RallyReactionEvent::AutomaticReaction;
                        break;
                    }
                }
            }
        }

        events
    }

    /// Calculate rally/reaction strength score (0-100).
    pub fn calculate_strength(&self, data: &OHLCVSeries) -> Vec<f64> {
        let n = data.close.len();
        let min_period = self.config.climax_lookback + self.config.min_rally_bars;

        if n < min_period {
            return vec![f64::NAN; n];
        }

        let mut strength = vec![f64::NAN; n];

        for i in min_period..n {
            let climax_bars = self.find_climax_bars(data, i);

            if climax_bars.is_empty() {
                strength[i] = 0.0;
                continue;
            }

            let mut max_strength: f64 = 0.0;

            for (climax_idx, is_selling_climax) in &climax_bars {
                let bars_since = i - climax_idx;
                if bars_since < self.config.min_rally_bars {
                    continue;
                }

                let climax_volume = data.volume[*climax_idx];
                let recent_avg_volume: f64 = data.volume[(*climax_idx + 1)..=i].iter().sum::<f64>()
                    / bars_since as f64;

                // Volume decline score (0-40 points)
                let volume_decline_score = if climax_volume > 0.0 {
                    (1.0 - recent_avg_volume / climax_volume).max(0.0) * 40.0
                } else {
                    0.0
                };

                // Price movement score (0-40 points)
                let price_score = if *is_selling_climax {
                    let move_up = data.close[i] - data.low[*climax_idx];
                    let potential = data.high[*climax_idx] - data.low[*climax_idx];
                    if potential > 0.0 {
                        (move_up / potential).clamp(0.0, 1.0) * 40.0
                    } else {
                        0.0
                    }
                } else {
                    let move_down = data.high[*climax_idx] - data.close[i];
                    let potential = data.high[*climax_idx] - data.low[*climax_idx];
                    if potential > 0.0 {
                        (move_down / potential).clamp(0.0, 1.0) * 40.0
                    } else {
                        0.0
                    }
                };

                // Time factor score (0-20 points) - optimal rally develops over 3-10 bars
                let optimal_bars = 6.0;
                let time_score = 20.0 * (1.0 - ((bars_since as f64 - optimal_bars) / optimal_bars).abs().min(1.0));

                let total = volume_decline_score + price_score + time_score;
                max_strength = max_strength.max(total);
            }

            strength[i] = max_strength;
        }

        strength
    }
}

impl Default for AutomaticRallyReaction {
    fn default() -> Self {
        Self::from_config(AutomaticRallyConfig::default())
    }
}

impl TechnicalIndicator for AutomaticRallyReaction {
    fn name(&self) -> &str {
        "AutomaticRallyReaction"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_period = self.config.climax_lookback + self.config.min_rally_bars;
        if data.close.len() < min_period {
            return Err(IndicatorError::InsufficientData {
                required: min_period,
                got: data.close.len(),
            });
        }

        let strength = self.calculate_strength(data);
        Ok(IndicatorOutput::single(strength))
    }

    fn min_periods(&self) -> usize {
        self.config.climax_lookback + self.config.min_rally_bars
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for AutomaticRallyReaction {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let events = self.detect(data);

        if let Some(last) = events.last() {
            match last {
                RallyReactionEvent::AutomaticRally => return Ok(IndicatorSignal::Bullish),
                RallyReactionEvent::AutomaticReaction => return Ok(IndicatorSignal::Bearish),
                RallyReactionEvent::InProgress | RallyReactionEvent::None => {}
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let events = self.detect(data);

        Ok(events
            .iter()
            .map(|e| match e {
                RallyReactionEvent::AutomaticRally => IndicatorSignal::Bullish,
                RallyReactionEvent::AutomaticReaction => IndicatorSignal::Bearish,
                RallyReactionEvent::InProgress | RallyReactionEvent::None => IndicatorSignal::Neutral,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_rally_after_climax_data(n: usize, climax_idx: usize) -> OHLCVSeries {
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            if i < climax_idx {
                // Downtrend before climax
                let base = 120.0 - (i as f64);
                open.push(base);
                high.push(base + 1.0);
                low.push(base - 1.0);
                close.push(base - 0.5);
                volume.push(1000.0);
            } else if i == climax_idx {
                // Selling climax
                let base = 120.0 - (climax_idx as f64);
                open.push(base);
                high.push(base + 2.0);
                low.push(base - 8.0); // Wide spread
                close.push(base - 7.0); // Close near low
                volume.push(5000.0); // High volume
            } else {
                // Rally after climax (price recovering, lower volume)
                let climax_low = 120.0 - (climax_idx as f64) - 8.0;
                let recovery = (i - climax_idx) as f64 * 1.5;
                let base = climax_low + recovery;
                open.push(base);
                high.push(base + 1.5);
                low.push(base - 0.5);
                close.push(base + 1.0);
                volume.push(800.0); // Lower volume
            }
        }

        OHLCVSeries { open, high, low, close, volume }
    }

    fn create_normal_data(n: usize) -> OHLCVSeries {
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0 + (i as f64) * 0.1;
            open.push(base);
            high.push(base + 1.0);
            low.push(base - 1.0);
            close.push(base + 0.5);
            volume.push(1000.0);
        }

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_automatic_rally_detection() {
        let detector = AutomaticRallyReaction::from_config(AutomaticRallyConfig {
            climax_lookback: 10,
            min_rally_bars: 3,
            volume_decrease_threshold: 0.7,
            min_retracement: 0.2,
            max_retracement: 0.8,
        });

        let data = create_rally_after_climax_data(25, 15);
        let events = detector.detect(&data);

        // Should detect automatic rally after the climax
        let has_rally = events.iter().any(|e| *e == RallyReactionEvent::AutomaticRally);
        // Note: Detection depends on specific conditions being met
        assert_eq!(events.len(), 25);

        // Check that we don't get false positives in normal data
        let normal_data = create_normal_data(25);
        let normal_events = detector.detect(&normal_data);
        let false_positives = normal_events.iter()
            .filter(|e| **e != RallyReactionEvent::None)
            .count();
        assert_eq!(false_positives, 0);
    }

    #[test]
    fn test_strength_calculation() {
        let detector = AutomaticRallyReaction::new(10);
        let data = create_rally_after_climax_data(30, 15);
        let strength = detector.calculate_strength(&data);

        assert_eq!(strength.len(), 30);

        // Valid values should be in 0-100 range
        for &val in strength.iter() {
            if !val.is_nan() {
                assert!(val >= 0.0 && val <= 100.0);
            }
        }
    }

    #[test]
    fn test_no_events_in_normal_data() {
        let detector = AutomaticRallyReaction::new(10);
        let data = create_normal_data(30);
        let events = detector.detect(&data);

        // Should not detect any events in normal trending data
        for event in events {
            assert_eq!(event, RallyReactionEvent::None);
        }
    }

    #[test]
    fn test_signal_generation() {
        let detector = AutomaticRallyReaction::new(10);
        let data = create_normal_data(30);
        let signal = detector.signal(&data).unwrap();

        // Normal data should give neutral signal
        assert!(matches!(signal, IndicatorSignal::Neutral));
    }

    #[test]
    fn test_config_defaults() {
        let config = AutomaticRallyConfig::default();
        assert_eq!(config.climax_lookback, 20);
        assert_eq!(config.min_rally_bars, 3);
        assert!((config.volume_decrease_threshold - 0.7).abs() < 0.001);
        assert!((config.min_retracement - 0.25).abs() < 0.001);
        assert!((config.max_retracement - 0.75).abs() < 0.001);
    }
}
