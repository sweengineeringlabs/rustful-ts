//! Spring/Upthrust Pattern Detection - False breakout identification (IND-234)
//!
//! Detects Wyckoff spring and upthrust patterns, which are false breakouts below
//! support (spring) or above resistance (upthrust) that often signal reversals.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Type of false breakout pattern detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalseBreakoutType {
    /// No pattern detected
    None,
    /// Spring - false breakout below support (bullish)
    Spring,
    /// Upthrust - false breakout above resistance (bearish)
    Upthrust,
}

/// Spring/Upthrust event with details.
#[derive(Debug, Clone)]
pub struct SpringUpthrustEvent {
    /// Type of pattern
    pub pattern_type: FalseBreakoutType,
    /// The level that was briefly broken
    pub broken_level: f64,
    /// Penetration depth (how far past the level)
    pub penetration: f64,
    /// Volume relative to average (ratio)
    pub volume_ratio: f64,
    /// Whether the pattern is confirmed by close
    pub confirmed: bool,
}

/// Spring/Upthrust configuration.
#[derive(Debug, Clone)]
pub struct SpringUpthrustConfig {
    /// Period for identifying support/resistance levels
    pub lookback_period: usize,
    /// Minimum penetration depth (as percentage of range)
    pub min_penetration: f64,
    /// Maximum penetration depth (as percentage of range)
    pub max_penetration: f64,
    /// Volume threshold (ratio to average volume)
    pub volume_threshold: f64,
    /// Require close back above/below the level
    pub require_close_confirmation: bool,
}

impl Default for SpringUpthrustConfig {
    fn default() -> Self {
        Self {
            lookback_period: 20,
            min_penetration: 0.001, // 0.1%
            max_penetration: 0.03,  // 3%
            volume_threshold: 1.2,  // 20% above average
            require_close_confirmation: true,
        }
    }
}

/// Spring/Upthrust output containing detection results.
#[derive(Debug, Clone)]
pub struct SpringUpthrustOutput {
    /// Pattern type at each bar
    pub pattern_type: Vec<FalseBreakoutType>,
    /// Signal strength (-1 to 1, negative for upthrust, positive for spring)
    pub signal_strength: Vec<f64>,
    /// Support levels used for detection
    pub support_levels: Vec<f64>,
    /// Resistance levels used for detection
    pub resistance_levels: Vec<f64>,
    /// Events with detailed pattern information
    pub events: Vec<Option<SpringUpthrustEvent>>,
}

/// Spring/Upthrust Pattern Detector (IND-234).
///
/// Identifies Wyckoff spring and upthrust patterns:
///
/// **Spring:**
/// - Price briefly breaks below support level
/// - Close recovers back above support
/// - Often accompanied by declining volume or volume spike
/// - Bullish reversal signal (bears trapped below support)
///
/// **Upthrust:**
/// - Price briefly breaks above resistance level
/// - Close falls back below resistance
/// - Often accompanied by high volume (distribution)
/// - Bearish reversal signal (bulls trapped above resistance)
///
/// These patterns are key Wyckoff signals indicating failed attempts by
/// one side to continue the trend, often marking significant reversals.
#[derive(Debug, Clone)]
pub struct SpringUpthrust {
    config: SpringUpthrustConfig,
}

impl SpringUpthrust {
    pub fn new(lookback_period: usize) -> Self {
        Self {
            config: SpringUpthrustConfig {
                lookback_period,
                ..Default::default()
            },
        }
    }

    pub fn from_config(config: SpringUpthrustConfig) -> Self {
        Self { config }
    }

    /// Calculate rolling minimum (support level).
    fn rolling_min(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in (period - 1)..n {
            let start = i + 1 - period;
            result[i] = data[start..=i]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
        }

        result
    }

    /// Calculate rolling maximum (resistance level).
    fn rolling_max(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        for i in (period - 1)..n {
            let start = i + 1 - period;
            result[i] = data[start..=i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        }

        result
    }

    /// Calculate average volume.
    fn average_volume(&self, volume: &[f64], period: usize) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![f64::NAN; n];

        for i in (period - 1)..n {
            let start = i + 1 - period;
            let sum: f64 = volume[start..=i].iter().sum();
            result[i] = sum / period as f64;
        }

        result
    }

    /// Calculate Spring/Upthrust patterns.
    pub fn calculate(&self, data: &OHLCVSeries) -> SpringUpthrustOutput {
        let n = data.close.len();
        let period = self.config.lookback_period;

        if n < period + 1 {
            return SpringUpthrustOutput {
                pattern_type: vec![FalseBreakoutType::None; n],
                signal_strength: vec![0.0; n],
                support_levels: vec![f64::NAN; n],
                resistance_levels: vec![f64::NAN; n],
                events: vec![None; n],
            };
        }

        let mut pattern_type = vec![FalseBreakoutType::None; n];
        let mut signal_strength = vec![0.0; n];
        let mut events: Vec<Option<SpringUpthrustEvent>> = vec![None; n];

        // Calculate support (rolling minimum of lows, excluding current bar)
        // and resistance (rolling maximum of highs, excluding current bar)
        let mut support_levels = vec![f64::NAN; n];
        let mut resistance_levels = vec![f64::NAN; n];

        for i in period..n {
            // Use prior period lows/highs to identify levels
            let start = i - period;
            let end = i - 1;

            support_levels[i] = data.low[start..=end]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));

            resistance_levels[i] = data.high[start..=end]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        }

        // Calculate average volume
        let avg_volume = self.average_volume(&data.volume, period);

        // Detect patterns
        for i in period..n {
            let support = support_levels[i];
            let resistance = resistance_levels[i];
            let range = resistance - support;

            if range <= 0.0 || support.is_nan() || resistance.is_nan() {
                continue;
            }

            let high = data.high[i];
            let low = data.low[i];
            let close = data.close[i];
            let volume = data.volume[i];
            let avg_vol = avg_volume[i];

            let vol_ratio = if avg_vol > 0.0 { volume / avg_vol } else { 1.0 };

            // Check for Spring (false break below support)
            if low < support {
                let penetration = (support - low) / support;

                if penetration >= self.config.min_penetration
                    && penetration <= self.config.max_penetration
                {
                    let close_confirmed = close > support;

                    if !self.config.require_close_confirmation || close_confirmed {
                        pattern_type[i] = FalseBreakoutType::Spring;

                        // Signal strength based on penetration depth, volume, and confirmation
                        let penetration_score = 1.0 - (penetration / self.config.max_penetration);
                        let volume_score = if vol_ratio < 1.0 {
                            1.0 // Declining volume on spring is bullish
                        } else {
                            0.5 + 0.5 / vol_ratio
                        };
                        let confirmation_score = if close_confirmed { 1.0 } else { 0.5 };

                        signal_strength[i] = penetration_score * volume_score * confirmation_score;

                        events[i] = Some(SpringUpthrustEvent {
                            pattern_type: FalseBreakoutType::Spring,
                            broken_level: support,
                            penetration,
                            volume_ratio: vol_ratio,
                            confirmed: close_confirmed,
                        });
                    }
                }
            }

            // Check for Upthrust (false break above resistance)
            if high > resistance {
                let penetration = (high - resistance) / resistance;

                if penetration >= self.config.min_penetration
                    && penetration <= self.config.max_penetration
                {
                    let close_confirmed = close < resistance;

                    if !self.config.require_close_confirmation || close_confirmed {
                        // Only mark as upthrust if not already marked as spring
                        if pattern_type[i] == FalseBreakoutType::None {
                            pattern_type[i] = FalseBreakoutType::Upthrust;

                            // Signal strength (negative for bearish)
                            let penetration_score = 1.0 - (penetration / self.config.max_penetration);
                            let volume_score = if vol_ratio >= self.config.volume_threshold {
                                1.0 // High volume on upthrust is bearish
                            } else {
                                0.5 + 0.5 * vol_ratio / self.config.volume_threshold
                            };
                            let confirmation_score = if close_confirmed { 1.0 } else { 0.5 };

                            signal_strength[i] =
                                -penetration_score * volume_score * confirmation_score;

                            events[i] = Some(SpringUpthrustEvent {
                                pattern_type: FalseBreakoutType::Upthrust,
                                broken_level: resistance,
                                penetration,
                                volume_ratio: vol_ratio,
                                confirmed: close_confirmed,
                            });
                        }
                    }
                }
            }
        }

        SpringUpthrustOutput {
            pattern_type,
            signal_strength,
            support_levels,
            resistance_levels,
            events,
        }
    }
}

impl Default for SpringUpthrust {
    fn default() -> Self {
        Self::from_config(SpringUpthrustConfig::default())
    }
}

impl TechnicalIndicator for SpringUpthrust {
    fn name(&self) -> &str {
        "SpringUpthrust"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.lookback_period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.config.lookback_period + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);

        // Encode pattern type: 0 = None, 1 = Spring, -1 = Upthrust
        let pattern_encoded: Vec<f64> = result
            .pattern_type
            .iter()
            .map(|&p| match p {
                FalseBreakoutType::None => 0.0,
                FalseBreakoutType::Spring => 1.0,
                FalseBreakoutType::Upthrust => -1.0,
            })
            .collect();

        Ok(IndicatorOutput::dual(pattern_encoded, result.signal_strength))
    }

    fn min_periods(&self) -> usize {
        self.config.lookback_period + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for SpringUpthrust {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);

        if let Some(&pattern) = result.pattern_type.last() {
            match pattern {
                FalseBreakoutType::Spring => return Ok(IndicatorSignal::Bullish),
                FalseBreakoutType::Upthrust => return Ok(IndicatorSignal::Bearish),
                FalseBreakoutType::None => {}
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);

        Ok(result
            .pattern_type
            .iter()
            .map(|&p| match p {
                FalseBreakoutType::Spring => IndicatorSignal::Bullish,
                FalseBreakoutType::Upthrust => IndicatorSignal::Bearish,
                FalseBreakoutType::None => IndicatorSignal::Neutral,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_spring_data() -> OHLCVSeries {
        // Create data with a spring pattern: price briefly breaks below support then recovers
        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();

        // Build up a range first (20 bars)
        for i in 0..20 {
            let base = 100.0 + (i % 3) as f64 * 0.5;
            open.push(base);
            high.push(base + 1.0);
            low.push(base - 0.5);
            close.push(base + 0.3);
            volume.push(1000.0);
        }

        // Support should be around 99.5, resistance around 102.0

        // Spring bar: break below support, close back above
        open.push(100.0);
        high.push(100.5);
        low.push(99.0); // Breaks below 99.5 support
        close.push(100.2); // Closes back above support
        volume.push(800.0); // Lower volume on spring

        OHLCVSeries { open, high, low, close, volume }
    }

    fn create_upthrust_data() -> OHLCVSeries {
        // Create data with an upthrust pattern
        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();

        // Build up a range first (20 bars)
        for i in 0..20 {
            let base = 100.0 + (i % 3) as f64 * 0.5;
            open.push(base);
            high.push(base + 1.0);
            low.push(base - 0.5);
            close.push(base + 0.3);
            volume.push(1000.0);
        }

        // Upthrust bar: break above resistance, close back below
        open.push(101.0);
        high.push(103.0); // Breaks above 102.0 resistance
        low.push(100.5);
        close.push(101.0); // Closes back below resistance
        volume.push(1500.0); // Higher volume on upthrust

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_spring_detection() {
        let detector = SpringUpthrust::new(20);
        let data = create_spring_data();
        let result = detector.calculate(&data);

        // Last bar should be detected as a spring
        let last_idx = result.pattern_type.len() - 1;
        assert_eq!(result.pattern_type[last_idx], FalseBreakoutType::Spring);
        assert!(result.signal_strength[last_idx] > 0.0);
    }

    #[test]
    fn test_upthrust_detection() {
        let detector = SpringUpthrust::new(20);
        let data = create_upthrust_data();
        let result = detector.calculate(&data);

        // Last bar should be detected as an upthrust
        let last_idx = result.pattern_type.len() - 1;
        assert_eq!(result.pattern_type[last_idx], FalseBreakoutType::Upthrust);
        assert!(result.signal_strength[last_idx] < 0.0);
    }

    #[test]
    fn test_support_resistance_levels() {
        let detector = SpringUpthrust::new(10);
        let mut data = OHLCVSeries::new();

        for i in 0..15 {
            let base = 100.0;
            data.open.push(base);
            data.high.push(base + 2.0);
            data.low.push(base - 1.0);
            data.close.push(base + 0.5);
            data.volume.push(1000.0);
        }

        let result = detector.calculate(&data);

        // Support should be around 99.0 (low), resistance around 102.0 (high)
        for i in 10..result.support_levels.len() {
            if !result.support_levels[i].is_nan() {
                assert!((result.support_levels[i] - 99.0).abs() < 0.1);
                assert!((result.resistance_levels[i] - 102.0).abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_signal_generation() {
        let detector = SpringUpthrust::new(20);
        let data = create_spring_data();

        let signal = detector.signal(&data).unwrap();
        assert_eq!(signal, IndicatorSignal::Bullish);
    }

    #[test]
    fn test_events_contain_details() {
        let detector = SpringUpthrust::new(20);
        let data = create_spring_data();
        let result = detector.calculate(&data);

        let last_idx = result.events.len() - 1;
        assert!(result.events[last_idx].is_some());

        let event = result.events[last_idx].as_ref().unwrap();
        assert_eq!(event.pattern_type, FalseBreakoutType::Spring);
        assert!(event.penetration > 0.0);
        assert!(event.confirmed);
    }

    #[test]
    fn test_insufficient_data() {
        let detector = SpringUpthrust::new(20);
        let mut data = OHLCVSeries::new();

        for _ in 0..10 {
            data.open.push(100.0);
            data.high.push(101.0);
            data.low.push(99.0);
            data.close.push(100.5);
            data.volume.push(1000.0);
        }

        let result = detector.compute(&data);
        assert!(result.is_err());
    }
}
