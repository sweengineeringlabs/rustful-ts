//! AbsorptionDetector (IND-221) - Large orders absorption detection
//!
//! Detects when large orders are being absorbed at support/resistance levels,
//! indicating potential reversal points or strong buying/selling interest.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator, SignalIndicator, IndicatorSignal,
};

/// Type of absorption detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AbsorptionType {
    /// No absorption detected
    None,
    /// Bullish absorption: high volume near lows with close near high
    BullishAbsorption,
    /// Bearish absorption: high volume near highs with close near low
    BearishAbsorption,
    /// Strong bullish: bullish absorption with follow-through
    StrongBullish,
    /// Strong bearish: bearish absorption with follow-through
    StrongBearish,
}

/// Absorption Detector Output
#[derive(Debug, Clone)]
pub struct AbsorptionDetectorOutput {
    /// Absorption type at each bar
    pub absorption_type: Vec<AbsorptionType>,
    /// Absorption strength (0-100)
    pub absorption_strength: Vec<f64>,
    /// Volume ratio to average
    pub volume_ratio: Vec<f64>,
    /// Signal values: 1 = bullish, -1 = bearish, 2 = strong bullish, -2 = strong bearish
    pub signal: Vec<f64>,
}

/// Absorption Detector Configuration
#[derive(Debug, Clone)]
pub struct AbsorptionDetectorConfig {
    /// Period for average volume calculation
    pub volume_period: usize,
    /// Volume threshold (multiplier of average)
    pub volume_threshold: f64,
    /// Minimum close position for bullish (0.0-1.0)
    pub bullish_close_threshold: f64,
    /// Maximum close position for bearish (0.0-1.0)
    pub bearish_close_threshold: f64,
    /// Lookback for confirming follow-through
    pub confirmation_bars: usize,
}

impl Default for AbsorptionDetectorConfig {
    fn default() -> Self {
        Self {
            volume_period: 20,
            volume_threshold: 1.5,
            bullish_close_threshold: 0.7,
            bearish_close_threshold: 0.3,
            confirmation_bars: 2,
        }
    }
}

/// AbsorptionDetector (IND-221)
///
/// Detects absorption patterns where large orders are being filled
/// at key price levels without moving the price significantly.
///
/// Bullish Absorption:
/// - High volume bar (above threshold)
/// - Wide range or trading near lows
/// - Close near the high of the bar
/// - Indicates buying absorption of selling pressure
///
/// Bearish Absorption:
/// - High volume bar (above threshold)
/// - Wide range or trading near highs
/// - Close near the low of the bar
/// - Indicates selling absorption of buying pressure
///
/// Strong signals require confirmation from subsequent bars.
#[derive(Debug, Clone)]
pub struct AbsorptionDetector {
    config: AbsorptionDetectorConfig,
}

impl AbsorptionDetector {
    pub fn new(config: AbsorptionDetectorConfig) -> Result<Self> {
        if config.volume_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if config.volume_threshold <= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_threshold".to_string(),
                reason: "must be greater than 1.0".to_string(),
            });
        }
        if config.bullish_close_threshold <= config.bearish_close_threshold {
            return Err(IndicatorError::InvalidParameter {
                name: "bullish_close_threshold".to_string(),
                reason: "must be greater than bearish_close_threshold".to_string(),
            });
        }
        if config.bullish_close_threshold > 1.0 || config.bearish_close_threshold < 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "close_threshold".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self {
            config: AbsorptionDetectorConfig::default(),
        }
    }

    /// Calculate absorption detection with full output
    pub fn calculate_full(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> AbsorptionDetectorOutput {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut absorption_type = vec![AbsorptionType::None; n];
        let mut absorption_strength = vec![0.0; n];
        let mut volume_ratio = vec![0.0; n];
        let mut signal = vec![0.0; n];

        if n < self.config.volume_period {
            return AbsorptionDetectorOutput {
                absorption_type,
                absorption_strength,
                volume_ratio,
                signal,
            };
        }

        // Calculate average volume
        let mut avg_volume = vec![0.0; n];
        let mut vol_sum = 0.0;

        for i in 0..self.config.volume_period {
            vol_sum += volume[i];
        }
        avg_volume[self.config.volume_period - 1] = vol_sum / self.config.volume_period as f64;

        for i in self.config.volume_period..n {
            vol_sum += volume[i] - volume[i - self.config.volume_period];
            avg_volume[i] = vol_sum / self.config.volume_period as f64;
        }

        // Track recent absorption signals for confirmation
        let mut recent_bullish: Vec<usize> = Vec::new();
        let mut recent_bearish: Vec<usize> = Vec::new();

        for i in self.config.volume_period..n {
            let range = high[i] - low[i];
            if range <= 0.0 || avg_volume[i] <= 0.0 {
                continue;
            }

            // Calculate volume ratio
            volume_ratio[i] = volume[i] / avg_volume[i];

            // Check for high volume
            if volume_ratio[i] < self.config.volume_threshold {
                continue;
            }

            // Calculate close position (0 = low, 1 = high)
            let close_position = (close[i] - low[i]) / range;

            // Bullish absorption: high volume with close near high
            if close_position >= self.config.bullish_close_threshold {
                // Check for previous price action (was there selling pressure?)
                let has_lower_wick = (close[i] - low[i]) > (high[i] - close[i]);

                if has_lower_wick || volume_ratio[i] >= self.config.volume_threshold * 1.5 {
                    // Calculate strength based on volume and close position
                    let vol_strength = ((volume_ratio[i] - 1.0) / (self.config.volume_threshold - 1.0)).min(2.0) * 50.0;
                    let close_strength = (close_position - 0.5) * 100.0;
                    absorption_strength[i] = (vol_strength + close_strength).min(100.0);

                    // Check for confirmation from previous absorption
                    let mut confirmed = false;
                    for &prev_idx in recent_bullish.iter().rev() {
                        if i - prev_idx <= self.config.confirmation_bars {
                            confirmed = true;
                            break;
                        }
                    }

                    if confirmed {
                        absorption_type[i] = AbsorptionType::StrongBullish;
                        signal[i] = 2.0;
                    } else {
                        absorption_type[i] = AbsorptionType::BullishAbsorption;
                        signal[i] = 1.0;
                    }

                    recent_bullish.push(i);
                    if recent_bullish.len() > 5 {
                        recent_bullish.remove(0);
                    }
                }
            }
            // Bearish absorption: high volume with close near low
            else if close_position <= self.config.bearish_close_threshold {
                // Check for previous price action (was there buying pressure?)
                let has_upper_wick = (high[i] - close[i]) > (close[i] - low[i]);

                if has_upper_wick || volume_ratio[i] >= self.config.volume_threshold * 1.5 {
                    // Calculate strength
                    let vol_strength = ((volume_ratio[i] - 1.0) / (self.config.volume_threshold - 1.0)).min(2.0) * 50.0;
                    let close_strength = (0.5 - close_position) * 100.0;
                    absorption_strength[i] = (vol_strength + close_strength).min(100.0);

                    // Check for confirmation
                    let mut confirmed = false;
                    for &prev_idx in recent_bearish.iter().rev() {
                        if i - prev_idx <= self.config.confirmation_bars {
                            confirmed = true;
                            break;
                        }
                    }

                    if confirmed {
                        absorption_type[i] = AbsorptionType::StrongBearish;
                        signal[i] = -2.0;
                    } else {
                        absorption_type[i] = AbsorptionType::BearishAbsorption;
                        signal[i] = -1.0;
                    }

                    recent_bearish.push(i);
                    if recent_bearish.len() > 5 {
                        recent_bearish.remove(0);
                    }
                }
            }
        }

        AbsorptionDetectorOutput {
            absorption_type,
            absorption_strength,
            volume_ratio,
            signal,
        }
    }

    /// Calculate absorption signal only
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        self.calculate_full(high, low, close, volume).signal
    }
}

impl TechnicalIndicator for AbsorptionDetector {
    fn name(&self) -> &str {
        "Absorption Detector"
    }

    fn min_periods(&self) -> usize {
        self.config.volume_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }

        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            output.signal,
            output.absorption_strength,
            output.volume_ratio,
        ))
    }
}

impl SignalIndicator for AbsorptionDetector {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);

        // Look at last few bars for recent signal
        let lookback = self.config.confirmation_bars + 1;
        let start = output.signal.len().saturating_sub(lookback);

        for i in (start..output.signal.len()).rev() {
            if output.signal[i] >= 2.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if output.signal[i] <= -2.0 {
                return Ok(IndicatorSignal::Bearish);
            } else if output.signal[i] > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if output.signal[i] < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);

        Ok(output
            .signal
            .iter()
            .map(|&s| {
                if s > 0.0 {
                    IndicatorSignal::Bullish
                } else if s < 0.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

impl Default for AbsorptionDetector {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_absorption_detector_bullish() {
        let detector = AbsorptionDetector::default_config();

        // Create data with bullish absorption pattern
        // Low volume baseline, then high volume bar with close near high
        let mut high = vec![105.0; 25];
        let mut low = vec![100.0; 25];
        let mut close = vec![102.5; 25]; // Neutral baseline
        let mut volume = vec![1000.0; 25];

        // Add bullish absorption bar at index 22
        high[22] = 106.0;
        low[22] = 100.0;
        close[22] = 105.5; // Close near high (92%)
        volume[22] = 2000.0; // 2x average

        let output = detector.calculate_full(&high, &low, &close, &volume);

        // Should detect bullish absorption
        assert!(output.signal[22] > 0.0, "Should detect bullish absorption");
        assert_eq!(output.absorption_type[22], AbsorptionType::BullishAbsorption);
    }

    #[test]
    fn test_absorption_detector_bearish() {
        let detector = AbsorptionDetector::default_config();

        // Create data with bearish absorption pattern
        let mut high = vec![105.0; 25];
        let mut low = vec![100.0; 25];
        let mut close = vec![102.5; 25];
        let mut volume = vec![1000.0; 25];

        // Add bearish absorption bar
        high[22] = 106.0;
        low[22] = 100.0;
        close[22] = 100.5; // Close near low (8%)
        volume[22] = 2000.0;

        let output = detector.calculate_full(&high, &low, &close, &volume);

        // Should detect bearish absorption
        assert!(output.signal[22] < 0.0, "Should detect bearish absorption");
        assert_eq!(output.absorption_type[22], AbsorptionType::BearishAbsorption);
    }

    #[test]
    fn test_absorption_detector_no_signal() {
        let detector = AbsorptionDetector::default_config();

        // Normal data without high volume
        let high = vec![105.0; 25];
        let low = vec![100.0; 25];
        let close = vec![104.5; 25]; // Near highs but normal volume
        let volume = vec![1000.0; 25];

        let output = detector.calculate_full(&high, &low, &close, &volume);

        // Should not detect absorption (volume not high enough)
        for &s in &output.signal[20..] {
            assert_eq!(s, 0.0, "Should not detect absorption with normal volume");
        }
    }

    #[test]
    fn test_absorption_detector_strong_signal() {
        let config = AbsorptionDetectorConfig {
            volume_period: 10,
            volume_threshold: 1.5,
            bullish_close_threshold: 0.7,
            bearish_close_threshold: 0.3,
            confirmation_bars: 2,
        };
        let detector = AbsorptionDetector::new(config).unwrap();

        // Create consecutive bullish absorption
        let mut high = vec![105.0; 15];
        let mut low = vec![100.0; 15];
        let mut close = vec![102.5; 15];
        let mut volume = vec![1000.0; 15];

        // First absorption
        high[11] = 106.0;
        low[11] = 100.0;
        close[11] = 105.5;
        volume[11] = 2000.0;

        // Second absorption (confirmation)
        high[12] = 107.0;
        low[12] = 101.0;
        close[12] = 106.5;
        volume[12] = 2200.0;

        let output = detector.calculate_full(&high, &low, &close, &volume);

        // Second should be strong bullish
        assert_eq!(output.signal[12], 2.0, "Should detect strong bullish");
        assert_eq!(output.absorption_type[12], AbsorptionType::StrongBullish);
    }

    #[test]
    fn test_absorption_detector_full_output() {
        let detector = AbsorptionDetector::default_config();
        let high = vec![105.0; 25];
        let low = vec![100.0; 25];
        let close = vec![103.0; 25];
        let volume = vec![1000.0; 25];

        let output = detector.calculate_full(&high, &low, &close, &volume);

        assert_eq!(output.absorption_type.len(), 25);
        assert_eq!(output.absorption_strength.len(), 25);
        assert_eq!(output.volume_ratio.len(), 25);
        assert_eq!(output.signal.len(), 25);
    }

    #[test]
    fn test_absorption_detector_invalid_config() {
        let config = AbsorptionDetectorConfig {
            volume_period: 1, // Invalid
            volume_threshold: 1.5,
            bullish_close_threshold: 0.7,
            bearish_close_threshold: 0.3,
            confirmation_bars: 2,
        };
        assert!(AbsorptionDetector::new(config).is_err());

        let config = AbsorptionDetectorConfig {
            volume_period: 20,
            volume_threshold: 0.5, // Invalid
            bullish_close_threshold: 0.7,
            bearish_close_threshold: 0.3,
            confirmation_bars: 2,
        };
        assert!(AbsorptionDetector::new(config).is_err());
    }

    #[test]
    fn test_absorption_type_enum() {
        let abs = AbsorptionType::BullishAbsorption;
        assert_eq!(abs, AbsorptionType::BullishAbsorption);
        assert_ne!(abs, AbsorptionType::BearishAbsorption);
    }

    #[test]
    fn test_absorption_detector_volume_ratio() {
        let detector = AbsorptionDetector::default_config();
        let high = vec![105.0; 25];
        let low = vec![100.0; 25];
        let close = vec![103.0; 25];
        let mut volume = vec![1000.0; 25];
        volume[22] = 2500.0; // 2.5x average

        let output = detector.calculate_full(&high, &low, &close, &volume);

        // Volume ratio should be ~2.5
        assert!((output.volume_ratio[22] - 2.5).abs() < 0.1);
    }
}
