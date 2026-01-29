//! Sign of Strength/Weakness - Trend quality indicator (IND-235)
//!
//! Identifies signs of strength (SOS) and signs of weakness (SOW) in Wyckoff
//! methodology, which indicate the quality and sustainability of trend moves.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Sign of Strength/Weakness signal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SOSSOWType {
    /// No significant sign detected
    None,
    /// Sign of Strength - bullish indication
    SignOfStrength,
    /// Sign of Weakness - bearish indication
    SignOfWeakness,
    /// Minor sign of strength
    MinorSOS,
    /// Minor sign of weakness
    MinorSOW,
}

/// Sign of Strength/Weakness configuration.
#[derive(Debug, Clone)]
pub struct SignOfStrengthConfig {
    /// Period for trend analysis
    pub period: usize,
    /// Volume analysis period
    pub volume_period: usize,
    /// Spread threshold (multiple of average spread)
    pub spread_threshold: f64,
    /// Volume threshold (multiple of average volume)
    pub volume_threshold: f64,
    /// Close position threshold (0-1, how close to high/low)
    pub close_position_threshold: f64,
}

impl Default for SignOfStrengthConfig {
    fn default() -> Self {
        Self {
            period: 14,
            volume_period: 10,
            spread_threshold: 1.5,
            volume_threshold: 1.3,
            close_position_threshold: 0.7,
        }
    }
}

/// Sign of Strength/Weakness output.
#[derive(Debug, Clone)]
pub struct SignOfStrengthOutput {
    /// Signal type at each bar
    pub signal_type: Vec<SOSSOWType>,
    /// Strength score (-1 to 1, negative for weakness, positive for strength)
    pub strength_score: Vec<f64>,
    /// Effort (volume relative to average)
    pub effort: Vec<f64>,
    /// Result (price spread relative to average)
    pub result: Vec<f64>,
    /// Close position within bar (0 = low, 1 = high)
    pub close_position: Vec<f64>,
}

/// Sign of Strength/Weakness Indicator (IND-235).
///
/// This indicator identifies Wyckoff signs of strength and weakness by analyzing:
///
/// **Sign of Strength (SOS):**
/// - Wide spread up bar (large price range with upward movement)
/// - High volume (effort matches result)
/// - Close near the high of the bar
/// - Indicates strong buying pressure, sustainable uptrend
///
/// **Sign of Weakness (SOW):**
/// - Wide spread down bar (large price range with downward movement)
/// - High volume on down bars
/// - Close near the low of the bar
/// - Indicates strong selling pressure, sustainable downtrend
///
/// The indicator also detects:
/// - No demand (narrow spread up bar on low volume)
/// - No supply (narrow spread down bar on low volume)
/// - Effort vs Result divergences
#[derive(Debug, Clone)]
pub struct SignOfStrengthWeakness {
    config: SignOfStrengthConfig,
}

impl SignOfStrengthWeakness {
    pub fn new(period: usize) -> Self {
        Self {
            config: SignOfStrengthConfig {
                period,
                ..Default::default()
            },
        }
    }

    pub fn from_config(config: SignOfStrengthConfig) -> Self {
        Self { config }
    }

    /// Calculate close position within bar (0 = low, 1 = high).
    fn calculate_close_position(high: f64, low: f64, close: f64) -> f64 {
        let range = high - low;
        if range <= 0.0 {
            return 0.5;
        }
        (close - low) / range
    }

    /// Calculate average spread.
    fn average_spread(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len());
        let mut result = vec![f64::NAN; n];

        let spreads: Vec<f64> = (0..n).map(|i| high[i] - low[i]).collect();

        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;
            let sum: f64 = spreads[start..=i].iter().sum();
            result[i] = sum / self.config.period as f64;
        }

        result
    }

    /// Calculate average volume.
    fn average_volume(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![f64::NAN; n];

        for i in (self.config.volume_period - 1)..n {
            let start = i + 1 - self.config.volume_period;
            let sum: f64 = volume[start..=i].iter().sum();
            result[i] = sum / self.config.volume_period as f64;
        }

        result
    }

    /// Calculate Sign of Strength/Weakness.
    pub fn calculate(&self, data: &OHLCVSeries) -> SignOfStrengthOutput {
        let n = data.close.len();
        let min_period = self.config.period.max(self.config.volume_period);

        if n < min_period + 1 {
            return SignOfStrengthOutput {
                signal_type: vec![SOSSOWType::None; n],
                strength_score: vec![0.0; n],
                effort: vec![f64::NAN; n],
                result: vec![f64::NAN; n],
                close_position: vec![f64::NAN; n],
            };
        }

        let mut signal_type = vec![SOSSOWType::None; n];
        let mut strength_score = vec![0.0; n];
        let mut effort = vec![f64::NAN; n];
        let mut result_vec = vec![f64::NAN; n];
        let mut close_position = vec![f64::NAN; n];

        // Calculate averages
        let avg_spread = self.average_spread(&data.high, &data.low);
        let avg_volume = self.average_volume(&data.volume);

        // Calculate indicators for each bar
        for i in min_period..n {
            let high = data.high[i];
            let low = data.low[i];
            let open = data.open[i];
            let close = data.close[i];
            let volume = data.volume[i];

            let spread = high - low;
            let avg_spr = avg_spread[i];
            let avg_vol = avg_volume[i];

            if avg_spr <= 0.0 || avg_vol <= 0.0 || avg_spr.is_nan() || avg_vol.is_nan() {
                continue;
            }

            // Calculate metrics
            let spread_ratio = spread / avg_spr;
            let volume_ratio = volume / avg_vol;
            let close_pos = Self::calculate_close_position(high, low, close);

            effort[i] = volume_ratio;
            result_vec[i] = spread_ratio;
            close_position[i] = close_pos;

            // Determine bar direction
            let is_up_bar = close > open;
            let is_down_bar = close < open;

            // Detect Sign of Strength
            if is_up_bar
                && spread_ratio >= self.config.spread_threshold
                && volume_ratio >= self.config.volume_threshold
                && close_pos >= self.config.close_position_threshold
            {
                signal_type[i] = SOSSOWType::SignOfStrength;

                // Calculate strength score
                let spread_score = (spread_ratio / self.config.spread_threshold).min(2.0) / 2.0;
                let volume_score = (volume_ratio / self.config.volume_threshold).min(2.0) / 2.0;
                let close_score = close_pos;

                strength_score[i] = (spread_score + volume_score + close_score) / 3.0;
            }
            // Detect Minor Sign of Strength (good close position but lower metrics)
            else if is_up_bar
                && spread_ratio >= 1.0
                && volume_ratio >= 1.0
                && close_pos >= self.config.close_position_threshold
            {
                signal_type[i] = SOSSOWType::MinorSOS;
                strength_score[i] = close_pos * 0.5;
            }
            // Detect Sign of Weakness
            else if is_down_bar
                && spread_ratio >= self.config.spread_threshold
                && volume_ratio >= self.config.volume_threshold
                && close_pos <= (1.0 - self.config.close_position_threshold)
            {
                signal_type[i] = SOSSOWType::SignOfWeakness;

                // Calculate weakness score (negative)
                let spread_score = (spread_ratio / self.config.spread_threshold).min(2.0) / 2.0;
                let volume_score = (volume_ratio / self.config.volume_threshold).min(2.0) / 2.0;
                let close_score = 1.0 - close_pos;

                strength_score[i] = -(spread_score + volume_score + close_score) / 3.0;
            }
            // Detect Minor Sign of Weakness
            else if is_down_bar
                && spread_ratio >= 1.0
                && volume_ratio >= 1.0
                && close_pos <= (1.0 - self.config.close_position_threshold)
            {
                signal_type[i] = SOSSOWType::MinorSOW;
                strength_score[i] = -(1.0 - close_pos) * 0.5;
            }
            // Check for No Demand (narrow spread up bar, low volume)
            else if is_up_bar && spread_ratio < 0.7 && volume_ratio < 0.7 {
                // No demand - bearish implication (hidden weakness)
                signal_type[i] = SOSSOWType::MinorSOW;
                strength_score[i] = -0.2;
            }
            // Check for No Supply (narrow spread down bar, low volume)
            else if is_down_bar && spread_ratio < 0.7 && volume_ratio < 0.7 {
                // No supply - bullish implication (hidden strength)
                signal_type[i] = SOSSOWType::MinorSOS;
                strength_score[i] = 0.2;
            }
        }

        SignOfStrengthOutput {
            signal_type,
            strength_score,
            effort,
            result: result_vec,
            close_position,
        }
    }
}

impl Default for SignOfStrengthWeakness {
    fn default() -> Self {
        Self::from_config(SignOfStrengthConfig::default())
    }
}

impl TechnicalIndicator for SignOfStrengthWeakness {
    fn name(&self) -> &str {
        "SignOfStrengthWeakness"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_period = self.config.period.max(self.config.volume_period);

        if data.close.len() < min_period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: min_period + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);

        // Encode signal type: 0 = None, 1 = SOS, -1 = SOW, 0.5 = MinorSOS, -0.5 = MinorSOW
        let signal_encoded: Vec<f64> = result
            .signal_type
            .iter()
            .map(|&s| match s {
                SOSSOWType::None => 0.0,
                SOSSOWType::SignOfStrength => 1.0,
                SOSSOWType::SignOfWeakness => -1.0,
                SOSSOWType::MinorSOS => 0.5,
                SOSSOWType::MinorSOW => -0.5,
            })
            .collect();

        Ok(IndicatorOutput::triple(
            signal_encoded,
            result.strength_score,
            result.close_position,
        ))
    }

    fn min_periods(&self) -> usize {
        self.config.period.max(self.config.volume_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for SignOfStrengthWeakness {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);

        if let Some(&signal) = result.signal_type.last() {
            match signal {
                SOSSOWType::SignOfStrength | SOSSOWType::MinorSOS => {
                    return Ok(IndicatorSignal::Bullish);
                }
                SOSSOWType::SignOfWeakness | SOSSOWType::MinorSOW => {
                    return Ok(IndicatorSignal::Bearish);
                }
                SOSSOWType::None => {}
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);

        Ok(result
            .signal_type
            .iter()
            .map(|&s| match s {
                SOSSOWType::SignOfStrength | SOSSOWType::MinorSOS => IndicatorSignal::Bullish,
                SOSSOWType::SignOfWeakness | SOSSOWType::MinorSOW => IndicatorSignal::Bearish,
                SOSSOWType::None => IndicatorSignal::Neutral,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_sos_data() -> OHLCVSeries {
        // Create data with a Sign of Strength bar at the end
        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();

        // Normal bars first (14 bars for averages)
        for _ in 0..14 {
            open.push(100.0);
            high.push(101.0);
            low.push(99.0);
            close.push(100.5);
            volume.push(1000.0);
        }

        // Sign of Strength bar: wide spread, high volume, close near high
        open.push(100.0);
        high.push(104.0); // Wide spread (4 points vs avg 2)
        low.push(99.5);
        close.push(103.8); // Close near high
        volume.push(2000.0); // High volume (2x avg)

        OHLCVSeries { open, high, low, close, volume }
    }

    fn create_sow_data() -> OHLCVSeries {
        // Create data with a Sign of Weakness bar at the end
        let mut open = Vec::new();
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();
        let mut volume = Vec::new();

        // Normal bars first
        for _ in 0..14 {
            open.push(100.0);
            high.push(101.0);
            low.push(99.0);
            close.push(100.5);
            volume.push(1000.0);
        }

        // Sign of Weakness bar: wide spread down, high volume, close near low
        open.push(100.0);
        high.push(100.5);
        low.push(96.0); // Wide spread down (4.5 points)
        close.push(96.2); // Close near low
        volume.push(2000.0); // High volume

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_sos_detection() {
        let indicator = SignOfStrengthWeakness::new(10);
        let data = create_sos_data();
        let result = indicator.calculate(&data);

        let last_idx = result.signal_type.len() - 1;
        assert_eq!(result.signal_type[last_idx], SOSSOWType::SignOfStrength);
        assert!(result.strength_score[last_idx] > 0.0);
    }

    #[test]
    fn test_sow_detection() {
        let indicator = SignOfStrengthWeakness::new(10);
        let data = create_sow_data();
        let result = indicator.calculate(&data);

        let last_idx = result.signal_type.len() - 1;
        assert_eq!(result.signal_type[last_idx], SOSSOWType::SignOfWeakness);
        assert!(result.strength_score[last_idx] < 0.0);
    }

    #[test]
    fn test_close_position() {
        // Close at high
        let pos = SignOfStrengthWeakness::calculate_close_position(110.0, 100.0, 110.0);
        assert!((pos - 1.0).abs() < 0.001);

        // Close at low
        let pos = SignOfStrengthWeakness::calculate_close_position(110.0, 100.0, 100.0);
        assert!(pos.abs() < 0.001);

        // Close at midpoint
        let pos = SignOfStrengthWeakness::calculate_close_position(110.0, 100.0, 105.0);
        assert!((pos - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_effort_result_calculation() {
        let indicator = SignOfStrengthWeakness::new(10);
        let data = create_sos_data();
        let result = indicator.calculate(&data);

        let last_idx = result.effort.len() - 1;

        // Effort should be ~2.0 (2000/1000)
        assert!(!result.effort[last_idx].is_nan());
        assert!(result.effort[last_idx] > 1.5);

        // Result (spread ratio) should also be high
        assert!(!result.result[last_idx].is_nan());
        assert!(result.result[last_idx] > 1.5);
    }

    #[test]
    fn test_signal_generation() {
        let indicator = SignOfStrengthWeakness::new(10);

        let sos_data = create_sos_data();
        let sos_signal = indicator.signal(&sos_data).unwrap();
        assert_eq!(sos_signal, IndicatorSignal::Bullish);

        let sow_data = create_sow_data();
        let sow_signal = indicator.signal(&sow_data).unwrap();
        assert_eq!(sow_signal, IndicatorSignal::Bearish);
    }

    #[test]
    fn test_neutral_bars() {
        let indicator = SignOfStrengthWeakness::new(10);

        let mut data = OHLCVSeries::new();
        for _ in 0..20 {
            data.open.push(100.0);
            data.high.push(101.0);
            data.low.push(99.0);
            data.close.push(100.5);
            data.volume.push(1000.0);
        }

        let result = indicator.calculate(&data);

        // Most bars should be None (no strong signals)
        let non_zero_count = result
            .signal_type
            .iter()
            .filter(|&&s| s != SOSSOWType::None)
            .count();
        assert!(non_zero_count < 5); // Most should be neutral
    }

    #[test]
    fn test_insufficient_data() {
        let indicator = SignOfStrengthWeakness::new(14);

        let mut data = OHLCVSeries::new();
        for _ in 0..10 {
            data.open.push(100.0);
            data.high.push(101.0);
            data.low.push(99.0);
            data.close.push(100.5);
            data.volume.push(1000.0);
        }

        let result = indicator.compute(&data);
        assert!(result.is_err());
    }
}
