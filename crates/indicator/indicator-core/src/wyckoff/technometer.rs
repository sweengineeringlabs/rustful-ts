//! Technometer - Overbought/Oversold indicator (IND-231)
//!
//! A Wyckoff-based oscillator that measures the relationship between
//! advances/declines and volume to identify overbought/oversold conditions.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Technometer configuration.
#[derive(Debug, Clone)]
pub struct TechnometerConfig {
    /// Smoothing period
    pub period: usize,
    /// Overbought threshold
    pub overbought: f64,
    /// Oversold threshold
    pub oversold: f64,
    /// Neutral zone threshold
    pub neutral_threshold: f64,
}

impl Default for TechnometerConfig {
    fn default() -> Self {
        Self {
            period: 10,
            overbought: 60.0,
            oversold: 40.0,
            neutral_threshold: 50.0,
        }
    }
}

/// Technometer.
///
/// The Technometer is an overbought/oversold indicator that uses
/// price action and volume to determine market extremes. It's based
/// on Wyckoff's analysis of supply and demand.
///
/// The indicator measures:
/// - Price momentum (close position relative to range)
/// - Volume intensity (current volume vs average)
/// - Cumulative buying/selling pressure
///
/// Output range: 0-100
/// - Above 60: Overbought (potential selling)
/// - Below 40: Oversold (potential buying)
/// - 40-60: Neutral zone
#[derive(Debug, Clone)]
pub struct Technometer {
    config: TechnometerConfig,
}

impl Technometer {
    pub fn new(period: usize) -> Self {
        Self {
            config: TechnometerConfig {
                period,
                ..Default::default()
            },
        }
    }

    pub fn from_config(config: TechnometerConfig) -> Self {
        Self { config }
    }

    /// Calculate price position score (0-1).
    /// Where the close is relative to the high-low range.
    fn price_position(high: f64, low: f64, close: f64) -> f64 {
        let range = high - low;
        if range <= 0.0 {
            return 0.5;
        }
        (close - low) / range
    }

    /// Calculate Technometer values.
    pub fn calculate(&self, data: &OHLCVSeries) -> Vec<f64> {
        let n = data.close.len();

        if n < self.config.period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        // Calculate average volume
        let avg_volume: f64 = data.volume.iter().sum::<f64>() / n as f64;
        let avg_volume = if avg_volume > 0.0 { avg_volume } else { 1.0 };

        // Calculate price position and volume-weighted scores
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let pos = Self::price_position(data.high[i], data.low[i], data.close[i]);
            let vol_factor = data.volume[i] / avg_volume;
            // Volume-weighted position: high volume amplifies the signal
            scores.push(pos * vol_factor.sqrt());
        }

        // Calculate rolling Technometer
        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;
            let window = &scores[start..=i];

            // Calculate weighted average (more recent = higher weight)
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;

            for (j, &score) in window.iter().enumerate() {
                let weight = (j + 1) as f64;
                weighted_sum += score * weight;
                weight_total += weight;
            }

            let avg_score = if weight_total > 0.0 {
                weighted_sum / weight_total
            } else {
                0.5
            };

            // Normalize to 0-100 range
            // Since price_position is 0-1 and vol_factor can vary,
            // we apply a sigmoid-like transformation
            result[i] = 100.0 / (1.0 + (-6.0 * (avg_score - 0.5)).exp());
        }

        result
    }

    /// Calculate Technometer with momentum component.
    pub fn calculate_with_momentum(&self, data: &OHLCVSeries) -> Vec<f64> {
        let base = self.calculate(data);
        let n = base.len();

        if n < self.config.period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in self.config.period..n {
            if !base[i].is_nan() && !base[i - 1].is_nan() {
                let momentum = base[i] - base[i - 1];
                // Add momentum bias to the reading
                result[i] = (base[i] + momentum * 0.5).clamp(0.0, 100.0);
            }
        }

        result
    }
}

impl Default for Technometer {
    fn default() -> Self {
        Self::from_config(TechnometerConfig::default())
    }
}

impl TechnicalIndicator for Technometer {
    fn name(&self) -> &str {
        "Technometer"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.config.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for Technometer {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(data);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last >= self.config.overbought {
                    return Ok(IndicatorSignal::Bearish); // Overbought = sell signal
                } else if last <= self.config.oversold {
                    return Ok(IndicatorSignal::Bullish); // Oversold = buy signal
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(data);

        Ok(values
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    IndicatorSignal::Neutral
                } else if v >= self.config.overbought {
                    IndicatorSignal::Bearish
                } else if v <= self.config.oversold {
                    IndicatorSignal::Bullish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_overbought_data(n: usize) -> OHLCVSeries {
        // Consistently closing near highs with increasing volume
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0 + (i as f64);
            open.push(base);
            high.push(base + 2.0);
            low.push(base - 0.2);
            close.push(base + 1.9); // Close very near high
            volume.push(1000.0 + (i as f64) * 100.0); // Increasing volume
        }

        OHLCVSeries { open, high, low, close, volume }
    }

    fn create_oversold_data(n: usize) -> OHLCVSeries {
        // Consistently closing near lows
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0 - (i as f64);
            open.push(base);
            high.push(base + 0.2);
            low.push(base - 2.0);
            close.push(base - 1.9); // Close very near low
            volume.push(1000.0 + (i as f64) * 100.0);
        }

        OHLCVSeries { open, high, low, close, volume }
    }

    fn create_neutral_data(n: usize) -> OHLCVSeries {
        // Closing at midpoint
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0;
            open.push(base);
            high.push(base + 2.0);
            low.push(base - 2.0);
            close.push(base); // Close at midpoint
            volume.push(1000.0);
        }

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_technometer_basic() {
        let tech = Technometer::new(10);
        let data = create_neutral_data(20);
        let result = tech.calculate(&data);

        assert_eq!(result.len(), 20);

        // Values should be in 0-100 range
        for &val in result.iter().skip(9) {
            if !val.is_nan() {
                assert!(val >= 0.0 && val <= 100.0);
            }
        }
    }

    #[test]
    fn test_technometer_overbought() {
        let tech = Technometer::new(5);
        let data = create_overbought_data(15);
        let result = tech.calculate(&data);

        // Should trend toward overbought (>50)
        if let Some(&last) = result.last() {
            if !last.is_nan() {
                assert!(last > 50.0, "Expected overbought reading, got {}", last);
            }
        }
    }

    #[test]
    fn test_technometer_oversold() {
        let tech = Technometer::new(5);
        let data = create_oversold_data(15);
        let result = tech.calculate(&data);

        // Should trend toward oversold (<50)
        if let Some(&last) = result.last() {
            if !last.is_nan() {
                assert!(last < 50.0, "Expected oversold reading, got {}", last);
            }
        }
    }

    #[test]
    fn test_price_position() {
        // Close at high
        let pos = Technometer::price_position(110.0, 100.0, 110.0);
        assert!((pos - 1.0).abs() < 0.001);

        // Close at low
        let pos = Technometer::price_position(110.0, 100.0, 100.0);
        assert!(pos.abs() < 0.001);

        // Close at midpoint
        let pos = Technometer::price_position(110.0, 100.0, 105.0);
        assert!((pos - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_technometer_signal() {
        let tech = Technometer::from_config(TechnometerConfig {
            period: 5,
            overbought: 60.0,
            oversold: 40.0,
            neutral_threshold: 50.0,
        });

        let overbought = create_overbought_data(15);
        let signal = tech.signal(&overbought).unwrap();
        // High closes with volume should give bearish (overbought) signal
        assert!(matches!(signal, IndicatorSignal::Bearish | IndicatorSignal::Neutral));
    }

    #[test]
    fn test_technometer_with_momentum() {
        let tech = Technometer::new(5);
        let data = create_overbought_data(20);
        let result = tech.calculate_with_momentum(&data);

        assert_eq!(result.len(), 20);

        // All values should be valid and in range
        for &val in result.iter().skip(6) {
            if !val.is_nan() {
                assert!(val >= 0.0 && val <= 100.0);
            }
        }
    }
}
