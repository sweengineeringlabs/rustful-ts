//! Selling Climax Detector - High volume reversal detection (IND-232)
//!
//! Identifies potential selling climaxes using Wyckoff principles,
//! where panic selling creates high volume at price lows.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Selling Climax Detector configuration.
#[derive(Debug, Clone)]
pub struct SellingClimaxConfig {
    /// Lookback period for volume comparison
    pub volume_period: usize,
    /// Volume spike threshold (multiple of average)
    pub volume_threshold: f64,
    /// Lookback period for price lows
    pub price_period: usize,
    /// Spread expansion threshold (multiple of average)
    pub spread_threshold: f64,
    /// Close position threshold (0-1, lower = closer to low)
    pub close_position_threshold: f64,
}

impl Default for SellingClimaxConfig {
    fn default() -> Self {
        Self {
            volume_period: 20,
            volume_threshold: 2.0,
            price_period: 20,
            spread_threshold: 1.5,
            close_position_threshold: 0.3,
        }
    }
}

/// Climax event detection result.
#[derive(Debug, Clone, PartialEq)]
pub enum ClimaxEvent {
    /// Potential selling climax detected
    SellingClimax,
    /// Potential buying climax detected
    BuyingClimax,
    /// No climax event
    None,
}

/// Selling Climax Detector.
///
/// A selling climax is a Wyckoff pattern characterized by:
/// 1. Exceptionally high volume (panic selling)
/// 2. Wide price spread (high volatility)
/// 3. Close near the low of the bar
/// 4. Often at or near recent lows
///
/// This often marks the end of a downtrend and the beginning
/// of accumulation or a reversal.
///
/// The detector also identifies buying climaxes (distribution tops)
/// with opposite characteristics.
#[derive(Debug, Clone)]
pub struct SellingClimaxDetector {
    config: SellingClimaxConfig,
}

impl SellingClimaxDetector {
    pub fn new(volume_period: usize) -> Self {
        Self {
            config: SellingClimaxConfig {
                volume_period,
                price_period: volume_period,
                ..Default::default()
            },
        }
    }

    pub fn from_config(config: SellingClimaxConfig) -> Self {
        Self { config }
    }

    /// Calculate close position within the bar (0 = at low, 1 = at high).
    fn close_position(high: f64, low: f64, close: f64) -> f64 {
        let range = high - low;
        if range <= 0.0 {
            return 0.5;
        }
        (close - low) / range
    }

    /// Detect climax events.
    pub fn detect(&self, data: &OHLCVSeries) -> Vec<ClimaxEvent> {
        let n = data.close.len();
        let period = self.config.volume_period.max(self.config.price_period);

        if n < period {
            return vec![ClimaxEvent::None; n];
        }

        let mut events = vec![ClimaxEvent::None; n];

        // Calculate spreads
        let spreads: Vec<f64> = data.high.iter()
            .zip(data.low.iter())
            .map(|(h, l)| h - l)
            .collect();

        for i in period..n {
            // Calculate average volume over lookback
            let vol_start = i - self.config.volume_period;
            let avg_volume: f64 = data.volume[vol_start..i].iter().sum::<f64>()
                / self.config.volume_period as f64;

            // Calculate average spread over lookback
            let spread_start = i - self.config.volume_period;
            let avg_spread: f64 = spreads[spread_start..i].iter().sum::<f64>()
                / self.config.volume_period as f64;

            // Find lowest low and highest high in price period
            let price_start = i - self.config.price_period;
            let lowest_low = data.low[price_start..i]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let highest_high = data.high[price_start..i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let current_volume = data.volume[i];
            let current_spread = spreads[i];
            let close_pos = Self::close_position(data.high[i], data.low[i], data.close[i]);

            // Check for selling climax conditions
            let volume_spike = avg_volume > 0.0 &&
                current_volume >= avg_volume * self.config.volume_threshold;
            let spread_expansion = avg_spread > 0.0 &&
                current_spread >= avg_spread * self.config.spread_threshold;
            let near_low = data.low[i] <= lowest_low * 1.02; // Within 2% of recent low
            let close_near_bar_low = close_pos <= self.config.close_position_threshold;

            if volume_spike && spread_expansion && near_low && close_near_bar_low {
                events[i] = ClimaxEvent::SellingClimax;
                continue;
            }

            // Check for buying climax conditions (opposite)
            let near_high = data.high[i] >= highest_high * 0.98; // Within 2% of recent high
            let close_near_bar_high = close_pos >= (1.0 - self.config.close_position_threshold);

            if volume_spike && spread_expansion && near_high && close_near_bar_high {
                events[i] = ClimaxEvent::BuyingClimax;
            }
        }

        events
    }

    /// Calculate climax intensity score (0-100).
    /// Higher values indicate stronger climax conditions.
    pub fn calculate_intensity(&self, data: &OHLCVSeries) -> Vec<f64> {
        let n = data.close.len();
        let period = self.config.volume_period.max(self.config.price_period);

        if n < period {
            return vec![f64::NAN; n];
        }

        let mut intensity = vec![f64::NAN; n];

        // Calculate spreads
        let spreads: Vec<f64> = data.high.iter()
            .zip(data.low.iter())
            .map(|(h, l)| h - l)
            .collect();

        for i in period..n {
            // Calculate average volume
            let vol_start = i - self.config.volume_period;
            let avg_volume: f64 = data.volume[vol_start..i].iter().sum::<f64>()
                / self.config.volume_period as f64;

            // Calculate average spread
            let spread_start = i - self.config.volume_period;
            let avg_spread: f64 = spreads[spread_start..i].iter().sum::<f64>()
                / self.config.volume_period as f64;

            if avg_volume <= 0.0 || avg_spread <= 0.0 {
                intensity[i] = 0.0;
                continue;
            }

            let volume_ratio = data.volume[i] / avg_volume;
            let spread_ratio = spreads[i] / avg_spread;
            let close_pos = Self::close_position(data.high[i], data.low[i], data.close[i]);

            // Intensity based on extremeness of close position (0 or 1 = extreme)
            let close_extremity = 2.0 * (0.5 - close_pos).abs();

            // Combine factors into intensity score
            let raw_intensity = (volume_ratio - 1.0).max(0.0) * 20.0
                + (spread_ratio - 1.0).max(0.0) * 20.0
                + close_extremity * 30.0;

            intensity[i] = raw_intensity.min(100.0);
        }

        intensity
    }
}

impl Default for SellingClimaxDetector {
    fn default() -> Self {
        Self::from_config(SellingClimaxConfig::default())
    }
}

impl TechnicalIndicator for SellingClimaxDetector {
    fn name(&self) -> &str {
        "SellingClimaxDetector"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let period = self.config.volume_period.max(self.config.price_period);
        if data.close.len() < period {
            return Err(IndicatorError::InsufficientData {
                required: period,
                got: data.close.len(),
            });
        }

        let intensity = self.calculate_intensity(data);
        Ok(IndicatorOutput::single(intensity))
    }

    fn min_periods(&self) -> usize {
        self.config.volume_period.max(self.config.price_period)
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for SellingClimaxDetector {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let events = self.detect(data);

        if let Some(last) = events.last() {
            match last {
                ClimaxEvent::SellingClimax => return Ok(IndicatorSignal::Bullish),
                ClimaxEvent::BuyingClimax => return Ok(IndicatorSignal::Bearish),
                ClimaxEvent::None => {}
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let events = self.detect(data);

        Ok(events
            .iter()
            .map(|e| match e {
                ClimaxEvent::SellingClimax => IndicatorSignal::Bullish,
                ClimaxEvent::BuyingClimax => IndicatorSignal::Bearish,
                ClimaxEvent::None => IndicatorSignal::Neutral,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_selling_climax_data(n: usize, climax_idx: usize) -> OHLCVSeries {
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0 - (i as f64) * 0.5; // Downtrend

            if i == climax_idx {
                // Create selling climax bar
                open.push(base);
                high.push(base + 1.0);
                low.push(base - 5.0); // Wide spread
                close.push(base - 4.5); // Close near low
                volume.push(5000.0); // High volume spike
            } else {
                open.push(base);
                high.push(base + 1.0);
                low.push(base - 1.0);
                close.push(base - 0.5);
                volume.push(1000.0);
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
    fn test_selling_climax_detection() {
        let detector = SellingClimaxDetector::from_config(SellingClimaxConfig {
            volume_period: 10,
            volume_threshold: 2.0,
            price_period: 10,
            spread_threshold: 1.5,
            close_position_threshold: 0.3,
        });

        let data = create_selling_climax_data(25, 20);
        let events = detector.detect(&data);

        // Should detect the selling climax at index 20
        assert_eq!(events[20], ClimaxEvent::SellingClimax);
    }

    #[test]
    fn test_no_climax_in_normal_data() {
        let detector = SellingClimaxDetector::new(10);
        let data = create_normal_data(30);
        let events = detector.detect(&data);

        // Should not detect any climax in normal data
        for event in events {
            assert_eq!(event, ClimaxEvent::None);
        }
    }

    #[test]
    fn test_climax_intensity() {
        let detector = SellingClimaxDetector::new(10);
        let data = create_selling_climax_data(25, 20);
        let intensity = detector.calculate_intensity(&data);

        assert_eq!(intensity.len(), 25);

        // Intensity at climax bar should be higher
        let climax_intensity = intensity[20];
        let normal_intensity = intensity[15];

        if !climax_intensity.is_nan() && !normal_intensity.is_nan() {
            assert!(climax_intensity > normal_intensity);
        }
    }

    #[test]
    fn test_close_position() {
        // Close at low
        let pos = SellingClimaxDetector::close_position(110.0, 100.0, 100.0);
        assert!(pos.abs() < 0.001);

        // Close at high
        let pos = SellingClimaxDetector::close_position(110.0, 100.0, 110.0);
        assert!((pos - 1.0).abs() < 0.001);

        // Close at midpoint
        let pos = SellingClimaxDetector::close_position(110.0, 100.0, 105.0);
        assert!((pos - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_signal_from_selling_climax() {
        let detector = SellingClimaxDetector::from_config(SellingClimaxConfig {
            volume_period: 10,
            volume_threshold: 2.0,
            price_period: 10,
            spread_threshold: 1.5,
            close_position_threshold: 0.3,
        });

        let data = create_selling_climax_data(25, 24); // Climax at last bar
        let signal = detector.signal(&data).unwrap();

        // Selling climax should give bullish signal (potential bottom)
        assert!(matches!(signal, IndicatorSignal::Bullish));
    }
}
