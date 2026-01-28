//! Elder's SafeZone Stop implementation.

use indicator_api::SafeZoneStopConfig;
use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// SafeZone Stop output.
#[derive(Debug, Clone)]
pub struct SafeZoneStopOutput {
    /// Long stop (trailing stop for uptrend/long positions).
    pub long_stop: Vec<f64>,
    /// Short stop (trailing stop for downtrend/short positions).
    pub short_stop: Vec<f64>,
}

/// Elder's SafeZone Stop indicator.
///
/// A directional stop loss indicator that adjusts based on recent price action.
/// Provides separate calculations for long and short positions.
///
/// ## Algorithm
///
/// **For uptrend (long stop):**
/// 1. Find downside penetrations: bars where low < previous low
/// 2. Calculate penetration distance: previous_low - current_low
/// 3. Average the penetration distances over the lookback period
/// 4. Long stop = current_low - (coefficient × average_penetration)
///
/// **For downtrend (short stop):**
/// 1. Find upside penetrations: bars where high > previous high
/// 2. Calculate penetration distance: current_high - previous_high
/// 3. Average the penetration distances over the lookback period
/// 4. Short stop = current_high + (coefficient × average_penetration)
///
/// ## Parameters
/// - `period`: Lookback period for averaging penetrations (default: 10)
/// - `coefficient`: Multiplier for average penetration distance (default: 2.5)
///
/// ## Interpretation
/// - In an uptrend, place stop loss at the long_stop level
/// - In a downtrend, place stop loss at the short_stop level
/// - Higher coefficient = wider stops, more room for price movement
/// - Lower coefficient = tighter stops, less tolerance for adverse moves
#[derive(Debug, Clone)]
pub struct SafeZoneStop {
    period: usize,
    coefficient: f64,
}

impl SafeZoneStop {
    /// Create a new SafeZone Stop indicator.
    pub fn new(period: usize, coefficient: f64) -> Self {
        Self {
            period,
            coefficient,
        }
    }

    /// Create from configuration.
    pub fn from_config(config: SafeZoneStopConfig) -> Self {
        Self {
            period: config.period,
            coefficient: config.coefficient,
        }
    }

    /// Calculate SafeZone Stop values.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> SafeZoneStopOutput {
        let n = high.len();

        // Need at least period + 1 bars (1 for previous comparison)
        if n < self.period + 1 {
            return SafeZoneStopOutput {
                long_stop: vec![f64::NAN; n],
                short_stop: vec![f64::NAN; n],
            };
        }

        let mut long_stop = vec![f64::NAN; n];
        let mut short_stop = vec![f64::NAN; n];

        // Calculate downside penetrations (for long stop)
        // Penetration occurs when low < previous low
        let mut down_penetrations = vec![0.0; n];
        for i in 1..n {
            if low[i] < low[i - 1] {
                down_penetrations[i] = low[i - 1] - low[i];
            }
        }

        // Calculate upside penetrations (for short stop)
        // Penetration occurs when high > previous high
        let mut up_penetrations = vec![0.0; n];
        for i in 1..n {
            if high[i] > high[i - 1] {
                up_penetrations[i] = high[i] - high[i - 1];
            }
        }

        // Calculate average penetrations and stops
        for i in self.period..n {
            // Average downside penetration over lookback period
            let mut down_sum = 0.0;
            let mut down_count = 0;
            for j in (i - self.period + 1)..=i {
                if down_penetrations[j] > 0.0 {
                    down_sum += down_penetrations[j];
                    down_count += 1;
                }
            }
            let avg_down = if down_count > 0 {
                down_sum / down_count as f64
            } else {
                0.0
            };

            // Average upside penetration over lookback period
            let mut up_sum = 0.0;
            let mut up_count = 0;
            for j in (i - self.period + 1)..=i {
                if up_penetrations[j] > 0.0 {
                    up_sum += up_penetrations[j];
                    up_count += 1;
                }
            }
            let avg_up = if up_count > 0 {
                up_sum / up_count as f64
            } else {
                0.0
            };

            // Long stop: below current low
            long_stop[i] = low[i] - self.coefficient * avg_down;

            // Short stop: above current high
            short_stop[i] = high[i] + self.coefficient * avg_up;
        }

        SafeZoneStopOutput {
            long_stop,
            short_stop,
        }
    }
}

impl Default for SafeZoneStop {
    fn default() -> Self {
        Self::from_config(SafeZoneStopConfig::default())
    }
}

impl TechnicalIndicator for SafeZoneStop {
    fn name(&self) -> &str {
        "SafeZoneStop"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.high.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(result.long_stop, result.short_stop))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for SafeZoneStop {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low);

        // Get the last valid values
        let last_long = result.long_stop.last().copied().unwrap_or(f64::NAN);
        let last_short = result.short_stop.last().copied().unwrap_or(f64::NAN);
        let last_close = data.close.last().copied().unwrap_or(f64::NAN);

        if last_long.is_nan() || last_short.is_nan() || last_close.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal based on price position relative to stops
        // Price above long stop = bullish (uptrend intact)
        // Price below short stop = bearish (downtrend intact)
        if last_close > last_long && last_close < last_short {
            // Price is within the "safe zone" - neutral
            Ok(IndicatorSignal::Neutral)
        } else if last_close > last_short {
            // Price broke above short stop - bullish breakout
            Ok(IndicatorSignal::Bullish)
        } else if last_close < last_long {
            // Price broke below long stop - bearish breakout
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low);
        let n = data.close.len();

        let signals = (0..n)
            .map(|i| {
                let long = result.long_stop[i];
                let short = result.short_stop[i];
                let close = data.close[i];

                if long.is_nan() || short.is_nan() || close.is_nan() {
                    IndicatorSignal::Neutral
                } else if close > short {
                    IndicatorSignal::Bullish
                } else if close < long {
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

    #[test]
    fn test_safezone_stop_basic() {
        let sz = SafeZoneStop::new(10, 2.5);
        let n = 30;

        // Create price data with clear trend patterns
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.3)).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.3)).collect();

        let result = sz.calculate(&high, &low);

        assert_eq!(result.long_stop.len(), n);
        assert_eq!(result.short_stop.len(), n);

        // First 10 values should be NaN (need period + 1 bars)
        for i in 0..10 {
            assert!(result.long_stop[i].is_nan());
            assert!(result.short_stop[i].is_nan());
        }

        // Values from period onwards should be valid
        for i in 10..n {
            assert!(
                !result.long_stop[i].is_nan(),
                "long_stop[{}] should not be NaN",
                i
            );
            assert!(
                !result.short_stop[i].is_nan(),
                "short_stop[{}] should not be NaN",
                i
            );

            // Long stop should be below low
            assert!(result.long_stop[i] <= low[i], "long_stop should be <= low");

            // Short stop should be above high
            assert!(
                result.short_stop[i] >= high[i],
                "short_stop should be >= high"
            );
        }
    }

    #[test]
    fn test_safezone_stop_with_penetrations() {
        let sz = SafeZoneStop::new(5, 2.0);

        // Create data with some penetrations
        let high = vec![
            100.0, 101.0, 100.5, 102.0, 101.5, // First 5 bars
            103.0, 102.5, 104.0, 103.5, 105.0, // Next 5 bars
            104.5, 106.0, 105.5, 107.0, 106.5, // Last 5 bars
        ];
        let low = vec![
            95.0, 94.0, 94.5, 93.0, 93.5, // Downside penetrations
            94.0, 94.5, 93.5, 94.0, 93.0, // Mixed
            94.5, 93.5, 95.0, 94.0, 95.5, // Mixed
        ];

        let result = sz.calculate(&high, &low);

        assert_eq!(result.long_stop.len(), high.len());
        assert_eq!(result.short_stop.len(), high.len());

        // Verify that stops are calculated for valid indices
        for i in 5..high.len() {
            assert!(!result.long_stop[i].is_nan());
            assert!(!result.short_stop[i].is_nan());
        }
    }

    #[test]
    fn test_safezone_stop_no_penetrations() {
        let sz = SafeZoneStop::new(5, 2.0);

        // Create monotonically increasing data (no downside penetrations)
        // and monotonically decreasing highs (no upside penetrations)
        let n = 15;
        let high: Vec<f64> = (0..n).map(|i| 100.0 - i as f64).collect(); // Decreasing
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect(); // Increasing

        let result = sz.calculate(&high, &low);

        // With no penetrations, avg penetration = 0
        // Long stop = low - 0 = low
        // Short stop = high + 0 = high
        for i in 5..n {
            assert!((result.long_stop[i] - low[i]).abs() < 1e-10);
            assert!((result.short_stop[i] - high[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_safezone_stop_default() {
        let sz = SafeZoneStop::default();
        assert_eq!(sz.period, 10);
        assert!((sz.coefficient - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_safezone_stop_technical_indicator() {
        let sz = SafeZoneStop::new(10, 2.5);

        assert_eq!(sz.name(), "SafeZoneStop");
        assert_eq!(sz.min_periods(), 11);
        assert_eq!(sz.output_features(), 2);
    }

    #[test]
    fn test_safezone_stop_insufficient_data() {
        let sz = SafeZoneStop::new(10, 2.5);

        let data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![101.0; 5],
            low: vec![99.0; 5],
            close: vec![100.5; 5],
            volume: vec![1000.0; 5],
        };

        let result = sz.compute(&data);
        assert!(result.is_err());

        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 11);
            assert_eq!(got, 5);
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    #[test]
    fn test_safezone_stop_signals() {
        let sz = SafeZoneStop::new(5, 2.0);

        let data = OHLCVSeries {
            open: vec![100.0; 20],
            high: (0..20).map(|i| 105.0 + (i as f64 * 0.2)).collect(),
            low: (0..20).map(|i| 95.0 + (i as f64 * 0.2)).collect(),
            close: (0..20).map(|i| 100.0 + (i as f64 * 0.2)).collect(),
            volume: vec![1000.0; 20],
        };

        let signals = sz.signals(&data).unwrap();
        assert_eq!(signals.len(), 20);

        // First few signals should be Neutral (insufficient data)
        for i in 0..5 {
            assert!(matches!(signals[i], IndicatorSignal::Neutral));
        }
    }
}
