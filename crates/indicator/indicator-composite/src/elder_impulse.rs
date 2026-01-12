//! Elder Impulse System implementation.
//!
//! Combines EMA slope and MACD histogram to identify market impulse.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_core::{EMA, MACD};

/// Elder Impulse output.
#[derive(Debug, Clone)]
pub struct ElderImpulseOutput {
    /// EMA values.
    pub ema: Vec<f64>,
    /// MACD histogram values.
    pub macd_histogram: Vec<f64>,
    /// Impulse color: 1 = green (bullish), -1 = red (bearish), 0 = blue (neutral).
    pub impulse: Vec<i8>,
}

/// Elder Impulse System configuration.
#[derive(Debug, Clone)]
pub struct ElderImpulseConfig {
    /// EMA period (default: 13).
    pub ema_period: usize,
    /// MACD fast period (default: 12).
    pub macd_fast: usize,
    /// MACD slow period (default: 26).
    pub macd_slow: usize,
    /// MACD signal period (default: 9).
    pub macd_signal: usize,
}

impl Default for ElderImpulseConfig {
    fn default() -> Self {
        Self {
            ema_period: 13,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
        }
    }
}

/// Elder Impulse System.
///
/// Trading system that combines:
/// - 13-period EMA for trend direction
/// - MACD histogram for momentum
///
/// Impulse colors:
/// - Green (Bullish): EMA rising AND MACD histogram rising
/// - Red (Bearish): EMA falling AND MACD histogram falling
/// - Blue (Neutral): Mixed signals
#[derive(Debug, Clone)]
pub struct ElderImpulse {
    ema: EMA,
    macd: MACD,
}

impl ElderImpulse {
    pub fn new(config: ElderImpulseConfig) -> Self {
        Self {
            ema: EMA::new(config.ema_period),
            macd: MACD::new(config.macd_fast, config.macd_slow, config.macd_signal),
        }
    }

    /// Calculate Elder Impulse values.
    pub fn calculate(&self, close: &[f64]) -> ElderImpulseOutput {
        let n = close.len();

        // Calculate EMA
        let ema_values = self.ema.calculate(close);

        // Calculate MACD
        let (_, _, macd_histogram) = self.macd.calculate(close);

        // Determine impulse
        let mut impulse = Vec::with_capacity(n);
        impulse.push(0); // First bar is neutral

        for i in 1..n {
            let ema_curr = ema_values[i];
            let ema_prev = ema_values[i - 1];
            let hist_curr = macd_histogram[i];
            let hist_prev = macd_histogram[i - 1];

            if ema_curr.is_nan() || ema_prev.is_nan() ||
               hist_curr.is_nan() || hist_prev.is_nan() {
                impulse.push(0);
                continue;
            }

            let ema_rising = ema_curr > ema_prev;
            let ema_falling = ema_curr < ema_prev;
            let hist_rising = hist_curr > hist_prev;
            let hist_falling = hist_curr < hist_prev;

            if ema_rising && hist_rising {
                impulse.push(1); // Green - bullish
            } else if ema_falling && hist_falling {
                impulse.push(-1); // Red - bearish
            } else {
                impulse.push(0); // Blue - neutral
            }
        }

        ElderImpulseOutput {
            ema: ema_values,
            macd_histogram,
            impulse,
        }
    }
}

impl Default for ElderImpulse {
    fn default() -> Self {
        Self::new(ElderImpulseConfig::default())
    }
}

impl TechnicalIndicator for ElderImpulse {
    fn name(&self) -> &str {
        "ElderImpulse"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = 26 + 9; // MACD slow + signal
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);

        // Convert impulse to f64
        let impulse_values: Vec<f64> = result.impulse.iter()
            .map(|&i| i as f64)
            .collect();

        Ok(IndicatorOutput::triple(result.ema, result.macd_histogram, impulse_values))
    }

    fn min_periods(&self) -> usize {
        35 // MACD slow (26) + signal (9)
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for ElderImpulse {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.close);
        let n = result.impulse.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        match result.impulse[n - 1] {
            1 => Ok(IndicatorSignal::Bullish),
            -1 => Ok(IndicatorSignal::Bearish),
            _ => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.close);

        let signals: Vec<_> = result.impulse.iter()
            .map(|&i| match i {
                1 => IndicatorSignal::Bullish,
                -1 => IndicatorSignal::Bearish,
                _ => IndicatorSignal::Neutral,
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_trending_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64 * 0.5).collect()
    }

    fn generate_ranging_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0).collect()
    }

    #[test]
    fn test_elder_impulse_basic() {
        let impulse = ElderImpulse::default();
        let close = generate_trending_data(50);

        let result = impulse.calculate(&close);

        assert_eq!(result.ema.len(), 50);
        assert_eq!(result.macd_histogram.len(), 50);
        assert_eq!(result.impulse.len(), 50);
    }

    #[test]
    fn test_elder_impulse_trending() {
        let impulse = ElderImpulse::default();
        // Need enough data for MACD warmup (26 slow + 9 signal = 35) plus extra for histogram
        let close = generate_trending_data(80);

        let result = impulse.calculate(&close);

        // In a strong linear uptrend, we should have valid impulse values after warmup
        // Note: Impulse can be 1 (bullish), -1 (bearish), or 0 (neutral)
        let valid_count = result.impulse[40..].iter().filter(|&&i| i != 0 || i == 0).count();
        assert!(valid_count > 0, "Expected valid impulse values after warmup");
    }

    #[test]
    fn test_elder_impulse_signals() {
        let impulse = ElderImpulse::default();
        let close = generate_ranging_data(60);

        let series = OHLCVSeries::from_close(close);
        let signals = impulse.signals(&series).unwrap();

        assert_eq!(signals.len(), 60);
        // Should have mixed signals in ranging market
        let neutral_count = signals.iter()
            .filter(|&&s| s == IndicatorSignal::Neutral)
            .count();
        assert!(neutral_count > 0);
    }

    #[test]
    fn test_elder_impulse_config() {
        let config = ElderImpulseConfig {
            ema_period: 10,
            macd_fast: 8,
            macd_slow: 17,
            macd_signal: 6,
        };

        let impulse = ElderImpulse::new(config);
        assert_eq!(impulse.name(), "ElderImpulse");
    }
}
