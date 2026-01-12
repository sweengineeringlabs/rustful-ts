//! Elder Triple Screen Trading System implementation.
//!
//! A multi-timeframe trading system that uses three "screens" for trade decisions.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use crate::{EMA, MACD, Stochastic};

/// Elder Triple Screen output.
#[derive(Debug, Clone)]
pub struct ElderTripleScreenOutput {
    /// First screen: Weekly trend (MACD histogram slope).
    pub trend_screen: Vec<i8>,
    /// Second screen: Daily oscillator (Stochastic or Force Index).
    pub oscillator_screen: Vec<i8>,
    /// Third screen: Entry signal.
    pub entry_signal: Vec<i8>,
    /// Combined signal strength (-3 to +3).
    pub signal_strength: Vec<i8>,
}

/// Elder Triple Screen configuration.
#[derive(Debug, Clone)]
pub struct ElderTripleScreenConfig {
    /// EMA period for trend detection (default: 13).
    pub ema_period: usize,
    /// MACD fast period (default: 12).
    pub macd_fast: usize,
    /// MACD slow period (default: 26).
    pub macd_slow: usize,
    /// MACD signal period (default: 9).
    pub macd_signal: usize,
    /// Stochastic K period (default: 14).
    pub stoch_k: usize,
    /// Stochastic D period (default: 3).
    pub stoch_d: usize,
    /// Overbought level (default: 80).
    pub overbought: f64,
    /// Oversold level (default: 20).
    pub oversold: f64,
}

impl Default for ElderTripleScreenConfig {
    fn default() -> Self {
        Self {
            ema_period: 13,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            stoch_k: 14,
            stoch_d: 3,
            overbought: 80.0,
            oversold: 20.0,
        }
    }
}

/// Elder Triple Screen Trading System.
///
/// The system uses three screens:
///
/// 1. **First Screen (Trend)**: Uses MACD histogram on higher timeframe
///    to determine overall trend direction.
///
/// 2. **Second Screen (Oscillator)**: Uses Stochastic on trading timeframe
///    to find pullbacks within the trend.
///
/// 3. **Third Screen (Entry)**: Uses price action for precise entry.
///
/// Trading rules:
/// - Only buy when weekly trend is UP and daily oscillator is oversold
/// - Only sell when weekly trend is DOWN and daily oscillator is overbought
#[derive(Debug, Clone)]
pub struct ElderTripleScreen {
    #[allow(dead_code)]
    ema: EMA,
    macd: MACD,
    stochastic: Stochastic,
    overbought: f64,
    oversold: f64,
}

impl ElderTripleScreen {
    pub fn new(config: ElderTripleScreenConfig) -> Self {
        Self {
            ema: EMA::new(config.ema_period),
            macd: MACD::new(config.macd_fast, config.macd_slow, config.macd_signal),
            stochastic: Stochastic::new(config.stoch_k, config.stoch_d),
            overbought: config.overbought,
            oversold: config.oversold,
        }
    }

    /// Calculate Elder Triple Screen values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> ElderTripleScreenOutput {
        let n = close.len();

        // First Screen: MACD histogram for trend
        let (_, _, macd_histogram) = self.macd.calculate(close);
        let mut trend_screen = vec![0i8; n];

        for i in 1..n {
            if macd_histogram[i].is_nan() || macd_histogram[i - 1].is_nan() {
                continue;
            }
            if macd_histogram[i] > macd_histogram[i - 1] {
                trend_screen[i] = 1; // Rising histogram = bullish trend
            } else if macd_histogram[i] < macd_histogram[i - 1] {
                trend_screen[i] = -1; // Falling histogram = bearish trend
            }
        }

        // Second Screen: Stochastic oscillator for pullbacks
        let (stoch_k, _) = self.stochastic.calculate(high, low, close);
        let mut oscillator_screen = vec![0i8; n];

        for i in 0..n {
            if stoch_k[i].is_nan() {
                continue;
            }
            if stoch_k[i] < self.oversold {
                oscillator_screen[i] = 1; // Oversold = bullish opportunity
            } else if stoch_k[i] > self.overbought {
                oscillator_screen[i] = -1; // Overbought = bearish opportunity
            }
        }

        // Third Screen: Entry signals (breakout of previous bar)
        let mut entry_signal = vec![0i8; n];

        for i in 1..n {
            // Buy signal: price breaks above previous high
            if close[i] > high[i - 1] {
                entry_signal[i] = 1;
            }
            // Sell signal: price breaks below previous low
            else if close[i] < low[i - 1] {
                entry_signal[i] = -1;
            }
        }

        // Combined signal strength
        let mut signal_strength = vec![0i8; n];

        for i in 0..n {
            let trend = trend_screen[i];
            let osc = oscillator_screen[i];
            let entry = entry_signal[i];

            // Strong buy: trend up + oversold + breakout up
            if trend == 1 && osc == 1 && entry == 1 {
                signal_strength[i] = 3;
            }
            // Moderate buy: trend up + oversold
            else if trend == 1 && osc == 1 {
                signal_strength[i] = 2;
            }
            // Weak buy: trend up + breakout up
            else if trend == 1 && entry == 1 {
                signal_strength[i] = 1;
            }
            // Strong sell: trend down + overbought + breakout down
            else if trend == -1 && osc == -1 && entry == -1 {
                signal_strength[i] = -3;
            }
            // Moderate sell: trend down + overbought
            else if trend == -1 && osc == -1 {
                signal_strength[i] = -2;
            }
            // Weak sell: trend down + breakout down
            else if trend == -1 && entry == -1 {
                signal_strength[i] = -1;
            }
        }

        ElderTripleScreenOutput {
            trend_screen,
            oscillator_screen,
            entry_signal,
            signal_strength,
        }
    }
}

impl Default for ElderTripleScreen {
    fn default() -> Self {
        Self::new(ElderTripleScreenConfig::default())
    }
}

impl TechnicalIndicator for ElderTripleScreen {
    fn name(&self) -> &str {
        "ElderTripleScreen"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = 35; // MACD slow + signal
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);

        // Output signal strength as primary
        let strength: Vec<f64> = result.signal_strength.iter()
            .map(|&s| s as f64)
            .collect();

        // Output trend as secondary
        let trend: Vec<f64> = result.trend_screen.iter()
            .map(|&t| t as f64)
            .collect();

        // Output oscillator as tertiary
        let osc: Vec<f64> = result.oscillator_screen.iter()
            .map(|&o| o as f64)
            .collect();

        Ok(IndicatorOutput::triple(strength, trend, osc))
    }

    fn min_periods(&self) -> usize {
        35
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for ElderTripleScreen {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let n = result.signal_strength.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let strength = result.signal_strength[n - 1];

        // Require at least moderate signal strength
        if strength >= 2 {
            Ok(IndicatorSignal::Bullish)
        } else if strength <= -2 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<_> = result.signal_strength.iter()
            .map(|&s| {
                if s >= 2 {
                    IndicatorSignal::Bullish
                } else if s <= -2 {
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

    fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        (high, low, close)
    }

    #[test]
    fn test_elder_triple_screen_basic() {
        let ets = ElderTripleScreen::default();
        let (high, low, close) = generate_test_data(50);

        let result = ets.calculate(&high, &low, &close);

        assert_eq!(result.trend_screen.len(), 50);
        assert_eq!(result.oscillator_screen.len(), 50);
        assert_eq!(result.entry_signal.len(), 50);
        assert_eq!(result.signal_strength.len(), 50);
    }

    #[test]
    fn test_elder_triple_screen_signals() {
        let ets = ElderTripleScreen::default();
        let (high, low, close) = generate_test_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let signals = ets.signals(&series).unwrap();
        assert_eq!(signals.len(), 50);
    }

    #[test]
    fn test_elder_triple_screen_strength_range() {
        let ets = ElderTripleScreen::default();
        let (high, low, close) = generate_test_data(50);

        let result = ets.calculate(&high, &low, &close);

        // Signal strength should be between -3 and +3
        for &s in &result.signal_strength {
            assert!(s >= -3 && s <= 3);
        }
    }

    #[test]
    fn test_elder_triple_screen_config() {
        let config = ElderTripleScreenConfig {
            ema_period: 10,
            macd_fast: 8,
            macd_slow: 17,
            macd_signal: 6,
            stoch_k: 10,
            stoch_d: 3,
            overbought: 70.0,
            oversold: 30.0,
        };

        let ets = ElderTripleScreen::new(config);
        assert_eq!(ets.name(), "ElderTripleScreen");
    }
}
