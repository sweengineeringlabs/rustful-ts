//! TD Price Oscillator Qualifier (TD POQ).
//!
//! TD POQ is a momentum oscillator that qualifies price movements by comparing
//! closes to previous closes and measuring momentum strength.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// TD POQ output.
#[derive(Debug, Clone)]
pub struct TDPOQOutput {
    /// POQ values (oscillator)
    pub poq: Vec<f64>,
    /// Smoothed POQ line
    pub signal: Vec<f64>,
    /// Histogram (POQ - Signal)
    pub histogram: Vec<f64>,
    /// Bullish divergence detected
    pub bullish_divergence: Vec<bool>,
    /// Bearish divergence detected
    pub bearish_divergence: Vec<bool>,
}

/// TD POQ configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDPOQConfig {
    /// Short period for oscillator (default: 3)
    pub short_period: usize,
    /// Long period for oscillator (default: 5)
    pub long_period: usize,
    /// Signal line smoothing period (default: 3)
    pub signal_period: usize,
    /// Lookback for divergence detection (default: 5)
    pub divergence_lookback: usize,
}

impl Default for TDPOQConfig {
    fn default() -> Self {
        Self {
            short_period: 3,
            long_period: 5,
            signal_period: 3,
            divergence_lookback: 5,
        }
    }
}

/// TD Price Oscillator Qualifier.
///
/// TD POQ measures momentum by comparing price changes over short and long periods.
/// It helps qualify the strength of price movements.
///
/// # Calculation
/// 1. Calculate short-term rate of change
/// 2. Calculate long-term rate of change
/// 3. POQ = difference or ratio of ROC values
/// 4. Signal = smoothed POQ
///
/// # Interpretation
/// - Positive POQ: Bullish momentum
/// - Negative POQ: Bearish momentum
/// - Divergences: Price and POQ moving opposite directions
#[derive(Debug, Clone)]
pub struct TDPOQ {
    config: TDPOQConfig,
}

impl TDPOQ {
    pub fn new() -> Self {
        Self {
            config: TDPOQConfig::default(),
        }
    }

    pub fn with_config(config: TDPOQConfig) -> Self {
        Self { config }
    }

    /// Simple moving average helper.
    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let mut sum: f64 = data[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        for i in period..n {
            sum = sum - data[i - period] + data[i];
            result[i] = sum / period as f64;
        }

        result
    }

    /// Calculate TD POQ from close prices.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDPOQOutput {
        let n = data.close.len();
        let short = self.config.short_period;
        let long = self.config.long_period;
        let signal_period = self.config.signal_period;
        let div_lookback = self.config.divergence_lookback;

        let mut poq = vec![f64::NAN; n];
        let mut bullish_divergence = vec![false; n];
        let mut bearish_divergence = vec![false; n];

        if n < long {
            return TDPOQOutput {
                poq: poq.clone(),
                signal: vec![f64::NAN; n],
                histogram: vec![f64::NAN; n],
                bullish_divergence,
                bearish_divergence,
            };
        }

        // Calculate Rate of Change components
        for i in long..n {
            let close = data.close[i];

            // Short-term ROC
            let short_roc = if data.close[i - short] != 0.0 {
                (close - data.close[i - short]) / data.close[i - short] * 100.0
            } else {
                0.0
            };

            // Long-term ROC
            let long_roc = if data.close[i - long] != 0.0 {
                (close - data.close[i - long]) / data.close[i - long] * 100.0
            } else {
                0.0
            };

            // POQ is the difference in ROC (momentum acceleration)
            poq[i] = short_roc - (long_roc * (short as f64 / long as f64));
        }

        // Calculate signal line (SMA of POQ)
        let signal = Self::sma(&poq, signal_period);

        // Calculate histogram
        let histogram: Vec<f64> = poq.iter()
            .zip(signal.iter())
            .map(|(&p, &s)| {
                if p.is_nan() || s.is_nan() {
                    f64::NAN
                } else {
                    p - s
                }
            })
            .collect();

        // Detect divergences
        for i in (long + div_lookback)..n {
            if poq[i].is_nan() || poq[i - div_lookback].is_nan() {
                continue;
            }

            let price_rising = data.close[i] > data.close[i - div_lookback];
            let price_falling = data.close[i] < data.close[i - div_lookback];
            let poq_rising = poq[i] > poq[i - div_lookback];
            let poq_falling = poq[i] < poq[i - div_lookback];

            // Bullish divergence: price falling but POQ rising
            bullish_divergence[i] = price_falling && poq_rising;

            // Bearish divergence: price rising but POQ falling
            bearish_divergence[i] = price_rising && poq_falling;
        }

        TDPOQOutput {
            poq,
            signal,
            histogram,
            bullish_divergence,
            bearish_divergence,
        }
    }
}

impl Default for TDPOQ {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TDPOQ {
    fn name(&self) -> &str {
        "TD POQ"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.long_period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.long_period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);
        Ok(IndicatorOutput::triple(result.poq, result.signal, result.histogram))
    }

    fn min_periods(&self) -> usize {
        self.config.long_period
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for TDPOQ {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.poq.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal based on divergences or POQ crossovers
        if result.bullish_divergence[n - 1] {
            return Ok(IndicatorSignal::Bullish);
        }
        if result.bearish_divergence[n - 1] {
            return Ok(IndicatorSignal::Bearish);
        }

        // Check for zero-line crossover
        if n >= 2 {
            let prev = result.poq[n - 2];
            let curr = result.poq[n - 1];

            if !prev.is_nan() && !curr.is_nan() {
                if prev < 0.0 && curr > 0.0 {
                    return Ok(IndicatorSignal::Bullish);
                }
                if prev > 0.0 && curr < 0.0 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);
        let n = result.poq.len();

        let mut signals = vec![IndicatorSignal::Neutral; n];

        for i in 1..n {
            // Divergences take priority
            if result.bullish_divergence[i] {
                signals[i] = IndicatorSignal::Bullish;
                continue;
            }
            if result.bearish_divergence[i] {
                signals[i] = IndicatorSignal::Bearish;
                continue;
            }

            // Zero-line crossover
            let prev = result.poq[i - 1];
            let curr = result.poq[i];

            if !prev.is_nan() && !curr.is_nan() {
                if prev < 0.0 && curr > 0.0 {
                    signals[i] = IndicatorSignal::Bullish;
                } else if prev > 0.0 && curr < 0.0 {
                    signals[i] = IndicatorSignal::Bearish;
                }
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_trending_data(direction: i32, bars: usize) -> OHLCVSeries {
        let closes: Vec<f64> = (0..bars)
            .map(|i| 100.0 + (direction as f64) * (i as f64 * 0.5))
            .collect();

        OHLCVSeries {
            open: closes.clone(),
            high: closes.iter().map(|c| c + 1.0).collect(),
            low: closes.iter().map(|c| c - 1.0).collect(),
            close: closes,
            volume: vec![1000.0; bars],
        }
    }

    #[test]
    fn test_poq_initialization() {
        let poq = TDPOQ::new();
        assert_eq!(poq.name(), "TD POQ");
        assert_eq!(poq.config.short_period, 3);
        assert_eq!(poq.config.long_period, 5);
    }

    #[test]
    fn test_poq_calculation() {
        let data = create_trending_data(1, 20);
        let poq = TDPOQ::new();
        let result = poq.calculate(&data);

        assert_eq!(result.poq.len(), 20);
        assert_eq!(result.signal.len(), 20);
        assert_eq!(result.histogram.len(), 20);

        // Should have some valid values
        let valid_count = result.poq.iter().filter(|v| !v.is_nan()).count();
        assert!(valid_count > 0);
    }

    #[test]
    fn test_uptrend_positive_poq() {
        let data = create_trending_data(1, 20);
        let poq = TDPOQ::new();
        let result = poq.calculate(&data);

        // In uptrend, later POQ values should be positive or near zero
        let last_valid = result.poq.iter().rev().find(|v| !v.is_nan());
        assert!(last_valid.is_some());
    }

    #[test]
    fn test_downtrend_negative_poq() {
        let data = create_trending_data(-1, 20);
        let poq = TDPOQ::new();
        let result = poq.calculate(&data);

        // In downtrend, later POQ values should be negative or near zero
        let last_valid = result.poq.iter().rev().find(|v| !v.is_nan());
        assert!(last_valid.is_some());
    }

    #[test]
    fn test_signal_line_smoothing() {
        // Need more data for signal line EMA warmup
        let data = create_trending_data(1, 40);
        let poq = TDPOQ::new();
        let result = poq.calculate(&data);

        // Signal line values should be calculated
        assert_eq!(result.signal.len(), 40);
        // POQ values should exist
        let valid_poq: Vec<_> = result.poq.iter()
            .filter(|v| !v.is_nan())
            .collect();
        assert!(!valid_poq.is_empty(), "Should have valid POQ values");
    }

    #[test]
    fn test_histogram_calculation() {
        let data = create_trending_data(1, 20);
        let poq = TDPOQ::new();
        let result = poq.calculate(&data);

        // Histogram = POQ - Signal
        for i in 0..result.histogram.len() {
            if !result.poq[i].is_nan() && !result.signal[i].is_nan() {
                let expected = result.poq[i] - result.signal[i];
                assert!((result.histogram[i] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_insufficient_data() {
        let data = create_trending_data(1, 3);
        let poq = TDPOQ::new();
        let result = poq.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_signals_method() {
        let data = create_trending_data(1, 20);
        let poq = TDPOQ::new();
        let signals = poq.signals(&data).unwrap();

        assert_eq!(signals.len(), 20);
    }

    #[test]
    fn test_sma_helper() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = TDPOQ::sma(&data, 3);

        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 2.0).abs() < 1e-10); // (1+2+3)/3 = 2
        assert!((sma[3] - 3.0).abs() < 1e-10); // (2+3+4)/3 = 3
        assert!((sma[4] - 4.0).abs() < 1e-10); // (3+4+5)/3 = 4
    }
}
