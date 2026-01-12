//! TD Range Expansion Index (TD REI).
//!
//! TD REI measures the strength of price movement by comparing today's high-low
//! range to recent ranges, with conditions that must be met for the value to count.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// TD REI output.
#[derive(Debug, Clone)]
pub struct TDREIOutput {
    /// REI values (-100 to +100)
    pub rei: Vec<f64>,
    /// Whether conditions are met for valid REI
    pub valid: Vec<bool>,
    /// Overbought signals (REI > upper threshold)
    pub overbought: Vec<bool>,
    /// Oversold signals (REI < lower threshold)
    pub oversold: Vec<bool>,
}

/// TD REI configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDREIConfig {
    /// Lookback period for calculation (default: 5)
    pub period: usize,
    /// Upper threshold for overbought (default: 45)
    pub overbought: f64,
    /// Lower threshold for oversold (default: -45)
    pub oversold: f64,
}

impl Default for TDREIConfig {
    fn default() -> Self {
        Self {
            period: 5,
            overbought: 45.0,
            oversold: -45.0,
        }
    }
}

/// TD Range Expansion Index.
///
/// TD REI identifies overbought/oversold conditions by measuring price expansion.
/// The indicator only provides valid readings when specific conditions are met.
///
/// # Calculation
/// 1. Check if today's high >= low[5 or 6] OR low <= high[5 or 6]
/// 2. Check if yesterday's high >= low[6 or 7] OR low <= high[6 or 7]
/// 3. If conditions met, calculate numerator and denominator
/// 4. REI = 100 * sum(numerator) / sum(abs(denominator))
///
/// # Interpretation
/// - REI > 45: Overbought
/// - REI < -45: Oversold
/// - Zero line crossovers indicate momentum shifts
#[derive(Debug, Clone)]
pub struct TDREI {
    config: TDREIConfig,
}

impl TDREI {
    pub fn new() -> Self {
        Self {
            config: TDREIConfig::default(),
        }
    }

    pub fn with_config(config: TDREIConfig) -> Self {
        Self { config }
    }

    pub fn with_period(mut self, period: usize) -> Self {
        self.config.period = period;
        self
    }

    pub fn with_thresholds(mut self, overbought: f64, oversold: f64) -> Self {
        self.config.overbought = overbought;
        self.config.oversold = oversold;
        self
    }

    /// Calculate TD REI from OHLC data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDREIOutput {
        let n = data.close.len();
        let period = self.config.period;
        let min_bars = period + 7; // Need lookback of 5-7 bars

        let mut rei = vec![f64::NAN; n];
        let mut valid = vec![false; n];
        let mut overbought = vec![false; n];
        let mut oversold = vec![false; n];

        if n < min_bars {
            return TDREIOutput {
                rei,
                valid,
                overbought,
                oversold,
            };
        }

        for i in min_bars..n {
            // Check condition 1: Today's range vs 5-6 bars ago
            let high_i = data.high[i];
            let low_i = data.low[i];
            let cond1 = (high_i >= data.low[i - 5] || high_i >= data.low[i - 6])
                || (low_i <= data.high[i - 5] || low_i <= data.high[i - 6]);

            // Check condition 2: Yesterday's range vs 6-7 bars ago
            let high_i1 = data.high[i - 1];
            let low_i1 = data.low[i - 1];
            let cond2 = (high_i1 >= data.low[i - 6] || high_i1 >= data.low[i - 7])
                || (low_i1 <= data.high[i - 6] || low_i1 <= data.high[i - 7]);

            valid[i] = cond1 && cond2;

            if valid[i] {
                // Calculate REI over the period
                let mut numerator_sum = 0.0;
                let mut denominator_sum = 0.0;

                for j in 0..period {
                    let idx = i - j;
                    if idx < 2 {
                        continue;
                    }

                    // Calculate high move and low move
                    let high_move = data.high[idx] - data.high[idx - 2];
                    let low_move = data.low[idx] - data.low[idx - 2];

                    // Numerator: directional move
                    numerator_sum += high_move + low_move;

                    // Denominator: absolute move
                    denominator_sum += high_move.abs() + low_move.abs();
                }

                if denominator_sum > 0.0 {
                    rei[i] = 100.0 * numerator_sum / denominator_sum;
                    overbought[i] = rei[i] > self.config.overbought;
                    oversold[i] = rei[i] < self.config.oversold;
                }
            }
        }

        TDREIOutput {
            rei,
            valid,
            overbought,
            oversold,
        }
    }
}

impl Default for TDREI {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TDREI {
    fn name(&self) -> &str {
        "TD REI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_bars = self.config.period + 7;
        if data.close.len() < min_bars {
            return Err(IndicatorError::InsufficientData {
                required: min_bars,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);
        let valid_f64: Vec<f64> = result.valid.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();

        Ok(IndicatorOutput::dual(result.rei, valid_f64))
    }

    fn min_periods(&self) -> usize {
        self.config.period + 7
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for TDREI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.rei.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal on overbought/oversold with valid reading
        if result.valid[n - 1] {
            if result.oversold[n - 1] {
                Ok(IndicatorSignal::Bullish) // Oversold = potential buy
            } else if result.overbought[n - 1] {
                Ok(IndicatorSignal::Bearish) // Overbought = potential sell
            } else {
                Ok(IndicatorSignal::Neutral)
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);

        let signals = result.valid.iter()
            .zip(result.oversold.iter())
            .zip(result.overbought.iter())
            .map(|((&v, &os), &ob)| {
                if v {
                    if os {
                        IndicatorSignal::Bullish
                    } else if ob {
                        IndicatorSignal::Bearish
                    } else {
                        IndicatorSignal::Neutral
                    }
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

    fn create_test_ohlc(bars: usize) -> OHLCVSeries {
        let mut high = Vec::with_capacity(bars);
        let mut low = Vec::with_capacity(bars);
        let mut close = Vec::with_capacity(bars);

        for i in 0..bars {
            let base = 100.0 + (i as f64 * 0.5);
            high.push(base + 2.0);
            low.push(base - 2.0);
            close.push(base);
        }

        OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; bars],
        }
    }

    #[test]
    fn test_rei_initialization() {
        let rei = TDREI::new();
        assert_eq!(rei.name(), "TD REI");
        assert_eq!(rei.config.period, 5);
        assert_eq!(rei.config.overbought, 45.0);
        assert_eq!(rei.config.oversold, -45.0);
    }

    #[test]
    fn test_rei_calculation() {
        let data = create_test_ohlc(20);
        let rei = TDREI::new();
        let result = rei.calculate(&data);

        assert_eq!(result.rei.len(), 20);
        assert_eq!(result.valid.len(), 20);

        // Some values should be valid
        let valid_count = result.valid.iter().filter(|&&v| v).count();
        // May have valid readings in trending data
        println!("Valid count: {}", valid_count);
    }

    #[test]
    fn test_rei_range() {
        let data = create_test_ohlc(25);
        let rei = TDREI::new();
        let result = rei.calculate(&data);

        // REI should be in [-100, 100] when valid
        for (i, &val) in result.rei.iter().enumerate() {
            if result.valid[i] && !val.is_nan() {
                assert!(val >= -100.0 && val <= 100.0,
                    "REI at {} is {}, should be in [-100, 100]", i, val);
            }
        }
    }

    #[test]
    fn test_overbought_oversold() {
        let config = TDREIConfig {
            period: 5,
            overbought: 30.0,  // Lower threshold for testing
            oversold: -30.0,
        };
        let rei = TDREI::with_config(config);
        let data = create_test_ohlc(25);
        let result = rei.calculate(&data);

        // Verify overbought/oversold flags are set correctly
        for i in 0..result.rei.len() {
            if result.valid[i] && !result.rei[i].is_nan() {
                if result.overbought[i] {
                    assert!(result.rei[i] > 30.0);
                }
                if result.oversold[i] {
                    assert!(result.rei[i] < -30.0);
                }
            }
        }
    }

    #[test]
    fn test_insufficient_data() {
        let data = create_test_ohlc(5);
        let rei = TDREI::new();
        let result = rei.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_pattern() {
        let rei = TDREI::new()
            .with_period(7)
            .with_thresholds(50.0, -50.0);

        assert_eq!(rei.config.period, 7);
        assert_eq!(rei.config.overbought, 50.0);
        assert_eq!(rei.config.oversold, -50.0);
    }

    #[test]
    fn test_signal_generation() {
        let data = create_test_ohlc(25);
        let rei = TDREI::new();
        let signals = rei.signals(&data).unwrap();

        assert_eq!(signals.len(), 25);
    }
}
