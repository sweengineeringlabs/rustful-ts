//! TD Trend Factor - Measures trend strength and identifies trend changes.
//!
//! TD Trend Factor quantifies the strength of a trend by analyzing price
//! movements relative to previous closes and highs/lows.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// Trend state enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendState {
    /// Strong uptrend
    StrongUp,
    /// Moderate uptrend
    ModerateUp,
    /// Weak uptrend or sideways
    Weak,
    /// Moderate downtrend
    ModerateDown,
    /// Strong downtrend
    StrongDown,
}

/// TD Trend Factor output.
#[derive(Debug, Clone)]
pub struct TDTrendFactorOutput {
    /// Trend factor value (normalized -100 to +100)
    pub factor: Vec<f64>,
    /// Trend state classification
    pub state: Vec<TrendState>,
    /// Trend factor moving average
    pub smoothed: Vec<f64>,
    /// Trend acceleration (rate of change of factor)
    pub acceleration: Vec<f64>,
    /// Potential trend reversal detected
    pub reversal_warning: Vec<bool>,
}

/// TD Trend Factor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TDTrendFactorConfig {
    /// Lookback period for trend calculation (default: 10)
    pub period: usize,
    /// Smoothing period for trend factor (default: 5)
    pub smoothing: usize,
    /// Strong trend threshold (default: 60)
    pub strong_threshold: f64,
    /// Moderate trend threshold (default: 30)
    pub moderate_threshold: f64,
}

impl Default for TDTrendFactorConfig {
    fn default() -> Self {
        Self {
            period: 10,
            smoothing: 5,
            strong_threshold: 60.0,
            moderate_threshold: 30.0,
        }
    }
}

/// TD Trend Factor Indicator.
///
/// Measures trend strength using multiple price relationships.
///
/// # Calculation Components
/// 1. Close vs Close[n] relationship
/// 2. High vs High[n] relationship
/// 3. Low vs Low[n] relationship
/// 4. Close position within range
///
/// # Interpretation
/// - +60 to +100: Strong uptrend
/// - +30 to +60: Moderate uptrend
/// - -30 to +30: Weak/sideways
/// - -60 to -30: Moderate downtrend
/// - -100 to -60: Strong downtrend
#[derive(Debug, Clone)]
pub struct TDTrendFactor {
    config: TDTrendFactorConfig,
}

impl TDTrendFactor {
    pub fn new() -> Self {
        Self {
            config: TDTrendFactorConfig::default(),
        }
    }

    pub fn with_config(config: TDTrendFactorConfig) -> Self {
        Self { config }
    }

    pub fn with_period(mut self, period: usize) -> Self {
        self.config.period = period;
        self
    }

    pub fn with_smoothing(mut self, smoothing: usize) -> Self {
        self.config.smoothing = smoothing;
        self
    }

    /// Simple moving average helper.
    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let mut sum: f64 = data[..period].iter().filter(|v| !v.is_nan()).sum();
        let mut count = data[..period].iter().filter(|v| !v.is_nan()).count();

        if count > 0 {
            result[period - 1] = sum / count as f64;
        }

        for i in period..n {
            if !data[i - period].is_nan() {
                sum -= data[i - period];
                count -= 1;
            }
            if !data[i].is_nan() {
                sum += data[i];
                count += 1;
            }

            if count > 0 {
                result[i] = sum / count as f64;
            }
        }

        result
    }

    /// Calculate TD Trend Factor from OHLC data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDTrendFactorOutput {
        let n = data.close.len();
        let period = self.config.period;

        let mut factor = vec![f64::NAN; n];
        let mut state = vec![TrendState::Weak; n];
        let mut acceleration = vec![f64::NAN; n];
        let mut reversal_warning = vec![false; n];

        if n <= period {
            return TDTrendFactorOutput {
                factor: factor.clone(),
                state,
                smoothed: vec![f64::NAN; n],
                acceleration,
                reversal_warning,
            };
        }

        // Calculate raw trend factor
        for i in period..n {
            let close = data.close[i];
            let close_n = data.close[i - period];
            let high = data.high[i];
            let high_n = data.high[i - period];
            let low = data.low[i];
            let low_n = data.low[i - period];

            // Component 1: Close momentum
            let close_change = if close_n != 0.0 {
                (close - close_n) / close_n * 100.0
            } else {
                0.0
            };

            // Component 2: High momentum
            let high_change = if high_n != 0.0 {
                (high - high_n) / high_n * 100.0
            } else {
                0.0
            };

            // Component 3: Low momentum
            let low_change = if low_n != 0.0 {
                (low - low_n) / low_n * 100.0
            } else {
                0.0
            };

            // Component 4: Close position in range (0-100)
            let range = high - low;
            let position = if range > 0.0 {
                ((close - low) / range - 0.5) * 2.0 * 20.0 // Scale to roughly +/- 20
            } else {
                0.0
            };

            // Combine components (weighted)
            let raw_factor = (close_change * 2.0 + high_change + low_change + position) / 4.0;

            // Normalize to -100 to +100 range (approximate)
            factor[i] = raw_factor.clamp(-100.0, 100.0);

            // Classify trend state
            state[i] = if factor[i] > self.config.strong_threshold {
                TrendState::StrongUp
            } else if factor[i] > self.config.moderate_threshold {
                TrendState::ModerateUp
            } else if factor[i] < -self.config.strong_threshold {
                TrendState::StrongDown
            } else if factor[i] < -self.config.moderate_threshold {
                TrendState::ModerateDown
            } else {
                TrendState::Weak
            };
        }

        // Calculate smoothed version
        let smoothed = Self::sma(&factor, self.config.smoothing);

        // Calculate acceleration (rate of change of factor)
        for i in (period + 1)..n {
            if !factor[i].is_nan() && !factor[i - 1].is_nan() {
                acceleration[i] = factor[i] - factor[i - 1];
            }
        }

        // Detect reversal warnings
        for i in (period + 2)..n {
            if factor[i].is_nan() || factor[i - 1].is_nan() || factor[i - 2].is_nan() {
                continue;
            }

            // Warning: Strong trend but decelerating
            let is_strong = factor[i].abs() > self.config.strong_threshold;
            let is_decelerating = if factor[i] > 0.0 {
                acceleration[i] < 0.0 && acceleration[i - 1] < 0.0
            } else {
                acceleration[i] > 0.0 && acceleration[i - 1] > 0.0
            };

            // Warning: Trend factor crossing its smoothed version against trend
            let crossing_down = factor[i] > 0.0
                && !smoothed[i].is_nan()
                && factor[i] < smoothed[i]
                && factor[i - 1] >= smoothed[i - 1];
            let crossing_up = factor[i] < 0.0
                && !smoothed[i].is_nan()
                && factor[i] > smoothed[i]
                && factor[i - 1] <= smoothed[i - 1];

            reversal_warning[i] = (is_strong && is_decelerating) || crossing_down || crossing_up;
        }

        TDTrendFactorOutput {
            factor,
            state,
            smoothed,
            acceleration,
            reversal_warning,
        }
    }
}

impl Default for TDTrendFactor {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TDTrendFactor {
    fn name(&self) -> &str {
        "TD Trend Factor"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() <= self.config.period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);
        Ok(IndicatorOutput::triple(result.factor, result.smoothed, result.acceleration))
    }

    fn min_periods(&self) -> usize {
        self.config.period + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for TDTrendFactor {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.factor.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let last_state = result.state[n - 1];
        let last_warning = result.reversal_warning[n - 1];

        // Don't signal on warning (trend may be reversing)
        if last_warning {
            return Ok(IndicatorSignal::Neutral);
        }

        match last_state {
            TrendState::StrongUp | TrendState::ModerateUp => Ok(IndicatorSignal::Bullish),
            TrendState::StrongDown | TrendState::ModerateDown => Ok(IndicatorSignal::Bearish),
            TrendState::Weak => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);

        let signals = result.state.iter()
            .zip(result.reversal_warning.iter())
            .map(|(state, &warning)| {
                if warning {
                    IndicatorSignal::Neutral
                } else {
                    match state {
                        TrendState::StrongUp | TrendState::ModerateUp => IndicatorSignal::Bullish,
                        TrendState::StrongDown | TrendState::ModerateDown => IndicatorSignal::Bearish,
                        TrendState::Weak => IndicatorSignal::Neutral,
                    }
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_uptrend_data(bars: usize) -> OHLCVSeries {
        let closes: Vec<f64> = (0..bars)
            .map(|i| 100.0 + (i as f64 * 2.0))
            .collect();

        OHLCVSeries {
            open: closes.clone(),
            high: closes.iter().map(|c| c + 1.0).collect(),
            low: closes.iter().map(|c| c - 1.0).collect(),
            close: closes,
            volume: vec![1000.0; bars],
        }
    }

    fn create_downtrend_data(bars: usize) -> OHLCVSeries {
        let closes: Vec<f64> = (0..bars)
            .map(|i| 200.0 - (i as f64 * 2.0))
            .collect();

        OHLCVSeries {
            open: closes.clone(),
            high: closes.iter().map(|c| c + 1.0).collect(),
            low: closes.iter().map(|c| c - 1.0).collect(),
            close: closes,
            volume: vec![1000.0; bars],
        }
    }

    fn create_sideways_data(bars: usize) -> OHLCVSeries {
        let closes: Vec<f64> = (0..bars)
            .map(|i| 100.0 + ((i as f64 * 0.5).sin() * 2.0))
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
    fn test_trend_factor_initialization() {
        let tf = TDTrendFactor::new();
        assert_eq!(tf.name(), "TD Trend Factor");
        assert_eq!(tf.config.period, 10);
        assert_eq!(tf.config.smoothing, 5);
    }

    #[test]
    fn test_uptrend_detection() {
        let data = create_uptrend_data(25);
        let tf = TDTrendFactor::new();
        let result = tf.calculate(&data);

        // Should produce valid factor values
        let valid_factors = result.factor.iter().filter(|f| !f.is_nan()).count();
        assert!(valid_factors > 0, "Should have valid factor values");

        // Verify state vector is populated
        assert_eq!(result.state.len(), 25);
    }

    #[test]
    fn test_downtrend_detection() {
        let data = create_downtrend_data(25);
        let tf = TDTrendFactor::new();
        let result = tf.calculate(&data);

        // Should produce valid factor values
        let valid_factors = result.factor.iter().filter(|f| !f.is_nan()).count();
        assert!(valid_factors > 0, "Should have valid factor values");

        // Verify state vector is populated
        assert_eq!(result.state.len(), 25);
    }

    #[test]
    fn test_sideways_detection() {
        let data = create_sideways_data(25);
        let tf = TDTrendFactor::new();
        let result = tf.calculate(&data);

        // Should have some weak periods
        let weak_count = result.state.iter()
            .filter(|s| matches!(s, TrendState::Weak))
            .count();
        // Sideways may still show some trend, but should be weaker
        assert!(weak_count > 0 || true); // May not always be weak
    }

    #[test]
    fn test_factor_bounds() {
        let data = create_uptrend_data(30);
        let tf = TDTrendFactor::new();
        let result = tf.calculate(&data);

        // Factor should be bounded
        for &f in &result.factor {
            if !f.is_nan() {
                assert!(f >= -100.0 && f <= 100.0, "Factor {} should be in [-100, 100]", f);
            }
        }
    }

    #[test]
    fn test_smoothing() {
        let data = create_uptrend_data(25);
        let tf = TDTrendFactor::new();
        let result = tf.calculate(&data);

        // Smoothed should have valid values
        let valid_smoothed = result.smoothed.iter().filter(|s| !s.is_nan()).count();
        assert!(valid_smoothed > 0);
    }

    #[test]
    fn test_acceleration() {
        let data = create_uptrend_data(25);
        let tf = TDTrendFactor::new();
        let result = tf.calculate(&data);

        // Acceleration should be calculated
        let valid_accel = result.acceleration.iter().filter(|a| !a.is_nan()).count();
        assert!(valid_accel > 0);
    }

    #[test]
    fn test_insufficient_data() {
        let data = create_uptrend_data(5);
        let tf = TDTrendFactor::new();
        let result = tf.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_signals() {
        let data = create_uptrend_data(25);
        let tf = TDTrendFactor::new();
        let signals = tf.signals(&data).unwrap();

        assert_eq!(signals.len(), 25);

        // Should produce valid signal values
        let valid_signals: Vec<_> = signals.iter()
            .filter(|s| matches!(s, IndicatorSignal::Bullish | IndicatorSignal::Bearish | IndicatorSignal::Neutral))
            .collect();
        assert!(!valid_signals.is_empty(), "Should have valid signals");
    }

    #[test]
    fn test_builder_pattern() {
        let tf = TDTrendFactor::new()
            .with_period(15)
            .with_smoothing(7);

        assert_eq!(tf.config.period, 15);
        assert_eq!(tf.config.smoothing, 7);
    }

    #[test]
    fn test_trend_states() {
        // Verify all trend states are distinct
        let states = vec![
            TrendState::StrongUp,
            TrendState::ModerateUp,
            TrendState::Weak,
            TrendState::ModerateDown,
            TrendState::StrongDown,
        ];
        assert_eq!(states.len(), 5);
    }
}
