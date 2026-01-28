//! George Lane Stochastic Oscillator Variants
//!
//! Comprehensive implementation of George Lane's stochastic oscillator family:
//! - Fast Stochastic (%K, %D)
//! - Slow Stochastic (%K, %D)
//! - Full Stochastic (configurable smoothing)
//! - Lane's Original Stochastic
//! - Stochastic Pop/Drop patterns
//! - Stochastic Divergence
//! - Stochastic Crossover signals

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use crate::SMA;

/// Fast Stochastic %K - Raw stochastic calculation.
///
/// %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
///
/// This is the most responsive stochastic variant, directly comparing
/// the current close to the recent trading range.
#[derive(Debug, Clone)]
pub struct FastStochasticK {
    period: usize,
}

impl FastStochasticK {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate raw %K values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut k_line = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window_high = &high[start..=i];
            let window_low = &low[start..=i];

            let highest = window_high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = window_low.iter().cloned().fold(f64::INFINITY, f64::min);

            let range = highest - lowest;
            if range.abs() < 1e-10 {
                k_line.push(50.0);
            } else {
                let k = ((close[i] - lowest) / range) * 100.0;
                k_line.push(k);
            }
        }

        k_line
    }
}

impl TechnicalIndicator for FastStochasticK {
    fn name(&self) -> &str {
        "Fast Stochastic %K"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let k_line = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(k_line))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for FastStochasticK {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let k_line = self.calculate(&data.high, &data.low, &data.close);

        if k_line.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let k = *k_line.last().unwrap();
        if k.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if k < 20.0 {
            Ok(IndicatorSignal::Bullish) // Oversold
        } else if k > 80.0 {
            Ok(IndicatorSignal::Bearish) // Overbought
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let k_line = self.calculate(&data.high, &data.low, &data.close);

        Ok(k_line.iter().map(|&k| {
            if k.is_nan() {
                IndicatorSignal::Neutral
            } else if k < 20.0 {
                IndicatorSignal::Bullish
            } else if k > 80.0 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

/// Fast Stochastic %D - SMA of Fast %K.
///
/// %D = SMA(%K, smoothing_period)
///
/// Smooths the raw %K with a simple moving average.
#[derive(Debug, Clone)]
pub struct FastStochasticD {
    k_period: usize,
    d_period: usize,
}

impl FastStochasticD {
    pub fn new(k_period: usize, d_period: usize) -> Self {
        Self { k_period, d_period }
    }

    /// Calculate Fast %K and %D.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let fast_k = FastStochasticK::new(self.k_period);
        let k_line = fast_k.calculate(high, low, close);

        let sma = SMA::new(self.d_period);
        let d_line = sma.calculate(&k_line);

        (k_line, d_line)
    }
}

impl TechnicalIndicator for FastStochasticD {
    fn name(&self) -> &str {
        "Fast Stochastic %D"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.k_period + self.d_period - 1;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(k_line, d_line))
    }

    fn min_periods(&self) -> usize {
        self.k_period + self.d_period - 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for FastStochasticD {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);

        if k_line.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = k_line.len();
        let k = k_line[n - 1];
        let d = d_line[n - 1];
        let prev_k = k_line[n - 2];
        let prev_d = d_line[n - 2];

        if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish crossover in oversold zone
        if prev_k <= prev_d && k > d && k < 30.0 {
            Ok(IndicatorSignal::Bullish)
        }
        // Bearish crossover in overbought zone
        else if prev_k >= prev_d && k < d && k > 70.0 {
            Ok(IndicatorSignal::Bearish)
        }
        else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);
        let n = k_line.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let k = k_line[i];
            let d = d_line[i];
            let prev_k = k_line[i - 1];
            let prev_d = d_line[i - 1];

            if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if prev_k <= prev_d && k > d && k < 30.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if prev_k >= prev_d && k < d && k > 70.0 {
                signals.push(IndicatorSignal::Bearish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

/// Slow Stochastic %K - Smoothed Fast %K (equals Fast %D).
///
/// Slow %K = SMA(Fast %K, smoothing_period)
///
/// The Slow %K is actually the same as Fast %D. It provides a smoother
/// version of the raw stochastic.
#[derive(Debug, Clone)]
pub struct SlowStochasticK {
    k_period: usize,
    smoothing_period: usize,
}

impl SlowStochasticK {
    pub fn new(k_period: usize, smoothing_period: usize) -> Self {
        Self { k_period, smoothing_period }
    }

    /// Calculate Slow %K.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let fast_k = FastStochasticK::new(self.k_period);
        let k_line = fast_k.calculate(high, low, close);

        let sma = SMA::new(self.smoothing_period);
        sma.calculate(&k_line)
    }
}

impl TechnicalIndicator for SlowStochasticK {
    fn name(&self) -> &str {
        "Slow Stochastic %K"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.k_period + self.smoothing_period - 1;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let slow_k = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(slow_k))
    }

    fn min_periods(&self) -> usize {
        self.k_period + self.smoothing_period - 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for SlowStochasticK {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let slow_k = self.calculate(&data.high, &data.low, &data.close);

        if slow_k.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let k = *slow_k.last().unwrap();
        if k.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if k < 20.0 {
            Ok(IndicatorSignal::Bullish)
        } else if k > 80.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let slow_k = self.calculate(&data.high, &data.low, &data.close);

        Ok(slow_k.iter().map(|&k| {
            if k.is_nan() {
                IndicatorSignal::Neutral
            } else if k < 20.0 {
                IndicatorSignal::Bullish
            } else if k > 80.0 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

/// Slow Stochastic %D - SMA of Slow %K.
///
/// Slow %D = SMA(Slow %K, d_period)
///
/// This is the slowest/smoothest stochastic variant, providing
/// the most filtered signals.
#[derive(Debug, Clone)]
pub struct SlowStochasticD {
    k_period: usize,
    k_smoothing: usize,
    d_period: usize,
}

impl SlowStochasticD {
    pub fn new(k_period: usize, k_smoothing: usize, d_period: usize) -> Self {
        Self { k_period, k_smoothing, d_period }
    }

    /// Calculate Slow %K and Slow %D.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let slow_k_calc = SlowStochasticK::new(self.k_period, self.k_smoothing);
        let slow_k = slow_k_calc.calculate(high, low, close);

        let sma = SMA::new(self.d_period);
        let slow_d = sma.calculate(&slow_k);

        (slow_k, slow_d)
    }
}

impl TechnicalIndicator for SlowStochasticD {
    fn name(&self) -> &str {
        "Slow Stochastic %D"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.k_period + self.k_smoothing + self.d_period - 2;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (slow_k, slow_d) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(slow_k, slow_d))
    }

    fn min_periods(&self) -> usize {
        self.k_period + self.k_smoothing + self.d_period - 2
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for SlowStochasticD {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (slow_k, slow_d) = self.calculate(&data.high, &data.low, &data.close);

        if slow_k.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = slow_k.len();
        let k = slow_k[n - 1];
        let d = slow_d[n - 1];
        let prev_k = slow_k[n - 2];
        let prev_d = slow_d[n - 2];

        if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if prev_k <= prev_d && k > d && k < 30.0 {
            Ok(IndicatorSignal::Bullish)
        } else if prev_k >= prev_d && k < d && k > 70.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (slow_k, slow_d) = self.calculate(&data.high, &data.low, &data.close);
        let n = slow_k.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let k = slow_k[i];
            let d = slow_d[i];
            let prev_k = slow_k[i - 1];
            let prev_d = slow_d[i - 1];

            if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if prev_k <= prev_d && k > d && k < 30.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if prev_k >= prev_d && k < d && k > 70.0 {
                signals.push(IndicatorSignal::Bearish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

/// Full Stochastic - Fully configurable stochastic oscillator.
///
/// Allows independent configuration of:
/// - %K lookback period
/// - %K smoothing period
/// - %D smoothing period
///
/// This is the most flexible stochastic variant.
#[derive(Debug, Clone)]
pub struct FullStochastic {
    k_period: usize,
    k_smoothing: usize,
    d_smoothing: usize,
    overbought: f64,
    oversold: f64,
}

/// Output for Full Stochastic with all components.
#[derive(Debug, Clone)]
pub struct FullStochasticOutput {
    /// Fast %K (raw stochastic)
    pub fast_k: Vec<f64>,
    /// Slow %K (smoothed %K)
    pub slow_k: Vec<f64>,
    /// Slow %D (smoothed slow %K)
    pub slow_d: Vec<f64>,
}

impl FullStochastic {
    pub fn new(k_period: usize, k_smoothing: usize, d_smoothing: usize) -> Self {
        Self {
            k_period,
            k_smoothing,
            d_smoothing,
            overbought: 80.0,
            oversold: 20.0,
        }
    }

    pub fn with_thresholds(
        k_period: usize,
        k_smoothing: usize,
        d_smoothing: usize,
        overbought: f64,
        oversold: f64,
    ) -> Self {
        Self {
            k_period,
            k_smoothing,
            d_smoothing,
            overbought,
            oversold,
        }
    }

    /// Calculate all stochastic components.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> FullStochasticOutput {
        let fast_k_calc = FastStochasticK::new(self.k_period);
        let fast_k = fast_k_calc.calculate(high, low, close);

        let sma_k = SMA::new(self.k_smoothing);
        let slow_k = sma_k.calculate(&fast_k);

        let sma_d = SMA::new(self.d_smoothing);
        let slow_d = sma_d.calculate(&slow_k);

        FullStochasticOutput {
            fast_k,
            slow_k,
            slow_d,
        }
    }
}

impl TechnicalIndicator for FullStochastic {
    fn name(&self) -> &str {
        "Full Stochastic"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.k_period + self.k_smoothing + self.d_smoothing - 2;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let output = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(output.fast_k, output.slow_k, output.slow_d))
    }

    fn min_periods(&self) -> usize {
        self.k_period + self.k_smoothing + self.d_smoothing - 2
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for FullStochastic {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate(&data.high, &data.low, &data.close);

        if output.slow_k.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = output.slow_k.len();
        let k = output.slow_k[n - 1];
        let d = output.slow_d[n - 1];
        let prev_k = output.slow_k[n - 2];
        let prev_d = output.slow_d[n - 2];

        if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if prev_k <= prev_d && k > d && k < self.oversold + 10.0 {
            Ok(IndicatorSignal::Bullish)
        } else if prev_k >= prev_d && k < d && k > self.overbought - 10.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate(&data.high, &data.low, &data.close);
        let n = output.slow_k.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let k = output.slow_k[i];
            let d = output.slow_d[i];
            let prev_k = output.slow_k[i - 1];
            let prev_d = output.slow_d[i - 1];

            if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if prev_k <= prev_d && k > d && k < self.oversold + 10.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if prev_k >= prev_d && k < d && k > self.overbought - 10.0 {
                signals.push(IndicatorSignal::Bearish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

/// Lane's Original Stochastic - George Lane's original formulation.
///
/// The original stochastic as described by George Lane uses:
/// - 14-period lookback for %K
/// - 3-period SMA for %D
/// - No additional smoothing on %K
///
/// This is equivalent to the Fast Stochastic (14, 3).
#[derive(Debug, Clone)]
pub struct LaneStochastic {
    k_period: usize,
    d_period: usize,
    overbought: f64,
    oversold: f64,
}

impl Default for LaneStochastic {
    fn default() -> Self {
        Self {
            k_period: 14,
            d_period: 3,
            overbought: 80.0,
            oversold: 20.0,
        }
    }
}

impl LaneStochastic {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_periods(k_period: usize, d_period: usize) -> Self {
        Self {
            k_period,
            d_period,
            overbought: 80.0,
            oversold: 20.0,
        }
    }

    /// Calculate Lane's original stochastic.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let fast_stoch = FastStochasticD::new(self.k_period, self.d_period);
        fast_stoch.calculate(high, low, close)
    }

    /// Get the overbought threshold.
    pub fn overbought(&self) -> f64 {
        self.overbought
    }

    /// Get the oversold threshold.
    pub fn oversold(&self) -> f64 {
        self.oversold
    }
}

impl TechnicalIndicator for LaneStochastic {
    fn name(&self) -> &str {
        "Lane's Stochastic"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.k_period + self.d_period - 1;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(k_line, d_line))
    }

    fn min_periods(&self) -> usize {
        self.k_period + self.d_period - 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for LaneStochastic {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);

        if k_line.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let n = k_line.len();
        let k = k_line[n - 1];
        let d = d_line[n - 1];
        let prev_k = k_line[n - 2];
        let prev_d = d_line[n - 2];

        if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Lane's classic signals
        if prev_k <= prev_d && k > d && k < self.oversold + 10.0 {
            Ok(IndicatorSignal::Bullish)
        } else if prev_k >= prev_d && k < d && k > self.overbought - 10.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (k_line, d_line) = self.calculate(&data.high, &data.low, &data.close);
        let n = k_line.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let k = k_line[i];
            let d = d_line[i];
            let prev_k = k_line[i - 1];
            let prev_d = d_line[i - 1];

            if k.is_nan() || d.is_nan() || prev_k.is_nan() || prev_d.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if prev_k <= prev_d && k > d && k < self.oversold + 10.0 {
                signals.push(IndicatorSignal::Bullish);
            } else if prev_k >= prev_d && k < d && k > self.overbought - 10.0 {
                signals.push(IndicatorSignal::Bearish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

/// Stochastic %K Divergence - Detects divergence between price and stochastic.
///
/// Bullish divergence: Price makes lower low, but %K makes higher low
/// Bearish divergence: Price makes higher high, but %K makes lower high
#[derive(Debug, Clone)]
pub struct StochasticDivergence {
    k_period: usize,
    lookback: usize,
}

/// Divergence detection result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceType {
    /// No divergence detected
    None,
    /// Bullish divergence (potential bottom)
    Bullish,
    /// Bearish divergence (potential top)
    Bearish,
    /// Hidden bullish (trend continuation)
    HiddenBullish,
    /// Hidden bearish (trend continuation)
    HiddenBearish,
}

/// Output for Stochastic Divergence indicator.
#[derive(Debug, Clone)]
pub struct DivergenceOutput {
    /// The %K values
    pub k_line: Vec<f64>,
    /// Divergence type at each point
    pub divergence: Vec<DivergenceType>,
}

impl StochasticDivergence {
    pub fn new(k_period: usize, lookback: usize) -> Self {
        Self { k_period, lookback }
    }

    /// Calculate divergence.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> DivergenceOutput {
        let fast_k = FastStochasticK::new(self.k_period);
        let k_line = fast_k.calculate(high, low, close);
        let n = close.len();

        let mut divergence = vec![DivergenceType::None; n];

        if n < self.lookback + self.k_period {
            return DivergenceOutput { k_line, divergence };
        }

        for i in (self.lookback + self.k_period - 1)..n {
            let start = i - self.lookback;

            // Find price swing points
            let price_low_idx = (start..=i).min_by(|&a, &b|
                low[a].partial_cmp(&low[b]).unwrap_or(std::cmp::Ordering::Equal)
            ).unwrap();
            let price_high_idx = (start..=i).max_by(|&a, &b|
                high[a].partial_cmp(&high[b]).unwrap_or(std::cmp::Ordering::Equal)
            ).unwrap();

            // Find %K swing points (skip NaN values)
            let k_values: Vec<(usize, f64)> = (start..=i)
                .filter(|&j| !k_line[j].is_nan())
                .map(|j| (j, k_line[j]))
                .collect();

            if k_values.len() < 2 {
                continue;
            }

            let k_low_point = k_values.iter().min_by(|a, b|
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            );
            let k_high_point = k_values.iter().max_by(|a, b|
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            );

            if let (Some(&(k_low_idx, k_low_val)), Some(&(k_high_idx, k_high_val))) = (k_low_point, k_high_point) {
                // Check for regular bullish divergence
                // Price makes lower low, %K makes higher low
                if i == n - 1 && price_low_idx == i {
                    let prev_low = low[start..i].iter().cloned().fold(f64::INFINITY, f64::min);
                    let prev_k_low = k_values.iter()
                        .filter(|&&(idx, _)| idx < i)
                        .map(|&(_, v)| v)
                        .fold(f64::INFINITY, f64::min);

                    if low[i] < prev_low && k_line[i] > prev_k_low && !k_line[i].is_nan() {
                        divergence[i] = DivergenceType::Bullish;
                        continue;
                    }
                }

                // Check for regular bearish divergence
                // Price makes higher high, %K makes lower high
                if i == n - 1 && price_high_idx == i {
                    let prev_high = high[start..i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let prev_k_high = k_values.iter()
                        .filter(|&&(idx, _)| idx < i)
                        .map(|&(_, v)| v)
                        .fold(f64::NEG_INFINITY, f64::max);

                    if high[i] > prev_high && k_line[i] < prev_k_high && !k_line[i].is_nan() {
                        divergence[i] = DivergenceType::Bearish;
                        continue;
                    }
                }
            }
        }

        DivergenceOutput { k_line, divergence }
    }
}

impl TechnicalIndicator for StochasticDivergence {
    fn name(&self) -> &str {
        "Stochastic Divergence"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.k_period + self.lookback;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let output = self.calculate(&data.high, &data.low, &data.close);
        let divergence_values: Vec<f64> = output.divergence.iter().map(|d| match d {
            DivergenceType::None => 0.0,
            DivergenceType::Bullish => 1.0,
            DivergenceType::Bearish => -1.0,
            DivergenceType::HiddenBullish => 0.5,
            DivergenceType::HiddenBearish => -0.5,
        }).collect();

        Ok(IndicatorOutput::dual(output.k_line, divergence_values))
    }

    fn min_periods(&self) -> usize {
        self.k_period + self.lookback
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for StochasticDivergence {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate(&data.high, &data.low, &data.close);

        if output.divergence.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        match output.divergence.last().unwrap() {
            DivergenceType::Bullish | DivergenceType::HiddenBullish => Ok(IndicatorSignal::Bullish),
            DivergenceType::Bearish | DivergenceType::HiddenBearish => Ok(IndicatorSignal::Bearish),
            DivergenceType::None => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate(&data.high, &data.low, &data.close);

        Ok(output.divergence.iter().map(|d| match d {
            DivergenceType::Bullish | DivergenceType::HiddenBullish => IndicatorSignal::Bullish,
            DivergenceType::Bearish | DivergenceType::HiddenBearish => IndicatorSignal::Bearish,
            DivergenceType::None => IndicatorSignal::Neutral,
        }).collect())
    }
}

/// Stochastic Pop - Detects momentum breakout from oversold zone.
///
/// A Stochastic Pop occurs when:
/// 1. %K is in oversold territory (<20)
/// 2. %K rapidly rises above 50
/// 3. Often accompanied by price breakout
#[derive(Debug, Clone)]
pub struct StochasticPop {
    k_period: usize,
    d_period: usize,
    oversold_threshold: f64,
    pop_threshold: f64,
}

impl StochasticPop {
    pub fn new(k_period: usize, d_period: usize) -> Self {
        Self {
            k_period,
            d_period,
            oversold_threshold: 20.0,
            pop_threshold: 50.0,
        }
    }

    pub fn with_thresholds(k_period: usize, d_period: usize, oversold: f64, pop: f64) -> Self {
        Self {
            k_period,
            d_period,
            oversold_threshold: oversold,
            pop_threshold: pop,
        }
    }

    /// Detect Stochastic Pop signals.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<bool> {
        let fast_k = FastStochasticK::new(self.k_period);
        let k_line = fast_k.calculate(high, low, close);
        let n = k_line.len();

        let mut pops = vec![false; n];

        for i in 1..n {
            if k_line[i].is_nan() || k_line[i - 1].is_nan() {
                continue;
            }

            // Pop: was oversold, now crossed above pop_threshold
            if k_line[i - 1] < self.oversold_threshold && k_line[i] > self.pop_threshold {
                pops[i] = true;
            }
        }

        pops
    }
}

impl TechnicalIndicator for StochasticPop {
    fn name(&self) -> &str {
        "Stochastic Pop"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.k_period {
            return Err(IndicatorError::InsufficientData {
                required: self.k_period,
                got: data.close.len(),
            });
        }

        let pops = self.calculate(&data.high, &data.low, &data.close);
        let values: Vec<f64> = pops.iter().map(|&p| if p { 1.0 } else { 0.0 }).collect();
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.k_period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for StochasticPop {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let pops = self.calculate(&data.high, &data.low, &data.close);

        if pops.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        if *pops.last().unwrap() {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let pops = self.calculate(&data.high, &data.low, &data.close);

        Ok(pops.iter().map(|&p| {
            if p { IndicatorSignal::Bullish } else { IndicatorSignal::Neutral }
        }).collect())
    }
}

/// Stochastic Drop - Detects momentum breakdown from overbought zone.
///
/// A Stochastic Drop occurs when:
/// 1. %K is in overbought territory (>80)
/// 2. %K rapidly falls below 50
/// 3. Often accompanied by price breakdown
#[derive(Debug, Clone)]
pub struct StochasticDrop {
    k_period: usize,
    d_period: usize,
    overbought_threshold: f64,
    drop_threshold: f64,
}

impl StochasticDrop {
    pub fn new(k_period: usize, d_period: usize) -> Self {
        Self {
            k_period,
            d_period,
            overbought_threshold: 80.0,
            drop_threshold: 50.0,
        }
    }

    pub fn with_thresholds(k_period: usize, d_period: usize, overbought: f64, drop: f64) -> Self {
        Self {
            k_period,
            d_period,
            overbought_threshold: overbought,
            drop_threshold: drop,
        }
    }

    /// Detect Stochastic Drop signals.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<bool> {
        let fast_k = FastStochasticK::new(self.k_period);
        let k_line = fast_k.calculate(high, low, close);
        let n = k_line.len();

        let mut drops = vec![false; n];

        for i in 1..n {
            if k_line[i].is_nan() || k_line[i - 1].is_nan() {
                continue;
            }

            // Drop: was overbought, now crossed below drop_threshold
            if k_line[i - 1] > self.overbought_threshold && k_line[i] < self.drop_threshold {
                drops[i] = true;
            }
        }

        drops
    }
}

impl TechnicalIndicator for StochasticDrop {
    fn name(&self) -> &str {
        "Stochastic Drop"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.k_period {
            return Err(IndicatorError::InsufficientData {
                required: self.k_period,
                got: data.close.len(),
            });
        }

        let drops = self.calculate(&data.high, &data.low, &data.close);
        let values: Vec<f64> = drops.iter().map(|&d| if d { 1.0 } else { 0.0 }).collect();
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.k_period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for StochasticDrop {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let drops = self.calculate(&data.high, &data.low, &data.close);

        if drops.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        if *drops.last().unwrap() {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let drops = self.calculate(&data.high, &data.low, &data.close);

        Ok(drops.iter().map(|&d| {
            if d { IndicatorSignal::Bearish } else { IndicatorSignal::Neutral }
        }).collect())
    }
}

/// Stochastic Crossover - Detects %K/%D crossover signals.
///
/// Classic Lane crossover system:
/// - Bullish: %K crosses above %D
/// - Bearish: %K crosses below %D
#[derive(Debug, Clone)]
pub struct StochasticCrossover {
    k_period: usize,
    d_period: usize,
    k_smoothing: usize,
}

/// Output for Stochastic Crossover.
#[derive(Debug, Clone)]
pub struct CrossoverOutput {
    /// %K line
    pub k_line: Vec<f64>,
    /// %D line
    pub d_line: Vec<f64>,
    /// Crossover signals: 1.0 = bullish cross, -1.0 = bearish cross, 0.0 = none
    pub crossovers: Vec<f64>,
}

impl StochasticCrossover {
    pub fn new(k_period: usize, d_period: usize) -> Self {
        Self {
            k_period,
            d_period,
            k_smoothing: 1, // No additional smoothing for fast stochastic
        }
    }

    pub fn slow(k_period: usize, k_smoothing: usize, d_period: usize) -> Self {
        Self {
            k_period,
            d_period,
            k_smoothing,
        }
    }

    /// Calculate crossover signals.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> CrossoverOutput {
        let fast_k = FastStochasticK::new(self.k_period);
        let raw_k = fast_k.calculate(high, low, close);

        // Apply smoothing if needed
        let k_line = if self.k_smoothing > 1 {
            let sma = SMA::new(self.k_smoothing);
            sma.calculate(&raw_k)
        } else {
            raw_k
        };

        let sma_d = SMA::new(self.d_period);
        let d_line = sma_d.calculate(&k_line);

        let n = k_line.len();
        let mut crossovers = vec![0.0; n];

        for i in 1..n {
            if k_line[i].is_nan() || d_line[i].is_nan() ||
               k_line[i - 1].is_nan() || d_line[i - 1].is_nan() {
                continue;
            }

            // Bullish crossover: %K crosses above %D
            if k_line[i - 1] <= d_line[i - 1] && k_line[i] > d_line[i] {
                crossovers[i] = 1.0;
            }
            // Bearish crossover: %K crosses below %D
            else if k_line[i - 1] >= d_line[i - 1] && k_line[i] < d_line[i] {
                crossovers[i] = -1.0;
            }
        }

        CrossoverOutput { k_line, d_line, crossovers }
    }
}

impl TechnicalIndicator for StochasticCrossover {
    fn name(&self) -> &str {
        "Stochastic Crossover"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.k_period + self.k_smoothing + self.d_period - 2;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let output = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(output.k_line, output.d_line, output.crossovers))
    }

    fn min_periods(&self) -> usize {
        self.k_period + self.k_smoothing + self.d_period - 2
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for StochasticCrossover {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate(&data.high, &data.low, &data.close);

        if output.crossovers.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let last = *output.crossovers.last().unwrap();
        if last > 0.5 {
            Ok(IndicatorSignal::Bullish)
        } else if last < -0.5 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate(&data.high, &data.low, &data.close);

        Ok(output.crossovers.iter().map(|&c| {
            if c > 0.5 {
                IndicatorSignal::Bullish
            } else if c < -0.5 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..50).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..50).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        (high, low, close)
    }

    #[test]
    fn test_fast_stochastic_k() {
        let (high, low, close) = make_test_data();
        let fast_k = FastStochasticK::new(14);
        let result = fast_k.calculate(&high, &low, &close);

        assert_eq!(result.len(), 50);

        // First 13 values should be NaN
        for i in 0..13 {
            assert!(result[i].is_nan());
        }

        // Valid values should be between 0 and 100
        for i in 13..50 {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_fast_stochastic_d() {
        let (high, low, close) = make_test_data();
        let fast_stoch = FastStochasticD::new(14, 3);
        let (k_line, d_line) = fast_stoch.calculate(&high, &low, &close);

        assert_eq!(k_line.len(), 50);
        assert_eq!(d_line.len(), 50);

        // %D is smoother, so it has more NaN values at the start
        for i in 15..50 {
            if !d_line[i].is_nan() {
                assert!(d_line[i] >= 0.0 && d_line[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_slow_stochastic_k() {
        let (high, low, close) = make_test_data();
        let slow_k = SlowStochasticK::new(14, 3);
        let result = slow_k.calculate(&high, &low, &close);

        assert_eq!(result.len(), 50);

        // Valid values should be between 0 and 100
        for i in 15..50 {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_slow_stochastic_d() {
        let (high, low, close) = make_test_data();
        let slow_stoch = SlowStochasticD::new(14, 3, 3);
        let (slow_k, slow_d) = slow_stoch.calculate(&high, &low, &close);

        assert_eq!(slow_k.len(), 50);
        assert_eq!(slow_d.len(), 50);

        // Valid values should be between 0 and 100
        for i in 18..50 {
            if !slow_d[i].is_nan() {
                assert!(slow_d[i] >= 0.0 && slow_d[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_full_stochastic() {
        let (high, low, close) = make_test_data();
        let full_stoch = FullStochastic::new(14, 3, 3);
        let output = full_stoch.calculate(&high, &low, &close);

        assert_eq!(output.fast_k.len(), 50);
        assert_eq!(output.slow_k.len(), 50);
        assert_eq!(output.slow_d.len(), 50);

        // Slow %K should be smoother than Fast %K
        // Slow %D should be smoother than Slow %K
    }

    #[test]
    fn test_lane_stochastic() {
        let (high, low, close) = make_test_data();
        let lane = LaneStochastic::new();
        let (k_line, d_line) = lane.calculate(&high, &low, &close);

        assert_eq!(k_line.len(), 50);
        assert_eq!(d_line.len(), 50);

        // Default periods are 14, 3
        assert_eq!(lane.k_period, 14);
        assert_eq!(lane.d_period, 3);
        assert_eq!(lane.overbought(), 80.0);
        assert_eq!(lane.oversold(), 20.0);
    }

    #[test]
    fn test_stochastic_boundary_values() {
        // Test with data that should produce extreme values
        let high = vec![100.0; 20];
        let low = vec![90.0; 20];

        // Close at the high - should give %K = 100
        let close_at_high = vec![100.0; 20];
        let fast_k = FastStochasticK::new(14);
        let result = fast_k.calculate(&high, &low, &close_at_high);
        assert!((result[19] - 100.0).abs() < 0.001);

        // Close at the low - should give %K = 0
        let close_at_low = vec![90.0; 20];
        let result2 = fast_k.calculate(&high, &low, &close_at_low);
        assert!(result2[19].abs() < 0.001);
    }

    #[test]
    fn test_stochastic_divergence() {
        let (high, low, close) = make_test_data();
        let divergence = StochasticDivergence::new(14, 10);
        let output = divergence.calculate(&high, &low, &close);

        assert_eq!(output.k_line.len(), 50);
        assert_eq!(output.divergence.len(), 50);
    }

    #[test]
    fn test_stochastic_pop() {
        // Create data with a pop scenario: oversold then rapid rise
        let mut high = vec![100.0; 30];
        let mut low = vec![90.0; 30];
        let mut close = vec![91.0; 30]; // Start oversold

        // Make close rise rapidly
        for i in 15..30 {
            close[i] = 99.0;
        }

        let pop = StochasticPop::new(14, 3);
        let pops = pop.calculate(&high, &low, &close);

        assert_eq!(pops.len(), 30);
    }

    #[test]
    fn test_stochastic_drop() {
        // Create data with a drop scenario: overbought then rapid fall
        let high = vec![100.0; 30];
        let low = vec![90.0; 30];
        let mut close = vec![99.0; 30]; // Start overbought

        // Make close fall rapidly
        for i in 15..30 {
            close[i] = 91.0;
        }

        let drop = StochasticDrop::new(14, 3);
        let drops = drop.calculate(&high, &low, &close);

        assert_eq!(drops.len(), 30);
    }

    #[test]
    fn test_stochastic_crossover() {
        let (high, low, close) = make_test_data();
        let crossover = StochasticCrossover::new(14, 3);
        let output = crossover.calculate(&high, &low, &close);

        assert_eq!(output.k_line.len(), 50);
        assert_eq!(output.d_line.len(), 50);
        assert_eq!(output.crossovers.len(), 50);

        // Crossovers should be -1, 0, or 1
        for c in &output.crossovers {
            assert!(*c >= -1.0 && *c <= 1.0);
        }
    }
}
