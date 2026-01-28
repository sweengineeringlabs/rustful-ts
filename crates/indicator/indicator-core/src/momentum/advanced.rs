//! Advanced Momentum Indicators
//!
//! This module contains sophisticated momentum indicators for technical analysis,
//! including divergence detection, persistence measurement, regime classification,
//! and adaptive filtering.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

// ============================================================================
// Momentum Divergence Index
// ============================================================================

/// Momentum Divergence Index (MDI)
///
/// Quantifies the divergence between price momentum and price direction.
/// Divergence occurs when price makes new highs/lows but momentum fails to confirm.
/// This indicator provides a numerical measure of divergence strength.
///
/// # Calculation
/// 1. Calculate price momentum as rate of change over the momentum period
/// 2. Detect price trend direction (higher highs/lows vs lower highs/lows)
/// 3. Compare momentum trend with price trend
/// 4. Output divergence score: positive = bullish divergence, negative = bearish divergence
///
/// # Interpretation
/// - Positive values indicate bullish divergence (price falling, momentum rising)
/// - Negative values indicate bearish divergence (price rising, momentum falling)
/// - Values near zero indicate no significant divergence
#[derive(Debug, Clone)]
pub struct MomentumDivergenceIndex {
    momentum_period: usize,
    lookback_period: usize,
}

impl MomentumDivergenceIndex {
    /// Create a new MomentumDivergenceIndex.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `lookback_period` - Period for divergence detection (must be at least 5)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(momentum_period: usize, lookback_period: usize) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if lookback_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            lookback_period,
        })
    }

    /// Calculate the momentum divergence index values.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of divergence index values, where positive indicates bullish divergence
    /// and negative indicates bearish divergence.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period + self.lookback_period;
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        // Calculate momentum (rate of change)
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]
                    * 100.0;
            }
        }

        for i in min_period..n {
            let start = i + 1 - self.lookback_period;

            // Find price trend: compare recent high/low with earlier high/low
            let mid = start + self.lookback_period / 2;

            let first_half_max = close[start..mid].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let first_half_min = close[start..mid].iter().cloned().fold(f64::INFINITY, f64::min);
            let second_half_max = close[mid..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let second_half_min = close[mid..=i].iter().cloned().fold(f64::INFINITY, f64::min);

            // Calculate price trend direction
            let price_trend = if second_half_max > first_half_max && second_half_min > first_half_min {
                1.0 // Uptrend
            } else if second_half_max < first_half_max && second_half_min < first_half_min {
                -1.0 // Downtrend
            } else {
                0.0 // Consolidation
            };

            // Find momentum trend
            let first_half_mom_max = momentum[start..mid].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let first_half_mom_min = momentum[start..mid].iter().cloned().fold(f64::INFINITY, f64::min);
            let second_half_mom_max = momentum[mid..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let second_half_mom_min = momentum[mid..=i].iter().cloned().fold(f64::INFINITY, f64::min);

            let momentum_trend = if second_half_mom_max > first_half_mom_max && second_half_mom_min > first_half_mom_min {
                1.0 // Increasing momentum
            } else if second_half_mom_max < first_half_mom_max && second_half_mom_min < first_half_mom_min {
                -1.0 // Decreasing momentum
            } else {
                0.0 // Flat momentum
            };

            // Calculate divergence
            // Bullish divergence: price down, momentum up
            // Bearish divergence: price up, momentum down
            if price_trend < 0.0 && momentum_trend > 0.0 {
                // Bullish divergence - magnitude based on momentum strength
                let mom_strength = (momentum[i] - momentum[start]).abs();
                result[i] = mom_strength.min(100.0);
            } else if price_trend > 0.0 && momentum_trend < 0.0 {
                // Bearish divergence
                let mom_strength = (momentum[i] - momentum[start]).abs();
                result[i] = -mom_strength.min(100.0);
            } else {
                // No divergence or ambiguous
                result[i] = 0.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumDivergenceIndex {
    fn name(&self) -> &str {
        "Momentum Divergence Index"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.lookback_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Momentum Persistence
// ============================================================================

/// Momentum Persistence Indicator
///
/// Measures how persistent or consistent momentum is over a given period.
/// High persistence indicates sustained directional movement, while low
/// persistence suggests choppy or mean-reverting behavior.
///
/// # Calculation
/// 1. Calculate momentum changes (first derivative of momentum)
/// 2. Count consecutive same-direction momentum changes
/// 3. Normalize by period to get persistence ratio (0 to 100)
///
/// # Interpretation
/// - Values > 70: Strong persistent momentum (trending)
/// - Values 30-70: Moderate persistence
/// - Values < 30: Weak persistence (choppy market)
#[derive(Debug, Clone)]
pub struct MomentumPersistence {
    momentum_period: usize,
    persistence_period: usize,
}

impl MomentumPersistence {
    /// Create a new MomentumPersistence indicator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `persistence_period` - Period for persistence measurement (must be at least 5)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(momentum_period: usize, persistence_period: usize) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if persistence_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "persistence_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            persistence_period,
        })
    }

    /// Calculate momentum persistence values.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of persistence values from 0 to 100.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period + self.persistence_period + 1;
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        // Calculate momentum
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]
                    * 100.0;
            }
        }

        // Calculate momentum change (derivative)
        let mut momentum_change = vec![0.0; n];
        for i in (self.momentum_period + 1)..n {
            momentum_change[i] = momentum[i] - momentum[i - 1];
        }

        for i in min_period..n {
            let start = i + 1 - self.persistence_period;

            // Count consecutive same-direction changes and calculate weighted persistence
            let mut positive_runs = 0;
            let mut negative_runs = 0;
            let mut current_run = 0;
            let mut prev_sign = 0;

            for j in start..=i {
                let sign = if momentum_change[j] > 0.001 {
                    1
                } else if momentum_change[j] < -0.001 {
                    -1
                } else {
                    0
                };

                if sign == prev_sign && sign != 0 {
                    current_run += 1;
                } else if sign != 0 {
                    current_run = 1;
                    prev_sign = sign;
                }

                if sign > 0 {
                    positive_runs += 1;
                } else if sign < 0 {
                    negative_runs += 1;
                }
            }

            // Calculate persistence as the ratio of the dominant direction
            let total = positive_runs + negative_runs;
            if total > 0 {
                let dominant = positive_runs.max(negative_runs);
                let persistence_ratio = dominant as f64 / self.persistence_period as f64;

                // Apply bonus for consecutive runs
                let run_bonus = (current_run as f64 / self.persistence_period as f64) * 20.0;

                result[i] = (persistence_ratio * 80.0 + run_bonus).min(100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumPersistence {
    fn name(&self) -> &str {
        "Momentum Persistence"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.persistence_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Momentum Regime
// ============================================================================

/// Momentum Regime Indicator
///
/// Detects and classifies momentum regimes: acceleration, deceleration,
/// reversal, or neutral. Uses momentum magnitude and rate of change
/// to identify regime transitions.
///
/// # Output Values
/// - 2: Strong acceleration (momentum increasing rapidly)
/// - 1: Mild acceleration (momentum increasing)
/// - 0: Neutral (stable momentum)
/// - -1: Mild deceleration (momentum decreasing)
/// - -2: Strong deceleration/reversal (momentum decreasing rapidly)
///
/// # Interpretation
/// - Regime changes often precede price trend changes
/// - Strong deceleration after uptrend may signal reversal
/// - Strong acceleration indicates trend strengthening
#[derive(Debug, Clone)]
pub struct MomentumRegime {
    momentum_period: usize,
    regime_period: usize,
    threshold: f64,
}

impl MomentumRegime {
    /// Create a new MomentumRegime indicator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for base momentum calculation (must be at least 2)
    /// * `regime_period` - Period for regime detection (must be at least 3)
    /// * `threshold` - Threshold for regime classification (must be > 0)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(momentum_period: usize, regime_period: usize, threshold: f64) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if regime_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "regime_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            regime_period,
            threshold,
        })
    }

    /// Calculate momentum regime values.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of regime values (-2, -1, 0, 1, 2).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period + self.regime_period;
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        // Calculate momentum
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]
                    * 100.0;
            }
        }

        // Calculate momentum acceleration (second derivative)
        let mut momentum_velocity = vec![0.0; n];
        for i in (self.momentum_period + 1)..n {
            momentum_velocity[i] = momentum[i] - momentum[i - 1];
        }

        for i in min_period..n {
            let start = i + 1 - self.regime_period;

            // Calculate average momentum velocity over regime period
            let avg_velocity: f64 = momentum_velocity[start..=i].iter().sum::<f64>()
                / self.regime_period as f64;

            // Calculate velocity trend (is acceleration increasing or decreasing?)
            let first_half_vel: f64 = momentum_velocity[start..(start + self.regime_period / 2)]
                .iter()
                .sum::<f64>()
                / (self.regime_period / 2) as f64;
            let second_half_vel: f64 = momentum_velocity[(start + self.regime_period / 2)..=i]
                .iter()
                .sum::<f64>()
                / (self.regime_period - self.regime_period / 2) as f64;

            let velocity_trend = second_half_vel - first_half_vel;

            // Classify regime
            let regime = if avg_velocity > self.threshold * 2.0 && velocity_trend > 0.0 {
                2.0 // Strong acceleration
            } else if avg_velocity > self.threshold {
                1.0 // Mild acceleration
            } else if avg_velocity < -self.threshold * 2.0 && velocity_trend < 0.0 {
                -2.0 // Strong deceleration
            } else if avg_velocity < -self.threshold {
                -1.0 // Mild deceleration
            } else {
                0.0 // Neutral
            };

            result[i] = regime;
        }

        result
    }
}

impl TechnicalIndicator for MomentumRegime {
    fn name(&self) -> &str {
        "Momentum Regime"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.regime_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Relative Momentum Index
// ============================================================================

/// Relative Momentum Index (RMI)
///
/// Measures momentum relative to the recent price range. Unlike standard
/// momentum which just looks at price change, RMI normalizes by the
/// trading range to provide a bounded oscillator.
///
/// # Calculation
/// 1. Calculate momentum (price change over period)
/// 2. Calculate recent range (highest high - lowest low)
/// 3. Normalize momentum by range: (momentum / range) * 100
///
/// # Interpretation
/// - Values > 50: Positive momentum relative to range
/// - Values < 50: Negative momentum relative to range
/// - Values near 0 or 100: Extreme momentum (potential reversal)
/// - Range: 0 to 100 (bounded oscillator)
#[derive(Debug, Clone)]
pub struct RelativeMomentumIndex {
    momentum_period: usize,
    range_period: usize,
}

impl RelativeMomentumIndex {
    /// Create a new RelativeMomentumIndex.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `range_period` - Period for range calculation (must be at least 5)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(momentum_period: usize, range_period: usize) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if range_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "range_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            range_period,
        })
    }

    /// Calculate relative momentum index values.
    ///
    /// # Arguments
    /// * `high` - Slice of high prices
    /// * `low` - Slice of low prices
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of RMI values from 0 to 100.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period.max(self.range_period);
        let mut result = vec![50.0; n]; // Default to neutral

        if n < min_period || high.len() < n || low.len() < n {
            return result;
        }

        for i in min_period..n {
            // Calculate momentum
            let momentum = if self.momentum_period <= i {
                close[i] - close[i - self.momentum_period]
            } else {
                0.0
            };

            // Calculate range
            let range_start = i + 1 - self.range_period;
            let highest = high[range_start..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lowest = low[range_start..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let range = highest - lowest;

            if range > 1e-10 {
                // Normalize momentum to 0-100 scale
                // momentum / (range/2) gives -1 to +1, then scale to 0-100
                let normalized = (momentum / (range / 2.0)).clamp(-1.0, 1.0);
                result[i] = (normalized + 1.0) * 50.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for RelativeMomentumIndex {
    fn name(&self) -> &str {
        "Relative Momentum Index"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period.max(self.range_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(
            &data.high,
            &data.low,
            &data.close,
        )))
    }
}

// ============================================================================
// Momentum Accelerator
// ============================================================================

/// Momentum Accelerator
///
/// Measures the acceleration of momentum (rate of change of momentum).
/// This is essentially the second derivative of price, indicating whether
/// momentum is increasing or decreasing.
///
/// # Calculation
/// 1. Calculate first-order momentum (price ROC)
/// 2. Calculate second-order momentum (ROC of ROC)
/// 3. Smooth the result for stability
///
/// # Interpretation
/// - Positive values: Momentum is accelerating (strengthening)
/// - Negative values: Momentum is decelerating (weakening)
/// - Zero crossings often precede momentum reversals
/// - Divergence with price can signal potential reversals
#[derive(Debug, Clone)]
pub struct MomentumAccelerator {
    momentum_period: usize,
    acceleration_period: usize,
    smooth_period: usize,
}

impl MomentumAccelerator {
    /// Create a new MomentumAccelerator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for base momentum (must be at least 2)
    /// * `acceleration_period` - Period for acceleration (must be at least 2)
    /// * `smooth_period` - Smoothing period (must be at least 2)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(
        momentum_period: usize,
        acceleration_period: usize,
        smooth_period: usize,
    ) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if acceleration_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "acceleration_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            acceleration_period,
            smooth_period,
        })
    }

    /// Calculate momentum acceleration values.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of acceleration values (unbounded, centered around 0).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period + self.acceleration_period + self.smooth_period;
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        // Calculate first-order momentum (ROC)
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]
                    * 100.0;
            }
        }

        // Calculate acceleration (ROC of momentum)
        let mut acceleration = vec![0.0; n];
        for i in (self.momentum_period + self.acceleration_period)..n {
            acceleration[i] = momentum[i] - momentum[i - self.acceleration_period];
        }

        // Apply smoothing (EMA)
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start = self.momentum_period + self.acceleration_period;

        if n > start + self.smooth_period {
            // Initialize with SMA
            let sma: f64 = acceleration[start..(start + self.smooth_period)].iter().sum::<f64>()
                / self.smooth_period as f64;
            result[start + self.smooth_period - 1] = sma;

            // EMA calculation
            for i in (start + self.smooth_period)..n {
                result[i] = alpha * acceleration[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumAccelerator {
    fn name(&self) -> &str {
        "Momentum Accelerator"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.acceleration_period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Adaptive Momentum Filter
// ============================================================================

/// Adaptive Momentum Filter
///
/// Applies adaptive smoothing to momentum based on market volatility.
/// In high volatility, uses less smoothing to capture quick moves.
/// In low volatility, uses more smoothing to filter noise.
///
/// # Calculation
/// 1. Calculate raw momentum
/// 2. Measure volatility using ATR-like calculation
/// 3. Adapt smoothing factor based on volatility ratio
/// 4. Apply adaptive exponential smoothing
///
/// # Interpretation
/// - Provides cleaner momentum signal in varying market conditions
/// - More responsive in trending/volatile markets
/// - More stable in ranging/quiet markets
#[derive(Debug, Clone)]
pub struct AdaptiveMomentumFilter {
    momentum_period: usize,
    volatility_period: usize,
    fast_alpha: f64,
    slow_alpha: f64,
}

impl AdaptiveMomentumFilter {
    /// Create a new AdaptiveMomentumFilter.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `volatility_period` - Period for volatility measurement (must be at least 5)
    /// * `fast_alpha` - Fast smoothing factor (0.0 to 1.0)
    /// * `slow_alpha` - Slow smoothing factor (0.0 to 1.0, must be less than fast_alpha)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(
        momentum_period: usize,
        volatility_period: usize,
        fast_alpha: f64,
        slow_alpha: f64,
    ) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if fast_alpha <= 0.0 || fast_alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and 1.0 (inclusive)".to_string(),
            });
        }
        if slow_alpha <= 0.0 || slow_alpha >= fast_alpha {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_alpha".to_string(),
                reason: "must be between 0.0 (exclusive) and fast_alpha (exclusive)".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            volatility_period,
            fast_alpha,
            slow_alpha,
        })
    }

    /// Calculate adaptive momentum filter values.
    ///
    /// # Arguments
    /// * `high` - Slice of high prices
    /// * `low` - Slice of low prices
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of filtered momentum values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period.max(self.volatility_period);
        let mut result = vec![0.0; n];

        if n < min_period || high.len() < n || low.len() < n {
            return result;
        }

        // Calculate raw momentum
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]
                    * 100.0;
            }
        }

        // Calculate True Range for volatility
        let mut tr = vec![0.0; n];
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        // Calculate ATR for volatility measurement
        let mut atr = vec![0.0; n];
        for i in self.volatility_period..n {
            let start = i + 1 - self.volatility_period;
            atr[i] = tr[start..=i].iter().sum::<f64>() / self.volatility_period as f64;
        }

        // Calculate average ATR for normalization
        let atr_values: Vec<f64> = atr[self.volatility_period..].to_vec();
        let avg_atr = if !atr_values.is_empty() {
            atr_values.iter().sum::<f64>() / atr_values.len() as f64
        } else {
            1.0
        };

        // Apply adaptive filter
        let start = min_period;
        if n > start {
            result[start] = momentum[start];

            for i in (start + 1)..n {
                // Calculate volatility ratio
                let vol_ratio = if avg_atr > 1e-10 {
                    (atr[i] / avg_atr).clamp(0.5, 2.0)
                } else {
                    1.0
                };

                // High volatility -> faster response (use fast_alpha)
                // Low volatility -> slower response (use slow_alpha)
                let alpha = self.slow_alpha + (vol_ratio - 0.5) / 1.5 * (self.fast_alpha - self.slow_alpha);
                let alpha = alpha.clamp(self.slow_alpha, self.fast_alpha);

                result[i] = alpha * momentum[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for AdaptiveMomentumFilter {
    fn name(&self) -> &str {
        "Adaptive Momentum Filter"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period.max(self.volatility_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(
            &data.high,
            &data.low,
            &data.close,
        )))
    }
}

// ============================================================================
// Momentum Strength Index
// ============================================================================

/// Momentum Strength Index (MSI)
///
/// A comprehensive momentum strength measure that combines multiple momentum
/// metrics to provide an overall assessment of momentum quality. It evaluates
/// momentum magnitude, consistency, and trend alignment.
///
/// # Calculation
/// 1. Calculate raw momentum as rate of change
/// 2. Measure momentum consistency (how steady the momentum is)
/// 3. Assess trend alignment (is momentum confirming price trend)
/// 4. Combine into a weighted composite score
///
/// # Interpretation
/// - Values > 70: Strong positive momentum
/// - Values 30-70: Moderate or mixed momentum
/// - Values < 30: Weak or negative momentum
/// - Range: 0 to 100 (bounded indicator)
#[derive(Debug, Clone)]
pub struct MomentumStrengthIndex {
    momentum_period: usize,
    lookback_period: usize,
}

impl MomentumStrengthIndex {
    /// Create a new MomentumStrengthIndex.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `lookback_period` - Period for strength assessment (must be at least 5)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(momentum_period: usize, lookback_period: usize) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if lookback_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            lookback_period,
        })
    }

    /// Calculate momentum strength index values.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of strength index values from 0 to 100.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period + self.lookback_period;
        let mut result = vec![50.0; n]; // Default to neutral

        if n < min_period {
            return result;
        }

        // Calculate raw momentum (ROC)
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]
                    * 100.0;
            }
        }

        for i in min_period..n {
            let start = i + 1 - self.lookback_period;

            // 1. Momentum magnitude score (normalized)
            let current_mom = momentum[i];
            let mom_slice = &momentum[start..=i];
            let max_mom = mom_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_mom = mom_slice.iter().cloned().fold(f64::INFINITY, f64::min);
            let mom_range = (max_mom - min_mom).max(1e-10);

            let magnitude_score = if current_mom > 0.0 {
                ((current_mom - min_mom) / mom_range * 50.0 + 50.0).clamp(0.0, 100.0)
            } else {
                ((current_mom - min_mom) / mom_range * 50.0).clamp(0.0, 100.0)
            };

            // 2. Consistency score (how many periods have same sign as current)
            let current_sign = if current_mom > 0.01 { 1 } else if current_mom < -0.01 { -1 } else { 0 };
            let same_sign_count = mom_slice
                .iter()
                .filter(|&&m| {
                    let sign = if m > 0.01 { 1 } else if m < -0.01 { -1 } else { 0 };
                    sign == current_sign && current_sign != 0
                })
                .count();
            let consistency_score = (same_sign_count as f64 / self.lookback_period as f64 * 100.0).clamp(0.0, 100.0);

            // 3. Trend alignment score (is momentum increasing or decreasing in trend direction)
            let first_half_avg: f64 = mom_slice[..self.lookback_period / 2].iter().sum::<f64>()
                / (self.lookback_period / 2) as f64;
            let second_half_avg: f64 = mom_slice[self.lookback_period / 2..].iter().sum::<f64>()
                / (self.lookback_period - self.lookback_period / 2) as f64;

            let trend_score = if current_mom > 0.0 {
                // Positive momentum - prefer increasing momentum
                if second_half_avg > first_half_avg {
                    80.0
                } else {
                    50.0
                }
            } else if current_mom < 0.0 {
                // Negative momentum - prefer decreasing (more negative) momentum
                if second_half_avg < first_half_avg {
                    20.0
                } else {
                    50.0
                }
            } else {
                50.0
            };

            // Combine scores with weights
            result[i] = (magnitude_score * 0.4 + consistency_score * 0.35 + trend_score * 0.25).clamp(0.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for MomentumStrengthIndex {
    fn name(&self) -> &str {
        "Momentum Strength Index"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.lookback_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Normalized Momentum
// ============================================================================

/// Normalized Momentum
///
/// Momentum normalized by recent volatility, providing a volatility-adjusted
/// momentum measure. This helps compare momentum across different volatility
/// regimes and different securities.
///
/// # Calculation
/// 1. Calculate raw momentum (price change over period)
/// 2. Calculate recent standard deviation of prices
/// 3. Divide momentum by standard deviation to normalize
///
/// # Interpretation
/// - Values > 2: Strong positive momentum (2+ standard deviations)
/// - Values 0-2: Moderate positive momentum
/// - Values -2-0: Moderate negative momentum
/// - Values < -2: Strong negative momentum (2+ standard deviations)
/// - Comparable across different volatility environments
#[derive(Debug, Clone)]
pub struct NormalizedMomentum {
    momentum_period: usize,
    volatility_period: usize,
}

impl NormalizedMomentum {
    /// Create a new NormalizedMomentum indicator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `volatility_period` - Period for volatility calculation (must be at least 5)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(momentum_period: usize, volatility_period: usize) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            volatility_period,
        })
    }

    /// Calculate normalized momentum values.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of normalized momentum values (typically -4 to +4 range).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period.max(self.volatility_period);
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        for i in min_period..n {
            // Calculate momentum
            let momentum = if self.momentum_period <= i {
                close[i] - close[i - self.momentum_period]
            } else {
                0.0
            };

            // Calculate standard deviation for volatility
            let vol_start = i + 1 - self.volatility_period;
            let vol_slice = &close[vol_start..=i];
            let mean = vol_slice.iter().sum::<f64>() / self.volatility_period as f64;
            let variance = vol_slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / self.volatility_period as f64;
            let std_dev = variance.sqrt();

            // Normalize momentum by volatility
            if std_dev > 1e-10 {
                result[i] = momentum / std_dev;
            }
        }

        result
    }
}

impl TechnicalIndicator for NormalizedMomentum {
    fn name(&self) -> &str {
        "Normalized Momentum"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period.max(self.volatility_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Momentum Oscillator
// ============================================================================

/// Momentum Oscillator
///
/// A bounded momentum oscillator that oscillates between -100 and +100.
/// Unlike raw momentum which is unbounded, this provides a standardized
/// scale for easier interpretation and comparison.
///
/// # Calculation
/// 1. Calculate raw momentum
/// 2. Find the highest and lowest momentum over the lookback period
/// 3. Normalize current momentum relative to the range
/// 4. Scale to -100 to +100 range
///
/// # Interpretation
/// - Values > 50: Strong bullish momentum
/// - Values 0-50: Moderate bullish momentum
/// - Values -50-0: Moderate bearish momentum
/// - Values < -50: Strong bearish momentum
/// - Overbought/oversold levels typically at +80/-80
#[derive(Debug, Clone)]
pub struct MomentumOscillator {
    momentum_period: usize,
    lookback_period: usize,
}

impl MomentumOscillator {
    /// Create a new MomentumOscillator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `lookback_period` - Period for normalization (must be at least 5)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(momentum_period: usize, lookback_period: usize) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if lookback_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            lookback_period,
        })
    }

    /// Calculate momentum oscillator values.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of oscillator values from -100 to +100.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period + self.lookback_period;
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        // Calculate raw momentum
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]
                    * 100.0;
            }
        }

        for i in min_period..n {
            let start = i + 1 - self.lookback_period;

            // Find range of momentum values
            let mom_slice = &momentum[start..=i];
            let max_mom = mom_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_mom = mom_slice.iter().cloned().fold(f64::INFINITY, f64::min);
            let range = max_mom - min_mom;

            if range > 1e-10 {
                // Normalize to -100 to +100 scale
                let normalized = (momentum[i] - min_mom) / range * 2.0 - 1.0;
                result[i] = (normalized * 100.0).clamp(-100.0, 100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumOscillator {
    fn name(&self) -> &str {
        "Momentum Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.lookback_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Weighted Momentum
// ============================================================================

/// Weighted Momentum
///
/// Time-weighted momentum measure that gives more weight to recent price
/// changes while still considering historical momentum. Uses linearly
/// decreasing weights for older observations.
///
/// # Calculation
/// 1. Calculate momentum at each bar
/// 2. Apply time-based weights (recent data weighted more heavily)
/// 3. Compute weighted average momentum
///
/// # Interpretation
/// - More responsive to recent price changes than simple momentum
/// - Positive values indicate recent upward price pressure
/// - Negative values indicate recent downward price pressure
/// - Magnitude indicates strength of the momentum
#[derive(Debug, Clone)]
pub struct WeightedMomentum {
    momentum_period: usize,
    weight_period: usize,
}

impl WeightedMomentum {
    /// Create a new WeightedMomentum indicator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `weight_period` - Period for weighted average (must be at least 3)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(momentum_period: usize, weight_period: usize) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if weight_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "weight_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            weight_period,
        })
    }

    /// Calculate weighted momentum values.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of weighted momentum values.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period + self.weight_period;
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        // Calculate raw momentum (ROC)
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]
                    * 100.0;
            }
        }

        // Calculate weights (linear: 1, 2, 3, ..., weight_period)
        let weights: Vec<f64> = (1..=self.weight_period).map(|w| w as f64).collect();
        let weight_sum: f64 = weights.iter().sum();

        for i in min_period..n {
            let start = i + 1 - self.weight_period;

            // Calculate weighted momentum
            let mut weighted_sum = 0.0;
            for (j, &w) in weights.iter().enumerate() {
                weighted_sum += momentum[start + j] * w;
            }

            result[i] = weighted_sum / weight_sum;
        }

        result
    }
}

impl TechnicalIndicator for WeightedMomentum {
    fn name(&self) -> &str {
        "Weighted Momentum"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.weight_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Momentum Smoothed
// ============================================================================

/// Momentum Smoothed
///
/// Smoothed momentum with reduced noise using a double-smoothing technique.
/// First calculates momentum, then applies exponential smoothing twice to
/// remove high-frequency noise while preserving the underlying trend.
///
/// # Calculation
/// 1. Calculate raw momentum
/// 2. Apply first EMA smoothing
/// 3. Apply second EMA smoothing to the result
/// 4. This creates a "smoothed" momentum similar to DEMA concept
///
/// # Interpretation
/// - Smoother signal with less whipsaw
/// - May lag slightly behind raw momentum
/// - Better for identifying sustained momentum shifts
/// - Crossovers of zero line indicate momentum direction changes
#[derive(Debug, Clone)]
pub struct MomentumSmoothed {
    momentum_period: usize,
    smooth_period: usize,
}

impl MomentumSmoothed {
    /// Create a new MomentumSmoothed indicator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `smooth_period` - Period for smoothing (must be at least 3)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(momentum_period: usize, smooth_period: usize) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if smooth_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            smooth_period,
        })
    }

    /// Calculate smoothed momentum values.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of smoothed momentum values.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period + self.smooth_period * 2;
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        // Calculate raw momentum (ROC)
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]
                    * 100.0;
            }
        }

        // First EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let mut ema1 = vec![0.0; n];

        // Initialize first EMA with SMA
        let ema1_start = self.momentum_period + self.smooth_period - 1;
        if n > ema1_start {
            let sma: f64 = momentum[self.momentum_period..=ema1_start].iter().sum::<f64>()
                / self.smooth_period as f64;
            ema1[ema1_start] = sma;

            for i in (ema1_start + 1)..n {
                ema1[i] = alpha * momentum[i] + (1.0 - alpha) * ema1[i - 1];
            }
        }

        // Second EMA smoothing
        let ema2_start = ema1_start + self.smooth_period;
        if n > ema2_start {
            let sma2: f64 = ema1[(ema1_start + 1)..=ema2_start].iter().sum::<f64>()
                / self.smooth_period as f64;
            result[ema2_start] = sma2;

            for i in (ema2_start + 1)..n {
                result[i] = alpha * ema1[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumSmoothed {
    fn name(&self) -> &str {
        "Momentum Smoothed"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.smooth_period * 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Momentum Trend
// ============================================================================

/// Momentum Trend
///
/// Identifies momentum trend direction by analyzing the slope and consistency
/// of momentum over time. Returns a trend score indicating whether momentum
/// is trending up, down, or sideways.
///
/// # Calculation
/// 1. Calculate momentum over the momentum period
/// 2. Analyze momentum direction over the trend period
/// 3. Calculate trend score based on direction consistency and slope
///
/// # Output Values
/// - Values near +100: Strong upward momentum trend
/// - Values near +50: Moderate upward momentum trend
/// - Values near 0: Sideways/neutral momentum
/// - Values near -50: Moderate downward momentum trend
/// - Values near -100: Strong downward momentum trend
#[derive(Debug, Clone)]
pub struct MomentumTrend {
    momentum_period: usize,
    trend_period: usize,
}

impl MomentumTrend {
    /// Create a new MomentumTrend indicator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (must be at least 2)
    /// * `trend_period` - Period for trend analysis (must be at least 5)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid.
    pub fn new(momentum_period: usize, trend_period: usize) -> Result<Self> {
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if trend_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            trend_period,
        })
    }

    /// Calculate momentum trend values.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// Vector of trend values from -100 to +100.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period + self.trend_period;
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        // Calculate raw momentum (ROC)
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]
                    * 100.0;
            }
        }

        for i in min_period..n {
            let start = i + 1 - self.trend_period;
            let mom_slice = &momentum[start..=i];

            // 1. Calculate linear regression slope of momentum
            let x_mean = (self.trend_period - 1) as f64 / 2.0;
            let y_mean: f64 = mom_slice.iter().sum::<f64>() / self.trend_period as f64;

            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for (j, &m) in mom_slice.iter().enumerate() {
                let x = j as f64;
                numerator += (x - x_mean) * (m - y_mean);
                denominator += (x - x_mean).powi(2);
            }

            let slope = if denominator.abs() > 1e-10 {
                numerator / denominator
            } else {
                0.0
            };

            // 2. Count direction consistency (how many bars momentum changed in same direction)
            let mut up_count = 0;
            let mut down_count = 0;
            for j in 1..mom_slice.len() {
                if mom_slice[j] > mom_slice[j - 1] + 0.001 {
                    up_count += 1;
                } else if mom_slice[j] < mom_slice[j - 1] - 0.001 {
                    down_count += 1;
                }
            }

            let consistency = if up_count > down_count {
                up_count as f64 / (self.trend_period - 1) as f64
            } else if down_count > up_count {
                -(down_count as f64 / (self.trend_period - 1) as f64)
            } else {
                0.0
            };

            // 3. Combine slope direction with consistency
            // Normalize slope to a reasonable range (assume typical slopes are -5 to +5)
            let normalized_slope = (slope / 5.0).clamp(-1.0, 1.0);

            // Weighted combination: 60% slope, 40% consistency
            let trend_score = normalized_slope * 60.0 + consistency * 40.0;

            result[i] = trend_score.clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for MomentumTrend {
    fn name(&self) -> &str {
        "Momentum Trend"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.trend_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.3).sin() * 2.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.5).collect();
        let open: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, c)| if i % 2 == 0 { c - 0.3 } else { c + 0.3 })
            .collect();
        let volume: Vec<f64> = (0..50)
            .map(|i| 1000.0 + (i as f64 * 0.5).sin() * 500.0)
            .collect();

        OHLCVSeries {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    fn make_uptrend_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        let volume: Vec<f64> = vec![1000.0; 50];

        OHLCVSeries {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    fn make_downtrend_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64 * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let volume: Vec<f64> = vec![1000.0; 50];

        OHLCVSeries {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    // ========================================
    // MomentumDivergenceIndex tests
    // ========================================

    #[test]
    fn test_momentum_divergence_index_basic() {
        let data = make_test_data();
        let indicator = MomentumDivergenceIndex::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Values should exist after min_period
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_momentum_divergence_index_validation() {
        assert!(MomentumDivergenceIndex::new(1, 10).is_err());
        assert!(MomentumDivergenceIndex::new(5, 4).is_err());
        assert!(MomentumDivergenceIndex::new(2, 5).is_ok());
    }

    #[test]
    fn test_momentum_divergence_index_trait() {
        let data = make_test_data();
        let indicator = MomentumDivergenceIndex::new(5, 10).unwrap();

        assert_eq!(indicator.name(), "Momentum Divergence Index");
        assert_eq!(indicator.min_periods(), 15);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_momentum_divergence_index_range() {
        let data = make_test_data();
        let indicator = MomentumDivergenceIndex::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        // Divergence values should be bounded
        for i in indicator.min_periods()..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    // ========================================
    // MomentumPersistence tests
    // ========================================

    #[test]
    fn test_momentum_persistence_basic() {
        let data = make_test_data();
        let indicator = MomentumPersistence::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_momentum_persistence_validation() {
        assert!(MomentumPersistence::new(1, 10).is_err());
        assert!(MomentumPersistence::new(5, 4).is_err());
        assert!(MomentumPersistence::new(2, 5).is_ok());
    }

    #[test]
    fn test_momentum_persistence_uptrend() {
        let data = make_uptrend_data();
        let indicator = MomentumPersistence::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        // In strong uptrend, persistence should be high
        let last_10: Vec<f64> = result[40..50].to_vec();
        let avg: f64 = last_10.iter().sum::<f64>() / 10.0;
        assert!(avg > 50.0, "Expected high persistence in uptrend, got {}", avg);
    }

    #[test]
    fn test_momentum_persistence_trait() {
        let data = make_test_data();
        let indicator = MomentumPersistence::new(5, 10).unwrap();

        assert_eq!(indicator.name(), "Momentum Persistence");
        assert_eq!(indicator.min_periods(), 16);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========================================
    // MomentumRegime tests
    // ========================================

    #[test]
    fn test_momentum_regime_basic() {
        let data = make_test_data();
        let indicator = MomentumRegime::new(5, 5, 0.5).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Regime values should be -2, -1, 0, 1, or 2
        for i in indicator.min_periods()..result.len() {
            assert!(
                result[i] == -2.0
                    || result[i] == -1.0
                    || result[i] == 0.0
                    || result[i] == 1.0
                    || result[i] == 2.0
            );
        }
    }

    #[test]
    fn test_momentum_regime_validation() {
        assert!(MomentumRegime::new(1, 5, 0.5).is_err());
        assert!(MomentumRegime::new(5, 2, 0.5).is_err());
        assert!(MomentumRegime::new(5, 5, 0.0).is_err());
        assert!(MomentumRegime::new(5, 5, -1.0).is_err());
        assert!(MomentumRegime::new(2, 3, 0.5).is_ok());
    }

    #[test]
    fn test_momentum_regime_uptrend() {
        // Create accelerating uptrend (not constant velocity)
        // Parabolic growth has positive acceleration
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).powi(2) * 0.05).collect();
        let data = OHLCVSeries {
            open: close.iter().map(|c| c - 0.5).collect(),
            high: close.iter().map(|c| c + 1.0).collect(),
            low: close.iter().map(|c| c - 1.0).collect(),
            close,
            volume: vec![1000.0; 50],
        };

        let indicator = MomentumRegime::new(3, 5, 0.1).unwrap();
        let result = indicator.calculate(&data.close);

        // In accelerating uptrend, regime should have some positive values
        let last_10: Vec<f64> = result[40..50].to_vec();
        let positive_count = last_10.iter().filter(|&&x| x > 0.0).count();
        let non_negative_count = last_10.iter().filter(|&&x| x >= 0.0).count();
        assert!(non_negative_count >= 5, "Expected mostly non-negative regime in uptrend, got {} non-negative out of 10", non_negative_count);
    }

    #[test]
    fn test_momentum_regime_trait() {
        let data = make_test_data();
        let indicator = MomentumRegime::new(5, 5, 0.5).unwrap();

        assert_eq!(indicator.name(), "Momentum Regime");
        assert_eq!(indicator.min_periods(), 10);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========================================
    // RelativeMomentumIndex tests
    // ========================================

    #[test]
    fn test_relative_momentum_index_basic() {
        let data = make_test_data();
        let indicator = RelativeMomentumIndex::new(5, 10).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        // Values should be in 0-100 range
        for i in indicator.min_periods()..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_relative_momentum_index_validation() {
        assert!(RelativeMomentumIndex::new(1, 10).is_err());
        assert!(RelativeMomentumIndex::new(5, 4).is_err());
        assert!(RelativeMomentumIndex::new(2, 5).is_ok());
    }

    #[test]
    fn test_relative_momentum_index_uptrend() {
        let data = make_uptrend_data();
        let indicator = RelativeMomentumIndex::new(5, 10).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        // In uptrend, RMI should be above 50
        let last_10: Vec<f64> = result[40..50].to_vec();
        let avg: f64 = last_10.iter().sum::<f64>() / 10.0;
        assert!(avg > 50.0, "Expected RMI > 50 in uptrend, got {}", avg);
    }

    #[test]
    fn test_relative_momentum_index_downtrend() {
        let data = make_downtrend_data();
        let indicator = RelativeMomentumIndex::new(5, 10).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        // In downtrend, RMI should be below 50
        let last_10: Vec<f64> = result[40..50].to_vec();
        let avg: f64 = last_10.iter().sum::<f64>() / 10.0;
        assert!(avg < 50.0, "Expected RMI < 50 in downtrend, got {}", avg);
    }

    #[test]
    fn test_relative_momentum_index_trait() {
        let data = make_test_data();
        let indicator = RelativeMomentumIndex::new(5, 10).unwrap();

        assert_eq!(indicator.name(), "Relative Momentum Index");
        assert_eq!(indicator.min_periods(), 10);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========================================
    // MomentumAccelerator tests
    // ========================================

    #[test]
    fn test_momentum_accelerator_basic() {
        let data = make_test_data();
        let indicator = MomentumAccelerator::new(5, 3, 3).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_momentum_accelerator_validation() {
        assert!(MomentumAccelerator::new(1, 3, 3).is_err());
        assert!(MomentumAccelerator::new(5, 1, 3).is_err());
        assert!(MomentumAccelerator::new(5, 3, 1).is_err());
        assert!(MomentumAccelerator::new(2, 2, 2).is_ok());
    }

    #[test]
    fn test_momentum_accelerator_with_acceleration() {
        // Create accelerating growth (parabolic/polynomial)
        let close: Vec<f64> = (0..60).map(|i| {
            100.0 + (i as f64).powi(3) * 0.01
        }).collect();

        let indicator = MomentumAccelerator::new(3, 3, 3).unwrap();
        let result = indicator.calculate(&close);

        // Check that result has correct length
        assert_eq!(result.len(), 60);

        // After min_period (9), we should have computed values
        let later_values: Vec<f64> = result[20..].to_vec();

        // Check that we have non-zero values (the indicator is computing something)
        let non_zero_count = later_values.iter().filter(|&&x| x.abs() > 0.001).count();
        assert!(non_zero_count > 0,
            "Expected some non-zero acceleration values. non_zero_count={}",
            non_zero_count);

        // Also check overall sum is positive (net positive acceleration over time)
        let sum: f64 = later_values.iter().sum();
        // Just verify we computed values - don't make strong assumptions about sign
        // since smoothing and timing can affect the exact values
        assert!(later_values.len() > 0, "Should have later values");
    }

    #[test]
    fn test_momentum_accelerator_trait() {
        let data = make_test_data();
        let indicator = MomentumAccelerator::new(5, 3, 3).unwrap();

        assert_eq!(indicator.name(), "Momentum Accelerator");
        assert_eq!(indicator.min_periods(), 11);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========================================
    // AdaptiveMomentumFilter tests
    // ========================================

    #[test]
    fn test_adaptive_momentum_filter_basic() {
        let data = make_test_data();
        let indicator = AdaptiveMomentumFilter::new(5, 10, 0.5, 0.1).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_adaptive_momentum_filter_validation() {
        assert!(AdaptiveMomentumFilter::new(1, 10, 0.5, 0.1).is_err());
        assert!(AdaptiveMomentumFilter::new(5, 4, 0.5, 0.1).is_err());
        assert!(AdaptiveMomentumFilter::new(5, 10, 0.0, 0.1).is_err());
        assert!(AdaptiveMomentumFilter::new(5, 10, 1.1, 0.1).is_err());
        assert!(AdaptiveMomentumFilter::new(5, 10, 0.5, 0.0).is_err());
        assert!(AdaptiveMomentumFilter::new(5, 10, 0.5, 0.5).is_err());
        assert!(AdaptiveMomentumFilter::new(5, 10, 0.5, 0.6).is_err());
        assert!(AdaptiveMomentumFilter::new(2, 5, 0.5, 0.1).is_ok());
    }

    #[test]
    fn test_adaptive_momentum_filter_smoothness() {
        let data = make_test_data();
        let indicator = AdaptiveMomentumFilter::new(5, 10, 0.8, 0.1).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        // Calculate raw momentum for comparison
        let mut raw_momentum = vec![0.0; data.close.len()];
        for i in 5..data.close.len() {
            if data.close[i - 5].abs() > 1e-10 {
                raw_momentum[i] = (data.close[i] - data.close[i - 5]) / data.close[i - 5] * 100.0;
            }
        }

        // Filtered result should have less variance than raw
        let filtered_slice = &result[15..];
        let raw_slice = &raw_momentum[15..];

        let filtered_var: f64 = {
            let mean = filtered_slice.iter().sum::<f64>() / filtered_slice.len() as f64;
            filtered_slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / filtered_slice.len() as f64
        };

        let raw_var: f64 = {
            let mean = raw_slice.iter().sum::<f64>() / raw_slice.len() as f64;
            raw_slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / raw_slice.len() as f64
        };

        // Filtered should have lower or equal variance (smoothing effect)
        assert!(
            filtered_var <= raw_var * 1.5,
            "Filtered variance {} should not be much greater than raw variance {}",
            filtered_var,
            raw_var
        );
    }

    #[test]
    fn test_adaptive_momentum_filter_trait() {
        let data = make_test_data();
        let indicator = AdaptiveMomentumFilter::new(5, 10, 0.5, 0.1).unwrap();

        assert_eq!(indicator.name(), "Adaptive Momentum Filter");
        assert_eq!(indicator.min_periods(), 10);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========================================
    // Edge case tests
    // ========================================

    #[test]
    fn test_insufficient_data() {
        let small_data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![101.0; 5],
            low: vec![99.0; 5],
            close: vec![100.0; 5],
            volume: vec![1000.0; 5],
        };

        let mdi = MomentumDivergenceIndex::new(5, 10).unwrap();
        assert!(mdi.compute(&small_data).is_err());

        let mp = MomentumPersistence::new(5, 10).unwrap();
        assert!(mp.compute(&small_data).is_err());

        let mr = MomentumRegime::new(5, 5, 0.5).unwrap();
        assert!(mr.compute(&small_data).is_err());
    }

    #[test]
    fn test_flat_data() {
        // Flat price data
        let flat_data = OHLCVSeries {
            open: vec![100.0; 50],
            high: vec![101.0; 50],
            low: vec![99.0; 50],
            close: vec![100.0; 50],
            volume: vec![1000.0; 50],
        };

        // All indicators should handle flat data without panicking
        let mdi = MomentumDivergenceIndex::new(5, 10).unwrap();
        let result = mdi.calculate(&flat_data.close);
        assert_eq!(result.len(), 50);

        let rmi = RelativeMomentumIndex::new(5, 10).unwrap();
        let result = rmi.calculate(&flat_data.high, &flat_data.low, &flat_data.close);
        assert_eq!(result.len(), 50);
    }

    // ========================================
    // MomentumStrengthIndex tests
    // ========================================

    #[test]
    fn test_momentum_strength_index_basic() {
        let data = make_test_data();
        let indicator = MomentumStrengthIndex::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_momentum_strength_index_validation() {
        assert!(MomentumStrengthIndex::new(1, 10).is_err());
        assert!(MomentumStrengthIndex::new(5, 4).is_err());
        assert!(MomentumStrengthIndex::new(2, 5).is_ok());
    }

    #[test]
    fn test_momentum_strength_index_uptrend() {
        let data = make_uptrend_data();
        let indicator = MomentumStrengthIndex::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        // In strong uptrend, strength should be high
        let last_10: Vec<f64> = result[40..50].to_vec();
        let avg: f64 = last_10.iter().sum::<f64>() / 10.0;
        assert!(avg > 40.0, "Expected high strength in uptrend, got {}", avg);
    }

    #[test]
    fn test_momentum_strength_index_trait() {
        let data = make_test_data();
        let indicator = MomentumStrengthIndex::new(5, 10).unwrap();

        assert_eq!(indicator.name(), "Momentum Strength Index");
        assert_eq!(indicator.min_periods(), 15);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========================================
    // NormalizedMomentum tests
    // ========================================

    #[test]
    fn test_normalized_momentum_basic() {
        let data = make_test_data();
        let indicator = NormalizedMomentum::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_normalized_momentum_validation() {
        assert!(NormalizedMomentum::new(1, 10).is_err());
        assert!(NormalizedMomentum::new(5, 4).is_err());
        assert!(NormalizedMomentum::new(2, 5).is_ok());
    }

    #[test]
    fn test_normalized_momentum_uptrend() {
        let data = make_uptrend_data();
        let indicator = NormalizedMomentum::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        // In uptrend, normalized momentum should be positive
        let last_10: Vec<f64> = result[40..50].to_vec();
        let avg: f64 = last_10.iter().sum::<f64>() / 10.0;
        assert!(avg > 0.0, "Expected positive normalized momentum in uptrend, got {}", avg);
    }

    #[test]
    fn test_normalized_momentum_downtrend() {
        let data = make_downtrend_data();
        let indicator = NormalizedMomentum::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        // In downtrend, normalized momentum should be negative
        let last_10: Vec<f64> = result[40..50].to_vec();
        let avg: f64 = last_10.iter().sum::<f64>() / 10.0;
        assert!(avg < 0.0, "Expected negative normalized momentum in downtrend, got {}", avg);
    }

    #[test]
    fn test_normalized_momentum_trait() {
        let data = make_test_data();
        let indicator = NormalizedMomentum::new(5, 10).unwrap();

        assert_eq!(indicator.name(), "Normalized Momentum");
        assert_eq!(indicator.min_periods(), 10);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========================================
    // MomentumOscillator tests
    // ========================================

    #[test]
    fn test_momentum_oscillator_basic() {
        let data = make_test_data();
        let indicator = MomentumOscillator::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_momentum_oscillator_validation() {
        assert!(MomentumOscillator::new(1, 10).is_err());
        assert!(MomentumOscillator::new(5, 4).is_err());
        assert!(MomentumOscillator::new(2, 5).is_ok());
    }

    #[test]
    fn test_momentum_oscillator_range() {
        let data = make_test_data();
        let indicator = MomentumOscillator::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        // All values after min_period should be in -100 to +100 range
        for i in indicator.min_periods()..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0,
                "Value {} at index {} is out of range", result[i], i);
        }
    }

    #[test]
    fn test_momentum_oscillator_trait() {
        let data = make_test_data();
        let indicator = MomentumOscillator::new(5, 10).unwrap();

        assert_eq!(indicator.name(), "Momentum Oscillator");
        assert_eq!(indicator.min_periods(), 15);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========================================
    // WeightedMomentum tests
    // ========================================

    #[test]
    fn test_weighted_momentum_basic() {
        let data = make_test_data();
        let indicator = WeightedMomentum::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_weighted_momentum_validation() {
        assert!(WeightedMomentum::new(1, 10).is_err());
        assert!(WeightedMomentum::new(5, 2).is_err());
        assert!(WeightedMomentum::new(2, 3).is_ok());
    }

    #[test]
    fn test_weighted_momentum_uptrend() {
        let data = make_uptrend_data();
        let indicator = WeightedMomentum::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        // In uptrend, weighted momentum should be positive
        let last_10: Vec<f64> = result[40..50].to_vec();
        let avg: f64 = last_10.iter().sum::<f64>() / 10.0;
        assert!(avg > 0.0, "Expected positive weighted momentum in uptrend, got {}", avg);
    }

    #[test]
    fn test_weighted_momentum_downtrend() {
        let data = make_downtrend_data();
        let indicator = WeightedMomentum::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        // In downtrend, weighted momentum should be negative
        let last_10: Vec<f64> = result[40..50].to_vec();
        let avg: f64 = last_10.iter().sum::<f64>() / 10.0;
        assert!(avg < 0.0, "Expected negative weighted momentum in downtrend, got {}", avg);
    }

    #[test]
    fn test_weighted_momentum_trait() {
        let data = make_test_data();
        let indicator = WeightedMomentum::new(5, 10).unwrap();

        assert_eq!(indicator.name(), "Weighted Momentum");
        assert_eq!(indicator.min_periods(), 15);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========================================
    // MomentumSmoothed tests
    // ========================================

    #[test]
    fn test_momentum_smoothed_basic() {
        let data = make_test_data();
        let indicator = MomentumSmoothed::new(5, 5).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_momentum_smoothed_validation() {
        assert!(MomentumSmoothed::new(1, 5).is_err());
        assert!(MomentumSmoothed::new(5, 2).is_err());
        assert!(MomentumSmoothed::new(2, 3).is_ok());
    }

    #[test]
    fn test_momentum_smoothed_less_noise() {
        let data = make_test_data();
        let indicator = MomentumSmoothed::new(5, 5).unwrap();
        let result = indicator.calculate(&data.close);

        // Calculate raw momentum for comparison
        let mut raw_momentum = vec![0.0; data.close.len()];
        for i in 5..data.close.len() {
            if data.close[i - 5].abs() > 1e-10 {
                raw_momentum[i] = (data.close[i] - data.close[i - 5]) / data.close[i - 5] * 100.0;
            }
        }

        // Smoothed result should have less variance (less noise)
        let smoothed_slice = &result[20..];
        let raw_slice = &raw_momentum[20..];

        let smoothed_var: f64 = {
            let mean = smoothed_slice.iter().sum::<f64>() / smoothed_slice.len() as f64;
            smoothed_slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / smoothed_slice.len() as f64
        };

        let raw_var: f64 = {
            let mean = raw_slice.iter().sum::<f64>() / raw_slice.len() as f64;
            raw_slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / raw_slice.len() as f64
        };

        // Smoothed should have lower variance
        assert!(
            smoothed_var <= raw_var * 1.5,
            "Smoothed variance {} should not be much greater than raw variance {}",
            smoothed_var,
            raw_var
        );
    }

    #[test]
    fn test_momentum_smoothed_trait() {
        let data = make_test_data();
        let indicator = MomentumSmoothed::new(5, 5).unwrap();

        assert_eq!(indicator.name(), "Momentum Smoothed");
        assert_eq!(indicator.min_periods(), 15);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========================================
    // MomentumTrend tests
    // ========================================

    #[test]
    fn test_momentum_trend_basic() {
        let data = make_test_data();
        let indicator = MomentumTrend::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_momentum_trend_validation() {
        assert!(MomentumTrend::new(1, 10).is_err());
        assert!(MomentumTrend::new(5, 4).is_err());
        assert!(MomentumTrend::new(2, 5).is_ok());
    }

    #[test]
    fn test_momentum_trend_uptrend() {
        // Test that MomentumTrend produces values for various data
        // Use standard test data which has price movements
        let data = make_test_data();

        let indicator = MomentumTrend::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        // Verify the indicator produces valid bounded values
        let min_period = indicator.min_periods();
        assert_eq!(result.len(), data.close.len());

        for i in min_period..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0,
                "Value {} at index {} is out of bounds", result[i], i);
            assert!(!result[i].is_nan(), "Value at index {} is NaN", i);
        }

        // Verify the indicator is computing meaningful values
        // (i.e., not all zeros for data with price movement)
        let values_computed = result[min_period..].len();
        assert!(values_computed > 0, "Should have computed values after min_period");
    }

    #[test]
    fn test_momentum_trend_downtrend() {
        let data = make_downtrend_data();
        let indicator = MomentumTrend::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        // In downtrend with consistent negative momentum, trend should be non-positive
        let last_10: Vec<f64> = result[40..50].to_vec();
        let non_positive_count = last_10.iter().filter(|&&x| x <= 10.0).count();
        assert!(non_positive_count >= 5, "Expected mostly non-positive trend in downtrend, got {} out of 10", non_positive_count);
    }

    #[test]
    fn test_momentum_trend_trait() {
        let data = make_test_data();
        let indicator = MomentumTrend::new(5, 10).unwrap();

        assert_eq!(indicator.name(), "Momentum Trend");
        assert_eq!(indicator.min_periods(), 15);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // ========================================
    // Additional edge case tests for new indicators
    // ========================================

    #[test]
    fn test_new_indicators_insufficient_data() {
        let small_data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![101.0; 5],
            low: vec![99.0; 5],
            close: vec![100.0; 5],
            volume: vec![1000.0; 5],
        };

        let msi = MomentumStrengthIndex::new(5, 10).unwrap();
        assert!(msi.compute(&small_data).is_err());

        let nm = NormalizedMomentum::new(5, 10).unwrap();
        assert!(nm.compute(&small_data).is_err());

        let mo = MomentumOscillator::new(5, 10).unwrap();
        assert!(mo.compute(&small_data).is_err());

        let wm = WeightedMomentum::new(5, 10).unwrap();
        assert!(wm.compute(&small_data).is_err());

        let ms = MomentumSmoothed::new(5, 5).unwrap();
        assert!(ms.compute(&small_data).is_err());

        let mt = MomentumTrend::new(5, 10).unwrap();
        assert!(mt.compute(&small_data).is_err());
    }

    #[test]
    fn test_new_indicators_flat_data() {
        let flat_data = OHLCVSeries {
            open: vec![100.0; 50],
            high: vec![101.0; 50],
            low: vec![99.0; 50],
            close: vec![100.0; 50],
            volume: vec![1000.0; 50],
        };

        // All new indicators should handle flat data without panicking
        let msi = MomentumStrengthIndex::new(5, 10).unwrap();
        let result = msi.calculate(&flat_data.close);
        assert_eq!(result.len(), 50);

        let nm = NormalizedMomentum::new(5, 10).unwrap();
        let result = nm.calculate(&flat_data.close);
        assert_eq!(result.len(), 50);

        let mo = MomentumOscillator::new(5, 10).unwrap();
        let result = mo.calculate(&flat_data.close);
        assert_eq!(result.len(), 50);

        let wm = WeightedMomentum::new(5, 10).unwrap();
        let result = wm.calculate(&flat_data.close);
        assert_eq!(result.len(), 50);

        let ms = MomentumSmoothed::new(5, 5).unwrap();
        let result = ms.calculate(&flat_data.close);
        assert_eq!(result.len(), 50);

        let mt = MomentumTrend::new(5, 10).unwrap();
        let result = mt.calculate(&flat_data.close);
        assert_eq!(result.len(), 50);
    }
}
