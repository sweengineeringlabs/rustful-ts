//! Advanced Composite Indicators implementation.
//!
//! Six composite indicators combining multiple technical analysis components.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

// ============================================================================
// 1. TrendVolatilityIndex
// ============================================================================

/// Trend Volatility Index output.
#[derive(Debug, Clone)]
pub struct TrendVolatilityIndexOutput {
    /// Combined trend-volatility index values (0-100 scale).
    pub index: Vec<f64>,
    /// Trend component (0-100).
    pub trend_component: Vec<f64>,
    /// Volatility component (0-100).
    pub volatility_component: Vec<f64>,
}

/// Trend Volatility Index configuration.
#[derive(Debug, Clone)]
pub struct TrendVolatilityIndexConfig {
    /// Period for trend calculation (default: 14).
    pub trend_period: usize,
    /// Period for volatility calculation (default: 14).
    pub volatility_period: usize,
    /// Weight for trend component (default: 0.6).
    pub trend_weight: f64,
    /// Weight for volatility component (default: 0.4).
    pub volatility_weight: f64,
}

impl Default for TrendVolatilityIndexConfig {
    fn default() -> Self {
        Self {
            trend_period: 14,
            volatility_period: 14,
            trend_weight: 0.6,
            volatility_weight: 0.4,
        }
    }
}

/// Trend Volatility Index.
///
/// Combines trend strength and volatility measures into a single composite indicator.
/// Higher values indicate strong trending conditions with increased volatility,
/// which often precedes significant price moves.
///
/// Formula:
/// - Trend Component: Normalized directional movement
/// - Volatility Component: Normalized ATR
/// - Index = trend_weight * trend + volatility_weight * volatility
#[derive(Debug, Clone)]
pub struct TrendVolatilityIndex {
    trend_period: usize,
    volatility_period: usize,
    trend_weight: f64,
    volatility_weight: f64,
}

impl TrendVolatilityIndex {
    /// Create a new TrendVolatilityIndex with the given configuration.
    pub fn new(config: TrendVolatilityIndexConfig) -> Result<Self> {
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.trend_weight < 0.0 || config.trend_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.volatility_weight < 0.0 || config.volatility_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(Self {
            trend_period: config.trend_period,
            volatility_period: config.volatility_period,
            trend_weight: config.trend_weight,
            volatility_weight: config.volatility_weight,
        })
    }

    /// Calculate the Trend Volatility Index values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> TrendVolatilityIndexOutput {
        let n = close.len();
        let mut index = vec![f64::NAN; n];
        let mut trend_component = vec![f64::NAN; n];
        let mut volatility_component = vec![f64::NAN; n];

        if n < 2 {
            return TrendVolatilityIndexOutput {
                index,
                trend_component,
                volatility_component,
            };
        }

        // Calculate True Range
        let mut tr = vec![0.0; n];
        tr[0] = high[0] - low[0];
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        // Calculate directional movement
        let mut plus_dm = vec![0.0; n];
        let mut minus_dm = vec![0.0; n];
        for i in 1..n {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];

            if up_move > down_move && up_move > 0.0 {
                plus_dm[i] = up_move;
            }
            if down_move > up_move && down_move > 0.0 {
                minus_dm[i] = down_move;
            }
        }

        // Calculate smoothed values
        let period = self.trend_period.max(self.volatility_period);
        let start_idx = period;

        if n <= start_idx {
            return TrendVolatilityIndexOutput {
                index,
                trend_component,
                volatility_component,
            };
        }

        // Calculate ATR (volatility)
        let mut atr = vec![f64::NAN; n];
        let mut sum_tr: f64 = tr[1..=self.volatility_period].iter().sum();
        atr[self.volatility_period] = sum_tr / self.volatility_period as f64;

        for i in (self.volatility_period + 1)..n {
            atr[i] = (atr[i - 1] * (self.volatility_period - 1) as f64 + tr[i])
                / self.volatility_period as f64;
        }

        // Calculate smoothed DM
        let mut smoothed_plus_dm = vec![f64::NAN; n];
        let mut smoothed_minus_dm = vec![f64::NAN; n];
        let mut smoothed_tr = vec![f64::NAN; n];

        let sum_plus: f64 = plus_dm[1..=self.trend_period].iter().sum();
        let sum_minus: f64 = minus_dm[1..=self.trend_period].iter().sum();
        sum_tr = tr[1..=self.trend_period].iter().sum();

        smoothed_plus_dm[self.trend_period] = sum_plus;
        smoothed_minus_dm[self.trend_period] = sum_minus;
        smoothed_tr[self.trend_period] = sum_tr;

        for i in (self.trend_period + 1)..n {
            smoothed_plus_dm[i] = smoothed_plus_dm[i - 1]
                - (smoothed_plus_dm[i - 1] / self.trend_period as f64)
                + plus_dm[i];
            smoothed_minus_dm[i] = smoothed_minus_dm[i - 1]
                - (smoothed_minus_dm[i - 1] / self.trend_period as f64)
                + minus_dm[i];
            smoothed_tr[i] =
                smoothed_tr[i - 1] - (smoothed_tr[i - 1] / self.trend_period as f64) + tr[i];
        }

        // Calculate trend and volatility components
        for i in start_idx..n {
            // Trend component: absolute difference of directional indicators
            if !smoothed_tr[i].is_nan() && smoothed_tr[i].abs() > 1e-10 {
                let plus_di = (smoothed_plus_dm[i] / smoothed_tr[i]) * 100.0;
                let minus_di = (smoothed_minus_dm[i] / smoothed_tr[i]) * 100.0;
                let di_diff = (plus_di - minus_di).abs();
                let di_sum = plus_di + minus_di;

                if di_sum > 0.0 {
                    // ADX-like calculation for trend strength
                    let dx = (di_diff / di_sum) * 100.0;
                    trend_component[i] = dx.clamp(0.0, 100.0);
                } else {
                    trend_component[i] = 0.0;
                }
            }

            // Volatility component: normalized ATR
            if !atr[i].is_nan() && close[i].abs() > 1e-10 {
                // ATR as percentage of price, scaled to 0-100
                let atr_percent = (atr[i] / close[i]) * 100.0;
                // Scale: 1% ATR = 50, 2% = 100
                volatility_component[i] = (atr_percent * 50.0).clamp(0.0, 100.0);
            }

            // Combined index
            if !trend_component[i].is_nan() && !volatility_component[i].is_nan() {
                index[i] = (self.trend_weight * trend_component[i]
                    + self.volatility_weight * volatility_component[i])
                    .clamp(0.0, 100.0);
            }
        }

        TrendVolatilityIndexOutput {
            index,
            trend_component,
            volatility_component,
        }
    }
}

impl TechnicalIndicator for TrendVolatilityIndex {
    fn name(&self) -> &str {
        "TrendVolatilityIndex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.index,
            result.trend_component,
            result.volatility_component,
        ))
    }

    fn min_periods(&self) -> usize {
        self.trend_period.max(self.volatility_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 2. MomentumQualityScore
// ============================================================================

/// Momentum Quality Score output.
#[derive(Debug, Clone)]
pub struct MomentumQualityScoreOutput {
    /// Quality-adjusted momentum score (-100 to 100).
    pub score: Vec<f64>,
    /// Raw momentum component.
    pub raw_momentum: Vec<f64>,
    /// Quality factor (0-1).
    pub quality_factor: Vec<f64>,
}

/// Momentum Quality Score configuration.
#[derive(Debug, Clone)]
pub struct MomentumQualityScoreConfig {
    /// Momentum lookback period (default: 14).
    pub momentum_period: usize,
    /// Quality assessment period (default: 20).
    pub quality_period: usize,
    /// Smoothing period (default: 3).
    pub smoothing_period: usize,
}

impl Default for MomentumQualityScoreConfig {
    fn default() -> Self {
        Self {
            momentum_period: 14,
            quality_period: 20,
            smoothing_period: 3,
        }
    }
}

/// Momentum Quality Score.
///
/// Evaluates momentum quality by considering:
/// - Consistency of price movement direction
/// - Smoothness of the trend (lower volatility = higher quality)
/// - Acceleration/deceleration of momentum
///
/// Higher quality momentum is more likely to continue.
#[derive(Debug, Clone)]
pub struct MomentumQualityScore {
    momentum_period: usize,
    quality_period: usize,
    smoothing_period: usize,
}

impl MomentumQualityScore {
    /// Create a new MomentumQualityScore with the given configuration.
    pub fn new(config: MomentumQualityScoreConfig) -> Result<Self> {
        if config.momentum_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.quality_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "quality_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            momentum_period: config.momentum_period,
            quality_period: config.quality_period,
            smoothing_period: config.smoothing_period,
        })
    }

    /// Calculate the Momentum Quality Score values.
    pub fn calculate(&self, close: &[f64]) -> MomentumQualityScoreOutput {
        let n = close.len();
        let mut score = vec![f64::NAN; n];
        let mut raw_momentum = vec![f64::NAN; n];
        let mut quality_factor = vec![f64::NAN; n];

        let min_required = self.momentum_period.max(self.quality_period);
        if n <= min_required {
            return MomentumQualityScoreOutput {
                score,
                raw_momentum,
                quality_factor,
            };
        }

        // Calculate raw momentum (rate of change)
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                raw_momentum[i] =
                    ((close[i] - close[i - self.momentum_period]) / close[i - self.momentum_period])
                        * 100.0;
            }
        }

        // Calculate quality factor
        for i in self.quality_period..n {
            let window = &close[(i - self.quality_period + 1)..=i];

            // Calculate direction consistency
            let mut up_count = 0;
            let mut down_count = 0;
            for j in 1..window.len() {
                if window[j] > window[j - 1] {
                    up_count += 1;
                } else if window[j] < window[j - 1] {
                    down_count += 1;
                }
            }
            let total_moves = (up_count + down_count).max(1);
            let consistency = (up_count.max(down_count) as f64) / (total_moves as f64);

            // Calculate smoothness (inverse of relative volatility)
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            if mean.abs() > 1e-10 {
                let variance: f64 =
                    window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
                let std_dev = variance.sqrt();
                let cv = std_dev / mean.abs(); // Coefficient of variation
                let smoothness = (1.0 - cv.min(1.0)).max(0.0);

                // Quality = consistency * smoothness
                quality_factor[i] = (consistency * smoothness).clamp(0.0, 1.0);
            } else {
                quality_factor[i] = 0.0;
            }
        }

        // Calculate quality-adjusted score
        for i in min_required..n {
            if !raw_momentum[i].is_nan() && !quality_factor[i].is_nan() {
                // Apply quality factor to momentum
                // High quality amplifies momentum, low quality dampens it
                let quality_multiplier = 0.5 + quality_factor[i]; // Range: 0.5 to 1.5
                score[i] = (raw_momentum[i] * quality_multiplier).clamp(-100.0, 100.0);
            }
        }

        // Apply smoothing
        if self.smoothing_period > 1 {
            let smoothed = self.ema_smooth(&score, self.smoothing_period);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    score[i] = smoothed[i];
                }
            }
        }

        MomentumQualityScoreOutput {
            score,
            raw_momentum,
            quality_factor,
        }
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        // Find first valid value
        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumQualityScore {
    fn name(&self) -> &str {
        "MomentumQualityScore"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(
            result.score,
            result.raw_momentum,
            result.quality_factor,
        ))
    }

    fn min_periods(&self) -> usize {
        self.momentum_period.max(self.quality_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 3. MarketPhaseIndicator
// ============================================================================

/// Market phase types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketPhase {
    /// Strong trending market.
    Trending,
    /// Range-bound market.
    Ranging,
    /// High volatility market.
    Volatile,
    /// Quiet/consolidating market.
    Quiet,
}

impl MarketPhase {
    /// Convert to numeric value for output.
    pub fn to_numeric(&self) -> f64 {
        match self {
            MarketPhase::Trending => 1.0,
            MarketPhase::Ranging => 0.0,
            MarketPhase::Volatile => 2.0,
            MarketPhase::Quiet => -1.0,
        }
    }
}

/// Market Phase Indicator output.
#[derive(Debug, Clone)]
pub struct MarketPhaseIndicatorOutput {
    /// Current market phase.
    pub phase: Vec<MarketPhase>,
    /// Phase numeric values.
    pub phase_value: Vec<f64>,
    /// Trend score (0-100).
    pub trend_score: Vec<f64>,
    /// Volatility score (0-100).
    pub volatility_score: Vec<f64>,
}

/// Market Phase Indicator configuration.
#[derive(Debug, Clone)]
pub struct MarketPhaseIndicatorConfig {
    /// Period for trend measurement (default: 20).
    pub trend_period: usize,
    /// Period for volatility measurement (default: 14).
    pub volatility_period: usize,
    /// Threshold for trending phase (default: 25.0).
    pub trend_threshold: f64,
    /// Threshold for volatile phase (default: 70.0).
    pub volatility_threshold: f64,
}

impl Default for MarketPhaseIndicatorConfig {
    fn default() -> Self {
        Self {
            trend_period: 20,
            volatility_period: 14,
            trend_threshold: 25.0,
            volatility_threshold: 70.0,
        }
    }
}

/// Market Phase Indicator.
///
/// Identifies the current market phase based on trend strength and volatility:
/// - Trending: Strong directional movement (high trend score)
/// - Ranging: Low trend strength, moderate volatility
/// - Volatile: High volatility with uncertain direction
/// - Quiet: Low trend and low volatility (consolidation)
#[derive(Debug, Clone)]
pub struct MarketPhaseIndicator {
    trend_period: usize,
    volatility_period: usize,
    trend_threshold: f64,
    volatility_threshold: f64,
}

impl MarketPhaseIndicator {
    /// Create a new MarketPhaseIndicator with the given configuration.
    pub fn new(config: MarketPhaseIndicatorConfig) -> Result<Self> {
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.trend_threshold <= 0.0 || config.trend_threshold >= 100.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_threshold".to_string(),
                reason: "must be between 0.0 and 100.0".to_string(),
            });
        }
        if config.volatility_threshold <= 0.0 || config.volatility_threshold >= 100.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_threshold".to_string(),
                reason: "must be between 0.0 and 100.0".to_string(),
            });
        }

        Ok(Self {
            trend_period: config.trend_period,
            volatility_period: config.volatility_period,
            trend_threshold: config.trend_threshold,
            volatility_threshold: config.volatility_threshold,
        })
    }

    /// Calculate the Market Phase Indicator values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> MarketPhaseIndicatorOutput {
        let n = close.len();
        let mut phase = vec![MarketPhase::Quiet; n];
        let mut phase_value = vec![f64::NAN; n];
        let mut trend_score = vec![f64::NAN; n];
        let mut volatility_score = vec![f64::NAN; n];

        let min_required = self.trend_period.max(self.volatility_period);
        if n <= min_required {
            return MarketPhaseIndicatorOutput {
                phase,
                phase_value,
                trend_score,
                volatility_score,
            };
        }

        // Calculate trend score using efficiency ratio
        for i in self.trend_period..n {
            let net_change = (close[i] - close[i - self.trend_period]).abs();
            let mut sum_changes = 0.0;
            for j in (i - self.trend_period + 1)..=i {
                sum_changes += (close[j] - close[j - 1]).abs();
            }

            if sum_changes > 1e-10 {
                // Efficiency ratio scaled to 0-100
                trend_score[i] = (net_change / sum_changes * 100.0).clamp(0.0, 100.0);
            } else {
                trend_score[i] = 0.0;
            }
        }

        // Calculate volatility score
        let mut atr = vec![f64::NAN; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i >= self.volatility_period {
                let window_start = i - self.volatility_period + 1;
                let mut tr_sum = 0.0;
                for j in window_start..=i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j.saturating_sub(1)]).abs())
                        .max((low[j] - close[j.saturating_sub(1)]).abs());
                    tr_sum += tr_j;
                }
                atr[i] = tr_sum / self.volatility_period as f64;
            }
        }

        // Calculate volatility percentile for volatility score
        for i in min_required..n {
            if !atr[i].is_nan() && close[i].abs() > 1e-10 {
                // ATR as percentage of price
                let atr_percent = (atr[i] / close[i]) * 100.0;
                // Scale: typical range is 0.5% to 3%
                volatility_score[i] = (atr_percent * 33.3).clamp(0.0, 100.0);
            }
        }

        // Determine phase
        for i in min_required..n {
            if trend_score[i].is_nan() || volatility_score[i].is_nan() {
                continue;
            }

            let is_trending = trend_score[i] >= self.trend_threshold;
            let is_volatile = volatility_score[i] >= self.volatility_threshold;

            phase[i] = match (is_trending, is_volatile) {
                (true, _) => MarketPhase::Trending,
                (false, true) => MarketPhase::Volatile,
                (false, false) if trend_score[i] < self.trend_threshold / 2.0 => MarketPhase::Quiet,
                (false, false) => MarketPhase::Ranging,
            };

            phase_value[i] = phase[i].to_numeric();
        }

        MarketPhaseIndicatorOutput {
            phase,
            phase_value,
            trend_score,
            volatility_score,
        }
    }
}

impl TechnicalIndicator for MarketPhaseIndicator {
    fn name(&self) -> &str {
        "MarketPhaseIndicator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.phase_value,
            result.trend_score,
            result.volatility_score,
        ))
    }

    fn min_periods(&self) -> usize {
        self.trend_period.max(self.volatility_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 4. PriceTrendStrength
// ============================================================================

/// Price Trend Strength output.
#[derive(Debug, Clone)]
pub struct PriceTrendStrengthOutput {
    /// Combined strength value (0-100).
    pub strength: Vec<f64>,
    /// Price momentum component.
    pub price_momentum: Vec<f64>,
    /// Trend persistence component.
    pub trend_persistence: Vec<f64>,
}

/// Price Trend Strength configuration.
#[derive(Debug, Clone)]
pub struct PriceTrendStrengthConfig {
    /// Short-term period (default: 5).
    pub short_period: usize,
    /// Medium-term period (default: 10).
    pub medium_period: usize,
    /// Long-term period (default: 20).
    pub long_period: usize,
}

impl Default for PriceTrendStrengthConfig {
    fn default() -> Self {
        Self {
            short_period: 5,
            medium_period: 10,
            long_period: 20,
        }
    }
}

/// Price Trend Strength.
///
/// Measures the combined strength of price movement and trend persistence
/// across multiple timeframes.
///
/// Components:
/// - Price momentum: Rate of change across timeframes
/// - Trend persistence: Consistency of direction across timeframes
#[derive(Debug, Clone)]
pub struct PriceTrendStrength {
    short_period: usize,
    medium_period: usize,
    long_period: usize,
}

impl PriceTrendStrength {
    /// Create a new PriceTrendStrength with the given configuration.
    pub fn new(config: PriceTrendStrengthConfig) -> Result<Self> {
        if config.short_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.medium_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.long_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.short_period >= config.medium_period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than medium_period".to_string(),
            });
        }
        if config.medium_period >= config.long_period {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be less than long_period".to_string(),
            });
        }

        Ok(Self {
            short_period: config.short_period,
            medium_period: config.medium_period,
            long_period: config.long_period,
        })
    }

    /// Calculate the Price Trend Strength values.
    pub fn calculate(&self, close: &[f64]) -> PriceTrendStrengthOutput {
        let n = close.len();
        let mut strength = vec![f64::NAN; n];
        let mut price_momentum = vec![f64::NAN; n];
        let mut trend_persistence = vec![f64::NAN; n];

        if n <= self.long_period {
            return PriceTrendStrengthOutput {
                strength,
                price_momentum,
                trend_persistence,
            };
        }

        // Calculate EMAs
        let ema_short = self.calculate_ema(close, self.short_period);
        let ema_medium = self.calculate_ema(close, self.medium_period);
        let ema_long = self.calculate_ema(close, self.long_period);

        for i in self.long_period..n {
            if ema_short[i].is_nan() || ema_medium[i].is_nan() || ema_long[i].is_nan() {
                continue;
            }

            // Price momentum: weighted average of ROCs
            let roc_short = if close[i - self.short_period].abs() > 1e-10 {
                ((close[i] - close[i - self.short_period]) / close[i - self.short_period]) * 100.0
            } else {
                0.0
            };
            let roc_medium = if close[i - self.medium_period].abs() > 1e-10 {
                ((close[i] - close[i - self.medium_period]) / close[i - self.medium_period]) * 100.0
            } else {
                0.0
            };
            let roc_long = if close[i - self.long_period].abs() > 1e-10 {
                ((close[i] - close[i - self.long_period]) / close[i - self.long_period]) * 100.0
            } else {
                0.0
            };

            // Weighted momentum (short-term gets more weight)
            let weighted_momentum = (roc_short * 0.5 + roc_medium * 0.3 + roc_long * 0.2).abs();
            price_momentum[i] = (weighted_momentum * 10.0).clamp(0.0, 100.0);

            // Trend persistence: EMA alignment
            let short_above_medium = ema_short[i] > ema_medium[i];
            let medium_above_long = ema_medium[i] > ema_long[i];
            let price_above_short = close[i] > ema_short[i];

            let alignment_score = if short_above_medium && medium_above_long && price_above_short {
                100.0 // Perfect bullish alignment
            } else if !short_above_medium && !medium_above_long && !price_above_short {
                100.0 // Perfect bearish alignment
            } else if (short_above_medium && medium_above_long)
                || (!short_above_medium && !medium_above_long)
            {
                66.0 // Partial alignment
            } else {
                33.0 // Mixed signals
            };

            trend_persistence[i] = alignment_score;

            // Combined strength
            strength[i] = (price_momentum[i] * 0.4 + trend_persistence[i] * 0.6).clamp(0.0, 100.0);
        }

        PriceTrendStrengthOutput {
            strength,
            price_momentum,
            trend_persistence,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            // Initialize with SMA
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            // Calculate EMA
            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }
}

impl TechnicalIndicator for PriceTrendStrength {
    fn name(&self) -> &str {
        "PriceTrendStrength"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(
            result.strength,
            result.price_momentum,
            result.trend_persistence,
        ))
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 5. AdaptiveMarketIndicator
// ============================================================================

/// Adaptive Market Indicator output.
#[derive(Debug, Clone)]
pub struct AdaptiveMarketIndicatorOutput {
    /// Adaptive indicator value (-100 to 100).
    pub value: Vec<f64>,
    /// Adaptation factor (0-1).
    pub adaptation_factor: Vec<f64>,
    /// Market efficiency measure.
    pub efficiency: Vec<f64>,
}

/// Adaptive Market Indicator configuration.
#[derive(Debug, Clone)]
pub struct AdaptiveMarketIndicatorConfig {
    /// Base period (default: 10).
    pub base_period: usize,
    /// Maximum adaptation period (default: 30).
    pub max_period: usize,
    /// Efficiency lookback (default: 10).
    pub efficiency_period: usize,
}

impl Default for AdaptiveMarketIndicatorConfig {
    fn default() -> Self {
        Self {
            base_period: 10,
            max_period: 30,
            efficiency_period: 10,
        }
    }
}

/// Adaptive Market Indicator.
///
/// A multi-factor indicator that adapts its sensitivity based on market conditions.
/// In trending markets, it becomes more responsive; in ranging markets, it becomes
/// smoother to reduce whipsaws.
///
/// Components:
/// - Efficiency Ratio: Measures how efficiently price moves
/// - Adaptive smoothing: Adjusts responsiveness based on efficiency
/// - Multi-factor signal: Combines trend and momentum adaptively
#[derive(Debug, Clone)]
pub struct AdaptiveMarketIndicator {
    base_period: usize,
    max_period: usize,
    efficiency_period: usize,
}

impl AdaptiveMarketIndicator {
    /// Create a new AdaptiveMarketIndicator with the given configuration.
    pub fn new(config: AdaptiveMarketIndicatorConfig) -> Result<Self> {
        if config.base_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.max_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.efficiency_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "efficiency_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.base_period >= config.max_period {
            return Err(IndicatorError::InvalidParameter {
                name: "base_period".to_string(),
                reason: "must be less than max_period".to_string(),
            });
        }

        Ok(Self {
            base_period: config.base_period,
            max_period: config.max_period,
            efficiency_period: config.efficiency_period,
        })
    }

    /// Calculate the Adaptive Market Indicator values.
    pub fn calculate(&self, close: &[f64]) -> AdaptiveMarketIndicatorOutput {
        let n = close.len();
        let mut value = vec![f64::NAN; n];
        let mut adaptation_factor = vec![f64::NAN; n];
        let mut efficiency = vec![f64::NAN; n];

        let min_required = self.max_period.max(self.efficiency_period);
        if n <= min_required {
            return AdaptiveMarketIndicatorOutput {
                value,
                adaptation_factor,
                efficiency,
            };
        }

        // Calculate efficiency ratio
        for i in self.efficiency_period..n {
            let net_change = (close[i] - close[i - self.efficiency_period]).abs();
            let mut sum_changes = 0.0;
            for j in (i - self.efficiency_period + 1)..=i {
                sum_changes += (close[j] - close[j - 1]).abs();
            }

            if sum_changes > 1e-10 {
                efficiency[i] = (net_change / sum_changes).clamp(0.0, 1.0);
            } else {
                efficiency[i] = 0.0;
            }
        }

        // Calculate adaptive values
        let fast_sc = 2.0 / (self.base_period as f64 + 1.0);
        let slow_sc = 2.0 / (self.max_period as f64 + 1.0);

        let mut adaptive_ma = vec![f64::NAN; n];

        // Initialize
        if n > min_required {
            adaptive_ma[min_required] = close[min_required];
        }

        for i in (min_required + 1)..n {
            if !efficiency[i].is_nan() && !adaptive_ma[i - 1].is_nan() {
                // Adaptation factor based on efficiency
                let er = efficiency[i];
                adaptation_factor[i] = er;

                // Smoothing constant adapts to market conditions
                let sc = (er * (fast_sc - slow_sc) + slow_sc).powi(2);

                // Adaptive moving average
                adaptive_ma[i] = adaptive_ma[i - 1] + sc * (close[i] - adaptive_ma[i - 1]);

                // Calculate indicator value as deviation from adaptive MA
                if adaptive_ma[i].abs() > 1e-10 {
                    let deviation = (close[i] - adaptive_ma[i]) / adaptive_ma[i] * 100.0;
                    // Scale by efficiency - more efficient markets get stronger signals
                    value[i] = (deviation * (1.0 + er)).clamp(-100.0, 100.0);
                }
            }
        }

        AdaptiveMarketIndicatorOutput {
            value,
            adaptation_factor,
            efficiency,
        }
    }
}

impl TechnicalIndicator for AdaptiveMarketIndicator {
    fn name(&self) -> &str {
        "AdaptiveMarketIndicator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(
            result.value,
            result.adaptation_factor,
            result.efficiency,
        ))
    }

    fn min_periods(&self) -> usize {
        self.max_period.max(self.efficiency_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 6. CompositeSignalStrength
// ============================================================================

/// Composite Signal Strength output.
#[derive(Debug, Clone)]
pub struct CompositeSignalStrengthOutput {
    /// Overall signal strength (-100 to 100).
    pub strength: Vec<f64>,
    /// Trend signal component.
    pub trend_signal: Vec<f64>,
    /// Momentum signal component.
    pub momentum_signal: Vec<f64>,
    /// Volatility signal component.
    pub volatility_signal: Vec<f64>,
}

/// Composite Signal Strength configuration.
#[derive(Debug, Clone)]
pub struct CompositeSignalStrengthConfig {
    /// Trend period (default: 20).
    pub trend_period: usize,
    /// Momentum period (default: 14).
    pub momentum_period: usize,
    /// Volatility period (default: 14).
    pub volatility_period: usize,
    /// Weight for trend signal (default: 0.4).
    pub trend_weight: f64,
    /// Weight for momentum signal (default: 0.35).
    pub momentum_weight: f64,
    /// Weight for volatility signal (default: 0.25).
    pub volatility_weight: f64,
}

impl Default for CompositeSignalStrengthConfig {
    fn default() -> Self {
        Self {
            trend_period: 20,
            momentum_period: 14,
            volatility_period: 14,
            trend_weight: 0.4,
            momentum_weight: 0.35,
            volatility_weight: 0.25,
        }
    }
}

/// Composite Signal Strength.
///
/// Combines multiple signal types into a unified strength measure:
/// - Trend Signal: Direction and strength of price trend
/// - Momentum Signal: Rate and quality of price change
/// - Volatility Signal: Market activity and breakout potential
///
/// Positive values indicate bullish conditions, negative bearish.
#[derive(Debug, Clone)]
pub struct CompositeSignalStrength {
    trend_period: usize,
    momentum_period: usize,
    volatility_period: usize,
    trend_weight: f64,
    momentum_weight: f64,
    volatility_weight: f64,
}

impl CompositeSignalStrength {
    /// Create a new CompositeSignalStrength with the given configuration.
    pub fn new(config: CompositeSignalStrengthConfig) -> Result<Self> {
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.momentum_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.trend_weight < 0.0 || config.trend_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.momentum_weight < 0.0 || config.momentum_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.volatility_weight < 0.0 || config.volatility_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(Self {
            trend_period: config.trend_period,
            momentum_period: config.momentum_period,
            volatility_period: config.volatility_period,
            trend_weight: config.trend_weight,
            momentum_weight: config.momentum_weight,
            volatility_weight: config.volatility_weight,
        })
    }

    /// Calculate the Composite Signal Strength values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> CompositeSignalStrengthOutput {
        let n = close.len();
        let mut strength = vec![f64::NAN; n];
        let mut trend_signal = vec![f64::NAN; n];
        let mut momentum_signal = vec![f64::NAN; n];
        let mut volatility_signal = vec![f64::NAN; n];

        let max_period = self
            .trend_period
            .max(self.momentum_period)
            .max(self.volatility_period);
        if n <= max_period {
            return CompositeSignalStrengthOutput {
                strength,
                trend_signal,
                momentum_signal,
                volatility_signal,
            };
        }

        // Calculate EMA for trend
        let ema = self.calculate_ema(close, self.trend_period);

        // Calculate trend signal
        for i in self.trend_period..n {
            if !ema[i].is_nan() && ema[i].abs() > 1e-10 {
                // Trend signal: deviation from EMA as percentage
                let deviation = (close[i] - ema[i]) / ema[i] * 100.0;
                // Scale to -100 to 100
                trend_signal[i] = (deviation * 10.0).clamp(-100.0, 100.0);
            }
        }

        // Calculate momentum signal (RSI-like)
        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let change = close[i] - close[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Smoothed averages
        for i in self.momentum_period..n {
            let avg_gain: f64 =
                gains[(i - self.momentum_period + 1)..=i].iter().sum::<f64>() / self.momentum_period as f64;
            let avg_loss: f64 =
                losses[(i - self.momentum_period + 1)..=i].iter().sum::<f64>() / self.momentum_period as f64;

            if avg_loss > 1e-10 {
                let rs = avg_gain / avg_loss;
                let rsi = 100.0 - (100.0 / (1.0 + rs));
                // Convert RSI (0-100) to signal (-100 to 100)
                momentum_signal[i] = (rsi - 50.0) * 2.0;
            } else if avg_gain > 0.0 {
                momentum_signal[i] = 100.0;
            } else {
                momentum_signal[i] = 0.0;
            }
        }

        // Calculate volatility signal
        let mut atr = vec![f64::NAN; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i >= self.volatility_period {
                let window_start = i - self.volatility_period + 1;
                let mut tr_sum = 0.0;
                for j in window_start..=i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j.saturating_sub(1)]).abs())
                        .max((low[j] - close[j.saturating_sub(1)]).abs());
                    tr_sum += tr_j;
                }
                atr[i] = tr_sum / self.volatility_period as f64;
            }
        }

        // Calculate volatility percentile and direction
        for i in max_period..n {
            if !atr[i].is_nan() && close[i].abs() > 1e-10 {
                let atr_percent = (atr[i] / close[i]) * 100.0;
                // High volatility in direction of trend = stronger signal
                let vol_magnitude = (atr_percent * 20.0).clamp(0.0, 100.0);

                // Direction based on recent price action
                let direction = if close[i] > close[i - 1] {
                    1.0
                } else if close[i] < close[i - 1] {
                    -1.0
                } else {
                    0.0
                };

                volatility_signal[i] = vol_magnitude * direction;
            }
        }

        // Combine signals
        for i in max_period..n {
            if !trend_signal[i].is_nan()
                && !momentum_signal[i].is_nan()
                && !volatility_signal[i].is_nan()
            {
                strength[i] = (self.trend_weight * trend_signal[i]
                    + self.momentum_weight * momentum_signal[i]
                    + self.volatility_weight * volatility_signal[i])
                    .clamp(-100.0, 100.0);
            }
        }

        CompositeSignalStrengthOutput {
            strength,
            trend_signal,
            momentum_signal,
            volatility_signal,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }
}

impl TechnicalIndicator for CompositeSignalStrength {
    fn name(&self) -> &str {
        "CompositeSignalStrength"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.strength,
            result.trend_signal,
            result.momentum_signal,
        ))
    }

    fn min_periods(&self) -> usize {
        self.trend_period
            .max(self.momentum_period)
            .max(self.volatility_period)
            + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 7. AdaptiveCompositeScore
// ============================================================================

/// Adaptive Composite Score output.
#[derive(Debug, Clone)]
pub struct AdaptiveCompositeScoreOutput {
    /// Adaptive composite score (-100 to 100).
    pub score: Vec<f64>,
    /// Regime factor (0-1).
    pub regime_factor: Vec<f64>,
    /// Base score before adaptation.
    pub base_score: Vec<f64>,
}

/// Adaptive Composite Score configuration.
#[derive(Debug, Clone)]
pub struct AdaptiveCompositeScoreConfig {
    /// Period for regime detection (default: 20).
    pub regime_period: usize,
    /// Period for score calculation (default: 14).
    pub score_period: usize,
    /// Smoothing period (default: 3).
    pub smoothing_period: usize,
}

impl Default for AdaptiveCompositeScoreConfig {
    fn default() -> Self {
        Self {
            regime_period: 20,
            score_period: 14,
            smoothing_period: 3,
        }
    }
}

/// Adaptive Composite Score.
///
/// A composite score that adapts to the current market regime.
/// In trending regimes, it emphasizes momentum; in ranging regimes,
/// it emphasizes mean reversion signals.
///
/// Components:
/// - Regime detection using efficiency ratio
/// - Base score from multiple factors
/// - Adaptive weighting based on regime
#[derive(Debug, Clone)]
pub struct AdaptiveCompositeScore {
    regime_period: usize,
    score_period: usize,
    smoothing_period: usize,
}

impl AdaptiveCompositeScore {
    /// Create a new AdaptiveCompositeScore with the given configuration.
    pub fn new(config: AdaptiveCompositeScoreConfig) -> Result<Self> {
        if config.regime_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "regime_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.score_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "score_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            regime_period: config.regime_period,
            score_period: config.score_period,
            smoothing_period: config.smoothing_period,
        })
    }

    /// Calculate the Adaptive Composite Score values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> AdaptiveCompositeScoreOutput {
        let n = close.len();
        let mut score = vec![f64::NAN; n];
        let mut regime_factor = vec![f64::NAN; n];
        let mut base_score = vec![f64::NAN; n];

        let min_required = self.regime_period.max(self.score_period);
        if n <= min_required {
            return AdaptiveCompositeScoreOutput {
                score,
                regime_factor,
                base_score,
            };
        }

        // Calculate efficiency ratio for regime detection
        for i in self.regime_period..n {
            let net_change = (close[i] - close[i - self.regime_period]).abs();
            let mut sum_changes = 0.0;
            for j in (i - self.regime_period + 1)..=i {
                sum_changes += (close[j] - close[j - 1]).abs();
            }

            if sum_changes > 1e-10 {
                regime_factor[i] = (net_change / sum_changes).clamp(0.0, 1.0);
            } else {
                regime_factor[i] = 0.0;
            }
        }

        // Calculate base score components
        for i in self.score_period..n {
            // Momentum component (ROC)
            let momentum = if close[i - self.score_period].abs() > 1e-10 {
                ((close[i] - close[i - self.score_period]) / close[i - self.score_period]) * 100.0
            } else {
                0.0
            };

            // Mean reversion component (deviation from SMA)
            let sma: f64 = close[(i - self.score_period + 1)..=i].iter().sum::<f64>()
                / self.score_period as f64;
            let mean_reversion = if sma.abs() > 1e-10 {
                ((close[i] - sma) / sma) * -100.0 // Negative = buy when below
            } else {
                0.0
            };

            // Volatility component
            let atr = self.calculate_atr(high, low, close, i, self.score_period);
            let vol_factor = if close[i].abs() > 1e-10 && !atr.is_nan() {
                (atr / close[i] * 100.0).clamp(0.0, 10.0) / 10.0
            } else {
                0.5
            };

            // Base score is weighted combination
            base_score[i] = (momentum * 0.5 + mean_reversion * 0.5).clamp(-100.0, 100.0);

            // Apply regime adaptation
            if !regime_factor[i].is_nan() {
                let regime = regime_factor[i];
                // High regime factor = trending, favor momentum
                // Low regime factor = ranging, favor mean reversion
                let momentum_weight = regime;
                let reversion_weight = 1.0 - regime;

                let adapted = momentum * momentum_weight + mean_reversion * reversion_weight;
                // Scale by volatility (higher vol = stronger signals)
                score[i] = (adapted * (0.5 + vol_factor)).clamp(-100.0, 100.0);
            }
        }

        // Apply smoothing
        if self.smoothing_period > 1 {
            let smoothed = self.ema_smooth(&score, self.smoothing_period);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    score[i] = smoothed[i];
                }
            }
        }

        AdaptiveCompositeScoreOutput {
            score,
            regime_factor,
            base_score,
        }
    }

    /// Calculate ATR for a specific index.
    fn calculate_atr(&self, high: &[f64], low: &[f64], close: &[f64], idx: usize, period: usize) -> f64 {
        if idx < period {
            return f64::NAN;
        }

        let mut sum = 0.0;
        for i in (idx - period + 1)..=idx {
            let tr = if i == 0 {
                high[i] - low[i]
            } else {
                (high[i] - low[i])
                    .max((high[i] - close[i - 1]).abs())
                    .max((low[i] - close[i - 1]).abs())
            };
            sum += tr;
        }
        sum / period as f64
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for AdaptiveCompositeScore {
    fn name(&self) -> &str {
        "AdaptiveCompositeScore"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.score,
            result.regime_factor,
            result.base_score,
        ))
    }

    fn min_periods(&self) -> usize {
        self.regime_period.max(self.score_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 8. MultiFactorMomentum
// ============================================================================

/// Multi-Factor Momentum output.
#[derive(Debug, Clone)]
pub struct MultiFactorMomentumOutput {
    /// Combined multi-factor momentum (-100 to 100).
    pub momentum: Vec<f64>,
    /// Price factor component.
    pub price_factor: Vec<f64>,
    /// Volume factor component.
    pub volume_factor: Vec<f64>,
}

/// Multi-Factor Momentum configuration.
#[derive(Debug, Clone)]
pub struct MultiFactorMomentumConfig {
    /// Short period for momentum (default: 5).
    pub short_period: usize,
    /// Medium period for momentum (default: 10).
    pub medium_period: usize,
    /// Long period for momentum (default: 20).
    pub long_period: usize,
    /// Volume period (default: 10).
    pub volume_period: usize,
}

impl Default for MultiFactorMomentumConfig {
    fn default() -> Self {
        Self {
            short_period: 5,
            medium_period: 10,
            long_period: 20,
            volume_period: 10,
        }
    }
}

/// Multi-Factor Momentum.
///
/// Calculates momentum from multiple factors including:
/// - Price momentum across multiple timeframes
/// - Volume confirmation
/// - Volatility-adjusted momentum
///
/// Higher values indicate strong bullish momentum with volume confirmation.
#[derive(Debug, Clone)]
pub struct MultiFactorMomentum {
    short_period: usize,
    medium_period: usize,
    long_period: usize,
    volume_period: usize,
}

impl MultiFactorMomentum {
    /// Create a new MultiFactorMomentum with the given configuration.
    pub fn new(config: MultiFactorMomentumConfig) -> Result<Self> {
        if config.short_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.medium_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.long_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volume_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            short_period: config.short_period,
            medium_period: config.medium_period,
            long_period: config.long_period,
            volume_period: config.volume_period,
        })
    }

    /// Calculate the Multi-Factor Momentum values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> MultiFactorMomentumOutput {
        let n = close.len();
        let mut momentum = vec![f64::NAN; n];
        let mut price_factor = vec![f64::NAN; n];
        let mut volume_factor = vec![f64::NAN; n];

        let min_required = self.long_period.max(self.volume_period);
        if n <= min_required {
            return MultiFactorMomentumOutput {
                momentum,
                price_factor,
                volume_factor,
            };
        }

        // Calculate price momentum across timeframes
        for i in self.long_period..n {
            // Short-term ROC
            let short_roc = if close[i - self.short_period].abs() > 1e-10 {
                ((close[i] - close[i - self.short_period]) / close[i - self.short_period]) * 100.0
            } else {
                0.0
            };

            // Medium-term ROC
            let medium_roc = if close[i - self.medium_period].abs() > 1e-10 {
                ((close[i] - close[i - self.medium_period]) / close[i - self.medium_period]) * 100.0
            } else {
                0.0
            };

            // Long-term ROC
            let long_roc = if close[i - self.long_period].abs() > 1e-10 {
                ((close[i] - close[i - self.long_period]) / close[i - self.long_period]) * 100.0
            } else {
                0.0
            };

            // Weighted price factor (short-term gets more weight)
            price_factor[i] = (short_roc * 0.5 + medium_roc * 0.3 + long_roc * 0.2).clamp(-100.0, 100.0);
        }

        // Calculate volume factor
        for i in self.volume_period..n {
            let avg_volume: f64 = volume[(i - self.volume_period + 1)..=i].iter().sum::<f64>()
                / self.volume_period as f64;

            if avg_volume > 1e-10 {
                // Volume ratio - current vs average
                let vol_ratio = volume[i] / avg_volume;

                // Direction based on price change
                let direction = if close[i] > close[i - 1] {
                    1.0
                } else if close[i] < close[i - 1] {
                    -1.0
                } else {
                    0.0
                };

                // Volume factor: high volume in trend direction = stronger signal
                volume_factor[i] = (direction * (vol_ratio - 1.0) * 50.0).clamp(-100.0, 100.0);
            } else {
                volume_factor[i] = 0.0;
            }
        }

        // Combine factors
        for i in min_required..n {
            if !price_factor[i].is_nan() && !volume_factor[i].is_nan() {
                // Price factor is primary, volume factor confirms
                let vol_confirm = if price_factor[i].signum() == volume_factor[i].signum() {
                    1.2 // Confirmation bonus
                } else {
                    0.8 // Divergence penalty
                };

                momentum[i] = ((price_factor[i] * 0.7 + volume_factor[i] * 0.3) * vol_confirm)
                    .clamp(-100.0, 100.0);
            }
        }

        MultiFactorMomentumOutput {
            momentum,
            price_factor,
            volume_factor,
        }
    }
}

impl TechnicalIndicator for MultiFactorMomentum {
    fn name(&self) -> &str {
        "MultiFactorMomentum"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            result.momentum,
            result.price_factor,
            result.volume_factor,
        ))
    }

    fn min_periods(&self) -> usize {
        self.long_period.max(self.volume_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 9. TrendQualityComposite
// ============================================================================

/// Trend Quality Composite output.
#[derive(Debug, Clone)]
pub struct TrendQualityCompositeOutput {
    /// Overall trend quality (0-100).
    pub quality: Vec<f64>,
    /// Trend strength component.
    pub trend_strength: Vec<f64>,
    /// Trend consistency component.
    pub consistency: Vec<f64>,
}

/// Trend Quality Composite configuration.
#[derive(Debug, Clone)]
pub struct TrendQualityCompositeConfig {
    /// Trend period (default: 20).
    pub trend_period: usize,
    /// Consistency period (default: 10).
    pub consistency_period: usize,
    /// Smoothing period (default: 3).
    pub smoothing_period: usize,
}

impl Default for TrendQualityCompositeConfig {
    fn default() -> Self {
        Self {
            trend_period: 20,
            consistency_period: 10,
            smoothing_period: 3,
        }
    }
}

/// Trend Quality Composite.
///
/// Measures the quality of the current trend based on:
/// - Trend strength (ADX-like measure)
/// - Trend consistency (directional persistence)
/// - Volatility adjustment
///
/// High quality trends are more likely to continue.
#[derive(Debug, Clone)]
pub struct TrendQualityComposite {
    trend_period: usize,
    consistency_period: usize,
    smoothing_period: usize,
}

impl TrendQualityComposite {
    /// Create a new TrendQualityComposite with the given configuration.
    pub fn new(config: TrendQualityCompositeConfig) -> Result<Self> {
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.consistency_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "consistency_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            trend_period: config.trend_period,
            consistency_period: config.consistency_period,
            smoothing_period: config.smoothing_period,
        })
    }

    /// Calculate the Trend Quality Composite values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> TrendQualityCompositeOutput {
        let n = close.len();
        let mut quality = vec![f64::NAN; n];
        let mut trend_strength = vec![f64::NAN; n];
        let mut consistency = vec![f64::NAN; n];

        let min_required = self.trend_period.max(self.consistency_period);
        if n <= min_required {
            return TrendQualityCompositeOutput {
                quality,
                trend_strength,
                consistency,
            };
        }

        // Calculate trend strength using efficiency ratio
        for i in self.trend_period..n {
            let net_change = (close[i] - close[i - self.trend_period]).abs();
            let mut sum_changes = 0.0;
            for j in (i - self.trend_period + 1)..=i {
                sum_changes += (close[j] - close[j - 1]).abs();
            }

            if sum_changes > 1e-10 {
                trend_strength[i] = (net_change / sum_changes * 100.0).clamp(0.0, 100.0);
            } else {
                trend_strength[i] = 0.0;
            }
        }

        // Calculate consistency (directional persistence)
        for i in self.consistency_period..n {
            let window = &close[(i - self.consistency_period + 1)..=i];
            let mut same_direction = 0;
            let overall_direction = (close[i] - close[i - self.consistency_period]).signum();

            for j in 1..window.len() {
                let bar_direction = (window[j] - window[j - 1]).signum();
                if bar_direction == overall_direction || bar_direction == 0.0 {
                    same_direction += 1;
                }
            }

            let total_bars = (window.len() - 1).max(1);
            consistency[i] = (same_direction as f64 / total_bars as f64 * 100.0).clamp(0.0, 100.0);
        }

        // Calculate volatility adjustment
        let mut vol_factor = vec![1.0; n];
        for i in self.trend_period..n {
            let atr = self.calculate_atr(high, low, close, i, self.trend_period);
            if !atr.is_nan() && close[i].abs() > 1e-10 {
                let atr_percent = atr / close[i];
                // Lower volatility = higher quality
                vol_factor[i] = (1.5 - atr_percent * 50.0).clamp(0.5, 1.5);
            }
        }

        // Combine components
        for i in min_required..n {
            if !trend_strength[i].is_nan() && !consistency[i].is_nan() {
                let raw_quality = (trend_strength[i] * 0.5 + consistency[i] * 0.5) * vol_factor[i];
                quality[i] = raw_quality.clamp(0.0, 100.0);
            }
        }

        // Apply smoothing
        if self.smoothing_period > 1 {
            let smoothed = self.ema_smooth(&quality, self.smoothing_period);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    quality[i] = smoothed[i];
                }
            }
        }

        TrendQualityCompositeOutput {
            quality,
            trend_strength,
            consistency,
        }
    }

    /// Calculate ATR for a specific index.
    fn calculate_atr(&self, high: &[f64], low: &[f64], close: &[f64], idx: usize, period: usize) -> f64 {
        if idx < period {
            return f64::NAN;
        }

        let mut sum = 0.0;
        for i in (idx - period + 1)..=idx {
            let tr = if i == 0 {
                high[i] - low[i]
            } else {
                (high[i] - low[i])
                    .max((high[i] - close[i - 1]).abs())
                    .max((low[i] - close[i - 1]).abs())
            };
            sum += tr;
        }
        sum / period as f64
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for TrendQualityComposite {
    fn name(&self) -> &str {
        "TrendQualityComposite"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.quality,
            result.trend_strength,
            result.consistency,
        ))
    }

    fn min_periods(&self) -> usize {
        self.trend_period.max(self.consistency_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 10. RiskOnRiskOff
// ============================================================================

/// Risk-On/Risk-Off output.
#[derive(Debug, Clone)]
pub struct RiskOnRiskOffOutput {
    /// Risk-on/risk-off indicator (-100 to 100).
    /// Positive = risk-on, Negative = risk-off.
    pub indicator: Vec<f64>,
    /// Risk score component (0-100).
    pub risk_score: Vec<f64>,
    /// Trend bias component (-100 to 100).
    pub trend_bias: Vec<f64>,
}

/// Risk-On/Risk-Off configuration.
#[derive(Debug, Clone)]
pub struct RiskOnRiskOffConfig {
    /// Momentum period (default: 14).
    pub momentum_period: usize,
    /// Volatility period (default: 20).
    pub volatility_period: usize,
    /// Trend period (default: 50).
    pub trend_period: usize,
}

impl Default for RiskOnRiskOffConfig {
    fn default() -> Self {
        Self {
            momentum_period: 14,
            volatility_period: 20,
            trend_period: 50,
        }
    }
}

/// Risk-On/Risk-Off Indicator.
///
/// Identifies risk-on or risk-off market conditions based on:
/// - Price momentum (positive = risk-on)
/// - Volatility levels (high vol = risk-off)
/// - Trend direction (uptrend = risk-on)
///
/// Positive values indicate risk-on environment (favor risky assets),
/// negative values indicate risk-off (favor safe-haven assets).
#[derive(Debug, Clone)]
pub struct RiskOnRiskOff {
    momentum_period: usize,
    volatility_period: usize,
    trend_period: usize,
}

impl RiskOnRiskOff {
    /// Create a new RiskOnRiskOff with the given configuration.
    pub fn new(config: RiskOnRiskOffConfig) -> Result<Self> {
        if config.momentum_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            momentum_period: config.momentum_period,
            volatility_period: config.volatility_period,
            trend_period: config.trend_period,
        })
    }

    /// Calculate the Risk-On/Risk-Off values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> RiskOnRiskOffOutput {
        let n = close.len();
        let mut indicator = vec![f64::NAN; n];
        let mut risk_score = vec![f64::NAN; n];
        let mut trend_bias = vec![f64::NAN; n];

        let min_required = self.momentum_period.max(self.volatility_period).max(self.trend_period);
        if n <= min_required {
            return RiskOnRiskOffOutput {
                indicator,
                risk_score,
                trend_bias,
            };
        }

        // Calculate EMA for trend
        let ema = self.calculate_ema(close, self.trend_period);

        // Calculate ATR for volatility
        let mut atr = vec![f64::NAN; n];
        for i in 1..n {
            if i >= self.volatility_period {
                let mut sum = 0.0;
                for j in (i - self.volatility_period + 1)..=i {
                    let tr = if j == 0 {
                        high[j] - low[j]
                    } else {
                        (high[j] - low[j])
                            .max((high[j] - close[j - 1]).abs())
                            .max((low[j] - close[j - 1]).abs())
                    };
                    sum += tr;
                }
                atr[i] = sum / self.volatility_period as f64;
            }
        }

        // Calculate momentum (RSI-based)
        let mut rsi = vec![f64::NAN; n];
        for i in self.momentum_period..n {
            let mut gains = 0.0;
            let mut losses = 0.0;
            for j in (i - self.momentum_period + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += -change;
                }
            }
            let avg_gain = gains / self.momentum_period as f64;
            let avg_loss = losses / self.momentum_period as f64;

            if avg_loss > 1e-10 {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            } else if gains > 0.0 {
                rsi[i] = 100.0;
            } else {
                rsi[i] = 50.0;
            }
        }

        // Calculate components and indicator
        for i in min_required..n {
            // Trend bias: position relative to EMA
            if !ema[i].is_nan() && ema[i].abs() > 1e-10 {
                let deviation = (close[i] - ema[i]) / ema[i] * 100.0;
                trend_bias[i] = (deviation * 10.0).clamp(-100.0, 100.0);
            }

            // Risk score: inverse volatility (high vol = high risk = low score)
            if !atr[i].is_nan() && close[i].abs() > 1e-10 {
                let atr_percent = atr[i] / close[i] * 100.0;
                // Scale: 0.5% ATR = 100, 3% ATR = 0
                risk_score[i] = (100.0 - atr_percent * 40.0).clamp(0.0, 100.0);
            }

            // Combine into risk-on/risk-off indicator
            if !rsi[i].is_nan() && !risk_score[i].is_nan() && !trend_bias[i].is_nan() {
                // Momentum component (RSI centered at 50)
                let momentum_component = (rsi[i] - 50.0) * 2.0;

                // Low risk + positive momentum + uptrend = risk-on
                // High risk + negative momentum + downtrend = risk-off
                let risk_component = (risk_score[i] - 50.0) * 2.0;

                indicator[i] = (momentum_component * 0.4 + risk_component * 0.3 + trend_bias[i] * 0.3)
                    .clamp(-100.0, 100.0);
            }
        }

        RiskOnRiskOffOutput {
            indicator,
            risk_score,
            trend_bias,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }
}

impl TechnicalIndicator for RiskOnRiskOff {
    fn name(&self) -> &str {
        "RiskOnRiskOff"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.indicator,
            result.risk_score,
            result.trend_bias,
        ))
    }

    fn min_periods(&self) -> usize {
        self.momentum_period.max(self.volatility_period).max(self.trend_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 11. MarketBreadthComposite
// ============================================================================

/// Market Breadth Composite output.
#[derive(Debug, Clone)]
pub struct MarketBreadthCompositeOutput {
    /// Overall breadth indicator (-100 to 100).
    pub breadth: Vec<f64>,
    /// Advance/decline component.
    pub advance_decline: Vec<f64>,
    /// Volume breadth component.
    pub volume_breadth: Vec<f64>,
}

/// Market Breadth Composite configuration.
#[derive(Debug, Clone)]
pub struct MarketBreadthCompositeConfig {
    /// Breadth calculation period (default: 20).
    pub breadth_period: usize,
    /// Volume period (default: 10).
    pub volume_period: usize,
    /// Smoothing period (default: 5).
    pub smoothing_period: usize,
}

impl Default for MarketBreadthCompositeConfig {
    fn default() -> Self {
        Self {
            breadth_period: 20,
            volume_period: 10,
            smoothing_period: 5,
        }
    }
}

/// Market Breadth Composite.
///
/// Simulates market breadth analysis for a single instrument by measuring:
/// - Advance/decline ratio (up bars vs down bars)
/// - Volume confirmation (up volume vs down volume)
/// - Momentum distribution
///
/// Positive values indicate broad participation in upward moves.
#[derive(Debug, Clone)]
pub struct MarketBreadthComposite {
    breadth_period: usize,
    volume_period: usize,
    smoothing_period: usize,
}

impl MarketBreadthComposite {
    /// Create a new MarketBreadthComposite with the given configuration.
    pub fn new(config: MarketBreadthCompositeConfig) -> Result<Self> {
        if config.breadth_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "breadth_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volume_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            breadth_period: config.breadth_period,
            volume_period: config.volume_period,
            smoothing_period: config.smoothing_period,
        })
    }

    /// Calculate the Market Breadth Composite values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> MarketBreadthCompositeOutput {
        let n = close.len();
        let mut breadth = vec![f64::NAN; n];
        let mut advance_decline = vec![f64::NAN; n];
        let mut volume_breadth = vec![f64::NAN; n];

        let min_required = self.breadth_period.max(self.volume_period);
        if n <= min_required {
            return MarketBreadthCompositeOutput {
                breadth,
                advance_decline,
                volume_breadth,
            };
        }

        // Calculate advance/decline ratio over period
        for i in self.breadth_period..n {
            let mut advances = 0;
            let mut declines = 0;

            for j in (i - self.breadth_period + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            if total > 0 {
                // AD ratio scaled to -100 to 100
                advance_decline[i] = ((advances as f64 - declines as f64) / total as f64 * 100.0)
                    .clamp(-100.0, 100.0);
            } else {
                advance_decline[i] = 0.0;
            }
        }

        // Calculate volume breadth (up volume vs down volume)
        for i in self.volume_period..n {
            let mut up_volume = 0.0;
            let mut down_volume = 0.0;

            for j in (i - self.volume_period + 1)..=i {
                if close[j] > close[j - 1] {
                    up_volume += volume[j];
                } else if close[j] < close[j - 1] {
                    down_volume += volume[j];
                }
            }

            let total_vol = up_volume + down_volume;
            if total_vol > 1e-10 {
                volume_breadth[i] = ((up_volume - down_volume) / total_vol * 100.0)
                    .clamp(-100.0, 100.0);
            } else {
                volume_breadth[i] = 0.0;
            }
        }

        // Combine into breadth indicator
        for i in min_required..n {
            if !advance_decline[i].is_nan() && !volume_breadth[i].is_nan() {
                breadth[i] = (advance_decline[i] * 0.6 + volume_breadth[i] * 0.4)
                    .clamp(-100.0, 100.0);
            }
        }

        // Apply smoothing
        if self.smoothing_period > 1 {
            let smoothed = self.ema_smooth(&breadth, self.smoothing_period);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    breadth[i] = smoothed[i];
                }
            }
        }

        MarketBreadthCompositeOutput {
            breadth,
            advance_decline,
            volume_breadth,
        }
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for MarketBreadthComposite {
    fn name(&self) -> &str {
        "MarketBreadthComposite"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            result.breadth,
            result.advance_decline,
            result.volume_breadth,
        ))
    }

    fn min_periods(&self) -> usize {
        self.breadth_period.max(self.volume_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 12. SentimentTrendComposite
// ============================================================================

/// Sentiment Trend Composite output.
#[derive(Debug, Clone)]
pub struct SentimentTrendCompositeOutput {
    /// Combined sentiment-trend composite (-100 to 100).
    pub composite: Vec<f64>,
    /// Sentiment score component.
    pub sentiment_score: Vec<f64>,
    /// Trend score component.
    pub trend_score: Vec<f64>,
}

/// Sentiment Trend Composite configuration.
#[derive(Debug, Clone)]
pub struct SentimentTrendCompositeConfig {
    /// Sentiment period (default: 14).
    pub sentiment_period: usize,
    /// Trend period (default: 20).
    pub trend_period: usize,
    /// Weight for sentiment (default: 0.5).
    pub sentiment_weight: f64,
    /// Weight for trend (default: 0.5).
    pub trend_weight: f64,
}

impl Default for SentimentTrendCompositeConfig {
    fn default() -> Self {
        Self {
            sentiment_period: 14,
            trend_period: 20,
            sentiment_weight: 0.5,
            trend_weight: 0.5,
        }
    }
}

/// Sentiment Trend Composite.
///
/// Combines sentiment analysis with trend analysis:
/// - Sentiment: Derived from price action patterns and volume
/// - Trend: Direction and strength of price movement
///
/// Useful for identifying aligned sentiment-trend conditions.
#[derive(Debug, Clone)]
pub struct SentimentTrendComposite {
    sentiment_period: usize,
    trend_period: usize,
    sentiment_weight: f64,
    trend_weight: f64,
}

impl SentimentTrendComposite {
    /// Create a new SentimentTrendComposite with the given configuration.
    pub fn new(config: SentimentTrendCompositeConfig) -> Result<Self> {
        if config.sentiment_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sentiment_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.sentiment_weight < 0.0 || config.sentiment_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "sentiment_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.trend_weight < 0.0 || config.trend_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(Self {
            sentiment_period: config.sentiment_period,
            trend_period: config.trend_period,
            sentiment_weight: config.sentiment_weight,
            trend_weight: config.trend_weight,
        })
    }

    /// Calculate the Sentiment Trend Composite values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> SentimentTrendCompositeOutput {
        let n = close.len();
        let mut composite = vec![f64::NAN; n];
        let mut sentiment_score = vec![f64::NAN; n];
        let mut trend_score = vec![f64::NAN; n];

        let min_required = self.sentiment_period.max(self.trend_period);
        if n <= min_required {
            return SentimentTrendCompositeOutput {
                composite,
                sentiment_score,
                trend_score,
            };
        }

        // Calculate sentiment from price action and volume
        for i in self.sentiment_period..n {
            // Bullish/bearish candle ratio
            let mut bullish = 0.0;
            let mut bearish = 0.0;
            let mut bullish_vol = 0.0;
            let mut bearish_vol = 0.0;

            for j in (i - self.sentiment_period + 1)..=i {
                if close[j] > close[j - 1] {
                    bullish += 1.0;
                    bullish_vol += volume[j];
                } else if close[j] < close[j - 1] {
                    bearish += 1.0;
                    bearish_vol += volume[j];
                }
            }

            let total = bullish + bearish;
            let total_vol = bullish_vol + bearish_vol;

            // Combine candle ratio and volume-weighted sentiment
            let candle_sentiment = if total > 0.0 {
                (bullish - bearish) / total * 100.0
            } else {
                0.0
            };

            let volume_sentiment = if total_vol > 1e-10 {
                (bullish_vol - bearish_vol) / total_vol * 100.0
            } else {
                0.0
            };

            sentiment_score[i] = ((candle_sentiment + volume_sentiment) / 2.0).clamp(-100.0, 100.0);
        }

        // Calculate trend score
        let ema = self.calculate_ema(close, self.trend_period);
        for i in self.trend_period..n {
            if !ema[i].is_nan() && ema[i].abs() > 1e-10 {
                // Deviation from EMA as trend indicator
                let deviation = (close[i] - ema[i]) / ema[i] * 100.0;

                // Efficiency ratio for trend quality
                let net_change = (close[i] - close[i - self.trend_period]).abs();
                let mut sum_changes = 0.0;
                for j in (i - self.trend_period + 1)..=i {
                    sum_changes += (close[j] - close[j - 1]).abs();
                }

                let efficiency = if sum_changes > 1e-10 {
                    net_change / sum_changes
                } else {
                    0.0
                };

                // Trend direction with quality adjustment
                let direction = if close[i] > close[i - self.trend_period] {
                    1.0
                } else {
                    -1.0
                };

                trend_score[i] = (direction * efficiency * 100.0 + deviation * 5.0).clamp(-100.0, 100.0);
            }
        }

        // Combine sentiment and trend
        for i in min_required..n {
            if !sentiment_score[i].is_nan() && !trend_score[i].is_nan() {
                // Alignment bonus when sentiment and trend agree
                let alignment = if sentiment_score[i].signum() == trend_score[i].signum() {
                    1.1
                } else {
                    0.9
                };

                composite[i] = ((self.sentiment_weight * sentiment_score[i]
                    + self.trend_weight * trend_score[i])
                    * alignment)
                    .clamp(-100.0, 100.0);
            }
        }

        SentimentTrendCompositeOutput {
            composite,
            sentiment_score,
            trend_score,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }
}

impl TechnicalIndicator for SentimentTrendComposite {
    fn name(&self) -> &str {
        "SentimentTrendComposite"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            result.composite,
            result.sentiment_score,
            result.trend_score,
        ))
    }

    fn min_periods(&self) -> usize {
        self.sentiment_period.max(self.trend_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 13. MarketStrengthIndex
// ============================================================================

/// Market Strength Index output.
#[derive(Debug, Clone)]
pub struct MarketStrengthIndexOutput {
    /// Overall market strength index (0-100).
    pub strength: Vec<f64>,
    /// Price strength component (0-100).
    pub price_strength: Vec<f64>,
    /// Volume strength component (0-100).
    pub volume_strength: Vec<f64>,
}

/// Market Strength Index configuration.
#[derive(Debug, Clone)]
pub struct MarketStrengthIndexConfig {
    /// Period for price strength calculation (default: 14).
    pub price_period: usize,
    /// Period for volume strength calculation (default: 14).
    pub volume_period: usize,
    /// Period for smoothing (default: 3).
    pub smoothing_period: usize,
    /// Weight for price component (default: 0.6).
    pub price_weight: f64,
    /// Weight for volume component (default: 0.4).
    pub volume_weight: f64,
}

impl Default for MarketStrengthIndexConfig {
    fn default() -> Self {
        Self {
            price_period: 14,
            volume_period: 14,
            smoothing_period: 3,
            price_weight: 0.6,
            volume_weight: 0.4,
        }
    }
}

/// Market Strength Index.
///
/// A composite indicator that measures overall market strength by combining:
/// - Price strength: Based on RSI-like momentum and trend direction
/// - Volume strength: Based on volume patterns and confirmation
///
/// Higher values indicate strong market conditions with good volume support.
/// Values above 70 suggest strong bullish conditions, below 30 suggest weakness.
///
/// # Formula
///
/// - Price Strength = Normalized RSI + Trend Efficiency
/// - Volume Strength = Volume ratio relative to moving average
/// - Strength = price_weight * price_strength + volume_weight * volume_strength
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{MarketStrengthIndex, MarketStrengthIndexConfig};
///
/// let config = MarketStrengthIndexConfig::default();
/// let indicator = MarketStrengthIndex::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close, &volume);
/// ```
#[derive(Debug, Clone)]
pub struct MarketStrengthIndex {
    price_period: usize,
    volume_period: usize,
    smoothing_period: usize,
    price_weight: f64,
    volume_weight: f64,
}

impl MarketStrengthIndex {
    /// Create a new MarketStrengthIndex with the given configuration.
    pub fn new(config: MarketStrengthIndexConfig) -> Result<Self> {
        if config.price_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "price_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volume_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.price_weight < 0.0 || config.price_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "price_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.volume_weight < 0.0 || config.volume_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(Self {
            price_period: config.price_period,
            volume_period: config.volume_period,
            smoothing_period: config.smoothing_period,
            price_weight: config.price_weight,
            volume_weight: config.volume_weight,
        })
    }

    /// Calculate the Market Strength Index values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> MarketStrengthIndexOutput {
        let n = close.len();
        let mut strength = vec![f64::NAN; n];
        let mut price_strength = vec![f64::NAN; n];
        let mut volume_strength = vec![f64::NAN; n];

        let min_required = self.price_period.max(self.volume_period);
        if n <= min_required {
            return MarketStrengthIndexOutput {
                strength,
                price_strength,
                volume_strength,
            };
        }

        // Calculate price strength (RSI-based)
        for i in self.price_period..n {
            let mut gains = 0.0;
            let mut losses = 0.0;

            for j in (i - self.price_period + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += -change;
                }
            }

            let avg_gain = gains / self.price_period as f64;
            let avg_loss = losses / self.price_period as f64;

            let rsi = if avg_loss > 1e-10 {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            } else if avg_gain > 0.0 {
                100.0
            } else {
                50.0
            };

            // Add trend efficiency component
            let net_change = (close[i] - close[i - self.price_period]).abs();
            let mut sum_changes = 0.0;
            for j in (i - self.price_period + 1)..=i {
                sum_changes += (close[j] - close[j - 1]).abs();
            }

            let efficiency = if sum_changes > 1e-10 {
                net_change / sum_changes
            } else {
                0.0
            };

            // Combine RSI and efficiency (both contribute to price strength)
            price_strength[i] = (rsi * 0.7 + efficiency * 100.0 * 0.3).clamp(0.0, 100.0);
        }

        // Calculate volume strength
        for i in self.volume_period..n {
            let avg_volume: f64 = volume[(i - self.volume_period + 1)..=i]
                .iter()
                .sum::<f64>()
                / self.volume_period as f64;

            if avg_volume > 1e-10 {
                let vol_ratio = volume[i] / avg_volume;

                // Count up-volume vs down-volume bars
                let mut up_vol = 0.0;
                let mut down_vol = 0.0;
                for j in (i - self.volume_period + 1)..=i {
                    if close[j] > close[j - 1] {
                        up_vol += volume[j];
                    } else if close[j] < close[j - 1] {
                        down_vol += volume[j];
                    }
                }

                let total_vol = up_vol + down_vol;
                let vol_direction = if total_vol > 1e-10 {
                    (up_vol - down_vol) / total_vol
                } else {
                    0.0
                };

                // Volume strength = base 50 + direction bias + current volume factor
                volume_strength[i] =
                    (50.0 + vol_direction * 30.0 + (vol_ratio - 1.0) * 20.0).clamp(0.0, 100.0);
            } else {
                volume_strength[i] = 50.0;
            }
        }

        // Combine components
        for i in min_required..n {
            if !price_strength[i].is_nan() && !volume_strength[i].is_nan() {
                strength[i] = (self.price_weight * price_strength[i]
                    + self.volume_weight * volume_strength[i])
                    .clamp(0.0, 100.0);
            }
        }

        // Apply smoothing
        if self.smoothing_period > 1 {
            let smoothed = self.ema_smooth(&strength, self.smoothing_period);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    strength[i] = smoothed[i];
                }
            }
        }

        MarketStrengthIndexOutput {
            strength,
            price_strength,
            volume_strength,
        }
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for MarketStrengthIndex {
    fn name(&self) -> &str {
        "MarketStrengthIndex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            result.strength,
            result.price_strength,
            result.volume_strength,
        ))
    }

    fn min_periods(&self) -> usize {
        self.price_period.max(self.volume_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 14. TrendMomentumComposite
// ============================================================================

/// Trend Momentum Composite output.
#[derive(Debug, Clone)]
pub struct TrendMomentumCompositeOutput {
    /// Combined trend-momentum signal (-100 to 100).
    pub signal: Vec<f64>,
    /// Trend component (-100 to 100).
    pub trend_component: Vec<f64>,
    /// Momentum component (-100 to 100).
    pub momentum_component: Vec<f64>,
}

/// Trend Momentum Composite configuration.
#[derive(Debug, Clone)]
pub struct TrendMomentumCompositeConfig {
    /// Short EMA period for trend (default: 12).
    pub short_ema_period: usize,
    /// Long EMA period for trend (default: 26).
    pub long_ema_period: usize,
    /// Momentum period (default: 14).
    pub momentum_period: usize,
    /// Signal smoothing period (default: 9).
    pub signal_period: usize,
    /// Weight for trend (default: 0.5).
    pub trend_weight: f64,
    /// Weight for momentum (default: 0.5).
    pub momentum_weight: f64,
}

impl Default for TrendMomentumCompositeConfig {
    fn default() -> Self {
        Self {
            short_ema_period: 12,
            long_ema_period: 26,
            momentum_period: 14,
            signal_period: 9,
            trend_weight: 0.5,
            momentum_weight: 0.5,
        }
    }
}

/// Trend Momentum Composite.
///
/// Combines trend-following and momentum indicators into a single composite signal:
/// - Trend Component: Based on MACD-like EMA crossover
/// - Momentum Component: Based on RSI-style momentum measurement
///
/// The indicator produces stronger signals when both trend and momentum align.
/// Positive values indicate bullish conditions, negative values bearish.
///
/// # Formula
///
/// - Trend = (Short EMA - Long EMA) / Price * 100, normalized
/// - Momentum = (RSI - 50) * 2 (centered and scaled)
/// - Signal = trend_weight * trend + momentum_weight * momentum
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{TrendMomentumComposite, TrendMomentumCompositeConfig};
///
/// let config = TrendMomentumCompositeConfig::default();
/// let indicator = TrendMomentumComposite::new(config).unwrap();
/// let result = indicator.calculate(&close);
/// ```
#[derive(Debug, Clone)]
pub struct TrendMomentumComposite {
    short_ema_period: usize,
    long_ema_period: usize,
    momentum_period: usize,
    signal_period: usize,
    trend_weight: f64,
    momentum_weight: f64,
}

impl TrendMomentumComposite {
    /// Create a new TrendMomentumComposite with the given configuration.
    pub fn new(config: TrendMomentumCompositeConfig) -> Result<Self> {
        if config.short_ema_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_ema_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.long_ema_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_ema_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.momentum_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.signal_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "signal_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.short_ema_period >= config.long_ema_period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_ema_period".to_string(),
                reason: "must be less than long_ema_period".to_string(),
            });
        }
        if config.trend_weight < 0.0 || config.trend_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.momentum_weight < 0.0 || config.momentum_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(Self {
            short_ema_period: config.short_ema_period,
            long_ema_period: config.long_ema_period,
            momentum_period: config.momentum_period,
            signal_period: config.signal_period,
            trend_weight: config.trend_weight,
            momentum_weight: config.momentum_weight,
        })
    }

    /// Calculate the Trend Momentum Composite values.
    pub fn calculate(&self, close: &[f64]) -> TrendMomentumCompositeOutput {
        let n = close.len();
        let mut signal = vec![f64::NAN; n];
        let mut trend_component = vec![f64::NAN; n];
        let mut momentum_component = vec![f64::NAN; n];

        let min_required = self.long_ema_period.max(self.momentum_period);
        if n <= min_required {
            return TrendMomentumCompositeOutput {
                signal,
                trend_component,
                momentum_component,
            };
        }

        // Calculate EMAs for trend
        let short_ema = self.calculate_ema(close, self.short_ema_period);
        let long_ema = self.calculate_ema(close, self.long_ema_period);

        // Calculate trend component (MACD-like)
        for i in self.long_ema_period..n {
            if !short_ema[i].is_nan() && !long_ema[i].is_nan() && close[i].abs() > 1e-10 {
                // Normalize MACD as percentage of price, then scale
                let macd = (short_ema[i] - long_ema[i]) / close[i] * 100.0;
                trend_component[i] = (macd * 20.0).clamp(-100.0, 100.0);
            }
        }

        // Calculate momentum component (RSI-based)
        for i in self.momentum_period..n {
            let mut gains = 0.0;
            let mut losses = 0.0;

            for j in (i - self.momentum_period + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += -change;
                }
            }

            let avg_gain = gains / self.momentum_period as f64;
            let avg_loss = losses / self.momentum_period as f64;

            let rsi = if avg_loss > 1e-10 {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            } else if avg_gain > 0.0 {
                100.0
            } else {
                50.0
            };

            // Convert RSI (0-100) to centered momentum (-100 to 100)
            momentum_component[i] = (rsi - 50.0) * 2.0;
        }

        // Combine components
        for i in min_required..n {
            if !trend_component[i].is_nan() && !momentum_component[i].is_nan() {
                // Apply alignment bonus when trend and momentum agree
                let alignment = if trend_component[i].signum() == momentum_component[i].signum() {
                    1.1
                } else {
                    0.9
                };

                let raw_signal = self.trend_weight * trend_component[i]
                    + self.momentum_weight * momentum_component[i];
                signal[i] = (raw_signal * alignment).clamp(-100.0, 100.0);
            }
        }

        // Apply signal line smoothing
        if self.signal_period > 1 {
            let smoothed = self.ema_smooth(&signal, self.signal_period);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    signal[i] = smoothed[i];
                }
            }
        }

        TrendMomentumCompositeOutput {
            signal,
            trend_component,
            momentum_component,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for TrendMomentumComposite {
    fn name(&self) -> &str {
        "TrendMomentumComposite"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(
            result.signal,
            result.trend_component,
            result.momentum_component,
        ))
    }

    fn min_periods(&self) -> usize {
        self.long_ema_period.max(self.momentum_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 15. VolatilityTrendIndex
// ============================================================================

/// Volatility Trend Index output.
#[derive(Debug, Clone)]
pub struct VolatilityTrendIndexOutput {
    /// Combined index value (-100 to 100).
    pub index: Vec<f64>,
    /// Volatility regime indicator (0-100).
    pub volatility_regime: Vec<f64>,
    /// Trend direction indicator (-100 to 100).
    pub trend_direction: Vec<f64>,
}

/// Volatility Trend Index configuration.
#[derive(Debug, Clone)]
pub struct VolatilityTrendIndexConfig {
    /// ATR period for volatility (default: 14).
    pub atr_period: usize,
    /// Trend period (default: 20).
    pub trend_period: usize,
    /// Volatility lookback for percentile (default: 50).
    pub volatility_lookback: usize,
    /// Smoothing period (default: 3).
    pub smoothing_period: usize,
}

impl Default for VolatilityTrendIndexConfig {
    fn default() -> Self {
        Self {
            atr_period: 14,
            trend_period: 20,
            volatility_lookback: 50,
            smoothing_period: 3,
        }
    }
}

/// Volatility Trend Index.
///
/// Combines volatility analysis with trend direction to identify:
/// - High volatility trending markets (breakouts)
/// - Low volatility trending markets (steady trends)
/// - High volatility ranging markets (choppy)
/// - Low volatility ranging markets (consolidation)
///
/// Positive values indicate uptrend, negative downtrend.
/// Magnitude indicates the combination of trend strength and volatility regime.
///
/// # Components
///
/// - Volatility Regime: ATR percentile rank (0 = low vol, 100 = high vol)
/// - Trend Direction: Normalized price position relative to trend
/// - Index: Combined signal that amplifies in trending + volatile conditions
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{VolatilityTrendIndex, VolatilityTrendIndexConfig};
///
/// let config = VolatilityTrendIndexConfig::default();
/// let indicator = VolatilityTrendIndex::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close);
/// ```
#[derive(Debug, Clone)]
pub struct VolatilityTrendIndex {
    atr_period: usize,
    trend_period: usize,
    volatility_lookback: usize,
    smoothing_period: usize,
}

impl VolatilityTrendIndex {
    /// Create a new VolatilityTrendIndex with the given configuration.
    pub fn new(config: VolatilityTrendIndexConfig) -> Result<Self> {
        if config.atr_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_lookback == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_lookback".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            atr_period: config.atr_period,
            trend_period: config.trend_period,
            volatility_lookback: config.volatility_lookback,
            smoothing_period: config.smoothing_period,
        })
    }

    /// Calculate the Volatility Trend Index values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> VolatilityTrendIndexOutput {
        let n = close.len();
        let mut index = vec![f64::NAN; n];
        let mut volatility_regime = vec![f64::NAN; n];
        let mut trend_direction = vec![f64::NAN; n];

        let min_required = self.atr_period.max(self.trend_period).max(self.volatility_lookback);
        if n <= min_required {
            return VolatilityTrendIndexOutput {
                index,
                volatility_regime,
                trend_direction,
            };
        }

        // Calculate ATR
        let mut atr = vec![f64::NAN; n];
        for i in 1..n {
            if i >= self.atr_period {
                let mut sum = 0.0;
                for j in (i - self.atr_period + 1)..=i {
                    let tr = if j == 0 {
                        high[j] - low[j]
                    } else {
                        (high[j] - low[j])
                            .max((high[j] - close[j - 1]).abs())
                            .max((low[j] - close[j - 1]).abs())
                    };
                    sum += tr;
                }
                atr[i] = sum / self.atr_period as f64;
            }
        }

        // Calculate volatility percentile
        for i in self.volatility_lookback..n {
            if !atr[i].is_nan() {
                let window = &atr[(i - self.volatility_lookback + 1)..=i];
                let current = atr[i];
                let count_below = window.iter().filter(|&&x| !x.is_nan() && x < current).count();
                let total = window.iter().filter(|x| !x.is_nan()).count();
                volatility_regime[i] = if total > 0 {
                    (count_below as f64 / total as f64 * 100.0).clamp(0.0, 100.0)
                } else {
                    50.0
                };
            }
        }

        // Calculate trend direction using EMA and price position
        let ema = self.calculate_ema(close, self.trend_period);
        for i in self.trend_period..n {
            if !ema[i].is_nan() && ema[i].abs() > 1e-10 {
                // Price deviation from EMA as percentage
                let deviation = (close[i] - ema[i]) / ema[i] * 100.0;

                // Add efficiency ratio for trend quality
                let net_change = close[i] - close[i - self.trend_period];
                let mut sum_changes = 0.0;
                for j in (i - self.trend_period + 1)..=i {
                    sum_changes += (close[j] - close[j - 1]).abs();
                }

                let efficiency = if sum_changes > 1e-10 {
                    net_change.abs() / sum_changes
                } else {
                    0.0
                };

                let direction = if net_change >= 0.0 { 1.0 } else { -1.0 };
                trend_direction[i] = (direction * (deviation.abs() * 5.0 + efficiency * 50.0))
                    .clamp(-100.0, 100.0);
            }
        }

        // Combine volatility and trend
        for i in min_required..n {
            if !volatility_regime[i].is_nan() && !trend_direction[i].is_nan() {
                // High volatility amplifies trend signal
                let vol_factor = 0.5 + volatility_regime[i] / 100.0; // 0.5 to 1.5

                index[i] = (trend_direction[i] * vol_factor).clamp(-100.0, 100.0);
            }
        }

        // Apply smoothing
        if self.smoothing_period > 1 {
            let smoothed = self.ema_smooth(&index, self.smoothing_period);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    index[i] = smoothed[i];
                }
            }
        }

        VolatilityTrendIndexOutput {
            index,
            volatility_regime,
            trend_direction,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilityTrendIndex {
    fn name(&self) -> &str {
        "VolatilityTrendIndex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.index,
            result.volatility_regime,
            result.trend_direction,
        ))
    }

    fn min_periods(&self) -> usize {
        self.atr_period.max(self.trend_period).max(self.volatility_lookback) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 16. MultiFactorSignal
// ============================================================================

/// Multi-Factor Signal output.
#[derive(Debug, Clone)]
pub struct MultiFactorSignalOutput {
    /// Combined multi-factor signal (-100 to 100).
    pub signal: Vec<f64>,
    /// Trend factor component (-100 to 100).
    pub trend_factor: Vec<f64>,
    /// Momentum factor component (-100 to 100).
    pub momentum_factor: Vec<f64>,
    /// Volatility factor component (-100 to 100).
    pub volatility_factor: Vec<f64>,
}

/// Multi-Factor Signal configuration.
#[derive(Debug, Clone)]
pub struct MultiFactorSignalConfig {
    /// Trend period (default: 20).
    pub trend_period: usize,
    /// Momentum period (default: 14).
    pub momentum_period: usize,
    /// Volatility period (default: 14).
    pub volatility_period: usize,
    /// Weight for trend factor (default: 0.4).
    pub trend_weight: f64,
    /// Weight for momentum factor (default: 0.35).
    pub momentum_weight: f64,
    /// Weight for volatility factor (default: 0.25).
    pub volatility_weight: f64,
    /// Smoothing period (default: 5).
    pub smoothing_period: usize,
}

impl Default for MultiFactorSignalConfig {
    fn default() -> Self {
        Self {
            trend_period: 20,
            momentum_period: 14,
            volatility_period: 14,
            trend_weight: 0.4,
            momentum_weight: 0.35,
            volatility_weight: 0.25,
            smoothing_period: 5,
        }
    }
}

/// Multi-Factor Signal.
///
/// A comprehensive composite indicator that combines multiple technical factors:
/// - Trend Factor: Direction and strength based on moving average analysis
/// - Momentum Factor: Rate of change and RSI-based momentum
/// - Volatility Factor: Volatility-adjusted signal strength
///
/// Each factor contributes to the overall signal based on configurable weights.
/// The indicator is designed to produce more reliable signals by requiring
/// confirmation from multiple factors.
///
/// # Signal Interpretation
///
/// - Values > 50: Strong bullish signal
/// - Values 25-50: Moderate bullish signal
/// - Values -25 to 25: Neutral/no clear signal
/// - Values -50 to -25: Moderate bearish signal
/// - Values < -50: Strong bearish signal
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{MultiFactorSignal, MultiFactorSignalConfig};
///
/// let config = MultiFactorSignalConfig::default();
/// let indicator = MultiFactorSignal::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close, &volume);
/// ```
#[derive(Debug, Clone)]
pub struct MultiFactorSignal {
    trend_period: usize,
    momentum_period: usize,
    volatility_period: usize,
    trend_weight: f64,
    momentum_weight: f64,
    volatility_weight: f64,
    smoothing_period: usize,
}

impl MultiFactorSignal {
    /// Create a new MultiFactorSignal with the given configuration.
    pub fn new(config: MultiFactorSignalConfig) -> Result<Self> {
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.momentum_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.trend_weight < 0.0 || config.trend_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.momentum_weight < 0.0 || config.momentum_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.volatility_weight < 0.0 || config.volatility_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(Self {
            trend_period: config.trend_period,
            momentum_period: config.momentum_period,
            volatility_period: config.volatility_period,
            trend_weight: config.trend_weight,
            momentum_weight: config.momentum_weight,
            volatility_weight: config.volatility_weight,
            smoothing_period: config.smoothing_period,
        })
    }

    /// Calculate the Multi-Factor Signal values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> MultiFactorSignalOutput {
        let n = close.len();
        let mut signal = vec![f64::NAN; n];
        let mut trend_factor = vec![f64::NAN; n];
        let mut momentum_factor = vec![f64::NAN; n];
        let mut volatility_factor = vec![f64::NAN; n];

        let max_period = self
            .trend_period
            .max(self.momentum_period)
            .max(self.volatility_period);
        if n <= max_period {
            return MultiFactorSignalOutput {
                signal,
                trend_factor,
                momentum_factor,
                volatility_factor,
            };
        }

        // Calculate trend factor
        let ema = self.calculate_ema(close, self.trend_period);
        for i in self.trend_period..n {
            if !ema[i].is_nan() && ema[i].abs() > 1e-10 {
                // Deviation from EMA
                let deviation = (close[i] - ema[i]) / ema[i] * 100.0;

                // EMA slope
                let ema_slope = if i > 0 && !ema[i - 1].is_nan() && ema[i - 1].abs() > 1e-10 {
                    (ema[i] - ema[i - 1]) / ema[i - 1] * 100.0
                } else {
                    0.0
                };

                trend_factor[i] = (deviation * 5.0 + ema_slope * 20.0).clamp(-100.0, 100.0);
            }
        }

        // Calculate momentum factor (ROC + RSI)
        for i in self.momentum_period..n {
            // Rate of change
            let roc = if close[i - self.momentum_period].abs() > 1e-10 {
                ((close[i] - close[i - self.momentum_period]) / close[i - self.momentum_period])
                    * 100.0
            } else {
                0.0
            };

            // RSI
            let mut gains = 0.0;
            let mut losses = 0.0;
            for j in (i - self.momentum_period + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += -change;
                }
            }

            let avg_gain = gains / self.momentum_period as f64;
            let avg_loss = losses / self.momentum_period as f64;

            let rsi = if avg_loss > 1e-10 {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            } else if avg_gain > 0.0 {
                100.0
            } else {
                50.0
            };

            // Combine ROC and centered RSI
            let rsi_centered = (rsi - 50.0) * 2.0;
            momentum_factor[i] = (roc * 5.0 + rsi_centered * 0.5).clamp(-100.0, 100.0);
        }

        // Calculate volatility factor
        let mut atr = vec![f64::NAN; n];
        for i in 1..n {
            if i >= self.volatility_period {
                let mut sum = 0.0;
                for j in (i - self.volatility_period + 1)..=i {
                    let tr = if j == 0 {
                        high[j] - low[j]
                    } else {
                        (high[j] - low[j])
                            .max((high[j] - close[j - 1]).abs())
                            .max((low[j] - close[j - 1]).abs())
                    };
                    sum += tr;
                }
                atr[i] = sum / self.volatility_period as f64;
            }
        }

        for i in max_period..n {
            if !atr[i].is_nan() && close[i].abs() > 1e-10 {
                let atr_percent = atr[i] / close[i] * 100.0;

                // Direction based on recent price movement
                let direction = if close[i] > close[i - 1] {
                    1.0
                } else if close[i] < close[i - 1] {
                    -1.0
                } else {
                    0.0
                };

                // High volatility amplifies signal in direction of move
                volatility_factor[i] = (direction * atr_percent * 25.0).clamp(-100.0, 100.0);
            }
        }

        // Combine all factors
        for i in max_period..n {
            if !trend_factor[i].is_nan()
                && !momentum_factor[i].is_nan()
                && !volatility_factor[i].is_nan()
            {
                // Count agreeing factors for confirmation bonus
                let signs = [
                    trend_factor[i].signum(),
                    momentum_factor[i].signum(),
                    volatility_factor[i].signum(),
                ];
                let positive_count = signs.iter().filter(|&&s| s > 0.0).count();
                let negative_count = signs.iter().filter(|&&s| s < 0.0).count();
                let agreement = positive_count.max(negative_count) as f64 / 3.0;

                let raw_signal = self.trend_weight * trend_factor[i]
                    + self.momentum_weight * momentum_factor[i]
                    + self.volatility_weight * volatility_factor[i];

                // Apply agreement bonus
                signal[i] = (raw_signal * (0.7 + agreement * 0.6)).clamp(-100.0, 100.0);
            }
        }

        // Apply smoothing
        if self.smoothing_period > 1 {
            let smoothed = self.ema_smooth(&signal, self.smoothing_period);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    signal[i] = smoothed[i];
                }
            }
        }

        MultiFactorSignalOutput {
            signal,
            trend_factor,
            momentum_factor,
            volatility_factor,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for MultiFactorSignal {
    fn name(&self) -> &str {
        "MultiFactorSignal"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            result.signal,
            result.trend_factor,
            result.momentum_factor,
        ))
    }

    fn min_periods(&self) -> usize {
        self.trend_period
            .max(self.momentum_period)
            .max(self.volatility_period)
            + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 17. AdaptiveMarketScore
// ============================================================================

/// Adaptive Market Score output.
#[derive(Debug, Clone)]
pub struct AdaptiveMarketScoreOutput {
    /// Adaptive market score (-100 to 100).
    pub score: Vec<f64>,
    /// Market condition indicator (0-100).
    pub market_condition: Vec<f64>,
    /// Adaptive weight applied (0-1).
    pub adaptive_weight: Vec<f64>,
}

/// Adaptive Market Score configuration.
#[derive(Debug, Clone)]
pub struct AdaptiveMarketScoreConfig {
    /// Fast period for responsive signals (default: 5).
    pub fast_period: usize,
    /// Slow period for stable signals (default: 20).
    pub slow_period: usize,
    /// Efficiency period for adaptation (default: 10).
    pub efficiency_period: usize,
    /// Volatility period for condition assessment (default: 14).
    pub volatility_period: usize,
}

impl Default for AdaptiveMarketScoreConfig {
    fn default() -> Self {
        Self {
            fast_period: 5,
            slow_period: 20,
            efficiency_period: 10,
            volatility_period: 14,
        }
    }
}

/// Adaptive Market Score.
///
/// A market scoring system that automatically adapts to current market conditions:
/// - In trending markets: Uses faster, more responsive calculations
/// - In ranging markets: Uses slower, more stable calculations
///
/// The adaptation is based on the Efficiency Ratio (ER), which measures how
/// efficiently price moves from point A to point B.
///
/// # Components
///
/// - Fast Score: Responsive to recent price changes
/// - Slow Score: Stable, smoothed signal
/// - Adaptive Weight: ER-based blend between fast and slow
/// - Market Condition: Assessment of current market state
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{AdaptiveMarketScore, AdaptiveMarketScoreConfig};
///
/// let config = AdaptiveMarketScoreConfig::default();
/// let indicator = AdaptiveMarketScore::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveMarketScore {
    fast_period: usize,
    slow_period: usize,
    efficiency_period: usize,
    volatility_period: usize,
}

impl AdaptiveMarketScore {
    /// Create a new AdaptiveMarketScore with the given configuration.
    pub fn new(config: AdaptiveMarketScoreConfig) -> Result<Self> {
        if config.fast_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.slow_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.efficiency_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "efficiency_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.fast_period >= config.slow_period {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be less than slow_period".to_string(),
            });
        }

        Ok(Self {
            fast_period: config.fast_period,
            slow_period: config.slow_period,
            efficiency_period: config.efficiency_period,
            volatility_period: config.volatility_period,
        })
    }

    /// Calculate the Adaptive Market Score values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> AdaptiveMarketScoreOutput {
        let n = close.len();
        let mut score = vec![f64::NAN; n];
        let mut market_condition = vec![f64::NAN; n];
        let mut adaptive_weight = vec![f64::NAN; n];

        let min_required = self
            .slow_period
            .max(self.efficiency_period)
            .max(self.volatility_period);
        if n <= min_required {
            return AdaptiveMarketScoreOutput {
                score,
                market_condition,
                adaptive_weight,
            };
        }

        // Calculate efficiency ratio
        let mut efficiency = vec![f64::NAN; n];
        for i in self.efficiency_period..n {
            let net_change = (close[i] - close[i - self.efficiency_period]).abs();
            let mut sum_changes = 0.0;
            for j in (i - self.efficiency_period + 1)..=i {
                sum_changes += (close[j] - close[j - 1]).abs();
            }

            if sum_changes > 1e-10 {
                efficiency[i] = (net_change / sum_changes).clamp(0.0, 1.0);
            } else {
                efficiency[i] = 0.0;
            }
        }

        // Calculate fast score (ROC-based)
        let mut fast_score = vec![f64::NAN; n];
        for i in self.fast_period..n {
            if close[i - self.fast_period].abs() > 1e-10 {
                let roc = ((close[i] - close[i - self.fast_period]) / close[i - self.fast_period])
                    * 100.0;
                fast_score[i] = (roc * 10.0).clamp(-100.0, 100.0);
            }
        }

        // Calculate slow score (smoothed deviation from SMA)
        let mut slow_score = vec![f64::NAN; n];
        for i in self.slow_period..n {
            let sma: f64 = close[(i - self.slow_period + 1)..=i].iter().sum::<f64>()
                / self.slow_period as f64;
            if sma.abs() > 1e-10 {
                let deviation = (close[i] - sma) / sma * 100.0;
                slow_score[i] = (deviation * 5.0).clamp(-100.0, 100.0);
            }
        }

        // Calculate volatility for market condition
        let mut atr = vec![f64::NAN; n];
        for i in 1..n {
            if i >= self.volatility_period {
                let mut sum = 0.0;
                for j in (i - self.volatility_period + 1)..=i {
                    let tr = if j == 0 {
                        high[j] - low[j]
                    } else {
                        (high[j] - low[j])
                            .max((high[j] - close[j - 1]).abs())
                            .max((low[j] - close[j - 1]).abs())
                    };
                    sum += tr;
                }
                atr[i] = sum / self.volatility_period as f64;
            }
        }

        // Calculate market condition and adaptive score
        for i in min_required..n {
            if !efficiency[i].is_nan() && !atr[i].is_nan() && close[i].abs() > 1e-10 {
                let er = efficiency[i];
                adaptive_weight[i] = er;

                // Market condition: combination of efficiency and volatility
                let atr_percent = atr[i] / close[i] * 100.0;
                // High efficiency + moderate volatility = good trending
                // Low efficiency + high volatility = choppy
                let trend_quality = er * 100.0;
                let vol_factor = (100.0 - atr_percent * 25.0).clamp(0.0, 100.0);
                market_condition[i] = (trend_quality * 0.7 + vol_factor * 0.3).clamp(0.0, 100.0);

                // Adaptive score: blend fast and slow based on efficiency
                if !fast_score[i].is_nan() && !slow_score[i].is_nan() {
                    // High efficiency = more weight to fast score
                    score[i] = (er * fast_score[i] + (1.0 - er) * slow_score[i]).clamp(-100.0, 100.0);
                }
            }
        }

        AdaptiveMarketScoreOutput {
            score,
            market_condition,
            adaptive_weight,
        }
    }
}

impl TechnicalIndicator for AdaptiveMarketScore {
    fn name(&self) -> &str {
        "AdaptiveMarketScore"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.score,
            result.market_condition,
            result.adaptive_weight,
        ))
    }

    fn min_periods(&self) -> usize {
        self.slow_period
            .max(self.efficiency_period)
            .max(self.volatility_period)
            + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 18. CompositeLeadingIndicator
// ============================================================================

/// Composite Leading Indicator output.
#[derive(Debug, Clone)]
pub struct CompositeLeadingIndicatorOutput {
    /// Combined leading indicator (-100 to 100).
    pub indicator: Vec<f64>,
    /// Price momentum lead component.
    pub price_lead: Vec<f64>,
    /// Volume lead component.
    pub volume_lead: Vec<f64>,
    /// Rate of change lead component.
    pub roc_lead: Vec<f64>,
}

/// Composite Leading Indicator configuration.
#[derive(Debug, Clone)]
pub struct CompositeLeadingIndicatorConfig {
    /// Short period for leading signals (default: 5).
    pub short_period: usize,
    /// Medium period for confirmation (default: 10).
    pub medium_period: usize,
    /// Long period for trend context (default: 20).
    pub long_period: usize,
    /// Volume period (default: 10).
    pub volume_period: usize,
    /// Smoothing period (default: 3).
    pub smoothing_period: usize,
}

impl Default for CompositeLeadingIndicatorConfig {
    fn default() -> Self {
        Self {
            short_period: 5,
            medium_period: 10,
            long_period: 20,
            volume_period: 10,
            smoothing_period: 3,
        }
    }
}

/// Composite Leading Indicator.
///
/// A forward-looking composite indicator designed to identify potential market
/// turning points by combining multiple leading signals:
///
/// - Price Lead: Short-term price momentum diverging from longer-term trend
/// - Volume Lead: Volume patterns that often precede price movements
/// - ROC Lead: Acceleration/deceleration of price change
///
/// The indicator attempts to signal market changes before they occur by
/// identifying early signs of momentum shifts.
///
/// # Signal Interpretation
///
/// - Rising from low values: Potential bullish reversal
/// - Falling from high values: Potential bearish reversal
/// - Divergence from price: Warning of potential trend change
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{CompositeLeadingIndicator, CompositeLeadingIndicatorConfig};
///
/// let config = CompositeLeadingIndicatorConfig::default();
/// let indicator = CompositeLeadingIndicator::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close, &volume);
/// ```
#[derive(Debug, Clone)]
pub struct CompositeLeadingIndicator {
    short_period: usize,
    medium_period: usize,
    long_period: usize,
    volume_period: usize,
    smoothing_period: usize,
}

impl CompositeLeadingIndicator {
    /// Create a new CompositeLeadingIndicator with the given configuration.
    pub fn new(config: CompositeLeadingIndicatorConfig) -> Result<Self> {
        if config.short_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.medium_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.long_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volume_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.short_period >= config.medium_period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than medium_period".to_string(),
            });
        }
        if config.medium_period >= config.long_period {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be less than long_period".to_string(),
            });
        }

        Ok(Self {
            short_period: config.short_period,
            medium_period: config.medium_period,
            long_period: config.long_period,
            volume_period: config.volume_period,
            smoothing_period: config.smoothing_period,
        })
    }

    /// Calculate the Composite Leading Indicator values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> CompositeLeadingIndicatorOutput {
        let n = close.len();
        let mut indicator = vec![f64::NAN; n];
        let mut price_lead = vec![f64::NAN; n];
        let mut volume_lead = vec![f64::NAN; n];
        let mut roc_lead = vec![f64::NAN; n];

        let min_required = self.long_period.max(self.volume_period);
        if n <= min_required {
            return CompositeLeadingIndicatorOutput {
                indicator,
                price_lead,
                volume_lead,
                roc_lead,
            };
        }

        // Calculate EMAs for price lead
        let short_ema = self.calculate_ema(close, self.short_period);
        let medium_ema = self.calculate_ema(close, self.medium_period);
        let long_ema = self.calculate_ema(close, self.long_period);

        // Calculate price lead (short-term deviation from longer-term)
        for i in self.long_period..n {
            if !short_ema[i].is_nan() && !medium_ema[i].is_nan() && !long_ema[i].is_nan() {
                if long_ema[i].abs() > 1e-10 {
                    // Short vs long deviation
                    let short_dev = (short_ema[i] - long_ema[i]) / long_ema[i] * 100.0;
                    // Medium vs long deviation (for confirmation)
                    let medium_dev = (medium_ema[i] - long_ema[i]) / long_ema[i] * 100.0;

                    // Price lead = short deviation with medium confirmation
                    let confirmation = if short_dev.signum() == medium_dev.signum() {
                        1.2
                    } else {
                        0.8
                    };
                    price_lead[i] = (short_dev * 10.0 * confirmation).clamp(-100.0, 100.0);
                }
            }
        }

        // Calculate volume lead (volume patterns preceding price)
        for i in self.volume_period..n {
            let avg_volume: f64 = volume[(i - self.volume_period + 1)..=i]
                .iter()
                .sum::<f64>()
                / self.volume_period as f64;

            if avg_volume > 1e-10 {
                // Volume ratio
                let vol_ratio = volume[i] / avg_volume;

                // Volume trend (is volume increasing?)
                let vol_prev: f64 = if i >= self.volume_period + self.short_period {
                    volume[(i - self.volume_period - self.short_period + 1)..=(i - self.short_period)]
                        .iter()
                        .sum::<f64>()
                        / self.volume_period as f64
                } else {
                    avg_volume
                };

                let vol_change = if vol_prev > 1e-10 {
                    (avg_volume - vol_prev) / vol_prev
                } else {
                    0.0
                };

                // Price direction for volume context
                let price_dir = if close[i] > close[i - 1] {
                    1.0
                } else if close[i] < close[i - 1] {
                    -1.0
                } else {
                    0.0
                };

                // Volume lead: rising volume in direction of move is leading
                volume_lead[i] =
                    (price_dir * (vol_ratio - 1.0) * 30.0 + vol_change * 50.0).clamp(-100.0, 100.0);
            }
        }

        // Calculate ROC lead (acceleration of price change)
        for i in self.medium_period..n {
            // Current ROC
            let roc_current = if close[i - self.short_period].abs() > 1e-10 {
                ((close[i] - close[i - self.short_period]) / close[i - self.short_period]) * 100.0
            } else {
                0.0
            };

            // Previous ROC
            let roc_prev = if i >= self.short_period + 1
                && close[i - self.short_period - 1].abs() > 1e-10
            {
                ((close[i - 1] - close[i - self.short_period - 1])
                    / close[i - self.short_period - 1])
                    * 100.0
            } else {
                0.0
            };

            // ROC of ROC (acceleration)
            let roc_accel = roc_current - roc_prev;

            // Longer-term ROC for context
            let roc_long = if close[i - self.medium_period].abs() > 1e-10 {
                ((close[i] - close[i - self.medium_period]) / close[i - self.medium_period]) * 100.0
            } else {
                0.0
            };

            // ROC lead: acceleration with trend context
            roc_lead[i] = (roc_accel * 20.0 + roc_current * 2.0).clamp(-100.0, 100.0);
        }

        // Combine leading components
        for i in min_required..n {
            if !price_lead[i].is_nan() && !volume_lead[i].is_nan() && !roc_lead[i].is_nan() {
                // Weight components
                let raw_indicator =
                    price_lead[i] * 0.4 + volume_lead[i] * 0.3 + roc_lead[i] * 0.3;

                // Agreement bonus
                let signs = [
                    price_lead[i].signum(),
                    volume_lead[i].signum(),
                    roc_lead[i].signum(),
                ];
                let positive_count = signs.iter().filter(|&&s| s > 0.0).count();
                let negative_count = signs.iter().filter(|&&s| s < 0.0).count();
                let agreement = positive_count.max(negative_count) as f64 / 3.0;

                indicator[i] = (raw_indicator * (0.7 + agreement * 0.6)).clamp(-100.0, 100.0);
            }
        }

        // Apply smoothing
        if self.smoothing_period > 1 {
            let smoothed = self.ema_smooth(&indicator, self.smoothing_period);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    indicator[i] = smoothed[i];
                }
            }
        }

        CompositeLeadingIndicatorOutput {
            indicator,
            price_lead,
            volume_lead,
            roc_lead,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for CompositeLeadingIndicator {
    fn name(&self) -> &str {
        "CompositeLeadingIndicator"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            result.indicator,
            result.price_lead,
            result.volume_lead,
        ))
    }

    fn min_periods(&self) -> usize {
        self.long_period.max(self.volume_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 19. TechnicalScorecard
// ============================================================================

/// Technical Scorecard output.
#[derive(Debug, Clone)]
pub struct TechnicalScorecardOutput {
    /// Overall technical score (0-100).
    pub score: Vec<f64>,
    /// Trend sub-score (0-100).
    pub trend_score: Vec<f64>,
    /// Momentum sub-score (0-100).
    pub momentum_score: Vec<f64>,
    /// Volatility sub-score (0-100).
    pub volatility_score: Vec<f64>,
}

/// Technical Scorecard configuration.
#[derive(Debug, Clone)]
pub struct TechnicalScorecardConfig {
    /// Period for trend analysis (default: 20).
    pub trend_period: usize,
    /// Period for momentum analysis (default: 14).
    pub momentum_period: usize,
    /// Period for volatility analysis (default: 14).
    pub volatility_period: usize,
    /// Weight for trend score (default: 0.4).
    pub trend_weight: f64,
    /// Weight for momentum score (default: 0.35).
    pub momentum_weight: f64,
    /// Weight for volatility score (default: 0.25).
    pub volatility_weight: f64,
}

impl Default for TechnicalScorecardConfig {
    fn default() -> Self {
        Self {
            trend_period: 20,
            momentum_period: 14,
            volatility_period: 14,
            trend_weight: 0.4,
            momentum_weight: 0.35,
            volatility_weight: 0.25,
        }
    }
}

/// Technical Scorecard.
///
/// A comprehensive technical analysis scoring system that evaluates market conditions
/// across multiple dimensions and produces a unified score from 0-100.
///
/// # Components
///
/// - **Trend Score**: Measures trend strength using EMA alignment and directional movement
/// - **Momentum Score**: RSI-based momentum with acceleration/deceleration analysis
/// - **Volatility Score**: ATR-based volatility assessment relative to historical norms
///
/// # Scoring Interpretation
///
/// - 80-100: Excellent technical conditions (strong trend, momentum, low volatility)
/// - 60-80: Good conditions with minor weaknesses
/// - 40-60: Neutral/mixed conditions
/// - 20-40: Weak conditions, caution advised
/// - 0-20: Poor technical conditions
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{TechnicalScorecard, TechnicalScorecardConfig};
///
/// let config = TechnicalScorecardConfig::default();
/// let indicator = TechnicalScorecard::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close, &volume);
/// ```
#[derive(Debug, Clone)]
pub struct TechnicalScorecard {
    trend_period: usize,
    momentum_period: usize,
    volatility_period: usize,
    trend_weight: f64,
    momentum_weight: f64,
    volatility_weight: f64,
}

impl TechnicalScorecard {
    /// Create a new TechnicalScorecard with the given configuration.
    pub fn new(config: TechnicalScorecardConfig) -> Result<Self> {
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.momentum_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.trend_weight < 0.0 || config.trend_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.momentum_weight < 0.0 || config.momentum_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.volatility_weight < 0.0 || config.volatility_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(Self {
            trend_period: config.trend_period,
            momentum_period: config.momentum_period,
            volatility_period: config.volatility_period,
            trend_weight: config.trend_weight,
            momentum_weight: config.momentum_weight,
            volatility_weight: config.volatility_weight,
        })
    }

    /// Calculate the Technical Scorecard values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        _volume: &[f64],
    ) -> TechnicalScorecardOutput {
        let n = close.len();
        let mut score = vec![f64::NAN; n];
        let mut trend_score = vec![f64::NAN; n];
        let mut momentum_score = vec![f64::NAN; n];
        let mut volatility_score = vec![f64::NAN; n];

        let max_period = self.trend_period.max(self.momentum_period).max(self.volatility_period);
        if n <= max_period {
            return TechnicalScorecardOutput {
                score,
                trend_score,
                momentum_score,
                volatility_score,
            };
        }

        // Calculate trend score
        let ema_short = self.calculate_ema(close, self.trend_period / 2);
        let ema_long = self.calculate_ema(close, self.trend_period);

        for i in self.trend_period..n {
            if !ema_short[i].is_nan() && !ema_long[i].is_nan() && ema_long[i].abs() > 1e-10 {
                // EMA alignment score
                let alignment = if ema_short[i] > ema_long[i] && close[i] > ema_short[i] {
                    100.0
                } else if ema_short[i] > ema_long[i] {
                    75.0
                } else if close[i] > ema_long[i] {
                    50.0
                } else {
                    25.0
                };

                // Efficiency ratio component
                let net_change = (close[i] - close[i - self.trend_period]).abs();
                let mut sum_changes = 0.0;
                for j in (i - self.trend_period + 1)..=i {
                    sum_changes += (close[j] - close[j - 1]).abs();
                }
                let efficiency = if sum_changes > 1e-10 {
                    (net_change / sum_changes * 100.0).clamp(0.0, 100.0)
                } else {
                    0.0
                };

                trend_score[i] = (alignment * 0.6 + efficiency * 0.4).clamp(0.0, 100.0);
            }
        }

        // Calculate momentum score (RSI-based)
        for i in self.momentum_period..n {
            let mut gains = 0.0;
            let mut losses = 0.0;
            for j in (i - self.momentum_period + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += -change;
                }
            }

            let avg_gain = gains / self.momentum_period as f64;
            let avg_loss = losses / self.momentum_period as f64;

            let rsi = if avg_loss > 1e-10 {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            } else if avg_gain > 0.0 {
                100.0
            } else {
                50.0
            };

            // Convert RSI to score (50 is neutral)
            // RSI 30-70 is normal, extremes are less favorable
            let rsi_score = if rsi >= 30.0 && rsi <= 70.0 {
                // Normal range, score based on direction
                50.0 + (rsi - 50.0) * 1.0
            } else if rsi > 70.0 {
                // Overbought, still bullish but decreasing score
                100.0 - (rsi - 70.0)
            } else {
                // Oversold, potential reversal
                rsi * 1.5
            };

            momentum_score[i] = rsi_score.clamp(0.0, 100.0);
        }

        // Calculate volatility score (lower volatility = higher score)
        let mut atr = vec![f64::NAN; n];
        for i in 1..n {
            if i >= self.volatility_period {
                let mut sum = 0.0;
                for j in (i - self.volatility_period + 1)..=i {
                    let tr = if j == 0 {
                        high[j] - low[j]
                    } else {
                        (high[j] - low[j])
                            .max((high[j] - close[j - 1]).abs())
                            .max((low[j] - close[j - 1]).abs())
                    };
                    sum += tr;
                }
                atr[i] = sum / self.volatility_period as f64;
            }
        }

        for i in max_period..n {
            if !atr[i].is_nan() && close[i].abs() > 1e-10 {
                let atr_percent = atr[i] / close[i] * 100.0;
                // Lower ATR% = higher score (stable conditions preferred)
                // 0.5% = 100, 1% = 75, 2% = 50, 3% = 25, 4%+ = 0
                volatility_score[i] = (100.0 - atr_percent * 25.0).clamp(0.0, 100.0);
            }
        }

        // Combine all scores
        for i in max_period..n {
            if !trend_score[i].is_nan()
                && !momentum_score[i].is_nan()
                && !volatility_score[i].is_nan()
            {
                score[i] = (self.trend_weight * trend_score[i]
                    + self.momentum_weight * momentum_score[i]
                    + self.volatility_weight * volatility_score[i])
                    .clamp(0.0, 100.0);
            }
        }

        TechnicalScorecardOutput {
            score,
            trend_score,
            momentum_score,
            volatility_score,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period && period > 0 {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }
}

impl TechnicalIndicator for TechnicalScorecard {
    fn name(&self) -> &str {
        "TechnicalScorecard"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            result.score,
            result.trend_score,
            result.momentum_score,
        ))
    }

    fn min_periods(&self) -> usize {
        self.trend_period
            .max(self.momentum_period)
            .max(self.volatility_period)
            + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 20. TrendMomentumVolume
// ============================================================================

/// Trend Momentum Volume output.
#[derive(Debug, Clone)]
pub struct TrendMomentumVolumeOutput {
    /// Combined trend-momentum-volume indicator (-100 to 100).
    pub indicator: Vec<f64>,
    /// Trend component (-100 to 100).
    pub trend_component: Vec<f64>,
    /// Momentum component (-100 to 100).
    pub momentum_component: Vec<f64>,
    /// Volume component (-100 to 100).
    pub volume_component: Vec<f64>,
}

/// Trend Momentum Volume configuration.
#[derive(Debug, Clone)]
pub struct TrendMomentumVolumeConfig {
    /// Period for trend analysis (default: 20).
    pub trend_period: usize,
    /// Period for momentum analysis (default: 14).
    pub momentum_period: usize,
    /// Period for volume analysis (default: 10).
    pub volume_period: usize,
    /// Weight for trend (default: 0.4).
    pub trend_weight: f64,
    /// Weight for momentum (default: 0.35).
    pub momentum_weight: f64,
    /// Weight for volume (default: 0.25).
    pub volume_weight: f64,
}

impl Default for TrendMomentumVolumeConfig {
    fn default() -> Self {
        Self {
            trend_period: 20,
            momentum_period: 14,
            volume_period: 10,
            trend_weight: 0.4,
            momentum_weight: 0.35,
            volume_weight: 0.25,
        }
    }
}

/// Trend Momentum Volume.
///
/// A composite indicator that integrates trend direction, momentum strength,
/// and volume confirmation into a single unified signal.
///
/// # Components
///
/// - **Trend**: Based on price position relative to EMAs and trend direction
/// - **Momentum**: Rate of change with smoothing for noise reduction
/// - **Volume**: Volume-weighted price confirmation
///
/// # Signal Interpretation
///
/// - Values > 50: Strong bullish alignment across all factors
/// - Values 0 to 50: Moderate bullish conditions
/// - Values -50 to 0: Moderate bearish conditions
/// - Values < -50: Strong bearish alignment
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{TrendMomentumVolume, TrendMomentumVolumeConfig};
///
/// let config = TrendMomentumVolumeConfig::default();
/// let indicator = TrendMomentumVolume::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close, &volume);
/// ```
#[derive(Debug, Clone)]
pub struct TrendMomentumVolume {
    trend_period: usize,
    momentum_period: usize,
    volume_period: usize,
    trend_weight: f64,
    momentum_weight: f64,
    volume_weight: f64,
}

impl TrendMomentumVolume {
    /// Create a new TrendMomentumVolume with the given configuration.
    pub fn new(config: TrendMomentumVolumeConfig) -> Result<Self> {
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.momentum_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volume_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.trend_weight < 0.0 || config.trend_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.momentum_weight < 0.0 || config.momentum_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if config.volume_weight < 0.0 || config.volume_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(Self {
            trend_period: config.trend_period,
            momentum_period: config.momentum_period,
            volume_period: config.volume_period,
            trend_weight: config.trend_weight,
            momentum_weight: config.momentum_weight,
            volume_weight: config.volume_weight,
        })
    }

    /// Calculate the Trend Momentum Volume values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> TrendMomentumVolumeOutput {
        let n = close.len();
        let mut indicator = vec![f64::NAN; n];
        let mut trend_component = vec![f64::NAN; n];
        let mut momentum_component = vec![f64::NAN; n];
        let mut volume_component = vec![f64::NAN; n];

        let max_period = self.trend_period.max(self.momentum_period).max(self.volume_period);
        if n <= max_period {
            return TrendMomentumVolumeOutput {
                indicator,
                trend_component,
                momentum_component,
                volume_component,
            };
        }

        // Calculate trend component
        let ema = self.calculate_ema(close, self.trend_period);
        for i in self.trend_period..n {
            if !ema[i].is_nan() && ema[i].abs() > 1e-10 {
                // Deviation from EMA
                let deviation = (close[i] - ema[i]) / ema[i] * 100.0;

                // EMA slope
                let slope = if i > 0 && !ema[i - 1].is_nan() && ema[i - 1].abs() > 1e-10 {
                    (ema[i] - ema[i - 1]) / ema[i - 1] * 100.0
                } else {
                    0.0
                };

                trend_component[i] = (deviation * 5.0 + slope * 50.0).clamp(-100.0, 100.0);
            }
        }

        // Calculate momentum component (smoothed ROC)
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                let roc = ((close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period])
                    * 100.0;

                // Add acceleration
                let prev_roc = if i > self.momentum_period
                    && close[i - self.momentum_period - 1].abs() > 1e-10
                {
                    ((close[i - 1] - close[i - self.momentum_period - 1])
                        / close[i - self.momentum_period - 1])
                        * 100.0
                } else {
                    roc
                };

                let acceleration = roc - prev_roc;
                momentum_component[i] = (roc * 5.0 + acceleration * 10.0).clamp(-100.0, 100.0);
            }
        }

        // Calculate volume component
        for i in self.volume_period..n {
            let avg_volume: f64 = volume[(i - self.volume_period + 1)..=i].iter().sum::<f64>()
                / self.volume_period as f64;

            if avg_volume > 1e-10 {
                let vol_ratio = volume[i] / avg_volume;

                // Direction based on price change
                let direction = if close[i] > close[i - 1] {
                    1.0
                } else if close[i] < close[i - 1] {
                    -1.0
                } else {
                    0.0
                };

                // Volume-weighted directional signal
                volume_component[i] = (direction * (vol_ratio - 0.5) * 100.0).clamp(-100.0, 100.0);
            }
        }

        // Combine components
        for i in max_period..n {
            if !trend_component[i].is_nan()
                && !momentum_component[i].is_nan()
                && !volume_component[i].is_nan()
            {
                // Check alignment for confirmation bonus
                let signs = [
                    trend_component[i].signum(),
                    momentum_component[i].signum(),
                    volume_component[i].signum(),
                ];
                let positive = signs.iter().filter(|&&s| s > 0.0).count();
                let negative = signs.iter().filter(|&&s| s < 0.0).count();
                let alignment_bonus = if positive >= 2 || negative >= 2 { 1.15 } else { 0.9 };

                indicator[i] = ((self.trend_weight * trend_component[i]
                    + self.momentum_weight * momentum_component[i]
                    + self.volume_weight * volume_component[i])
                    * alignment_bonus)
                    .clamp(-100.0, 100.0);
            }
        }

        TrendMomentumVolumeOutput {
            indicator,
            trend_component,
            momentum_component,
            volume_component,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }
}

impl TechnicalIndicator for TrendMomentumVolume {
    fn name(&self) -> &str {
        "TrendMomentumVolume"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            result.indicator,
            result.trend_component,
            result.momentum_component,
        ))
    }

    fn min_periods(&self) -> usize {
        self.trend_period
            .max(self.momentum_period)
            .max(self.volume_period)
            + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 21. MultiIndicatorConfluence
// ============================================================================

/// Multi-Indicator Confluence output.
#[derive(Debug, Clone)]
pub struct MultiIndicatorConfluenceOutput {
    /// Confluence score (0-100).
    pub confluence: Vec<f64>,
    /// Number of agreeing indicators (0-5).
    pub agreement_count: Vec<f64>,
    /// Average signal strength.
    pub avg_signal_strength: Vec<f64>,
}

/// Multi-Indicator Confluence configuration.
#[derive(Debug, Clone)]
pub struct MultiIndicatorConfluenceConfig {
    /// RSI period (default: 14).
    pub rsi_period: usize,
    /// MACD fast period (default: 12).
    pub macd_fast: usize,
    /// MACD slow period (default: 26).
    pub macd_slow: usize,
    /// Stochastic period (default: 14).
    pub stoch_period: usize,
    /// EMA period for trend (default: 20).
    pub ema_period: usize,
}

impl Default for MultiIndicatorConfluenceConfig {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            stoch_period: 14,
            ema_period: 20,
        }
    }
}

/// Multi-Indicator Confluence.
///
/// Measures the degree of agreement between multiple popular technical indicators.
/// When multiple indicators agree, the signal has higher confidence.
///
/// # Indicators Evaluated
///
/// 1. **RSI**: Bullish if > 50, bearish if < 50
/// 2. **MACD**: Bullish if MACD > signal line
/// 3. **Stochastic**: Bullish if %K > 50
/// 4. **EMA Trend**: Bullish if price > EMA
/// 5. **Momentum**: Bullish if ROC > 0
///
/// # Confluence Score
///
/// - 100: All indicators agree (strong signal)
/// - 80: 4 of 5 agree
/// - 60: 3 of 5 agree
/// - 40: 2 of 5 agree (mixed)
/// - 20: 1 of 5 agree
/// - 0: No agreement
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{MultiIndicatorConfluence, MultiIndicatorConfluenceConfig};
///
/// let config = MultiIndicatorConfluenceConfig::default();
/// let indicator = MultiIndicatorConfluence::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close);
/// ```
#[derive(Debug, Clone)]
pub struct MultiIndicatorConfluence {
    rsi_period: usize,
    macd_fast: usize,
    macd_slow: usize,
    stoch_period: usize,
    ema_period: usize,
}

impl MultiIndicatorConfluence {
    /// Create a new MultiIndicatorConfluence with the given configuration.
    pub fn new(config: MultiIndicatorConfluenceConfig) -> Result<Self> {
        if config.rsi_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "rsi_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.macd_fast == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "macd_fast".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.macd_slow == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "macd_slow".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.stoch_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "stoch_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.ema_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "ema_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.macd_fast >= config.macd_slow {
            return Err(IndicatorError::InvalidParameter {
                name: "macd_fast".to_string(),
                reason: "must be less than macd_slow".to_string(),
            });
        }

        Ok(Self {
            rsi_period: config.rsi_period,
            macd_fast: config.macd_fast,
            macd_slow: config.macd_slow,
            stoch_period: config.stoch_period,
            ema_period: config.ema_period,
        })
    }

    /// Calculate the Multi-Indicator Confluence values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> MultiIndicatorConfluenceOutput {
        let n = close.len();
        let mut confluence = vec![f64::NAN; n];
        let mut agreement_count = vec![f64::NAN; n];
        let mut avg_signal_strength = vec![f64::NAN; n];

        let max_period = self.macd_slow.max(self.rsi_period).max(self.stoch_period).max(self.ema_period);
        if n <= max_period {
            return MultiIndicatorConfluenceOutput {
                confluence,
                agreement_count,
                avg_signal_strength,
            };
        }

        // Calculate RSI
        let mut rsi = vec![f64::NAN; n];
        for i in self.rsi_period..n {
            let mut gains = 0.0;
            let mut losses = 0.0;
            for j in (i - self.rsi_period + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += -change;
                }
            }
            let avg_gain = gains / self.rsi_period as f64;
            let avg_loss = losses / self.rsi_period as f64;
            rsi[i] = if avg_loss > 1e-10 {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            } else if avg_gain > 0.0 {
                100.0
            } else {
                50.0
            };
        }

        // Calculate MACD
        let ema_fast = self.calculate_ema(close, self.macd_fast);
        let ema_slow = self.calculate_ema(close, self.macd_slow);
        let mut macd_line = vec![f64::NAN; n];
        for i in self.macd_slow..n {
            if !ema_fast[i].is_nan() && !ema_slow[i].is_nan() {
                macd_line[i] = ema_fast[i] - ema_slow[i];
            }
        }
        let signal_line = self.calculate_ema(&macd_line, 9);

        // Calculate Stochastic %K
        let mut stoch_k = vec![f64::NAN; n];
        for i in self.stoch_period..n {
            let window_high = high[(i - self.stoch_period + 1)..=i]
                .iter()
                .cloned()
                .fold(f64::MIN, f64::max);
            let window_low = low[(i - self.stoch_period + 1)..=i]
                .iter()
                .cloned()
                .fold(f64::MAX, f64::min);
            let range = window_high - window_low;
            stoch_k[i] = if range > 1e-10 {
                ((close[i] - window_low) / range * 100.0).clamp(0.0, 100.0)
            } else {
                50.0
            };
        }

        // Calculate EMA for trend
        let trend_ema = self.calculate_ema(close, self.ema_period);

        // Calculate momentum (ROC)
        let mut momentum = vec![f64::NAN; n];
        let mom_period = 10;
        for i in mom_period..n {
            if close[i - mom_period].abs() > 1e-10 {
                momentum[i] = ((close[i] - close[i - mom_period]) / close[i - mom_period]) * 100.0;
            }
        }

        // Calculate confluence
        for i in max_period..n {
            let mut bullish_count = 0;
            let mut bearish_count = 0;
            let mut signals: Vec<f64> = Vec::new();

            // RSI signal
            if !rsi[i].is_nan() {
                if rsi[i] > 50.0 {
                    bullish_count += 1;
                    signals.push((rsi[i] - 50.0) * 2.0);
                } else {
                    bearish_count += 1;
                    signals.push((rsi[i] - 50.0) * 2.0);
                }
            }

            // MACD signal
            if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
                let macd_diff = macd_line[i] - signal_line[i];
                if macd_diff > 0.0 {
                    bullish_count += 1;
                    signals.push(macd_diff.abs().min(5.0) * 20.0);
                } else {
                    bearish_count += 1;
                    signals.push(-macd_diff.abs().min(5.0) * 20.0);
                }
            }

            // Stochastic signal
            if !stoch_k[i].is_nan() {
                if stoch_k[i] > 50.0 {
                    bullish_count += 1;
                    signals.push((stoch_k[i] - 50.0) * 2.0);
                } else {
                    bearish_count += 1;
                    signals.push((stoch_k[i] - 50.0) * 2.0);
                }
            }

            // EMA trend signal
            if !trend_ema[i].is_nan() {
                if close[i] > trend_ema[i] {
                    bullish_count += 1;
                    signals.push(((close[i] - trend_ema[i]) / trend_ema[i] * 100.0).min(100.0));
                } else {
                    bearish_count += 1;
                    signals.push(((close[i] - trend_ema[i]) / trend_ema[i] * 100.0).max(-100.0));
                }
            }

            // Momentum signal
            if !momentum[i].is_nan() {
                if momentum[i] > 0.0 {
                    bullish_count += 1;
                    signals.push(momentum[i].min(100.0));
                } else {
                    bearish_count += 1;
                    signals.push(momentum[i].max(-100.0));
                }
            }

            // Calculate confluence
            let max_agreement = bullish_count.max(bearish_count);
            agreement_count[i] = max_agreement as f64;
            confluence[i] = (max_agreement as f64 / 5.0 * 100.0).clamp(0.0, 100.0);

            // Average signal strength
            if !signals.is_empty() {
                avg_signal_strength[i] = signals.iter().sum::<f64>() / signals.len() as f64;
            }
        }

        MultiIndicatorConfluenceOutput {
            confluence,
            agreement_count,
            avg_signal_strength,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        // Find first valid value
        let mut first_valid = None;
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..n {
            if !data[i].is_nan() {
                sum += data[i];
                count += 1;
                if count == period {
                    first_valid = Some(i);
                    ema[i] = sum / period as f64;
                    break;
                }
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() && !ema[i - 1].is_nan() {
                    ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
                }
            }
        }

        ema
    }
}

impl TechnicalIndicator for MultiIndicatorConfluence {
    fn name(&self) -> &str {
        "MultiIndicatorConfluence"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.confluence,
            result.agreement_count,
            result.avg_signal_strength,
        ))
    }

    fn min_periods(&self) -> usize {
        self.macd_slow.max(self.rsi_period).max(self.stoch_period).max(self.ema_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 22. MarketConditionIndex
// ============================================================================

/// Market Condition Index output.
#[derive(Debug, Clone)]
pub struct MarketConditionIndexOutput {
    /// Overall market condition index (0-100).
    pub index: Vec<f64>,
    /// Trend condition (0-100).
    pub trend_condition: Vec<f64>,
    /// Volatility condition (0-100).
    pub volatility_condition: Vec<f64>,
    /// Momentum condition (0-100).
    pub momentum_condition: Vec<f64>,
}

/// Market Condition Index configuration.
#[derive(Debug, Clone)]
pub struct MarketConditionIndexConfig {
    /// Period for trend analysis (default: 20).
    pub trend_period: usize,
    /// Period for volatility analysis (default: 14).
    pub volatility_period: usize,
    /// Period for momentum analysis (default: 10).
    pub momentum_period: usize,
    /// Smoothing period (default: 5).
    pub smoothing_period: usize,
}

impl Default for MarketConditionIndexConfig {
    fn default() -> Self {
        Self {
            trend_period: 20,
            volatility_period: 14,
            momentum_period: 10,
            smoothing_period: 5,
        }
    }
}

/// Market Condition Index.
///
/// Provides an overall assessment of market conditions by evaluating multiple
/// dimensions of market behavior and producing a normalized index from 0-100.
///
/// # Components
///
/// - **Trend Condition**: How clearly defined the current trend is
/// - **Volatility Condition**: Current volatility relative to historical norms
/// - **Momentum Condition**: Strength and consistency of price momentum
///
/// # Index Interpretation
///
/// - 80-100: Excellent market conditions (clear trend, stable volatility)
/// - 60-80: Good conditions suitable for most strategies
/// - 40-60: Mixed conditions, selective approach recommended
/// - 20-40: Challenging conditions, reduced exposure advised
/// - 0-20: Poor conditions, high caution warranted
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{MarketConditionIndex, MarketConditionIndexConfig};
///
/// let config = MarketConditionIndexConfig::default();
/// let indicator = MarketConditionIndex::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close);
/// ```
#[derive(Debug, Clone)]
pub struct MarketConditionIndex {
    trend_period: usize,
    volatility_period: usize,
    momentum_period: usize,
    smoothing_period: usize,
}

impl MarketConditionIndex {
    /// Create a new MarketConditionIndex with the given configuration.
    pub fn new(config: MarketConditionIndexConfig) -> Result<Self> {
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.momentum_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            trend_period: config.trend_period,
            volatility_period: config.volatility_period,
            momentum_period: config.momentum_period,
            smoothing_period: config.smoothing_period,
        })
    }

    /// Calculate the Market Condition Index values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> MarketConditionIndexOutput {
        let n = close.len();
        let mut index = vec![f64::NAN; n];
        let mut trend_condition = vec![f64::NAN; n];
        let mut volatility_condition = vec![f64::NAN; n];
        let mut momentum_condition = vec![f64::NAN; n];

        let max_period = self.trend_period.max(self.volatility_period).max(self.momentum_period);
        if n <= max_period {
            return MarketConditionIndexOutput {
                index,
                trend_condition,
                volatility_condition,
                momentum_condition,
            };
        }

        // Calculate trend condition using efficiency ratio
        for i in self.trend_period..n {
            let net_change = (close[i] - close[i - self.trend_period]).abs();
            let mut sum_changes = 0.0;
            for j in (i - self.trend_period + 1)..=i {
                sum_changes += (close[j] - close[j - 1]).abs();
            }

            if sum_changes > 1e-10 {
                // Efficiency ratio: 1.0 = perfect trend, 0.0 = random walk
                let efficiency = net_change / sum_changes;
                trend_condition[i] = (efficiency * 100.0).clamp(0.0, 100.0);
            } else {
                trend_condition[i] = 50.0;
            }
        }

        // Calculate volatility condition
        let mut atr = vec![f64::NAN; n];
        let mut atr_history: Vec<f64> = Vec::new();

        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i >= self.volatility_period {
                let mut sum = 0.0;
                for j in (i - self.volatility_period + 1)..=i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j.saturating_sub(1)]).abs())
                        .max((low[j] - close[j.saturating_sub(1)]).abs());
                    sum += tr_j;
                }
                atr[i] = sum / self.volatility_period as f64;
                atr_history.push(atr[i]);
            }
        }

        // Calculate volatility percentile
        for i in max_period..n {
            if !atr[i].is_nan() && close[i].abs() > 1e-10 {
                let atr_percent = atr[i] / close[i] * 100.0;
                // Moderate volatility is best: 0.5-1.5% is ideal
                // Too low = no movement, too high = dangerous
                let ideal_vol = 1.0;
                let vol_deviation = (atr_percent - ideal_vol).abs();
                volatility_condition[i] = (100.0 - vol_deviation * 40.0).clamp(0.0, 100.0);
            }
        }

        // Calculate momentum condition
        for i in self.momentum_period..n {
            // Direction consistency
            let mut same_direction = 0;
            let overall_direction = (close[i] - close[i - self.momentum_period]).signum();

            for j in (i - self.momentum_period + 1)..=i {
                let bar_direction = (close[j] - close[j - 1]).signum();
                if bar_direction == overall_direction || bar_direction == 0.0 {
                    same_direction += 1;
                }
            }

            let consistency = same_direction as f64 / self.momentum_period as f64;

            // Momentum magnitude
            let roc = if close[i - self.momentum_period].abs() > 1e-10 {
                ((close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]).abs()
                    * 100.0
            } else {
                0.0
            };

            // Combine consistency and magnitude
            let momentum_quality = (consistency * 50.0 + (roc * 5.0).min(50.0)).clamp(0.0, 100.0);
            momentum_condition[i] = momentum_quality;
        }

        // Combine into overall index
        for i in max_period..n {
            if !trend_condition[i].is_nan()
                && !volatility_condition[i].is_nan()
                && !momentum_condition[i].is_nan()
            {
                index[i] = (trend_condition[i] * 0.4
                    + volatility_condition[i] * 0.3
                    + momentum_condition[i] * 0.3)
                    .clamp(0.0, 100.0);
            }
        }

        // Apply smoothing
        if self.smoothing_period > 1 {
            let smoothed = self.ema_smooth(&index, self.smoothing_period);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    index[i] = smoothed[i];
                }
            }
        }

        MarketConditionIndexOutput {
            index,
            trend_condition,
            volatility_condition,
            momentum_condition,
        }
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for MarketConditionIndex {
    fn name(&self) -> &str {
        "MarketConditionIndex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.index,
            result.trend_condition,
            result.volatility_condition,
        ))
    }

    fn min_periods(&self) -> usize {
        self.trend_period.max(self.volatility_period).max(self.momentum_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 23. SignalStrengthComposite
// ============================================================================

/// Signal Strength Composite output.
#[derive(Debug, Clone)]
pub struct SignalStrengthCompositeOutput {
    /// Combined signal strength (-100 to 100).
    pub strength: Vec<f64>,
    /// Primary signal component.
    pub primary_signal: Vec<f64>,
    /// Confirmation signal component.
    pub confirmation_signal: Vec<f64>,
    /// Quality score (0-100).
    pub quality_score: Vec<f64>,
}

/// Signal Strength Composite configuration.
#[derive(Debug, Clone)]
pub struct SignalStrengthCompositeConfig {
    /// Primary signal period (default: 14).
    pub primary_period: usize,
    /// Confirmation period (default: 20).
    pub confirmation_period: usize,
    /// Quality assessment period (default: 10).
    pub quality_period: usize,
    /// Smoothing period (default: 3).
    pub smoothing_period: usize,
}

impl Default for SignalStrengthCompositeConfig {
    fn default() -> Self {
        Self {
            primary_period: 14,
            confirmation_period: 20,
            quality_period: 10,
            smoothing_period: 3,
        }
    }
}

/// Signal Strength Composite.
///
/// Combines multiple signal sources with a quality-weighted approach to produce
/// a robust trading signal with strength measurement.
///
/// # Components
///
/// - **Primary Signal**: Fast-responding momentum-based signal
/// - **Confirmation Signal**: Slower trend-based confirmation
/// - **Quality Score**: Signal reliability assessment
///
/// # Signal Interpretation
///
/// - Strength > 50 with Quality > 70: Strong buy signal
/// - Strength > 25 with Quality > 50: Moderate buy signal
/// - Strength -25 to 25: Neutral
/// - Strength < -25 with Quality > 50: Moderate sell signal
/// - Strength < -50 with Quality > 70: Strong sell signal
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{SignalStrengthComposite, SignalStrengthCompositeConfig};
///
/// let config = SignalStrengthCompositeConfig::default();
/// let indicator = SignalStrengthComposite::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close, &volume);
/// ```
#[derive(Debug, Clone)]
pub struct SignalStrengthComposite {
    primary_period: usize,
    confirmation_period: usize,
    quality_period: usize,
    smoothing_period: usize,
}

impl SignalStrengthComposite {
    /// Create a new SignalStrengthComposite with the given configuration.
    pub fn new(config: SignalStrengthCompositeConfig) -> Result<Self> {
        if config.primary_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "primary_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.confirmation_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "confirmation_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.quality_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "quality_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            primary_period: config.primary_period,
            confirmation_period: config.confirmation_period,
            quality_period: config.quality_period,
            smoothing_period: config.smoothing_period,
        })
    }

    /// Calculate the Signal Strength Composite values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> SignalStrengthCompositeOutput {
        let n = close.len();
        let mut strength = vec![f64::NAN; n];
        let mut primary_signal = vec![f64::NAN; n];
        let mut confirmation_signal = vec![f64::NAN; n];
        let mut quality_score = vec![f64::NAN; n];

        let max_period = self.primary_period.max(self.confirmation_period).max(self.quality_period);
        if n <= max_period {
            return SignalStrengthCompositeOutput {
                strength,
                primary_signal,
                confirmation_signal,
                quality_score,
            };
        }

        // Calculate primary signal (RSI-based momentum)
        for i in self.primary_period..n {
            let mut gains = 0.0;
            let mut losses = 0.0;
            for j in (i - self.primary_period + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += -change;
                }
            }

            let avg_gain = gains / self.primary_period as f64;
            let avg_loss = losses / self.primary_period as f64;

            let rsi = if avg_loss > 1e-10 {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            } else if avg_gain > 0.0 {
                100.0
            } else {
                50.0
            };

            // Convert RSI to centered signal
            primary_signal[i] = (rsi - 50.0) * 2.0;
        }

        // Calculate confirmation signal (trend-based)
        let ema = self.calculate_ema(close, self.confirmation_period);
        for i in self.confirmation_period..n {
            if !ema[i].is_nan() && ema[i].abs() > 1e-10 {
                // Price deviation from EMA
                let deviation = (close[i] - ema[i]) / ema[i] * 100.0;

                // EMA slope
                let slope = if i > 0 && !ema[i - 1].is_nan() && ema[i - 1].abs() > 1e-10 {
                    (ema[i] - ema[i - 1]) / ema[i - 1] * 100.0
                } else {
                    0.0
                };

                confirmation_signal[i] = (deviation * 10.0 + slope * 50.0).clamp(-100.0, 100.0);
            }
        }

        // Calculate quality score
        for i in self.quality_period..n {
            // Signal consistency
            let mut signal_direction_consistency = 0;
            for j in (i - self.quality_period + 1)..=i {
                if j > 0 {
                    let bar_direction = (close[j] - close[j - 1]).signum();
                    let overall_direction = (close[i] - close[i - self.quality_period]).signum();
                    if bar_direction == overall_direction {
                        signal_direction_consistency += 1;
                    }
                }
            }

            let consistency = signal_direction_consistency as f64 / self.quality_period as f64;

            // Volume confirmation
            let avg_volume: f64 = volume[(i - self.quality_period + 1)..=i].iter().sum::<f64>()
                / self.quality_period as f64;
            let vol_quality = if avg_volume > 1e-10 {
                let vol_ratio = volume[i] / avg_volume;
                // Higher volume = better quality signal
                (vol_ratio * 50.0).clamp(0.0, 100.0)
            } else {
                50.0
            };

            quality_score[i] = (consistency * 50.0 + vol_quality * 0.5).clamp(0.0, 100.0);
        }

        // Combine signals with quality weighting
        for i in max_period..n {
            if !primary_signal[i].is_nan()
                && !confirmation_signal[i].is_nan()
                && !quality_score[i].is_nan()
            {
                // Agreement bonus
                let agreement = if primary_signal[i].signum() == confirmation_signal[i].signum() {
                    1.2
                } else {
                    0.8
                };

                // Quality-weighted combination
                let quality_factor = quality_score[i] / 100.0;
                let raw_strength = (primary_signal[i] * 0.6 + confirmation_signal[i] * 0.4)
                    * agreement
                    * (0.5 + quality_factor * 0.5);

                strength[i] = raw_strength.clamp(-100.0, 100.0);
            }
        }

        // Apply smoothing
        if self.smoothing_period > 1 {
            let smoothed = self.ema_smooth(&strength, self.smoothing_period);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    strength[i] = smoothed[i];
                }
            }
        }

        SignalStrengthCompositeOutput {
            strength,
            primary_signal,
            confirmation_signal,
            quality_score,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for SignalStrengthComposite {
    fn name(&self) -> &str {
        "SignalStrengthComposite"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            result.strength,
            result.primary_signal,
            result.confirmation_signal,
        ))
    }

    fn min_periods(&self) -> usize {
        self.primary_period.max(self.confirmation_period).max(self.quality_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 24. AdaptiveCompositeMA
// ============================================================================

/// Adaptive Composite MA output.
#[derive(Debug, Clone)]
pub struct AdaptiveCompositeMAOutput {
    /// Adaptive composite moving average values.
    pub ma: Vec<f64>,
    /// Adaptation speed (0-1).
    pub adaptation_speed: Vec<f64>,
    /// Trend direction (-1, 0, 1).
    pub trend_direction: Vec<f64>,
}

/// Adaptive Composite MA configuration.
#[derive(Debug, Clone)]
pub struct AdaptiveCompositeMAConfig {
    /// Fast period for responsive component (default: 5).
    pub fast_period: usize,
    /// Slow period for stable component (default: 30).
    pub slow_period: usize,
    /// Efficiency ratio period (default: 10).
    pub efficiency_period: usize,
    /// Volatility period for adaptation (default: 14).
    pub volatility_period: usize,
}

impl Default for AdaptiveCompositeMAConfig {
    fn default() -> Self {
        Self {
            fast_period: 5,
            slow_period: 30,
            efficiency_period: 10,
            volatility_period: 14,
        }
    }
}

/// Adaptive Composite MA.
///
/// A sophisticated moving average that combines multiple MA types and adapts
/// its behavior based on market conditions using efficiency ratio and volatility.
///
/// # Adaptation Logic
///
/// - **Trending Markets** (high efficiency): Uses faster, more responsive MA
/// - **Ranging Markets** (low efficiency): Uses slower, smoother MA
/// - **High Volatility**: Reduces sensitivity to avoid whipsaws
/// - **Low Volatility**: Increases sensitivity to catch breakouts
///
/// # Components
///
/// - Fast EMA: Quick response to price changes
/// - Slow SMA: Stable, noise-filtered signal
/// - Efficiency Ratio: Market efficiency measurement
/// - Volatility Adjustment: Dynamic sensitivity control
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{AdaptiveCompositeMA, AdaptiveCompositeMAConfig};
///
/// let config = AdaptiveCompositeMAConfig::default();
/// let indicator = AdaptiveCompositeMA::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveCompositeMA {
    fast_period: usize,
    slow_period: usize,
    efficiency_period: usize,
    volatility_period: usize,
}

impl AdaptiveCompositeMA {
    /// Create a new AdaptiveCompositeMA with the given configuration.
    pub fn new(config: AdaptiveCompositeMAConfig) -> Result<Self> {
        if config.fast_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.slow_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.efficiency_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "efficiency_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.fast_period >= config.slow_period {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be less than slow_period".to_string(),
            });
        }

        Ok(Self {
            fast_period: config.fast_period,
            slow_period: config.slow_period,
            efficiency_period: config.efficiency_period,
            volatility_period: config.volatility_period,
        })
    }

    /// Calculate the Adaptive Composite MA values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> AdaptiveCompositeMAOutput {
        let n = close.len();
        let mut ma = vec![f64::NAN; n];
        let mut adaptation_speed = vec![f64::NAN; n];
        let mut trend_direction = vec![f64::NAN; n];

        let max_period = self.slow_period.max(self.efficiency_period).max(self.volatility_period);
        if n <= max_period {
            return AdaptiveCompositeMAOutput {
                ma,
                adaptation_speed,
                trend_direction,
            };
        }

        // Calculate efficiency ratio
        let mut efficiency = vec![f64::NAN; n];
        for i in self.efficiency_period..n {
            let net_change = (close[i] - close[i - self.efficiency_period]).abs();
            let mut sum_changes = 0.0;
            for j in (i - self.efficiency_period + 1)..=i {
                sum_changes += (close[j] - close[j - 1]).abs();
            }

            if sum_changes > 1e-10 {
                efficiency[i] = (net_change / sum_changes).clamp(0.0, 1.0);
            } else {
                efficiency[i] = 0.0;
            }
        }

        // Calculate volatility adjustment
        let mut vol_adjustment = vec![1.0; n];
        let mut atr = vec![f64::NAN; n];
        for i in 1..n {
            if i >= self.volatility_period {
                let mut sum = 0.0;
                for j in (i - self.volatility_period + 1)..=i {
                    let tr = if j == 0 {
                        high[j] - low[j]
                    } else {
                        (high[j] - low[j])
                            .max((high[j] - close[j - 1]).abs())
                            .max((low[j] - close[j - 1]).abs())
                    };
                    sum += tr;
                }
                atr[i] = sum / self.volatility_period as f64;

                if close[i].abs() > 1e-10 {
                    let atr_percent = atr[i] / close[i];
                    // Higher volatility = slower adaptation (lower adjustment)
                    vol_adjustment[i] = (1.0 - atr_percent * 10.0).clamp(0.3, 1.0);
                }
            }
        }

        // Calculate fast and slow MAs
        let fast_ema = self.calculate_ema(close, self.fast_period);
        let slow_sma = self.calculate_sma(close, self.slow_period);

        // Calculate adaptive MA
        let fast_sc = 2.0 / (self.fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (self.slow_period as f64 + 1.0);

        // Initialize with SMA
        if n > max_period {
            ma[max_period] = close[max_period];
        }

        for i in (max_period + 1)..n {
            if !efficiency[i].is_nan() && !ma[i - 1].is_nan() {
                let er = efficiency[i];
                let vol_adj = vol_adjustment[i];

                // Adaptive smoothing constant
                // High efficiency = use fast SC
                // Low efficiency = use slow SC
                // Volatility adjustment reduces responsiveness in volatile conditions
                let adaptive_sc = ((er * (fast_sc - slow_sc) + slow_sc) * vol_adj).powi(2);

                // Calculate adaptive MA
                ma[i] = ma[i - 1] + adaptive_sc * (close[i] - ma[i - 1]);

                // Adaptation speed
                adaptation_speed[i] = er * vol_adj;

                // Trend direction based on MA slope
                if !ma[i - 1].is_nan() && ma[i - 1].abs() > 1e-10 {
                    let slope = (ma[i] - ma[i - 1]) / ma[i - 1];
                    trend_direction[i] = if slope > 0.0001 {
                        1.0
                    } else if slope < -0.0001 {
                        -1.0
                    } else {
                        0.0
                    };
                }
            }
        }

        AdaptiveCompositeMAOutput {
            ma,
            adaptation_speed,
            trend_direction,
        }
    }

    /// Calculate EMA for given period.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }

    /// Calculate SMA for given period.
    fn calculate_sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut sma = vec![f64::NAN; n];

        if period > 0 && n >= period {
            for i in (period - 1)..n {
                let start = i.saturating_sub(period - 1);
                let sum: f64 = data[start..=i].iter().sum();
                sma[i] = sum / period as f64;
            }
        }

        sma
    }
}

impl TechnicalIndicator for AdaptiveCompositeMA {
    fn name(&self) -> &str {
        "AdaptiveCompositeMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.ma,
            result.adaptation_speed,
            result.trend_direction,
        ))
    }

    fn min_periods(&self) -> usize {
        self.slow_period.max(self.efficiency_period).max(self.volatility_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 25. TrendStrengthComposite
// ============================================================================

/// Trend Strength Composite output.
#[derive(Debug, Clone)]
pub struct TrendStrengthCompositeOutput {
    /// Combined trend strength score (0-100).
    pub strength: Vec<f64>,
    /// ADX-based trend component (0-100).
    pub adx_component: Vec<f64>,
    /// Price position component (0-100).
    pub position_component: Vec<f64>,
    /// Momentum direction component (-100 to 100).
    pub direction_component: Vec<f64>,
}

/// Trend Strength Composite configuration.
#[derive(Debug, Clone)]
pub struct TrendStrengthCompositeConfig {
    /// Period for ADX calculation (default: 14).
    pub adx_period: usize,
    /// Period for price position calculation (default: 20).
    pub position_period: usize,
    /// Period for momentum direction (default: 10).
    pub momentum_period: usize,
    /// Weight for ADX component (default: 0.4).
    pub adx_weight: f64,
    /// Weight for position component (default: 0.3).
    pub position_weight: f64,
    /// Weight for direction component (default: 0.3).
    pub direction_weight: f64,
}

impl Default for TrendStrengthCompositeConfig {
    fn default() -> Self {
        Self {
            adx_period: 14,
            position_period: 20,
            momentum_period: 10,
            adx_weight: 0.4,
            position_weight: 0.3,
            direction_weight: 0.3,
        }
    }
}

/// Trend Strength Composite.
///
/// Combines multiple trend measurement techniques to provide a comprehensive
/// assessment of trend strength. Unlike single-indicator approaches, this
/// composite considers directional movement, price position within its range,
/// and momentum alignment.
///
/// # Components
///
/// - **ADX Component**: Measures trend strength using directional movement
/// - **Position Component**: Where price is relative to recent high/low range
/// - **Direction Component**: Momentum direction and consistency
///
/// # Interpretation
///
/// - 80-100: Very strong trend
/// - 60-80: Strong trend
/// - 40-60: Moderate trend
/// - 20-40: Weak trend
/// - 0-20: No clear trend
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{TrendStrengthComposite, TrendStrengthCompositeConfig};
///
/// let config = TrendStrengthCompositeConfig::default();
/// let indicator = TrendStrengthComposite::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close);
/// ```
#[derive(Debug, Clone)]
pub struct TrendStrengthComposite {
    adx_period: usize,
    position_period: usize,
    momentum_period: usize,
    adx_weight: f64,
    position_weight: f64,
    direction_weight: f64,
}

impl TrendStrengthComposite {
    /// Create a new TrendStrengthComposite with the given configuration.
    pub fn new(config: TrendStrengthCompositeConfig) -> Result<Self> {
        if config.adx_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "adx_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.position_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "position_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.momentum_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        let total_weight = config.adx_weight + config.position_weight + config.direction_weight;
        if (total_weight - 1.0).abs() > 0.01 {
            return Err(IndicatorError::InvalidParameter {
                name: "weights".to_string(),
                reason: "must sum to 1.0".to_string(),
            });
        }

        Ok(Self {
            adx_period: config.adx_period,
            position_period: config.position_period,
            momentum_period: config.momentum_period,
            adx_weight: config.adx_weight,
            position_weight: config.position_weight,
            direction_weight: config.direction_weight,
        })
    }

    /// Calculate the Trend Strength Composite values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> TrendStrengthCompositeOutput {
        let n = close.len();
        let mut strength = vec![f64::NAN; n];
        let mut adx_component = vec![f64::NAN; n];
        let mut position_component = vec![f64::NAN; n];
        let mut direction_component = vec![f64::NAN; n];

        let max_period = self.adx_period.max(self.position_period).max(self.momentum_period);
        if n <= max_period + self.adx_period {
            return TrendStrengthCompositeOutput {
                strength,
                adx_component,
                position_component,
                direction_component,
            };
        }

        // Calculate True Range and Directional Movement
        let mut tr = vec![0.0; n];
        let mut plus_dm = vec![0.0; n];
        let mut minus_dm = vec![0.0; n];

        tr[0] = high[0] - low[0];
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);

            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];

            if up_move > down_move && up_move > 0.0 {
                plus_dm[i] = up_move;
            }
            if down_move > up_move && down_move > 0.0 {
                minus_dm[i] = down_move;
            }
        }

        // Calculate smoothed values for ADX
        let mut smoothed_tr = vec![f64::NAN; n];
        let mut smoothed_plus_dm = vec![f64::NAN; n];
        let mut smoothed_minus_dm = vec![f64::NAN; n];
        let mut adx = vec![f64::NAN; n];

        if n > self.adx_period {
            let sum_tr: f64 = tr[1..=self.adx_period].iter().sum();
            let sum_plus: f64 = plus_dm[1..=self.adx_period].iter().sum();
            let sum_minus: f64 = minus_dm[1..=self.adx_period].iter().sum();

            smoothed_tr[self.adx_period] = sum_tr;
            smoothed_plus_dm[self.adx_period] = sum_plus;
            smoothed_minus_dm[self.adx_period] = sum_minus;

            for i in (self.adx_period + 1)..n {
                smoothed_tr[i] = smoothed_tr[i - 1] - (smoothed_tr[i - 1] / self.adx_period as f64) + tr[i];
                smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / self.adx_period as f64) + plus_dm[i];
                smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / self.adx_period as f64) + minus_dm[i];
            }

            // Calculate DI+ and DI- and DX
            let mut dx = vec![f64::NAN; n];
            for i in self.adx_period..n {
                if !smoothed_tr[i].is_nan() && smoothed_tr[i].abs() > 1e-10 {
                    let plus_di = (smoothed_plus_dm[i] / smoothed_tr[i]) * 100.0;
                    let minus_di = (smoothed_minus_dm[i] / smoothed_tr[i]) * 100.0;
                    let di_sum = plus_di + minus_di;
                    if di_sum > 0.0 {
                        dx[i] = ((plus_di - minus_di).abs() / di_sum) * 100.0;
                    }
                }
            }

            // Calculate ADX as smoothed DX
            let adx_start = self.adx_period * 2;
            if n > adx_start {
                let mut sum_dx = 0.0;
                let mut count = 0;
                for i in self.adx_period..adx_start {
                    if !dx[i].is_nan() {
                        sum_dx += dx[i];
                        count += 1;
                    }
                }
                if count > 0 {
                    adx[adx_start] = sum_dx / count as f64;
                }

                for i in (adx_start + 1)..n {
                    if !adx[i - 1].is_nan() && !dx[i].is_nan() {
                        adx[i] = (adx[i - 1] * (self.adx_period - 1) as f64 + dx[i]) / self.adx_period as f64;
                    }
                }
            }
        }

        // Calculate position component (where price is in the range)
        for i in self.position_period..n {
            let highest = high[(i - self.position_period + 1)..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = low[(i - self.position_period + 1)..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let range = highest - lowest;

            if range > 1e-10 {
                // Position as percentage: 0 = at low, 100 = at high
                position_component[i] = ((close[i] - lowest) / range * 100.0).clamp(0.0, 100.0);
            } else {
                position_component[i] = 50.0;
            }
        }

        // Calculate direction component (momentum consistency)
        for i in self.momentum_period..n {
            let mut up_count = 0;
            let mut down_count = 0;
            let mut net_change = 0.0;

            for j in (i - self.momentum_period + 1)..=i {
                let change = close[j] - close[j - 1];
                net_change += change;
                if change > 0.0 {
                    up_count += 1;
                } else if change < 0.0 {
                    down_count += 1;
                }
            }

            let total_bars = self.momentum_period;
            let dominant_direction = if up_count > down_count { 1.0 } else { -1.0 };
            let consistency = (up_count.max(down_count) as f64) / (total_bars as f64);

            // Direction component: direction * consistency * 100
            direction_component[i] = (dominant_direction * consistency * 100.0).clamp(-100.0, 100.0);
        }

        // Combine components into strength
        let start_idx = max_period + self.adx_period;
        for i in start_idx..n {
            if !adx[i].is_nan() && !position_component[i].is_nan() && !direction_component[i].is_nan() {
                adx_component[i] = adx[i].clamp(0.0, 100.0);

                // Convert position to trend strength (extremes = strong trend)
                let pos_strength = ((position_component[i] - 50.0).abs() * 2.0).clamp(0.0, 100.0);

                // Convert direction to absolute strength
                let dir_strength = direction_component[i].abs();

                strength[i] = (self.adx_weight * adx_component[i]
                    + self.position_weight * pos_strength
                    + self.direction_weight * dir_strength)
                    .clamp(0.0, 100.0);
            }
        }

        TrendStrengthCompositeOutput {
            strength,
            adx_component,
            position_component,
            direction_component,
        }
    }
}

impl TechnicalIndicator for TrendStrengthComposite {
    fn name(&self) -> &str {
        "TrendStrengthComposite"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.strength,
            result.adx_component,
            result.position_component,
        ))
    }

    fn min_periods(&self) -> usize {
        self.adx_period.max(self.position_period).max(self.momentum_period) + self.adx_period + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 26. MomentumQualityComposite
// ============================================================================

/// Momentum Quality Composite output.
#[derive(Debug, Clone)]
pub struct MomentumQualityCompositeOutput {
    /// Overall momentum quality score (0-100).
    pub quality: Vec<f64>,
    /// RSI-based momentum (0-100).
    pub rsi_momentum: Vec<f64>,
    /// Rate of change momentum (-100 to 100).
    pub roc_momentum: Vec<f64>,
    /// Consistency score (0-100).
    pub consistency: Vec<f64>,
}

/// Momentum Quality Composite configuration.
#[derive(Debug, Clone)]
pub struct MomentumQualityCompositeConfig {
    /// Period for RSI calculation (default: 14).
    pub rsi_period: usize,
    /// Period for ROC calculation (default: 10).
    pub roc_period: usize,
    /// Period for consistency measurement (default: 20).
    pub consistency_period: usize,
    /// Smoothing period (default: 3).
    pub smoothing: usize,
}

impl Default for MomentumQualityCompositeConfig {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            roc_period: 10,
            consistency_period: 20,
            smoothing: 3,
        }
    }
}

/// Momentum Quality Composite.
///
/// Evaluates momentum quality by combining multiple momentum measures with
/// a consistency filter. High-quality momentum occurs when RSI, ROC, and
/// price direction are aligned and consistent.
///
/// # Components
///
/// - **RSI Momentum**: Relative strength of gains vs losses
/// - **ROC Momentum**: Rate of price change
/// - **Consistency**: How consistently price moves in one direction
///
/// # Quality Levels
///
/// - 80-100: Excellent momentum quality
/// - 60-80: Good momentum quality
/// - 40-60: Moderate quality
/// - 20-40: Poor quality
/// - 0-20: Very poor quality
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{MomentumQualityComposite, MomentumQualityCompositeConfig};
///
/// let config = MomentumQualityCompositeConfig::default();
/// let indicator = MomentumQualityComposite::new(config).unwrap();
/// let result = indicator.calculate(&close);
/// ```
#[derive(Debug, Clone)]
pub struct MomentumQualityComposite {
    rsi_period: usize,
    roc_period: usize,
    consistency_period: usize,
    smoothing: usize,
}

impl MomentumQualityComposite {
    /// Create a new MomentumQualityComposite with the given configuration.
    pub fn new(config: MomentumQualityCompositeConfig) -> Result<Self> {
        if config.rsi_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "rsi_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.roc_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "roc_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.consistency_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "consistency_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            rsi_period: config.rsi_period,
            roc_period: config.roc_period,
            consistency_period: config.consistency_period,
            smoothing: config.smoothing,
        })
    }

    /// Calculate the Momentum Quality Composite values.
    pub fn calculate(&self, close: &[f64]) -> MomentumQualityCompositeOutput {
        let n = close.len();
        let mut quality = vec![f64::NAN; n];
        let mut rsi_momentum = vec![f64::NAN; n];
        let mut roc_momentum = vec![f64::NAN; n];
        let mut consistency = vec![f64::NAN; n];

        let max_period = self.rsi_period.max(self.roc_period).max(self.consistency_period);
        if n <= max_period {
            return MomentumQualityCompositeOutput {
                quality,
                rsi_momentum,
                roc_momentum,
                consistency,
            };
        }

        // Calculate RSI
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;

        for i in 1..=self.rsi_period {
            let change = close[i] - close[i - 1];
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss += -change;
            }
        }
        avg_gain /= self.rsi_period as f64;
        avg_loss /= self.rsi_period as f64;

        if avg_loss > 1e-10 {
            rsi_momentum[self.rsi_period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
        } else {
            rsi_momentum[self.rsi_period] = if avg_gain > 0.0 { 100.0 } else { 50.0 };
        }

        for i in (self.rsi_period + 1)..n {
            let change = close[i] - close[i - 1];
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };

            avg_gain = (avg_gain * (self.rsi_period - 1) as f64 + gain) / self.rsi_period as f64;
            avg_loss = (avg_loss * (self.rsi_period - 1) as f64 + loss) / self.rsi_period as f64;

            if avg_loss > 1e-10 {
                rsi_momentum[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
            } else {
                rsi_momentum[i] = if avg_gain > 0.0 { 100.0 } else { 50.0 };
            }
        }

        // Calculate ROC
        for i in self.roc_period..n {
            if close[i - self.roc_period].abs() > 1e-10 {
                roc_momentum[i] = ((close[i] - close[i - self.roc_period]) / close[i - self.roc_period] * 100.0)
                    .clamp(-100.0, 100.0);
            }
        }

        // Calculate consistency
        for i in self.consistency_period..n {
            let mut up_count = 0;
            let mut down_count = 0;

            for j in (i - self.consistency_period + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    up_count += 1;
                } else if change < 0.0 {
                    down_count += 1;
                }
            }

            let dominant = up_count.max(down_count) as f64;
            let total = (up_count + down_count).max(1) as f64;
            consistency[i] = (dominant / total * 100.0).clamp(0.0, 100.0);
        }

        // Calculate quality score
        for i in max_period..n {
            if !rsi_momentum[i].is_nan() && !roc_momentum[i].is_nan() && !consistency[i].is_nan() {
                // RSI quality: how far from 50 (neutral)
                let rsi_strength = (rsi_momentum[i] - 50.0).abs() * 2.0;

                // ROC quality: absolute momentum
                let roc_strength = roc_momentum[i].abs();

                // Agreement between RSI and ROC directions
                let rsi_direction = if rsi_momentum[i] > 50.0 { 1.0 } else { -1.0 };
                let roc_direction = roc_momentum[i].signum();
                let agreement = if rsi_direction == roc_direction { 1.0 } else { 0.5 };

                // Combine components
                quality[i] = ((rsi_strength * 0.35 + roc_strength * 0.35 + consistency[i] * 0.30) * agreement)
                    .clamp(0.0, 100.0);
            }
        }

        // Apply smoothing
        if self.smoothing > 1 {
            let smoothed = self.ema_smooth(&quality, self.smoothing);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    quality[i] = smoothed[i];
                }
            }
        }

        MomentumQualityCompositeOutput {
            quality,
            rsi_momentum,
            roc_momentum,
            consistency,
        }
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumQualityComposite {
    fn name(&self) -> &str {
        "MomentumQualityComposite"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(
            result.quality,
            result.rsi_momentum,
            result.roc_momentum,
        ))
    }

    fn min_periods(&self) -> usize {
        self.rsi_period.max(self.roc_period).max(self.consistency_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 27. VolatilityAdjustedSignal
// ============================================================================

/// Volatility Adjusted Signal output.
#[derive(Debug, Clone)]
pub struct VolatilityAdjustedSignalOutput {
    /// Volatility-adjusted signal (-100 to 100).
    pub signal: Vec<f64>,
    /// Raw unadjusted signal (-100 to 100).
    pub raw_signal: Vec<f64>,
    /// Volatility factor (0-1).
    pub volatility_factor: Vec<f64>,
    /// Signal confidence (0-100).
    pub confidence: Vec<f64>,
}

/// Volatility Adjusted Signal configuration.
#[derive(Debug, Clone)]
pub struct VolatilityAdjustedSignalConfig {
    /// Period for signal calculation (default: 14).
    pub signal_period: usize,
    /// Period for volatility calculation (default: 20).
    pub volatility_period: usize,
    /// Volatility lookback for percentile (default: 50).
    pub volatility_lookback: usize,
    /// Signal smoothing period (default: 3).
    pub smoothing: usize,
}

impl Default for VolatilityAdjustedSignalConfig {
    fn default() -> Self {
        Self {
            signal_period: 14,
            volatility_period: 20,
            volatility_lookback: 50,
            smoothing: 3,
        }
    }
}

/// Volatility Adjusted Signal.
///
/// Generates trading signals that are dynamically adjusted based on current
/// volatility conditions. In high volatility, signals are dampened to reduce
/// false positives. In low volatility, signals are amplified to catch early moves.
///
/// # Volatility Adjustment Logic
///
/// - **High Volatility** (above 75th percentile): Signal dampened by 50%
/// - **Normal Volatility** (25th-75th percentile): Signal unchanged
/// - **Low Volatility** (below 25th percentile): Signal amplified by 25%
///
/// # Components
///
/// - **Raw Signal**: Base momentum signal before adjustment
/// - **Volatility Factor**: Current volatility relative to historical
/// - **Confidence**: Signal reliability based on volatility regime
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{VolatilityAdjustedSignal, VolatilityAdjustedSignalConfig};
///
/// let config = VolatilityAdjustedSignalConfig::default();
/// let indicator = VolatilityAdjustedSignal::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close);
/// ```
#[derive(Debug, Clone)]
pub struct VolatilityAdjustedSignal {
    signal_period: usize,
    volatility_period: usize,
    volatility_lookback: usize,
    smoothing: usize,
}

impl VolatilityAdjustedSignal {
    /// Create a new VolatilityAdjustedSignal with the given configuration.
    pub fn new(config: VolatilityAdjustedSignalConfig) -> Result<Self> {
        if config.signal_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "signal_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_lookback == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_lookback".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.smoothing == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            signal_period: config.signal_period,
            volatility_period: config.volatility_period,
            volatility_lookback: config.volatility_lookback,
            smoothing: config.smoothing,
        })
    }

    /// Calculate the Volatility Adjusted Signal values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> VolatilityAdjustedSignalOutput {
        let n = close.len();
        let mut signal = vec![f64::NAN; n];
        let mut raw_signal = vec![f64::NAN; n];
        let mut volatility_factor = vec![f64::NAN; n];
        let mut confidence = vec![f64::NAN; n];

        let max_period = self.signal_period.max(self.volatility_period).max(self.volatility_lookback);
        if n <= max_period {
            return VolatilityAdjustedSignalOutput {
                signal,
                raw_signal,
                volatility_factor,
                confidence,
            };
        }

        // Calculate ATR for volatility
        let mut atr = vec![f64::NAN; n];
        for i in 1..n {
            if i >= self.volatility_period {
                let mut sum = 0.0;
                for j in (i - self.volatility_period + 1)..=i {
                    let tr = if j == 0 {
                        high[j] - low[j]
                    } else {
                        (high[j] - low[j])
                            .max((high[j] - close[j - 1]).abs())
                            .max((low[j] - close[j - 1]).abs())
                    };
                    sum += tr;
                }
                atr[i] = sum / self.volatility_period as f64;
            }
        }

        // Calculate raw signal (momentum-based)
        for i in self.signal_period..n {
            let mut gains = 0.0;
            let mut losses = 0.0;

            for j in (i - self.signal_period + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += -change;
                }
            }

            let avg_gain = gains / self.signal_period as f64;
            let avg_loss = losses / self.signal_period as f64;

            // RSI-based signal converted to -100 to 100
            let rsi = if avg_loss > 1e-10 {
                100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            } else if avg_gain > 0.0 {
                100.0
            } else {
                50.0
            };

            raw_signal[i] = (rsi - 50.0) * 2.0;
        }

        // Calculate volatility percentile and factor
        for i in max_period..n {
            if !atr[i].is_nan() {
                // Get volatility history
                let start = i.saturating_sub(self.volatility_lookback);
                let mut vol_history: Vec<f64> = atr[start..=i]
                    .iter()
                    .filter(|x| !x.is_nan())
                    .cloned()
                    .collect();

                if !vol_history.is_empty() {
                    vol_history.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let current_vol = atr[i];
                    let rank = vol_history.iter().position(|&x| x >= current_vol).unwrap_or(vol_history.len());
                    let percentile = (rank as f64) / (vol_history.len() as f64);

                    volatility_factor[i] = percentile;

                    // Calculate adjustment multiplier
                    let adjustment = if percentile > 0.75 {
                        // High volatility: dampen signal
                        0.5 + (1.0 - percentile) * 2.0
                    } else if percentile < 0.25 {
                        // Low volatility: amplify signal
                        1.0 + (0.25 - percentile)
                    } else {
                        // Normal volatility
                        1.0
                    };

                    if !raw_signal[i].is_nan() {
                        signal[i] = (raw_signal[i] * adjustment).clamp(-100.0, 100.0);
                    }

                    // Confidence based on volatility stability
                    let mean_vol: f64 = vol_history.iter().sum::<f64>() / vol_history.len() as f64;
                    let vol_variance: f64 = vol_history.iter().map(|x| (x - mean_vol).powi(2)).sum::<f64>() / vol_history.len() as f64;
                    let vol_cv = if mean_vol > 1e-10 { vol_variance.sqrt() / mean_vol } else { 0.0 };

                    // Lower CV = more stable volatility = higher confidence
                    confidence[i] = ((1.0 - vol_cv.min(1.0)) * 100.0).clamp(0.0, 100.0);
                }
            }
        }

        // Apply smoothing to signal
        if self.smoothing > 1 {
            let smoothed = self.ema_smooth(&signal, self.smoothing);
            for i in 0..n {
                if !smoothed[i].is_nan() {
                    signal[i] = smoothed[i];
                }
            }
        }

        VolatilityAdjustedSignalOutput {
            signal,
            raw_signal,
            volatility_factor,
            confidence,
        }
    }

    /// Simple EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        let mut first_valid = None;
        for i in 0..n {
            if !data[i].is_nan() {
                first_valid = Some(i);
                result[i] = data[i];
                break;
            }
        }

        if let Some(start) = first_valid {
            for i in (start + 1)..n {
                if !data[i].is_nan() {
                    if !result[i - 1].is_nan() {
                        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
                    } else {
                        result[i] = data[i];
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilityAdjustedSignal {
    fn name(&self) -> &str {
        "VolatilityAdjustedSignal"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.signal,
            result.raw_signal,
            result.volatility_factor,
        ))
    }

    fn min_periods(&self) -> usize {
        self.signal_period.max(self.volatility_period).max(self.volatility_lookback) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 28. MultiFactorMomentumV2
// ============================================================================

/// Multi-Factor Momentum V2 output.
#[derive(Debug, Clone)]
pub struct MultiFactorMomentumV2Output {
    /// Combined multi-factor momentum score (-100 to 100).
    pub momentum: Vec<f64>,
    /// Price momentum factor.
    pub price_factor: Vec<f64>,
    /// Volume momentum factor.
    pub volume_factor: Vec<f64>,
    /// Volatility momentum factor.
    pub volatility_factor: Vec<f64>,
}

/// Multi-Factor Momentum V2 configuration.
#[derive(Debug, Clone)]
pub struct MultiFactorMomentumV2Config {
    /// Short-term momentum period (default: 5).
    pub short_period: usize,
    /// Medium-term momentum period (default: 10).
    pub medium_period: usize,
    /// Long-term momentum period (default: 20).
    pub long_period: usize,
    /// Volume period (default: 10).
    pub volume_period: usize,
}

impl Default for MultiFactorMomentumV2Config {
    fn default() -> Self {
        Self {
            short_period: 5,
            medium_period: 10,
            long_period: 20,
            volume_period: 10,
        }
    }
}

/// Multi-Factor Momentum V2.
///
/// An enhanced multi-factor momentum indicator that combines price momentum
/// across multiple timeframes with volume and volatility factors for a more
/// complete momentum picture.
///
/// # Factors
///
/// - **Price Factor**: Weighted combination of short, medium, and long-term ROC
/// - **Volume Factor**: Volume trend relative to average (confirms moves)
/// - **Volatility Factor**: Volatility expansion/contraction momentum
///
/// # Signal Interpretation
///
/// - Strong positive (>50): Strong upward momentum across factors
/// - Moderate positive (25-50): Moderate bullish momentum
/// - Neutral (-25 to 25): No clear momentum direction
/// - Moderate negative (-50 to -25): Moderate bearish momentum
/// - Strong negative (<-50): Strong downward momentum across factors
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{MultiFactorMomentumV2, MultiFactorMomentumV2Config};
///
/// let config = MultiFactorMomentumV2Config::default();
/// let indicator = MultiFactorMomentumV2::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close, &volume);
/// ```
#[derive(Debug, Clone)]
pub struct MultiFactorMomentumV2 {
    short_period: usize,
    medium_period: usize,
    long_period: usize,
    volume_period: usize,
}

impl MultiFactorMomentumV2 {
    /// Create a new MultiFactorMomentumV2 with the given configuration.
    pub fn new(config: MultiFactorMomentumV2Config) -> Result<Self> {
        if config.short_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.medium_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.long_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volume_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.short_period >= config.medium_period || config.medium_period >= config.long_period {
            return Err(IndicatorError::InvalidParameter {
                name: "periods".to_string(),
                reason: "short_period < medium_period < long_period required".to_string(),
            });
        }

        Ok(Self {
            short_period: config.short_period,
            medium_period: config.medium_period,
            long_period: config.long_period,
            volume_period: config.volume_period,
        })
    }

    /// Calculate the Multi-Factor Momentum V2 values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> MultiFactorMomentumV2Output {
        let n = close.len();
        let mut momentum = vec![f64::NAN; n];
        let mut price_factor = vec![f64::NAN; n];
        let mut volume_factor = vec![f64::NAN; n];
        let mut volatility_factor = vec![f64::NAN; n];

        let max_period = self.long_period.max(self.volume_period);
        if n <= max_period {
            return MultiFactorMomentumV2Output {
                momentum,
                price_factor,
                volume_factor,
                volatility_factor,
            };
        }

        // Calculate price factor (multi-timeframe ROC)
        for i in self.long_period..n {
            let short_roc = if close[i - self.short_period].abs() > 1e-10 {
                (close[i] - close[i - self.short_period]) / close[i - self.short_period] * 100.0
            } else {
                0.0
            };

            let medium_roc = if close[i - self.medium_period].abs() > 1e-10 {
                (close[i] - close[i - self.medium_period]) / close[i - self.medium_period] * 100.0
            } else {
                0.0
            };

            let long_roc = if close[i - self.long_period].abs() > 1e-10 {
                (close[i] - close[i - self.long_period]) / close[i - self.long_period] * 100.0
            } else {
                0.0
            };

            // Weight: 50% short, 30% medium, 20% long
            price_factor[i] = (short_roc * 0.5 + medium_roc * 0.3 + long_roc * 0.2).clamp(-100.0, 100.0);
        }

        // Calculate volume factor
        for i in self.volume_period..n {
            let avg_volume: f64 = volume[(i - self.volume_period + 1)..=i].iter().sum::<f64>()
                / self.volume_period as f64;

            if avg_volume > 1e-10 {
                // Compare current volume to average
                let vol_ratio = volume[i] / avg_volume;

                // Volume trend: are recent volumes increasing or decreasing?
                let recent_vol: f64 = volume[(i - self.volume_period / 2)..=i].iter().sum::<f64>()
                    / (self.volume_period / 2 + 1) as f64;
                let older_vol: f64 = volume[(i - self.volume_period + 1)..(i - self.volume_period / 2)]
                    .iter()
                    .sum::<f64>()
                    / (self.volume_period / 2) as f64;

                let vol_trend = if older_vol > 1e-10 {
                    (recent_vol - older_vol) / older_vol
                } else {
                    0.0
                };

                // Volume factor: positive if volume confirms price direction
                volume_factor[i] = ((vol_ratio - 1.0) * 50.0 + vol_trend * 50.0).clamp(-100.0, 100.0);
            } else {
                volume_factor[i] = 0.0;
            }
        }

        // Calculate volatility factor
        for i in max_period..n {
            // Calculate current and historical ATR
            let mut current_atr = 0.0;
            for j in (i - self.short_period + 1)..=i {
                let tr = if j == 0 {
                    high[j] - low[j]
                } else {
                    (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs())
                };
                current_atr += tr;
            }
            current_atr /= self.short_period as f64;

            let mut historical_atr = 0.0;
            for j in (i - self.long_period + 1)..=i {
                let tr = if j == 0 {
                    high[j] - low[j]
                } else {
                    (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs())
                };
                historical_atr += tr;
            }
            historical_atr /= self.long_period as f64;

            // Volatility expansion = positive momentum (breakout potential)
            // Volatility contraction = negative momentum (consolidation)
            if historical_atr > 1e-10 {
                let vol_change = (current_atr - historical_atr) / historical_atr;
                volatility_factor[i] = (vol_change * 100.0).clamp(-100.0, 100.0);
            } else {
                volatility_factor[i] = 0.0;
            }
        }

        // Combine factors into momentum
        for i in max_period..n {
            if !price_factor[i].is_nan() && !volume_factor[i].is_nan() && !volatility_factor[i].is_nan() {
                // Price factor weighted highest, then volume confirmation
                let vol_confirm = if (price_factor[i].signum() == volume_factor[i].signum()) || volume_factor[i].abs() < 10.0 {
                    1.0
                } else {
                    0.7
                };

                momentum[i] = (price_factor[i] * 0.6 + volume_factor[i] * 0.25 + volatility_factor[i] * 0.15)
                    * vol_confirm;
                momentum[i] = momentum[i].clamp(-100.0, 100.0);
            }
        }

        MultiFactorMomentumV2Output {
            momentum,
            price_factor,
            volume_factor,
            volatility_factor,
        }
    }
}

impl TechnicalIndicator for MultiFactorMomentumV2 {
    fn name(&self) -> &str {
        "MultiFactorMomentumV2"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            result.momentum,
            result.price_factor,
            result.volume_factor,
        ))
    }

    fn min_periods(&self) -> usize {
        self.long_period.max(self.volume_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 29. TechnicalRating
// ============================================================================

/// Technical Rating output.
#[derive(Debug, Clone)]
pub struct TechnicalRatingOutput {
    /// Overall technical rating (-100 to 100).
    pub rating: Vec<f64>,
    /// Moving average rating (-100 to 100).
    pub ma_rating: Vec<f64>,
    /// Oscillator rating (-100 to 100).
    pub oscillator_rating: Vec<f64>,
    /// Summary rating (Strong Sell=-2, Sell=-1, Neutral=0, Buy=1, Strong Buy=2).
    pub summary: Vec<f64>,
}

/// Technical Rating configuration.
#[derive(Debug, Clone)]
pub struct TechnicalRatingConfig {
    /// Short EMA period (default: 10).
    pub ema_short: usize,
    /// Medium EMA period (default: 20).
    pub ema_medium: usize,
    /// Long EMA period (default: 50).
    pub ema_long: usize,
    /// RSI period (default: 14).
    pub rsi_period: usize,
    /// Stochastic period (default: 14).
    pub stoch_period: usize,
}

impl Default for TechnicalRatingConfig {
    fn default() -> Self {
        Self {
            ema_short: 10,
            ema_medium: 20,
            ema_long: 50,
            rsi_period: 14,
            stoch_period: 14,
        }
    }
}

/// Technical Rating.
///
/// Provides an overall technical rating similar to TradingView's Technical
/// Analysis summary. Combines moving average analysis and oscillator signals
/// into a unified rating.
///
/// # Rating Components
///
/// ## Moving Average Rating
/// - Price above EMA = Buy signal (+1)
/// - Price below EMA = Sell signal (-1)
/// - Considers short, medium, and long-term EMAs
///
/// ## Oscillator Rating
/// - RSI: >70 Sell, <30 Buy, else Neutral
/// - Stochastic: >80 Sell, <20 Buy, else Neutral
///
/// # Summary Interpretation
///
/// - +2: Strong Buy (rating > 50)
/// - +1: Buy (rating 25 to 50)
/// - 0: Neutral (rating -25 to 25)
/// - -1: Sell (rating -50 to -25)
/// - -2: Strong Sell (rating < -50)
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{TechnicalRating, TechnicalRatingConfig};
///
/// let config = TechnicalRatingConfig::default();
/// let indicator = TechnicalRating::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close);
/// ```
#[derive(Debug, Clone)]
pub struct TechnicalRating {
    ema_short: usize,
    ema_medium: usize,
    ema_long: usize,
    rsi_period: usize,
    stoch_period: usize,
}

impl TechnicalRating {
    /// Create a new TechnicalRating with the given configuration.
    pub fn new(config: TechnicalRatingConfig) -> Result<Self> {
        if config.ema_short == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "ema_short".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.ema_medium == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "ema_medium".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.ema_long == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "ema_long".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.rsi_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "rsi_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.stoch_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "stoch_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            ema_short: config.ema_short,
            ema_medium: config.ema_medium,
            ema_long: config.ema_long,
            rsi_period: config.rsi_period,
            stoch_period: config.stoch_period,
        })
    }

    /// Calculate the Technical Rating values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> TechnicalRatingOutput {
        let n = close.len();
        let mut rating = vec![f64::NAN; n];
        let mut ma_rating = vec![f64::NAN; n];
        let mut oscillator_rating = vec![f64::NAN; n];
        let mut summary = vec![f64::NAN; n];

        let max_period = self.ema_long.max(self.rsi_period).max(self.stoch_period);
        if n <= max_period {
            return TechnicalRatingOutput {
                rating,
                ma_rating,
                oscillator_rating,
                summary,
            };
        }

        // Calculate EMAs
        let ema_short = self.calculate_ema(close, self.ema_short);
        let ema_medium = self.calculate_ema(close, self.ema_medium);
        let ema_long = self.calculate_ema(close, self.ema_long);

        // Calculate RSI
        let mut rsi = vec![f64::NAN; n];
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;

        for i in 1..=self.rsi_period {
            let change = close[i] - close[i - 1];
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss += -change;
            }
        }
        avg_gain /= self.rsi_period as f64;
        avg_loss /= self.rsi_period as f64;

        if avg_loss > 1e-10 {
            rsi[self.rsi_period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
        } else {
            rsi[self.rsi_period] = if avg_gain > 0.0 { 100.0 } else { 50.0 };
        }

        for i in (self.rsi_period + 1)..n {
            let change = close[i] - close[i - 1];
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };

            avg_gain = (avg_gain * (self.rsi_period - 1) as f64 + gain) / self.rsi_period as f64;
            avg_loss = (avg_loss * (self.rsi_period - 1) as f64 + loss) / self.rsi_period as f64;

            if avg_loss > 1e-10 {
                rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
            } else {
                rsi[i] = if avg_gain > 0.0 { 100.0 } else { 50.0 };
            }
        }

        // Calculate Stochastic %K
        let mut stoch_k = vec![f64::NAN; n];
        for i in self.stoch_period..n {
            let highest = high[(i - self.stoch_period + 1)..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = low[(i - self.stoch_period + 1)..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let range = highest - lowest;

            if range > 1e-10 {
                stoch_k[i] = ((close[i] - lowest) / range * 100.0).clamp(0.0, 100.0);
            } else {
                stoch_k[i] = 50.0;
            }
        }

        // Calculate ratings
        for i in max_period..n {
            // MA Rating: count signals from each EMA
            let mut ma_signals = 0.0;
            let mut ma_count = 0;

            if !ema_short[i].is_nan() {
                ma_count += 1;
                ma_signals += if close[i] > ema_short[i] { 1.0 } else { -1.0 };
            }
            if !ema_medium[i].is_nan() {
                ma_count += 1;
                ma_signals += if close[i] > ema_medium[i] { 1.0 } else { -1.0 };
            }
            if !ema_long[i].is_nan() {
                ma_count += 1;
                ma_signals += if close[i] > ema_long[i] { 1.0 } else { -1.0 };
            }

            // EMA crossover signals
            if !ema_short[i].is_nan() && !ema_medium[i].is_nan() {
                ma_count += 1;
                ma_signals += if ema_short[i] > ema_medium[i] { 1.0 } else { -1.0 };
            }
            if !ema_medium[i].is_nan() && !ema_long[i].is_nan() {
                ma_count += 1;
                ma_signals += if ema_medium[i] > ema_long[i] { 1.0 } else { -1.0 };
            }

            if ma_count > 0 {
                ma_rating[i] = (ma_signals / ma_count as f64 * 100.0).clamp(-100.0, 100.0);
            }

            // Oscillator Rating
            let mut osc_signals = 0.0;
            let mut osc_count = 0;

            if !rsi[i].is_nan() {
                osc_count += 1;
                if rsi[i] > 70.0 {
                    osc_signals += -1.0; // Overbought = Sell
                } else if rsi[i] < 30.0 {
                    osc_signals += 1.0; // Oversold = Buy
                } else {
                    osc_signals += (rsi[i] - 50.0) / 50.0; // Neutral range
                }
            }

            if !stoch_k[i].is_nan() {
                osc_count += 1;
                if stoch_k[i] > 80.0 {
                    osc_signals += -1.0; // Overbought = Sell
                } else if stoch_k[i] < 20.0 {
                    osc_signals += 1.0; // Oversold = Buy
                } else {
                    osc_signals += (stoch_k[i] - 50.0) / 50.0; // Neutral range
                }
            }

            if osc_count > 0 {
                oscillator_rating[i] = (osc_signals / osc_count as f64 * 100.0).clamp(-100.0, 100.0);
            }

            // Combined rating (60% MA, 40% Oscillator)
            if !ma_rating[i].is_nan() && !oscillator_rating[i].is_nan() {
                rating[i] = (ma_rating[i] * 0.6 + oscillator_rating[i] * 0.4).clamp(-100.0, 100.0);

                // Summary rating
                summary[i] = if rating[i] > 50.0 {
                    2.0 // Strong Buy
                } else if rating[i] > 25.0 {
                    1.0 // Buy
                } else if rating[i] > -25.0 {
                    0.0 // Neutral
                } else if rating[i] > -50.0 {
                    -1.0 // Sell
                } else {
                    -2.0 // Strong Sell
                };
            }
        }

        TechnicalRatingOutput {
            rating,
            ma_rating,
            oscillator_rating,
            summary,
        }
    }

    /// Calculate EMA.
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];
        let multiplier = 2.0 / (period + 1) as f64;

        if n >= period {
            let sum: f64 = data[0..period].iter().sum();
            ema[period - 1] = sum / period as f64;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }
}

impl TechnicalIndicator for TechnicalRating {
    fn name(&self) -> &str {
        "TechnicalRating"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(
            result.rating,
            result.ma_rating,
            result.oscillator_rating,
        ))
    }

    fn min_periods(&self) -> usize {
        self.ema_long.max(self.rsi_period).max(self.stoch_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// 30. MarketPhaseDetector
// ============================================================================

/// Market phase types detected by MarketPhaseDetector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedPhase {
    /// Accumulation phase (bottoming).
    Accumulation,
    /// Markup phase (uptrend).
    Markup,
    /// Distribution phase (topping).
    Distribution,
    /// Markdown phase (downtrend).
    Markdown,
}

impl DetectedPhase {
    /// Convert phase to numeric value.
    pub fn to_value(&self) -> f64 {
        match self {
            DetectedPhase::Accumulation => 1.0,
            DetectedPhase::Markup => 2.0,
            DetectedPhase::Distribution => 3.0,
            DetectedPhase::Markdown => 4.0,
        }
    }
}

/// Market Phase Detector output.
#[derive(Debug, Clone)]
pub struct MarketPhaseDetectorOutput {
    /// Detected market phase.
    pub phase: Vec<DetectedPhase>,
    /// Phase value for charting (1-4).
    pub phase_value: Vec<f64>,
    /// Phase confidence (0-100).
    pub confidence: Vec<f64>,
    /// Trend strength indicator.
    pub trend_strength: Vec<f64>,
}

/// Market Phase Detector configuration.
#[derive(Debug, Clone)]
pub struct MarketPhaseDetectorConfig {
    /// Period for trend analysis (default: 20).
    pub trend_period: usize,
    /// Period for volatility analysis (default: 14).
    pub volatility_period: usize,
    /// Period for volume analysis (default: 20).
    pub volume_period: usize,
    /// Threshold for trend strength (default: 25.0).
    pub trend_threshold: f64,
}

impl Default for MarketPhaseDetectorConfig {
    fn default() -> Self {
        Self {
            trend_period: 20,
            volatility_period: 14,
            volume_period: 20,
            trend_threshold: 25.0,
        }
    }
}

/// Market Phase Detector.
///
/// Detects the current market phase using Wyckoff market cycle theory.
/// Identifies four distinct phases: Accumulation, Markup, Distribution, and Markdown.
///
/// # Phases
///
/// ## Accumulation
/// - Price at low levels after markdown
/// - Volatility contracting
/// - Smart money accumulating positions
///
/// ## Markup
/// - Price trending upward
/// - Higher highs and higher lows
/// - Strong bullish momentum
///
/// ## Distribution
/// - Price at high levels after markup
/// - Volatility may increase
/// - Smart money distributing positions
///
/// ## Markdown
/// - Price trending downward
/// - Lower highs and lower lows
/// - Strong bearish momentum
///
/// # Example
///
/// ```ignore
/// use indicator_core::composite::advanced::{MarketPhaseDetector, MarketPhaseDetectorConfig};
///
/// let config = MarketPhaseDetectorConfig::default();
/// let indicator = MarketPhaseDetector::new(config).unwrap();
/// let result = indicator.calculate(&high, &low, &close, &volume);
/// ```
#[derive(Debug, Clone)]
pub struct MarketPhaseDetector {
    trend_period: usize,
    volatility_period: usize,
    volume_period: usize,
    trend_threshold: f64,
}

impl MarketPhaseDetector {
    /// Create a new MarketPhaseDetector with the given configuration.
    pub fn new(config: MarketPhaseDetectorConfig) -> Result<Self> {
        if config.trend_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volatility_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.volume_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if config.trend_threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_threshold".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            trend_period: config.trend_period,
            volatility_period: config.volatility_period,
            volume_period: config.volume_period,
            trend_threshold: config.trend_threshold,
        })
    }

    /// Calculate the Market Phase Detector values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> MarketPhaseDetectorOutput {
        let n = close.len();
        let mut phase = vec![DetectedPhase::Accumulation; n];
        let mut phase_value = vec![f64::NAN; n];
        let mut confidence = vec![f64::NAN; n];
        let mut trend_strength = vec![f64::NAN; n];

        let max_period = self.trend_period.max(self.volatility_period).max(self.volume_period);
        if n <= max_period {
            return MarketPhaseDetectorOutput {
                phase,
                phase_value,
                confidence,
                trend_strength,
            };
        }

        // Calculate price trend direction and strength
        for i in self.trend_period..n {
            let net_change = close[i] - close[i - self.trend_period];
            let mut sum_abs_change = 0.0;
            for j in (i - self.trend_period + 1)..=i {
                sum_abs_change += (close[j] - close[j - 1]).abs();
            }

            // Efficiency ratio for trend strength
            let efficiency = if sum_abs_change > 1e-10 {
                (net_change.abs() / sum_abs_change * 100.0).clamp(0.0, 100.0)
            } else {
                0.0
            };

            trend_strength[i] = efficiency;

            // Calculate price position in range
            let highest = high[(i - self.trend_period + 1)..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = low[(i - self.trend_period + 1)..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let range = highest - lowest;
            let position = if range > 1e-10 {
                (close[i] - lowest) / range
            } else {
                0.5
            };

            // Calculate volatility (ATR as % of price)
            let mut atr = 0.0;
            for j in (i - self.volatility_period + 1)..=i {
                let tr = if j == 0 {
                    high[j] - low[j]
                } else {
                    (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs())
                };
                atr += tr;
            }
            atr /= self.volatility_period as f64;
            let volatility_pct = if close[i].abs() > 1e-10 { atr / close[i] * 100.0 } else { 0.0 };

            // Volume analysis
            let avg_volume: f64 = volume[(i - self.volume_period + 1)..=i].iter().sum::<f64>()
                / self.volume_period as f64;
            let volume_ratio = if avg_volume > 1e-10 { volume[i] / avg_volume } else { 1.0 };

            // Determine market phase
            let is_trending = trend_strength[i] > self.trend_threshold;
            let is_uptrend = net_change > 0.0;
            let is_high_position = position > 0.7;
            let is_low_position = position < 0.3;

            let detected_phase = if is_trending {
                if is_uptrend {
                    DetectedPhase::Markup
                } else {
                    DetectedPhase::Markdown
                }
            } else {
                // Ranging market - determine accumulation or distribution
                if is_high_position {
                    DetectedPhase::Distribution
                } else if is_low_position {
                    DetectedPhase::Accumulation
                } else {
                    // Middle of range - use volume for hints
                    if volume_ratio > 1.1 && is_uptrend {
                        DetectedPhase::Accumulation
                    } else if volume_ratio > 1.1 && !is_uptrend {
                        DetectedPhase::Distribution
                    } else if position > 0.5 {
                        DetectedPhase::Distribution
                    } else {
                        DetectedPhase::Accumulation
                    }
                }
            };

            phase[i] = detected_phase;
            phase_value[i] = detected_phase.to_value();

            // Calculate confidence based on how clear the signals are
            let trend_confidence = if is_trending { trend_strength[i] } else { (100.0 - trend_strength[i]) * 0.5 };
            let position_confidence = if is_high_position || is_low_position {
                (position - 0.5).abs() * 200.0
            } else {
                30.0
            };
            let volume_confidence = (volume_ratio - 1.0).abs() * 30.0;

            confidence[i] = ((trend_confidence + position_confidence + volume_confidence) / 3.0).clamp(0.0, 100.0);
        }

        MarketPhaseDetectorOutput {
            phase,
            phase_value,
            confidence,
            trend_strength,
        }
    }
}

impl TechnicalIndicator for MarketPhaseDetector {
    fn name(&self) -> &str {
        "MarketPhaseDetector"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            result.phase_value,
            result.confidence,
            result.trend_strength,
        ))
    }

    fn min_periods(&self) -> usize {
        self.trend_period.max(self.volatility_period).max(self.volume_period) + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper functions for generating test data
    fn generate_uptrend_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64 * 0.5).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64 * 0.5).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect();
        (high, low, close)
    }

    fn generate_ranging_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n)
            .map(|i| 105.0 + (i as f64 * 0.3).sin() * 3.0)
            .collect();
        let low: Vec<f64> = (0..n)
            .map(|i| 95.0 + (i as f64 * 0.3).sin() * 3.0)
            .collect();
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 3.0)
            .collect();
        (high, low, close)
    }

    fn generate_volatile_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n)
            .map(|i| 110.0 + (i as f64 * 0.5).sin() * 10.0)
            .collect();
        let low: Vec<f64> = (0..n)
            .map(|i| 90.0 + (i as f64 * 0.5).sin() * 10.0)
            .collect();
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 8.0)
            .collect();
        (high, low, close)
    }

    // ========== TrendVolatilityIndex Tests ==========

    #[test]
    fn test_trend_volatility_index_new() {
        let config = TrendVolatilityIndexConfig::default();
        let indicator = TrendVolatilityIndex::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_trend_volatility_index_invalid_trend_period() {
        let config = TrendVolatilityIndexConfig {
            trend_period: 0,
            ..Default::default()
        };
        let result = TrendVolatilityIndex::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "trend_period");
        }
    }

    #[test]
    fn test_trend_volatility_index_invalid_weight() {
        let config = TrendVolatilityIndexConfig {
            trend_weight: 1.5,
            ..Default::default()
        };
        let result = TrendVolatilityIndex::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_volatility_index_calculate() {
        let config = TrendVolatilityIndexConfig::default();
        let indicator = TrendVolatilityIndex::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.index.len(), 50);
        assert_eq!(result.trend_component.len(), 50);
        assert_eq!(result.volatility_component.len(), 50);

        // Check valid values are in range
        for i in 20..50 {
            if !result.index[i].is_nan() {
                assert!(result.index[i] >= 0.0 && result.index[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_trend_volatility_index_trait() {
        let config = TrendVolatilityIndexConfig::default();
        let indicator = TrendVolatilityIndex::new(config).unwrap();

        assert_eq!(indicator.name(), "TrendVolatilityIndex");
        assert_eq!(indicator.min_periods(), 15);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_trend_volatility_index_compute() {
        let config = TrendVolatilityIndexConfig::default();
        let indicator = TrendVolatilityIndex::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== MomentumQualityScore Tests ==========

    #[test]
    fn test_momentum_quality_score_new() {
        let config = MomentumQualityScoreConfig::default();
        let indicator = MomentumQualityScore::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_momentum_quality_score_invalid_period() {
        let config = MomentumQualityScoreConfig {
            momentum_period: 0,
            ..Default::default()
        };
        let result = MomentumQualityScore::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_momentum_quality_score_calculate() {
        let config = MomentumQualityScoreConfig::default();
        let indicator = MomentumQualityScore::new(config).unwrap();
        let (_, _, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&close);

        assert_eq!(result.score.len(), 50);
        assert_eq!(result.raw_momentum.len(), 50);
        assert_eq!(result.quality_factor.len(), 50);

        // Quality factor should be between 0 and 1
        for i in 25..50 {
            if !result.quality_factor[i].is_nan() {
                assert!(result.quality_factor[i] >= 0.0 && result.quality_factor[i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_momentum_quality_score_trait() {
        let config = MomentumQualityScoreConfig::default();
        let indicator = MomentumQualityScore::new(config).unwrap();

        assert_eq!(indicator.name(), "MomentumQualityScore");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_momentum_quality_score_compute() {
        let config = MomentumQualityScoreConfig::default();
        let indicator = MomentumQualityScore::new(config).unwrap();
        let (_, _, close) = generate_uptrend_data(50);

        let series = OHLCVSeries::from_close(close);

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
    }

    // ========== MarketPhaseIndicator Tests ==========

    #[test]
    fn test_market_phase_indicator_new() {
        let config = MarketPhaseIndicatorConfig::default();
        let indicator = MarketPhaseIndicator::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_market_phase_indicator_invalid_period() {
        let config = MarketPhaseIndicatorConfig {
            trend_period: 0,
            ..Default::default()
        };
        let result = MarketPhaseIndicator::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_market_phase_indicator_invalid_threshold() {
        let config = MarketPhaseIndicatorConfig {
            trend_threshold: 0.0,
            ..Default::default()
        };
        let result = MarketPhaseIndicator::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_market_phase_indicator_calculate_trending() {
        let config = MarketPhaseIndicatorConfig::default();
        let indicator = MarketPhaseIndicator::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.phase.len(), 50);

        // In uptrend, should see some trending phases
        let trending_count = result
            .phase
            .iter()
            .filter(|&&p| p == MarketPhase::Trending)
            .count();
        assert!(trending_count > 0, "Expected trending phases in uptrend data");
    }

    #[test]
    fn test_market_phase_indicator_calculate_ranging() {
        let config = MarketPhaseIndicatorConfig::default();
        let indicator = MarketPhaseIndicator::new(config).unwrap();
        let (high, low, close) = generate_ranging_data(50);

        let result = indicator.calculate(&high, &low, &close);

        // In ranging market, should see ranging or quiet phases
        let non_trending_count = result
            .phase
            .iter()
            .filter(|&&p| p == MarketPhase::Ranging || p == MarketPhase::Quiet)
            .count();
        assert!(non_trending_count > 0);
    }

    #[test]
    fn test_market_phase_numeric_conversion() {
        assert_eq!(MarketPhase::Trending.to_numeric(), 1.0);
        assert_eq!(MarketPhase::Ranging.to_numeric(), 0.0);
        assert_eq!(MarketPhase::Volatile.to_numeric(), 2.0);
        assert_eq!(MarketPhase::Quiet.to_numeric(), -1.0);
    }

    #[test]
    fn test_market_phase_indicator_trait() {
        let config = MarketPhaseIndicatorConfig::default();
        let indicator = MarketPhaseIndicator::new(config).unwrap();

        assert_eq!(indicator.name(), "MarketPhaseIndicator");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    // ========== PriceTrendStrength Tests ==========

    #[test]
    fn test_price_trend_strength_new() {
        let config = PriceTrendStrengthConfig::default();
        let indicator = PriceTrendStrength::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_price_trend_strength_invalid_period_order() {
        let config = PriceTrendStrengthConfig {
            short_period: 10,
            medium_period: 5,
            long_period: 20,
        };
        let result = PriceTrendStrength::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_price_trend_strength_calculate() {
        let config = PriceTrendStrengthConfig::default();
        let indicator = PriceTrendStrength::new(config).unwrap();
        let (_, _, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&close);

        assert_eq!(result.strength.len(), 50);
        assert_eq!(result.price_momentum.len(), 50);
        assert_eq!(result.trend_persistence.len(), 50);

        // Check valid values are in range
        for i in 25..50 {
            if !result.strength[i].is_nan() {
                assert!(result.strength[i] >= 0.0 && result.strength[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_price_trend_strength_trait() {
        let config = PriceTrendStrengthConfig::default();
        let indicator = PriceTrendStrength::new(config).unwrap();

        assert_eq!(indicator.name(), "PriceTrendStrength");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_price_trend_strength_compute() {
        let config = PriceTrendStrengthConfig::default();
        let indicator = PriceTrendStrength::new(config).unwrap();
        let (_, _, close) = generate_uptrend_data(50);

        let series = OHLCVSeries::from_close(close);

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
    }

    // ========== AdaptiveMarketIndicator Tests ==========

    #[test]
    fn test_adaptive_market_indicator_new() {
        let config = AdaptiveMarketIndicatorConfig::default();
        let indicator = AdaptiveMarketIndicator::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_adaptive_market_indicator_invalid_period_order() {
        let config = AdaptiveMarketIndicatorConfig {
            base_period: 30,
            max_period: 10,
            efficiency_period: 10,
        };
        let result = AdaptiveMarketIndicator::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaptive_market_indicator_calculate() {
        let config = AdaptiveMarketIndicatorConfig::default();
        let indicator = AdaptiveMarketIndicator::new(config).unwrap();
        let (_, _, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&close);

        assert_eq!(result.value.len(), 50);
        assert_eq!(result.adaptation_factor.len(), 50);
        assert_eq!(result.efficiency.len(), 50);

        // Efficiency should be between 0 and 1
        for i in 35..50 {
            if !result.efficiency[i].is_nan() {
                assert!(result.efficiency[i] >= 0.0 && result.efficiency[i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_adaptive_market_indicator_trending_efficiency() {
        let config = AdaptiveMarketIndicatorConfig::default();
        let indicator = AdaptiveMarketIndicator::new(config).unwrap();
        let (_, _, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&close);

        // In a linear uptrend, efficiency should be high
        let avg_efficiency: f64 = result.efficiency[35..50]
            .iter()
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / result.efficiency[35..50]
                .iter()
                .filter(|x| !x.is_nan())
                .count() as f64;
        assert!(avg_efficiency > 0.5, "Expected high efficiency in uptrend");
    }

    #[test]
    fn test_adaptive_market_indicator_trait() {
        let config = AdaptiveMarketIndicatorConfig::default();
        let indicator = AdaptiveMarketIndicator::new(config).unwrap();

        assert_eq!(indicator.name(), "AdaptiveMarketIndicator");
        assert_eq!(indicator.min_periods(), 31);
        assert_eq!(indicator.output_features(), 3);
    }

    // ========== CompositeSignalStrength Tests ==========

    #[test]
    fn test_composite_signal_strength_new() {
        let config = CompositeSignalStrengthConfig::default();
        let indicator = CompositeSignalStrength::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_composite_signal_strength_invalid_period() {
        let config = CompositeSignalStrengthConfig {
            trend_period: 0,
            ..Default::default()
        };
        let result = CompositeSignalStrength::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_composite_signal_strength_invalid_weight() {
        let config = CompositeSignalStrengthConfig {
            trend_weight: 1.5,
            ..Default::default()
        };
        let result = CompositeSignalStrength::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_composite_signal_strength_calculate() {
        let config = CompositeSignalStrengthConfig::default();
        let indicator = CompositeSignalStrength::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.strength.len(), 50);
        assert_eq!(result.trend_signal.len(), 50);
        assert_eq!(result.momentum_signal.len(), 50);
        assert_eq!(result.volatility_signal.len(), 50);

        // Check valid values are in range
        for i in 25..50 {
            if !result.strength[i].is_nan() {
                assert!(result.strength[i] >= -100.0 && result.strength[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_composite_signal_strength_uptrend() {
        let config = CompositeSignalStrengthConfig::default();
        let indicator = CompositeSignalStrength::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&high, &low, &close);

        // In uptrend, strength should tend positive
        let avg_strength: f64 = result.strength[25..50]
            .iter()
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / result.strength[25..50]
                .iter()
                .filter(|x| !x.is_nan())
                .count() as f64;
        assert!(avg_strength > 0.0, "Expected positive strength in uptrend");
    }

    #[test]
    fn test_composite_signal_strength_trait() {
        let config = CompositeSignalStrengthConfig::default();
        let indicator = CompositeSignalStrength::new(config).unwrap();

        assert_eq!(indicator.name(), "CompositeSignalStrength");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_composite_signal_strength_compute() {
        let config = CompositeSignalStrengthConfig::default();
        let indicator = CompositeSignalStrength::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_composite_signal_strength_insufficient_data() {
        let config = CompositeSignalStrengthConfig::default();
        let indicator = CompositeSignalStrength::new(config).unwrap();

        let series = OHLCVSeries::from_close(vec![100.0; 10]);

        let result = indicator.compute(&series);
        assert!(result.is_err());
        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 21);
            assert_eq!(got, 10);
        }
    }

    // ========== AdaptiveCompositeScore Tests ==========

    #[test]
    fn test_adaptive_composite_score_new() {
        let config = AdaptiveCompositeScoreConfig::default();
        let indicator = AdaptiveCompositeScore::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_adaptive_composite_score_invalid_period() {
        let config = AdaptiveCompositeScoreConfig {
            regime_period: 0,
            ..Default::default()
        };
        let result = AdaptiveCompositeScore::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaptive_composite_score_calculate() {
        let config = AdaptiveCompositeScoreConfig::default();
        let indicator = AdaptiveCompositeScore::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.score.len(), 50);
        assert_eq!(result.regime_factor.len(), 50);
        assert_eq!(result.base_score.len(), 50);

        for i in 25..50 {
            if !result.score[i].is_nan() {
                assert!(result.score[i] >= -100.0 && result.score[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_adaptive_composite_score_trait() {
        let config = AdaptiveCompositeScoreConfig::default();
        let indicator = AdaptiveCompositeScore::new(config).unwrap();

        assert_eq!(indicator.name(), "AdaptiveCompositeScore");
        assert_eq!(indicator.output_features(), 3);
    }

    // ========== MultiFactorMomentum Tests ==========

    #[test]
    fn test_multi_factor_momentum_new() {
        let config = MultiFactorMomentumConfig::default();
        let indicator = MultiFactorMomentum::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_multi_factor_momentum_invalid_period() {
        let config = MultiFactorMomentumConfig {
            short_period: 0,
            ..Default::default()
        };
        let result = MultiFactorMomentum::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_factor_momentum_calculate() {
        let config = MultiFactorMomentumConfig::default();
        let indicator = MultiFactorMomentum::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.momentum.len(), 50);
        assert_eq!(result.price_factor.len(), 50);
        assert_eq!(result.volume_factor.len(), 50);

        for i in 30..50 {
            if !result.momentum[i].is_nan() {
                assert!(result.momentum[i] >= -100.0 && result.momentum[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_multi_factor_momentum_trait() {
        let config = MultiFactorMomentumConfig::default();
        let indicator = MultiFactorMomentum::new(config).unwrap();

        assert_eq!(indicator.name(), "MultiFactorMomentum");
        assert_eq!(indicator.output_features(), 3);
    }

    // ========== TrendQualityComposite Tests ==========

    #[test]
    fn test_trend_quality_composite_new() {
        let config = TrendQualityCompositeConfig::default();
        let indicator = TrendQualityComposite::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_trend_quality_composite_invalid_period() {
        let config = TrendQualityCompositeConfig {
            trend_period: 0,
            ..Default::default()
        };
        let result = TrendQualityComposite::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_quality_composite_calculate() {
        let config = TrendQualityCompositeConfig::default();
        let indicator = TrendQualityComposite::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.quality.len(), 50);
        assert_eq!(result.trend_strength.len(), 50);
        assert_eq!(result.consistency.len(), 50);

        for i in 25..50 {
            if !result.quality[i].is_nan() {
                assert!(result.quality[i] >= 0.0 && result.quality[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_trend_quality_composite_trait() {
        let config = TrendQualityCompositeConfig::default();
        let indicator = TrendQualityComposite::new(config).unwrap();

        assert_eq!(indicator.name(), "TrendQualityComposite");
        assert_eq!(indicator.output_features(), 3);
    }

    // ========== RiskOnRiskOff Tests ==========

    #[test]
    fn test_risk_on_risk_off_new() {
        let config = RiskOnRiskOffConfig::default();
        let indicator = RiskOnRiskOff::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_risk_on_risk_off_invalid_period() {
        let config = RiskOnRiskOffConfig {
            momentum_period: 0,
            ..Default::default()
        };
        let result = RiskOnRiskOff::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_risk_on_risk_off_calculate() {
        let config = RiskOnRiskOffConfig::default();
        let indicator = RiskOnRiskOff::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.indicator.len(), 50);
        assert_eq!(result.risk_score.len(), 50);
        assert_eq!(result.trend_bias.len(), 50);

        for i in 25..50 {
            if !result.indicator[i].is_nan() {
                assert!(result.indicator[i] >= -100.0 && result.indicator[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_risk_on_risk_off_trait() {
        let config = RiskOnRiskOffConfig::default();
        let indicator = RiskOnRiskOff::new(config).unwrap();

        assert_eq!(indicator.name(), "RiskOnRiskOff");
        assert_eq!(indicator.output_features(), 3);
    }

    // ========== MarketBreadthComposite Tests ==========

    #[test]
    fn test_market_breadth_composite_new() {
        let config = MarketBreadthCompositeConfig::default();
        let indicator = MarketBreadthComposite::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_market_breadth_composite_invalid_period() {
        let config = MarketBreadthCompositeConfig {
            breadth_period: 0,
            ..Default::default()
        };
        let result = MarketBreadthComposite::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_market_breadth_composite_calculate() {
        let config = MarketBreadthCompositeConfig::default();
        let indicator = MarketBreadthComposite::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.breadth.len(), 50);
        assert_eq!(result.advance_decline.len(), 50);
        assert_eq!(result.volume_breadth.len(), 50);

        for i in 25..50 {
            if !result.breadth[i].is_nan() {
                assert!(result.breadth[i] >= -100.0 && result.breadth[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_market_breadth_composite_trait() {
        let config = MarketBreadthCompositeConfig::default();
        let indicator = MarketBreadthComposite::new(config).unwrap();

        assert_eq!(indicator.name(), "MarketBreadthComposite");
        assert_eq!(indicator.output_features(), 3);
    }

    // ========== SentimentTrendComposite Tests ==========

    #[test]
    fn test_sentiment_trend_composite_new() {
        let config = SentimentTrendCompositeConfig::default();
        let indicator = SentimentTrendComposite::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_sentiment_trend_composite_invalid_period() {
        let config = SentimentTrendCompositeConfig {
            sentiment_period: 0,
            ..Default::default()
        };
        let result = SentimentTrendComposite::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_sentiment_trend_composite_invalid_weight() {
        let config = SentimentTrendCompositeConfig {
            sentiment_weight: 1.5,
            ..Default::default()
        };
        let result = SentimentTrendComposite::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_sentiment_trend_composite_calculate() {
        let config = SentimentTrendCompositeConfig::default();
        let indicator = SentimentTrendComposite::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.composite.len(), 50);
        assert_eq!(result.sentiment_score.len(), 50);
        assert_eq!(result.trend_score.len(), 50);

        for i in 25..50 {
            if !result.composite[i].is_nan() {
                assert!(result.composite[i] >= -100.0 && result.composite[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_sentiment_trend_composite_trait() {
        let config = SentimentTrendCompositeConfig::default();
        let indicator = SentimentTrendComposite::new(config).unwrap();

        assert_eq!(indicator.name(), "SentimentTrendComposite");
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_sentiment_trend_composite_compute() {
        let config = SentimentTrendCompositeConfig::default();
        let indicator = SentimentTrendComposite::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== MarketStrengthIndex Tests ==========

    #[test]
    fn test_market_strength_index_new() {
        let config = MarketStrengthIndexConfig::default();
        let indicator = MarketStrengthIndex::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_market_strength_index_invalid_price_period() {
        let config = MarketStrengthIndexConfig {
            price_period: 0,
            ..Default::default()
        };
        let result = MarketStrengthIndex::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "price_period");
        }
    }

    #[test]
    fn test_market_strength_index_invalid_weight() {
        let config = MarketStrengthIndexConfig {
            price_weight: 1.5,
            ..Default::default()
        };
        let result = MarketStrengthIndex::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_market_strength_index_calculate() {
        let config = MarketStrengthIndexConfig::default();
        let indicator = MarketStrengthIndex::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.strength.len(), 50);
        assert_eq!(result.price_strength.len(), 50);
        assert_eq!(result.volume_strength.len(), 50);

        for i in 20..50 {
            if !result.strength[i].is_nan() {
                assert!(result.strength[i] >= 0.0 && result.strength[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_market_strength_index_trait() {
        let config = MarketStrengthIndexConfig::default();
        let indicator = MarketStrengthIndex::new(config).unwrap();

        assert_eq!(indicator.name(), "MarketStrengthIndex");
        assert_eq!(indicator.min_periods(), 15);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_market_strength_index_compute() {
        let config = MarketStrengthIndexConfig::default();
        let indicator = MarketStrengthIndex::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== TrendMomentumComposite Tests ==========

    #[test]
    fn test_trend_momentum_composite_new() {
        let config = TrendMomentumCompositeConfig::default();
        let indicator = TrendMomentumComposite::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_trend_momentum_composite_invalid_period() {
        let config = TrendMomentumCompositeConfig {
            short_ema_period: 0,
            ..Default::default()
        };
        let result = TrendMomentumComposite::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_momentum_composite_invalid_period_order() {
        let config = TrendMomentumCompositeConfig {
            short_ema_period: 30,
            long_ema_period: 20,
            ..Default::default()
        };
        let result = TrendMomentumComposite::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_momentum_composite_invalid_weight() {
        let config = TrendMomentumCompositeConfig {
            trend_weight: 1.5,
            ..Default::default()
        };
        let result = TrendMomentumComposite::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_momentum_composite_calculate() {
        let config = TrendMomentumCompositeConfig::default();
        let indicator = TrendMomentumComposite::new(config).unwrap();
        let (_, _, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&close);

        assert_eq!(result.signal.len(), 50);
        assert_eq!(result.trend_component.len(), 50);
        assert_eq!(result.momentum_component.len(), 50);

        for i in 30..50 {
            if !result.signal[i].is_nan() {
                assert!(result.signal[i] >= -100.0 && result.signal[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_trend_momentum_composite_trait() {
        let config = TrendMomentumCompositeConfig::default();
        let indicator = TrendMomentumComposite::new(config).unwrap();

        assert_eq!(indicator.name(), "TrendMomentumComposite");
        assert_eq!(indicator.min_periods(), 27);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_trend_momentum_composite_compute() {
        let config = TrendMomentumCompositeConfig::default();
        let indicator = TrendMomentumComposite::new(config).unwrap();
        let (_, _, close) = generate_uptrend_data(50);

        let series = OHLCVSeries::from_close(close);

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== VolatilityTrendIndex Tests ==========

    #[test]
    fn test_volatility_trend_index_new() {
        let config = VolatilityTrendIndexConfig::default();
        let indicator = VolatilityTrendIndex::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_volatility_trend_index_invalid_atr_period() {
        let config = VolatilityTrendIndexConfig {
            atr_period: 0,
            ..Default::default()
        };
        let result = VolatilityTrendIndex::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "atr_period");
        }
    }

    #[test]
    fn test_volatility_trend_index_calculate() {
        let config = VolatilityTrendIndexConfig::default();
        let indicator = VolatilityTrendIndex::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(70);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.index.len(), 70);
        assert_eq!(result.volatility_regime.len(), 70);
        assert_eq!(result.trend_direction.len(), 70);

        for i in 55..70 {
            if !result.index[i].is_nan() {
                assert!(result.index[i] >= -100.0 && result.index[i] <= 100.0);
            }
            if !result.volatility_regime[i].is_nan() {
                assert!(result.volatility_regime[i] >= 0.0 && result.volatility_regime[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_volatility_trend_index_trait() {
        let config = VolatilityTrendIndexConfig::default();
        let indicator = VolatilityTrendIndex::new(config).unwrap();

        assert_eq!(indicator.name(), "VolatilityTrendIndex");
        assert_eq!(indicator.min_periods(), 51);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_volatility_trend_index_compute() {
        let config = VolatilityTrendIndexConfig::default();
        let indicator = VolatilityTrendIndex::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(70);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 70],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 70);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== MultiFactorSignal Tests ==========

    #[test]
    fn test_multi_factor_signal_new() {
        let config = MultiFactorSignalConfig::default();
        let indicator = MultiFactorSignal::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_multi_factor_signal_invalid_trend_period() {
        let config = MultiFactorSignalConfig {
            trend_period: 0,
            ..Default::default()
        };
        let result = MultiFactorSignal::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "trend_period");
        }
    }

    #[test]
    fn test_multi_factor_signal_invalid_weight() {
        let config = MultiFactorSignalConfig {
            trend_weight: 1.5,
            ..Default::default()
        };
        let result = MultiFactorSignal::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_factor_signal_calculate() {
        let config = MultiFactorSignalConfig::default();
        let indicator = MultiFactorSignal::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.signal.len(), 50);
        assert_eq!(result.trend_factor.len(), 50);
        assert_eq!(result.momentum_factor.len(), 50);
        assert_eq!(result.volatility_factor.len(), 50);

        for i in 25..50 {
            if !result.signal[i].is_nan() {
                assert!(result.signal[i] >= -100.0 && result.signal[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_multi_factor_signal_trait() {
        let config = MultiFactorSignalConfig::default();
        let indicator = MultiFactorSignal::new(config).unwrap();

        assert_eq!(indicator.name(), "MultiFactorSignal");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_multi_factor_signal_compute() {
        let config = MultiFactorSignalConfig::default();
        let indicator = MultiFactorSignal::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== AdaptiveMarketScore Tests ==========

    #[test]
    fn test_adaptive_market_score_new() {
        let config = AdaptiveMarketScoreConfig::default();
        let indicator = AdaptiveMarketScore::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_adaptive_market_score_invalid_fast_period() {
        let config = AdaptiveMarketScoreConfig {
            fast_period: 0,
            ..Default::default()
        };
        let result = AdaptiveMarketScore::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "fast_period");
        }
    }

    #[test]
    fn test_adaptive_market_score_invalid_period_order() {
        let config = AdaptiveMarketScoreConfig {
            fast_period: 25,
            slow_period: 20,
            ..Default::default()
        };
        let result = AdaptiveMarketScore::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaptive_market_score_calculate() {
        let config = AdaptiveMarketScoreConfig::default();
        let indicator = AdaptiveMarketScore::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.score.len(), 50);
        assert_eq!(result.market_condition.len(), 50);
        assert_eq!(result.adaptive_weight.len(), 50);

        for i in 25..50 {
            if !result.score[i].is_nan() {
                assert!(result.score[i] >= -100.0 && result.score[i] <= 100.0);
            }
            if !result.market_condition[i].is_nan() {
                assert!(result.market_condition[i] >= 0.0 && result.market_condition[i] <= 100.0);
            }
            if !result.adaptive_weight[i].is_nan() {
                assert!(result.adaptive_weight[i] >= 0.0 && result.adaptive_weight[i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_adaptive_market_score_trait() {
        let config = AdaptiveMarketScoreConfig::default();
        let indicator = AdaptiveMarketScore::new(config).unwrap();

        assert_eq!(indicator.name(), "AdaptiveMarketScore");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_adaptive_market_score_compute() {
        let config = AdaptiveMarketScoreConfig::default();
        let indicator = AdaptiveMarketScore::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== CompositeLeadingIndicator Tests ==========

    #[test]
    fn test_composite_leading_indicator_new() {
        let config = CompositeLeadingIndicatorConfig::default();
        let indicator = CompositeLeadingIndicator::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_composite_leading_indicator_invalid_short_period() {
        let config = CompositeLeadingIndicatorConfig {
            short_period: 0,
            ..Default::default()
        };
        let result = CompositeLeadingIndicator::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "short_period");
        }
    }

    #[test]
    fn test_composite_leading_indicator_invalid_period_order() {
        let config = CompositeLeadingIndicatorConfig {
            short_period: 15,
            medium_period: 10,
            long_period: 20,
            ..Default::default()
        };
        let result = CompositeLeadingIndicator::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_composite_leading_indicator_calculate() {
        let config = CompositeLeadingIndicatorConfig::default();
        let indicator = CompositeLeadingIndicator::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.indicator.len(), 50);
        assert_eq!(result.price_lead.len(), 50);
        assert_eq!(result.volume_lead.len(), 50);
        assert_eq!(result.roc_lead.len(), 50);

        for i in 25..50 {
            if !result.indicator[i].is_nan() {
                assert!(result.indicator[i] >= -100.0 && result.indicator[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_composite_leading_indicator_trait() {
        let config = CompositeLeadingIndicatorConfig::default();
        let indicator = CompositeLeadingIndicator::new(config).unwrap();

        assert_eq!(indicator.name(), "CompositeLeadingIndicator");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_composite_leading_indicator_compute() {
        let config = CompositeLeadingIndicatorConfig::default();
        let indicator = CompositeLeadingIndicator::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_composite_leading_indicator_insufficient_data() {
        let config = CompositeLeadingIndicatorConfig::default();
        let indicator = CompositeLeadingIndicator::new(config).unwrap();

        let series = OHLCVSeries::from_close(vec![100.0; 10]);

        let result = indicator.compute(&series);
        assert!(result.is_err());
        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 21);
            assert_eq!(got, 10);
        }
    }

    // ========== TechnicalScorecard Tests ==========

    #[test]
    fn test_technical_scorecard_new() {
        let config = TechnicalScorecardConfig::default();
        let indicator = TechnicalScorecard::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_technical_scorecard_invalid_trend_period() {
        let config = TechnicalScorecardConfig {
            trend_period: 0,
            ..Default::default()
        };
        let result = TechnicalScorecard::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "trend_period");
        }
    }

    #[test]
    fn test_technical_scorecard_invalid_weight() {
        let config = TechnicalScorecardConfig {
            trend_weight: 1.5,
            ..Default::default()
        };
        let result = TechnicalScorecard::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_technical_scorecard_calculate() {
        let config = TechnicalScorecardConfig::default();
        let indicator = TechnicalScorecard::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = vec![1000.0; 50];

        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.score.len(), 50);
        assert_eq!(result.trend_score.len(), 50);
        assert_eq!(result.momentum_score.len(), 50);
        assert_eq!(result.volatility_score.len(), 50);

        for i in 25..50 {
            if !result.score[i].is_nan() {
                assert!(result.score[i] >= 0.0 && result.score[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_technical_scorecard_trait() {
        let config = TechnicalScorecardConfig::default();
        let indicator = TechnicalScorecard::new(config).unwrap();

        assert_eq!(indicator.name(), "TechnicalScorecard");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_technical_scorecard_compute() {
        let config = TechnicalScorecardConfig::default();
        let indicator = TechnicalScorecard::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== TrendMomentumVolume Tests ==========

    #[test]
    fn test_trend_momentum_volume_new() {
        let config = TrendMomentumVolumeConfig::default();
        let indicator = TrendMomentumVolume::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_trend_momentum_volume_invalid_trend_period() {
        let config = TrendMomentumVolumeConfig {
            trend_period: 0,
            ..Default::default()
        };
        let result = TrendMomentumVolume::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "trend_period");
        }
    }

    #[test]
    fn test_trend_momentum_volume_invalid_weight() {
        let config = TrendMomentumVolumeConfig {
            trend_weight: 1.5,
            ..Default::default()
        };
        let result = TrendMomentumVolume::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_momentum_volume_calculate() {
        let config = TrendMomentumVolumeConfig::default();
        let indicator = TrendMomentumVolume::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.indicator.len(), 50);
        assert_eq!(result.trend_component.len(), 50);
        assert_eq!(result.momentum_component.len(), 50);
        assert_eq!(result.volume_component.len(), 50);

        for i in 25..50 {
            if !result.indicator[i].is_nan() {
                assert!(result.indicator[i] >= -100.0 && result.indicator[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_trend_momentum_volume_trait() {
        let config = TrendMomentumVolumeConfig::default();
        let indicator = TrendMomentumVolume::new(config).unwrap();

        assert_eq!(indicator.name(), "TrendMomentumVolume");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_trend_momentum_volume_compute() {
        let config = TrendMomentumVolumeConfig::default();
        let indicator = TrendMomentumVolume::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== MultiIndicatorConfluence Tests ==========

    #[test]
    fn test_multi_indicator_confluence_new() {
        let config = MultiIndicatorConfluenceConfig::default();
        let indicator = MultiIndicatorConfluence::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_multi_indicator_confluence_invalid_rsi_period() {
        let config = MultiIndicatorConfluenceConfig {
            rsi_period: 0,
            ..Default::default()
        };
        let result = MultiIndicatorConfluence::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "rsi_period");
        }
    }

    #[test]
    fn test_multi_indicator_confluence_invalid_macd_order() {
        let config = MultiIndicatorConfluenceConfig {
            macd_fast: 30,
            macd_slow: 20,
            ..Default::default()
        };
        let result = MultiIndicatorConfluence::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_indicator_confluence_calculate() {
        let config = MultiIndicatorConfluenceConfig::default();
        let indicator = MultiIndicatorConfluence::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.confluence.len(), 50);
        assert_eq!(result.agreement_count.len(), 50);
        assert_eq!(result.avg_signal_strength.len(), 50);

        for i in 30..50 {
            if !result.confluence[i].is_nan() {
                assert!(result.confluence[i] >= 0.0 && result.confluence[i] <= 100.0);
            }
            if !result.agreement_count[i].is_nan() {
                assert!(result.agreement_count[i] >= 0.0 && result.agreement_count[i] <= 5.0);
            }
        }
    }

    #[test]
    fn test_multi_indicator_confluence_trait() {
        let config = MultiIndicatorConfluenceConfig::default();
        let indicator = MultiIndicatorConfluence::new(config).unwrap();

        assert_eq!(indicator.name(), "MultiIndicatorConfluence");
        assert_eq!(indicator.min_periods(), 27);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_multi_indicator_confluence_compute() {
        let config = MultiIndicatorConfluenceConfig::default();
        let indicator = MultiIndicatorConfluence::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== MarketConditionIndex Tests ==========

    #[test]
    fn test_market_condition_index_new() {
        let config = MarketConditionIndexConfig::default();
        let indicator = MarketConditionIndex::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_market_condition_index_invalid_trend_period() {
        let config = MarketConditionIndexConfig {
            trend_period: 0,
            ..Default::default()
        };
        let result = MarketConditionIndex::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "trend_period");
        }
    }

    #[test]
    fn test_market_condition_index_calculate() {
        let config = MarketConditionIndexConfig::default();
        let indicator = MarketConditionIndex::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.index.len(), 50);
        assert_eq!(result.trend_condition.len(), 50);
        assert_eq!(result.volatility_condition.len(), 50);
        assert_eq!(result.momentum_condition.len(), 50);

        for i in 25..50 {
            if !result.index[i].is_nan() {
                assert!(result.index[i] >= 0.0 && result.index[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_market_condition_index_trait() {
        let config = MarketConditionIndexConfig::default();
        let indicator = MarketConditionIndex::new(config).unwrap();

        assert_eq!(indicator.name(), "MarketConditionIndex");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_market_condition_index_compute() {
        let config = MarketConditionIndexConfig::default();
        let indicator = MarketConditionIndex::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== SignalStrengthComposite Tests ==========

    #[test]
    fn test_signal_strength_composite_new() {
        let config = SignalStrengthCompositeConfig::default();
        let indicator = SignalStrengthComposite::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_signal_strength_composite_invalid_primary_period() {
        let config = SignalStrengthCompositeConfig {
            primary_period: 0,
            ..Default::default()
        };
        let result = SignalStrengthComposite::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "primary_period");
        }
    }

    #[test]
    fn test_signal_strength_composite_calculate() {
        let config = SignalStrengthCompositeConfig::default();
        let indicator = SignalStrengthComposite::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.strength.len(), 50);
        assert_eq!(result.primary_signal.len(), 50);
        assert_eq!(result.confirmation_signal.len(), 50);
        assert_eq!(result.quality_score.len(), 50);

        for i in 25..50 {
            if !result.strength[i].is_nan() {
                assert!(result.strength[i] >= -100.0 && result.strength[i] <= 100.0);
            }
            if !result.quality_score[i].is_nan() {
                assert!(result.quality_score[i] >= 0.0 && result.quality_score[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_signal_strength_composite_trait() {
        let config = SignalStrengthCompositeConfig::default();
        let indicator = SignalStrengthComposite::new(config).unwrap();

        assert_eq!(indicator.name(), "SignalStrengthComposite");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_signal_strength_composite_compute() {
        let config = SignalStrengthCompositeConfig::default();
        let indicator = SignalStrengthComposite::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== AdaptiveCompositeMA Tests ==========

    #[test]
    fn test_adaptive_composite_ma_new() {
        let config = AdaptiveCompositeMAConfig::default();
        let indicator = AdaptiveCompositeMA::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_adaptive_composite_ma_invalid_fast_period() {
        let config = AdaptiveCompositeMAConfig {
            fast_period: 0,
            ..Default::default()
        };
        let result = AdaptiveCompositeMA::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "fast_period");
        }
    }

    #[test]
    fn test_adaptive_composite_ma_invalid_period_order() {
        let config = AdaptiveCompositeMAConfig {
            fast_period: 40,
            slow_period: 30,
            ..Default::default()
        };
        let result = AdaptiveCompositeMA::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaptive_composite_ma_calculate() {
        let config = AdaptiveCompositeMAConfig::default();
        let indicator = AdaptiveCompositeMA::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(60);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.ma.len(), 60);
        assert_eq!(result.adaptation_speed.len(), 60);
        assert_eq!(result.trend_direction.len(), 60);

        for i in 35..60 {
            if !result.ma[i].is_nan() {
                // MA should be within reasonable range of price
                assert!(result.ma[i] > 50.0 && result.ma[i] < 200.0);
            }
            if !result.adaptation_speed[i].is_nan() {
                assert!(result.adaptation_speed[i] >= 0.0 && result.adaptation_speed[i] <= 1.0);
            }
            if !result.trend_direction[i].is_nan() {
                assert!(result.trend_direction[i] >= -1.0 && result.trend_direction[i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_adaptive_composite_ma_trait() {
        let config = AdaptiveCompositeMAConfig::default();
        let indicator = AdaptiveCompositeMA::new(config).unwrap();

        assert_eq!(indicator.name(), "AdaptiveCompositeMA");
        assert_eq!(indicator.min_periods(), 31);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_adaptive_composite_ma_compute() {
        let config = AdaptiveCompositeMAConfig::default();
        let indicator = AdaptiveCompositeMA::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(60);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 60],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 60);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_adaptive_composite_ma_insufficient_data() {
        let config = AdaptiveCompositeMAConfig::default();
        let indicator = AdaptiveCompositeMA::new(config).unwrap();

        let series = OHLCVSeries::from_close(vec![100.0; 10]);

        let result = indicator.compute(&series);
        assert!(result.is_err());
        if let Err(IndicatorError::InsufficientData { required, got }) = result {
            assert_eq!(required, 31);
            assert_eq!(got, 10);
        }
    }

    // ========== TrendStrengthComposite Tests ==========

    #[test]
    fn test_trend_strength_composite_new() {
        let config = TrendStrengthCompositeConfig::default();
        let indicator = TrendStrengthComposite::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_trend_strength_composite_invalid_adx_period() {
        let config = TrendStrengthCompositeConfig {
            adx_period: 0,
            ..Default::default()
        };
        let result = TrendStrengthComposite::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "adx_period");
        }
    }

    #[test]
    fn test_trend_strength_composite_invalid_weights() {
        let config = TrendStrengthCompositeConfig {
            adx_weight: 0.5,
            position_weight: 0.5,
            direction_weight: 0.5, // Total > 1.0
            ..Default::default()
        };
        let result = TrendStrengthComposite::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_strength_composite_calculate() {
        let config = TrendStrengthCompositeConfig::default();
        let indicator = TrendStrengthComposite::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(60);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.strength.len(), 60);
        assert_eq!(result.adx_component.len(), 60);
        assert_eq!(result.position_component.len(), 60);
        assert_eq!(result.direction_component.len(), 60);

        for i in 35..60 {
            if !result.strength[i].is_nan() {
                assert!(result.strength[i] >= 0.0 && result.strength[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_trend_strength_composite_trait() {
        let config = TrendStrengthCompositeConfig::default();
        let indicator = TrendStrengthComposite::new(config).unwrap();

        assert_eq!(indicator.name(), "TrendStrengthComposite");
        assert!(indicator.min_periods() > 20);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_trend_strength_composite_compute() {
        let config = TrendStrengthCompositeConfig::default();
        let indicator = TrendStrengthComposite::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(60);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 60],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 60);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== MomentumQualityComposite Tests ==========

    #[test]
    fn test_momentum_quality_composite_new() {
        let config = MomentumQualityCompositeConfig::default();
        let indicator = MomentumQualityComposite::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_momentum_quality_composite_invalid_rsi_period() {
        let config = MomentumQualityCompositeConfig {
            rsi_period: 0,
            ..Default::default()
        };
        let result = MomentumQualityComposite::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "rsi_period");
        }
    }

    #[test]
    fn test_momentum_quality_composite_calculate() {
        let config = MomentumQualityCompositeConfig::default();
        let indicator = MomentumQualityComposite::new(config).unwrap();
        let (_, _, close) = generate_uptrend_data(50);

        let result = indicator.calculate(&close);

        assert_eq!(result.quality.len(), 50);
        assert_eq!(result.rsi_momentum.len(), 50);
        assert_eq!(result.roc_momentum.len(), 50);
        assert_eq!(result.consistency.len(), 50);

        for i in 25..50 {
            if !result.quality[i].is_nan() {
                assert!(result.quality[i] >= 0.0 && result.quality[i] <= 100.0);
            }
            if !result.rsi_momentum[i].is_nan() {
                assert!(result.rsi_momentum[i] >= 0.0 && result.rsi_momentum[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_momentum_quality_composite_trait() {
        let config = MomentumQualityCompositeConfig::default();
        let indicator = MomentumQualityComposite::new(config).unwrap();

        assert_eq!(indicator.name(), "MomentumQualityComposite");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_momentum_quality_composite_compute() {
        let config = MomentumQualityCompositeConfig::default();
        let indicator = MomentumQualityComposite::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== VolatilityAdjustedSignal Tests ==========

    #[test]
    fn test_volatility_adjusted_signal_new() {
        let config = VolatilityAdjustedSignalConfig::default();
        let indicator = VolatilityAdjustedSignal::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_volatility_adjusted_signal_invalid_signal_period() {
        let config = VolatilityAdjustedSignalConfig {
            signal_period: 0,
            ..Default::default()
        };
        let result = VolatilityAdjustedSignal::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "signal_period");
        }
    }

    #[test]
    fn test_volatility_adjusted_signal_calculate() {
        let config = VolatilityAdjustedSignalConfig::default();
        let indicator = VolatilityAdjustedSignal::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(70);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.signal.len(), 70);
        assert_eq!(result.raw_signal.len(), 70);
        assert_eq!(result.volatility_factor.len(), 70);
        assert_eq!(result.confidence.len(), 70);

        for i in 55..70 {
            if !result.signal[i].is_nan() {
                assert!(result.signal[i] >= -100.0 && result.signal[i] <= 100.0);
            }
            if !result.volatility_factor[i].is_nan() {
                assert!(result.volatility_factor[i] >= 0.0 && result.volatility_factor[i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_volatility_adjusted_signal_trait() {
        let config = VolatilityAdjustedSignalConfig::default();
        let indicator = VolatilityAdjustedSignal::new(config).unwrap();

        assert_eq!(indicator.name(), "VolatilityAdjustedSignal");
        assert_eq!(indicator.min_periods(), 51);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_volatility_adjusted_signal_compute() {
        let config = VolatilityAdjustedSignalConfig::default();
        let indicator = VolatilityAdjustedSignal::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(70);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 70],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 70);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== MultiFactorMomentumV2 Tests ==========

    #[test]
    fn test_multi_factor_momentum_v2_new() {
        let config = MultiFactorMomentumV2Config::default();
        let indicator = MultiFactorMomentumV2::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_multi_factor_momentum_v2_invalid_short_period() {
        let config = MultiFactorMomentumV2Config {
            short_period: 0,
            ..Default::default()
        };
        let result = MultiFactorMomentumV2::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "short_period");
        }
    }

    #[test]
    fn test_multi_factor_momentum_v2_invalid_period_order() {
        let config = MultiFactorMomentumV2Config {
            short_period: 15,
            medium_period: 10,
            long_period: 20,
            ..Default::default()
        };
        let result = MultiFactorMomentumV2::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_factor_momentum_v2_calculate() {
        let config = MultiFactorMomentumV2Config::default();
        let indicator = MultiFactorMomentumV2::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.momentum.len(), 50);
        assert_eq!(result.price_factor.len(), 50);
        assert_eq!(result.volume_factor.len(), 50);
        assert_eq!(result.volatility_factor.len(), 50);

        for i in 25..50 {
            if !result.momentum[i].is_nan() {
                assert!(result.momentum[i] >= -100.0 && result.momentum[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_multi_factor_momentum_v2_trait() {
        let config = MultiFactorMomentumV2Config::default();
        let indicator = MultiFactorMomentumV2::new(config).unwrap();

        assert_eq!(indicator.name(), "MultiFactorMomentumV2");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_multi_factor_momentum_v2_compute() {
        let config = MultiFactorMomentumV2Config::default();
        let indicator = MultiFactorMomentumV2::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 10.0)).collect();

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== TechnicalRating Tests ==========

    #[test]
    fn test_technical_rating_new() {
        let config = TechnicalRatingConfig::default();
        let indicator = TechnicalRating::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_technical_rating_invalid_ema_short() {
        let config = TechnicalRatingConfig {
            ema_short: 0,
            ..Default::default()
        };
        let result = TechnicalRating::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "ema_short");
        }
    }

    #[test]
    fn test_technical_rating_calculate() {
        let config = TechnicalRatingConfig::default();
        let indicator = TechnicalRating::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(70);

        let result = indicator.calculate(&high, &low, &close);

        assert_eq!(result.rating.len(), 70);
        assert_eq!(result.ma_rating.len(), 70);
        assert_eq!(result.oscillator_rating.len(), 70);
        assert_eq!(result.summary.len(), 70);

        for i in 55..70 {
            if !result.rating[i].is_nan() {
                assert!(result.rating[i] >= -100.0 && result.rating[i] <= 100.0);
            }
            if !result.summary[i].is_nan() {
                assert!(result.summary[i] >= -2.0 && result.summary[i] <= 2.0);
            }
        }
    }

    #[test]
    fn test_technical_rating_uptrend() {
        let config = TechnicalRatingConfig::default();
        let indicator = TechnicalRating::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(70);

        let result = indicator.calculate(&high, &low, &close);

        // In an uptrend, the rating should generally be positive
        let mut positive_count = 0;
        for i in 55..70 {
            if !result.rating[i].is_nan() && result.rating[i] > 0.0 {
                positive_count += 1;
            }
        }
        assert!(positive_count > 5, "Uptrend should have mostly positive ratings");
    }

    #[test]
    fn test_technical_rating_trait() {
        let config = TechnicalRatingConfig::default();
        let indicator = TechnicalRating::new(config).unwrap();

        assert_eq!(indicator.name(), "TechnicalRating");
        assert_eq!(indicator.min_periods(), 51);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_technical_rating_compute() {
        let config = TechnicalRatingConfig::default();
        let indicator = TechnicalRating::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(70);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 70],
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 70);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ========== MarketPhaseDetector Tests ==========

    #[test]
    fn test_market_phase_detector_new() {
        let config = MarketPhaseDetectorConfig::default();
        let indicator = MarketPhaseDetector::new(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_market_phase_detector_invalid_trend_period() {
        let config = MarketPhaseDetectorConfig {
            trend_period: 0,
            ..Default::default()
        };
        let result = MarketPhaseDetector::new(config);
        assert!(result.is_err());
        if let Err(IndicatorError::InvalidParameter { name, .. }) = result {
            assert_eq!(name, "trend_period");
        }
    }

    #[test]
    fn test_market_phase_detector_invalid_threshold() {
        let config = MarketPhaseDetectorConfig {
            trend_threshold: 0.0,
            ..Default::default()
        };
        let result = MarketPhaseDetector::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_market_phase_detector_calculate() {
        let config = MarketPhaseDetectorConfig::default();
        let indicator = MarketPhaseDetector::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = vec![1000.0; 50];

        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.phase.len(), 50);
        assert_eq!(result.phase_value.len(), 50);
        assert_eq!(result.confidence.len(), 50);
        assert_eq!(result.trend_strength.len(), 50);

        for i in 25..50 {
            if !result.phase_value[i].is_nan() {
                assert!(result.phase_value[i] >= 1.0 && result.phase_value[i] <= 4.0);
            }
            if !result.confidence[i].is_nan() {
                assert!(result.confidence[i] >= 0.0 && result.confidence[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_market_phase_detector_uptrend_markup() {
        let config = MarketPhaseDetectorConfig::default();
        let indicator = MarketPhaseDetector::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = vec![1000.0; 50];

        let result = indicator.calculate(&high, &low, &close, &volume);

        // In strong uptrend, should detect Markup phase
        let mut markup_count = 0;
        for i in 25..50 {
            if result.phase[i] == DetectedPhase::Markup {
                markup_count += 1;
            }
        }
        assert!(markup_count > 0, "Strong uptrend should have Markup phases");
    }

    #[test]
    fn test_market_phase_detector_trait() {
        let config = MarketPhaseDetectorConfig::default();
        let indicator = MarketPhaseDetector::new(config).unwrap();

        assert_eq!(indicator.name(), "MarketPhaseDetector");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 3);
    }

    #[test]
    fn test_market_phase_detector_compute() {
        let config = MarketPhaseDetectorConfig::default();
        let indicator = MarketPhaseDetector::new(config).unwrap();
        let (high, low, close) = generate_uptrend_data(50);
        let volume: Vec<f64> = vec![1000.0; 50];

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let output = indicator.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_detected_phase_to_value() {
        assert_eq!(DetectedPhase::Accumulation.to_value(), 1.0);
        assert_eq!(DetectedPhase::Markup.to_value(), 2.0);
        assert_eq!(DetectedPhase::Distribution.to_value(), 3.0);
        assert_eq!(DetectedPhase::Markdown.to_value(), 4.0);
    }
}
