//! Advanced DeMark Indicators - Extended TD indicator suite.
//!
//! This module contains advanced DeMark-style indicators:
//! - TDLine: TD Trend Line indicator
//! - TDRange: TD Range indicator for volatility
//! - TDChannel: TD Channel indicator
//! - TDQualifier: TD Setup qualifier indicator
//! - TDAlignment: Measures alignment of TD components
//! - TDExhaustion: Enhanced TD exhaustion signal

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

// ============================================================================
// TDLine - TD Trend Line Indicator
// ============================================================================

/// TD Line output containing trend line values.
#[derive(Debug, Clone)]
pub struct TDLineOutput {
    /// Upper trend line values
    pub upper_line: Vec<f64>,
    /// Lower trend line values
    pub lower_line: Vec<f64>,
    /// Trend line slope
    pub slope: Vec<f64>,
    /// Breakout signal: 1.0 = bullish break, -1.0 = bearish break, 0.0 = none
    pub breakout: Vec<f64>,
}

/// TD Trend Line indicator.
///
/// Identifies significant swing highs and lows to draw dynamic trend lines
/// that adapt to price action. Breakouts from these lines signal potential
/// trend changes.
///
/// # Parameters
/// - `lookback`: Period for swing point detection (default: 5)
/// - `confirmation_bars`: Bars required to confirm a breakout (default: 2)
#[derive(Debug, Clone)]
pub struct TDLine {
    lookback: usize,
    confirmation_bars: usize,
}

impl TDLine {
    /// Create a new TDLine indicator with validation.
    pub fn new(lookback: usize, confirmation_bars: usize) -> Result<Self> {
        if lookback < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if confirmation_bars < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "confirmation_bars".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            lookback,
            confirmation_bars,
        })
    }

    /// Calculate TD Line values from OHLCV data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDLineOutput {
        let n = data.close.len();
        let mut upper_line = vec![f64::NAN; n];
        let mut lower_line = vec![f64::NAN; n];
        let mut slope = vec![0.0; n];
        let mut breakout = vec![0.0; n];

        if n < self.lookback * 2 + 1 {
            return TDLineOutput {
                upper_line,
                lower_line,
                slope,
                breakout,
            };
        }

        // Find swing highs and lows
        let mut swing_highs: Vec<(usize, f64)> = Vec::new();
        let mut swing_lows: Vec<(usize, f64)> = Vec::new();

        for i in self.lookback..(n - self.lookback) {
            // Check for swing high
            let mut is_swing_high = true;
            for j in 1..=self.lookback {
                if data.high[i] <= data.high[i - j] || data.high[i] <= data.high[i + j] {
                    is_swing_high = false;
                    break;
                }
            }
            if is_swing_high {
                swing_highs.push((i, data.high[i]));
            }

            // Check for swing low
            let mut is_swing_low = true;
            for j in 1..=self.lookback {
                if data.low[i] >= data.low[i - j] || data.low[i] >= data.low[i + j] {
                    is_swing_low = false;
                    break;
                }
            }
            if is_swing_low {
                swing_lows.push((i, data.low[i]));
            }
        }

        // Calculate trend lines from swing points
        let mut last_upper: Option<(f64, f64)> = None; // (value, slope)
        let mut last_lower: Option<(f64, f64)> = None;

        if swing_highs.len() >= 2 {
            let (idx1, val1) = swing_highs[swing_highs.len() - 2];
            let (idx2, val2) = swing_highs[swing_highs.len() - 1];
            if idx2 > idx1 {
                let s = (val2 - val1) / (idx2 - idx1) as f64;
                last_upper = Some((val2, s));
            }
        }

        if swing_lows.len() >= 2 {
            let (idx1, val1) = swing_lows[swing_lows.len() - 2];
            let (idx2, val2) = swing_lows[swing_lows.len() - 1];
            if idx2 > idx1 {
                let s = (val2 - val1) / (idx2 - idx1) as f64;
                last_lower = Some((val2, s));
            }
        }

        // Project trend lines forward
        if let Some((base_val, s)) = last_upper {
            if let Some(&(base_idx, _)) = swing_highs.last() {
                for i in base_idx..n {
                    let projected = base_val + s * (i - base_idx) as f64;
                    upper_line[i] = projected;
                    slope[i] = s;

                    // Check for breakout above upper line
                    if i >= self.confirmation_bars {
                        let mut confirmed = true;
                        for j in 0..self.confirmation_bars {
                            if data.close[i - j] <= upper_line[i - j] {
                                confirmed = false;
                                break;
                            }
                        }
                        if confirmed {
                            breakout[i] = 1.0;
                        }
                    }
                }
            }
        }

        if let Some((base_val, s)) = last_lower {
            if let Some(&(base_idx, _)) = swing_lows.last() {
                for i in base_idx..n {
                    let projected = base_val + s * (i - base_idx) as f64;
                    lower_line[i] = projected;

                    // Check for breakout below lower line
                    if i >= self.confirmation_bars {
                        let mut confirmed = true;
                        for j in 0..self.confirmation_bars {
                            if data.close[i - j] >= lower_line[i - j] {
                                confirmed = false;
                                break;
                            }
                        }
                        if confirmed && breakout[i] == 0.0 {
                            breakout[i] = -1.0;
                        }
                    }
                }
            }
        }

        TDLineOutput {
            upper_line,
            lower_line,
            slope,
            breakout,
        }
    }
}

impl TechnicalIndicator for TDLine {
    fn name(&self) -> &str {
        "TD Line"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min = self.min_periods();
        if data.close.len() < min {
            return Err(IndicatorError::InsufficientData {
                required: min,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);
        Ok(IndicatorOutput::triple(
            result.upper_line,
            result.lower_line,
            result.breakout,
        ))
    }

    fn min_periods(&self) -> usize {
        self.lookback * 2 + 1
    }
}

// ============================================================================
// TDRange - TD Range Volatility Indicator
// ============================================================================

/// TD Range output containing volatility metrics.
#[derive(Debug, Clone)]
pub struct TDRangeOutput {
    /// TD Range values (normalized volatility)
    pub range: Vec<f64>,
    /// Range expansion signal
    pub expansion: Vec<f64>,
    /// Range contraction signal
    pub contraction: Vec<f64>,
    /// Volatility percentile (0-100)
    pub percentile: Vec<f64>,
}

/// TD Range indicator for volatility measurement.
///
/// Measures price volatility using DeMark-style range analysis.
/// Identifies periods of range expansion and contraction.
///
/// # Parameters
/// - `period`: Lookback period for range calculation (default: 14)
/// - `expansion_threshold`: Multiplier for expansion detection (default: 1.5)
/// - `contraction_threshold`: Multiplier for contraction detection (default: 0.5)
#[derive(Debug, Clone)]
pub struct TDRange {
    period: usize,
    expansion_threshold: f64,
    contraction_threshold: f64,
}

impl TDRange {
    /// Create a new TDRange indicator with validation.
    pub fn new(period: usize, expansion_threshold: f64, contraction_threshold: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if expansion_threshold <= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "expansion_threshold".to_string(),
                reason: "must be greater than 1.0".to_string(),
            });
        }
        if contraction_threshold <= 0.0 || contraction_threshold >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "contraction_threshold".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self {
            period,
            expansion_threshold,
            contraction_threshold,
        })
    }

    /// Calculate TD Range values from OHLCV data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDRangeOutput {
        let n = data.close.len();
        let mut range = vec![f64::NAN; n];
        let mut expansion = vec![0.0; n];
        let mut contraction = vec![0.0; n];
        let mut percentile = vec![f64::NAN; n];

        if n < self.period {
            return TDRangeOutput {
                range,
                expansion,
                contraction,
                percentile,
            };
        }

        // Calculate true range for each bar
        let mut true_ranges = vec![0.0; n];
        true_ranges[0] = data.high[0] - data.low[0];
        for i in 1..n {
            let hl = data.high[i] - data.low[i];
            let hc = (data.high[i] - data.close[i - 1]).abs();
            let lc = (data.low[i] - data.close[i - 1]).abs();
            true_ranges[i] = hl.max(hc).max(lc);
        }

        // Calculate average true range and TD Range
        for i in (self.period - 1)..n {
            let avg_tr: f64 = true_ranges[(i + 1 - self.period)..=i].iter().sum::<f64>()
                / self.period as f64;

            let current_tr = true_ranges[i];
            range[i] = if avg_tr > 0.0 {
                current_tr / avg_tr
            } else {
                1.0
            };

            // Detect expansion/contraction
            if range[i] > self.expansion_threshold {
                expansion[i] = 1.0;
            } else if range[i] < self.contraction_threshold {
                contraction[i] = 1.0;
            }

            // Calculate percentile within lookback
            let lookback_ranges: Vec<f64> = true_ranges[(i + 1 - self.period)..=i]
                .iter()
                .copied()
                .collect();
            let mut sorted = lookback_ranges.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let count_below = sorted.iter().filter(|&&v| v < current_tr).count();
            percentile[i] = (count_below as f64 / self.period as f64) * 100.0;
        }

        TDRangeOutput {
            range,
            expansion,
            contraction,
            percentile,
        }
    }
}

impl TechnicalIndicator for TDRange {
    fn name(&self) -> &str {
        "TD Range"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);
        Ok(IndicatorOutput::triple(
            result.range,
            result.expansion,
            result.percentile,
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

// ============================================================================
// TDChannel - TD Channel Indicator
// ============================================================================

/// TD Channel output containing channel bounds.
#[derive(Debug, Clone)]
pub struct TDChannelOutput {
    /// Upper channel boundary
    pub upper: Vec<f64>,
    /// Lower channel boundary
    pub lower: Vec<f64>,
    /// Channel midline
    pub midline: Vec<f64>,
    /// Channel width (percentage)
    pub width: Vec<f64>,
    /// Position within channel (0-1)
    pub position: Vec<f64>,
}

/// TD Channel indicator.
///
/// Constructs a price channel based on DeMark principles using
/// qualified highs and lows over a lookback period.
///
/// # Parameters
/// - `period`: Lookback period for channel (default: 20)
/// - `offset`: Channel offset multiplier (default: 0.0)
#[derive(Debug, Clone)]
pub struct TDChannel {
    period: usize,
    offset: f64,
}

impl TDChannel {
    /// Create a new TDChannel indicator with validation.
    pub fn new(period: usize, offset: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if offset < 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "offset".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        Ok(Self { period, offset })
    }

    /// Calculate TD Channel values from OHLCV data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDChannelOutput {
        let n = data.close.len();
        let mut upper = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];
        let mut midline = vec![f64::NAN; n];
        let mut width = vec![f64::NAN; n];
        let mut position = vec![f64::NAN; n];

        if n < self.period {
            return TDChannelOutput {
                upper,
                lower,
                midline,
                width,
                position,
            };
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;

            // Find qualified high and low (excluding current bar for DeMark style)
            let mut highest = f64::NEG_INFINITY;
            let mut lowest = f64::INFINITY;

            for j in start..=i {
                if data.high[j] > highest {
                    highest = data.high[j];
                }
                if data.low[j] < lowest {
                    lowest = data.low[j];
                }
            }

            // Apply offset
            let channel_range = highest - lowest;
            let offset_amount = channel_range * self.offset;

            upper[i] = highest + offset_amount;
            lower[i] = lowest - offset_amount;
            midline[i] = (upper[i] + lower[i]) / 2.0;

            // Calculate width as percentage of midline
            if midline[i] > 0.0 {
                width[i] = ((upper[i] - lower[i]) / midline[i]) * 100.0;
            }

            // Calculate position within channel
            let effective_range = upper[i] - lower[i];
            if effective_range > 0.0 {
                position[i] = (data.close[i] - lower[i]) / effective_range;
                position[i] = position[i].clamp(0.0, 1.0);
            }
        }

        TDChannelOutput {
            upper,
            lower,
            midline,
            width,
            position,
        }
    }
}

impl TechnicalIndicator for TDChannel {
    fn name(&self) -> &str {
        "TD Channel"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);
        Ok(IndicatorOutput::triple(
            result.upper,
            result.lower,
            result.position,
        ))
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

// ============================================================================
// TDQualifier - TD Setup Qualifier Indicator
// ============================================================================

/// TD Qualifier output containing qualification status.
#[derive(Debug, Clone)]
pub struct TDQualifierOutput {
    /// Qualification score (0-100)
    pub score: Vec<f64>,
    /// True if bar qualifies for buy setup
    pub buy_qualified: Vec<bool>,
    /// True if bar qualifies for sell setup
    pub sell_qualified: Vec<bool>,
    /// Momentum component of qualification
    pub momentum: Vec<f64>,
    /// Range component of qualification
    pub range_score: Vec<f64>,
}

/// TD Qualifier indicator for setup validation.
///
/// Evaluates the quality of TD Setup bars by analyzing momentum,
/// range, and price position characteristics.
///
/// # Parameters
/// - `lookback`: Comparison lookback period (default: 4)
/// - `momentum_weight`: Weight for momentum component (default: 0.5)
/// - `range_weight`: Weight for range component (default: 0.5)
#[derive(Debug, Clone)]
pub struct TDQualifier {
    lookback: usize,
    momentum_weight: f64,
    range_weight: f64,
}

impl TDQualifier {
    /// Create a new TDQualifier indicator with validation.
    pub fn new(lookback: usize, momentum_weight: f64, range_weight: f64) -> Result<Self> {
        if lookback < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if momentum_weight < 0.0 || momentum_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if range_weight < 0.0 || range_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "range_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self {
            lookback,
            momentum_weight,
            range_weight,
        })
    }

    /// Calculate TD Qualifier values from OHLCV data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDQualifierOutput {
        let n = data.close.len();
        let mut score = vec![f64::NAN; n];
        let mut buy_qualified = vec![false; n];
        let mut sell_qualified = vec![false; n];
        let mut momentum = vec![f64::NAN; n];
        let mut range_score = vec![f64::NAN; n];

        if n <= self.lookback {
            return TDQualifierOutput {
                score,
                buy_qualified,
                sell_qualified,
                momentum,
                range_score,
            };
        }

        for i in self.lookback..n {
            let current_close = data.close[i];
            let lookback_close = data.close[i - self.lookback];

            // Calculate momentum component
            let price_change = current_close - lookback_close;
            let mom = if lookback_close != 0.0 {
                (price_change / lookback_close) * 100.0
            } else {
                0.0
            };
            momentum[i] = mom;

            // Calculate range component
            let current_range = data.high[i] - data.low[i];
            let lookback_range = data.high[i - self.lookback] - data.low[i - self.lookback];
            let range_ratio = if lookback_range > 0.0 {
                current_range / lookback_range
            } else {
                1.0
            };

            // Normalize range score (0-100)
            let rs = (range_ratio.min(2.0) / 2.0) * 100.0;
            range_score[i] = rs;

            // Calculate overall qualification score
            let mom_normalized = (mom.abs().min(10.0) / 10.0) * 100.0;
            let s = self.momentum_weight * mom_normalized + self.range_weight * rs;
            score[i] = s;

            // Determine buy/sell qualification
            // Buy qualification: close < lookback close (bearish momentum)
            // Sell qualification: close > lookback close (bullish momentum)
            if current_close < lookback_close && s >= 50.0 {
                buy_qualified[i] = true;
            } else if current_close > lookback_close && s >= 50.0 {
                sell_qualified[i] = true;
            }
        }

        TDQualifierOutput {
            score,
            buy_qualified,
            sell_qualified,
            momentum,
            range_score,
        }
    }
}

impl TechnicalIndicator for TDQualifier {
    fn name(&self) -> &str {
        "TD Qualifier"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min = self.lookback + 1;
        if data.close.len() < min {
            return Err(IndicatorError::InsufficientData {
                required: min,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);
        let buy_f64: Vec<f64> = result.buy_qualified.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let sell_f64: Vec<f64> = result.sell_qualified.iter().map(|&b| if b { -1.0 } else { 0.0 }).collect();

        Ok(IndicatorOutput::triple(result.score, buy_f64, sell_f64))
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }
}

// ============================================================================
// TDAlignment - TD Component Alignment Indicator
// ============================================================================

/// TD Alignment output containing alignment metrics.
#[derive(Debug, Clone)]
pub struct TDAlignmentOutput {
    /// Overall alignment score (-100 to 100)
    pub alignment: Vec<f64>,
    /// Trend alignment component
    pub trend: Vec<f64>,
    /// Momentum alignment component
    pub momentum: Vec<f64>,
    /// Volatility alignment component
    pub volatility: Vec<f64>,
    /// True when all components align bullish
    pub bullish_aligned: Vec<bool>,
    /// True when all components align bearish
    pub bearish_aligned: Vec<bool>,
}

/// TD Alignment indicator.
///
/// Measures the alignment of multiple TD components (trend, momentum,
/// volatility) to identify high-probability trade setups.
///
/// # Parameters
/// - `short_period`: Short-term lookback (default: 5)
/// - `long_period`: Long-term lookback (default: 20)
/// - `alignment_threshold`: Threshold for alignment detection (default: 0.7)
#[derive(Debug, Clone)]
pub struct TDAlignment {
    short_period: usize,
    long_period: usize,
    alignment_threshold: f64,
}

impl TDAlignment {
    /// Create a new TDAlignment indicator with validation.
    pub fn new(short_period: usize, long_period: usize, alignment_threshold: f64) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if alignment_threshold <= 0.0 || alignment_threshold > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "alignment_threshold".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self {
            short_period,
            long_period,
            alignment_threshold,
        })
    }

    /// Calculate TD Alignment values from OHLCV data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDAlignmentOutput {
        let n = data.close.len();
        let mut alignment = vec![f64::NAN; n];
        let mut trend = vec![f64::NAN; n];
        let mut momentum = vec![f64::NAN; n];
        let mut volatility = vec![f64::NAN; n];
        let mut bullish_aligned = vec![false; n];
        let mut bearish_aligned = vec![false; n];

        if n < self.long_period {
            return TDAlignmentOutput {
                alignment,
                trend,
                momentum,
                volatility,
                bullish_aligned,
                bearish_aligned,
            };
        }

        for i in (self.long_period - 1)..n {
            // Calculate short-term SMA
            let short_start = i + 1 - self.short_period;
            let short_sum: f64 = data.close[short_start..=i].iter().sum();
            let short_sma = short_sum / self.short_period as f64;

            // Calculate long-term SMA
            let long_start = i + 1 - self.long_period;
            let long_sum: f64 = data.close[long_start..=i].iter().sum();
            let long_sma = long_sum / self.long_period as f64;

            // Trend component: short vs long SMA
            let trend_val = if long_sma > 0.0 {
                ((short_sma - long_sma) / long_sma) * 100.0
            } else {
                0.0
            };
            trend[i] = trend_val.clamp(-100.0, 100.0);

            // Momentum component: price vs short SMA
            let mom_val = if short_sma > 0.0 {
                ((data.close[i] - short_sma) / short_sma) * 100.0
            } else {
                0.0
            };
            momentum[i] = mom_val.clamp(-100.0, 100.0);

            // Volatility component: current range vs average range
            let current_range = data.high[i] - data.low[i];
            let avg_range: f64 = (long_start..=i)
                .map(|j| data.high[j] - data.low[j])
                .sum::<f64>()
                / self.long_period as f64;

            let vol_val = if avg_range > 0.0 {
                ((current_range / avg_range) - 1.0) * 50.0
            } else {
                0.0
            };
            volatility[i] = vol_val.clamp(-100.0, 100.0);

            // Overall alignment score
            alignment[i] = (trend[i] + momentum[i] + volatility[i]) / 3.0;

            // Detect aligned conditions
            let threshold_pct = self.alignment_threshold * 100.0;
            let all_positive = trend[i] > threshold_pct
                && momentum[i] > threshold_pct
                && volatility[i] > 0.0;
            let all_negative = trend[i] < -threshold_pct
                && momentum[i] < -threshold_pct
                && volatility[i] > 0.0;

            bullish_aligned[i] = all_positive;
            bearish_aligned[i] = all_negative;
        }

        TDAlignmentOutput {
            alignment,
            trend,
            momentum,
            volatility,
            bullish_aligned,
            bearish_aligned,
        }
    }
}

impl TechnicalIndicator for TDAlignment {
    fn name(&self) -> &str {
        "TD Alignment"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.long_period {
            return Err(IndicatorError::InsufficientData {
                required: self.long_period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);
        Ok(IndicatorOutput::triple(
            result.alignment,
            result.trend,
            result.momentum,
        ))
    }

    fn min_periods(&self) -> usize {
        self.long_period
    }
}

// ============================================================================
// TDExhaustion - Enhanced TD Exhaustion Signal Indicator
// ============================================================================

/// TD Exhaustion output containing exhaustion signals.
#[derive(Debug, Clone)]
pub struct TDExhaustionOutput {
    /// Exhaustion score (-100 to 100, positive = bullish exhaustion, negative = bearish)
    pub exhaustion: Vec<f64>,
    /// Buy exhaustion signal (potential bottom)
    pub buy_signal: Vec<bool>,
    /// Sell exhaustion signal (potential top)
    pub sell_signal: Vec<bool>,
    /// Momentum exhaustion component
    pub momentum_exhaustion: Vec<f64>,
    /// Price exhaustion component
    pub price_exhaustion: Vec<f64>,
    /// Confirmation strength (0-100)
    pub confirmation: Vec<f64>,
}

/// TD Exhaustion indicator for enhanced exhaustion signals.
///
/// Combines multiple exhaustion criteria to identify high-probability
/// reversal points. Uses momentum divergence, price extension, and
/// volume analysis.
///
/// # Parameters
/// - `lookback`: Lookback period for exhaustion detection (default: 9)
/// - `momentum_period`: Period for momentum calculation (default: 14)
/// - `threshold`: Exhaustion signal threshold (default: 70.0)
#[derive(Debug, Clone)]
pub struct TDExhaustion {
    lookback: usize,
    momentum_period: usize,
    threshold: f64,
}

impl TDExhaustion {
    /// Create a new TDExhaustion indicator with validation.
    pub fn new(lookback: usize, momentum_period: usize, threshold: f64) -> Result<Self> {
        if lookback < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if threshold <= 0.0 || threshold > 100.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0.0 and 100.0".to_string(),
            });
        }
        Ok(Self {
            lookback,
            momentum_period,
            threshold,
        })
    }

    /// Calculate TD Exhaustion values from OHLCV data.
    pub fn calculate(&self, data: &OHLCVSeries) -> TDExhaustionOutput {
        let n = data.close.len();
        let required = self.lookback.max(self.momentum_period);

        let mut exhaustion = vec![f64::NAN; n];
        let mut buy_signal = vec![false; n];
        let mut sell_signal = vec![false; n];
        let mut momentum_exhaustion = vec![f64::NAN; n];
        let mut price_exhaustion = vec![f64::NAN; n];
        let mut confirmation = vec![f64::NAN; n];

        if n < required {
            return TDExhaustionOutput {
                exhaustion,
                buy_signal,
                sell_signal,
                momentum_exhaustion,
                price_exhaustion,
                confirmation,
            };
        }

        // Calculate RSI-style momentum
        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let change = data.close[i] - data.close[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Calculate smoothed gains/losses
        let mut avg_gain = vec![0.0; n];
        let mut avg_loss = vec![0.0; n];

        if n >= self.momentum_period {
            // Initial average
            let initial_gain: f64 = gains[1..=self.momentum_period].iter().sum::<f64>()
                / self.momentum_period as f64;
            let initial_loss: f64 = losses[1..=self.momentum_period].iter().sum::<f64>()
                / self.momentum_period as f64;

            avg_gain[self.momentum_period] = initial_gain;
            avg_loss[self.momentum_period] = initial_loss;

            // Smooth subsequent values
            for i in (self.momentum_period + 1)..n {
                avg_gain[i] = (avg_gain[i - 1] * (self.momentum_period - 1) as f64 + gains[i])
                    / self.momentum_period as f64;
                avg_loss[i] = (avg_loss[i - 1] * (self.momentum_period - 1) as f64 + losses[i])
                    / self.momentum_period as f64;
            }
        }

        for i in required..n {
            // Momentum exhaustion (RSI-based)
            let rsi = if avg_loss[i] == 0.0 {
                100.0
            } else if avg_gain[i] == 0.0 {
                0.0
            } else {
                100.0 - (100.0 / (1.0 + avg_gain[i] / avg_loss[i]))
            };

            // Convert RSI to exhaustion scale
            momentum_exhaustion[i] = if rsi > 70.0 {
                ((rsi - 70.0) / 30.0) * 100.0  // Overbought (sell exhaustion)
            } else if rsi < 30.0 {
                -((30.0 - rsi) / 30.0) * 100.0  // Oversold (buy exhaustion)
            } else {
                0.0
            };

            // Price exhaustion: how far price is from recent range
            let start = i + 1 - self.lookback;
            let highest: f64 = data.high[start..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lowest: f64 = data.low[start..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);

            let range = highest - lowest;
            if range > 0.0 {
                let position = (data.close[i] - lowest) / range;
                // Convert to exhaustion: near highs = sell exhaustion, near lows = buy exhaustion
                price_exhaustion[i] = (position - 0.5) * 200.0; // -100 to 100
            }

            // Combined exhaustion score
            let exhaust = (momentum_exhaustion[i] + price_exhaustion[i]) / 2.0;
            exhaustion[i] = exhaust;

            // Confirmation based on consistency
            let consistent = (momentum_exhaustion[i].signum() == price_exhaustion[i].signum())
                || (momentum_exhaustion[i].abs() < 10.0 || price_exhaustion[i].abs() < 10.0);
            confirmation[i] = if consistent {
                exhaust.abs().min(100.0)
            } else {
                exhaust.abs().min(100.0) * 0.5
            };

            // Generate signals
            if exhaust < -self.threshold && confirmation[i] >= 50.0 {
                buy_signal[i] = true;
            } else if exhaust > self.threshold && confirmation[i] >= 50.0 {
                sell_signal[i] = true;
            }
        }

        TDExhaustionOutput {
            exhaustion,
            buy_signal,
            sell_signal,
            momentum_exhaustion,
            price_exhaustion,
            confirmation,
        }
    }
}

impl TechnicalIndicator for TDExhaustion {
    fn name(&self) -> &str {
        "TD Exhaustion"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min = self.lookback.max(self.momentum_period);
        if data.close.len() < min {
            return Err(IndicatorError::InsufficientData {
                required: min,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);
        let buy_f64: Vec<f64> = result.buy_signal.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let sell_f64: Vec<f64> = result.sell_signal.iter().map(|&b| if b { -1.0 } else { 0.0 }).collect();

        Ok(IndicatorOutput::triple(result.exhaustion, buy_f64, sell_f64))
    }

    fn min_periods(&self) -> usize {
        self.lookback.max(self.momentum_period)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(closes: Vec<f64>) -> OHLCVSeries {
        let n = closes.len();
        OHLCVSeries {
            open: closes.clone(),
            high: closes.iter().map(|c| c + 2.0).collect(),
            low: closes.iter().map(|c| c - 2.0).collect(),
            close: closes,
            volume: vec![1000.0; n],
        }
    }

    fn create_trending_data(direction: i32, bars: usize) -> OHLCVSeries {
        let mut closes = Vec::with_capacity(bars);
        let base = 100.0;

        for i in 0..bars {
            let price = if direction > 0 {
                base + (i as f64 * 1.5)
            } else {
                base - (i as f64 * 1.5)
            };
            closes.push(price);
        }

        OHLCVSeries {
            open: closes.clone(),
            high: closes.iter().map(|c| c + 2.0).collect(),
            low: closes.iter().map(|c| c - 2.0).collect(),
            close: closes,
            volume: vec![1000.0; bars],
        }
    }

    // ======================== TDLine Tests ========================

    #[test]
    fn test_td_line_new_valid() {
        let result = TDLine::new(5, 2);
        assert!(result.is_ok());
        let indicator = result.unwrap();
        assert_eq!(indicator.name(), "TD Line");
        assert_eq!(indicator.min_periods(), 11);
    }

    #[test]
    fn test_td_line_new_invalid_lookback() {
        let result = TDLine::new(1, 2);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "lookback");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_line_new_invalid_confirmation() {
        let result = TDLine::new(5, 0);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "confirmation_bars");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_line_calculate() {
        let data = create_trending_data(1, 30);
        let indicator = TDLine::new(3, 2).unwrap();
        let result = indicator.calculate(&data);

        assert_eq!(result.upper_line.len(), 30);
        assert_eq!(result.lower_line.len(), 30);
        assert_eq!(result.slope.len(), 30);
        assert_eq!(result.breakout.len(), 30);
    }

    #[test]
    fn test_td_line_compute() {
        let data = create_trending_data(1, 30);
        let indicator = TDLine::new(3, 2).unwrap();
        let output = indicator.compute(&data);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_td_line_insufficient_data() {
        let data = create_test_data(vec![100.0, 101.0, 102.0]);
        let indicator = TDLine::new(3, 2).unwrap();
        let result = indicator.compute(&data);

        assert!(result.is_err());
    }

    // ======================== TDRange Tests ========================

    #[test]
    fn test_td_range_new_valid() {
        let result = TDRange::new(14, 1.5, 0.5);
        assert!(result.is_ok());
        let indicator = result.unwrap();
        assert_eq!(indicator.name(), "TD Range");
        assert_eq!(indicator.min_periods(), 14);
    }

    #[test]
    fn test_td_range_new_invalid_period() {
        let result = TDRange::new(1, 1.5, 0.5);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "period");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_range_new_invalid_expansion() {
        let result = TDRange::new(14, 0.8, 0.5);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "expansion_threshold");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_range_new_invalid_contraction() {
        let result = TDRange::new(14, 1.5, 1.2);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "contraction_threshold");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_range_calculate() {
        let data = create_trending_data(1, 30);
        let indicator = TDRange::new(10, 1.5, 0.5).unwrap();
        let result = indicator.calculate(&data);

        assert_eq!(result.range.len(), 30);
        assert_eq!(result.expansion.len(), 30);
        assert_eq!(result.contraction.len(), 30);
        assert_eq!(result.percentile.len(), 30);

        // Check that percentile values are in valid range
        for &p in result.percentile.iter().skip(10) {
            if !p.is_nan() {
                assert!(p >= 0.0 && p <= 100.0);
            }
        }
    }

    #[test]
    fn test_td_range_compute() {
        let data = create_trending_data(1, 20);
        let indicator = TDRange::new(10, 1.5, 0.5).unwrap();
        let output = indicator.compute(&data);

        assert!(output.is_ok());
    }

    // ======================== TDChannel Tests ========================

    #[test]
    fn test_td_channel_new_valid() {
        let result = TDChannel::new(20, 0.1);
        assert!(result.is_ok());
        let indicator = result.unwrap();
        assert_eq!(indicator.name(), "TD Channel");
        assert_eq!(indicator.min_periods(), 20);
    }

    #[test]
    fn test_td_channel_new_invalid_period() {
        let result = TDChannel::new(1, 0.1);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "period");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_channel_new_invalid_offset() {
        let result = TDChannel::new(20, -0.1);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "offset");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_channel_calculate() {
        let data = create_trending_data(1, 30);
        let indicator = TDChannel::new(10, 0.0).unwrap();
        let result = indicator.calculate(&data);

        assert_eq!(result.upper.len(), 30);
        assert_eq!(result.lower.len(), 30);
        assert_eq!(result.midline.len(), 30);
        assert_eq!(result.width.len(), 30);
        assert_eq!(result.position.len(), 30);

        // Upper should be >= lower
        for i in 10..30 {
            if !result.upper[i].is_nan() && !result.lower[i].is_nan() {
                assert!(result.upper[i] >= result.lower[i]);
            }
        }

        // Position should be between 0 and 1
        for &p in result.position.iter().skip(10) {
            if !p.is_nan() {
                assert!(p >= 0.0 && p <= 1.0, "Position {} out of range", p);
            }
        }
    }

    #[test]
    fn test_td_channel_compute() {
        let data = create_trending_data(1, 25);
        let indicator = TDChannel::new(10, 0.1).unwrap();
        let output = indicator.compute(&data);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.primary.len(), 25);
    }

    // ======================== TDQualifier Tests ========================

    #[test]
    fn test_td_qualifier_new_valid() {
        let result = TDQualifier::new(4, 0.5, 0.5);
        assert!(result.is_ok());
        let indicator = result.unwrap();
        assert_eq!(indicator.name(), "TD Qualifier");
        assert_eq!(indicator.min_periods(), 5);
    }

    #[test]
    fn test_td_qualifier_new_invalid_lookback() {
        let result = TDQualifier::new(0, 0.5, 0.5);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "lookback");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_qualifier_new_invalid_momentum_weight() {
        let result = TDQualifier::new(4, 1.5, 0.5);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "momentum_weight");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_qualifier_new_invalid_range_weight() {
        let result = TDQualifier::new(4, 0.5, -0.1);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "range_weight");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_qualifier_calculate() {
        let data = create_trending_data(1, 20);
        let indicator = TDQualifier::new(4, 0.5, 0.5).unwrap();
        let result = indicator.calculate(&data);

        assert_eq!(result.score.len(), 20);
        assert_eq!(result.buy_qualified.len(), 20);
        assert_eq!(result.sell_qualified.len(), 20);
        assert_eq!(result.momentum.len(), 20);
        assert_eq!(result.range_score.len(), 20);
    }

    #[test]
    fn test_td_qualifier_buy_sell_detection() {
        // Uptrend should have sell qualified bars
        let uptrend = create_trending_data(1, 15);
        let indicator = TDQualifier::new(4, 0.5, 0.5).unwrap();
        let result = indicator.calculate(&uptrend);

        let sell_count = result.sell_qualified.iter().filter(|&&x| x).count();
        assert!(sell_count > 0, "Should have some sell qualified bars in uptrend");

        // Downtrend should have buy qualified bars
        let downtrend = create_trending_data(-1, 15);
        let result = indicator.calculate(&downtrend);

        let buy_count = result.buy_qualified.iter().filter(|&&x| x).count();
        assert!(buy_count > 0, "Should have some buy qualified bars in downtrend");
    }

    // ======================== TDAlignment Tests ========================

    #[test]
    fn test_td_alignment_new_valid() {
        let result = TDAlignment::new(5, 20, 0.7);
        assert!(result.is_ok());
        let indicator = result.unwrap();
        assert_eq!(indicator.name(), "TD Alignment");
        assert_eq!(indicator.min_periods(), 20);
    }

    #[test]
    fn test_td_alignment_new_invalid_short_period() {
        let result = TDAlignment::new(1, 20, 0.7);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "short_period");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_alignment_new_invalid_long_period() {
        let result = TDAlignment::new(10, 5, 0.7);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "long_period");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_alignment_new_invalid_threshold() {
        let result = TDAlignment::new(5, 20, 0.0);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "alignment_threshold");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_alignment_calculate() {
        let data = create_trending_data(1, 30);
        let indicator = TDAlignment::new(5, 15, 0.5).unwrap();
        let result = indicator.calculate(&data);

        assert_eq!(result.alignment.len(), 30);
        assert_eq!(result.trend.len(), 30);
        assert_eq!(result.momentum.len(), 30);
        assert_eq!(result.volatility.len(), 30);
        assert_eq!(result.bullish_aligned.len(), 30);
        assert_eq!(result.bearish_aligned.len(), 30);
    }

    #[test]
    fn test_td_alignment_compute() {
        let data = create_trending_data(1, 25);
        let indicator = TDAlignment::new(5, 15, 0.5).unwrap();
        let output = indicator.compute(&data);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.primary.len(), 25);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    // ======================== TDExhaustion Tests ========================

    #[test]
    fn test_td_exhaustion_new_valid() {
        let result = TDExhaustion::new(9, 14, 70.0);
        assert!(result.is_ok());
        let indicator = result.unwrap();
        assert_eq!(indicator.name(), "TD Exhaustion");
        assert_eq!(indicator.min_periods(), 14);
    }

    #[test]
    fn test_td_exhaustion_new_invalid_lookback() {
        let result = TDExhaustion::new(1, 14, 70.0);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "lookback");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_exhaustion_new_invalid_momentum_period() {
        let result = TDExhaustion::new(9, 1, 70.0);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "momentum_period");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_exhaustion_new_invalid_threshold() {
        let result = TDExhaustion::new(9, 14, 0.0);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::InvalidParameter { name, .. }) => {
                assert_eq!(name, "threshold");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_td_exhaustion_calculate() {
        let data = create_trending_data(1, 30);
        let indicator = TDExhaustion::new(9, 14, 70.0).unwrap();
        let result = indicator.calculate(&data);

        assert_eq!(result.exhaustion.len(), 30);
        assert_eq!(result.buy_signal.len(), 30);
        assert_eq!(result.sell_signal.len(), 30);
        assert_eq!(result.momentum_exhaustion.len(), 30);
        assert_eq!(result.price_exhaustion.len(), 30);
        assert_eq!(result.confirmation.len(), 30);
    }

    #[test]
    fn test_td_exhaustion_signals() {
        // Strong uptrend should eventually produce sell exhaustion signals
        let mut closes = vec![100.0];
        for i in 1..50 {
            closes.push(100.0 + (i as f64 * 2.0));
        }
        let data = create_test_data(closes);
        let indicator = TDExhaustion::new(9, 14, 50.0).unwrap();
        let result = indicator.calculate(&data);

        // Should have at least some positive exhaustion (sell signals) in strong uptrend
        let has_positive_exhaustion = result.exhaustion.iter()
            .any(|&e| !e.is_nan() && e > 30.0);
        assert!(has_positive_exhaustion, "Strong uptrend should show exhaustion");
    }

    #[test]
    fn test_td_exhaustion_compute() {
        let data = create_trending_data(1, 30);
        let indicator = TDExhaustion::new(9, 14, 70.0).unwrap();
        let output = indicator.compute(&data);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_td_exhaustion_insufficient_data() {
        let data = create_test_data(vec![100.0, 101.0, 102.0]);
        let indicator = TDExhaustion::new(9, 14, 70.0).unwrap();
        let result = indicator.compute(&data);

        assert!(result.is_err());
    }

    // ======================== Cross-Indicator Tests ========================

    #[test]
    fn test_all_indicators_empty_data() {
        let empty_data = OHLCVSeries {
            open: vec![],
            high: vec![],
            low: vec![],
            close: vec![],
            volume: vec![],
        };

        let td_line = TDLine::new(3, 2).unwrap();
        let td_range = TDRange::new(10, 1.5, 0.5).unwrap();
        let td_channel = TDChannel::new(10, 0.0).unwrap();
        let td_qualifier = TDQualifier::new(4, 0.5, 0.5).unwrap();
        let td_alignment = TDAlignment::new(5, 15, 0.5).unwrap();
        let td_exhaustion = TDExhaustion::new(9, 14, 70.0).unwrap();

        assert!(td_line.compute(&empty_data).is_err());
        assert!(td_range.compute(&empty_data).is_err());
        assert!(td_channel.compute(&empty_data).is_err());
        assert!(td_qualifier.compute(&empty_data).is_err());
        assert!(td_alignment.compute(&empty_data).is_err());
        assert!(td_exhaustion.compute(&empty_data).is_err());
    }

    #[test]
    fn test_all_indicators_names() {
        let td_line = TDLine::new(3, 2).unwrap();
        let td_range = TDRange::new(10, 1.5, 0.5).unwrap();
        let td_channel = TDChannel::new(10, 0.0).unwrap();
        let td_qualifier = TDQualifier::new(4, 0.5, 0.5).unwrap();
        let td_alignment = TDAlignment::new(5, 15, 0.5).unwrap();
        let td_exhaustion = TDExhaustion::new(9, 14, 70.0).unwrap();

        assert_eq!(td_line.name(), "TD Line");
        assert_eq!(td_range.name(), "TD Range");
        assert_eq!(td_channel.name(), "TD Channel");
        assert_eq!(td_qualifier.name(), "TD Qualifier");
        assert_eq!(td_alignment.name(), "TD Alignment");
        assert_eq!(td_exhaustion.name(), "TD Exhaustion");
    }
}
