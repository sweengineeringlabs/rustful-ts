//! Final Advanced Swing Indicators
//!
//! Comprehensive set of advanced swing trading indicators for projection,
//! confirmation, range analysis, breakout detection, momentum flow, and trend analysis.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Swing Projection - Projects future swing targets
///
/// Analyzes historical swing patterns to project potential future price targets
/// based on measured moves, Fibonacci projections, and swing amplitude analysis.
///
/// Output:
/// - Primary: Upper projection target
/// - Secondary: Lower projection target
#[derive(Debug, Clone)]
pub struct SwingProjection {
    period: usize,
    projection_ratio: f64,
}

impl SwingProjection {
    /// Create a new Swing Projection indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for swing analysis (minimum 5)
    /// * `projection_ratio` - Ratio for target projection (e.g., 1.0 = 100%, 1.618 = Fibonacci)
    pub fn new(period: usize, projection_ratio: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if projection_ratio <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "projection_ratio".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            period,
            projection_ratio,
        })
    }

    /// Calculate swing projection targets
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut upper_projection = vec![f64::NAN; n];
        let mut lower_projection = vec![f64::NAN; n];

        if n < self.period * 2 {
            return (upper_projection, lower_projection);
        }

        // Identify swing points
        let swing_lookback = self.period / 2;
        let swing_lookback = swing_lookback.max(2);

        for i in (self.period * 2)..n {
            // Collect recent swing highs and lows
            let mut swing_highs: Vec<f64> = Vec::new();
            let mut swing_lows: Vec<f64> = Vec::new();

            for j in (swing_lookback..(i.saturating_sub(swing_lookback))).rev() {
                // Check if j is a swing high
                let mut is_high = true;
                for k in 1..=swing_lookback {
                    if j >= k && j + k < n {
                        if high[j] <= high[j - k] || high[j] <= high[j + k] {
                            is_high = false;
                            break;
                        }
                    } else {
                        is_high = false;
                        break;
                    }
                }
                if is_high && swing_highs.len() < 3 {
                    swing_highs.push(high[j]);
                }

                // Check if j is a swing low
                let mut is_low = true;
                for k in 1..=swing_lookback {
                    if j >= k && j + k < n {
                        if low[j] >= low[j - k] || low[j] >= low[j + k] {
                            is_low = false;
                            break;
                        }
                    } else {
                        is_low = false;
                        break;
                    }
                }
                if is_low && swing_lows.len() < 3 {
                    swing_lows.push(low[j]);
                }

                if swing_highs.len() >= 3 && swing_lows.len() >= 3 {
                    break;
                }
            }

            // Calculate average swing amplitude
            let avg_swing_high = if !swing_highs.is_empty() {
                swing_highs.iter().sum::<f64>() / swing_highs.len() as f64
            } else {
                high[i]
            };

            let avg_swing_low = if !swing_lows.is_empty() {
                swing_lows.iter().sum::<f64>() / swing_lows.len() as f64
            } else {
                low[i]
            };

            let avg_swing_range = avg_swing_high - avg_swing_low;

            // Recent range for context
            let start = i.saturating_sub(self.period);
            let recent_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let recent_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);

            // Project targets
            upper_projection[i] = recent_high + avg_swing_range * self.projection_ratio;
            lower_projection[i] = recent_low - avg_swing_range * self.projection_ratio;
        }

        (upper_projection, lower_projection)
    }
}

impl TechnicalIndicator for SwingProjection {
    fn name(&self) -> &str {
        "Swing Projection"
    }

    fn min_periods(&self) -> usize {
        self.period * 2 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(upper, lower))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Confirmation - Confirms swing high/low formations
///
/// Validates swing point formations by analyzing price action confirmation,
/// volume patterns, and follow-through behavior.
///
/// Output:
/// - Primary: Confirmation signal (1 = bullish swing confirmed, -1 = bearish swing confirmed, 0 = none)
/// - Secondary: Confirmation strength (0 to 100)
#[derive(Debug, Clone)]
pub struct SwingConfirmation {
    period: usize,
    confirmation_bars: usize,
}

impl SwingConfirmation {
    /// Create a new Swing Confirmation indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for swing detection (minimum 3)
    /// * `confirmation_bars` - Bars required for confirmation (minimum 2)
    pub fn new(period: usize, confirmation_bars: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if confirmation_bars < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "confirmation_bars".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            period,
            confirmation_bars,
        })
    }

    /// Calculate swing confirmation signals
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut signal = vec![0.0; n];
        let mut strength = vec![0.0; n];

        let total_lookback = self.period + self.confirmation_bars;
        if n <= total_lookback {
            return (signal, strength);
        }

        for i in total_lookback..n {
            let swing_idx = i - self.confirmation_bars;

            // Check for swing high at swing_idx
            let mut is_swing_high = true;
            for j in 1..=self.period {
                if swing_idx >= j {
                    if high[swing_idx] <= high[swing_idx - j] {
                        is_swing_high = false;
                        break;
                    }
                } else {
                    is_swing_high = false;
                    break;
                }
            }
            // Check bars after swing point
            if is_swing_high {
                for j in 1..=self.confirmation_bars {
                    if swing_idx + j < n {
                        if high[swing_idx + j] >= high[swing_idx] {
                            is_swing_high = false;
                            break;
                        }
                    }
                }
            }

            // Check for swing low at swing_idx
            let mut is_swing_low = true;
            for j in 1..=self.period {
                if swing_idx >= j {
                    if low[swing_idx] >= low[swing_idx - j] {
                        is_swing_low = false;
                        break;
                    }
                } else {
                    is_swing_low = false;
                    break;
                }
            }
            // Check bars after swing point
            if is_swing_low {
                for j in 1..=self.confirmation_bars {
                    if swing_idx + j < n {
                        if low[swing_idx + j] <= low[swing_idx] {
                            is_swing_low = false;
                            break;
                        }
                    }
                }
            }

            if is_swing_high {
                // Bearish swing high confirmed - price moving lower
                let price_decline = high[swing_idx] - close[i];
                let range = high[swing_idx] - low[swing_idx];

                // Volume confirmation: higher volume on decline
                let swing_volume = volume[swing_idx];
                let avg_confirm_volume: f64 = volume[(swing_idx + 1)..=i].iter().sum::<f64>()
                    / self.confirmation_bars as f64;
                let volume_factor = if swing_volume > 1e-10 {
                    (avg_confirm_volume / swing_volume).clamp(0.5, 2.0)
                } else {
                    1.0
                };

                // Calculate strength
                let price_strength = if range > 1e-10 {
                    (price_decline / range).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                let confirm_strength = (price_strength * 50.0 + volume_factor * 25.0 + 25.0)
                    .clamp(0.0, 100.0);

                if confirm_strength > 30.0 {
                    signal[i] = -1.0;
                    strength[i] = confirm_strength;
                }
            } else if is_swing_low {
                // Bullish swing low confirmed - price moving higher
                let price_advance = close[i] - low[swing_idx];
                let range = high[swing_idx] - low[swing_idx];

                // Volume confirmation: higher volume on advance
                let swing_volume = volume[swing_idx];
                let avg_confirm_volume: f64 = volume[(swing_idx + 1)..=i].iter().sum::<f64>()
                    / self.confirmation_bars as f64;
                let volume_factor = if swing_volume > 1e-10 {
                    (avg_confirm_volume / swing_volume).clamp(0.5, 2.0)
                } else {
                    1.0
                };

                // Calculate strength
                let price_strength = if range > 1e-10 {
                    (price_advance / range).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                let confirm_strength = (price_strength * 50.0 + volume_factor * 25.0 + 25.0)
                    .clamp(0.0, 100.0);

                if confirm_strength > 30.0 {
                    signal[i] = 1.0;
                    strength[i] = confirm_strength;
                }
            }
        }

        (signal, strength)
    }
}

impl TechnicalIndicator for SwingConfirmation {
    fn name(&self) -> &str {
        "Swing Confirmation"
    }

    fn min_periods(&self) -> usize {
        self.period + self.confirmation_bars + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (signal, strength) =
            self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::dual(signal, strength))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Range Analysis - Analyzes swing trading ranges
///
/// Comprehensive analysis of price ranges within swing movements,
/// identifying range expansion, contraction, and optimal trading zones.
///
/// Output:
/// - Primary: Range width as percentage of price
/// - Secondary: Range position (0 = at low, 100 = at high)
#[derive(Debug, Clone)]
pub struct SwingRangeAnalysis {
    period: usize,
    smoothing: usize,
}

impl SwingRangeAnalysis {
    /// Create a new Swing Range Analysis indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for range analysis (minimum 5)
    /// * `smoothing` - Smoothing period for output (minimum 1)
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smoothing < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Calculate swing range analysis metrics
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut range_width = vec![0.0; n];
        let mut range_position = vec![50.0; n];

        let mut raw_width = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate swing range
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            // Average price for normalization
            let avg_price = close[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;

            if avg_price > 1e-10 {
                raw_width[i] = range / avg_price * 100.0;
            }

            // Range position
            if range > 1e-10 {
                range_position[i] = ((close[i] - period_low) / range * 100.0).clamp(0.0, 100.0);
            }
        }

        // Apply smoothing to range width
        for i in (self.period + self.smoothing - 1)..n {
            let start = i - self.smoothing + 1;
            let sum: f64 = raw_width[start..=i].iter().sum();
            range_width[i] = sum / self.smoothing as f64;
        }

        (range_width, range_position)
    }
}

impl TechnicalIndicator for SwingRangeAnalysis {
    fn name(&self) -> &str {
        "Swing Range Analysis"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (width, position) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(width, position))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Breakout - Detects swing breakouts
///
/// Identifies breakout signals when price moves beyond established swing levels,
/// with confirmation based on momentum, volume, and follow-through.
///
/// Output:
/// - Primary: Breakout signal (1 = bullish breakout, -1 = bearish breakout, 0 = none)
/// - Secondary: Breakout strength (0 to 100)
#[derive(Debug, Clone)]
pub struct SwingBreakout {
    period: usize,
    threshold: f64,
}

impl SwingBreakout {
    /// Create a new Swing Breakout indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for swing level detection (minimum 5)
    /// * `threshold` - Minimum breakout threshold as percentage (0.0 to 10.0)
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if !(0.0..=10.0).contains(&threshold) {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0.0 and 10.0".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate swing breakout signals
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut signal = vec![0.0; n];
        let mut strength = vec![0.0; n];

        if n <= self.period {
            return (signal, strength);
        }

        for i in self.period..n {
            let start = i - self.period;

            // Calculate prior swing range (excluding current bar)
            let prior_high = high[start..(i)].iter().cloned().fold(f64::MIN, f64::max);
            let prior_low = low[start..(i)].iter().cloned().fold(f64::MAX, f64::min);
            let prior_range = prior_high - prior_low;

            if prior_range < 1e-10 {
                continue;
            }

            // Average price and volume for normalization
            let avg_price = close[start..i].iter().sum::<f64>() / self.period as f64;
            let avg_volume = volume[start..i].iter().sum::<f64>() / self.period as f64;

            let threshold_amount = avg_price * self.threshold / 100.0;

            // Check for bullish breakout
            if close[i] > prior_high + threshold_amount {
                let breakout_amount = close[i] - prior_high;
                let breakout_pct = breakout_amount / prior_range * 100.0;

                // Volume confirmation
                let volume_ratio = if avg_volume > 1e-10 {
                    (volume[i] / avg_volume).clamp(0.5, 3.0)
                } else {
                    1.0
                };

                // Bar strength (close near high)
                let bar_range = high[i] - low[i];
                let bar_strength = if bar_range > 1e-10 {
                    (close[i] - low[i]) / bar_range
                } else {
                    0.5
                };

                let breakout_strength =
                    (breakout_pct * 30.0 + volume_ratio * 35.0 + bar_strength * 35.0)
                        .clamp(0.0, 100.0);

                signal[i] = 1.0;
                strength[i] = breakout_strength;
            }
            // Check for bearish breakout
            else if close[i] < prior_low - threshold_amount {
                let breakout_amount = prior_low - close[i];
                let breakout_pct = breakout_amount / prior_range * 100.0;

                // Volume confirmation
                let volume_ratio = if avg_volume > 1e-10 {
                    (volume[i] / avg_volume).clamp(0.5, 3.0)
                } else {
                    1.0
                };

                // Bar strength (close near low)
                let bar_range = high[i] - low[i];
                let bar_strength = if bar_range > 1e-10 {
                    (high[i] - close[i]) / bar_range
                } else {
                    0.5
                };

                let breakout_strength =
                    (breakout_pct * 30.0 + volume_ratio * 35.0 + bar_strength * 35.0)
                        .clamp(0.0, 100.0);

                signal[i] = -1.0;
                strength[i] = breakout_strength;
            }
        }

        (signal, strength)
    }
}

impl TechnicalIndicator for SwingBreakout {
    fn name(&self) -> &str {
        "Swing Breakout"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (signal, strength) =
            self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::dual(signal, strength))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Momentum Flow - Tracks momentum flow through swings
///
/// Measures the flow of momentum through successive swing movements,
/// identifying momentum accumulation, distribution, and divergences.
///
/// Output:
/// - Primary: Momentum flow value (positive = bullish flow, negative = bearish flow)
/// - Secondary: Flow rate of change
#[derive(Debug, Clone)]
pub struct SwingMomentumFlow {
    period: usize,
    flow_period: usize,
}

impl SwingMomentumFlow {
    /// Create a new Swing Momentum Flow indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for swing analysis (minimum 5)
    /// * `flow_period` - Period for flow rate calculation (minimum 2)
    pub fn new(period: usize, flow_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if flow_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "flow_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, flow_period })
    }

    /// Calculate swing momentum flow
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut flow = vec![0.0; n];
        let mut flow_roc = vec![0.0; n];

        if n <= self.period {
            return (flow, flow_roc);
        }

        for i in self.period..n {
            let start = i - self.period;

            // Calculate range and position
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            if range < 1e-10 {
                continue;
            }

            // Momentum components
            let mut bullish_momentum = 0.0;
            let mut bearish_momentum = 0.0;

            for j in (start + 1)..=i {
                let price_change = close[j] - close[j - 1];
                let typical_volume = volume[j];

                if price_change > 0.0 {
                    bullish_momentum += price_change * typical_volume;
                } else if price_change < 0.0 {
                    bearish_momentum += price_change.abs() * typical_volume;
                }
            }

            // Normalize by total volume
            let total_volume: f64 = volume[start..=i].iter().sum();
            let total_momentum = bullish_momentum + bearish_momentum;

            if total_momentum > 1e-10 && total_volume > 1e-10 {
                // Net momentum flow
                let net_flow = bullish_momentum - bearish_momentum;
                // Normalize to -100 to 100 range
                flow[i] = (net_flow / total_momentum * 100.0).clamp(-100.0, 100.0);
            }
        }

        // Calculate flow rate of change
        for i in (self.period + self.flow_period)..n {
            let flow_change = flow[i] - flow[i - self.flow_period];
            flow_roc[i] = flow_change;
        }

        (flow, flow_roc)
    }
}

impl TechnicalIndicator for SwingMomentumFlow {
    fn name(&self) -> &str {
        "Swing Momentum Flow"
    }

    fn min_periods(&self) -> usize {
        self.period + self.flow_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (flow, roc) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::dual(flow, roc))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Swing Trend Analysis - Analyzes overall swing trend
///
/// Comprehensive trend analysis based on swing structure, including trend direction,
/// strength, maturity, and potential exhaustion signals.
///
/// Output:
/// - Primary: Trend score (-100 to 100, positive = bullish, negative = bearish)
/// - Secondary: Trend maturity (0 = early trend, 100 = mature/exhausted trend)
#[derive(Debug, Clone)]
pub struct SwingTrendAnalysis {
    short_period: usize,
    long_period: usize,
}

impl SwingTrendAnalysis {
    /// Create a new Swing Trend Analysis indicator.
    ///
    /// # Arguments
    /// * `short_period` - Short-term swing period (minimum 5)
    /// * `long_period` - Long-term swing period (minimum 10, must be > short_period)
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if long_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self {
            short_period,
            long_period,
        })
    }

    /// Calculate swing trend analysis
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut trend_score = vec![0.0; n];
        let mut maturity = vec![0.0; n];

        if n <= self.long_period {
            return (trend_score, maturity);
        }

        for i in self.long_period..n {
            // Short-term analysis
            let short_start = i - self.short_period;
            let short_high = high[short_start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let short_low = low[short_start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let short_range = short_high - short_low;
            let short_trend = close[i] - close[short_start];

            // Long-term analysis
            let long_start = i - self.long_period;
            let long_high = high[long_start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let long_low = low[long_start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let long_range = long_high - long_low;
            let long_trend = close[i] - close[long_start];

            if short_range < 1e-10 || long_range < 1e-10 {
                continue;
            }

            // Trend direction and strength
            let short_trend_normalized = short_trend / short_range * 50.0;
            let long_trend_normalized = long_trend / long_range * 50.0;

            // Count higher highs and higher lows (bullish) or lower highs and lower lows (bearish)
            let mut higher_highs = 0;
            let mut higher_lows = 0;
            let mut lower_highs = 0;
            let mut lower_lows = 0;

            let check_period = self.short_period.min(10);
            for j in 1..check_period {
                if i >= j + 1 {
                    if high[i - j] > high[i - j - 1] {
                        higher_highs += 1;
                    } else {
                        lower_highs += 1;
                    }
                    if low[i - j] > low[i - j - 1] {
                        higher_lows += 1;
                    } else {
                        lower_lows += 1;
                    }
                }
            }

            let structure_score = ((higher_highs + higher_lows) as i32
                - (lower_highs + lower_lows) as i32) as f64
                / (check_period as f64 * 2.0)
                * 100.0;

            // Combine components
            let raw_score = short_trend_normalized * 0.4
                + long_trend_normalized * 0.3
                + structure_score * 0.3;
            trend_score[i] = raw_score.clamp(-100.0, 100.0);

            // Maturity analysis
            // Check position within long-term range
            let position_in_range = (close[i] - long_low) / long_range;

            // Trend extension: how far has price moved from midpoint
            let midpoint = (long_high + long_low) / 2.0;
            let extension = ((close[i] - midpoint).abs() / (long_range / 2.0)).clamp(0.0, 1.0);

            // Check for momentum deceleration
            let mid_point_idx = long_start + self.long_period / 2;
            let first_half_move = close[mid_point_idx] - close[long_start];
            let second_half_move = close[i] - close[mid_point_idx];

            let deceleration = if first_half_move.abs() > 1e-10 {
                let ratio = second_half_move.abs() / first_half_move.abs();
                if (first_half_move > 0.0 && second_half_move > 0.0)
                    || (first_half_move < 0.0 && second_half_move < 0.0)
                {
                    // Same direction - check if slowing
                    if ratio < 1.0 {
                        (1.0 - ratio) * 50.0
                    } else {
                        0.0
                    }
                } else {
                    // Direction change - high maturity
                    50.0
                }
            } else {
                0.0
            };

            maturity[i] = (extension * 50.0 + deceleration).clamp(0.0, 100.0);
        }

        (trend_score, maturity)
    }
}

impl TechnicalIndicator for SwingTrendAnalysis {
    fn name(&self) -> &str {
        "Swing Trend Analysis"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (trend, maturity) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(trend, maturity))
    }

    fn output_features(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create data with clear swing patterns
        let high = vec![
            102.0, 104.0, 106.0, 105.0, 103.0, 104.0, 108.0, 110.0, 109.0, 107.0, 108.0, 112.0,
            114.0, 113.0, 111.0, 112.0, 116.0, 118.0, 117.0, 115.0, 116.0, 120.0, 122.0, 121.0,
            119.0, 120.0, 124.0, 126.0, 125.0, 123.0, 124.0, 128.0, 130.0, 129.0, 127.0, 128.0,
            132.0, 134.0, 133.0, 131.0,
        ];
        let low = vec![
            98.0, 100.0, 102.0, 101.0, 99.0, 100.0, 104.0, 106.0, 105.0, 103.0, 104.0, 108.0,
            110.0, 109.0, 107.0, 108.0, 112.0, 114.0, 113.0, 111.0, 112.0, 116.0, 118.0, 117.0,
            115.0, 116.0, 120.0, 122.0, 121.0, 119.0, 120.0, 124.0, 126.0, 125.0, 123.0, 124.0,
            128.0, 130.0, 129.0, 127.0,
        ];
        let close = vec![
            100.0, 102.0, 104.0, 103.0, 101.0, 102.0, 106.0, 108.0, 107.0, 105.0, 106.0, 110.0,
            112.0, 111.0, 109.0, 110.0, 114.0, 116.0, 115.0, 113.0, 114.0, 118.0, 120.0, 119.0,
            117.0, 118.0, 122.0, 124.0, 123.0, 121.0, 122.0, 126.0, 128.0, 127.0, 125.0, 126.0,
            130.0, 132.0, 131.0, 129.0,
        ];
        let volume = vec![
            1000.0, 1100.0, 1200.0, 1100.0, 1000.0, 1050.0, 1300.0, 1400.0, 1300.0, 1100.0, 1150.0,
            1500.0, 1600.0, 1500.0, 1200.0, 1250.0, 1700.0, 1800.0, 1700.0, 1400.0, 1450.0, 1900.0,
            2000.0, 1900.0, 1600.0, 1650.0, 2100.0, 2200.0, 2100.0, 1800.0, 1850.0, 2300.0, 2400.0,
            2300.0, 2000.0, 2050.0, 2500.0, 2600.0, 2500.0, 2200.0,
        ];
        (high, low, close, volume)
    }

    // SwingProjection tests
    #[test]
    fn test_swing_projection() {
        let (high, low, close, _) = make_test_data();
        let sp = SwingProjection::new(5, 1.0).unwrap();
        let (upper, lower) = sp.calculate(&high, &low, &close);

        assert_eq!(upper.len(), close.len());
        assert_eq!(lower.len(), close.len());

        // Check that projections are valid after warmup
        for i in 11..close.len() {
            if !upper[i].is_nan() && !lower[i].is_nan() {
                assert!(
                    upper[i] > lower[i],
                    "Upper {} should be > lower {} at {}",
                    upper[i],
                    lower[i],
                    i
                );
            }
        }
    }

    #[test]
    fn test_swing_projection_validation() {
        assert!(SwingProjection::new(4, 1.0).is_err());
        assert!(SwingProjection::new(5, 0.0).is_err());
        assert!(SwingProjection::new(5, -1.0).is_err());
        assert!(SwingProjection::new(5, 1.618).is_ok());
    }

    #[test]
    fn test_swing_projection_fibonacci() {
        let (high, low, close, _) = make_test_data();
        let sp = SwingProjection::new(5, 1.618).unwrap();
        let (upper, lower) = sp.calculate(&high, &low, &close);

        assert_eq!(upper.len(), close.len());
        // Fibonacci projection should extend further
        for i in 11..close.len() {
            if !upper[i].is_nan() {
                assert!(upper[i] > high[i], "Fibonacci upper should project beyond current high");
            }
        }
    }

    // SwingConfirmation tests
    #[test]
    fn test_swing_confirmation() {
        let (high, low, close, volume) = make_test_data();
        let sc = SwingConfirmation::new(3, 2).unwrap();
        let (signal, strength) = sc.calculate(&high, &low, &close, &volume);

        assert_eq!(signal.len(), close.len());
        assert_eq!(strength.len(), close.len());

        // Signals should be -1, 0, or 1
        for s in &signal {
            assert!(
                *s == -1.0 || *s == 0.0 || *s == 1.0,
                "Signal should be -1, 0, or 1, got {}",
                s
            );
        }

        // Strength should be 0-100
        for s in &strength {
            assert!(
                *s >= 0.0 && *s <= 100.0,
                "Strength should be 0-100, got {}",
                s
            );
        }
    }

    #[test]
    fn test_swing_confirmation_validation() {
        assert!(SwingConfirmation::new(2, 2).is_err());
        assert!(SwingConfirmation::new(3, 1).is_err());
        assert!(SwingConfirmation::new(3, 2).is_ok());
        assert!(SwingConfirmation::new(5, 3).is_ok());
    }

    // SwingRangeAnalysis tests
    #[test]
    fn test_swing_range_analysis() {
        let (high, low, close, _) = make_test_data();
        let sra = SwingRangeAnalysis::new(10, 3).unwrap();
        let (width, position) = sra.calculate(&high, &low, &close);

        assert_eq!(width.len(), close.len());
        assert_eq!(position.len(), close.len());

        // Range width should be non-negative
        for i in 13..width.len() {
            assert!(width[i] >= 0.0, "Width at {} was {}", i, width[i]);
        }

        // Position should be 0-100
        for i in 10..position.len() {
            assert!(
                position[i] >= 0.0 && position[i] <= 100.0,
                "Position at {} was {}",
                i,
                position[i]
            );
        }
    }

    #[test]
    fn test_swing_range_analysis_validation() {
        assert!(SwingRangeAnalysis::new(4, 1).is_err());
        assert!(SwingRangeAnalysis::new(5, 0).is_err());
        assert!(SwingRangeAnalysis::new(5, 1).is_ok());
        assert!(SwingRangeAnalysis::new(10, 3).is_ok());
    }

    // SwingBreakout tests
    #[test]
    fn test_swing_breakout() {
        let (high, low, close, volume) = make_test_data();
        let sb = SwingBreakout::new(5, 0.5).unwrap();
        let (signal, strength) = sb.calculate(&high, &low, &close, &volume);

        assert_eq!(signal.len(), close.len());
        assert_eq!(strength.len(), close.len());

        // Signals should be -1, 0, or 1
        for s in &signal {
            assert!(
                *s == -1.0 || *s == 0.0 || *s == 1.0,
                "Signal should be -1, 0, or 1"
            );
        }

        // Strength should be 0-100
        for s in &strength {
            assert!(*s >= 0.0 && *s <= 100.0, "Strength should be 0-100");
        }
    }

    #[test]
    fn test_swing_breakout_validation() {
        assert!(SwingBreakout::new(4, 0.5).is_err());
        assert!(SwingBreakout::new(5, -0.1).is_err());
        assert!(SwingBreakout::new(5, 10.1).is_err());
        assert!(SwingBreakout::new(5, 0.0).is_ok());
        assert!(SwingBreakout::new(5, 10.0).is_ok());
    }

    #[test]
    fn test_swing_breakout_detection() {
        // Create data with a clear breakout
        let high = vec![
            100.0, 101.0, 102.0, 101.5, 101.0, 100.5, 100.0, 100.5, 101.0, 110.0,
        ];
        let low = vec![
            98.0, 99.0, 100.0, 99.5, 99.0, 98.5, 98.0, 98.5, 99.0, 105.0,
        ];
        let close = vec![
            99.0, 100.0, 101.0, 100.5, 100.0, 99.5, 99.0, 99.5, 100.0, 108.0,
        ];
        let volume = vec![
            1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 2000.0,
        ];

        let sb = SwingBreakout::new(5, 0.5).unwrap();
        let (signal, strength) = sb.calculate(&high, &low, &close, &volume);

        // Should detect bullish breakout at bar 9
        assert_eq!(signal[9], 1.0, "Should detect bullish breakout");
        assert!(strength[9] > 0.0, "Breakout strength should be positive");
    }

    // SwingMomentumFlow tests
    #[test]
    fn test_swing_momentum_flow() {
        let (high, low, close, volume) = make_test_data();
        let smf = SwingMomentumFlow::new(5, 3).unwrap();
        let (flow, roc) = smf.calculate(&high, &low, &close, &volume);

        assert_eq!(flow.len(), close.len());
        assert_eq!(roc.len(), close.len());

        // Flow should be in -100 to 100 range
        for i in 5..flow.len() {
            assert!(
                flow[i] >= -100.0 && flow[i] <= 100.0,
                "Flow at {} was {}",
                i,
                flow[i]
            );
        }
    }

    #[test]
    fn test_swing_momentum_flow_validation() {
        assert!(SwingMomentumFlow::new(4, 2).is_err());
        assert!(SwingMomentumFlow::new(5, 1).is_err());
        assert!(SwingMomentumFlow::new(5, 2).is_ok());
    }

    #[test]
    fn test_swing_momentum_flow_bullish() {
        // Create clearly bullish data
        let high: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0 + 2.0).collect();
        let low: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0 - 2.0).collect();
        let close: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume = vec![1000.0; 20];

        let smf = SwingMomentumFlow::new(5, 2).unwrap();
        let (flow, _) = smf.calculate(&high, &low, &close, &volume);

        // Flow should be positive in uptrend
        for i in 10..flow.len() {
            assert!(flow[i] > 0.0, "Flow should be positive in uptrend at {}", i);
        }
    }

    // SwingTrendAnalysis tests
    #[test]
    fn test_swing_trend_analysis() {
        let (high, low, close, _) = make_test_data();
        let sta = SwingTrendAnalysis::new(5, 15).unwrap();
        let (trend, maturity) = sta.calculate(&high, &low, &close);

        assert_eq!(trend.len(), close.len());
        assert_eq!(maturity.len(), close.len());

        // Trend should be in -100 to 100 range
        for i in 15..trend.len() {
            assert!(
                trend[i] >= -100.0 && trend[i] <= 100.0,
                "Trend at {} was {}",
                i,
                trend[i]
            );
        }

        // Maturity should be 0-100
        for i in 15..maturity.len() {
            assert!(
                maturity[i] >= 0.0 && maturity[i] <= 100.0,
                "Maturity at {} was {}",
                i,
                maturity[i]
            );
        }
    }

    #[test]
    fn test_swing_trend_analysis_validation() {
        assert!(SwingTrendAnalysis::new(4, 15).is_err());
        assert!(SwingTrendAnalysis::new(5, 9).is_err());
        assert!(SwingTrendAnalysis::new(5, 5).is_err()); // long must be > short
        assert!(SwingTrendAnalysis::new(5, 10).is_ok());
        assert!(SwingTrendAnalysis::new(5, 20).is_ok());
    }

    #[test]
    fn test_swing_trend_analysis_uptrend() {
        // Create clearly bullish data
        let high: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 1.5 + 2.0).collect();
        let low: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 1.5 - 2.0).collect();
        let close: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 1.5).collect();

        let sta = SwingTrendAnalysis::new(5, 15).unwrap();
        let (trend, _) = sta.calculate(&high, &low, &close);

        // Trend should be positive in uptrend
        for i in 20..trend.len() {
            assert!(
                trend[i] > 0.0,
                "Trend should be positive in uptrend at {}",
                i
            );
        }
    }

    // TechnicalIndicator trait tests
    #[test]
    fn test_technical_indicator_impl() {
        let mut data = OHLCVSeries::new();
        for i in 0..40 {
            let base = 100.0 + (i as f64 * 0.3).sin() * 10.0 + i as f64 * 0.5;
            data.open.push(base);
            data.high.push(base + 2.0);
            data.low.push(base - 2.0);
            data.close.push(base + 1.0);
            data.volume.push(1000.0 + i as f64 * 10.0);
        }

        // Test SwingProjection
        let sp = SwingProjection::new(5, 1.0).unwrap();
        let output = sp.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert!(output.secondary.is_some());
        assert_eq!(sp.name(), "Swing Projection");
        assert_eq!(sp.output_features(), 2);

        // Test SwingConfirmation
        let sc = SwingConfirmation::new(3, 2).unwrap();
        let output = sc.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert!(output.secondary.is_some());
        assert_eq!(sc.name(), "Swing Confirmation");
        assert_eq!(sc.output_features(), 2);

        // Test SwingRangeAnalysis
        let sra = SwingRangeAnalysis::new(5, 2).unwrap();
        let output = sra.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert!(output.secondary.is_some());
        assert_eq!(sra.name(), "Swing Range Analysis");
        assert_eq!(sra.output_features(), 2);

        // Test SwingBreakout
        let sb = SwingBreakout::new(5, 0.5).unwrap();
        let output = sb.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert!(output.secondary.is_some());
        assert_eq!(sb.name(), "Swing Breakout");
        assert_eq!(sb.output_features(), 2);

        // Test SwingMomentumFlow
        let smf = SwingMomentumFlow::new(5, 3).unwrap();
        let output = smf.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert!(output.secondary.is_some());
        assert_eq!(smf.name(), "Swing Momentum Flow");
        assert_eq!(smf.output_features(), 2);

        // Test SwingTrendAnalysis
        let sta = SwingTrendAnalysis::new(5, 15).unwrap();
        let output = sta.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 40);
        assert!(output.secondary.is_some());
        assert_eq!(sta.name(), "Swing Trend Analysis");
        assert_eq!(sta.output_features(), 2);
    }

    #[test]
    fn test_min_periods() {
        let sp = SwingProjection::new(5, 1.0).unwrap();
        assert_eq!(sp.min_periods(), 11);

        let sc = SwingConfirmation::new(3, 2).unwrap();
        assert_eq!(sc.min_periods(), 6);

        let sra = SwingRangeAnalysis::new(5, 2).unwrap();
        assert_eq!(sra.min_periods(), 7);

        let sb = SwingBreakout::new(5, 0.5).unwrap();
        assert_eq!(sb.min_periods(), 6);

        let smf = SwingMomentumFlow::new(5, 3).unwrap();
        assert_eq!(smf.min_periods(), 9);

        let sta = SwingTrendAnalysis::new(5, 15).unwrap();
        assert_eq!(sta.min_periods(), 16);
    }

    #[test]
    fn test_empty_data() {
        let high: Vec<f64> = vec![];
        let low: Vec<f64> = vec![];
        let close: Vec<f64> = vec![];
        let volume: Vec<f64> = vec![];

        let sp = SwingProjection::new(5, 1.0).unwrap();
        let (upper, lower) = sp.calculate(&high, &low, &close);
        assert!(upper.is_empty());
        assert!(lower.is_empty());

        let sc = SwingConfirmation::new(3, 2).unwrap();
        let (signal, strength) = sc.calculate(&high, &low, &close, &volume);
        assert!(signal.is_empty());
        assert!(strength.is_empty());

        let sra = SwingRangeAnalysis::new(5, 1).unwrap();
        let (width, position) = sra.calculate(&high, &low, &close);
        assert!(width.is_empty());
        assert!(position.is_empty());

        let sb = SwingBreakout::new(5, 0.5).unwrap();
        let (signal, strength) = sb.calculate(&high, &low, &close, &volume);
        assert!(signal.is_empty());
        assert!(strength.is_empty());

        let smf = SwingMomentumFlow::new(5, 2).unwrap();
        let (flow, roc) = smf.calculate(&high, &low, &close, &volume);
        assert!(flow.is_empty());
        assert!(roc.is_empty());

        let sta = SwingTrendAnalysis::new(5, 10).unwrap();
        let (trend, maturity) = sta.calculate(&high, &low, &close);
        assert!(trend.is_empty());
        assert!(maturity.is_empty());
    }

    #[test]
    fn test_small_data() {
        // Data smaller than required periods
        let high = vec![100.0, 101.0, 102.0];
        let low = vec![98.0, 99.0, 100.0];
        let close = vec![99.0, 100.0, 101.0];
        let volume = vec![1000.0, 1000.0, 1000.0];

        let sp = SwingProjection::new(5, 1.0).unwrap();
        let (upper, lower) = sp.calculate(&high, &low, &close);
        assert_eq!(upper.len(), 3);
        assert!(upper.iter().all(|x| x.is_nan()));
        assert!(lower.iter().all(|x| x.is_nan()));

        let sb = SwingBreakout::new(5, 0.5).unwrap();
        let (signal, strength) = sb.calculate(&high, &low, &close, &volume);
        assert_eq!(signal.len(), 3);
        assert!(signal.iter().all(|&x| x == 0.0));
        assert!(strength.iter().all(|&x| x == 0.0));
    }
}
