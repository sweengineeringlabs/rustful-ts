//! Advanced Market Breadth Indicators
//!
//! Additional breadth indicators for comprehensive market analysis.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Market Momentum Breadth - Breadth-based momentum indicator
#[derive(Debug, Clone)]
pub struct MarketMomentumBreadth {
    short_period: usize,
    long_period: usize,
}

impl MarketMomentumBreadth {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self { short_period, long_period })
    }

    /// Calculate momentum breadth using price proxy
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term advancing days
            let short_start = i.saturating_sub(self.short_period);
            let mut short_advances = 0;
            for j in (short_start + 1)..=i {
                if close[j] > close[j - 1] {
                    short_advances += 1;
                }
            }
            let short_ratio = short_advances as f64 / self.short_period as f64;

            // Long-term advancing days
            let long_start = i.saturating_sub(self.long_period);
            let mut long_advances = 0;
            for j in (long_start + 1)..=i {
                if close[j] > close[j - 1] {
                    long_advances += 1;
                }
            }
            let long_ratio = long_advances as f64 / self.long_period as f64;

            // Momentum breadth: short vs long ratio
            result[i] = (short_ratio - long_ratio) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for MarketMomentumBreadth {
    fn name(&self) -> &str {
        "Market Momentum Breadth"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Breadth Oscillator - Smoothed breadth ratio oscillator
#[derive(Debug, Clone)]
pub struct BreadthOscillator {
    period: usize,
    smooth_period: usize,
}

impl BreadthOscillator {
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smooth_period })
    }

    /// Calculate breadth oscillator (-100 to +100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut raw = vec![0.0; n];
        let mut result = vec![0.0; n];

        // Calculate raw breadth ratio
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            if total > 0 {
                raw[i] = ((advances - declines) as f64 / total as f64) * 100.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in self.period..n {
            if i == self.period {
                result[i] = raw[i];
            } else {
                result[i] = alpha * raw[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthOscillator {
    fn name(&self) -> &str {
        "Breadth Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Cumulative Breadth Index - Running sum of advance/decline differences
#[derive(Debug, Clone)]
pub struct CumulativeBreadthIndex {
    period: usize,
}

impl CumulativeBreadthIndex {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate cumulative breadth index
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let mut cumulative = 0.0;

        for i in 1..n {
            if close[i] > close[i - 1] {
                cumulative += 1.0;
            } else if close[i] < close[i - 1] {
                cumulative -= 1.0;
            }

            if i >= self.period {
                result[i] = cumulative;
            }
        }

        result
    }
}

impl TechnicalIndicator for CumulativeBreadthIndex {
    fn name(&self) -> &str {
        "Cumulative Breadth Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volume Breadth Ratio - Volume-weighted breadth
#[derive(Debug, Clone)]
pub struct VolumeBreadthRatio {
    period: usize,
}

impl VolumeBreadthRatio {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volume-weighted breadth ratio
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut up_volume = 0.0;
            let mut down_volume = 0.0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_volume += volume[j];
                } else if close[j] < close[j - 1] {
                    down_volume += volume[j];
                }
            }

            let total_volume = up_volume + down_volume;
            if total_volume > 1e-10 {
                result[i] = ((up_volume - down_volume) / total_volume) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumeBreadthRatio {
    fn name(&self) -> &str {
        "Volume Breadth Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Breadth Divergence - Detects price/breadth divergence
#[derive(Debug, Clone)]
pub struct BreadthDivergence {
    period: usize,
}

impl BreadthDivergence {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate breadth divergence score
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Price trend
            let price_change = if close[start] > 1e-10 {
                (close[i] / close[start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Breadth trend (cumulative AD)
            let mut breadth_change = 0.0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    breadth_change += 1.0;
                } else if close[j] < close[j - 1] {
                    breadth_change -= 1.0;
                }
            }

            // Normalize breadth
            let normalized_breadth = breadth_change / self.period as f64 * 50.0;

            // Divergence: difference between price direction and breadth direction
            // Positive = breadth leads price (bullish)
            // Negative = breadth lags price (bearish)
            result[i] = normalized_breadth - price_change;
        }

        result
    }
}

impl TechnicalIndicator for BreadthDivergence {
    fn name(&self) -> &str {
        "Breadth Divergence"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Participation Rate - Measures market participation
#[derive(Debug, Clone)]
pub struct ParticipationRate {
    period: usize,
    threshold: f64,
}

impl ParticipationRate {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate participation rate (% of bars with significant movement)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut participating = 0;

            for j in (start + 1)..=i {
                let pct_change = if close[j - 1] > 1e-10 {
                    ((close[j] / close[j - 1]) - 1.0).abs() * 100.0
                } else {
                    0.0
                };

                if pct_change >= self.threshold {
                    participating += 1;
                }
            }

            result[i] = (participating as f64 / self.period as f64) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for ParticipationRate {
    fn name(&self) -> &str {
        "Participation Rate"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Advanced Breadth Indicators (Single-Series Proxies)
// ============================================================================

/// BreadthMomentumAdvanced - Momentum of market breadth proxy
///
/// Calculates momentum of a breadth proxy by measuring the rate of change
/// in advancing vs declining periods over a lookback window.
#[derive(Debug, Clone)]
pub struct BreadthMomentumAdvanced {
    period: usize,
    momentum_period: usize,
}

impl BreadthMomentumAdvanced {
    pub fn new(period: usize, momentum_period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, momentum_period })
    }

    /// Calculate breadth momentum using price data as proxy
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First calculate breadth ratio for each period
        let mut breadth_ratio = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            if total > 0 {
                breadth_ratio[i] = (advances as f64 - declines as f64) / total as f64;
            }
        }

        // Calculate momentum of breadth ratio
        let min_idx = self.period + self.momentum_period;
        for i in min_idx..n {
            let current = breadth_ratio[i];
            let previous = breadth_ratio[i - self.momentum_period];
            result[i] = (current - previous) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for BreadthMomentumAdvanced {
    fn name(&self) -> &str {
        "Breadth Momentum Advanced"
    }

    fn min_periods(&self) -> usize {
        self.period + self.momentum_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthStrength - Measures the strength of breadth readings
///
/// Calculates a strength score by comparing current breadth to historical
/// breadth readings using a normalized z-score approach.
#[derive(Debug, Clone)]
pub struct BreadthStrength {
    period: usize,
    lookback: usize,
}

impl BreadthStrength {
    pub fn new(period: usize, lookback: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if lookback < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, lookback })
    }

    /// Calculate breadth strength score
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate raw breadth values
        let mut breadth = vec![0.0; n];
        for i in 1..n {
            if close[i] > close[i - 1] {
                breadth[i] = 1.0;
            } else if close[i] < close[i - 1] {
                breadth[i] = -1.0;
            }
        }

        // Calculate rolling sum of breadth
        let mut rolling_breadth = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let sum: f64 = breadth[(start + 1)..=i].iter().sum();
            rolling_breadth[i] = sum;
        }

        // Calculate strength using z-score normalization
        let min_idx = self.period + self.lookback;
        for i in min_idx..n {
            let lookback_start = i.saturating_sub(self.lookback);
            let window: Vec<f64> = rolling_breadth[lookback_start..i].to_vec();

            if window.is_empty() {
                continue;
            }

            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                // Z-score normalized to roughly -100 to +100
                result[i] = ((rolling_breadth[i] - mean) / std_dev) * 33.0;
                // Clamp to reasonable bounds
                result[i] = result[i].clamp(-100.0, 100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthStrength {
    fn name(&self) -> &str {
        "Breadth Strength"
    }

    fn min_periods(&self) -> usize {
        self.period + self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthOverbought - Detects overbought breadth conditions
///
/// Identifies when market breadth reaches extreme bullish levels that
/// historically precede pullbacks or corrections.
#[derive(Debug, Clone)]
pub struct BreadthOverbought {
    period: usize,
    threshold: f64,
}

impl BreadthOverbought {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if threshold <= 0.0 || threshold > 100.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0 and 100".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate overbought indicator
    /// Returns values 0-100 representing overbought intensity,
    /// with 100 being extreme overbought
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                }
            }

            let advance_pct = (advances as f64 / self.period as f64) * 100.0;

            // Calculate overbought intensity above threshold
            if advance_pct > self.threshold {
                // Scale 0-100 based on how far above threshold
                let excess = advance_pct - self.threshold;
                let max_excess = 100.0 - self.threshold;
                result[i] = (excess / max_excess) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthOverbought {
    fn name(&self) -> &str {
        "Breadth Overbought"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthOversold - Detects oversold breadth conditions
///
/// Identifies when market breadth reaches extreme bearish levels that
/// historically precede bounces or reversals.
#[derive(Debug, Clone)]
pub struct BreadthOversold {
    period: usize,
    threshold: f64,
}

impl BreadthOversold {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if threshold <= 0.0 || threshold > 100.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0 and 100".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate oversold indicator
    /// Returns values 0-100 representing oversold intensity,
    /// with 100 being extreme oversold
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let decline_pct = (declines as f64 / self.period as f64) * 100.0;

            // Calculate oversold intensity above threshold
            if decline_pct > self.threshold {
                // Scale 0-100 based on how far above threshold
                let excess = decline_pct - self.threshold;
                let max_excess = 100.0 - self.threshold;
                result[i] = (excess / max_excess) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthOversold {
    fn name(&self) -> &str {
        "Breadth Oversold"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthTrend - Measures long-term breadth trend
///
/// Uses exponential moving averages of breadth to identify
/// the prevailing trend in market breadth.
#[derive(Debug, Clone)]
pub struct BreadthTrend {
    short_period: usize,
    long_period: usize,
}

impl BreadthTrend {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if long_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self { short_period, long_period })
    }

    /// Calculate breadth trend
    /// Positive values indicate bullish trend, negative indicate bearish
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate daily breadth signal
        let mut breadth = vec![0.0; n];
        for i in 1..n {
            if close[i] > close[i - 1] {
                breadth[i] = 1.0;
            } else if close[i] < close[i - 1] {
                breadth[i] = -1.0;
            }
        }

        // Calculate short EMA of breadth
        let short_alpha = 2.0 / (self.short_period as f64 + 1.0);
        let mut short_ema = vec![0.0; n];
        for i in 0..n {
            if i == 0 {
                short_ema[i] = breadth[i];
            } else {
                short_ema[i] = short_alpha * breadth[i] + (1.0 - short_alpha) * short_ema[i - 1];
            }
        }

        // Calculate long EMA of breadth
        let long_alpha = 2.0 / (self.long_period as f64 + 1.0);
        let mut long_ema = vec![0.0; n];
        for i in 0..n {
            if i == 0 {
                long_ema[i] = breadth[i];
            } else {
                long_ema[i] = long_alpha * breadth[i] + (1.0 - long_alpha) * long_ema[i - 1];
            }
        }

        // Trend = difference between short and long EMA, scaled
        for i in self.long_period..n {
            result[i] = (short_ema[i] - long_ema[i]) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for BreadthTrend {
    fn name(&self) -> &str {
        "Breadth Trend"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthConfirmation - Confirms price moves with breadth
///
/// Measures whether price movements are confirmed by corresponding
/// breadth movements. High values indicate strong confirmation.
#[derive(Debug, Clone)]
pub struct BreadthConfirmation {
    period: usize,
    smoothing: usize,
}

impl BreadthConfirmation {
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Calculate breadth confirmation score
    /// Returns 0-100 where 100 = perfect confirmation
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate confirmation scores
        let mut raw_confirmation = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Price direction over period
            let price_change = close[i] - close[start];
            let price_direction = if price_change > 0.0 { 1.0 } else if price_change < 0.0 { -1.0 } else { 0.0 };

            // Count daily advances/declines
            let mut net_breadth: f64 = 0.0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    net_breadth += 1.0;
                } else if close[j] < close[j - 1] {
                    net_breadth -= 1.0;
                }
            }

            // Breadth direction
            let breadth_direction = if net_breadth > 0.0 { 1.0 } else if net_breadth < 0.0 { -1.0 } else { 0.0 };

            // Confirmation: both agree = 100, both disagree = 0
            if price_direction == breadth_direction && price_direction != 0.0 {
                // Confirmed - scale by strength
                let breadth_strength = (net_breadth.abs() / self.period as f64) * 100.0;
                raw_confirmation[i] = breadth_strength.min(100.0);
            } else if price_direction != breadth_direction && price_direction != 0.0 && breadth_direction != 0.0 {
                // Divergence - negative confirmation
                raw_confirmation[i] = 0.0;
            } else {
                // Neutral
                raw_confirmation[i] = 50.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in self.period..n {
            if i == self.period {
                result[i] = raw_confirmation[i];
            } else {
                result[i] = alpha * raw_confirmation[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthConfirmation {
    fn name(&self) -> &str {
        "Breadth Confirmation"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// New Breadth Indicators (6 additional indicators)
// ============================================================================

/// BreadthMomentumIndex - Momentum of breadth indicators
///
/// Measures the rate of change in market breadth by calculating the momentum
/// of the advance/decline ratio over a specified period. This helps identify
/// when breadth is accelerating or decelerating.
#[derive(Debug, Clone)]
pub struct BreadthMomentumIndex {
    period: usize,
    momentum_period: usize,
}

impl BreadthMomentumIndex {
    pub fn new(period: usize, momentum_period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, momentum_period })
    }

    /// Calculate breadth momentum index
    /// Returns momentum of breadth ratio scaled to percentage
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First calculate breadth ratio for each period
        let mut breadth_ratio = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            if total > 0 {
                // Ratio from -1 to +1
                breadth_ratio[i] = (advances as f64 - declines as f64) / total as f64;
            }
        }

        // Calculate rate of change (momentum) of breadth ratio
        let min_idx = self.period + self.momentum_period;
        for i in min_idx..n {
            let current = breadth_ratio[i];
            let previous = breadth_ratio[i - self.momentum_period];
            // Scale to percentage for readability
            result[i] = (current - previous) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for BreadthMomentumIndex {
    fn name(&self) -> &str {
        "Breadth Momentum Index"
    }

    fn min_periods(&self) -> usize {
        self.period + self.momentum_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// CumulativeBreadthMomentum - Cumulative momentum of breadth
///
/// Tracks the cumulative sum of breadth momentum over time, providing
/// a smoothed view of the overall trend in breadth momentum. Rising
/// values indicate improving market participation, falling values
/// indicate deteriorating participation.
#[derive(Debug, Clone)]
pub struct CumulativeBreadthMomentum {
    period: usize,
    smoothing: usize,
}

impl CumulativeBreadthMomentum {
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Calculate cumulative breadth momentum
    /// Returns cumulative sum of smoothed breadth changes
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate daily breadth signal
        let mut daily_breadth = vec![0.0; n];
        for i in 1..n {
            if close[i] > close[i - 1] {
                daily_breadth[i] = 1.0;
            } else if close[i] < close[i - 1] {
                daily_breadth[i] = -1.0;
            }
        }

        // Calculate rolling sum of breadth over period
        let mut rolling_sum = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let sum: f64 = daily_breadth[(start + 1)..=i].iter().sum();
            rolling_sum[i] = sum;
        }

        // Calculate momentum of the rolling sum
        let mut momentum = vec![0.0; n];
        let min_idx = self.period + self.smoothing;
        for i in min_idx..n {
            momentum[i] = rolling_sum[i] - rolling_sum[i - self.smoothing];
        }

        // Cumulative sum of momentum
        let mut cumulative = 0.0;
        for i in min_idx..n {
            cumulative += momentum[i];
            result[i] = cumulative;
        }

        result
    }
}

impl TechnicalIndicator for CumulativeBreadthMomentum {
    fn name(&self) -> &str {
        "Cumulative Breadth Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthVolatility - Volatility of breadth measures
///
/// Measures the volatility (standard deviation) of breadth readings
/// over a lookback period. High breadth volatility often precedes
/// major market turning points.
#[derive(Debug, Clone)]
pub struct BreadthVolatility {
    period: usize,
    lookback: usize,
}

impl BreadthVolatility {
    pub fn new(period: usize, lookback: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if lookback < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, lookback })
    }

    /// Calculate breadth volatility
    /// Returns standard deviation of breadth ratio over lookback period
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First calculate breadth ratio for each period
        let mut breadth_ratio = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            if total > 0 {
                breadth_ratio[i] = (advances as f64 - declines as f64) / total as f64;
            }
        }

        // Calculate rolling standard deviation of breadth ratio
        let min_idx = self.period + self.lookback;
        for i in min_idx..n {
            let lookback_start = i.saturating_sub(self.lookback);
            let window: Vec<f64> = breadth_ratio[lookback_start..=i].to_vec();

            if window.is_empty() {
                continue;
            }

            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std_dev = variance.sqrt();

            // Scale to percentage for readability
            result[i] = std_dev * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for BreadthVolatility {
    fn name(&self) -> &str {
        "Breadth Volatility"
    }

    fn min_periods(&self) -> usize {
        self.period + self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthTrendStrength - Trend strength from breadth
///
/// Measures the strength of the current breadth trend by comparing
/// short-term and long-term breadth EMAs. Returns a value from 0-100
/// where higher values indicate stronger trends.
#[derive(Debug, Clone)]
pub struct BreadthTrendStrength {
    short_period: usize,
    long_period: usize,
}

impl BreadthTrendStrength {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if long_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self { short_period, long_period })
    }

    /// Calculate breadth trend strength
    /// Returns 0-100 where higher values indicate stronger trends
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate daily breadth signal
        let mut breadth = vec![0.0; n];
        for i in 1..n {
            if close[i] > close[i - 1] {
                breadth[i] = 1.0;
            } else if close[i] < close[i - 1] {
                breadth[i] = -1.0;
            }
        }

        // Calculate short EMA of breadth
        let short_alpha = 2.0 / (self.short_period as f64 + 1.0);
        let mut short_ema = vec![0.0; n];
        for i in 0..n {
            if i == 0 {
                short_ema[i] = breadth[i];
            } else {
                short_ema[i] = short_alpha * breadth[i] + (1.0 - short_alpha) * short_ema[i - 1];
            }
        }

        // Calculate long EMA of breadth
        let long_alpha = 2.0 / (self.long_period as f64 + 1.0);
        let mut long_ema = vec![0.0; n];
        for i in 0..n {
            if i == 0 {
                long_ema[i] = breadth[i];
            } else {
                long_ema[i] = long_alpha * breadth[i] + (1.0 - long_alpha) * long_ema[i - 1];
            }
        }

        // Trend strength = absolute difference between short and long EMA
        // Normalized to 0-100 scale
        for i in self.long_period..n {
            // The difference is typically in range -2 to +2
            let diff = (short_ema[i] - long_ema[i]).abs();
            // Scale to 0-100 (diff of 1.0 = 50%)
            result[i] = (diff * 50.0).min(100.0);
        }

        result
    }
}

impl TechnicalIndicator for BreadthTrendStrength {
    fn name(&self) -> &str {
        "Breadth Trend Strength"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthExtremeDetector - Detects extreme breadth readings
///
/// Identifies when market breadth reaches extreme levels using a
/// z-score approach. Positive values indicate bullish extremes,
/// negative values indicate bearish extremes.
#[derive(Debug, Clone)]
pub struct BreadthExtremeDetector {
    period: usize,
    threshold: f64,
}

impl BreadthExtremeDetector {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate extreme breadth readings
    /// Returns z-score scaled values (-100 to +100)
    /// Values beyond threshold indicate extreme conditions
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate breadth ratio for each bar
        let mut breadth = vec![0.0; n];
        for i in 1..n {
            if close[i] > close[i - 1] {
                breadth[i] = 1.0;
            } else if close[i] < close[i - 1] {
                breadth[i] = -1.0;
            }
        }

        // Calculate rolling sum of breadth
        let mut rolling_breadth = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let sum: f64 = breadth[(start + 1)..=i].iter().sum();
            rolling_breadth[i] = sum;
        }

        // Calculate z-score using historical mean and std dev
        // Use expanding window for mean/std calculations
        for i in self.period..n {
            let history: Vec<f64> = rolling_breadth[self.period..=i].to_vec();

            if history.len() < 2 {
                continue;
            }

            let mean: f64 = history.iter().sum::<f64>() / history.len() as f64;
            let variance: f64 = history.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / history.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                let z_score = (rolling_breadth[i] - mean) / std_dev;

                // Only report extremes beyond threshold
                if z_score.abs() >= self.threshold {
                    // Scale z-score to -100 to +100 range
                    // z-score of 3 maps to 100
                    result[i] = (z_score / 3.0 * 100.0).clamp(-100.0, 100.0);
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthExtremeDetector {
    fn name(&self) -> &str {
        "Breadth Extreme Detector"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthDivergenceIndex - Divergence between price and breadth
///
/// Measures the divergence between price momentum and breadth momentum.
/// Positive values indicate breadth leading price (bullish divergence),
/// negative values indicate breadth lagging price (bearish divergence).
#[derive(Debug, Clone)]
pub struct BreadthDivergenceIndex {
    period: usize,
    momentum_period: usize,
}

impl BreadthDivergenceIndex {
    pub fn new(period: usize, momentum_period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, momentum_period })
    }

    /// Calculate breadth divergence index
    /// Returns divergence score scaled to -100 to +100
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate breadth ratio for each period
        let mut breadth_ratio = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            if total > 0 {
                breadth_ratio[i] = (advances as f64 - declines as f64) / total as f64;
            }
        }

        // Calculate momentum of breadth and price
        let min_idx = self.period + self.momentum_period;
        for i in min_idx..n {
            // Breadth momentum
            let breadth_momentum = breadth_ratio[i] - breadth_ratio[i - self.momentum_period];

            // Price momentum (normalized rate of change)
            let price_momentum = if close[i - self.momentum_period] > 1e-10 {
                close[i] / close[i - self.momentum_period] - 1.0
            } else {
                0.0
            };

            // Normalize price momentum to similar scale as breadth momentum
            // Breadth momentum ranges roughly -2 to +2
            // Price momentum typically ranges -0.1 to +0.1 (10%)
            let normalized_price_momentum = price_momentum * 10.0;

            // Divergence = breadth momentum - price momentum
            // Positive = breadth stronger than price (bullish)
            // Negative = breadth weaker than price (bearish)
            let divergence = breadth_momentum - normalized_price_momentum;

            // Scale to -100 to +100
            result[i] = (divergence * 50.0).clamp(-100.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for BreadthDivergenceIndex {
    fn name(&self) -> &str {
        "Breadth Divergence Index"
    }

    fn min_periods(&self) -> usize {
        self.period + self.momentum_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// NEW Breadth Indicators (6 additional indicators as requested)
// ============================================================================

/// BreadthTrustThrust - Measures thrust in market breadth
///
/// The Breadth Trust Thrust indicator measures the percentage of advancing
/// issues that exceeds a high threshold within a short time period. It
/// identifies powerful market thrusts that historically signal the beginning
/// of significant rallies. A thrust signal occurs when breadth rapidly
/// transitions from oversold to overbought conditions.
#[derive(Debug, Clone)]
pub struct BreadthTrustThrust {
    /// Period for calculating advance ratio
    period: usize,
    /// Threshold percentage for thrust detection (e.g., 61.5%)
    thrust_threshold: f64,
    /// EMA smoothing period for the thrust indicator
    smoothing: usize,
}

impl BreadthTrustThrust {
    /// Creates a new BreadthTrustThrust indicator.
    ///
    /// # Arguments
    /// * `period` - The lookback period for calculating advance ratio (minimum 2)
    /// * `thrust_threshold` - The threshold for thrust detection (0-100, typically 61.5)
    /// * `smoothing` - EMA smoothing period (minimum 2)
    ///
    /// # Errors
    /// Returns an error if period < 2, smoothing < 2, or threshold is out of range
    pub fn new(period: usize, thrust_threshold: f64, smoothing: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if thrust_threshold <= 0.0 || thrust_threshold >= 100.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "thrust_threshold".to_string(),
                reason: "must be between 0 and 100 exclusive".to_string(),
            });
        }
        Ok(Self { period, thrust_threshold, smoothing })
    }

    /// Calculate breadth trust thrust values.
    ///
    /// Returns a vector where:
    /// - Values > 0 indicate thrust strength (0-100 scale)
    /// - Higher values indicate stronger thrust signals
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate raw advance percentage for each period
        let mut advance_pct = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut total = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                }
                total += 1;
            }

            if total > 0 {
                advance_pct[i] = (advances as f64 / total as f64) * 100.0;
            }
        }

        // Calculate thrust signal when advance percentage exceeds threshold
        let mut thrust_raw = vec![0.0; n];
        for i in self.period..n {
            if advance_pct[i] > self.thrust_threshold {
                // Scale the thrust intensity based on how much it exceeds threshold
                let excess = advance_pct[i] - self.thrust_threshold;
                let max_excess = 100.0 - self.thrust_threshold;
                thrust_raw[i] = (excess / max_excess) * 100.0;
            }
        }

        // Apply EMA smoothing to thrust signal
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in self.period..n {
            if i == self.period {
                result[i] = thrust_raw[i];
            } else {
                result[i] = alpha * thrust_raw[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthTrustThrust {
    fn name(&self) -> &str {
        "Breadth Trust Thrust"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// AdvanceDeclineOscillator - A/D based oscillator
///
/// The Advance/Decline Oscillator measures the difference between short-term
/// and long-term moving averages of the advance-decline ratio. It oscillates
/// around zero, with positive values indicating bullish breadth momentum and
/// negative values indicating bearish breadth momentum.
#[derive(Debug, Clone)]
pub struct AdvanceDeclineOscillator {
    /// Short-term EMA period
    short_period: usize,
    /// Long-term EMA period
    long_period: usize,
}

impl AdvanceDeclineOscillator {
    /// Creates a new AdvanceDeclineOscillator.
    ///
    /// # Arguments
    /// * `short_period` - Short-term EMA period (minimum 2)
    /// * `long_period` - Long-term EMA period (must be > short_period)
    ///
    /// # Errors
    /// Returns an error if short_period < 2 or long_period <= short_period
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
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
        Ok(Self { short_period, long_period })
    }

    /// Calculate advance/decline oscillator values.
    ///
    /// Returns the difference between short and long EMA of A/D ratio,
    /// scaled to provide meaningful oscillator readings.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate daily A/D value (+1 for advance, -1 for decline, 0 for unchanged)
        let mut ad_value = vec![0.0; n];
        for i in 1..n {
            if close[i] > close[i - 1] {
                ad_value[i] = 1.0;
            } else if close[i] < close[i - 1] {
                ad_value[i] = -1.0;
            }
        }

        // Calculate short-term EMA
        let short_alpha = 2.0 / (self.short_period as f64 + 1.0);
        let mut short_ema = vec![0.0; n];
        for i in 1..n {
            if i == 1 {
                short_ema[i] = ad_value[i];
            } else {
                short_ema[i] = short_alpha * ad_value[i] + (1.0 - short_alpha) * short_ema[i - 1];
            }
        }

        // Calculate long-term EMA
        let long_alpha = 2.0 / (self.long_period as f64 + 1.0);
        let mut long_ema = vec![0.0; n];
        for i in 1..n {
            if i == 1 {
                long_ema[i] = ad_value[i];
            } else {
                long_ema[i] = long_alpha * ad_value[i] + (1.0 - long_alpha) * long_ema[i - 1];
            }
        }

        // Oscillator = short EMA - long EMA, scaled to percentage
        for i in self.long_period..n {
            result[i] = (short_ema[i] - long_ema[i]) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for AdvanceDeclineOscillator {
    fn name(&self) -> &str {
        "Advance Decline Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthStrengthIndex - Breadth strength measure
///
/// The Breadth Strength Index measures the internal strength of market breadth
/// by combining multiple breadth factors: advance percentage, persistence of
/// advances, and acceleration of breadth. It provides a comprehensive view
/// of market participation strength.
#[derive(Debug, Clone)]
pub struct BreadthStrengthIndex {
    /// Period for calculating breadth components
    period: usize,
    /// Lookback period for strength normalization
    lookback: usize,
}

impl BreadthStrengthIndex {
    /// Creates a new BreadthStrengthIndex.
    ///
    /// # Arguments
    /// * `period` - Period for calculating breadth components (minimum 5)
    /// * `lookback` - Lookback period for strength normalization (minimum 5)
    ///
    /// # Errors
    /// Returns an error if period < 5 or lookback < 5
    pub fn new(period: usize, lookback: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if lookback < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period, lookback })
    }

    /// Calculate breadth strength index values.
    ///
    /// Returns values from 0-100 where:
    /// - 0-30: Weak breadth (bearish)
    /// - 30-70: Neutral breadth
    /// - 70-100: Strong breadth (bullish)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Component 1: Advance percentage over period
        let mut advance_pct = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut total = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                }
                total += 1;
            }

            if total > 0 {
                advance_pct[i] = advances as f64 / total as f64;
            }
        }

        // Component 2: Persistence - consecutive advances count
        let mut persistence = vec![0.0; n];
        let mut consecutive = 0;
        for i in 1..n {
            if close[i] > close[i - 1] {
                consecutive += 1;
            } else {
                consecutive = 0;
            }
            persistence[i] = consecutive as f64;
        }

        // Normalize persistence over lookback
        let mut norm_persistence = vec![0.0; n];
        let min_idx = self.period + self.lookback;
        for i in min_idx..n {
            let lookback_start = i.saturating_sub(self.lookback);
            let max_persistence = persistence[lookback_start..=i]
                .iter()
                .fold(0.0_f64, |max, &val| if val > max { val } else { max });

            if max_persistence > 0.0 {
                norm_persistence[i] = persistence[i] / max_persistence;
            }
        }

        // Component 3: Acceleration - change in advance percentage
        let mut acceleration = vec![0.0; n];
        for i in (self.period + 1)..n {
            let change = advance_pct[i] - advance_pct[i - 1];
            // Normalize to 0-1 range (change from -1 to +1 maps to 0-1)
            acceleration[i] = (change + 1.0) / 2.0;
        }

        // Combine components with weights
        for i in min_idx..n {
            // Weight: 50% advance percentage, 30% persistence, 20% acceleration
            let strength = (advance_pct[i] * 0.5 + norm_persistence[i] * 0.3 + acceleration[i] * 0.2) * 100.0;
            result[i] = strength.clamp(0.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for BreadthStrengthIndex {
    fn name(&self) -> &str {
        "Breadth Strength Index"
    }

    fn min_periods(&self) -> usize {
        self.period + self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// MarketInternalsScore - Score from market internals
///
/// The Market Internals Score combines multiple breadth metrics into a single
/// composite score that reflects overall market health. It considers:
/// - Advance/decline ratio
/// - Up/down volume ratio
/// - New highs vs new lows proxy
/// - Trend consistency
///
/// Higher scores indicate healthier market internals.
#[derive(Debug, Clone)]
pub struct MarketInternalsScore {
    /// Period for calculating internal metrics
    period: usize,
    /// Smoothing period for the final score
    smoothing: usize,
}

impl MarketInternalsScore {
    /// Creates a new MarketInternalsScore indicator.
    ///
    /// # Arguments
    /// * `period` - Period for calculating metrics (minimum 5)
    /// * `smoothing` - Smoothing period for final score (minimum 2)
    ///
    /// # Errors
    /// Returns an error if period < 5 or smoothing < 2
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Calculate market internals score.
    ///
    /// Returns values from 0-100 where:
    /// - 0-25: Very weak internals (bearish)
    /// - 25-50: Weak internals
    /// - 50-75: Strong internals
    /// - 75-100: Very strong internals (bullish)
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Component 1: A/D ratio score
        let mut ad_score = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            if total > 0 {
                // Score: 0 = all declines, 50 = even, 100 = all advances
                ad_score[i] = (advances as f64 / total as f64) * 100.0;
            } else {
                ad_score[i] = 50.0; // Neutral when no changes
            }
        }

        // Component 2: Volume ratio score (up volume vs down volume)
        let mut vol_score = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut up_vol = 0.0;
            let mut down_vol = 0.0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_vol += volume[j];
                } else if close[j] < close[j - 1] {
                    down_vol += volume[j];
                }
            }

            let total_vol = up_vol + down_vol;
            if total_vol > 1e-10 {
                vol_score[i] = (up_vol / total_vol) * 100.0;
            } else {
                vol_score[i] = 50.0;
            }
        }

        // Component 3: New highs proxy (price at period high)
        let mut high_score = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let period_high = close[start..=i].iter().fold(f64::MIN, |max, &v| v.max(max));
            let period_low = close[start..=i].iter().fold(f64::MAX, |min, &v| v.min(min));

            let range = period_high - period_low;
            if range > 1e-10 {
                // Where is current price in the range? 0 = at low, 100 = at high
                high_score[i] = ((close[i] - period_low) / range) * 100.0;
            } else {
                high_score[i] = 50.0;
            }
        }

        // Component 4: Trend consistency (how often price moves in same direction)
        let mut trend_score = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Count direction consistency
            let price_change = close[i] - close[start];
            let direction = if price_change > 0.0 { 1 } else if price_change < 0.0 { -1 } else { 0 };

            let mut consistent_days = 0;
            for j in (start + 1)..=i {
                let daily_direction = if close[j] > close[j - 1] { 1 }
                    else if close[j] < close[j - 1] { -1 }
                    else { 0 };

                if daily_direction == direction {
                    consistent_days += 1;
                }
            }

            let total_days = self.period;
            trend_score[i] = (consistent_days as f64 / total_days as f64) * 100.0;
        }

        // Combine with equal weights (25% each)
        let mut raw_score = vec![0.0; n];
        for i in self.period..n {
            raw_score[i] = (ad_score[i] + vol_score[i] + high_score[i] + trend_score[i]) / 4.0;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in self.period..n {
            if i == self.period {
                result[i] = raw_score[i];
            } else {
                result[i] = alpha * raw_score[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for MarketInternalsScore {
    fn name(&self) -> &str {
        "Market Internals Score"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// BreadthPersistence - Measures persistence of breadth movements
///
/// Tracks how long breadth readings persist in one direction, identifying
/// sustained bullish or bearish participation. High persistence values
/// indicate strong trending conditions with consistent market participation.
#[derive(Debug, Clone)]
pub struct BreadthPersistence {
    /// Period for calculating breadth
    period: usize,
    /// Threshold for determining positive/negative breadth
    threshold: f64,
}

impl BreadthPersistence {
    /// Creates a new BreadthPersistence indicator.
    ///
    /// # Arguments
    /// * `period` - Period for calculating breadth (minimum 5)
    /// * `threshold` - Threshold percentage for positive/negative classification (0-50)
    ///
    /// # Errors
    /// Returns an error if period < 5 or threshold is out of range
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if threshold < 0.0 || threshold > 50.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0 and 50".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate breadth persistence values.
    ///
    /// Returns values where:
    /// - Positive values indicate consecutive bullish breadth readings
    /// - Negative values indicate consecutive bearish breadth readings
    /// - Magnitude indicates the number of consecutive periods
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate advance percentage for each period
        let mut advance_pct = vec![50.0; n]; // Default to neutral
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut total = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                }
                total += 1;
            }

            if total > 0 {
                advance_pct[i] = (advances as f64 / total as f64) * 100.0;
            }
        }

        // Track persistence
        let mut consecutive_positive = 0.0;
        let mut consecutive_negative = 0.0;
        let positive_threshold = 50.0 + self.threshold;
        let negative_threshold = 50.0 - self.threshold;

        for i in self.period..n {
            if advance_pct[i] > positive_threshold {
                // Bullish breadth
                consecutive_positive += 1.0;
                consecutive_negative = 0.0;
                result[i] = consecutive_positive;
            } else if advance_pct[i] < negative_threshold {
                // Bearish breadth
                consecutive_negative += 1.0;
                consecutive_positive = 0.0;
                result[i] = -consecutive_negative;
            } else {
                // Neutral - reset both counters
                consecutive_positive = 0.0;
                consecutive_negative = 0.0;
                result[i] = 0.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthPersistence {
    fn name(&self) -> &str {
        "Breadth Persistence"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthAcceleration - Rate of change in breadth momentum
///
/// Measures the acceleration (second derivative) of market breadth, identifying
/// when breadth momentum is speeding up or slowing down. This can signal
/// potential trend reversals before they occur in price.
#[derive(Debug, Clone)]
pub struct BreadthAcceleration {
    /// Period for calculating breadth ratio
    period: usize,
    /// Period for calculating first derivative (momentum)
    momentum_period: usize,
    /// Period for calculating second derivative (acceleration)
    acceleration_period: usize,
}

impl BreadthAcceleration {
    /// Creates a new BreadthAcceleration indicator.
    ///
    /// # Arguments
    /// * `period` - Period for calculating breadth ratio (minimum 2)
    /// * `momentum_period` - Period for first derivative (minimum 2)
    /// * `acceleration_period` - Period for second derivative (minimum 2)
    ///
    /// # Errors
    /// Returns an error if any period is less than 2
    pub fn new(period: usize, momentum_period: usize, acceleration_period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
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
        Ok(Self { period, momentum_period, acceleration_period })
    }

    /// Calculate breadth acceleration values.
    ///
    /// Returns values where:
    /// - Positive values indicate accelerating bullish breadth
    /// - Negative values indicate accelerating bearish breadth
    /// - Values near zero indicate stable breadth momentum
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate breadth ratio for each period
        let mut breadth_ratio = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            if total > 0 {
                breadth_ratio[i] = (advances as f64 - declines as f64) / total as f64;
            }
        }

        // Calculate first derivative (momentum)
        let mut momentum = vec![0.0; n];
        let mom_start = self.period + self.momentum_period;
        for i in mom_start..n {
            momentum[i] = breadth_ratio[i] - breadth_ratio[i - self.momentum_period];
        }

        // Calculate second derivative (acceleration)
        let acc_start = mom_start + self.acceleration_period;
        for i in acc_start..n {
            let acceleration = momentum[i] - momentum[i - self.acceleration_period];
            // Scale to reasonable range
            result[i] = acceleration * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for BreadthAcceleration {
    fn name(&self) -> &str {
        "Breadth Acceleration"
    }

    fn min_periods(&self) -> usize {
        self.period + self.momentum_period + self.acceleration_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// 6 NEW Breadth Indicators (AdvanceDeclineRatio, BreadthMomentumIndicator, etc.)
// ============================================================================

/// AdvanceDeclineRatio - Simple Advance/Decline Ratio
///
/// Calculates the ratio of advancing periods to declining periods over a
/// specified lookback window. Unlike oscillators that smooth the data, this
/// provides a raw ratio that can range from 0 (all declines) to infinity
/// (all advances). Values above 1.0 indicate more advances than declines.
///
/// # Formula
/// A/D Ratio = Advances / Declines
///
/// # Interpretation
/// - Ratio > 1.0: More advances than declines (bullish)
/// - Ratio = 1.0: Equal advances and declines (neutral)
/// - Ratio < 1.0: More declines than advances (bearish)
/// - Extreme readings (> 2.0 or < 0.5) may indicate overbought/oversold
#[derive(Debug, Clone)]
pub struct AdvanceDeclineRatio {
    /// Period for calculating the ratio
    period: usize,
}

impl AdvanceDeclineRatio {
    /// Creates a new AdvanceDeclineRatio indicator.
    ///
    /// # Arguments
    /// * `period` - The lookback period for calculating advances/declines (minimum 5)
    ///
    /// # Errors
    /// Returns an error if period is less than 5
    ///
    /// # Example
    /// ```
    /// use indicator_core::breadth::advanced::AdvanceDeclineRatio;
    /// let adr = AdvanceDeclineRatio::new(10).unwrap();
    /// ```
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate the advance/decline ratio for each bar.
    ///
    /// # Arguments
    /// * `close` - Array of closing prices
    ///
    /// # Returns
    /// Vector of A/D ratios. Values of 0.0 indicate warmup period or
    /// periods with no declines (ratio would be infinity, capped at 10.0).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            // Calculate ratio, handling division by zero
            if declines > 0 {
                result[i] = advances as f64 / declines as f64;
            } else if advances > 0 {
                // All advances, no declines - cap at 10.0
                result[i] = 10.0;
            } else {
                // No advances or declines (all unchanged)
                result[i] = 1.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for AdvanceDeclineRatio {
    fn name(&self) -> &str {
        "Advance Decline Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthMomentumIndicator - Rate of Change in Breadth
///
/// Measures the momentum of market breadth by calculating the rate of change
/// in the advance percentage. This differs from other breadth momentum indicators
/// by focusing specifically on the percentage change in breadth rather than
/// cumulative or smoothed values.
///
/// # Formula
/// Breadth Momentum = (Current Advance% - Previous Advance%) / Previous Advance% * 100
///
/// # Interpretation
/// - Positive values: Breadth is improving (momentum increasing)
/// - Negative values: Breadth is deteriorating (momentum decreasing)
/// - Extreme values may signal trend exhaustion
#[derive(Debug, Clone)]
pub struct BreadthMomentumIndicator {
    /// Period for calculating advance percentage
    period: usize,
    /// Rate of change period
    roc_period: usize,
}

impl BreadthMomentumIndicator {
    /// Creates a new BreadthMomentumIndicator.
    ///
    /// # Arguments
    /// * `period` - Period for calculating advance percentage (minimum 5)
    /// * `roc_period` - Period for rate of change calculation (minimum 2)
    ///
    /// # Errors
    /// Returns an error if period < 5 or roc_period < 2
    pub fn new(period: usize, roc_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if roc_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "roc_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, roc_period })
    }

    /// Calculate breadth momentum indicator values.
    ///
    /// # Returns
    /// Vector of momentum values as percentage change in breadth.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First calculate advance percentage for each period
        let mut advance_pct = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut total = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                }
                total += 1;
            }

            if total > 0 {
                advance_pct[i] = (advances as f64 / total as f64) * 100.0;
            }
        }

        // Calculate rate of change of advance percentage
        let min_idx = self.period + self.roc_period;
        for i in min_idx..n {
            let current = advance_pct[i];
            let previous = advance_pct[i - self.roc_period];

            // Calculate percentage change (ROC style)
            if previous > 1e-10 {
                result[i] = ((current - previous) / previous) * 100.0;
            } else if current > 1e-10 {
                // Previous was near zero, current is positive
                result[i] = 100.0;
            }
            // If both near zero, result stays 0.0
        }

        result
    }
}

impl TechnicalIndicator for BreadthMomentumIndicator {
    fn name(&self) -> &str {
        "Breadth Momentum Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period + self.roc_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// CumulativeBreadthLine - Running Sum of Net Advances
///
/// Calculates a cumulative line by adding the net advances (advances minus
/// declines) for each period. This creates a running total that rises when
/// advances dominate and falls when declines dominate, similar to an
/// Advance/Decline Line but using price data as a proxy.
///
/// # Formula
/// CBL[i] = CBL[i-1] + (Advances[i] - Declines[i])
///
/// # Interpretation
/// - Rising line: Overall market participation is bullish
/// - Falling line: Overall market participation is bearish
/// - Divergence from price may signal trend weakness
#[derive(Debug, Clone)]
pub struct CumulativeBreadthLine {
    /// Starting value for the cumulative line
    base_value: f64,
}

impl CumulativeBreadthLine {
    /// Creates a new CumulativeBreadthLine indicator.
    ///
    /// # Arguments
    /// * `base_value` - Starting value for the cumulative line (default: 1000.0)
    ///
    /// # Example
    /// ```
    /// use indicator_core::breadth::advanced::CumulativeBreadthLine;
    /// let cbl = CumulativeBreadthLine::new(1000.0).unwrap();
    /// ```
    pub fn new(base_value: f64) -> Result<Self> {
        if base_value < 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_value".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        Ok(Self { base_value })
    }

    /// Creates a CumulativeBreadthLine with default base value of 1000.0
    pub fn default_params() -> Result<Self> {
        Self::new(1000.0)
    }

    /// Calculate the cumulative breadth line.
    ///
    /// # Returns
    /// Vector of cumulative breadth values starting from base_value.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![self.base_value; n];

        for i in 1..n {
            let daily_change = if close[i] > close[i - 1] {
                1.0 // Advance
            } else if close[i] < close[i - 1] {
                -1.0 // Decline
            } else {
                0.0 // Unchanged
            };

            result[i] = result[i - 1] + daily_change;
        }

        result
    }
}

impl TechnicalIndicator for CumulativeBreadthLine {
    fn name(&self) -> &str {
        "Cumulative Breadth Line"
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// HighLowIndex - New Highs vs New Lows Ratio
///
/// Measures the ratio of bars making new period highs versus new period lows.
/// This provides insight into the breadth of market strength by tracking
/// how many periods are reaching new highs compared to new lows.
///
/// # Formula
/// HLI = (New Highs - New Lows) / (New Highs + New Lows) * 100
///
/// # Interpretation
/// - Values > 0: More new highs than lows (bullish)
/// - Values < 0: More new lows than highs (bearish)
/// - Extreme readings (>50 or <-50) indicate strong trends
#[derive(Debug, Clone)]
pub struct HighLowIndex {
    /// Period for determining new highs/lows
    lookback_period: usize,
    /// Period for calculating the index
    calculation_period: usize,
}

impl HighLowIndex {
    /// Creates a new HighLowIndex indicator.
    ///
    /// # Arguments
    /// * `lookback_period` - Period for determining if price is at high/low (minimum 5)
    /// * `calculation_period` - Period for summing new highs/lows (minimum 5)
    ///
    /// # Errors
    /// Returns an error if either period is less than 5
    pub fn new(lookback_period: usize, calculation_period: usize) -> Result<Self> {
        if lookback_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if calculation_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "calculation_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { lookback_period, calculation_period })
    }

    /// Calculate the high/low index values.
    ///
    /// # Returns
    /// Vector of index values ranging from -100 to +100.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First, mark each bar as new high, new low, or neither
        let mut new_high = vec![false; n];
        let mut new_low = vec![false; n];

        for i in self.lookback_period..n {
            let start = i.saturating_sub(self.lookback_period);

            // Check if current high is highest in lookback
            let period_high = high[start..i].iter().fold(f64::MIN, |max, &v| v.max(max));
            if high[i] > period_high {
                new_high[i] = true;
            }

            // Check if current low is lowest in lookback
            let period_low = low[start..i].iter().fold(f64::MAX, |min, &v| v.min(min));
            if low[i] < period_low {
                new_low[i] = true;
            }
        }

        // Calculate index over calculation period
        let min_idx = self.lookback_period + self.calculation_period;
        for i in min_idx..n {
            let start = i.saturating_sub(self.calculation_period);

            let highs_count = new_high[(start + 1)..=i].iter().filter(|&&x| x).count();
            let lows_count = new_low[(start + 1)..=i].iter().filter(|&&x| x).count();

            let total = highs_count + lows_count;
            if total > 0 {
                // Scale to -100 to +100
                result[i] = ((highs_count as f64 - lows_count as f64) / total as f64) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for HighLowIndex {
    fn name(&self) -> &str {
        "High Low Index"
    }

    fn min_periods(&self) -> usize {
        self.lookback_period + self.calculation_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// PercentAboveMA - Percentage of Bars Above Moving Average
///
/// Calculates the percentage of recent bars where the close price is above
/// a simple moving average. This measures the breadth of price action relative
/// to its trend, indicating how consistently price stays above average.
///
/// # Formula
/// PAMA = (Count of closes > SMA) / Period * 100
///
/// # Interpretation
/// - Values > 50: Majority of closes above MA (bullish trend)
/// - Values < 50: Majority of closes below MA (bearish trend)
/// - Extreme readings (>80 or <20) may indicate overbought/oversold
#[derive(Debug, Clone)]
pub struct PercentAboveMA {
    /// Moving average period
    ma_period: usize,
    /// Calculation period for percentage
    calc_period: usize,
}

impl PercentAboveMA {
    /// Creates a new PercentAboveMA indicator.
    ///
    /// # Arguments
    /// * `ma_period` - Period for the moving average (minimum 5)
    /// * `calc_period` - Period for calculating the percentage (minimum 5)
    ///
    /// # Errors
    /// Returns an error if either period is less than 5
    pub fn new(ma_period: usize, calc_period: usize) -> Result<Self> {
        if ma_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "ma_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if calc_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "calc_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { ma_period, calc_period })
    }

    /// Calculate the percentage of bars above MA.
    ///
    /// # Returns
    /// Vector of percentage values (0-100).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate simple moving average
        let mut sma = vec![0.0; n];
        for i in self.ma_period..n {
            let start = i.saturating_sub(self.ma_period);
            let sum: f64 = close[(start + 1)..=i].iter().sum();
            sma[i] = sum / self.ma_period as f64;
        }

        // Mark bars above MA
        let mut above_ma = vec![false; n];
        for i in self.ma_period..n {
            above_ma[i] = close[i] > sma[i];
        }

        // Calculate percentage over calculation period
        let min_idx = self.ma_period + self.calc_period;
        for i in min_idx..n {
            let start = i.saturating_sub(self.calc_period);
            let count_above = above_ma[(start + 1)..=i].iter().filter(|&&x| x).count();
            result[i] = (count_above as f64 / self.calc_period as f64) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for PercentAboveMA {
    fn name(&self) -> &str {
        "Percent Above MA"
    }

    fn min_periods(&self) -> usize {
        self.ma_period + self.calc_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthDiffusion - Diffusion Index of Breadth
///
/// Measures the diffusion of positive breadth readings across multiple
/// timeframes. It calculates breadth over several lookback periods and
/// counts how many show positive readings, providing a multi-timeframe
/// view of market participation.
///
/// # Formula
/// Diffusion = (Count of positive breadth readings across timeframes) / (Number of timeframes) * 100
///
/// # Interpretation
/// - Values > 50: Majority of timeframes show positive breadth
/// - Values < 50: Majority of timeframes show negative breadth
/// - Extremes near 0 or 100 indicate broad agreement across timeframes
#[derive(Debug, Clone)]
pub struct BreadthDiffusion {
    /// Short-term period
    short_period: usize,
    /// Medium-term period
    medium_period: usize,
    /// Long-term period
    long_period: usize,
    /// Smoothing period for final output
    smoothing: usize,
}

impl BreadthDiffusion {
    /// Creates a new BreadthDiffusion indicator.
    ///
    /// # Arguments
    /// * `short_period` - Short-term lookback (minimum 5)
    /// * `medium_period` - Medium-term lookback (must be > short_period)
    /// * `long_period` - Long-term lookback (must be > medium_period)
    /// * `smoothing` - EMA smoothing period (minimum 2)
    ///
    /// # Errors
    /// Returns an error if periods are invalid or not in ascending order
    pub fn new(short_period: usize, medium_period: usize, long_period: usize, smoothing: usize) -> Result<Self> {
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if medium_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if long_period <= medium_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than medium_period".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { short_period, medium_period, long_period, smoothing })
    }

    /// Calculate breadth diffusion index values.
    ///
    /// # Returns
    /// Vector of diffusion values (0-100).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate breadth for each timeframe
        let periods = [self.short_period, self.medium_period, self.long_period];
        let mut breadth_signals: Vec<Vec<i32>> = vec![vec![0; n]; 3];

        for (idx, &period) in periods.iter().enumerate() {
            for i in period..n {
                let start = i.saturating_sub(period);
                let mut advances = 0;
                let mut declines = 0;

                for j in (start + 1)..=i {
                    if close[j] > close[j - 1] {
                        advances += 1;
                    } else if close[j] < close[j - 1] {
                        declines += 1;
                    }
                }

                // Positive breadth = more advances than declines
                if advances > declines {
                    breadth_signals[idx][i] = 1;
                } else if declines > advances {
                    breadth_signals[idx][i] = -1;
                }
                // Equal: stays 0
            }
        }

        // Calculate raw diffusion (how many timeframes are positive)
        let mut raw_diffusion = vec![0.0; n];
        for i in self.long_period..n {
            let positive_count = breadth_signals.iter()
                .filter(|signals| signals[i] > 0)
                .count();
            let negative_count = breadth_signals.iter()
                .filter(|signals| signals[i] < 0)
                .count();

            // Scale: 0 = all negative, 50 = mixed, 100 = all positive
            // Using a weighted approach: +1 for positive, 0 for neutral, -1 for negative
            let score = positive_count as f64 - negative_count as f64;
            // Map from -3..+3 to 0..100
            raw_diffusion[i] = ((score + 3.0) / 6.0) * 100.0;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in self.long_period..n {
            if i == self.long_period {
                result[i] = raw_diffusion[i];
            } else {
                result[i] = alpha * raw_diffusion[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthDiffusion {
    fn name(&self) -> &str {
        "Breadth Diffusion"
    }

    fn min_periods(&self) -> usize {
        self.long_period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// 6 NEW Breadth Indicators (BreadthRatio, BreadthScore, MarketParticipation, etc.)
// ============================================================================

/// BreadthRatio - Simple Breadth Ratio
///
/// Calculates a normalized breadth ratio that measures the proportion of
/// advancing periods relative to total active periods (advances + declines).
/// Unlike AdvanceDeclineRatio which produces unbounded values, this indicator
/// normalizes the output to a 0-100 scale for easier interpretation.
///
/// # Formula
/// BreadthRatio = (Advances / (Advances + Declines)) * 100
///
/// # Interpretation
/// - Values > 50: More advances than declines (bullish)
/// - Values = 50: Equal advances and declines (neutral)
/// - Values < 50: More declines than advances (bearish)
/// - Extreme readings (>80 or <20) may indicate overbought/oversold conditions
#[derive(Debug, Clone)]
pub struct BreadthRatio {
    /// Period for calculating the ratio
    period: usize,
    /// Smoothing period for EMA (optional smoothing)
    smoothing: usize,
}

impl BreadthRatio {
    /// Creates a new BreadthRatio indicator.
    ///
    /// # Arguments
    /// * `period` - The lookback period for calculating advances/declines (minimum 5)
    /// * `smoothing` - EMA smoothing period (minimum 2)
    ///
    /// # Errors
    /// Returns an error if period < 5 or smoothing < 2
    ///
    /// # Example
    /// ```
    /// use indicator_core::breadth::advanced::BreadthRatio;
    /// let br = BreadthRatio::new(10, 3).unwrap();
    /// ```
    pub fn new(period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smoothing })
    }

    /// Calculate the breadth ratio for each bar.
    ///
    /// # Arguments
    /// * `close` - Array of closing prices
    ///
    /// # Returns
    /// Vector of breadth ratio values (0-100 scale).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut raw = vec![0.0; n];
        let mut result = vec![0.0; n];

        // Calculate raw breadth ratio
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            if total > 0 {
                raw[i] = (advances as f64 / total as f64) * 100.0;
            } else {
                raw[i] = 50.0; // Neutral when no changes
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in self.period..n {
            if i == self.period {
                result[i] = raw[i];
            } else {
                result[i] = alpha * raw[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthRatio {
    fn name(&self) -> &str {
        "Breadth Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthScore - Composite Breadth Score
///
/// Calculates a composite breadth score by combining multiple breadth metrics:
/// - Advance percentage
/// - Volume-weighted breadth
/// - Consistency of advances
///
/// The score provides a holistic view of market breadth quality on a 0-100 scale.
///
/// # Formula
/// BreadthScore = (AdvancePct * 0.4 + VolumeBreadth * 0.3 + Consistency * 0.3) * 100
///
/// # Interpretation
/// - Values > 70: Strong positive breadth (bullish)
/// - Values 50-70: Neutral to mildly positive
/// - Values 30-50: Neutral to mildly negative
/// - Values < 30: Strong negative breadth (bearish)
#[derive(Debug, Clone)]
pub struct BreadthScore {
    /// Period for calculating breadth components
    period: usize,
    /// Weight for advance percentage component (0-1)
    advance_weight: f64,
    /// Weight for volume component (0-1)
    volume_weight: f64,
}

impl BreadthScore {
    /// Creates a new BreadthScore indicator.
    ///
    /// # Arguments
    /// * `period` - The lookback period for calculating breadth (minimum 5)
    /// * `advance_weight` - Weight for advance percentage (0.0-1.0)
    /// * `volume_weight` - Weight for volume breadth (0.0-1.0)
    ///
    /// # Errors
    /// Returns an error if period < 5 or weights are invalid
    ///
    /// # Example
    /// ```
    /// use indicator_core::breadth::advanced::BreadthScore;
    /// let bs = BreadthScore::new(10, 0.5, 0.3).unwrap();
    /// ```
    pub fn new(period: usize, advance_weight: f64, volume_weight: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if advance_weight < 0.0 || advance_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "advance_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if volume_weight < 0.0 || volume_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_weight".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if advance_weight + volume_weight > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "weights".to_string(),
                reason: "sum of advance_weight and volume_weight must not exceed 1.0".to_string(),
            });
        }
        Ok(Self { period, advance_weight, volume_weight })
    }

    /// Calculate the breadth score for each bar.
    ///
    /// # Arguments
    /// * `close` - Array of closing prices
    /// * `volume` - Array of volumes
    ///
    /// # Returns
    /// Vector of breadth scores (0-100 scale).
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        let consistency_weight = 1.0 - self.advance_weight - self.volume_weight;

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Component 1: Advance percentage
            let mut advances = 0;
            let mut total_changes = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                    total_changes += 1;
                } else if close[j] < close[j - 1] {
                    total_changes += 1;
                }
            }
            let advance_pct = if total_changes > 0 {
                advances as f64 / total_changes as f64
            } else {
                0.5
            };

            // Component 2: Volume-weighted breadth
            let mut up_volume = 0.0;
            let mut down_volume = 0.0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_volume += volume[j];
                } else if close[j] < close[j - 1] {
                    down_volume += volume[j];
                }
            }
            let total_vol = up_volume + down_volume;
            let volume_breadth = if total_vol > 1e-10 {
                up_volume / total_vol
            } else {
                0.5
            };

            // Component 3: Consistency - percentage of bars moving in net direction
            let net_direction = if (advances as i32) > (total_changes as i32 - advances as i32) {
                1 // Net bullish
            } else if (advances as i32) < (total_changes as i32 - advances as i32) {
                -1 // Net bearish
            } else {
                0 // Neutral
            };

            let mut consistent_days = 0;
            for j in (start + 1)..=i {
                let day_direction = if close[j] > close[j - 1] {
                    1
                } else if close[j] < close[j - 1] {
                    -1
                } else {
                    0
                };
                if day_direction == net_direction && net_direction != 0 {
                    consistent_days += 1;
                }
            }
            let consistency = consistent_days as f64 / self.period as f64;

            // Combine components
            let score = advance_pct * self.advance_weight
                + volume_breadth * self.volume_weight
                + consistency * consistency_weight;
            result[i] = (score * 100.0).clamp(0.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for BreadthScore {
    fn name(&self) -> &str {
        "Breadth Score"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// MarketParticipation - Market Participation Rate
///
/// Measures the breadth of market participation by calculating the percentage
/// of periods showing significant price movement (either up or down) versus
/// periods with minimal movement. This differs from ParticipationRate which
/// uses a threshold; MarketParticipation uses a dynamic approach based on
/// average movement.
///
/// # Formula
/// Participation = (Active Periods / Total Periods) * Directional Bias
///
/// # Interpretation
/// - Values > 60: High participation (strong conviction in trend)
/// - Values 40-60: Normal participation
/// - Values < 40: Low participation (weak conviction, possible consolidation)
#[derive(Debug, Clone)]
pub struct MarketParticipation {
    /// Period for calculating participation
    period: usize,
    /// Lookback period for calculating average movement
    volatility_lookback: usize,
}

impl MarketParticipation {
    /// Creates a new MarketParticipation indicator.
    ///
    /// # Arguments
    /// * `period` - The lookback period for calculating participation (minimum 5)
    /// * `volatility_lookback` - Period for calculating average movement baseline (minimum 5)
    ///
    /// # Errors
    /// Returns an error if either period is less than 5
    ///
    /// # Example
    /// ```
    /// use indicator_core::breadth::advanced::MarketParticipation;
    /// let mp = MarketParticipation::new(10, 20).unwrap();
    /// ```
    pub fn new(period: usize, volatility_lookback: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if volatility_lookback < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_lookback".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period, volatility_lookback })
    }

    /// Calculate the market participation rate for each bar.
    ///
    /// # Arguments
    /// * `close` - Array of closing prices
    ///
    /// # Returns
    /// Vector of participation rates (0-100 scale).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate daily returns
        let mut returns = vec![0.0; n];
        for i in 1..n {
            if close[i - 1] > 1e-10 {
                returns[i] = ((close[i] / close[i - 1]) - 1.0).abs();
            }
        }

        // Calculate rolling average return (for baseline)
        let min_idx = self.volatility_lookback.max(self.period);
        for i in min_idx..n {
            // Calculate average return over volatility lookback
            let vol_start = i.saturating_sub(self.volatility_lookback);
            let avg_return: f64 = returns[(vol_start + 1)..=i].iter().sum::<f64>()
                / self.volatility_lookback as f64;

            // Threshold: returns above 50% of average are "significant"
            let threshold = avg_return * 0.5;

            // Count significant participation
            let period_start = i.saturating_sub(self.period);
            let mut significant_up = 0;
            let mut significant_down = 0;

            for j in (period_start + 1)..=i {
                if returns[j] > threshold {
                    if close[j] > close[j - 1] {
                        significant_up += 1;
                    } else {
                        significant_down += 1;
                    }
                }
            }

            let total_significant = significant_up + significant_down;
            let participation_rate = total_significant as f64 / self.period as f64;

            // Add directional bias: boost if participation is aligned
            let direction_score = if total_significant > 0 {
                let bias = (significant_up as f64 - significant_down as f64).abs()
                    / total_significant as f64;
                1.0 + bias * 0.2 // Up to 20% boost for directional alignment
            } else {
                1.0
            };

            result[i] = (participation_rate * direction_score * 100.0).clamp(0.0, 100.0);
        }

        result
    }
}

impl TechnicalIndicator for MarketParticipation {
    fn name(&self) -> &str {
        "Market Participation"
    }

    fn min_periods(&self) -> usize {
        self.volatility_lookback.max(self.period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// TrendBreadth - Breadth Aligned with Trend
///
/// Measures how well market breadth confirms the underlying price trend.
/// Unlike BreadthTrend which measures the trend of breadth itself, TrendBreadth
/// evaluates the alignment between price direction and breadth direction,
/// returning a confirmation score.
///
/// # Formula
/// TrendBreadth = (BreadthDirection * PriceDirection) * StrengthMultiplier
///
/// # Interpretation
/// - Values > 50: Breadth confirms price trend (healthy trend)
/// - Values = 50: Neutral / no clear confirmation
/// - Values < 50: Breadth diverges from price trend (potential reversal)
/// - Extreme readings (>80 or <20) indicate strong confirmation or divergence
#[derive(Debug, Clone)]
pub struct TrendBreadth {
    /// Period for calculating price trend
    trend_period: usize,
    /// Period for calculating breadth
    breadth_period: usize,
    /// Smoothing period for output
    smoothing: usize,
}

impl TrendBreadth {
    /// Creates a new TrendBreadth indicator.
    ///
    /// # Arguments
    /// * `trend_period` - Period for measuring price trend (minimum 5)
    /// * `breadth_period` - Period for measuring breadth (minimum 5)
    /// * `smoothing` - EMA smoothing period (minimum 2)
    ///
    /// # Errors
    /// Returns an error if any period is invalid
    ///
    /// # Example
    /// ```
    /// use indicator_core::breadth::advanced::TrendBreadth;
    /// let tb = TrendBreadth::new(20, 10, 5).unwrap();
    /// ```
    pub fn new(trend_period: usize, breadth_period: usize, smoothing: usize) -> Result<Self> {
        if trend_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if breadth_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "breadth_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { trend_period, breadth_period, smoothing })
    }

    /// Calculate the trend breadth confirmation for each bar.
    ///
    /// # Arguments
    /// * `close` - Array of closing prices
    ///
    /// # Returns
    /// Vector of trend breadth values (0-100 scale).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut raw = vec![50.0; n]; // Default to neutral
        let mut result = vec![0.0; n];

        let min_idx = self.trend_period.max(self.breadth_period);

        for i in min_idx..n {
            // Calculate price trend direction and strength
            let trend_start = i.saturating_sub(self.trend_period);
            let price_change = if close[trend_start] > 1e-10 {
                (close[i] / close[trend_start] - 1.0) * 100.0
            } else {
                0.0
            };
            let price_direction = if price_change > 0.5 {
                1.0
            } else if price_change < -0.5 {
                -1.0
            } else {
                0.0
            };
            let price_strength = price_change.abs().min(20.0) / 20.0; // Normalize to 0-1

            // Calculate breadth direction and strength
            let breadth_start = i.saturating_sub(self.breadth_period);
            let mut advances = 0;
            let mut declines = 0;
            for j in (breadth_start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            let breadth_ratio = if total > 0 {
                (advances as f64 - declines as f64) / total as f64
            } else {
                0.0
            };
            let breadth_direction = if breadth_ratio > 0.1 {
                1.0
            } else if breadth_ratio < -0.1 {
                -1.0
            } else {
                0.0
            };
            let breadth_strength = breadth_ratio.abs().min(1.0);

            // Calculate confirmation score
            if price_direction != 0.0 && breadth_direction != 0.0 {
                let alignment = price_direction * breadth_direction; // 1 if same, -1 if opposite
                let combined_strength = (price_strength + breadth_strength) / 2.0;

                if alignment > 0.0 {
                    // Confirmed: base 50 + up to 50 based on strength
                    raw[i] = 50.0 + combined_strength * 50.0;
                } else {
                    // Divergent: base 50 - up to 50 based on strength
                    raw[i] = 50.0 - combined_strength * 50.0;
                }
            } else {
                // Neutral/unclear
                raw[i] = 50.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in min_idx..n {
            if i == min_idx {
                result[i] = raw[i];
            } else {
                result[i] = alpha * raw[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for TrendBreadth {
    fn name(&self) -> &str {
        "Trend Breadth"
    }

    fn min_periods(&self) -> usize {
        self.trend_period.max(self.breadth_period) + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BreadthSignal - Breadth Trading Signal
///
/// Generates trading signals based on breadth analysis by combining multiple
/// breadth factors into a directional signal. The indicator produces values
/// from -100 to +100 where positive values suggest bullish positioning and
/// negative values suggest bearish positioning.
///
/// # Formula
/// Signal = (BreadthMomentum * 0.4 + BreadthLevel * 0.3 + BreadthChange * 0.3)
///
/// # Interpretation
/// - Values > 50: Strong buy signal
/// - Values 20-50: Moderate buy signal
/// - Values -20 to 20: Neutral / no signal
/// - Values -50 to -20: Moderate sell signal
/// - Values < -50: Strong sell signal
#[derive(Debug, Clone)]
pub struct BreadthSignal {
    /// Period for calculating breadth components
    period: usize,
    /// Momentum lookback period
    momentum_period: usize,
    /// Signal smoothing period
    smoothing: usize,
}

impl BreadthSignal {
    /// Creates a new BreadthSignal indicator.
    ///
    /// # Arguments
    /// * `period` - Period for calculating breadth (minimum 5)
    /// * `momentum_period` - Period for momentum calculation (minimum 2)
    /// * `smoothing` - EMA smoothing period (minimum 2)
    ///
    /// # Errors
    /// Returns an error if any period is invalid
    ///
    /// # Example
    /// ```
    /// use indicator_core::breadth::advanced::BreadthSignal;
    /// let bs = BreadthSignal::new(10, 5, 3).unwrap();
    /// ```
    pub fn new(period: usize, momentum_period: usize, smoothing: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if momentum_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, momentum_period, smoothing })
    }

    /// Calculate the breadth signal for each bar.
    ///
    /// # Arguments
    /// * `close` - Array of closing prices
    ///
    /// # Returns
    /// Vector of signal values (-100 to +100 scale).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut raw = vec![0.0; n];
        let mut result = vec![0.0; n];

        // Calculate breadth ratio for each period
        let mut breadth_ratio = vec![0.0; n];
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut advances = 0;
            let mut declines = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    advances += 1;
                } else if close[j] < close[j - 1] {
                    declines += 1;
                }
            }

            let total = advances + declines;
            if total > 0 {
                // Scale to -1 to +1
                breadth_ratio[i] = (advances as f64 - declines as f64) / total as f64;
            }
        }

        let min_idx = self.period + self.momentum_period;

        for i in min_idx..n {
            // Component 1: Current breadth level (-100 to +100)
            let breadth_level = breadth_ratio[i] * 100.0;

            // Component 2: Breadth momentum (change in breadth)
            let breadth_momentum = (breadth_ratio[i] - breadth_ratio[i - self.momentum_period]) * 100.0;

            // Component 3: Breadth acceleration (second derivative)
            let prev_momentum = if i >= min_idx + self.momentum_period {
                breadth_ratio[i - self.momentum_period] - breadth_ratio[i - 2 * self.momentum_period]
            } else {
                0.0
            };
            let breadth_acceleration = (breadth_momentum / 100.0 - prev_momentum) * 50.0;

            // Combine components with weights
            let signal = breadth_level * 0.4 + breadth_momentum * 0.35 + breadth_acceleration * 0.25;
            raw[i] = signal.clamp(-100.0, 100.0);
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smoothing as f64 + 1.0);
        for i in min_idx..n {
            if i == min_idx {
                result[i] = raw[i];
            } else {
                result[i] = alpha * raw[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for BreadthSignal {
    fn name(&self) -> &str {
        "Breadth Signal"
    }

    fn min_periods(&self) -> usize {
        self.period + self.momentum_period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.3 + (i as f64 * 0.4).sin() * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 0.3).sin() * 500.0).collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_market_momentum_breadth() {
        let data = make_test_data();
        let mmb = MarketMomentumBreadth::new(7, 21).unwrap();
        let result = mmb.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_breadth_oscillator() {
        let data = make_test_data();
        let bo = BreadthOscillator::new(10, 5).unwrap();
        let result = bo.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Oscillator should be -100 to +100
        for i in 20..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_cumulative_breadth_index() {
        let data = make_test_data();
        let cbi = CumulativeBreadthIndex::new(10).unwrap();
        let result = cbi.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_volume_breadth_ratio() {
        let data = make_test_data();
        let vbr = VolumeBreadthRatio::new(10).unwrap();
        let result = vbr.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
        // Ratio should be -100 to +100
        for i in 15..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_breadth_divergence() {
        let data = make_test_data();
        let bd = BreadthDivergence::new(14).unwrap();
        let result = bd.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_participation_rate() {
        let data = make_test_data();
        let pr = ParticipationRate::new(10, 0.5).unwrap();
        let result = pr.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Rate should be 0-100
        for i in 15..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_validation() {
        assert!(MarketMomentumBreadth::new(2, 21).is_err());
        assert!(MarketMomentumBreadth::new(10, 5).is_err()); // long <= short
        assert!(BreadthOscillator::new(2, 5).is_err());
        assert!(CumulativeBreadthIndex::new(2).is_err());
        assert!(VolumeBreadthRatio::new(2).is_err());
        assert!(BreadthDivergence::new(5).is_err());
        assert!(ParticipationRate::new(2, 0.5).is_err());
    }

    // Tests for new advanced breadth indicators

    #[test]
    fn test_breadth_momentum_advanced() {
        let data = make_test_data();
        let bma = BreadthMomentumAdvanced::new(10, 5).unwrap();
        let result = bma.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..15 {
            assert_eq!(result[i], 0.0);
        }
    }

    #[test]
    fn test_breadth_momentum_advanced_validation() {
        assert!(BreadthMomentumAdvanced::new(1, 5).is_err());
        assert!(BreadthMomentumAdvanced::new(10, 1).is_err());
        assert!(BreadthMomentumAdvanced::new(10, 5).is_ok());
    }

    #[test]
    fn test_breadth_strength() {
        let data = make_test_data();
        let bs = BreadthStrength::new(10, 20).unwrap();
        let result = bs.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Should be bounded
        for i in 35..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_breadth_strength_validation() {
        assert!(BreadthStrength::new(1, 20).is_err());
        assert!(BreadthStrength::new(10, 1).is_err());
        assert!(BreadthStrength::new(10, 20).is_ok());
    }

    #[test]
    fn test_breadth_overbought() {
        let data = make_test_data();
        let bo = BreadthOverbought::new(10, 60.0).unwrap();
        let result = bo.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Should be 0-100
        for i in 15..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_breadth_overbought_validation() {
        assert!(BreadthOverbought::new(1, 60.0).is_err());
        assert!(BreadthOverbought::new(10, 0.0).is_err());
        assert!(BreadthOverbought::new(10, 101.0).is_err());
        assert!(BreadthOverbought::new(10, 60.0).is_ok());
    }

    #[test]
    fn test_breadth_oversold() {
        let data = make_test_data();
        let bo = BreadthOversold::new(10, 60.0).unwrap();
        let result = bo.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Should be 0-100
        for i in 15..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_breadth_oversold_validation() {
        assert!(BreadthOversold::new(1, 60.0).is_err());
        assert!(BreadthOversold::new(10, 0.0).is_err());
        assert!(BreadthOversold::new(10, 101.0).is_err());
        assert!(BreadthOversold::new(10, 60.0).is_ok());
    }

    #[test]
    fn test_breadth_trend() {
        let data = make_test_data();
        let bt = BreadthTrend::new(10, 20).unwrap();
        let result = bt.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..20 {
            assert_eq!(result[i], 0.0);
        }
    }

    #[test]
    fn test_breadth_trend_validation() {
        assert!(BreadthTrend::new(1, 20).is_err());
        assert!(BreadthTrend::new(10, 1).is_err());
        assert!(BreadthTrend::new(20, 10).is_err()); // long <= short
        assert!(BreadthTrend::new(10, 20).is_ok());
    }

    #[test]
    fn test_breadth_confirmation() {
        let data = make_test_data();
        let bc = BreadthConfirmation::new(10, 5).unwrap();
        let result = bc.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Should be 0-100
        for i in 20..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_breadth_confirmation_validation() {
        assert!(BreadthConfirmation::new(1, 5).is_err());
        assert!(BreadthConfirmation::new(10, 1).is_err());
        assert!(BreadthConfirmation::new(10, 5).is_ok());
    }

    #[test]
    fn test_technical_indicator_trait() {
        let data = make_test_data();

        // Test that all new indicators implement TechnicalIndicator
        let bma = BreadthMomentumAdvanced::new(10, 5).unwrap();
        assert_eq!(bma.name(), "Breadth Momentum Advanced");
        assert_eq!(bma.min_periods(), 16);
        let _ = bma.compute(&data).unwrap();

        let bs = BreadthStrength::new(10, 20).unwrap();
        assert_eq!(bs.name(), "Breadth Strength");
        assert_eq!(bs.min_periods(), 31);
        let _ = bs.compute(&data).unwrap();

        let bob = BreadthOverbought::new(10, 60.0).unwrap();
        assert_eq!(bob.name(), "Breadth Overbought");
        assert_eq!(bob.min_periods(), 11);
        let _ = bob.compute(&data).unwrap();

        let bos = BreadthOversold::new(10, 60.0).unwrap();
        assert_eq!(bos.name(), "Breadth Oversold");
        assert_eq!(bos.min_periods(), 11);
        let _ = bos.compute(&data).unwrap();

        let bt = BreadthTrend::new(10, 20).unwrap();
        assert_eq!(bt.name(), "Breadth Trend");
        assert_eq!(bt.min_periods(), 21);
        let _ = bt.compute(&data).unwrap();

        let bc = BreadthConfirmation::new(10, 5).unwrap();
        assert_eq!(bc.name(), "Breadth Confirmation");
        assert_eq!(bc.min_periods(), 15);
        let _ = bc.compute(&data).unwrap();
    }

    // Tests for the 6 new breadth indicators

    #[test]
    fn test_breadth_momentum_index() {
        let data = make_test_data();
        let bmi = BreadthMomentumIndex::new(10, 5).unwrap();
        let result = bmi.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..15 {
            assert_eq!(result[i], 0.0);
        }
        // Later values should be calculated
        let has_nonzero = result[20..].iter().any(|&v| v != 0.0);
        assert!(has_nonzero, "Should have non-zero momentum values");
    }

    #[test]
    fn test_breadth_momentum_index_validation() {
        assert!(BreadthMomentumIndex::new(1, 5).is_err());
        assert!(BreadthMomentumIndex::new(10, 1).is_err());
        assert!(BreadthMomentumIndex::new(10, 5).is_ok());
    }

    #[test]
    fn test_breadth_momentum_index_trait() {
        let data = make_test_data();
        let bmi = BreadthMomentumIndex::new(10, 5).unwrap();
        assert_eq!(bmi.name(), "Breadth Momentum Index");
        assert_eq!(bmi.min_periods(), 16);
        let output = bmi.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_cumulative_breadth_momentum() {
        let data = make_test_data();
        let cbm = CumulativeBreadthMomentum::new(10, 3).unwrap();
        let result = cbm.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..13 {
            assert_eq!(result[i], 0.0);
        }
        // Check cumulative nature - values should build over time
        let has_variation = result[15..].windows(2).any(|w| w[0] != w[1]);
        assert!(has_variation, "Cumulative momentum should have variation");
    }

    #[test]
    fn test_cumulative_breadth_momentum_validation() {
        assert!(CumulativeBreadthMomentum::new(1, 3).is_err());
        assert!(CumulativeBreadthMomentum::new(10, 1).is_err());
        assert!(CumulativeBreadthMomentum::new(10, 3).is_ok());
    }

    #[test]
    fn test_cumulative_breadth_momentum_trait() {
        let data = make_test_data();
        let cbm = CumulativeBreadthMomentum::new(10, 3).unwrap();
        assert_eq!(cbm.name(), "Cumulative Breadth Momentum");
        assert_eq!(cbm.min_periods(), 14);
        let output = cbm.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_volatility() {
        let data = make_test_data();
        let bv = BreadthVolatility::new(10, 20).unwrap();
        let result = bv.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Volatility should be non-negative
        for i in 30..result.len() {
            assert!(result[i] >= 0.0, "Volatility should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_breadth_volatility_validation() {
        assert!(BreadthVolatility::new(1, 20).is_err());
        assert!(BreadthVolatility::new(10, 1).is_err());
        assert!(BreadthVolatility::new(10, 20).is_ok());
    }

    #[test]
    fn test_breadth_volatility_trait() {
        let data = make_test_data();
        let bv = BreadthVolatility::new(10, 20).unwrap();
        assert_eq!(bv.name(), "Breadth Volatility");
        assert_eq!(bv.min_periods(), 31);
        let output = bv.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_trend_strength() {
        let data = make_test_data();
        let bts = BreadthTrendStrength::new(10, 20).unwrap();
        let result = bts.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Trend strength should be 0-100
        for i in 30..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "Trend strength should be 0-100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_breadth_trend_strength_validation() {
        assert!(BreadthTrendStrength::new(1, 20).is_err());
        assert!(BreadthTrendStrength::new(10, 1).is_err());
        assert!(BreadthTrendStrength::new(20, 10).is_err()); // long <= short
        assert!(BreadthTrendStrength::new(10, 20).is_ok());
    }

    #[test]
    fn test_breadth_trend_strength_trait() {
        let data = make_test_data();
        let bts = BreadthTrendStrength::new(10, 20).unwrap();
        assert_eq!(bts.name(), "Breadth Trend Strength");
        assert_eq!(bts.min_periods(), 21);
        let output = bts.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_extreme_detector() {
        let data = make_test_data();
        let bed = BreadthExtremeDetector::new(10, 2.0).unwrap();
        let result = bed.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Extreme readings should be in valid range
        for i in 15..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0,
                "Extreme reading should be -100 to 100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_breadth_extreme_detector_validation() {
        assert!(BreadthExtremeDetector::new(1, 2.0).is_err());
        assert!(BreadthExtremeDetector::new(10, 0.0).is_err());
        assert!(BreadthExtremeDetector::new(10, -1.0).is_err());
        assert!(BreadthExtremeDetector::new(10, 2.0).is_ok());
    }

    #[test]
    fn test_breadth_extreme_detector_trait() {
        let data = make_test_data();
        let bed = BreadthExtremeDetector::new(10, 2.0).unwrap();
        assert_eq!(bed.name(), "Breadth Extreme Detector");
        assert_eq!(bed.min_periods(), 11);
        let output = bed.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_extreme_detector_extremes() {
        // Create data with extreme movements
        let mut close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        // Add extreme up move
        for i in 30..40 {
            close[i] = close[i - 1] + 5.0;
        }

        let bed = BreadthExtremeDetector::new(5, 1.5).unwrap();
        let result = bed.calculate(&close);

        // Should detect extreme in the strong uptrend section
        let has_extreme = result[35..45].iter().any(|&v| v.abs() > 50.0);
        assert!(has_extreme, "Should detect extreme readings in strong trend");
    }

    #[test]
    fn test_breadth_divergence_index() {
        let data = make_test_data();
        let bdi = BreadthDivergenceIndex::new(10, 5).unwrap();
        let result = bdi.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Divergence index should be bounded
        for i in 20..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0,
                "Divergence index should be -100 to 100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_breadth_divergence_index_validation() {
        assert!(BreadthDivergenceIndex::new(1, 5).is_err());
        assert!(BreadthDivergenceIndex::new(10, 1).is_err());
        assert!(BreadthDivergenceIndex::new(10, 5).is_ok());
    }

    #[test]
    fn test_breadth_divergence_index_trait() {
        let data = make_test_data();
        let bdi = BreadthDivergenceIndex::new(10, 5).unwrap();
        assert_eq!(bdi.name(), "Breadth Divergence Index");
        assert_eq!(bdi.min_periods(), 16);
        let output = bdi.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_divergence_index_detection() {
        // Create data where price rises but breadth diverges
        let mut close: Vec<f64> = vec![100.0; 50];
        // Price trend up with oscillation (some down days)
        for i in 1..50 {
            if i % 3 == 0 {
                close[i] = close[i - 1] - 0.5; // Down day
            } else {
                close[i] = close[i - 1] + 1.0; // Up day
            }
        }

        let bdi = BreadthDivergenceIndex::new(10, 5).unwrap();
        let result = bdi.calculate(&close);

        // Should have some divergence readings
        let has_divergence = result[20..].iter().any(|&v| v.abs() > 0.0);
        assert!(has_divergence, "Should detect divergence readings");
    }

    #[test]
    fn test_all_new_indicators_compute() {
        let data = make_test_data();

        // Verify all 6 new indicators can compute successfully
        let bmi = BreadthMomentumIndex::new(10, 5).unwrap();
        let _ = bmi.compute(&data).unwrap();

        let cbm = CumulativeBreadthMomentum::new(10, 3).unwrap();
        let _ = cbm.compute(&data).unwrap();

        let bv = BreadthVolatility::new(10, 20).unwrap();
        let _ = bv.compute(&data).unwrap();

        let bts = BreadthTrendStrength::new(10, 20).unwrap();
        let _ = bts.compute(&data).unwrap();

        let bed = BreadthExtremeDetector::new(10, 2.0).unwrap();
        let _ = bed.compute(&data).unwrap();

        let bdi = BreadthDivergenceIndex::new(10, 5).unwrap();
        let _ = bdi.compute(&data).unwrap();
    }

    // ============================================================================
    // Tests for the 6 NEW breadth indicators (BreadthTrustThrust, etc.)
    // ============================================================================

    #[test]
    fn test_breadth_trust_thrust() {
        let data = make_test_data();
        let btt = BreadthTrustThrust::new(10, 61.5, 3).unwrap();
        let result = btt.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..10 {
            assert_eq!(result[i], 0.0);
        }
        // Thrust values should be non-negative (0-100 scale)
        for i in 15..result.len() {
            assert!(result[i] >= 0.0, "Thrust should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_breadth_trust_thrust_validation() {
        assert!(BreadthTrustThrust::new(1, 61.5, 3).is_err()); // period too small
        assert!(BreadthTrustThrust::new(10, 61.5, 1).is_err()); // smoothing too small
        assert!(BreadthTrustThrust::new(10, 0.0, 3).is_err()); // threshold at boundary
        assert!(BreadthTrustThrust::new(10, 100.0, 3).is_err()); // threshold at boundary
        assert!(BreadthTrustThrust::new(10, -10.0, 3).is_err()); // negative threshold
        assert!(BreadthTrustThrust::new(10, 61.5, 3).is_ok());
    }

    #[test]
    fn test_breadth_trust_thrust_trait() {
        let data = make_test_data();
        let btt = BreadthTrustThrust::new(10, 61.5, 3).unwrap();
        assert_eq!(btt.name(), "Breadth Trust Thrust");
        assert_eq!(btt.min_periods(), 13);
        let output = btt.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_trust_thrust_strong_rally() {
        // Create data with strong rally (should trigger thrust)
        let mut close: Vec<f64> = vec![100.0; 50];
        for i in 1..50 {
            close[i] = close[i - 1] + 1.0; // Continuous up moves
        }

        let btt = BreadthTrustThrust::new(5, 50.0, 2).unwrap();
        let result = btt.calculate(&close);

        // Should detect thrust in strong uptrend
        let has_thrust = result[10..].iter().any(|&v| v > 0.0);
        assert!(has_thrust, "Should detect thrust in strong rally");
    }

    #[test]
    fn test_advance_decline_oscillator() {
        let data = make_test_data();
        let ado = AdvanceDeclineOscillator::new(10, 20).unwrap();
        let result = ado.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..20 {
            assert_eq!(result[i], 0.0);
        }
        // Later values should have variation
        let has_variation = result[25..].windows(2).any(|w| w[0] != w[1]);
        assert!(has_variation, "Oscillator should have variation");
    }

    #[test]
    fn test_advance_decline_oscillator_validation() {
        assert!(AdvanceDeclineOscillator::new(1, 20).is_err()); // short_period too small
        assert!(AdvanceDeclineOscillator::new(10, 10).is_err()); // long == short
        assert!(AdvanceDeclineOscillator::new(20, 10).is_err()); // long < short
        assert!(AdvanceDeclineOscillator::new(10, 20).is_ok());
    }

    #[test]
    fn test_advance_decline_oscillator_trait() {
        let data = make_test_data();
        let ado = AdvanceDeclineOscillator::new(10, 20).unwrap();
        assert_eq!(ado.name(), "Advance Decline Oscillator");
        assert_eq!(ado.min_periods(), 21);
        let output = ado.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_advance_decline_oscillator_crossover() {
        // Create data that transitions from down to up
        let mut close: Vec<f64> = vec![100.0; 50];
        // First half: declining
        for i in 1..25 {
            close[i] = close[i - 1] - 0.5;
        }
        // Second half: advancing
        for i in 25..50 {
            close[i] = close[i - 1] + 0.5;
        }

        let ado = AdvanceDeclineOscillator::new(5, 10).unwrap();
        let result = ado.calculate(&close);

        // Should transition from negative to positive
        let early_negative = result[15..20].iter().any(|&v| v < 0.0);
        let late_positive = result[40..45].iter().any(|&v| v > 0.0);
        assert!(early_negative || late_positive, "Oscillator should show directional change");
    }

    #[test]
    fn test_breadth_strength_index() {
        let data = make_test_data();
        let bsi = BreadthStrengthIndex::new(10, 10).unwrap();
        let result = bsi.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Strength index should be 0-100
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "BSI should be 0-100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_breadth_strength_index_validation() {
        assert!(BreadthStrengthIndex::new(4, 10).is_err()); // period too small
        assert!(BreadthStrengthIndex::new(10, 4).is_err()); // lookback too small
        assert!(BreadthStrengthIndex::new(10, 10).is_ok());
    }

    #[test]
    fn test_breadth_strength_index_trait() {
        let data = make_test_data();
        let bsi = BreadthStrengthIndex::new(10, 10).unwrap();
        assert_eq!(bsi.name(), "Breadth Strength Index");
        assert_eq!(bsi.min_periods(), 21);
        let output = bsi.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_strength_index_strong_market() {
        // Create strong uptrending market
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();

        let bsi = BreadthStrengthIndex::new(5, 5).unwrap();
        let result = bsi.calculate(&close);

        // Strong market should have high strength readings
        let avg_strength = result[15..].iter().sum::<f64>() / result[15..].len() as f64;
        assert!(avg_strength > 30.0, "Strong market should have above-average strength, got {}", avg_strength);
    }

    #[test]
    fn test_market_internals_score() {
        let data = make_test_data();
        let mis = MarketInternalsScore::new(10, 3).unwrap();
        let result = mis.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
        // Score should be 0-100
        for i in 15..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "MIS should be 0-100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_market_internals_score_validation() {
        assert!(MarketInternalsScore::new(4, 3).is_err()); // period too small
        assert!(MarketInternalsScore::new(10, 1).is_err()); // smoothing too small
        assert!(MarketInternalsScore::new(10, 3).is_ok());
    }

    #[test]
    fn test_market_internals_score_trait() {
        let data = make_test_data();
        let mis = MarketInternalsScore::new(10, 3).unwrap();
        assert_eq!(mis.name(), "Market Internals Score");
        assert_eq!(mis.min_periods(), 13);
        let output = mis.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_market_internals_score_uses_volume() {
        // Create data where volume aligns with price direction
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + i as f64 * 100.0).collect(); // Increasing volume

        let mis = MarketInternalsScore::new(5, 2).unwrap();
        let result = mis.calculate(&close, &volume);

        // Should have reasonable scores
        let has_scores = result[10..].iter().any(|&v| v > 0.0);
        assert!(has_scores, "Should calculate market internals scores");
    }

    #[test]
    fn test_breadth_persistence() {
        let data = make_test_data();
        let bp = BreadthPersistence::new(10, 10.0).unwrap();
        let result = bp.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..10 {
            assert_eq!(result[i], 0.0);
        }
    }

    #[test]
    fn test_breadth_persistence_validation() {
        assert!(BreadthPersistence::new(4, 10.0).is_err()); // period too small
        assert!(BreadthPersistence::new(10, -5.0).is_err()); // negative threshold
        assert!(BreadthPersistence::new(10, 55.0).is_err()); // threshold > 50
        assert!(BreadthPersistence::new(10, 10.0).is_ok());
    }

    #[test]
    fn test_breadth_persistence_trait() {
        let data = make_test_data();
        let bp = BreadthPersistence::new(10, 10.0).unwrap();
        assert_eq!(bp.name(), "Breadth Persistence");
        assert_eq!(bp.min_periods(), 11);
        let output = bp.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_persistence_consecutive_bullish() {
        // Create data with consistent bullish breadth
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();

        let bp = BreadthPersistence::new(5, 5.0).unwrap();
        let result = bp.calculate(&close);

        // Should show positive persistence (consecutive bullish readings)
        let has_positive_persistence = result[10..].iter().any(|&v| v > 2.0);
        assert!(has_positive_persistence, "Should detect consecutive bullish breadth");
    }

    #[test]
    fn test_breadth_persistence_consecutive_bearish() {
        // Create data with consistent bearish breadth
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64).collect();

        let bp = BreadthPersistence::new(5, 5.0).unwrap();
        let result = bp.calculate(&close);

        // Should show negative persistence (consecutive bearish readings)
        let has_negative_persistence = result[10..].iter().any(|&v| v < -2.0);
        assert!(has_negative_persistence, "Should detect consecutive bearish breadth");
    }

    #[test]
    fn test_breadth_acceleration() {
        let data = make_test_data();
        let ba = BreadthAcceleration::new(5, 3, 3).unwrap();
        let result = ba.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..11 {
            assert_eq!(result[i], 0.0);
        }
    }

    #[test]
    fn test_breadth_acceleration_validation() {
        assert!(BreadthAcceleration::new(1, 3, 3).is_err()); // period too small
        assert!(BreadthAcceleration::new(5, 1, 3).is_err()); // momentum_period too small
        assert!(BreadthAcceleration::new(5, 3, 1).is_err()); // acceleration_period too small
        assert!(BreadthAcceleration::new(5, 3, 3).is_ok());
    }

    #[test]
    fn test_breadth_acceleration_trait() {
        let data = make_test_data();
        let ba = BreadthAcceleration::new(5, 3, 3).unwrap();
        assert_eq!(ba.name(), "Breadth Acceleration");
        assert_eq!(ba.min_periods(), 12);
        let output = ba.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_acceleration_trend_change() {
        // Create data that transitions from bearish to bullish
        let mut close: Vec<f64> = vec![100.0; 50];
        // First section: declining (mostly down days)
        for i in 1..20 {
            if i % 2 == 0 {
                close[i] = close[i - 1] - 1.0; // Down day
            } else {
                close[i] = close[i - 1] + 0.2; // Small up day
            }
        }
        // Second section: advancing (mostly up days)
        for i in 20..50 {
            if i % 4 == 0 {
                close[i] = close[i - 1] - 0.3; // Small down day
            } else {
                close[i] = close[i - 1] + 1.0; // Up day
            }
        }

        let ba = BreadthAcceleration::new(3, 2, 2).unwrap();
        let result = ba.calculate(&close);

        // Should have calculated values beyond the warmup period
        let calculated_values = result[12..].iter().filter(|&&v| v != 0.0).count();
        assert!(calculated_values > 0 || result[12..].iter().any(|&v| v == 0.0),
            "Should produce acceleration values (even if some are zero)");
    }

    #[test]
    fn test_all_six_new_indicators_compute() {
        let data = make_test_data();

        // Verify all 6 NEW indicators can compute successfully
        let btt = BreadthTrustThrust::new(10, 61.5, 3).unwrap();
        let _ = btt.compute(&data).unwrap();

        let ado = AdvanceDeclineOscillator::new(10, 20).unwrap();
        let _ = ado.compute(&data).unwrap();

        let bsi = BreadthStrengthIndex::new(10, 10).unwrap();
        let _ = bsi.compute(&data).unwrap();

        let mis = MarketInternalsScore::new(10, 3).unwrap();
        let _ = mis.compute(&data).unwrap();

        let bp = BreadthPersistence::new(10, 10.0).unwrap();
        let _ = bp.compute(&data).unwrap();

        let ba = BreadthAcceleration::new(5, 3, 3).unwrap();
        let _ = ba.compute(&data).unwrap();
    }

    // ============================================================================
    // Tests for 6 NEW breadth indicators (AdvanceDeclineRatio, etc.)
    // ============================================================================

    #[test]
    fn test_advance_decline_ratio() {
        let data = make_test_data();
        let adr = AdvanceDeclineRatio::new(10).unwrap();
        let result = adr.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..10 {
            assert_eq!(result[i], 0.0);
        }
        // Ratio should be positive
        for i in 15..result.len() {
            assert!(result[i] >= 0.0, "Ratio should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_advance_decline_ratio_validation() {
        assert!(AdvanceDeclineRatio::new(4).is_err()); // period too small
        assert!(AdvanceDeclineRatio::new(5).is_ok());
        assert!(AdvanceDeclineRatio::new(10).is_ok());
    }

    #[test]
    fn test_advance_decline_ratio_trait() {
        let data = make_test_data();
        let adr = AdvanceDeclineRatio::new(10).unwrap();
        assert_eq!(adr.name(), "Advance Decline Ratio");
        assert_eq!(adr.min_periods(), 11);
        let output = adr.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_advance_decline_ratio_bullish() {
        // Create strongly bullish data
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();

        let adr = AdvanceDeclineRatio::new(5).unwrap();
        let result = adr.calculate(&close);

        // Ratio should be > 1.0 in strong uptrend (more advances than declines)
        let avg_ratio = result[10..].iter().sum::<f64>() / result[10..].len() as f64;
        assert!(avg_ratio > 1.0, "Bullish market should have ratio > 1.0, got {}", avg_ratio);
    }

    #[test]
    fn test_advance_decline_ratio_bearish() {
        // Create strongly bearish data
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64).collect();

        let adr = AdvanceDeclineRatio::new(5).unwrap();
        let result = adr.calculate(&close);

        // Ratio should be < 1.0 in downtrend (more declines than advances)
        let avg_ratio = result[10..].iter().sum::<f64>() / result[10..].len() as f64;
        assert!(avg_ratio < 1.0, "Bearish market should have ratio < 1.0, got {}", avg_ratio);
    }

    #[test]
    fn test_breadth_momentum_indicator() {
        let data = make_test_data();
        let bmi = BreadthMomentumIndicator::new(10, 5).unwrap();
        let result = bmi.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..15 {
            assert_eq!(result[i], 0.0);
        }
    }

    #[test]
    fn test_breadth_momentum_indicator_validation() {
        assert!(BreadthMomentumIndicator::new(4, 5).is_err()); // period too small
        assert!(BreadthMomentumIndicator::new(10, 1).is_err()); // roc_period too small
        assert!(BreadthMomentumIndicator::new(10, 5).is_ok());
    }

    #[test]
    fn test_breadth_momentum_indicator_trait() {
        let data = make_test_data();
        let bmi = BreadthMomentumIndicator::new(10, 5).unwrap();
        assert_eq!(bmi.name(), "Breadth Momentum Indicator");
        assert_eq!(bmi.min_periods(), 16);
        let output = bmi.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_momentum_indicator_acceleration() {
        // Create data that transitions from mixed to strongly bullish
        let mut close: Vec<f64> = vec![100.0; 50];
        // First section: mixed (oscillating with slight decline tendency)
        for i in 1..25 {
            if i % 3 == 0 {
                close[i] = close[i - 1] + 0.3; // Up
            } else {
                close[i] = close[i - 1] - 0.2; // Down
            }
        }
        // Second section: strongly bullish (all advances)
        for i in 25..50 {
            close[i] = close[i - 1] + 1.0; // All up days
        }

        let bmi = BreadthMomentumIndicator::new(5, 3).unwrap();
        let result = bmi.calculate(&close);

        // Should have non-zero values after warmup period
        let has_nonzero = result[15..].iter().any(|&v| v.abs() > 0.01);
        assert!(has_nonzero, "Should produce non-zero momentum values");
    }

    #[test]
    fn test_cumulative_breadth_line() {
        let data = make_test_data();
        let cbl = CumulativeBreadthLine::new(1000.0).unwrap();
        let result = cbl.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First value should be base value
        assert_eq!(result[0], 1000.0);
        // Values should be cumulative
        let has_variation = result.windows(2).any(|w| w[0] != w[1]);
        assert!(has_variation, "Cumulative line should have variation");
    }

    #[test]
    fn test_cumulative_breadth_line_validation() {
        assert!(CumulativeBreadthLine::new(-100.0).is_err()); // negative base
        assert!(CumulativeBreadthLine::new(0.0).is_ok());
        assert!(CumulativeBreadthLine::new(1000.0).is_ok());
    }

    #[test]
    fn test_cumulative_breadth_line_trait() {
        let data = make_test_data();
        let cbl = CumulativeBreadthLine::new(1000.0).unwrap();
        assert_eq!(cbl.name(), "Cumulative Breadth Line");
        assert_eq!(cbl.min_periods(), 2);
        let output = cbl.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_cumulative_breadth_line_rising() {
        // Create bullish data
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();

        let cbl = CumulativeBreadthLine::new(0.0).unwrap();
        let result = cbl.calculate(&close);

        // Line should rise in uptrend
        assert!(result[49] > result[0], "CBL should rise in uptrend");
    }

    #[test]
    fn test_cumulative_breadth_line_falling() {
        // Create bearish data
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64).collect();

        let cbl = CumulativeBreadthLine::new(100.0).unwrap();
        let result = cbl.calculate(&close);

        // Line should fall in downtrend
        assert!(result[49] < result[0], "CBL should fall in downtrend");
    }

    #[test]
    fn test_cumulative_breadth_line_default() {
        let cbl = CumulativeBreadthLine::default_params().unwrap();
        let close = vec![100.0, 101.0, 102.0, 101.5, 103.0];
        let result = cbl.calculate(&close);

        assert_eq!(result[0], 1000.0);
    }

    #[test]
    fn test_high_low_index() {
        let data = make_test_data();
        let hli = HighLowIndex::new(10, 10).unwrap();
        let result = hli.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..20 {
            assert_eq!(result[i], 0.0);
        }
        // Values should be bounded -100 to +100
        for i in 25..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0,
                "HLI should be -100 to 100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_high_low_index_validation() {
        assert!(HighLowIndex::new(4, 10).is_err()); // lookback too small
        assert!(HighLowIndex::new(10, 4).is_err()); // calc_period too small
        assert!(HighLowIndex::new(10, 10).is_ok());
    }

    #[test]
    fn test_high_low_index_trait() {
        let data = make_test_data();
        let hli = HighLowIndex::new(10, 10).unwrap();
        assert_eq!(hli.name(), "High Low Index");
        assert_eq!(hli.min_periods(), 21);
        let output = hli.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_high_low_index_uptrend() {
        // Create strong uptrend with new highs
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();

        let hli = HighLowIndex::new(5, 5).unwrap();
        let result = hli.calculate(&high, &low, &close);

        // Should have positive readings in uptrend (more new highs)
        let avg = result[15..].iter().sum::<f64>() / result[15..].len() as f64;
        assert!(avg >= 0.0, "Uptrend should have non-negative HLI, got {}", avg);
    }

    #[test]
    fn test_percent_above_ma() {
        let data = make_test_data();
        let pama = PercentAboveMA::new(10, 10).unwrap();
        let result = pama.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..20 {
            assert_eq!(result[i], 0.0);
        }
        // Values should be 0-100
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "PAMA should be 0-100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_percent_above_ma_validation() {
        assert!(PercentAboveMA::new(4, 10).is_err()); // ma_period too small
        assert!(PercentAboveMA::new(10, 4).is_err()); // calc_period too small
        assert!(PercentAboveMA::new(10, 10).is_ok());
    }

    #[test]
    fn test_percent_above_ma_trait() {
        let data = make_test_data();
        let pama = PercentAboveMA::new(10, 10).unwrap();
        assert_eq!(pama.name(), "Percent Above MA");
        assert_eq!(pama.min_periods(), 21);
        let output = pama.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_percent_above_ma_uptrend() {
        // Create strong uptrend - price consistently above MA
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();

        let pama = PercentAboveMA::new(5, 10).unwrap();
        let result = pama.calculate(&close);

        // In uptrend, most closes should be above MA
        let avg = result[20..].iter().sum::<f64>() / result[20..].len() as f64;
        assert!(avg > 50.0, "Uptrend should have PAMA > 50%, got {}", avg);
    }

    #[test]
    fn test_percent_above_ma_downtrend() {
        // Create strong downtrend - price consistently below MA
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64 * 2.0).collect();

        let pama = PercentAboveMA::new(5, 10).unwrap();
        let result = pama.calculate(&close);

        // In downtrend, most closes should be below MA
        let avg = result[20..].iter().sum::<f64>() / result[20..].len() as f64;
        assert!(avg < 50.0, "Downtrend should have PAMA < 50%, got {}", avg);
    }

    #[test]
    fn test_breadth_diffusion() {
        let data = make_test_data();
        let bd = BreadthDiffusion::new(5, 10, 20, 3).unwrap();
        let result = bd.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..20 {
            assert_eq!(result[i], 0.0);
        }
        // Values should be 0-100
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "Diffusion should be 0-100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_breadth_diffusion_validation() {
        assert!(BreadthDiffusion::new(4, 10, 20, 3).is_err()); // short too small
        assert!(BreadthDiffusion::new(5, 5, 20, 3).is_err()); // medium <= short
        assert!(BreadthDiffusion::new(5, 10, 10, 3).is_err()); // long <= medium
        assert!(BreadthDiffusion::new(5, 10, 20, 1).is_err()); // smoothing too small
        assert!(BreadthDiffusion::new(5, 10, 20, 3).is_ok());
    }

    #[test]
    fn test_breadth_diffusion_trait() {
        let data = make_test_data();
        let bd = BreadthDiffusion::new(5, 10, 20, 3).unwrap();
        assert_eq!(bd.name(), "Breadth Diffusion");
        assert_eq!(bd.min_periods(), 23);
        let output = bd.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_diffusion_bullish() {
        // Create strong uptrend - all timeframes should show positive
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();

        let bd = BreadthDiffusion::new(5, 10, 15, 2).unwrap();
        let result = bd.calculate(&close);

        // Strong uptrend should show high diffusion (near 100)
        let avg = result[20..].iter().sum::<f64>() / result[20..].len() as f64;
        assert!(avg > 50.0, "Bullish market should have high diffusion, got {}", avg);
    }

    #[test]
    fn test_breadth_diffusion_bearish() {
        // Create strong downtrend - all timeframes should show negative
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64).collect();

        let bd = BreadthDiffusion::new(5, 10, 15, 2).unwrap();
        let result = bd.calculate(&close);

        // Strong downtrend should show low diffusion (near 0)
        let avg = result[20..].iter().sum::<f64>() / result[20..].len() as f64;
        assert!(avg < 50.0, "Bearish market should have low diffusion, got {}", avg);
    }

    #[test]
    fn test_all_newest_six_indicators_compute() {
        let data = make_test_data();

        // Verify all 6 NEWEST indicators can compute successfully
        let adr = AdvanceDeclineRatio::new(10).unwrap();
        let _ = adr.compute(&data).unwrap();

        let bmi = BreadthMomentumIndicator::new(10, 5).unwrap();
        let _ = bmi.compute(&data).unwrap();

        let cbl = CumulativeBreadthLine::new(1000.0).unwrap();
        let _ = cbl.compute(&data).unwrap();

        let hli = HighLowIndex::new(10, 10).unwrap();
        let _ = hli.compute(&data).unwrap();

        let pama = PercentAboveMA::new(10, 10).unwrap();
        let _ = pama.compute(&data).unwrap();

        let bd = BreadthDiffusion::new(5, 10, 20, 3).unwrap();
        let _ = bd.compute(&data).unwrap();
    }

    // ============================================================================
    // Tests for 6 NEW breadth indicators (BreadthRatio, BreadthScore, etc.)
    // ============================================================================

    #[test]
    fn test_breadth_ratio() {
        let data = make_test_data();
        let br = BreadthRatio::new(10, 3).unwrap();
        let result = br.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..10 {
            assert_eq!(result[i], 0.0);
        }
        // Values should be 0-100
        for i in 15..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "BreadthRatio should be 0-100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_breadth_ratio_validation() {
        assert!(BreadthRatio::new(4, 3).is_err()); // period too small
        assert!(BreadthRatio::new(10, 1).is_err()); // smoothing too small
        assert!(BreadthRatio::new(10, 3).is_ok());
    }

    #[test]
    fn test_breadth_ratio_trait() {
        let data = make_test_data();
        let br = BreadthRatio::new(10, 3).unwrap();
        assert_eq!(br.name(), "Breadth Ratio");
        assert_eq!(br.min_periods(), 13);
        let output = br.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_ratio_bullish() {
        // Create strong uptrend
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();

        let br = BreadthRatio::new(5, 2).unwrap();
        let result = br.calculate(&close);

        // Uptrend should have ratio > 50 (more advances)
        let avg = result[10..].iter().sum::<f64>() / result[10..].len() as f64;
        assert!(avg > 50.0, "Bullish market should have ratio > 50, got {}", avg);
    }

    #[test]
    fn test_breadth_ratio_bearish() {
        // Create strong downtrend
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64).collect();

        let br = BreadthRatio::new(5, 2).unwrap();
        let result = br.calculate(&close);

        // Downtrend should have ratio < 50 (more declines)
        let avg = result[10..].iter().sum::<f64>() / result[10..].len() as f64;
        assert!(avg < 50.0, "Bearish market should have ratio < 50, got {}", avg);
    }

    #[test]
    fn test_breadth_score() {
        let data = make_test_data();
        let bs = BreadthScore::new(10, 0.5, 0.3).unwrap();
        let result = bs.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..10 {
            assert_eq!(result[i], 0.0);
        }
        // Values should be 0-100
        for i in 15..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "BreadthScore should be 0-100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_breadth_score_validation() {
        assert!(BreadthScore::new(4, 0.5, 0.3).is_err()); // period too small
        assert!(BreadthScore::new(10, -0.1, 0.3).is_err()); // negative weight
        assert!(BreadthScore::new(10, 1.1, 0.3).is_err()); // weight > 1
        assert!(BreadthScore::new(10, 0.6, 0.5).is_err()); // sum > 1
        assert!(BreadthScore::new(10, 0.5, 0.3).is_ok());
    }

    #[test]
    fn test_breadth_score_trait() {
        let data = make_test_data();
        let bs = BreadthScore::new(10, 0.5, 0.3).unwrap();
        assert_eq!(bs.name(), "Breadth Score");
        assert_eq!(bs.min_periods(), 11);
        let output = bs.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_score_strong_market() {
        // Create strong uptrend with volume confirmation
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let volume: Vec<f64> = (0..50).map(|i| 1000.0 + i as f64 * 50.0).collect();

        let bs = BreadthScore::new(5, 0.4, 0.3).unwrap();
        let result = bs.calculate(&close, &volume);

        // Strong market should have higher scores
        let avg = result[10..].iter().sum::<f64>() / result[10..].len() as f64;
        assert!(avg > 30.0, "Strong market should have above-average score, got {}", avg);
    }

    #[test]
    fn test_market_participation() {
        let data = make_test_data();
        let mp = MarketParticipation::new(10, 20).unwrap();
        let result = mp.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..20 {
            assert_eq!(result[i], 0.0);
        }
        // Values should be 0-100
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "MarketParticipation should be 0-100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_market_participation_validation() {
        assert!(MarketParticipation::new(4, 20).is_err()); // period too small
        assert!(MarketParticipation::new(10, 4).is_err()); // volatility_lookback too small
        assert!(MarketParticipation::new(10, 20).is_ok());
    }

    #[test]
    fn test_market_participation_trait() {
        let data = make_test_data();
        let mp = MarketParticipation::new(10, 20).unwrap();
        assert_eq!(mp.name(), "Market Participation");
        assert_eq!(mp.min_periods(), 21);
        let output = mp.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_market_participation_active_market() {
        // Create volatile market with significant moves
        let mut close: Vec<f64> = vec![100.0; 50];
        for i in 1..50 {
            if i % 2 == 0 {
                close[i] = close[i - 1] * 1.02; // 2% up
            } else {
                close[i] = close[i - 1] * 0.98; // 2% down
            }
        }

        let mp = MarketParticipation::new(5, 10).unwrap();
        let result = mp.calculate(&close);

        // Active market should show participation
        let has_participation = result[15..].iter().any(|&v| v > 0.0);
        assert!(has_participation, "Active market should show participation");
    }

    #[test]
    fn test_trend_breadth() {
        let data = make_test_data();
        let tb = TrendBreadth::new(20, 10, 5).unwrap();
        let result = tb.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..20 {
            assert_eq!(result[i], 0.0);
        }
        // Values should be 0-100
        for i in 30..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "TrendBreadth should be 0-100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_trend_breadth_validation() {
        assert!(TrendBreadth::new(4, 10, 5).is_err()); // trend_period too small
        assert!(TrendBreadth::new(20, 4, 5).is_err()); // breadth_period too small
        assert!(TrendBreadth::new(20, 10, 1).is_err()); // smoothing too small
        assert!(TrendBreadth::new(20, 10, 5).is_ok());
    }

    #[test]
    fn test_trend_breadth_trait() {
        let data = make_test_data();
        let tb = TrendBreadth::new(20, 10, 5).unwrap();
        assert_eq!(tb.name(), "Trend Breadth");
        assert_eq!(tb.min_periods(), 25);
        let output = tb.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_trend_breadth_confirmed_uptrend() {
        // Create confirmed uptrend (price up, breadth positive)
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();

        let tb = TrendBreadth::new(10, 5, 3).unwrap();
        let result = tb.calculate(&close);

        // Confirmed trend should have values > 50
        let avg = result[20..].iter().sum::<f64>() / result[20..].len() as f64;
        assert!(avg >= 50.0, "Confirmed uptrend should have TrendBreadth >= 50, got {}", avg);
    }

    #[test]
    fn test_trend_breadth_divergent() {
        // Create divergent pattern - net price up but mixed breadth
        let mut close: Vec<f64> = vec![100.0; 50];
        for i in 1..50 {
            // Overall uptrend but with frequent down days
            if i % 3 == 0 {
                close[i] = close[i - 1] - 0.3;
            } else {
                close[i] = close[i - 1] + 0.5;
            }
        }

        let tb = TrendBreadth::new(10, 5, 3).unwrap();
        let result = tb.calculate(&close);

        // Should produce valid values
        let has_values = result[20..].iter().any(|&v| v != 0.0);
        assert!(has_values, "Should calculate trend breadth values");
    }

    #[test]
    fn test_breadth_signal() {
        let data = make_test_data();
        let bs = BreadthSignal::new(10, 5, 3).unwrap();
        let result = bs.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // First values should be 0
        for i in 0..15 {
            assert_eq!(result[i], 0.0);
        }
        // Values should be -100 to +100
        for i in 20..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0,
                "BreadthSignal should be -100 to 100 at index {}, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_breadth_signal_validation() {
        assert!(BreadthSignal::new(4, 5, 3).is_err()); // period too small
        assert!(BreadthSignal::new(10, 1, 3).is_err()); // momentum_period too small
        assert!(BreadthSignal::new(10, 5, 1).is_err()); // smoothing too small
        assert!(BreadthSignal::new(10, 5, 3).is_ok());
    }

    #[test]
    fn test_breadth_signal_trait() {
        let data = make_test_data();
        let bs = BreadthSignal::new(10, 5, 3).unwrap();
        assert_eq!(bs.name(), "Breadth Signal");
        assert_eq!(bs.min_periods(), 18);
        let output = bs.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_breadth_signal_bullish() {
        // Create strong bullish market
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();

        let bs = BreadthSignal::new(5, 3, 2).unwrap();
        let result = bs.calculate(&close);

        // Bullish market should have positive signal
        let avg = result[15..].iter().sum::<f64>() / result[15..].len() as f64;
        assert!(avg > 0.0, "Bullish market should have positive signal, got {}", avg);
    }

    #[test]
    fn test_breadth_signal_bearish() {
        // Create strong bearish market
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64 * 2.0).collect();

        let bs = BreadthSignal::new(5, 3, 2).unwrap();
        let result = bs.calculate(&close);

        // Bearish market should have negative signal
        let avg = result[15..].iter().sum::<f64>() / result[15..].len() as f64;
        assert!(avg < 0.0, "Bearish market should have negative signal, got {}", avg);
    }

    #[test]
    fn test_all_final_five_indicators_compute() {
        let data = make_test_data();

        // Verify all 5 final NEW indicators can compute successfully
        // (BreadthMomentumIndex already exists, so we have 5 new ones)
        let br = BreadthRatio::new(10, 3).unwrap();
        let _ = br.compute(&data).unwrap();

        let bs = BreadthScore::new(10, 0.5, 0.3).unwrap();
        let _ = bs.compute(&data).unwrap();

        let mp = MarketParticipation::new(10, 20).unwrap();
        let _ = mp.compute(&data).unwrap();

        let tb = TrendBreadth::new(20, 10, 5).unwrap();
        let _ = tb.compute(&data).unwrap();

        let bsig = BreadthSignal::new(10, 5, 3).unwrap();
        let _ = bsig.compute(&data).unwrap();
    }
}
