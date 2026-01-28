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
        assert_eq!(output.values.len(), data.close.len());
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
        assert_eq!(output.values.len(), data.close.len());
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
        assert_eq!(output.values.len(), data.close.len());
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
        assert_eq!(output.values.len(), data.close.len());
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
        assert_eq!(output.values.len(), data.close.len());
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
        assert_eq!(output.values.len(), data.close.len());
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
}
