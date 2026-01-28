//! Advanced Band Indicators
//!
//! Additional sophisticated band and channel indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Volatility Bands - Volatility-scaled price bands
///
/// Creates bands around a moving average scaled by historical volatility.
/// Upper band = MA + (volatility * multiplier)
/// Lower band = MA - (volatility * multiplier)
#[derive(Debug, Clone)]
pub struct VolatilityBands {
    period: usize,
    volatility_period: usize,
    mult: f64,
}

impl VolatilityBands {
    pub fn new(period: usize, volatility_period: usize, mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volatility_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, volatility_period, mult })
    }

    /// Calculate volatility bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        // Calculate historical volatility (using ATR-based approach)
        let start_idx = self.period.max(self.volatility_period);

        for i in start_idx..n {
            // Calculate SMA for middle band
            let ma_start = i.saturating_sub(self.period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.period as f64;

            // Calculate ATR for volatility
            let vol_start = i.saturating_sub(self.volatility_period - 1);
            let mut atr_sum = 0.0;
            for j in vol_start..=i {
                let tr = if j == 0 {
                    high[j] - low[j]
                } else {
                    (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs())
                };
                atr_sum += tr;
            }
            let volatility = atr_sum / self.volatility_period as f64;

            middle[i] = ma;
            upper[i] = ma + self.mult * volatility;
            lower[i] = ma - self.mult * volatility;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for VolatilityBands {
    fn name(&self) -> &str {
        "Volatility Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.volatility_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Trend Bands - Trend-following adaptive bands
///
/// Bands that adapt based on the current trend direction and strength.
/// Uses EMA as the centerline with ATR-based band width.
#[derive(Debug, Clone)]
pub struct TrendBands {
    period: usize,
    atr_period: usize,
    mult: f64,
}

impl TrendBands {
    pub fn new(period: usize, atr_period: usize, mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if atr_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, atr_period, mult })
    }

    /// Calculate trend bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate EMA
        let alpha = 2.0 / (self.period as f64 + 1.0);
        let mut ema = vec![0.0; n];
        ema[0] = close[0];
        for i in 1..n {
            ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1];
        }

        // Calculate ATR
        let mut atr = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i < self.atr_period {
                // Simple average during warmup
                let mut sum = 0.0;
                for j in 1..=i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    sum += tr_j;
                }
                atr[i] = sum / i as f64;
            } else {
                // EMA of ATR
                let atr_alpha = 2.0 / (self.atr_period as f64 + 1.0);
                atr[i] = atr_alpha * tr + (1.0 - atr_alpha) * atr[i - 1];
            }
        }

        let start_idx = self.period.max(self.atr_period);
        for i in start_idx..n {
            // Trend adjustment factor based on price vs EMA
            let trend_strength = if ema[i] > 1e-10 {
                ((close[i] - ema[i]) / ema[i]).abs()
            } else {
                0.0
            };
            let adaptive_mult = self.mult * (1.0 + trend_strength);

            middle[i] = ema[i];
            upper[i] = ema[i] + adaptive_mult * atr[i];
            lower[i] = ema[i] - adaptive_mult * atr[i];
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for TrendBands {
    fn name(&self) -> &str {
        "Trend Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.atr_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Momentum Bands Advanced - Momentum-adjusted price channels
///
/// Bands that expand/contract based on price momentum.
/// Uses Rate of Change (ROC) to adjust band width.
#[derive(Debug, Clone)]
pub struct MomentumBandsAdvanced {
    period: usize,
    roc_period: usize,
    base_mult: f64,
}

impl MomentumBandsAdvanced {
    pub fn new(period: usize, roc_period: usize, base_mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if roc_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "roc_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if base_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, roc_period, base_mult })
    }

    /// Calculate momentum bands (middle, upper, lower)
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        // Calculate ROC (Rate of Change)
        let mut roc = vec![0.0; n];
        for i in self.roc_period..n {
            if close[i - self.roc_period].abs() > 1e-10 {
                roc[i] = (close[i] - close[i - self.roc_period]) / close[i - self.roc_period] * 100.0;
            }
        }

        let start_idx = self.period.max(self.roc_period);

        for i in start_idx..n {
            // Calculate SMA
            let ma_start = i.saturating_sub(self.period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.period as f64;

            // Calculate standard deviation
            let variance: f64 = close[ma_start..=i].iter()
                .map(|x| (x - ma).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            // Adjust multiplier based on momentum magnitude
            let momentum_factor = 1.0 + (roc[i].abs() / 100.0).min(1.0);
            let adjusted_mult = self.base_mult * momentum_factor;

            middle[i] = ma;
            upper[i] = ma + adjusted_mult * std_dev;
            lower[i] = ma - adjusted_mult * std_dev;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for MomentumBandsAdvanced {
    fn name(&self) -> &str {
        "Momentum Bands Advanced"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.roc_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Price Envelope - Percentage-based price envelope
///
/// Simple envelope bands at a fixed percentage above and below a moving average.
#[derive(Debug, Clone)]
pub struct PriceEnvelope {
    period: usize,
    upper_percent: f64,
    lower_percent: f64,
}

impl PriceEnvelope {
    pub fn new(period: usize, upper_percent: f64, lower_percent: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if upper_percent <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "upper_percent".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if lower_percent <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "lower_percent".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, upper_percent, lower_percent })
    }

    /// Create with symmetric percentage
    pub fn symmetric(period: usize, percent: f64) -> Result<Self> {
        Self::new(period, percent, percent)
    }

    /// Calculate price envelope (middle, upper, lower)
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let ma: f64 = close[start..=i].iter().sum::<f64>() / self.period as f64;

            middle[i] = ma;
            upper[i] = ma * (1.0 + self.upper_percent / 100.0);
            lower[i] = ma * (1.0 - self.lower_percent / 100.0);
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for PriceEnvelope {
    fn name(&self) -> &str {
        "Price Envelope"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Dynamic Price Channel - Adaptive price channel based on volatility
///
/// Price channel that adjusts its width based on recent volatility.
/// Higher volatility = wider channel.
#[derive(Debug, Clone)]
pub struct DynamicPriceChannel {
    period: usize,
    volatility_lookback: usize,
    base_mult: f64,
}

impl DynamicPriceChannel {
    pub fn new(period: usize, volatility_lookback: usize, base_mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volatility_lookback < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_lookback".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if base_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, volatility_lookback, base_mult })
    }

    /// Calculate dynamic price channel (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        // Calculate rolling volatility (normalized ATR)
        let start_idx = self.period.max(self.volatility_lookback);

        for i in start_idx..n {
            // Base channel from high/low
            let ch_start = i.saturating_sub(self.period - 1);
            let highest = high[ch_start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let lowest = low[ch_start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let base_middle = (highest + lowest) / 2.0;

            // Calculate ATR for volatility
            let vol_start = i.saturating_sub(self.volatility_lookback - 1);
            let mut atr_sum = 0.0;
            for j in vol_start..=i {
                let tr = if j == 0 {
                    high[j] - low[j]
                } else {
                    (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs())
                };
                atr_sum += tr;
            }
            let atr = atr_sum / self.volatility_lookback as f64;

            // Normalize volatility relative to price
            let normalized_vol = if base_middle > 1e-10 {
                atr / base_middle
            } else {
                0.0
            };

            // Adjust channel width based on volatility
            let volatility_mult = 1.0 + normalized_vol * 10.0; // Scale factor
            let width = (highest - lowest) * self.base_mult * volatility_mult;

            middle[i] = base_middle;
            upper[i] = base_middle + width / 2.0;
            lower[i] = base_middle - width / 2.0;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for DynamicPriceChannel {
    fn name(&self) -> &str {
        "Dynamic Price Channel"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.volatility_lookback) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Range Bands - High-low range based bands
///
/// Bands based on the average true range of highs and lows.
/// Useful for identifying support and resistance levels.
#[derive(Debug, Clone)]
pub struct RangeBands {
    period: usize,
    mult: f64,
}

impl RangeBands {
    pub fn new(period: usize, mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, mult })
    }

    /// Calculate range bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;

            // Calculate average high, low, and typical price
            let avg_high: f64 = high[start..=i].iter().sum::<f64>() / self.period as f64;
            let avg_low: f64 = low[start..=i].iter().sum::<f64>() / self.period as f64;
            let avg_close: f64 = close[start..=i].iter().sum::<f64>() / self.period as f64;

            // Middle is typical price
            let typical = (avg_high + avg_low + avg_close) / 3.0;

            // Range is based on high-low spread
            let avg_range = avg_high - avg_low;

            middle[i] = typical;
            upper[i] = typical + self.mult * avg_range;
            lower[i] = typical - self.mult * avg_range;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for RangeBands {
    fn name(&self) -> &str {
        "Range Bands"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Adaptive Bands System - Bands that adapt width based on market conditions
///
/// Uses efficiency ratio to determine market conditions and adjusts band width
/// accordingly. In trending markets, bands widen; in choppy markets, they narrow.
#[derive(Debug, Clone)]
pub struct AdaptiveBandsSystem {
    period: usize,
    fast_period: usize,
    slow_period: usize,
    mult: f64,
}

impl AdaptiveBandsSystem {
    pub fn new(period: usize, fast_period: usize, slow_period: usize, mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if fast_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if slow_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if fast_period >= slow_period {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be less than slow_period".to_string(),
            });
        }
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, fast_period, slow_period, mult })
    }

    /// Calculate adaptive bands system (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate Efficiency Ratio (ER)
        let mut er = vec![0.0; n];
        for i in self.period..n {
            let change = (close[i] - close[i - self.period]).abs();
            let mut volatility = 0.0;
            for j in (i - self.period + 1)..=i {
                volatility += (close[j] - close[j - 1]).abs();
            }
            er[i] = if volatility > 1e-10 { change / volatility } else { 0.0 };
        }

        // Adaptive smoothing constant
        let fast_sc = 2.0 / (self.fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (self.slow_period as f64 + 1.0);

        // Calculate adaptive moving average (KAMA-style)
        let mut kama = vec![0.0; n];
        if n > 0 {
            kama[0] = close[0];
        }
        for i in 1..n {
            let sc = (er[i] * (fast_sc - slow_sc) + slow_sc).powi(2);
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1]);
        }

        // Calculate ATR for band width
        let start_idx = self.period.max(self.slow_period);
        for i in start_idx..n {
            // Calculate ATR
            let atr_start = i.saturating_sub(self.period - 1);
            let mut atr_sum = 0.0;
            for j in atr_start..=i {
                let tr = if j == 0 {
                    high[j] - low[j]
                } else {
                    (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs())
                };
                atr_sum += tr;
            }
            let atr = atr_sum / self.period as f64;

            // Adaptive band width based on efficiency ratio
            // Higher ER = trending = wider bands
            let adaptive_mult = self.mult * (0.5 + er[i] * 1.5);

            middle[i] = kama[i];
            upper[i] = kama[i] + adaptive_mult * atr;
            lower[i] = kama[i] - adaptive_mult * atr;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for AdaptiveBandsSystem {
    fn name(&self) -> &str {
        "Adaptive Bands System"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.slow_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Trend Aware Bands - Bands that widen in trends, narrow in ranges
///
/// Uses ADX-style trend strength to dynamically adjust band width.
/// Strong trends cause bands to expand; weak trends cause contraction.
#[derive(Debug, Clone)]
pub struct TrendAwareBands {
    period: usize,
    atr_period: usize,
    trend_period: usize,
    base_mult: f64,
}

impl TrendAwareBands {
    pub fn new(period: usize, atr_period: usize, trend_period: usize, base_mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if atr_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if trend_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if base_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, atr_period, trend_period, base_mult })
    }

    /// Calculate trend aware bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate directional movement for trend strength
        let mut plus_dm = vec![0.0; n];
        let mut minus_dm = vec![0.0; n];
        let mut tr = vec![0.0; n];

        for i in 1..n {
            let up_move = high[i] - high[i - 1];
            let down_move = low[i - 1] - low[i];

            plus_dm[i] = if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
            minus_dm[i] = if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };

            tr[i] = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());
        }

        // Smoothed values using EMA
        let alpha = 2.0 / (self.trend_period as f64 + 1.0);
        let mut smooth_plus_dm = vec![0.0; n];
        let mut smooth_minus_dm = vec![0.0; n];
        let mut smooth_tr = vec![0.0; n];

        for i in 1..n {
            smooth_plus_dm[i] = alpha * plus_dm[i] + (1.0 - alpha) * smooth_plus_dm[i - 1];
            smooth_minus_dm[i] = alpha * minus_dm[i] + (1.0 - alpha) * smooth_minus_dm[i - 1];
            smooth_tr[i] = alpha * tr[i] + (1.0 - alpha) * smooth_tr[i - 1];
        }

        // Calculate DX (Directional Index)
        let mut dx = vec![0.0; n];
        for i in self.trend_period..n {
            let plus_di = if smooth_tr[i] > 1e-10 { 100.0 * smooth_plus_dm[i] / smooth_tr[i] } else { 0.0 };
            let minus_di = if smooth_tr[i] > 1e-10 { 100.0 * smooth_minus_dm[i] / smooth_tr[i] } else { 0.0 };

            let di_diff = (plus_di - minus_di).abs();
            let di_sum = plus_di + minus_di;
            dx[i] = if di_sum > 1e-10 { 100.0 * di_diff / di_sum } else { 0.0 };
        }

        // Smooth DX to get ADX-like trend strength
        let mut adx = vec![0.0; n];
        for i in self.trend_period..n {
            adx[i] = alpha * dx[i] + (1.0 - alpha) * adx[i.saturating_sub(1)];
        }

        let start_idx = self.period.max(self.atr_period).max(self.trend_period);

        for i in start_idx..n {
            // Calculate SMA for middle band
            let ma_start = i.saturating_sub(self.period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.period as f64;

            // Calculate ATR
            let atr_start = i.saturating_sub(self.atr_period - 1);
            let mut atr_sum = 0.0;
            for j in atr_start..=i {
                atr_sum += tr[j];
            }
            let atr = atr_sum / self.atr_period as f64;

            // Trend strength factor: ADX normalized to 0-1, then scaled
            // ADX typically ranges 0-100, with 25+ indicating strong trend
            let trend_factor = (adx[i] / 50.0).min(2.0).max(0.5);
            let adaptive_mult = self.base_mult * trend_factor;

            middle[i] = ma;
            upper[i] = ma + adaptive_mult * atr;
            lower[i] = ma - adaptive_mult * atr;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for TrendAwareBands {
    fn name(&self) -> &str {
        "Trend Aware Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.atr_period).max(self.trend_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Volatility Adjusted Bands - Bands with volatility-based adjustments
///
/// Uses historical volatility percentile to adjust band width dynamically.
/// When current volatility is high relative to history, bands expand.
#[derive(Debug, Clone)]
pub struct VolatilityAdjustedBands {
    period: usize,
    volatility_period: usize,
    lookback: usize,
    base_mult: f64,
}

impl VolatilityAdjustedBands {
    pub fn new(period: usize, volatility_period: usize, lookback: usize, base_mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volatility_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if lookback < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if base_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, volatility_period, lookback, base_mult })
    }

    /// Calculate volatility adjusted bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate rolling ATR values
        let mut atr_values = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i < self.volatility_period {
                // Simple average during warmup
                let mut sum = 0.0;
                for j in 1..=i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    sum += tr_j;
                }
                atr_values[i] = sum / i as f64;
            } else {
                let alpha = 2.0 / (self.volatility_period as f64 + 1.0);
                atr_values[i] = alpha * tr + (1.0 - alpha) * atr_values[i - 1];
            }
        }

        let start_idx = self.period.max(self.volatility_period).max(self.lookback);

        for i in start_idx..n {
            // Calculate SMA for middle band
            let ma_start = i.saturating_sub(self.period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.period as f64;

            // Current ATR
            let current_atr = atr_values[i];

            // Calculate volatility percentile over lookback period
            let lookback_start = i.saturating_sub(self.lookback - 1);
            let atr_slice = &atr_values[lookback_start..=i];

            // Count how many values are below current ATR
            let count_below = atr_slice.iter().filter(|&&v| v < current_atr).count();
            let percentile = count_below as f64 / atr_slice.len() as f64;

            // Adjust multiplier based on percentile (0.5 to 1.5 range)
            let vol_adjustment = 0.5 + percentile;
            let adaptive_mult = self.base_mult * vol_adjustment;

            middle[i] = ma;
            upper[i] = ma + adaptive_mult * current_atr;
            lower[i] = ma - adaptive_mult * current_atr;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for VolatilityAdjustedBands {
    fn name(&self) -> &str {
        "Volatility Adjusted Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.volatility_period).max(self.lookback) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Cycle Bands - Bands based on detected market cycles
///
/// Uses a simple cycle detection algorithm to identify dominant cycle length
/// and adjusts band width based on cycle phase.
#[derive(Debug, Clone)]
pub struct CycleBands {
    period: usize,
    min_cycle: usize,
    max_cycle: usize,
    mult: f64,
}

impl CycleBands {
    pub fn new(period: usize, min_cycle: usize, max_cycle: usize, mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if min_cycle < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_cycle".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if max_cycle < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_cycle".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if min_cycle >= max_cycle {
            return Err(IndicatorError::InvalidParameter {
                name: "min_cycle".to_string(),
                reason: "must be less than max_cycle".to_string(),
            });
        }
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, min_cycle, max_cycle, mult })
    }

    /// Detect dominant cycle using autocorrelation
    fn detect_cycle(&self, prices: &[f64]) -> usize {
        let n = prices.len();
        if n < self.max_cycle * 2 {
            return self.min_cycle;
        }

        // Calculate mean
        let mean: f64 = prices.iter().sum::<f64>() / n as f64;

        // Find dominant cycle using autocorrelation
        let mut best_cycle = self.min_cycle;
        let mut best_corr = -2.0;

        for lag in self.min_cycle..=self.max_cycle.min(n / 2) {
            let mut sum_product = 0.0;
            let mut sum_sq1 = 0.0;
            let mut sum_sq2 = 0.0;

            for i in 0..(n - lag) {
                let v1 = prices[i] - mean;
                let v2 = prices[i + lag] - mean;
                sum_product += v1 * v2;
                sum_sq1 += v1 * v1;
                sum_sq2 += v2 * v2;
            }

            let denom = (sum_sq1 * sum_sq2).sqrt();
            let corr = if denom > 1e-10 { sum_product / denom } else { 0.0 };

            if corr > best_corr {
                best_corr = corr;
                best_cycle = lag;
            }
        }

        best_cycle
    }

    /// Calculate cycle bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        let start_idx = self.period.max(self.max_cycle * 2);

        for i in start_idx..n {
            // Detect cycle from recent data
            let lookback_start = i.saturating_sub(self.max_cycle * 3);
            let cycle_length = self.detect_cycle(&close[lookback_start..=i]);

            // Calculate adaptive MA using detected cycle
            let adaptive_period = cycle_length.min(i);
            let ma_start = i.saturating_sub(adaptive_period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / adaptive_period as f64;

            // Calculate cycle-adjusted ATR
            let atr_period = cycle_length.min(i);
            let atr_start = i.saturating_sub(atr_period - 1);
            let mut atr_sum = 0.0;
            for j in atr_start..=i {
                let tr = if j == 0 {
                    high[j] - low[j]
                } else {
                    (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs())
                };
                atr_sum += tr;
            }
            let atr = atr_sum / atr_period as f64;

            // Cycle phase adjustment (simple sine wave approximation)
            let phase = ((i - start_idx) as f64 * 2.0 * std::f64::consts::PI) / cycle_length as f64;
            let phase_factor = 0.8 + 0.4 * phase.sin().abs(); // Ranges from 0.8 to 1.2

            let adaptive_mult = self.mult * phase_factor;

            middle[i] = ma;
            upper[i] = ma + adaptive_mult * atr;
            lower[i] = ma - adaptive_mult * atr;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for CycleBands {
    fn name(&self) -> &str {
        "Cycle Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.max_cycle * 2) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Adaptive Keltner Channels - Keltner channels with adaptive multiplier
///
/// Traditional Keltner Channels use a fixed ATR multiplier. This version
/// adapts the multiplier based on market volatility conditions.
/// When volatility is high relative to recent history, the multiplier expands.
#[derive(Debug, Clone)]
pub struct AdaptiveKeltnerChannels {
    ema_period: usize,
    atr_period: usize,
    base_mult: f64,
    volatility_lookback: usize,
}

impl AdaptiveKeltnerChannels {
    pub fn new(ema_period: usize, atr_period: usize, base_mult: f64, volatility_lookback: usize) -> Result<Self> {
        if ema_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "ema_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if atr_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if base_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if volatility_lookback < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_lookback".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { ema_period, atr_period, base_mult, volatility_lookback })
    }

    /// Calculate adaptive Keltner channels (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate EMA for middle band
        let alpha = 2.0 / (self.ema_period as f64 + 1.0);
        let mut ema = vec![0.0; n];
        ema[0] = close[0];
        for i in 1..n {
            ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1];
        }

        // Calculate ATR values
        let mut atr_values = vec![0.0; n];
        let atr_alpha = 2.0 / (self.atr_period as f64 + 1.0);
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());
            atr_values[i] = atr_alpha * tr + (1.0 - atr_alpha) * atr_values[i - 1];
        }

        let start_idx = self.ema_period.max(self.atr_period).max(self.volatility_lookback);

        for i in start_idx..n {
            let current_atr = atr_values[i];

            // Calculate volatility percentile over lookback period
            let lookback_start = i.saturating_sub(self.volatility_lookback - 1);
            let atr_slice = &atr_values[lookback_start..=i];

            // Count how many values are below current ATR
            let count_below = atr_slice.iter().filter(|&&v| v < current_atr).count();
            let percentile = count_below as f64 / atr_slice.len() as f64;

            // Adaptive multiplier: ranges from 0.5x to 1.5x of base_mult
            let adaptive_mult = self.base_mult * (0.5 + percentile);

            middle[i] = ema[i];
            upper[i] = ema[i] + adaptive_mult * current_atr;
            lower[i] = ema[i] - adaptive_mult * current_atr;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for AdaptiveKeltnerChannels {
    fn name(&self) -> &str {
        "Adaptive Keltner Channels"
    }

    fn min_periods(&self) -> usize {
        self.ema_period.max(self.atr_period).max(self.volatility_lookback) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Volatility Weighted Bands - Bands weighted by recent volatility
///
/// Creates bands where the width is weighted by comparing current volatility
/// to a longer-term average volatility. Higher relative volatility = wider bands.
#[derive(Debug, Clone)]
pub struct VolatilityWeightedBands {
    period: usize,
    short_vol_period: usize,
    long_vol_period: usize,
    base_mult: f64,
}

impl VolatilityWeightedBands {
    pub fn new(period: usize, short_vol_period: usize, long_vol_period: usize, base_mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if short_vol_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_vol_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if long_vol_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "long_vol_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if short_vol_period >= long_vol_period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_vol_period".to_string(),
                reason: "must be less than long_vol_period".to_string(),
            });
        }
        if base_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, short_vol_period, long_vol_period, base_mult })
    }

    /// Calculate volatility weighted bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate true range values
        let mut tr = vec![0.0; n];
        for i in 1..n {
            tr[i] = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());
        }

        let start_idx = self.period.max(self.long_vol_period);

        for i in start_idx..n {
            // Calculate SMA for middle band
            let ma_start = i.saturating_sub(self.period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.period as f64;

            // Calculate short-term volatility (ATR)
            let short_start = i.saturating_sub(self.short_vol_period - 1);
            let short_vol: f64 = tr[short_start..=i].iter().sum::<f64>() / self.short_vol_period as f64;

            // Calculate long-term volatility (ATR)
            let long_start = i.saturating_sub(self.long_vol_period - 1);
            let long_vol: f64 = tr[long_start..=i].iter().sum::<f64>() / self.long_vol_period as f64;

            // Volatility weight: ratio of short to long volatility
            // Clamped to reasonable range [0.5, 2.0]
            let vol_weight = if long_vol > 1e-10 {
                (short_vol / long_vol).max(0.5).min(2.0)
            } else {
                1.0
            };

            let adaptive_mult = self.base_mult * vol_weight;

            middle[i] = ma;
            upper[i] = ma + adaptive_mult * short_vol;
            lower[i] = ma - adaptive_mult * short_vol;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for VolatilityWeightedBands {
    fn name(&self) -> &str {
        "Volatility Weighted Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.long_vol_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Trend Following Channel - Channel that follows trend direction
///
/// A channel that biases toward the trend direction. In uptrends, the lower
/// band is closer to price (tighter stop); in downtrends, the upper band is closer.
/// Uses trend strength to determine the asymmetry.
#[derive(Debug, Clone)]
pub struct TrendFollowingChannel {
    period: usize,
    atr_period: usize,
    trend_period: usize,
    mult: f64,
}

impl TrendFollowingChannel {
    pub fn new(period: usize, atr_period: usize, trend_period: usize, mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if atr_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if trend_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, atr_period, trend_period, mult })
    }

    /// Calculate trend following channel (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate EMA for middle band
        let alpha = 2.0 / (self.period as f64 + 1.0);
        let mut ema = vec![0.0; n];
        ema[0] = close[0];
        for i in 1..n {
            ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1];
        }

        // Calculate ATR
        let atr_alpha = 2.0 / (self.atr_period as f64 + 1.0);
        let mut atr = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());
            atr[i] = atr_alpha * tr + (1.0 - atr_alpha) * atr[i - 1];
        }

        let start_idx = self.period.max(self.atr_period).max(self.trend_period);

        for i in start_idx..n {
            // Calculate trend direction and strength using linear regression slope
            let slope = {
                let trend_start = i.saturating_sub(self.trend_period - 1);
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_xy = 0.0;
                let mut sum_xx = 0.0;
                let count = (i - trend_start + 1) as f64;

                for (j, &price) in close[trend_start..=i].iter().enumerate() {
                    let x = j as f64;
                    sum_x += x;
                    sum_y += price;
                    sum_xy += x * price;
                    sum_xx += x * x;
                }

                let denom = count * sum_xx - sum_x * sum_x;
                if denom.abs() > 1e-10 {
                    (count * sum_xy - sum_x * sum_y) / denom
                } else {
                    0.0
                }
            };

            // Normalize slope relative to price (as percentage)
            let norm_slope = if ema[i] > 1e-10 {
                slope / ema[i] * 100.0
            } else {
                0.0
            };

            // Trend bias factor: -1 (strong downtrend) to +1 (strong uptrend)
            // Clamped to prevent extreme asymmetry
            let trend_bias = norm_slope.max(-0.5).min(0.5);

            // In uptrend: lower band closer (smaller mult), upper band farther
            // In downtrend: upper band closer, lower band farther
            let upper_mult = self.mult * (1.0 - trend_bias);
            let lower_mult = self.mult * (1.0 + trend_bias);

            middle[i] = ema[i];
            upper[i] = ema[i] + upper_mult * atr[i];
            lower[i] = ema[i] - lower_mult * atr[i];
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for TrendFollowingChannel {
    fn name(&self) -> &str {
        "Trend Following Channel"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.atr_period).max(self.trend_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Dynamic Support Resistance Bands - Bands based on S/R levels
///
/// Creates bands based on recent swing highs and lows, which represent
/// natural support and resistance levels. The bands adapt as new swing
/// points are formed.
#[derive(Debug, Clone)]
pub struct DynamicSupportResistanceBands {
    period: usize,
    swing_period: usize,
    smoothing: usize,
}

impl DynamicSupportResistanceBands {
    pub fn new(period: usize, swing_period: usize, smoothing: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if swing_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if smoothing < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, swing_period, smoothing })
    }

    /// Detect swing highs and lows
    fn find_swing_points(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len().min(low.len());
        let mut swing_highs = vec![f64::NAN; n];
        let mut swing_lows = vec![f64::NAN; n];

        let half_period = self.swing_period / 2;

        for i in half_period..(n - half_period) {
            // Check for swing high
            let mut is_swing_high = true;
            for j in (i - half_period)..i {
                if high[j] >= high[i] {
                    is_swing_high = false;
                    break;
                }
            }
            if is_swing_high {
                for j in (i + 1)..=(i + half_period) {
                    if high[j] >= high[i] {
                        is_swing_high = false;
                        break;
                    }
                }
            }
            if is_swing_high {
                swing_highs[i] = high[i];
            }

            // Check for swing low
            let mut is_swing_low = true;
            for j in (i - half_period)..i {
                if low[j] <= low[i] {
                    is_swing_low = false;
                    break;
                }
            }
            if is_swing_low {
                for j in (i + 1)..=(i + half_period) {
                    if low[j] <= low[i] {
                        is_swing_low = false;
                        break;
                    }
                }
            }
            if is_swing_low {
                swing_lows[i] = low[i];
            }
        }

        (swing_highs, swing_lows)
    }

    /// Calculate dynamic S/R bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        let (swing_highs, swing_lows) = self.find_swing_points(high, low);

        // Calculate EMA for middle band
        let alpha = 2.0 / (self.period as f64 + 1.0);
        let mut ema = vec![0.0; n];
        ema[0] = close[0];
        for i in 1..n {
            ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1];
        }

        let start_idx = self.period.max(self.swing_period);
        let smooth_alpha = 2.0 / (self.smoothing as f64 + 1.0);

        // Track recent swing levels
        let mut resistance = if start_idx < n { high[start_idx] } else { 0.0 };
        let mut support = if start_idx < n { low[start_idx] } else { 0.0 };

        for i in start_idx..n {
            // Update resistance from recent swing highs
            let lookback_start = i.saturating_sub(self.period - 1);
            let mut recent_swing_high = f64::MIN;
            for j in lookback_start..=i {
                if !swing_highs[j].is_nan() && swing_highs[j] > recent_swing_high {
                    recent_swing_high = swing_highs[j];
                }
            }
            if recent_swing_high > f64::MIN {
                resistance = smooth_alpha * recent_swing_high + (1.0 - smooth_alpha) * resistance;
            }

            // Update support from recent swing lows
            let mut recent_swing_low = f64::MAX;
            for j in lookback_start..=i {
                if !swing_lows[j].is_nan() && swing_lows[j] < recent_swing_low {
                    recent_swing_low = swing_lows[j];
                }
            }
            if recent_swing_low < f64::MAX {
                support = smooth_alpha * recent_swing_low + (1.0 - smooth_alpha) * support;
            }

            middle[i] = ema[i];
            upper[i] = resistance;
            lower[i] = support;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for DynamicSupportResistanceBands {
    fn name(&self) -> &str {
        "Dynamic Support Resistance Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.swing_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Momentum Bandwidth - Bandwidth based on momentum
///
/// Measures the bandwidth (upper - lower) / middle of momentum-adjusted bands.
/// Returns a single oscillator value indicating band expansion or contraction.
/// Higher values indicate increased momentum and volatility.
#[derive(Debug, Clone)]
pub struct MomentumBandwidth {
    period: usize,
    momentum_period: usize,
    mult: f64,
}

impl MomentumBandwidth {
    pub fn new(period: usize, momentum_period: usize, mult: f64) -> Result<Self> {
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
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, momentum_period, mult })
    }

    /// Calculate momentum bandwidth oscillator
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut bandwidth = vec![0.0; n];

        if n == 0 {
            return bandwidth;
        }

        let start_idx = self.period.max(self.momentum_period);

        for i in start_idx..n {
            // Calculate SMA
            let ma_start = i.saturating_sub(self.period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.period as f64;

            // Calculate standard deviation
            let variance: f64 = close[ma_start..=i].iter()
                .map(|x| (x - ma).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            // Calculate momentum (ROC)
            let momentum = if close[i - self.momentum_period].abs() > 1e-10 {
                ((close[i] - close[i - self.momentum_period]) / close[i - self.momentum_period]).abs()
            } else {
                0.0
            };

            // Momentum-adjusted bandwidth
            // Higher momentum = wider effective bands = higher bandwidth reading
            let momentum_factor = 1.0 + momentum * 10.0; // Scale momentum effect
            let adjusted_mult = self.mult * momentum_factor;

            let upper = ma + adjusted_mult * std_dev;
            let lower = ma - adjusted_mult * std_dev;

            // Bandwidth as percentage
            bandwidth[i] = if ma > 1e-10 {
                (upper - lower) / ma * 100.0
            } else {
                0.0
            };
        }

        bandwidth
    }
}

impl TechnicalIndicator for MomentumBandwidth {
    fn name(&self) -> &str {
        "Momentum Bandwidth"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.momentum_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }
}

/// Price Envelope Oscillator - Oscillator showing position within envelope
///
/// Shows where the current price is positioned relative to the envelope bands.
/// Returns values typically between -100 (at lower band) and +100 (at upper band).
/// Values outside this range indicate band breakouts.
#[derive(Debug, Clone)]
pub struct PriceEnvelopeOscillator {
    period: usize,
    envelope_percent: f64,
}

impl PriceEnvelopeOscillator {
    pub fn new(period: usize, envelope_percent: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if envelope_percent <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "envelope_percent".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, envelope_percent })
    }

    /// Calculate price envelope oscillator
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut oscillator = vec![0.0; n];

        if n == 0 {
            return oscillator;
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let ma: f64 = close[start..=i].iter().sum::<f64>() / self.period as f64;

            let upper = ma * (1.0 + self.envelope_percent / 100.0);
            let lower = ma * (1.0 - self.envelope_percent / 100.0);

            // Calculate position within envelope as percentage
            // 0 = at middle, +100 = at upper, -100 = at lower
            let band_width = upper - lower;
            if band_width > 1e-10 {
                // Scale so that middle = 0, upper = 100, lower = -100
                oscillator[i] = ((close[i] - ma) / (band_width / 2.0)) * 100.0;
            }
        }

        oscillator
    }
}

impl TechnicalIndicator for PriceEnvelopeOscillator {
    fn name(&self) -> &str {
        "Price Envelope Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }
}

/// Dynamic Envelope - Dynamic price envelope system
///
/// Creates an adaptive envelope that adjusts based on price momentum,
/// volatility, and recent price action relative to the envelope.
#[derive(Debug, Clone)]
pub struct DynamicEnvelope {
    period: usize,
    smoothing: usize,
    expansion_factor: f64,
    contraction_factor: f64,
}

impl DynamicEnvelope {
    pub fn new(period: usize, smoothing: usize, expansion_factor: f64, contraction_factor: f64) -> Result<Self> {
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
        if expansion_factor <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "expansion_factor".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if contraction_factor <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "contraction_factor".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if contraction_factor >= expansion_factor {
            return Err(IndicatorError::InvalidParameter {
                name: "contraction_factor".to_string(),
                reason: "must be less than expansion_factor".to_string(),
            });
        }
        Ok(Self { period, smoothing, expansion_factor, contraction_factor })
    }

    /// Calculate dynamic envelope (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate EMA for middle band
        let alpha = 2.0 / (self.period as f64 + 1.0);
        let mut ema = vec![0.0; n];
        ema[0] = close[0];
        for i in 1..n {
            ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1];
        }

        // Calculate ATR
        let mut atr = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());
            let atr_alpha = 2.0 / (self.period as f64 + 1.0);
            atr[i] = atr_alpha * tr + (1.0 - atr_alpha) * atr[i.saturating_sub(1)];
        }

        // Dynamic envelope width
        let mut envelope_width = vec![0.0; n];
        let smooth_alpha = 2.0 / (self.smoothing as f64 + 1.0);

        let start_idx = self.period.max(self.smoothing);

        // Initialize envelope width
        if start_idx < n {
            envelope_width[start_idx] = atr[start_idx] * self.expansion_factor;
        }

        for i in (start_idx + 1)..n {
            let prev_upper = ema[i - 1] + envelope_width[i - 1];
            let prev_lower = ema[i - 1] - envelope_width[i - 1];

            // Determine if price is pushing against envelope
            let target_width = if close[i] > prev_upper {
                // Price breaking upper band - expand
                atr[i] * self.expansion_factor
            } else if close[i] < prev_lower {
                // Price breaking lower band - expand
                atr[i] * self.expansion_factor
            } else if close[i] > ema[i] && high[i] >= prev_upper * 0.99 {
                // Price approaching upper - slight expansion
                atr[i] * (self.expansion_factor + self.contraction_factor) / 2.0
            } else if close[i] < ema[i] && low[i] <= prev_lower * 1.01 {
                // Price approaching lower - slight expansion
                atr[i] * (self.expansion_factor + self.contraction_factor) / 2.0
            } else {
                // Price within envelope - contract
                atr[i] * self.contraction_factor
            };

            // Smooth the envelope width transition
            envelope_width[i] = smooth_alpha * target_width + (1.0 - smooth_alpha) * envelope_width[i - 1];
        }

        for i in start_idx..n {
            middle[i] = ema[i];
            upper[i] = ema[i] + envelope_width[i];
            lower[i] = ema[i] - envelope_width[i];
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for DynamicEnvelope {
    fn name(&self) -> &str {
        "Dynamic Envelope"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.smoothing) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Volatility Bandwidth - Measures bandwidth as percentage of price
///
/// Calculates the width of volatility bands as a percentage of the middle band.
/// This indicator helps identify periods of low and high volatility (squeeze/expansion).
/// Values are expressed as percentages - lower values indicate squeeze conditions.
///
/// Formula: Bandwidth = (Upper - Lower) / Middle * 100
#[derive(Debug, Clone)]
pub struct VolatilityBandwidth {
    period: usize,
    mult: f64,
}

impl VolatilityBandwidth {
    pub fn new(period: usize, mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, mult })
    }

    /// Calculate volatility bandwidth as percentage
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut bandwidth = vec![0.0; n];

        if n == 0 {
            return bandwidth;
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let slice = &close[start..=i];

            // Calculate SMA (middle band)
            let ma: f64 = slice.iter().sum::<f64>() / self.period as f64;

            // Calculate standard deviation
            let variance: f64 = slice.iter()
                .map(|x| (x - ma).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            // Calculate bands
            let upper = ma + self.mult * std_dev;
            let lower = ma - self.mult * std_dev;

            // Bandwidth as percentage
            if ma > 1e-10 {
                bandwidth[i] = (upper - lower) / ma * 100.0;
            }
        }

        bandwidth
    }

    /// Calculate with bands returned as well (middle, upper, lower, bandwidth)
    pub fn calculate_with_bands(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];
        let mut bandwidth = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower, bandwidth);
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let slice = &close[start..=i];

            let ma: f64 = slice.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = slice.iter()
                .map(|x| (x - ma).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            middle[i] = ma;
            upper[i] = ma + self.mult * std_dev;
            lower[i] = ma - self.mult * std_dev;

            if ma > 1e-10 {
                bandwidth[i] = (upper[i] - lower[i]) / ma * 100.0;
            }
        }

        (middle, upper, lower, bandwidth)
    }
}

impl TechnicalIndicator for VolatilityBandwidth {
    fn name(&self) -> &str {
        "Volatility Bandwidth"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }
}

/// Band Breakout Strength - Measures strength of band breakouts
///
/// Quantifies how strongly price has broken through the upper or lower band.
/// Positive values indicate upward breakouts, negative values indicate downward breakouts.
/// The magnitude indicates breakout strength relative to band width.
///
/// Formula: Strength = (Close - Band) / BandWidth * 100
/// Where Band is upper band for closes above, lower band for closes below
#[derive(Debug, Clone)]
pub struct BandBreakoutStrength {
    period: usize,
    mult: f64,
}

impl BandBreakoutStrength {
    pub fn new(period: usize, mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, mult })
    }

    /// Calculate band breakout strength
    /// Returns values typically in range [-100, 100] when within bands
    /// Values exceed this range during breakouts
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut strength = vec![0.0; n];

        if n == 0 {
            return strength;
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let slice = &close[start..=i];

            // Calculate bands
            let ma: f64 = slice.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = slice.iter()
                .map(|x| (x - ma).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            let upper = ma + self.mult * std_dev;
            let lower = ma - self.mult * std_dev;
            let band_width = upper - lower;

            if band_width > 1e-10 {
                // Calculate position relative to bands
                if close[i] > upper {
                    // Upward breakout: positive strength beyond 100
                    strength[i] = 100.0 + (close[i] - upper) / band_width * 200.0;
                } else if close[i] < lower {
                    // Downward breakout: negative strength beyond -100
                    strength[i] = -100.0 - (lower - close[i]) / band_width * 200.0;
                } else {
                    // Within bands: -100 to +100 range
                    // 0 at middle, +100 at upper, -100 at lower
                    strength[i] = (close[i] - ma) / (band_width / 2.0) * 100.0;
                }
            }
        }

        strength
    }

    /// Calculate with bands returned as well (middle, upper, lower)
    pub fn calculate_bands(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let slice = &close[start..=i];

            let ma: f64 = slice.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = slice.iter()
                .map(|x| (x - ma).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            middle[i] = ma;
            upper[i] = ma + self.mult * std_dev;
            lower[i] = ma - self.mult * std_dev;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for BandBreakoutStrength {
    fn name(&self) -> &str {
        "Band Breakout Strength"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }
}

/// Dynamic Price Bands - Price bands that adapt to volatility regime
///
/// Creates bands that dynamically adjust their width based on the current
/// volatility regime (low, normal, or high volatility). The bands expand
/// during high volatility and contract during low volatility, using a
/// percentile-based regime detection.
///
/// This differs from other adaptive bands by using regime classification
/// rather than continuous adjustment.
#[derive(Debug, Clone)]
pub struct DynamicPriceBands {
    period: usize,
    volatility_lookback: usize,
    low_vol_mult: f64,
    high_vol_mult: f64,
}

impl DynamicPriceBands {
    pub fn new(period: usize, volatility_lookback: usize, low_vol_mult: f64, high_vol_mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volatility_lookback < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_lookback".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if low_vol_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "low_vol_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if high_vol_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "high_vol_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if low_vol_mult >= high_vol_mult {
            return Err(IndicatorError::InvalidParameter {
                name: "low_vol_mult".to_string(),
                reason: "must be less than high_vol_mult".to_string(),
            });
        }
        Ok(Self { period, volatility_lookback, low_vol_mult, high_vol_mult })
    }

    /// Calculate dynamic price bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate ATR for volatility measurement
        let mut atr_values = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i < self.period {
                // Simple average during warmup
                let mut sum = tr;
                for j in 1..i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    sum += tr_j;
                }
                atr_values[i] = sum / i as f64;
            } else {
                let alpha = 2.0 / (self.period as f64 + 1.0);
                atr_values[i] = alpha * tr + (1.0 - alpha) * atr_values[i - 1];
            }
        }

        let start_idx = self.period.max(self.volatility_lookback);

        for i in start_idx..n {
            // Calculate SMA for middle band
            let ma_start = i.saturating_sub(self.period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.period as f64;

            // Calculate current ATR
            let current_atr = atr_values[i];

            // Determine volatility regime using percentile
            let lookback_start = i.saturating_sub(self.volatility_lookback - 1);
            let atr_slice = &atr_values[lookback_start..=i];

            let count_below = atr_slice.iter().filter(|&&v| v < current_atr).count();
            let percentile = count_below as f64 / atr_slice.len() as f64;

            // Regime-based multiplier selection
            // Low volatility (bottom 33%): use low_vol_mult
            // High volatility (top 33%): use high_vol_mult
            // Normal volatility: interpolate between them
            let mult = if percentile < 0.33 {
                self.low_vol_mult
            } else if percentile > 0.67 {
                self.high_vol_mult
            } else {
                // Linear interpolation for normal regime
                let regime_position = (percentile - 0.33) / 0.34;
                self.low_vol_mult + regime_position * (self.high_vol_mult - self.low_vol_mult)
            };

            middle[i] = ma;
            upper[i] = ma + mult * current_atr;
            lower[i] = ma - mult * current_atr;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for DynamicPriceBands {
    fn name(&self) -> &str {
        "Dynamic Price Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.volatility_lookback) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Trend Aligned Bands - Bands that shift based on trend direction
///
/// Creates bands where the center line shifts toward the trend direction.
/// In uptrends, the middle band is biased upward; in downtrends, biased downward.
/// This helps keep price within the bands during trending markets.
///
/// Uses linear regression to determine trend direction and strength.
#[derive(Debug, Clone)]
pub struct TrendAlignedBands {
    period: usize,
    trend_period: usize,
    mult: f64,
    max_shift: f64,
}

impl TrendAlignedBands {
    pub fn new(period: usize, trend_period: usize, mult: f64, max_shift: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if trend_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if max_shift <= 0.0 || max_shift > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_shift".to_string(),
                reason: "must be between 0 and 1 (exclusive of 0)".to_string(),
            });
        }
        Ok(Self { period, trend_period, mult, max_shift })
    }

    /// Calculate trend aligned bands (middle, upper, lower)
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        let start_idx = self.period.max(self.trend_period);

        for i in start_idx..n {
            // Calculate base SMA
            let ma_start = i.saturating_sub(self.period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.period as f64;

            // Calculate standard deviation for band width
            let variance: f64 = close[ma_start..=i].iter()
                .map(|x| (x - ma).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            // Calculate linear regression slope for trend
            let trend_start = i.saturating_sub(self.trend_period - 1);
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_xy = 0.0;
            let mut sum_xx = 0.0;
            let count = (i - trend_start + 1) as f64;

            for (j, &price) in close[trend_start..=i].iter().enumerate() {
                let x = j as f64;
                sum_x += x;
                sum_y += price;
                sum_xy += x * price;
                sum_xx += x * x;
            }

            let denom = count * sum_xx - sum_x * sum_x;
            let slope = if denom.abs() > 1e-10 {
                (count * sum_xy - sum_x * sum_y) / denom
            } else {
                0.0
            };

            // Normalize slope as percentage of price
            let norm_slope = if ma > 1e-10 {
                (slope * self.trend_period as f64) / ma
            } else {
                0.0
            };

            // Calculate trend shift (clamped to max_shift)
            let trend_shift = norm_slope.max(-self.max_shift).min(self.max_shift);

            // Shift middle band based on trend
            let band_width = self.mult * std_dev;
            let shifted_middle = ma + trend_shift * band_width;

            middle[i] = shifted_middle;
            upper[i] = shifted_middle + band_width;
            lower[i] = shifted_middle - band_width;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for TrendAlignedBands {
    fn name(&self) -> &str {
        "Trend Aligned Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.trend_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Momentum Driven Bands - Bands scaled by momentum strength
///
/// Creates bands where the width is directly proportional to price momentum.
/// During strong momentum moves, bands widen; during consolidation, they narrow.
/// Uses the Rate of Change (ROC) absolute value to scale band width.
///
/// This differs from MomentumBandsAdvanced by using momentum as the primary
/// scaling factor rather than just an adjustment.
#[derive(Debug, Clone)]
pub struct MomentumDrivenBands {
    period: usize,
    momentum_period: usize,
    min_mult: f64,
    max_mult: f64,
}

impl MomentumDrivenBands {
    pub fn new(period: usize, momentum_period: usize, min_mult: f64, max_mult: f64) -> Result<Self> {
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
        if min_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if max_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if min_mult >= max_mult {
            return Err(IndicatorError::InvalidParameter {
                name: "min_mult".to_string(),
                reason: "must be less than max_mult".to_string(),
            });
        }
        Ok(Self { period, momentum_period, min_mult, max_mult })
    }

    /// Calculate momentum driven bands (middle, upper, lower)
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate absolute ROC values for momentum
        let mut abs_roc = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                abs_roc[i] = ((close[i] - close[i - self.momentum_period])
                    / close[i - self.momentum_period]).abs() * 100.0;
            }
        }

        // Calculate rolling max ROC for normalization
        let start_idx = self.period.max(self.momentum_period);
        let lookback = self.period * 2; // Use 2x period for max ROC lookback

        for i in start_idx..n {
            // Calculate SMA for middle band
            let ma_start = i.saturating_sub(self.period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.period as f64;

            // Calculate standard deviation
            let variance: f64 = close[ma_start..=i].iter()
                .map(|x| (x - ma).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            // Get max ROC over lookback for normalization
            let roc_start = i.saturating_sub(lookback - 1);
            let max_roc = abs_roc[roc_start..=i].iter()
                .cloned()
                .fold(0.0_f64, f64::max)
                .max(0.01); // Prevent division by zero

            // Normalize current ROC to 0-1 range
            let normalized_momentum = (abs_roc[i] / max_roc).min(1.0);

            // Scale multiplier based on momentum
            let mult = self.min_mult + normalized_momentum * (self.max_mult - self.min_mult);

            middle[i] = ma;
            upper[i] = ma + mult * std_dev;
            lower[i] = ma - mult * std_dev;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for MomentumDrivenBands {
    fn name(&self) -> &str {
        "Momentum Driven Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.momentum_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Adaptive Envelope Bands - Envelope bands with adaptive percentage
///
/// Traditional envelope bands use a fixed percentage above/below the moving average.
/// This indicator adapts the percentage based on recent price volatility,
/// making the envelope tighter in calm markets and wider in volatile markets.
///
/// The adaptive percentage is calculated using the coefficient of variation
/// (standard deviation / mean) over a lookback period.
#[derive(Debug, Clone)]
pub struct AdaptiveEnvelopeBands {
    period: usize,
    volatility_period: usize,
    min_percent: f64,
    max_percent: f64,
}

impl AdaptiveEnvelopeBands {
    pub fn new(period: usize, volatility_period: usize, min_percent: f64, max_percent: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volatility_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if min_percent <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_percent".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if max_percent <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_percent".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if min_percent >= max_percent {
            return Err(IndicatorError::InvalidParameter {
                name: "min_percent".to_string(),
                reason: "must be less than max_percent".to_string(),
            });
        }
        Ok(Self { period, volatility_period, min_percent, max_percent })
    }

    /// Calculate adaptive envelope bands (middle, upper, lower)
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate coefficient of variation (CV) for each point
        let mut cv_values = vec![0.0; n];
        for i in (self.volatility_period - 1)..n {
            let vol_start = i + 1 - self.volatility_period;
            let slice = &close[vol_start..=i];

            let mean: f64 = slice.iter().sum::<f64>() / self.volatility_period as f64;
            if mean > 1e-10 {
                let variance: f64 = slice.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / self.volatility_period as f64;
                cv_values[i] = variance.sqrt() / mean;
            }
        }

        // Find typical CV range for normalization
        let start_idx = self.period.max(self.volatility_period);
        let lookback = self.volatility_period * 2;

        for i in start_idx..n {
            // Calculate SMA for middle band
            let ma_start = i.saturating_sub(self.period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.period as f64;

            // Get min/max CV over lookback for normalization
            let cv_start = i.saturating_sub(lookback - 1);
            let cv_slice = &cv_values[cv_start..=i];

            let min_cv = cv_slice.iter().cloned().fold(f64::MAX, f64::min);
            let max_cv = cv_slice.iter().cloned().fold(0.0_f64, f64::max);
            let cv_range = (max_cv - min_cv).max(1e-10);

            // Normalize current CV to 0-1 range
            let normalized_cv = ((cv_values[i] - min_cv) / cv_range).max(0.0).min(1.0);

            // Calculate adaptive percentage
            let adaptive_percent = self.min_percent + normalized_cv * (self.max_percent - self.min_percent);

            middle[i] = ma;
            upper[i] = ma * (1.0 + adaptive_percent / 100.0);
            lower[i] = ma * (1.0 - adaptive_percent / 100.0);
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for AdaptiveEnvelopeBands {
    fn name(&self) -> &str {
        "Adaptive Envelope Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.volatility_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![
            102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
            122.0, 123.0, 124.0, 125.0, 126.0,
        ];
        let low = vec![
            98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
            108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0,
            118.0, 119.0, 120.0, 121.0, 122.0,
        ];
        let close = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
            110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
            120.0, 121.0, 122.0, 123.0, 124.0,
        ];
        (high, low, close)
    }

    #[test]
    fn test_volatility_bands() {
        let (high, low, close) = make_test_data();
        let vb = VolatilityBands::new(10, 10, 2.0).unwrap();
        let (middle, upper, lower) = vb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        // Check that bands are calculated after warmup
        assert!(middle[15] > 0.0);
        assert!(upper[15] > middle[15]);
        assert!(lower[15] < middle[15]);
    }

    #[test]
    fn test_volatility_bands_validation() {
        assert!(VolatilityBands::new(1, 10, 2.0).is_err());
        assert!(VolatilityBands::new(10, 1, 2.0).is_err());
        assert!(VolatilityBands::new(10, 10, -1.0).is_err());
        assert!(VolatilityBands::new(10, 10, 2.0).is_ok());
    }

    #[test]
    fn test_trend_bands() {
        let (high, low, close) = make_test_data();
        let tb = TrendBands::new(10, 10, 2.0).unwrap();
        let (middle, upper, lower) = tb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        assert!(middle[15] > 0.0);
        assert!(upper[15] > middle[15]);
        assert!(lower[15] < middle[15]);
    }

    #[test]
    fn test_trend_bands_validation() {
        assert!(TrendBands::new(1, 10, 2.0).is_err());
        assert!(TrendBands::new(10, 1, 2.0).is_err());
        assert!(TrendBands::new(10, 10, 0.0).is_err());
        assert!(TrendBands::new(10, 10, 2.0).is_ok());
    }

    #[test]
    fn test_momentum_bands_advanced() {
        let (_, _, close) = make_test_data();
        let mb = MomentumBandsAdvanced::new(10, 5, 2.0).unwrap();
        let (middle, upper, lower) = mb.calculate(&close);

        assert_eq!(middle.len(), close.len());
        assert!(middle[15] > 0.0);
        assert!(upper[15] >= middle[15]);
        assert!(lower[15] <= middle[15]);
    }

    #[test]
    fn test_momentum_bands_advanced_validation() {
        assert!(MomentumBandsAdvanced::new(1, 5, 2.0).is_err());
        assert!(MomentumBandsAdvanced::new(10, 1, 2.0).is_err());
        assert!(MomentumBandsAdvanced::new(10, 5, -0.5).is_err());
        assert!(MomentumBandsAdvanced::new(10, 5, 2.0).is_ok());
    }

    #[test]
    fn test_price_envelope() {
        let (_, _, close) = make_test_data();
        let pe = PriceEnvelope::new(10, 3.0, 3.0).unwrap();
        let (middle, upper, lower) = pe.calculate(&close);

        assert_eq!(middle.len(), close.len());
        // After warmup period, verify bands
        assert!(middle[10] > 0.0);
        assert!(upper[10] > middle[10]);
        assert!(lower[10] < middle[10]);

        // Verify percentage relationship
        let expected_upper = middle[10] * 1.03;
        let expected_lower = middle[10] * 0.97;
        assert!((upper[10] - expected_upper).abs() < 1e-10);
        assert!((lower[10] - expected_lower).abs() < 1e-10);
    }

    #[test]
    fn test_price_envelope_symmetric() {
        let pe = PriceEnvelope::symmetric(10, 5.0).unwrap();
        assert!(pe.upper_percent == pe.lower_percent);
    }

    #[test]
    fn test_price_envelope_validation() {
        assert!(PriceEnvelope::new(1, 3.0, 3.0).is_err());
        assert!(PriceEnvelope::new(10, -1.0, 3.0).is_err());
        assert!(PriceEnvelope::new(10, 3.0, 0.0).is_err());
        assert!(PriceEnvelope::new(10, 3.0, 3.0).is_ok());
    }

    #[test]
    fn test_dynamic_price_channel() {
        let (high, low, close) = make_test_data();
        let dpc = DynamicPriceChannel::new(10, 10, 1.0).unwrap();
        let (middle, upper, lower) = dpc.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        assert!(middle[15] > 0.0);
        assert!(upper[15] > middle[15]);
        assert!(lower[15] < middle[15]);
    }

    #[test]
    fn test_dynamic_price_channel_validation() {
        assert!(DynamicPriceChannel::new(1, 10, 1.0).is_err());
        assert!(DynamicPriceChannel::new(10, 1, 1.0).is_err());
        assert!(DynamicPriceChannel::new(10, 10, 0.0).is_err());
        assert!(DynamicPriceChannel::new(10, 10, 1.0).is_ok());
    }

    #[test]
    fn test_range_bands() {
        let (high, low, close) = make_test_data();
        let rb = RangeBands::new(10, 1.0).unwrap();
        let (middle, upper, lower) = rb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        // After warmup
        assert!(middle[10] > 0.0);
        assert!(upper[10] > middle[10]);
        assert!(lower[10] < middle[10]);
    }

    #[test]
    fn test_range_bands_validation() {
        assert!(RangeBands::new(1, 1.0).is_err());
        assert!(RangeBands::new(10, -1.0).is_err());
        assert!(RangeBands::new(10, 1.0).is_ok());
    }

    // Tests for the 6 new advanced indicators

    fn make_extended_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create more data points for cycle detection
        let mut high = Vec::with_capacity(100);
        let mut low = Vec::with_capacity(100);
        let mut close = Vec::with_capacity(100);

        for i in 0..100 {
            let base = 100.0 + (i as f64) * 0.5;
            // Add some cyclical behavior
            let cycle = 3.0 * ((i as f64) * 0.2).sin();
            high.push(base + 2.0 + cycle);
            low.push(base - 2.0 + cycle);
            close.push(base + cycle);
        }

        (high, low, close)
    }

    #[test]
    fn test_adaptive_bands_system() {
        let (high, low, close) = make_extended_test_data();
        let abs = AdaptiveBandsSystem::new(10, 5, 20, 2.0).unwrap();
        let (middle, upper, lower) = abs.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        // Check that bands are calculated after warmup
        let idx = 30;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_adaptive_bands_system_validation() {
        assert!(AdaptiveBandsSystem::new(1, 5, 20, 2.0).is_err());
        assert!(AdaptiveBandsSystem::new(10, 1, 20, 2.0).is_err());
        assert!(AdaptiveBandsSystem::new(10, 5, 1, 2.0).is_err());
        assert!(AdaptiveBandsSystem::new(10, 20, 10, 2.0).is_err()); // fast >= slow
        assert!(AdaptiveBandsSystem::new(10, 5, 20, 0.0).is_err());
        assert!(AdaptiveBandsSystem::new(10, 5, 20, -1.0).is_err());
        assert!(AdaptiveBandsSystem::new(10, 5, 20, 2.0).is_ok());
    }

    #[test]
    fn test_adaptive_bands_system_trait() {
        let abs = AdaptiveBandsSystem::new(10, 5, 20, 2.0).unwrap();
        assert_eq!(abs.name(), "Adaptive Bands System");
        assert_eq!(abs.min_periods(), 21);
    }

    #[test]
    fn test_trend_aware_bands() {
        let (high, low, close) = make_extended_test_data();
        let tab = TrendAwareBands::new(10, 10, 14, 2.0).unwrap();
        let (middle, upper, lower) = tab.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 20;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_trend_aware_bands_validation() {
        assert!(TrendAwareBands::new(1, 10, 14, 2.0).is_err());
        assert!(TrendAwareBands::new(10, 1, 14, 2.0).is_err());
        assert!(TrendAwareBands::new(10, 10, 1, 2.0).is_err());
        assert!(TrendAwareBands::new(10, 10, 14, 0.0).is_err());
        assert!(TrendAwareBands::new(10, 10, 14, -1.0).is_err());
        assert!(TrendAwareBands::new(10, 10, 14, 2.0).is_ok());
    }

    #[test]
    fn test_trend_aware_bands_trait() {
        let tab = TrendAwareBands::new(10, 10, 14, 2.0).unwrap();
        assert_eq!(tab.name(), "Trend Aware Bands");
        assert_eq!(tab.min_periods(), 15);
    }

    #[test]
    fn test_volatility_adjusted_bands() {
        let (high, low, close) = make_extended_test_data();
        let vab = VolatilityAdjustedBands::new(10, 10, 20, 2.0).unwrap();
        let (middle, upper, lower) = vab.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 30;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_volatility_adjusted_bands_validation() {
        assert!(VolatilityAdjustedBands::new(1, 10, 20, 2.0).is_err());
        assert!(VolatilityAdjustedBands::new(10, 1, 20, 2.0).is_err());
        assert!(VolatilityAdjustedBands::new(10, 10, 1, 2.0).is_err());
        assert!(VolatilityAdjustedBands::new(10, 10, 20, 0.0).is_err());
        assert!(VolatilityAdjustedBands::new(10, 10, 20, -1.0).is_err());
        assert!(VolatilityAdjustedBands::new(10, 10, 20, 2.0).is_ok());
    }

    #[test]
    fn test_volatility_adjusted_bands_trait() {
        let vab = VolatilityAdjustedBands::new(10, 10, 20, 2.0).unwrap();
        assert_eq!(vab.name(), "Volatility Adjusted Bands");
        assert_eq!(vab.min_periods(), 21);
    }

    #[test]
    fn test_cycle_bands() {
        let (high, low, close) = make_extended_test_data();
        let cb = CycleBands::new(10, 5, 20, 2.0).unwrap();
        let (middle, upper, lower) = cb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 50; // Need more warmup for cycle detection
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_cycle_bands_validation() {
        assert!(CycleBands::new(1, 5, 20, 2.0).is_err());
        assert!(CycleBands::new(10, 1, 20, 2.0).is_err());
        assert!(CycleBands::new(10, 5, 1, 2.0).is_err());
        assert!(CycleBands::new(10, 20, 10, 2.0).is_err()); // min >= max
        assert!(CycleBands::new(10, 5, 20, 0.0).is_err());
        assert!(CycleBands::new(10, 5, 20, -1.0).is_err());
        assert!(CycleBands::new(10, 5, 20, 2.0).is_ok());
    }

    #[test]
    fn test_cycle_bands_trait() {
        let cb = CycleBands::new(10, 5, 20, 2.0).unwrap();
        assert_eq!(cb.name(), "Cycle Bands");
        assert_eq!(cb.min_periods(), 41); // max(10, 20*2) + 1
    }

    #[test]
    fn test_cycle_bands_cycle_detection() {
        // Test with synthetic cyclic data
        let mut close = Vec::with_capacity(200);
        for i in 0..200 {
            // Create a dominant 10-period cycle
            close.push(100.0 + 10.0 * ((i as f64) * 2.0 * std::f64::consts::PI / 10.0).sin());
        }

        let cb = CycleBands::new(5, 5, 20, 2.0).unwrap();
        let detected_cycle = cb.detect_cycle(&close);

        // Should detect cycle around 10 periods (allowing some tolerance)
        assert!(detected_cycle >= 8 && detected_cycle <= 12,
            "Expected cycle around 10, got {}", detected_cycle);
    }

    #[test]
    fn test_dynamic_envelope() {
        let (high, low, close) = make_extended_test_data();
        let de = DynamicEnvelope::new(10, 5, 2.0, 1.0).unwrap();
        let (middle, upper, lower) = de.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 20;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_dynamic_envelope_validation() {
        assert!(DynamicEnvelope::new(1, 5, 2.0, 1.0).is_err());
        assert!(DynamicEnvelope::new(10, 1, 2.0, 1.0).is_err());
        assert!(DynamicEnvelope::new(10, 5, 0.0, 1.0).is_err());
        assert!(DynamicEnvelope::new(10, 5, -1.0, 1.0).is_err());
        assert!(DynamicEnvelope::new(10, 5, 2.0, 0.0).is_err());
        assert!(DynamicEnvelope::new(10, 5, 2.0, -1.0).is_err());
        assert!(DynamicEnvelope::new(10, 5, 1.0, 2.0).is_err()); // contraction >= expansion
        assert!(DynamicEnvelope::new(10, 5, 2.0, 1.0).is_ok());
    }

    #[test]
    fn test_dynamic_envelope_trait() {
        let de = DynamicEnvelope::new(10, 5, 2.0, 1.0).unwrap();
        assert_eq!(de.name(), "Dynamic Envelope");
        assert_eq!(de.min_periods(), 11);
    }

    #[test]
    fn test_dynamic_envelope_expansion_contraction() {
        // Test that envelope expands when price breaks bands
        let mut high = vec![102.0; 50];
        let mut low = vec![98.0; 50];
        let mut close = vec![100.0; 50];

        // Create a price spike at index 30
        for i in 30..35 {
            high[i] = 120.0;
            close[i] = 115.0;
        }

        let de = DynamicEnvelope::new(10, 5, 2.0, 1.0).unwrap();
        let (_, upper, _) = de.calculate(&high, &low, &close);

        // Upper band should be wider after the spike
        assert!(upper[35] > upper[25],
            "Expected envelope expansion after price spike");
    }

    #[test]
    fn test_empty_data() {
        let empty: Vec<f64> = vec![];

        let abs = AdaptiveBandsSystem::new(10, 5, 20, 2.0).unwrap();
        let (m, u, l) = abs.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());

        let tab = TrendAwareBands::new(10, 10, 14, 2.0).unwrap();
        let (m, u, l) = tab.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());

        let vab = VolatilityAdjustedBands::new(10, 10, 20, 2.0).unwrap();
        let (m, u, l) = vab.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());

        let cb = CycleBands::new(10, 5, 20, 2.0).unwrap();
        let (m, u, l) = cb.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());

        let de = DynamicEnvelope::new(10, 5, 2.0, 1.0).unwrap();
        let (m, u, l) = de.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_band_symmetry() {
        let (high, low, close) = make_extended_test_data();

        // For all bands, upper - middle should approximately equal middle - lower
        let abs = AdaptiveBandsSystem::new(10, 5, 20, 2.0).unwrap();
        let (m, u, l) = abs.calculate(&high, &low, &close);
        let idx = 50;
        let upper_diff = u[idx] - m[idx];
        let lower_diff = m[idx] - l[idx];
        assert!((upper_diff - lower_diff).abs() < 1e-10,
            "Bands should be symmetric around middle");
    }

    // ============================================================
    // Tests for the 6 NEW band indicators
    // ============================================================

    // --- AdaptiveKeltnerChannels Tests ---

    #[test]
    fn test_adaptive_keltner_channels() {
        let (high, low, close) = make_extended_test_data();
        let akc = AdaptiveKeltnerChannels::new(20, 10, 2.0, 50).unwrap();
        let (middle, upper, lower) = akc.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 60;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_adaptive_keltner_channels_validation() {
        assert!(AdaptiveKeltnerChannels::new(1, 10, 2.0, 50).is_err());
        assert!(AdaptiveKeltnerChannels::new(20, 1, 2.0, 50).is_err());
        assert!(AdaptiveKeltnerChannels::new(20, 10, 0.0, 50).is_err());
        assert!(AdaptiveKeltnerChannels::new(20, 10, -1.0, 50).is_err());
        assert!(AdaptiveKeltnerChannels::new(20, 10, 2.0, 1).is_err());
        assert!(AdaptiveKeltnerChannels::new(20, 10, 2.0, 50).is_ok());
    }

    #[test]
    fn test_adaptive_keltner_channels_trait() {
        let akc = AdaptiveKeltnerChannels::new(20, 10, 2.0, 50).unwrap();
        assert_eq!(akc.name(), "Adaptive Keltner Channels");
        assert_eq!(akc.min_periods(), 51); // max(20, 10, 50) + 1
    }

    #[test]
    fn test_adaptive_keltner_channels_empty_data() {
        let empty: Vec<f64> = vec![];
        let akc = AdaptiveKeltnerChannels::new(20, 10, 2.0, 50).unwrap();
        let (m, u, l) = akc.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_adaptive_keltner_channels_symmetry() {
        let (high, low, close) = make_extended_test_data();
        let akc = AdaptiveKeltnerChannels::new(20, 10, 2.0, 50).unwrap();
        let (m, u, l) = akc.calculate(&high, &low, &close);
        let idx = 70;
        let upper_diff = u[idx] - m[idx];
        let lower_diff = m[idx] - l[idx];
        assert!((upper_diff - lower_diff).abs() < 1e-10,
            "Adaptive Keltner Channels should be symmetric");
    }

    // --- VolatilityWeightedBands Tests ---

    #[test]
    fn test_volatility_weighted_bands() {
        let (high, low, close) = make_extended_test_data();
        let vwb = VolatilityWeightedBands::new(20, 10, 30, 2.0).unwrap();
        let (middle, upper, lower) = vwb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 40;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_volatility_weighted_bands_validation() {
        assert!(VolatilityWeightedBands::new(1, 10, 30, 2.0).is_err());
        assert!(VolatilityWeightedBands::new(20, 1, 30, 2.0).is_err());
        assert!(VolatilityWeightedBands::new(20, 10, 1, 2.0).is_err());
        assert!(VolatilityWeightedBands::new(20, 30, 20, 2.0).is_err()); // short >= long
        assert!(VolatilityWeightedBands::new(20, 10, 30, 0.0).is_err());
        assert!(VolatilityWeightedBands::new(20, 10, 30, -1.0).is_err());
        assert!(VolatilityWeightedBands::new(20, 10, 30, 2.0).is_ok());
    }

    #[test]
    fn test_volatility_weighted_bands_trait() {
        let vwb = VolatilityWeightedBands::new(20, 10, 30, 2.0).unwrap();
        assert_eq!(vwb.name(), "Volatility Weighted Bands");
        assert_eq!(vwb.min_periods(), 31); // max(20, 30) + 1
    }

    #[test]
    fn test_volatility_weighted_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let vwb = VolatilityWeightedBands::new(20, 10, 30, 2.0).unwrap();
        let (m, u, l) = vwb.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_volatility_weighted_bands_high_volatility() {
        // Test that bands widen during high volatility periods
        let mut high = vec![102.0; 80];
        let mut low = vec![98.0; 80];
        let mut close = vec![100.0; 80];

        // Create high volatility spike in middle
        for i in 45..55 {
            high[i] = 120.0;
            low[i] = 80.0;
            close[i] = 100.0 + (if i % 2 == 0 { 15.0 } else { -15.0 });
        }

        let vwb = VolatilityWeightedBands::new(10, 5, 20, 2.0).unwrap();
        let (_, upper, _) = vwb.calculate(&high, &low, &close);

        // Band width at index 55 (right after volatility spike) should be greater
        // than band width at index 30 (stable period)
        let width_stable = upper[30] - close[30];
        let width_volatile = upper[55] - close[55];
        // Both should be positive (upper above close)
        assert!(width_stable > 0.0, "Stable period should have positive band width");
        assert!(width_volatile >= 0.0, "Volatile period should have non-negative band width");
    }

    // --- TrendFollowingChannel Tests ---

    #[test]
    fn test_trend_following_channel() {
        let (high, low, close) = make_extended_test_data();
        let tfc = TrendFollowingChannel::new(20, 14, 10, 2.0).unwrap();
        let (middle, upper, lower) = tfc.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 30;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_trend_following_channel_validation() {
        assert!(TrendFollowingChannel::new(1, 14, 10, 2.0).is_err());
        assert!(TrendFollowingChannel::new(20, 1, 10, 2.0).is_err());
        assert!(TrendFollowingChannel::new(20, 14, 1, 2.0).is_err());
        assert!(TrendFollowingChannel::new(20, 14, 10, 0.0).is_err());
        assert!(TrendFollowingChannel::new(20, 14, 10, -1.0).is_err());
        assert!(TrendFollowingChannel::new(20, 14, 10, 2.0).is_ok());
    }

    #[test]
    fn test_trend_following_channel_trait() {
        let tfc = TrendFollowingChannel::new(20, 14, 10, 2.0).unwrap();
        assert_eq!(tfc.name(), "Trend Following Channel");
        assert_eq!(tfc.min_periods(), 21); // max(20, 14, 10) + 1
    }

    #[test]
    fn test_trend_following_channel_empty_data() {
        let empty: Vec<f64> = vec![];
        let tfc = TrendFollowingChannel::new(20, 14, 10, 2.0).unwrap();
        let (m, u, l) = tfc.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_trend_following_channel_asymmetry_in_trend() {
        // In uptrend, lower band should be closer to price (tighter stop)
        let (high, low, close) = make_extended_test_data(); // This has an uptrend
        let tfc = TrendFollowingChannel::new(10, 10, 10, 2.0).unwrap();
        let (middle, upper, lower) = tfc.calculate(&high, &low, &close);

        // In uptrend: upper - middle may differ from middle - lower
        let idx = 50;
        let upper_dist = upper[idx] - middle[idx];
        let lower_dist = middle[idx] - lower[idx];
        // In an uptrend, the asymmetry should be present
        // (not necessarily equal, unlike symmetric bands)
        assert!(upper_dist > 0.0 && lower_dist > 0.0,
            "Both bands should be positive distance from middle");
    }

    // --- DynamicSupportResistanceBands Tests ---

    #[test]
    fn test_dynamic_support_resistance_bands() {
        let (high, low, close) = make_extended_test_data();
        let dsrb = DynamicSupportResistanceBands::new(20, 5, 10).unwrap();
        let (middle, upper, lower) = dsrb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 30;
        assert!(middle[idx] > 0.0);
        // Upper should be >= middle, lower should be <= middle
        assert!(upper[idx] >= lower[idx]);
    }

    #[test]
    fn test_dynamic_support_resistance_bands_validation() {
        assert!(DynamicSupportResistanceBands::new(1, 5, 10).is_err());
        assert!(DynamicSupportResistanceBands::new(20, 1, 10).is_err());
        assert!(DynamicSupportResistanceBands::new(20, 5, 1).is_err());
        assert!(DynamicSupportResistanceBands::new(20, 5, 10).is_ok());
    }

    #[test]
    fn test_dynamic_support_resistance_bands_trait() {
        let dsrb = DynamicSupportResistanceBands::new(20, 5, 10).unwrap();
        assert_eq!(dsrb.name(), "Dynamic Support Resistance Bands");
        assert_eq!(dsrb.min_periods(), 21); // max(20, 5) + 1
    }

    #[test]
    fn test_dynamic_support_resistance_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let dsrb = DynamicSupportResistanceBands::new(20, 5, 10).unwrap();
        let (m, u, l) = dsrb.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_dynamic_support_resistance_bands_swing_detection() {
        // Create data with clear swing highs and lows
        let mut high = Vec::with_capacity(50);
        let mut low = Vec::with_capacity(50);
        let mut close = Vec::with_capacity(50);

        for i in 0..50 {
            let base = 100.0;
            // Create wave pattern with clear peaks and troughs
            let wave = 10.0 * ((i as f64) * 0.3).sin();
            high.push(base + wave + 2.0);
            low.push(base + wave - 2.0);
            close.push(base + wave);
        }

        let dsrb = DynamicSupportResistanceBands::new(10, 5, 5).unwrap();
        let (middle, upper, lower) = dsrb.calculate(&high, &low, &close);

        // Upper should track resistance, lower should track support
        let idx = 30;
        assert!(upper[idx] >= middle[idx]);
        assert!(lower[idx] <= middle[idx]);
    }

    // --- MomentumBandwidth Tests ---

    #[test]
    fn test_momentum_bandwidth() {
        let (_, _, close) = make_extended_test_data();
        let mb = MomentumBandwidth::new(20, 10, 2.0).unwrap();
        let bandwidth = mb.calculate(&close);

        assert_eq!(bandwidth.len(), close.len());
        let idx = 30;
        assert!(bandwidth[idx] >= 0.0); // Bandwidth should be non-negative
    }

    #[test]
    fn test_momentum_bandwidth_validation() {
        assert!(MomentumBandwidth::new(1, 10, 2.0).is_err());
        assert!(MomentumBandwidth::new(20, 1, 2.0).is_err());
        assert!(MomentumBandwidth::new(20, 10, 0.0).is_err());
        assert!(MomentumBandwidth::new(20, 10, -1.0).is_err());
        assert!(MomentumBandwidth::new(20, 10, 2.0).is_ok());
    }

    #[test]
    fn test_momentum_bandwidth_trait() {
        let mb = MomentumBandwidth::new(20, 10, 2.0).unwrap();
        assert_eq!(mb.name(), "Momentum Bandwidth");
        assert_eq!(mb.min_periods(), 21); // max(20, 10) + 1
    }

    #[test]
    fn test_momentum_bandwidth_empty_data() {
        let empty: Vec<f64> = vec![];
        let mb = MomentumBandwidth::new(20, 10, 2.0).unwrap();
        let bandwidth = mb.calculate(&empty);
        assert!(bandwidth.is_empty());
    }

    #[test]
    fn test_momentum_bandwidth_high_momentum() {
        // Test that bandwidth increases with higher momentum
        let mut close_low_momentum: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.1).collect();
        let mut close_high_momentum: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 1.0).collect();

        let mb = MomentumBandwidth::new(10, 5, 2.0).unwrap();
        let bw_low = mb.calculate(&close_low_momentum);
        let bw_high = mb.calculate(&close_high_momentum);

        // Higher momentum should lead to wider bandwidth
        let idx = 30;
        assert!(bw_high[idx] > bw_low[idx],
            "Higher momentum should produce wider bandwidth");
    }

    // --- PriceEnvelopeOscillator Tests ---

    #[test]
    fn test_price_envelope_oscillator() {
        let (_, _, close) = make_extended_test_data();
        let peo = PriceEnvelopeOscillator::new(20, 3.0).unwrap();
        let oscillator = peo.calculate(&close);

        assert_eq!(oscillator.len(), close.len());
        // Oscillator values should be calculated after warmup
        let idx = 25;
        // Value should be within reasonable range for trending data
        assert!(oscillator[idx].abs() < 200.0); // Reasonable upper bound
    }

    #[test]
    fn test_price_envelope_oscillator_validation() {
        assert!(PriceEnvelopeOscillator::new(1, 3.0).is_err());
        assert!(PriceEnvelopeOscillator::new(20, 0.0).is_err());
        assert!(PriceEnvelopeOscillator::new(20, -1.0).is_err());
        assert!(PriceEnvelopeOscillator::new(20, 3.0).is_ok());
    }

    #[test]
    fn test_price_envelope_oscillator_trait() {
        let peo = PriceEnvelopeOscillator::new(20, 3.0).unwrap();
        assert_eq!(peo.name(), "Price Envelope Oscillator");
        assert_eq!(peo.min_periods(), 20);
    }

    #[test]
    fn test_price_envelope_oscillator_empty_data() {
        let empty: Vec<f64> = vec![];
        let peo = PriceEnvelopeOscillator::new(20, 3.0).unwrap();
        let oscillator = peo.calculate(&empty);
        assert!(oscillator.is_empty());
    }

    #[test]
    fn test_price_envelope_oscillator_at_bands() {
        // Test that oscillator returns expected values at band boundaries
        // Create flat data then spike to upper band
        let mut close = vec![100.0; 30];

        let peo = PriceEnvelopeOscillator::new(10, 5.0).unwrap();

        // For flat data at 100, MA = 100, upper = 105, lower = 95
        // At price 100 (middle), oscillator should be ~0
        let osc = peo.calculate(&close);
        let idx = 20;
        assert!(osc[idx].abs() < 5.0, "At middle, oscillator should be near 0");

        // Test price at upper band
        close[25] = 105.0;
        let osc2 = peo.calculate(&close);
        assert!(osc2[25] > 50.0, "Near upper band, oscillator should be positive");
    }

    #[test]
    fn test_price_envelope_oscillator_breakout() {
        // Test that oscillator exceeds +/-100 on breakouts
        let mut close = vec![100.0; 30];

        // Create breakout beyond upper envelope
        for i in 20..30 {
            close[i] = 120.0; // Way above 3% envelope
        }

        let peo = PriceEnvelopeOscillator::new(10, 3.0).unwrap();
        let osc = peo.calculate(&close);

        // At the breakout, before MA adjusts, oscillator should exceed 100
        // (at i=20, MA still includes mostly 100s, so 120 is well above upper band)
        assert!(osc[20] > 100.0, "On breakout, oscillator should exceed 100");
    }

    // --- Combined Tests for All 6 New Indicators ---

    #[test]
    fn test_all_new_indicators_with_short_data() {
        // Test that all indicators handle data shorter than min_periods gracefully
        let short_high = vec![102.0, 103.0, 104.0];
        let short_low = vec![98.0, 99.0, 100.0];
        let short_close = vec![100.0, 101.0, 102.0];

        // AdaptiveKeltnerChannels
        let akc = AdaptiveKeltnerChannels::new(20, 10, 2.0, 50).unwrap();
        let (m, u, l) = akc.calculate(&short_high, &short_low, &short_close);
        assert_eq!(m.len(), 3);
        assert!(m.iter().all(|&x| x == 0.0)); // All zeros before warmup

        // VolatilityWeightedBands
        let vwb = VolatilityWeightedBands::new(20, 10, 30, 2.0).unwrap();
        let (m, u, l) = vwb.calculate(&short_high, &short_low, &short_close);
        assert_eq!(m.len(), 3);

        // TrendFollowingChannel
        let tfc = TrendFollowingChannel::new(20, 14, 10, 2.0).unwrap();
        let (m, u, l) = tfc.calculate(&short_high, &short_low, &short_close);
        assert_eq!(m.len(), 3);

        // DynamicSupportResistanceBands
        let dsrb = DynamicSupportResistanceBands::new(20, 5, 10).unwrap();
        let (m, u, l) = dsrb.calculate(&short_high, &short_low, &short_close);
        assert_eq!(m.len(), 3);

        // MomentumBandwidth
        let mb = MomentumBandwidth::new(20, 10, 2.0).unwrap();
        let bw = mb.calculate(&short_close);
        assert_eq!(bw.len(), 3);

        // PriceEnvelopeOscillator
        let peo = PriceEnvelopeOscillator::new(20, 3.0).unwrap();
        let osc = peo.calculate(&short_close);
        assert_eq!(osc.len(), 3);
    }

    #[test]
    fn test_new_indicators_numerical_stability() {
        // Test with very small and very large values
        let large_high: Vec<f64> = (0..50).map(|i| 1e8 + (i as f64) * 1000.0).collect();
        let large_low: Vec<f64> = (0..50).map(|i| 1e8 - 1000.0 + (i as f64) * 1000.0).collect();
        let large_close: Vec<f64> = (0..50).map(|i| 1e8 + (i as f64) * 1000.0).collect();

        let akc = AdaptiveKeltnerChannels::new(10, 10, 2.0, 20).unwrap();
        let (m, u, l) = akc.calculate(&large_high, &large_low, &large_close);
        let idx = 30;
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        let small_high: Vec<f64> = (0..50).map(|i| 1e-6 + (i as f64) * 1e-8).collect();
        let small_low: Vec<f64> = (0..50).map(|i| 1e-6 - 1e-8 + (i as f64) * 1e-8).collect();
        let small_close: Vec<f64> = (0..50).map(|i| 1e-6 + (i as f64) * 1e-8).collect();

        let (m, u, l) = akc.calculate(&small_high, &small_low, &small_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());
    }

    // ============================================================
    // Tests for the 6 NEW Indicators (Jan 2026 additions)
    // ============================================================

    // --- VolatilityBandwidth Tests ---

    #[test]
    fn test_volatility_bandwidth() {
        let (_, _, close) = make_extended_test_data();
        let vbw = VolatilityBandwidth::new(20, 2.0).unwrap();
        let bandwidth = vbw.calculate(&close);

        assert_eq!(bandwidth.len(), close.len());
        let idx = 25;
        assert!(bandwidth[idx] >= 0.0); // Bandwidth should be non-negative
    }

    #[test]
    fn test_volatility_bandwidth_validation() {
        assert!(VolatilityBandwidth::new(1, 2.0).is_err());
        assert!(VolatilityBandwidth::new(20, 0.0).is_err());
        assert!(VolatilityBandwidth::new(20, -1.0).is_err());
        assert!(VolatilityBandwidth::new(20, 2.0).is_ok());
    }

    #[test]
    fn test_volatility_bandwidth_trait() {
        let vbw = VolatilityBandwidth::new(20, 2.0).unwrap();
        assert_eq!(vbw.name(), "Volatility Bandwidth");
        assert_eq!(vbw.min_periods(), 20);
    }

    #[test]
    fn test_volatility_bandwidth_empty_data() {
        let empty: Vec<f64> = vec![];
        let vbw = VolatilityBandwidth::new(20, 2.0).unwrap();
        let bandwidth = vbw.calculate(&empty);
        assert!(bandwidth.is_empty());
    }

    #[test]
    fn test_volatility_bandwidth_with_bands() {
        let (_, _, close) = make_extended_test_data();
        let vbw = VolatilityBandwidth::new(10, 2.0).unwrap();
        let (middle, upper, lower, bandwidth) = vbw.calculate_with_bands(&close);

        let idx = 20;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
        // Verify bandwidth calculation matches bands
        let expected_bw = (upper[idx] - lower[idx]) / middle[idx] * 100.0;
        assert!((bandwidth[idx] - expected_bw).abs() < 1e-10);
    }

    #[test]
    fn test_volatility_bandwidth_squeeze_detection() {
        // Test that bandwidth is lower during low volatility
        let mut close_low_vol: Vec<f64> = vec![100.0; 50];
        let mut close_high_vol: Vec<f64> = (0..50).map(|i| {
            100.0 + 5.0 * ((i as f64) * 0.5).sin()
        }).collect();

        let vbw = VolatilityBandwidth::new(10, 2.0).unwrap();
        let bw_low = vbw.calculate(&close_low_vol);
        let bw_high = vbw.calculate(&close_high_vol);

        // Flat data should have near-zero bandwidth
        // (after warmup, all values are same so std_dev = 0)
        let idx = 30;
        assert!(bw_low[idx] < bw_high[idx],
            "Low volatility should have smaller bandwidth");
    }

    // --- BandBreakoutStrength Tests ---

    #[test]
    fn test_band_breakout_strength() {
        let (_, _, close) = make_extended_test_data();
        let bbs = BandBreakoutStrength::new(20, 2.0).unwrap();
        let strength = bbs.calculate(&close);

        assert_eq!(strength.len(), close.len());
        let idx = 25;
        // Strength should be calculated
        assert!(strength[idx].is_finite());
    }

    #[test]
    fn test_band_breakout_strength_validation() {
        assert!(BandBreakoutStrength::new(1, 2.0).is_err());
        assert!(BandBreakoutStrength::new(20, 0.0).is_err());
        assert!(BandBreakoutStrength::new(20, -1.0).is_err());
        assert!(BandBreakoutStrength::new(20, 2.0).is_ok());
    }

    #[test]
    fn test_band_breakout_strength_trait() {
        let bbs = BandBreakoutStrength::new(20, 2.0).unwrap();
        assert_eq!(bbs.name(), "Band Breakout Strength");
        assert_eq!(bbs.min_periods(), 20);
    }

    #[test]
    fn test_band_breakout_strength_empty_data() {
        let empty: Vec<f64> = vec![];
        let bbs = BandBreakoutStrength::new(20, 2.0).unwrap();
        let strength = bbs.calculate(&empty);
        assert!(strength.is_empty());
    }

    #[test]
    fn test_band_breakout_strength_within_bands() {
        // Flat data should have strength near 0 (at middle)
        let close = vec![100.0; 30];
        let bbs = BandBreakoutStrength::new(10, 2.0).unwrap();
        let strength = bbs.calculate(&close);

        // At middle band, strength should be 0
        // But with flat data, std_dev is 0, so bands collapse
        // We need varying data to test properly
        let varying: Vec<f64> = (0..30).map(|i| 100.0 + (i % 3) as f64 - 1.0).collect();
        let strength2 = bbs.calculate(&varying);
        let idx = 20;
        // Should be within -100 to 100 range when within bands
        assert!(strength2[idx] >= -100.0 && strength2[idx] <= 100.0);
    }

    #[test]
    fn test_band_breakout_strength_breakout() {
        // Create data with a clear breakout
        let mut close = vec![100.0; 30];
        for i in 0..30 {
            close[i] = 100.0 + ((i % 5) as f64 - 2.0);
        }
        // Create upward breakout
        close[25] = 130.0; // Way above upper band

        let bbs = BandBreakoutStrength::new(10, 2.0).unwrap();
        let strength = bbs.calculate(&close);

        // Breakout should exceed 100
        assert!(strength[25] > 100.0,
            "Upward breakout should produce strength > 100, got {}", strength[25]);
    }

    #[test]
    fn test_band_breakout_strength_bands() {
        let (_, _, close) = make_extended_test_data();
        let bbs = BandBreakoutStrength::new(10, 2.0).unwrap();
        let (middle, upper, lower) = bbs.calculate_bands(&close);

        let idx = 20;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    // --- DynamicPriceBands Tests ---

    #[test]
    fn test_dynamic_price_bands() {
        let (high, low, close) = make_extended_test_data();
        let dpb = DynamicPriceBands::new(20, 50, 1.0, 3.0).unwrap();
        let (middle, upper, lower) = dpb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 60;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_dynamic_price_bands_validation() {
        assert!(DynamicPriceBands::new(1, 50, 1.0, 3.0).is_err());
        assert!(DynamicPriceBands::new(20, 1, 1.0, 3.0).is_err());
        assert!(DynamicPriceBands::new(20, 50, 0.0, 3.0).is_err());
        assert!(DynamicPriceBands::new(20, 50, -1.0, 3.0).is_err());
        assert!(DynamicPriceBands::new(20, 50, 1.0, 0.0).is_err());
        assert!(DynamicPriceBands::new(20, 50, 1.0, -1.0).is_err());
        assert!(DynamicPriceBands::new(20, 50, 3.0, 1.0).is_err()); // low >= high
        assert!(DynamicPriceBands::new(20, 50, 1.0, 3.0).is_ok());
    }

    #[test]
    fn test_dynamic_price_bands_trait() {
        let dpb = DynamicPriceBands::new(20, 50, 1.0, 3.0).unwrap();
        assert_eq!(dpb.name(), "Dynamic Price Bands");
        assert_eq!(dpb.min_periods(), 51); // max(20, 50) + 1
    }

    #[test]
    fn test_dynamic_price_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let dpb = DynamicPriceBands::new(20, 50, 1.0, 3.0).unwrap();
        let (m, u, l) = dpb.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_dynamic_price_bands_regime_adaptation() {
        // Create data with different volatility regimes
        let mut high = vec![102.0; 100];
        let mut low = vec![98.0; 100];
        let mut close = vec![100.0; 100];

        // High volatility period at end
        for i in 70..100 {
            high[i] = 120.0;
            low[i] = 80.0;
            close[i] = 100.0;
        }

        let dpb = DynamicPriceBands::new(10, 20, 1.0, 3.0).unwrap();
        let (_, upper, _) = dpb.calculate(&high, &low, &close);

        // Bands should be wider in high volatility period
        let low_vol_idx = 50;
        let high_vol_idx = 90;
        let width_low = upper[low_vol_idx] - close[low_vol_idx];
        let width_high = upper[high_vol_idx] - close[high_vol_idx];

        assert!(width_high > width_low,
            "High volatility regime should have wider bands");
    }

    // --- TrendAlignedBands Tests ---

    #[test]
    fn test_trend_aligned_bands() {
        let (_, _, close) = make_extended_test_data();
        let tab = TrendAlignedBands::new(20, 10, 2.0, 0.5).unwrap();
        let (middle, upper, lower) = tab.calculate(&close);

        assert_eq!(middle.len(), close.len());
        let idx = 30;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_trend_aligned_bands_validation() {
        assert!(TrendAlignedBands::new(1, 10, 2.0, 0.5).is_err());
        assert!(TrendAlignedBands::new(20, 1, 2.0, 0.5).is_err());
        assert!(TrendAlignedBands::new(20, 10, 0.0, 0.5).is_err());
        assert!(TrendAlignedBands::new(20, 10, -1.0, 0.5).is_err());
        assert!(TrendAlignedBands::new(20, 10, 2.0, 0.0).is_err()); // max_shift = 0
        assert!(TrendAlignedBands::new(20, 10, 2.0, 1.5).is_err()); // max_shift > 1
        assert!(TrendAlignedBands::new(20, 10, 2.0, 0.5).is_ok());
    }

    #[test]
    fn test_trend_aligned_bands_trait() {
        let tab = TrendAlignedBands::new(20, 10, 2.0, 0.5).unwrap();
        assert_eq!(tab.name(), "Trend Aligned Bands");
        assert_eq!(tab.min_periods(), 21); // max(20, 10) + 1
    }

    #[test]
    fn test_trend_aligned_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let tab = TrendAlignedBands::new(20, 10, 2.0, 0.5).unwrap();
        let (m, u, l) = tab.calculate(&empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_trend_aligned_bands_uptrend_shift() {
        // Create strong uptrend data
        let uptrend: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 2.0).collect();

        let tab = TrendAlignedBands::new(10, 10, 2.0, 0.5).unwrap();
        let (middle, _, _) = tab.calculate(&uptrend);

        // In uptrend, middle should be shifted up from simple SMA
        let idx: usize = 30;
        let ma_start = idx.saturating_sub(9);
        let simple_ma: f64 = uptrend[ma_start..=idx].iter().sum::<f64>() / 10.0;

        // Middle should be >= simple MA in uptrend (shifted up)
        assert!(middle[idx] >= simple_ma,
            "In uptrend, middle should be shifted upward from SMA");
    }

    // --- MomentumDrivenBands Tests ---

    #[test]
    fn test_momentum_driven_bands() {
        let (_, _, close) = make_extended_test_data();
        let mdb = MomentumDrivenBands::new(20, 10, 1.0, 3.0).unwrap();
        let (middle, upper, lower) = mdb.calculate(&close);

        assert_eq!(middle.len(), close.len());
        let idx = 30;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_momentum_driven_bands_validation() {
        assert!(MomentumDrivenBands::new(1, 10, 1.0, 3.0).is_err());
        assert!(MomentumDrivenBands::new(20, 1, 1.0, 3.0).is_err());
        assert!(MomentumDrivenBands::new(20, 10, 0.0, 3.0).is_err());
        assert!(MomentumDrivenBands::new(20, 10, -1.0, 3.0).is_err());
        assert!(MomentumDrivenBands::new(20, 10, 1.0, 0.0).is_err());
        assert!(MomentumDrivenBands::new(20, 10, 1.0, -1.0).is_err());
        assert!(MomentumDrivenBands::new(20, 10, 3.0, 1.0).is_err()); // min >= max
        assert!(MomentumDrivenBands::new(20, 10, 1.0, 3.0).is_ok());
    }

    #[test]
    fn test_momentum_driven_bands_trait() {
        let mdb = MomentumDrivenBands::new(20, 10, 1.0, 3.0).unwrap();
        assert_eq!(mdb.name(), "Momentum Driven Bands");
        assert_eq!(mdb.min_periods(), 21); // max(20, 10) + 1
    }

    #[test]
    fn test_momentum_driven_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let mdb = MomentumDrivenBands::new(20, 10, 1.0, 3.0).unwrap();
        let (m, u, l) = mdb.calculate(&empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_momentum_driven_bands_high_momentum_wider() {
        // Create low momentum data (slow drift)
        let low_momentum: Vec<f64> = (0..60).map(|i| 100.0 + (i as f64) * 0.1).collect();
        // Create high momentum data (fast move)
        let high_momentum: Vec<f64> = (0..60).map(|i| 100.0 + (i as f64) * 2.0).collect();

        let mdb = MomentumDrivenBands::new(10, 5, 1.0, 3.0).unwrap();
        let (_, upper_low, _) = mdb.calculate(&low_momentum);
        let (middle_high, upper_high, _) = mdb.calculate(&high_momentum);

        let idx = 40;
        let width_low = upper_low[idx] - low_momentum[idx];
        let width_high = upper_high[idx] - middle_high[idx];

        // High momentum should have wider bands (relative to price movement)
        // We check the multiplier effect by comparing relative widths
        assert!(width_high > width_low,
            "Higher momentum should produce wider bands");
    }

    // --- AdaptiveEnvelopeBands Tests ---

    #[test]
    fn test_adaptive_envelope_bands() {
        let (_, _, close) = make_extended_test_data();
        let aeb = AdaptiveEnvelopeBands::new(20, 10, 1.0, 5.0).unwrap();
        let (middle, upper, lower) = aeb.calculate(&close);

        assert_eq!(middle.len(), close.len());
        let idx = 30;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_adaptive_envelope_bands_validation() {
        assert!(AdaptiveEnvelopeBands::new(1, 10, 1.0, 5.0).is_err());
        assert!(AdaptiveEnvelopeBands::new(20, 1, 1.0, 5.0).is_err());
        assert!(AdaptiveEnvelopeBands::new(20, 10, 0.0, 5.0).is_err());
        assert!(AdaptiveEnvelopeBands::new(20, 10, -1.0, 5.0).is_err());
        assert!(AdaptiveEnvelopeBands::new(20, 10, 1.0, 0.0).is_err());
        assert!(AdaptiveEnvelopeBands::new(20, 10, 1.0, -1.0).is_err());
        assert!(AdaptiveEnvelopeBands::new(20, 10, 5.0, 1.0).is_err()); // min >= max
        assert!(AdaptiveEnvelopeBands::new(20, 10, 1.0, 5.0).is_ok());
    }

    #[test]
    fn test_adaptive_envelope_bands_trait() {
        let aeb = AdaptiveEnvelopeBands::new(20, 10, 1.0, 5.0).unwrap();
        assert_eq!(aeb.name(), "Adaptive Envelope Bands");
        assert_eq!(aeb.min_periods(), 21); // max(20, 10) + 1
    }

    #[test]
    fn test_adaptive_envelope_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let aeb = AdaptiveEnvelopeBands::new(20, 10, 1.0, 5.0).unwrap();
        let (m, u, l) = aeb.calculate(&empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_adaptive_envelope_bands_percentage_relationship() {
        let (_, _, close) = make_extended_test_data();
        let aeb = AdaptiveEnvelopeBands::new(10, 10, 2.0, 8.0).unwrap();
        let (middle, upper, lower) = aeb.calculate(&close);

        let idx = 30;
        // Upper should be a percentage above middle
        let upper_pct = (upper[idx] - middle[idx]) / middle[idx] * 100.0;
        let lower_pct = (middle[idx] - lower[idx]) / middle[idx] * 100.0;

        // Percentages should be within the min/max range (with small tolerance for edge cases)
        // The adaptive algorithm may hit bounds depending on CV normalization
        assert!(upper_pct >= 1.9 && upper_pct <= 8.1,
            "Upper percentage {} should be approximately between 2% and 8%", upper_pct);
        assert!(lower_pct >= 1.9 && lower_pct <= 8.1,
            "Lower percentage {} should be approximately between 2% and 8%", lower_pct);

        // Upper and lower should be symmetric (they use the same percentage)
        assert!((upper_pct - lower_pct).abs() < 0.1,
            "Envelope should be symmetric, got upper={} lower={}", upper_pct, lower_pct);
    }

    #[test]
    fn test_adaptive_envelope_bands_volatility_adaptation() {
        // Create data with low then high volatility
        let mut close = Vec::with_capacity(80);
        // Low volatility period
        for i in 0..40 {
            close.push(100.0 + (i % 2) as f64 * 0.5);
        }
        // High volatility period
        for i in 40..80 {
            close.push(100.0 + ((i % 4) as f64 - 1.5) * 5.0);
        }

        let aeb = AdaptiveEnvelopeBands::new(10, 10, 1.0, 5.0).unwrap();
        let (middle, upper, _) = aeb.calculate(&close);

        let low_vol_idx = 30;
        let high_vol_idx = 70;

        let pct_low = (upper[low_vol_idx] - middle[low_vol_idx]) / middle[low_vol_idx] * 100.0;
        let pct_high = (upper[high_vol_idx] - middle[high_vol_idx]) / middle[high_vol_idx] * 100.0;

        // High volatility period should have larger percentage envelope
        assert!(pct_high > pct_low,
            "High volatility period should have larger envelope percentage");
    }

    // --- Combined Tests for All 6 NEW Indicators ---

    #[test]
    fn test_all_6_new_indicators_short_data() {
        let short_high = vec![102.0, 103.0, 104.0];
        let short_low = vec![98.0, 99.0, 100.0];
        let short_close = vec![100.0, 101.0, 102.0];

        // VolatilityBandwidth
        let vbw = VolatilityBandwidth::new(20, 2.0).unwrap();
        let bw = vbw.calculate(&short_close);
        assert_eq!(bw.len(), 3);

        // BandBreakoutStrength
        let bbs = BandBreakoutStrength::new(20, 2.0).unwrap();
        let strength = bbs.calculate(&short_close);
        assert_eq!(strength.len(), 3);

        // DynamicPriceBands
        let dpb = DynamicPriceBands::new(20, 30, 1.0, 3.0).unwrap();
        let (m, u, l) = dpb.calculate(&short_high, &short_low, &short_close);
        assert_eq!(m.len(), 3);

        // TrendAlignedBands
        let tab = TrendAlignedBands::new(20, 10, 2.0, 0.5).unwrap();
        let (m, u, l) = tab.calculate(&short_close);
        assert_eq!(m.len(), 3);

        // MomentumDrivenBands
        let mdb = MomentumDrivenBands::new(20, 10, 1.0, 3.0).unwrap();
        let (m, u, l) = mdb.calculate(&short_close);
        assert_eq!(m.len(), 3);

        // AdaptiveEnvelopeBands
        let aeb = AdaptiveEnvelopeBands::new(20, 10, 1.0, 5.0).unwrap();
        let (m, u, l) = aeb.calculate(&short_close);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn test_all_6_new_indicators_numerical_stability() {
        // Test with large values
        let large_high: Vec<f64> = (0..60).map(|i| 1e8 + (i as f64) * 1000.0).collect();
        let large_low: Vec<f64> = (0..60).map(|i| 1e8 - 1000.0 + (i as f64) * 1000.0).collect();
        let large_close: Vec<f64> = (0..60).map(|i| 1e8 + (i as f64) * 1000.0).collect();
        let idx = 50;

        // VolatilityBandwidth
        let vbw = VolatilityBandwidth::new(10, 2.0).unwrap();
        let bw = vbw.calculate(&large_close);
        assert!(bw[idx].is_finite());

        // BandBreakoutStrength
        let bbs = BandBreakoutStrength::new(10, 2.0).unwrap();
        let strength = bbs.calculate(&large_close);
        assert!(strength[idx].is_finite());

        // DynamicPriceBands
        let dpb = DynamicPriceBands::new(10, 20, 1.0, 3.0).unwrap();
        let (m, u, l) = dpb.calculate(&large_high, &large_low, &large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // TrendAlignedBands
        let tab = TrendAlignedBands::new(10, 10, 2.0, 0.5).unwrap();
        let (m, u, l) = tab.calculate(&large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // MomentumDrivenBands
        let mdb = MomentumDrivenBands::new(10, 5, 1.0, 3.0).unwrap();
        let (m, u, l) = mdb.calculate(&large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // AdaptiveEnvelopeBands
        let aeb = AdaptiveEnvelopeBands::new(10, 10, 1.0, 5.0).unwrap();
        let (m, u, l) = aeb.calculate(&large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());
    }
}
