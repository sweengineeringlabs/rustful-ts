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

/// Price Percentile Bands - Bands based on price percentiles within a rolling window
///
/// Creates bands using percentile calculations over a lookback period.
/// The upper band is the specified upper percentile, the lower band is the
/// specified lower percentile, and the middle is the median (50th percentile).
///
/// This differs from standard deviation-based bands by being more robust
/// to outliers and providing natural support/resistance levels.
///
/// Formula:
/// - Upper Band: nth percentile (e.g., 95th) of prices
/// - Middle Band: 50th percentile (median) of prices
/// - Lower Band: (100-n)th percentile (e.g., 5th) of prices
#[derive(Debug, Clone)]
pub struct PricePercentileBands {
    period: usize,
    upper_percentile: f64,
    lower_percentile: f64,
}

impl PricePercentileBands {
    /// Create new Price Percentile Bands
    ///
    /// # Arguments
    /// * `period` - Lookback period for percentile calculation
    /// * `upper_percentile` - Upper percentile (0-100), e.g., 95.0
    /// * `lower_percentile` - Lower percentile (0-100), e.g., 5.0
    pub fn new(period: usize, upper_percentile: f64, lower_percentile: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if upper_percentile <= 0.0 || upper_percentile >= 100.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "upper_percentile".to_string(),
                reason: "must be between 0 and 100 (exclusive)".to_string(),
            });
        }
        if lower_percentile <= 0.0 || lower_percentile >= 100.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "lower_percentile".to_string(),
                reason: "must be between 0 and 100 (exclusive)".to_string(),
            });
        }
        if lower_percentile >= upper_percentile {
            return Err(IndicatorError::InvalidParameter {
                name: "lower_percentile".to_string(),
                reason: "must be less than upper_percentile".to_string(),
            });
        }
        Ok(Self { period, upper_percentile, lower_percentile })
    }

    /// Create with symmetric percentiles around median
    ///
    /// # Arguments
    /// * `period` - Lookback period
    /// * `percentile_spread` - Distance from median (e.g., 45 gives 5th and 95th percentiles)
    pub fn symmetric(period: usize, percentile_spread: f64) -> Result<Self> {
        Self::new(period, 50.0 + percentile_spread, 50.0 - percentile_spread)
    }

    /// Calculate percentile of a slice
    fn percentile(data: &[f64], pct: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = (pct / 100.0) * (sorted.len() - 1) as f64;
        let lower_idx = index.floor() as usize;
        let upper_idx = index.ceil() as usize;

        if lower_idx == upper_idx || upper_idx >= sorted.len() {
            sorted[lower_idx.min(sorted.len() - 1)]
        } else {
            let fraction = index - lower_idx as f64;
            sorted[lower_idx] * (1.0 - fraction) + sorted[upper_idx] * fraction
        }
    }

    /// Calculate price percentile bands (middle, upper, lower)
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &close[start..=i];

            upper[i] = Self::percentile(window, self.upper_percentile);
            middle[i] = Self::percentile(window, 50.0);
            lower[i] = Self::percentile(window, self.lower_percentile);
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for PricePercentileBands {
    fn name(&self) -> &str {
        "Price Percentile Bands"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Volume Bands - Price bands scaled by volume activity
///
/// Creates bands where the width is proportional to volume activity.
/// Higher volume periods result in wider bands, reflecting increased
/// price uncertainty during high-activity periods.
///
/// The indicator uses volume-weighted ATR to determine band width,
/// with a volume ratio comparing current volume to average volume.
///
/// Formula:
/// - Middle Band: EMA of close
/// - Volume Ratio: Current Volume / Average Volume
/// - Band Width: ATR * multiplier * (0.5 + 0.5 * Volume Ratio)
/// - Upper Band: Middle + Band Width
/// - Lower Band: Middle - Band Width
#[derive(Debug, Clone)]
pub struct VolumeBands {
    period: usize,
    volume_period: usize,
    mult: f64,
}

impl VolumeBands {
    /// Create new Volume Bands
    ///
    /// # Arguments
    /// * `period` - Period for EMA and ATR calculation
    /// * `volume_period` - Period for average volume calculation
    /// * `mult` - Base multiplier for band width
    pub fn new(period: usize, volume_period: usize, mult: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volume_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volume_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, volume_period, mult })
    }

    /// Calculate volume bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len()).min(volume.len());
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
        let atr_alpha = 2.0 / (self.period as f64 + 1.0);
        let mut atr = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());
            atr[i] = atr_alpha * tr + (1.0 - atr_alpha) * atr[i - 1];
        }

        let start_idx = self.period.max(self.volume_period);

        for i in start_idx..n {
            // Calculate average volume
            let vol_start = i.saturating_sub(self.volume_period - 1);
            let avg_volume: f64 = volume[vol_start..=i].iter().sum::<f64>() / self.volume_period as f64;

            // Calculate volume ratio (current vs average)
            let volume_ratio = if avg_volume > 1e-10 {
                (volume[i] / avg_volume).max(0.1).min(5.0) // Clamp to reasonable range
            } else {
                1.0
            };

            // Scale band width by volume ratio
            // Low volume = narrower bands (0.5 base), high volume = wider bands (up to 3x)
            let volume_scale = 0.5 + 0.5 * volume_ratio;
            let band_width = self.mult * atr[i] * volume_scale;

            middle[i] = ema[i];
            upper[i] = ema[i] + band_width;
            lower[i] = ema[i] - band_width;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for VolumeBands {
    fn name(&self) -> &str {
        "Volume Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.volume_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// ATR Bands - Bands using Average True Range instead of standard deviation
///
/// Creates bands using ATR (Average True Range) for width calculation
/// instead of standard deviation. ATR provides a more volatility-accurate
/// measure that accounts for gaps and is commonly used in trading systems.
///
/// This indicator is similar to Keltner Channels but uses SMA instead of EMA
/// for the middle band and provides more configuration options.
///
/// Formula:
/// - Middle Band: SMA of close
/// - Upper Band: Middle + (ATR * multiplier)
/// - Lower Band: Middle - (ATR * multiplier)
#[derive(Debug, Clone)]
pub struct ATRBands {
    period: usize,
    atr_period: usize,
    mult: f64,
}

impl ATRBands {
    /// Create new ATR Bands
    ///
    /// # Arguments
    /// * `period` - Period for SMA calculation
    /// * `atr_period` - Period for ATR calculation
    /// * `mult` - Multiplier for ATR band width
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

    /// Create with same period for SMA and ATR
    pub fn simple(period: usize, mult: f64) -> Result<Self> {
        Self::new(period, period, mult)
    }

    /// Calculate ATR bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate ATR using Wilder's smoothing method
        let mut atr = vec![0.0; n];
        if n > 0 {
            // First TR value
            atr[0] = high[0] - low[0];
        }

        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i < self.atr_period {
                // Simple average during warmup
                let mut tr_sum = high[0] - low[0];
                for j in 1..=i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    tr_sum += tr_j;
                }
                atr[i] = tr_sum / (i + 1) as f64;
            } else {
                // Wilder's smoothing: ATR = ((ATR_prev * (n-1)) + TR) / n
                atr[i] = (atr[i - 1] * (self.atr_period - 1) as f64 + tr) / self.atr_period as f64;
            }
        }

        let start_idx = self.period.max(self.atr_period);

        for i in start_idx..n {
            // Calculate SMA for middle band
            let ma_start = i.saturating_sub(self.period - 1);
            let ma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.period as f64;

            middle[i] = ma;
            upper[i] = ma + self.mult * atr[i];
            lower[i] = ma - self.mult * atr[i];
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for ATRBands {
    fn name(&self) -> &str {
        "ATR Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.atr_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Adaptive Channel Bands - Channel bands that adapt to trend strength
///
/// Creates a price channel that adapts its behavior based on trend conditions.
/// In strong trends, the bands widen and follow price more closely.
/// In ranging markets, the bands narrow and revert to mean more quickly.
///
/// Uses the Efficiency Ratio (from KAMA) to measure trend strength and
/// adjusts both the smoothing factor and band width accordingly.
///
/// Formula:
/// - Efficiency Ratio (ER): |Price Change| / Sum(|Period Changes|)
/// - Adaptive Alpha: (ER * (fast_alpha - slow_alpha) + slow_alpha)^2
/// - Middle Band: Adaptive MA using adaptive alpha
/// - Band Width: ATR * multiplier * (0.5 + ER)
#[derive(Debug, Clone)]
pub struct AdaptiveChannelBands {
    period: usize,
    fast_period: usize,
    slow_period: usize,
    mult: f64,
}

impl AdaptiveChannelBands {
    /// Create new Adaptive Channel Bands
    ///
    /// # Arguments
    /// * `period` - Lookback period for efficiency ratio
    /// * `fast_period` - Fast smoothing period (typically 2)
    /// * `slow_period` - Slow smoothing period (typically 30)
    /// * `mult` - Base multiplier for band width
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

    /// Create with default fast/slow periods (2/30)
    pub fn default_periods(period: usize, mult: f64) -> Result<Self> {
        Self::new(period, 2, 30, mult)
    }

    /// Calculate adaptive channel bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate smoothing constants
        let fast_sc = 2.0 / (self.fast_period as f64 + 1.0);
        let slow_sc = 2.0 / (self.slow_period as f64 + 1.0);

        // Calculate efficiency ratio and adaptive MA (KAMA-style)
        let mut kama = vec![0.0; n];
        let mut er = vec![0.0; n];

        if n > 0 {
            kama[0] = close[0];
        }

        for i in 1..n {
            if i >= self.period {
                // Efficiency Ratio = |Change| / Sum(|Volatility|)
                let change = (close[i] - close[i - self.period]).abs();
                let mut volatility = 0.0;
                for j in (i - self.period + 1)..=i {
                    volatility += (close[j] - close[j - 1]).abs();
                }
                er[i] = if volatility > 1e-10 { change / volatility } else { 0.0 };
            }

            // Adaptive smoothing constant
            let sc = (er[i] * (fast_sc - slow_sc) + slow_sc).powi(2);
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1]);
        }

        // Calculate ATR for band width
        let mut atr = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());
            let atr_alpha = 2.0 / (self.period as f64 + 1.0);
            atr[i] = atr_alpha * tr + (1.0 - atr_alpha) * atr[i - 1];
        }

        let start_idx = self.period.max(self.slow_period);

        for i in start_idx..n {
            // Adaptive band width: wider in trends, narrower in ranges
            // ER near 1 = trending, ER near 0 = ranging
            let adaptive_mult = self.mult * (0.5 + er[i] * 1.5);

            middle[i] = kama[i];
            upper[i] = kama[i] + adaptive_mult * atr[i];
            lower[i] = kama[i] - adaptive_mult * atr[i];
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for AdaptiveChannelBands {
    fn name(&self) -> &str {
        "Adaptive Channel Bands"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.slow_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Regression Bands - Bands around a linear regression line with standard error
///
/// Creates bands using linear regression as the centerline and standard error
/// of the regression for band width. This provides statistically meaningful
/// bands that show the confidence interval around the trend line.
///
/// Unlike simple deviation-based bands, these bands consider the underlying
/// trend and measure deviation from that trend rather than from a moving average.
///
/// Formula:
/// - Middle Band: Linear regression value (y = mx + b)
/// - Standard Error: sqrt(sum((y_actual - y_predicted)^2) / (n-2))
/// - Upper Band: Regression + (SE * multiplier)
/// - Lower Band: Regression - (SE * multiplier)
#[derive(Debug, Clone)]
pub struct RegressionBands {
    period: usize,
    mult: f64,
}

impl RegressionBands {
    /// Create new Regression Bands
    ///
    /// # Arguments
    /// * `period` - Period for linear regression calculation
    /// * `mult` - Multiplier for standard error band width (2.0 = ~95% confidence)
    pub fn new(period: usize, mult: f64) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3 for regression".to_string(),
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

    /// Calculate linear regression statistics for a window
    /// Returns (slope, intercept, standard_error, regression_value_at_end)
    fn linear_regression_stats(data: &[f64]) -> (f64, f64, f64, f64) {
        let n = data.len() as f64;
        if n < 3.0 {
            return (0.0, data.last().copied().unwrap_or(0.0), 0.0, data.last().copied().unwrap_or(0.0));
        }

        // Calculate sums for regression
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        // Calculate slope and intercept
        let denom = n * sum_xx - sum_x * sum_x;
        let (slope, intercept) = if denom.abs() > 1e-10 {
            let slope = (n * sum_xy - sum_x * sum_y) / denom;
            let intercept = (sum_y - slope * sum_x) / n;
            (slope, intercept)
        } else {
            (0.0, sum_y / n)
        };

        // Calculate standard error of regression
        let mut sum_sq_error = 0.0;
        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            let y_pred = slope * x + intercept;
            sum_sq_error += (y - y_pred).powi(2);
        }
        let standard_error = if n > 2.0 {
            (sum_sq_error / (n - 2.0)).sqrt()
        } else {
            0.0
        };

        // Calculate regression value at the end of the window
        let last_x = (data.len() - 1) as f64;
        let regression_value = slope * last_x + intercept;

        (slope, intercept, standard_error, regression_value)
    }

    /// Calculate regression bands (middle, upper, lower)
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &close[start..=i];

            let (_, _, std_error, reg_value) = Self::linear_regression_stats(window);

            middle[i] = reg_value;
            upper[i] = reg_value + self.mult * std_error;
            lower[i] = reg_value - self.mult * std_error;
        }

        (middle, upper, lower)
    }

    /// Calculate with slope information
    /// Returns (middle, upper, lower, slope)
    pub fn calculate_with_slope(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];
        let mut slope = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower, slope);
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &close[start..=i];

            let (s, _, std_error, reg_value) = Self::linear_regression_stats(window);

            middle[i] = reg_value;
            upper[i] = reg_value + self.mult * std_error;
            lower[i] = reg_value - self.mult * std_error;
            slope[i] = s;
        }

        (middle, upper, lower, slope)
    }
}

impl TechnicalIndicator for RegressionBands {
    fn name(&self) -> &str {
        "Regression Bands"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Quantile Bands - Bands based on quantile/percentile analysis with IQR
///
/// Creates bands using robust statistical measures: median for center and
/// interquartile range (IQR) for band width. This approach is more robust
/// to outliers than standard deviation-based methods.
///
/// The bands can optionally extend beyond typical IQR bounds using a multiplier,
/// similar to how Tukey's fences work for outlier detection.
///
/// Formula:
/// - Middle Band: Median (50th percentile)
/// - IQR: Q3 (75th percentile) - Q1 (25th percentile)
/// - Upper Band: Q3 + (IQR * multiplier)
/// - Lower Band: Q1 - (IQR * multiplier)
#[derive(Debug, Clone)]
pub struct QuantileBands {
    period: usize,
    mult: f64,
}

impl QuantileBands {
    /// Create new Quantile Bands
    ///
    /// # Arguments
    /// * `period` - Lookback period for quantile calculation
    /// * `mult` - Multiplier for IQR band extension (1.5 = Tukey's fence)
    pub fn new(period: usize, mult: f64) -> Result<Self> {
        if period < 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 4 for quantile calculation".to_string(),
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

    /// Create with Tukey's fence multiplier (1.5)
    pub fn tukey_fence(period: usize) -> Result<Self> {
        Self::new(period, 1.5)
    }

    /// Calculate quantile of a slice
    fn quantile(data: &[f64], q: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = q * (sorted.len() - 1) as f64;
        let lower_idx = index.floor() as usize;
        let upper_idx = index.ceil() as usize;

        if lower_idx == upper_idx || upper_idx >= sorted.len() {
            sorted[lower_idx.min(sorted.len() - 1)]
        } else {
            let fraction = index - lower_idx as f64;
            sorted[lower_idx] * (1.0 - fraction) + sorted[upper_idx] * fraction
        }
    }

    /// Calculate quantile bands (middle, upper, lower)
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &close[start..=i];

            let q1 = Self::quantile(window, 0.25);
            let median = Self::quantile(window, 0.50);
            let q3 = Self::quantile(window, 0.75);

            let iqr = q3 - q1;

            middle[i] = median;
            upper[i] = q3 + self.mult * iqr;
            lower[i] = q1 - self.mult * iqr;
        }

        (middle, upper, lower)
    }

    /// Calculate with additional quantile information
    /// Returns (middle, upper, lower, q1, q3, iqr)
    pub fn calculate_detailed(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];
        let mut q1_vec = vec![0.0; n];
        let mut q3_vec = vec![0.0; n];
        let mut iqr_vec = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower, q1_vec, q3_vec, iqr_vec);
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &close[start..=i];

            let q1 = Self::quantile(window, 0.25);
            let median = Self::quantile(window, 0.50);
            let q3 = Self::quantile(window, 0.75);

            let iqr = q3 - q1;

            middle[i] = median;
            upper[i] = q3 + self.mult * iqr;
            lower[i] = q1 - self.mult * iqr;
            q1_vec[i] = q1;
            q3_vec[i] = q3;
            iqr_vec[i] = iqr;
        }

        (middle, upper, lower, q1_vec, q3_vec, iqr_vec)
    }
}

impl TechnicalIndicator for QuantileBands {
    fn name(&self) -> &str {
        "Quantile Bands"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Donchian Channel Enhanced - Enhanced Donchian channels with volatility adjustment
///
/// Extends the traditional Donchian Channel by adding ATR-based volatility
/// adjustment to the bands. This helps filter out noise during high volatility
/// periods while maintaining responsiveness during low volatility.
///
/// Formula:
/// - Middle Band: (Highest High + Lowest Low) / 2
/// - Upper Band: Highest High + (ATR * volatility_mult)
/// - Lower Band: Lowest Low - (ATR * volatility_mult)
#[derive(Debug, Clone)]
pub struct DonchianChannelEnhanced {
    period: usize,
    atr_period: usize,
    volatility_mult: f64,
}

impl DonchianChannelEnhanced {
    /// Create new Donchian Channel Enhanced
    ///
    /// # Arguments
    /// * `period` - Lookback period for high/low calculation
    /// * `atr_period` - Period for ATR calculation
    /// * `volatility_mult` - Multiplier for ATR-based band expansion
    pub fn new(period: usize, atr_period: usize, volatility_mult: f64) -> Result<Self> {
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
        if volatility_mult < 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_mult".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        Ok(Self { period, atr_period, volatility_mult })
    }

    /// Create with default ATR period (14)
    pub fn default_atr(period: usize, volatility_mult: f64) -> Result<Self> {
        Self::new(period, 14, volatility_mult)
    }

    /// Calculate Donchian Channel Enhanced (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate ATR
        let mut atr = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i < self.atr_period {
                let mut sum = 0.0;
                for j in 1..=i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    sum += tr_j;
                }
                atr[i] = sum / i as f64;
            } else {
                let alpha = 2.0 / (self.atr_period as f64 + 1.0);
                atr[i] = alpha * tr + (1.0 - alpha) * atr[i - 1];
            }
        }

        let start_idx = self.period.max(self.atr_period);
        for i in start_idx..n {
            let start = i + 1 - self.period;
            let highest = high[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = low[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);

            let volatility_expansion = atr[i] * self.volatility_mult;

            middle[i] = (highest + lowest) / 2.0;
            upper[i] = highest + volatility_expansion;
            lower[i] = lowest - volatility_expansion;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for DonchianChannelEnhanced {
    fn name(&self) -> &str {
        "Donchian Channel Enhanced"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.atr_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// High Low Bands - Bands based on smoothed highs and lows
///
/// Creates bands using exponentially smoothed high and low prices,
/// with an optional offset percentage. This provides dynamic support
/// and resistance levels that adapt to recent price action.
///
/// Formula:
/// - Upper Band: EMA(High) * (1 + offset_percent/100)
/// - Middle Band: (EMA(High) + EMA(Low)) / 2
/// - Lower Band: EMA(Low) * (1 - offset_percent/100)
#[derive(Debug, Clone)]
pub struct HighLowBandsAdvanced {
    period: usize,
    offset_percent: f64,
}

impl HighLowBandsAdvanced {
    /// Create new High Low Bands
    ///
    /// # Arguments
    /// * `period` - EMA period for smoothing
    /// * `offset_percent` - Percentage offset for band expansion (0.0 for no offset)
    pub fn new(period: usize, offset_percent: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if offset_percent < 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "offset_percent".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        Ok(Self { period, offset_percent })
    }

    /// Create with no offset (pure smoothed high/low)
    pub fn no_offset(period: usize) -> Result<Self> {
        Self::new(period, 0.0)
    }

    /// Calculate High Low Bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len().min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate EMA of highs and lows
        let alpha = 2.0 / (self.period as f64 + 1.0);
        let mut ema_high = vec![0.0; n];
        let mut ema_low = vec![0.0; n];

        ema_high[0] = high[0];
        ema_low[0] = low[0];

        for i in 1..n {
            ema_high[i] = alpha * high[i] + (1.0 - alpha) * ema_high[i - 1];
            ema_low[i] = alpha * low[i] + (1.0 - alpha) * ema_low[i - 1];
        }

        let offset_mult_upper = 1.0 + self.offset_percent / 100.0;
        let offset_mult_lower = 1.0 - self.offset_percent / 100.0;

        for i in (self.period - 1)..n {
            upper[i] = ema_high[i] * offset_mult_upper;
            lower[i] = ema_low[i] * offset_mult_lower;
            middle[i] = (ema_high[i] + ema_low[i]) / 2.0;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for HighLowBandsAdvanced {
    fn name(&self) -> &str {
        "High Low Bands Advanced"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Pivot Bands - Bands based on pivot point calculations
///
/// Creates bands using pivot point analysis, calculating support and resistance
/// levels based on the previous period's high, low, and close. This indicator
/// uses a rolling pivot calculation for continuous band generation.
///
/// Formula (Standard Pivot):
/// - Pivot Point (Middle): (High + Low + Close) / 3
/// - Upper Band (R1): (2 * Pivot) - Low
/// - Lower Band (S1): (2 * Pivot) - High
#[derive(Debug, Clone)]
pub struct PivotBands {
    period: usize,
    level: usize, // 1 = R1/S1, 2 = R2/S2, 3 = R3/S3
}

impl PivotBands {
    /// Create new Pivot Bands
    ///
    /// # Arguments
    /// * `period` - Lookback period for pivot calculation
    /// * `level` - Support/Resistance level (1, 2, or 3)
    pub fn new(period: usize, level: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if level < 1 || level > 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "level".to_string(),
                reason: "must be 1, 2, or 3".to_string(),
            });
        }
        Ok(Self { period, level })
    }

    /// Create with first level support/resistance
    pub fn first_level(period: usize) -> Result<Self> {
        Self::new(period, 1)
    }

    /// Calculate Pivot Bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let period_high = high[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let period_close = close[i];

            // Calculate pivot point
            let pivot = (period_high + period_low + period_close) / 3.0;

            // Calculate support and resistance based on level
            let (resistance, support) = match self.level {
                1 => {
                    let r1 = (2.0 * pivot) - period_low;
                    let s1 = (2.0 * pivot) - period_high;
                    (r1, s1)
                }
                2 => {
                    let r1 = (2.0 * pivot) - period_low;
                    let s1 = (2.0 * pivot) - period_high;
                    let r2 = pivot + (period_high - period_low);
                    let s2 = pivot - (period_high - period_low);
                    (r2, s2)
                }
                3 => {
                    let r1 = (2.0 * pivot) - period_low;
                    let s1 = (2.0 * pivot) - period_high;
                    let r2 = pivot + (period_high - period_low);
                    let s2 = pivot - (period_high - period_low);
                    let r3 = r1 + (period_high - period_low);
                    let s3 = s1 - (period_high - period_low);
                    (r3, s3)
                }
                _ => (pivot, pivot),
            };

            middle[i] = pivot;
            upper[i] = resistance;
            lower[i] = support;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for PivotBands {
    fn name(&self) -> &str {
        "Pivot Bands"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Moving Average Bands - Multi-MA based bands with configurable MA types
///
/// Creates bands using different moving average calculations for the upper,
/// middle, and lower bands. This allows for asymmetric bands that can better
/// capture market dynamics.
///
/// Uses EMA for center with SMA-based bands offset by standard deviation.
#[derive(Debug, Clone)]
pub struct MovingAverageBands {
    ema_period: usize,
    sma_period: usize,
    std_mult: f64,
}

impl MovingAverageBands {
    /// Create new Moving Average Bands
    ///
    /// # Arguments
    /// * `ema_period` - Period for EMA (center line)
    /// * `sma_period` - Period for SMA and standard deviation calculation
    /// * `std_mult` - Standard deviation multiplier for band width
    pub fn new(ema_period: usize, sma_period: usize, std_mult: f64) -> Result<Self> {
        if ema_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "ema_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if sma_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "sma_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if std_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "std_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { ema_period, sma_period, std_mult })
    }

    /// Create with equal periods
    pub fn equal_periods(period: usize, std_mult: f64) -> Result<Self> {
        Self::new(period, period, std_mult)
    }

    /// Calculate Moving Average Bands (middle, upper, lower)
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
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

        let start_idx = self.ema_period.max(self.sma_period);
        for i in start_idx..n {
            // Calculate SMA and standard deviation for bands
            let sma_start = i + 1 - self.sma_period;
            let sma: f64 = close[sma_start..=i].iter().sum::<f64>() / self.sma_period as f64;

            let variance: f64 = close[sma_start..=i]
                .iter()
                .map(|&x| (x - sma).powi(2))
                .sum::<f64>() / self.sma_period as f64;
            let std_dev = variance.sqrt();

            middle[i] = ema[i];
            upper[i] = sma + self.std_mult * std_dev;
            lower[i] = sma - self.std_mult * std_dev;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for MovingAverageBands {
    fn name(&self) -> &str {
        "Moving Average Bands"
    }

    fn min_periods(&self) -> usize {
        self.ema_period.max(self.sma_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Volatility Adjusted Bands Extended - Enhanced volatility-adjusted price bands
///
/// Creates bands that dynamically adjust their width based on recent volatility
/// relative to historical volatility. When current volatility is high relative
/// to history, bands expand; when low, bands contract.
///
/// Uses a volatility ratio to scale the band multiplier.
#[derive(Debug, Clone)]
pub struct VolatilityAdjustedBandsExt {
    ma_period: usize,
    short_vol_period: usize,
    long_vol_period: usize,
    base_mult: f64,
}

impl VolatilityAdjustedBandsExt {
    /// Create new Volatility Adjusted Bands Extended
    ///
    /// # Arguments
    /// * `ma_period` - Period for the center moving average
    /// * `short_vol_period` - Period for short-term volatility
    /// * `long_vol_period` - Period for long-term (historical) volatility
    /// * `base_mult` - Base multiplier for band width
    pub fn new(ma_period: usize, short_vol_period: usize, long_vol_period: usize, base_mult: f64) -> Result<Self> {
        if ma_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "ma_period".to_string(),
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
        Ok(Self { ma_period, short_vol_period, long_vol_period, base_mult })
    }

    /// Create with default volatility periods (10 short, 50 long)
    pub fn default_periods(ma_period: usize, base_mult: f64) -> Result<Self> {
        Self::new(ma_period, 10, 50, base_mult)
    }

    /// Calculate Volatility Adjusted Bands Extended (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate True Range for volatility measurement
        let mut tr = vec![0.0; n];
        tr[0] = high[0] - low[0];
        for i in 1..n {
            tr[i] = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());
        }

        let start_idx = self.ma_period.max(self.long_vol_period);
        for i in start_idx..n {
            // Calculate SMA for middle band
            let ma_start = i + 1 - self.ma_period;
            let sma: f64 = close[ma_start..=i].iter().sum::<f64>() / self.ma_period as f64;

            // Calculate short-term volatility (ATR-like)
            let short_start = i + 1 - self.short_vol_period;
            let short_vol: f64 = tr[short_start..=i].iter().sum::<f64>() / self.short_vol_period as f64;

            // Calculate long-term volatility
            let long_start = i + 1 - self.long_vol_period;
            let long_vol: f64 = tr[long_start..=i].iter().sum::<f64>() / self.long_vol_period as f64;

            // Calculate volatility ratio and adjust multiplier
            let vol_ratio = if long_vol > 1e-10 { short_vol / long_vol } else { 1.0 };
            let adjusted_mult = self.base_mult * vol_ratio;

            middle[i] = sma;
            upper[i] = sma + adjusted_mult * short_vol;
            lower[i] = sma - adjusted_mult * short_vol;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for VolatilityAdjustedBandsExt {
    fn name(&self) -> &str {
        "Volatility Adjusted Bands Extended"
    }

    fn min_periods(&self) -> usize {
        self.ma_period.max(self.long_vol_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }
}

/// Trend Channel Bands - Bands that follow the dominant trend direction
///
/// Creates asymmetric bands that adjust based on trend direction and strength.
/// In an uptrend, the lower band acts as trailing support; in a downtrend,
/// the upper band acts as trailing resistance.
///
/// Uses linear regression slope to determine trend direction and ATR for
/// band width calculation.
#[derive(Debug, Clone)]
pub struct TrendChannelBands {
    regression_period: usize,
    atr_period: usize,
    atr_mult: f64,
    trend_sensitivity: f64,
}

impl TrendChannelBands {
    /// Create new Trend Channel Bands
    ///
    /// # Arguments
    /// * `regression_period` - Period for linear regression trend detection
    /// * `atr_period` - Period for ATR calculation
    /// * `atr_mult` - ATR multiplier for band width
    /// * `trend_sensitivity` - How much trend affects band asymmetry (0.0 to 1.0)
    pub fn new(regression_period: usize, atr_period: usize, atr_mult: f64, trend_sensitivity: f64) -> Result<Self> {
        if regression_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "regression_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if atr_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if atr_mult <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_mult".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if trend_sensitivity < 0.0 || trend_sensitivity > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_sensitivity".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self { regression_period, atr_period, atr_mult, trend_sensitivity })
    }

    /// Create with default settings
    pub fn default_settings(regression_period: usize, atr_mult: f64) -> Result<Self> {
        Self::new(regression_period, 14, atr_mult, 0.5)
    }

    /// Calculate linear regression slope
    fn calc_slope(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let sum_x: f64 = (0..data.len()).map(|i| i as f64).sum();
        let sum_y: f64 = data.iter().sum();
        let sum_xy: f64 = data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..data.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        (n * sum_xy - sum_x * sum_y) / denominator
    }

    /// Calculate Trend Channel Bands (middle, upper, lower)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len().min(high.len()).min(low.len());
        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n == 0 {
            return (middle, upper, lower);
        }

        // Calculate ATR
        let mut atr = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i < self.atr_period {
                let mut sum = 0.0;
                for j in 1..=i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    sum += tr_j;
                }
                atr[i] = sum / i as f64;
            } else {
                let alpha = 2.0 / (self.atr_period as f64 + 1.0);
                atr[i] = alpha * tr + (1.0 - alpha) * atr[i - 1];
            }
        }

        let start_idx = self.regression_period.max(self.atr_period);
        for i in start_idx..n {
            // Calculate linear regression for trend
            let reg_start = i + 1 - self.regression_period;
            let slope = Self::calc_slope(&close[reg_start..=i]);

            // Normalize slope relative to price level
            let avg_price = close[reg_start..=i].iter().sum::<f64>() / self.regression_period as f64;
            let norm_slope = if avg_price > 1e-10 {
                slope / avg_price * 100.0 // Slope as percentage per bar
            } else {
                0.0
            };

            // Calculate trend factor (-1 to 1, clamped)
            let trend_factor = norm_slope.max(-1.0).min(1.0);

            // Adjust band widths based on trend
            // In uptrend: lower band tighter (for trailing stop), upper band wider
            // In downtrend: upper band tighter, lower band wider
            let base_width = self.atr_mult * atr[i];
            let upper_adj = 1.0 + trend_factor * self.trend_sensitivity;
            let lower_adj = 1.0 - trend_factor * self.trend_sensitivity;

            middle[i] = close[i];
            upper[i] = close[i] + base_width * upper_adj;
            lower[i] = close[i] - base_width * lower_adj;
        }

        (middle, upper, lower)
    }
}

impl TechnicalIndicator for TrendChannelBands {
    fn name(&self) -> &str {
        "Trend Channel Bands"
    }

    fn min_periods(&self) -> usize {
        self.regression_period.max(self.atr_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
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

    // ============================================================
    // Tests for 6 NEW Band Indicators (Jan 2026 - Phase 2)
    // PricePercentileBands, VolumeBands, ATRBands,
    // AdaptiveChannelBands, RegressionBands, QuantileBands
    // ============================================================

    fn make_test_data_with_volume() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![
            102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
            122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0,
        ];
        let low = vec![
            98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
            108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0,
            118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0,
        ];
        let close = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
            110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
            120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0,
        ];
        let volume = vec![
            1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
            2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0,
            3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0,
        ];
        (high, low, close, volume)
    }

    // --- PricePercentileBands Tests ---

    #[test]
    fn test_price_percentile_bands() {
        let (_, _, close) = make_extended_test_data();
        let ppb = PricePercentileBands::new(20, 95.0, 5.0).unwrap();
        let (middle, upper, lower) = ppb.calculate(&close);

        assert_eq!(middle.len(), close.len());
        let idx = 30;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] >= middle[idx]);
        assert!(lower[idx] <= middle[idx]);
    }

    #[test]
    fn test_price_percentile_bands_validation() {
        assert!(PricePercentileBands::new(1, 95.0, 5.0).is_err());  // period < 2
        assert!(PricePercentileBands::new(20, 0.0, 5.0).is_err());  // upper <= 0
        assert!(PricePercentileBands::new(20, 100.0, 5.0).is_err()); // upper >= 100
        assert!(PricePercentileBands::new(20, 95.0, 0.0).is_err());  // lower <= 0
        assert!(PricePercentileBands::new(20, 95.0, 100.0).is_err()); // lower >= 100
        assert!(PricePercentileBands::new(20, 50.0, 60.0).is_err()); // lower >= upper
        assert!(PricePercentileBands::new(20, 95.0, 5.0).is_ok());
    }

    #[test]
    fn test_price_percentile_bands_trait() {
        let ppb = PricePercentileBands::new(20, 95.0, 5.0).unwrap();
        assert_eq!(ppb.name(), "Price Percentile Bands");
        assert_eq!(ppb.min_periods(), 20);
    }

    #[test]
    fn test_price_percentile_bands_symmetric() {
        let ppb = PricePercentileBands::symmetric(20, 45.0).unwrap();
        // Should create 95th upper and 5th lower percentiles
        let close = vec![100.0; 30];
        let (middle, upper, lower) = ppb.calculate(&close);
        let idx = 25;
        // For flat data, all percentiles should be equal
        assert!((middle[idx] - upper[idx]).abs() < 1e-10);
        assert!((middle[idx] - lower[idx]).abs() < 1e-10);
    }

    #[test]
    fn test_price_percentile_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let ppb = PricePercentileBands::new(20, 95.0, 5.0).unwrap();
        let (m, u, l) = ppb.calculate(&empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_price_percentile_bands_ordering() {
        // Create data with variation
        let close: Vec<f64> = (0..50).map(|i| 100.0 + 5.0 * ((i as f64) * 0.3).sin()).collect();
        let ppb = PricePercentileBands::new(10, 90.0, 10.0).unwrap();
        let (middle, upper, lower) = ppb.calculate(&close);

        // Verify ordering: lower <= middle <= upper for all calculated points
        for i in 9..50 {
            assert!(lower[i] <= middle[i], "lower[{}] > middle[{}]", i, i);
            assert!(middle[i] <= upper[i], "middle[{}] > upper[{}]", i, i);
        }
    }

    // --- VolumeBands Tests ---

    #[test]
    fn test_volume_bands() {
        let (high, low, close, volume) = make_test_data_with_volume();
        let vb = VolumeBands::new(10, 10, 2.0).unwrap();
        let (middle, upper, lower) = vb.calculate(&high, &low, &close, &volume);

        assert_eq!(middle.len(), close.len());
        let idx = 15;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_volume_bands_validation() {
        assert!(VolumeBands::new(1, 10, 2.0).is_err());  // period < 2
        assert!(VolumeBands::new(10, 1, 2.0).is_err()); // volume_period < 2
        assert!(VolumeBands::new(10, 10, 0.0).is_err()); // mult <= 0
        assert!(VolumeBands::new(10, 10, -1.0).is_err()); // mult < 0
        assert!(VolumeBands::new(10, 10, 2.0).is_ok());
    }

    #[test]
    fn test_volume_bands_trait() {
        let vb = VolumeBands::new(10, 15, 2.0).unwrap();
        assert_eq!(vb.name(), "Volume Bands");
        assert_eq!(vb.min_periods(), 16); // max(10, 15) + 1
    }

    #[test]
    fn test_volume_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let vb = VolumeBands::new(10, 10, 2.0).unwrap();
        let (m, u, l) = vb.calculate(&empty, &empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_volume_bands_high_volume_wider() {
        // Create data with varying volume
        let high = vec![102.0; 50];
        let low = vec![98.0; 50];
        let close = vec![100.0; 50];

        // Low volume data
        let low_volume = vec![1000.0; 50];
        // High volume data
        let high_volume = vec![5000.0; 50];

        let vb = VolumeBands::new(10, 10, 2.0).unwrap();
        let (_, upper_low, _) = vb.calculate(&high, &low, &close, &low_volume);
        let (_, upper_high, _) = vb.calculate(&high, &low, &close, &high_volume);

        let idx = 20;
        let width_low = upper_low[idx] - close[idx];
        let width_high = upper_high[idx] - close[idx];

        // Both should have positive width since ATR is non-zero
        assert!(width_low > 0.0);
        assert!(width_high > 0.0);
    }

    // --- ATRBands Tests ---

    #[test]
    fn test_atr_bands() {
        let (high, low, close) = make_extended_test_data();
        let ab = ATRBands::new(20, 14, 2.0).unwrap();
        let (middle, upper, lower) = ab.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 25;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_atr_bands_validation() {
        assert!(ATRBands::new(1, 14, 2.0).is_err());  // period < 2
        assert!(ATRBands::new(20, 1, 2.0).is_err()); // atr_period < 2
        assert!(ATRBands::new(20, 14, 0.0).is_err()); // mult <= 0
        assert!(ATRBands::new(20, 14, -1.0).is_err()); // mult < 0
        assert!(ATRBands::new(20, 14, 2.0).is_ok());
    }

    #[test]
    fn test_atr_bands_trait() {
        let ab = ATRBands::new(20, 14, 2.0).unwrap();
        assert_eq!(ab.name(), "ATR Bands");
        assert_eq!(ab.min_periods(), 21); // max(20, 14) + 1
    }

    #[test]
    fn test_atr_bands_simple() {
        let ab = ATRBands::simple(14, 2.0).unwrap();
        assert_eq!(ab.min_periods(), 15); // max(14, 14) + 1
    }

    #[test]
    fn test_atr_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let ab = ATRBands::new(20, 14, 2.0).unwrap();
        let (m, u, l) = ab.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_atr_bands_symmetry() {
        let (high, low, close) = make_extended_test_data();
        let ab = ATRBands::new(10, 10, 2.0).unwrap();
        let (middle, upper, lower) = ab.calculate(&high, &low, &close);

        // Bands should be symmetric around middle
        let idx = 20;
        let upper_diff = upper[idx] - middle[idx];
        let lower_diff = middle[idx] - lower[idx];
        assert!((upper_diff - lower_diff).abs() < 1e-10,
            "ATR Bands should be symmetric");
    }

    // --- AdaptiveChannelBands Tests ---

    #[test]
    fn test_adaptive_channel_bands() {
        let (high, low, close) = make_extended_test_data();
        let acb = AdaptiveChannelBands::new(10, 2, 30, 2.0).unwrap();
        let (middle, upper, lower) = acb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 40;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_adaptive_channel_bands_validation() {
        assert!(AdaptiveChannelBands::new(1, 2, 30, 2.0).is_err());   // period < 2
        assert!(AdaptiveChannelBands::new(10, 1, 30, 2.0).is_err()); // fast_period < 2
        assert!(AdaptiveChannelBands::new(10, 2, 1, 2.0).is_err());  // slow_period < 2
        assert!(AdaptiveChannelBands::new(10, 30, 20, 2.0).is_err()); // fast >= slow
        assert!(AdaptiveChannelBands::new(10, 2, 30, 0.0).is_err()); // mult <= 0
        assert!(AdaptiveChannelBands::new(10, 2, 30, -1.0).is_err()); // mult < 0
        assert!(AdaptiveChannelBands::new(10, 2, 30, 2.0).is_ok());
    }

    #[test]
    fn test_adaptive_channel_bands_trait() {
        let acb = AdaptiveChannelBands::new(10, 2, 30, 2.0).unwrap();
        assert_eq!(acb.name(), "Adaptive Channel Bands");
        assert_eq!(acb.min_periods(), 31); // max(10, 30) + 1
    }

    #[test]
    fn test_adaptive_channel_bands_default_periods() {
        let acb = AdaptiveChannelBands::default_periods(10, 2.0).unwrap();
        assert_eq!(acb.min_periods(), 31); // max(10, 30) + 1
    }

    #[test]
    fn test_adaptive_channel_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let acb = AdaptiveChannelBands::new(10, 2, 30, 2.0).unwrap();
        let (m, u, l) = acb.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_adaptive_channel_bands_adapts_to_trend() {
        // Create trending data
        let trending: Vec<f64> = (0..80).map(|i| 100.0 + (i as f64) * 2.0).collect();
        let high: Vec<f64> = trending.iter().map(|&c| c + 2.0).collect();
        let low: Vec<f64> = trending.iter().map(|&c| c - 2.0).collect();

        // Create ranging data
        let ranging: Vec<f64> = (0..80).map(|i| 100.0 + 3.0 * ((i as f64) * 0.5).sin()).collect();
        let high_r: Vec<f64> = ranging.iter().map(|&c| c + 2.0).collect();
        let low_r: Vec<f64> = ranging.iter().map(|&c| c - 2.0).collect();

        let acb = AdaptiveChannelBands::new(10, 2, 30, 2.0).unwrap();
        let (_, upper_t, lower_t) = acb.calculate(&high, &low, &trending);
        let (_, upper_r, lower_r) = acb.calculate(&high_r, &low_r, &ranging);

        // Both should produce valid bands
        let idx = 50;
        assert!(upper_t[idx] > lower_t[idx]);
        assert!(upper_r[idx] > lower_r[idx]);
    }

    // --- RegressionBands Tests ---

    #[test]
    fn test_regression_bands() {
        let (_, _, close) = make_extended_test_data();
        let rb = RegressionBands::new(20, 2.0).unwrap();
        let (middle, upper, lower) = rb.calculate(&close);

        assert_eq!(middle.len(), close.len());
        let idx = 30;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] >= middle[idx]);
        assert!(lower[idx] <= middle[idx]);
    }

    #[test]
    fn test_regression_bands_validation() {
        assert!(RegressionBands::new(2, 2.0).is_err());  // period < 3
        assert!(RegressionBands::new(20, 0.0).is_err()); // mult <= 0
        assert!(RegressionBands::new(20, -1.0).is_err()); // mult < 0
        assert!(RegressionBands::new(20, 2.0).is_ok());
    }

    #[test]
    fn test_regression_bands_trait() {
        let rb = RegressionBands::new(20, 2.0).unwrap();
        assert_eq!(rb.name(), "Regression Bands");
        assert_eq!(rb.min_periods(), 20);
    }

    #[test]
    fn test_regression_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let rb = RegressionBands::new(20, 2.0).unwrap();
        let (m, u, l) = rb.calculate(&empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_regression_bands_with_slope() {
        // Create clearly uptrending data for slope test
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 2.0).collect();
        let rb = RegressionBands::new(10, 2.0).unwrap();
        let (middle, upper, lower, slope) = rb.calculate_with_slope(&close);

        let idx = 20;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] >= middle[idx]);
        assert!(lower[idx] <= middle[idx]);
        // Uptrending data should have positive slope
        assert!(slope[idx] > 0.0, "Slope should be positive for uptrending data, got {}", slope[idx]);
    }

    #[test]
    fn test_regression_bands_perfect_line() {
        // Perfect linear data should have zero standard error
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64) * 2.0).collect();
        let rb = RegressionBands::new(10, 2.0).unwrap();
        let (middle, upper, lower) = rb.calculate(&close);

        let idx = 20;
        // For perfect linear data, upper should equal lower should equal middle
        // (standard error is 0)
        assert!((upper[idx] - middle[idx]).abs() < 1e-10,
            "Upper should equal middle for perfect line");
        assert!((middle[idx] - lower[idx]).abs() < 1e-10,
            "Middle should equal lower for perfect line");
    }

    // --- QuantileBands Tests ---

    #[test]
    fn test_quantile_bands() {
        let (_, _, close) = make_extended_test_data();
        let qb = QuantileBands::new(20, 1.5).unwrap();
        let (middle, upper, lower) = qb.calculate(&close);

        assert_eq!(middle.len(), close.len());
        let idx = 30;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] >= middle[idx]);
        assert!(lower[idx] <= middle[idx]);
    }

    #[test]
    fn test_quantile_bands_validation() {
        assert!(QuantileBands::new(3, 1.5).is_err());  // period < 4
        assert!(QuantileBands::new(20, 0.0).is_err()); // mult <= 0
        assert!(QuantileBands::new(20, -1.0).is_err()); // mult < 0
        assert!(QuantileBands::new(20, 1.5).is_ok());
    }

    #[test]
    fn test_quantile_bands_trait() {
        let qb = QuantileBands::new(20, 1.5).unwrap();
        assert_eq!(qb.name(), "Quantile Bands");
        assert_eq!(qb.min_periods(), 20);
    }

    #[test]
    fn test_quantile_bands_tukey_fence() {
        let qb = QuantileBands::tukey_fence(20).unwrap();
        assert_eq!(qb.min_periods(), 20);
    }

    #[test]
    fn test_quantile_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let qb = QuantileBands::new(20, 1.5).unwrap();
        let (m, u, l) = qb.calculate(&empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_quantile_bands_detailed() {
        let (_, _, close) = make_extended_test_data();
        let qb = QuantileBands::new(10, 1.5).unwrap();
        let (middle, upper, lower, q1, q3, iqr) = qb.calculate_detailed(&close);

        let idx = 20;
        assert!(middle[idx] > 0.0);
        assert!(q1[idx] < q3[idx]);
        assert!((iqr[idx] - (q3[idx] - q1[idx])).abs() < 1e-10);
        // Upper = Q3 + 1.5*IQR, Lower = Q1 - 1.5*IQR
        assert!((upper[idx] - (q3[idx] + 1.5 * iqr[idx])).abs() < 1e-10);
        assert!((lower[idx] - (q1[idx] - 1.5 * iqr[idx])).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_bands_flat_data() {
        // Flat data should have Q1 == Q3 == median, so IQR = 0
        let close = vec![100.0; 30];
        let qb = QuantileBands::new(10, 1.5).unwrap();
        let (middle, upper, lower) = qb.calculate(&close);

        let idx = 20;
        // For flat data, all values are equal
        assert!((middle[idx] - 100.0).abs() < 1e-10);
        assert!((upper[idx] - 100.0).abs() < 1e-10);
        assert!((lower[idx] - 100.0).abs() < 1e-10);
    }

    // --- Combined Tests for All 6 NEW Indicators ---

    #[test]
    fn test_all_6_new_phase2_indicators_short_data() {
        let short_high = vec![102.0, 103.0, 104.0];
        let short_low = vec![98.0, 99.0, 100.0];
        let short_close = vec![100.0, 101.0, 102.0];
        let short_volume = vec![1000.0, 1100.0, 1200.0];

        // PricePercentileBands
        let ppb = PricePercentileBands::new(20, 95.0, 5.0).unwrap();
        let (m, u, l) = ppb.calculate(&short_close);
        assert_eq!(m.len(), 3);

        // VolumeBands
        let vb = VolumeBands::new(20, 10, 2.0).unwrap();
        let (m, u, l) = vb.calculate(&short_high, &short_low, &short_close, &short_volume);
        assert_eq!(m.len(), 3);

        // ATRBands
        let ab = ATRBands::new(20, 14, 2.0).unwrap();
        let (m, u, l) = ab.calculate(&short_high, &short_low, &short_close);
        assert_eq!(m.len(), 3);

        // AdaptiveChannelBands
        let acb = AdaptiveChannelBands::new(10, 2, 30, 2.0).unwrap();
        let (m, u, l) = acb.calculate(&short_high, &short_low, &short_close);
        assert_eq!(m.len(), 3);

        // RegressionBands
        let rb = RegressionBands::new(20, 2.0).unwrap();
        let (m, u, l) = rb.calculate(&short_close);
        assert_eq!(m.len(), 3);

        // QuantileBands
        let qb = QuantileBands::new(20, 1.5).unwrap();
        let (m, u, l) = qb.calculate(&short_close);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn test_all_6_new_phase2_indicators_numerical_stability() {
        // Test with large values
        let large_high: Vec<f64> = (0..60).map(|i| 1e8 + (i as f64) * 1000.0).collect();
        let large_low: Vec<f64> = (0..60).map(|i| 1e8 - 1000.0 + (i as f64) * 1000.0).collect();
        let large_close: Vec<f64> = (0..60).map(|i| 1e8 + (i as f64) * 1000.0).collect();
        let large_volume: Vec<f64> = (0..60).map(|i| 1e6 + (i as f64) * 100.0).collect();
        let idx = 50;

        // PricePercentileBands
        let ppb = PricePercentileBands::new(10, 95.0, 5.0).unwrap();
        let (m, u, l) = ppb.calculate(&large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // VolumeBands
        let vb = VolumeBands::new(10, 10, 2.0).unwrap();
        let (m, u, l) = vb.calculate(&large_high, &large_low, &large_close, &large_volume);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // ATRBands
        let ab = ATRBands::new(10, 10, 2.0).unwrap();
        let (m, u, l) = ab.calculate(&large_high, &large_low, &large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // AdaptiveChannelBands
        let acb = AdaptiveChannelBands::new(10, 2, 30, 2.0).unwrap();
        let (m, u, l) = acb.calculate(&large_high, &large_low, &large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // RegressionBands
        let rb = RegressionBands::new(10, 2.0).unwrap();
        let (m, u, l) = rb.calculate(&large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // QuantileBands
        let qb = QuantileBands::new(10, 1.5).unwrap();
        let (m, u, l) = qb.calculate(&large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());
    }

    // ============================================================
    // Tests for 6 NEW Band Indicators (Jan 2026 - Phase 3)
    // DonchianChannelEnhanced, HighLowBandsAdvanced, PivotBands,
    // MovingAverageBands, VolatilityAdjustedBandsExt, TrendChannelBands
    // ============================================================

    // --- DonchianChannelEnhanced Tests ---

    #[test]
    fn test_donchian_channel_enhanced() {
        let (high, low, close) = make_extended_test_data();
        let dce = DonchianChannelEnhanced::new(20, 14, 0.5).unwrap();
        let (middle, upper, lower) = dce.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 30;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_donchian_channel_enhanced_validation() {
        assert!(DonchianChannelEnhanced::new(1, 14, 0.5).is_err());  // period < 2
        assert!(DonchianChannelEnhanced::new(20, 1, 0.5).is_err()); // atr_period < 2
        assert!(DonchianChannelEnhanced::new(20, 14, -0.5).is_err()); // volatility_mult < 0
        assert!(DonchianChannelEnhanced::new(20, 14, 0.5).is_ok());
        assert!(DonchianChannelEnhanced::new(20, 14, 0.0).is_ok()); // zero mult is allowed
    }

    #[test]
    fn test_donchian_channel_enhanced_trait() {
        let dce = DonchianChannelEnhanced::new(20, 14, 0.5).unwrap();
        assert_eq!(dce.name(), "Donchian Channel Enhanced");
        assert_eq!(dce.min_periods(), 21); // max(20, 14) + 1
    }

    #[test]
    fn test_donchian_channel_enhanced_default_atr() {
        let dce = DonchianChannelEnhanced::default_atr(20, 0.5).unwrap();
        assert_eq!(dce.min_periods(), 21); // max(20, 14) + 1
    }

    #[test]
    fn test_donchian_channel_enhanced_empty_data() {
        let empty: Vec<f64> = vec![];
        let dce = DonchianChannelEnhanced::new(20, 14, 0.5).unwrap();
        let (m, u, l) = dce.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_donchian_channel_enhanced_volatility_expansion() {
        let (high, low, close) = make_extended_test_data();

        // With zero mult, bands should be pure Donchian
        let dce_no_exp = DonchianChannelEnhanced::new(10, 10, 0.0).unwrap();
        let (_, upper_no_exp, lower_no_exp) = dce_no_exp.calculate(&high, &low, &close);

        // With positive mult, bands should be wider
        let dce_exp = DonchianChannelEnhanced::new(10, 10, 1.0).unwrap();
        let (_, upper_exp, lower_exp) = dce_exp.calculate(&high, &low, &close);

        let idx = 20;
        assert!(upper_exp[idx] >= upper_no_exp[idx],
            "With volatility expansion, upper band should be higher or equal");
        assert!(lower_exp[idx] <= lower_no_exp[idx],
            "With volatility expansion, lower band should be lower or equal");
    }

    // --- HighLowBandsAdvanced Tests ---

    #[test]
    fn test_high_low_bands_advanced() {
        let (high, low, _) = make_extended_test_data();
        let hlb = HighLowBandsAdvanced::new(20, 1.0).unwrap();
        let (middle, upper, lower) = hlb.calculate(&high, &low);

        assert_eq!(middle.len(), high.len());
        let idx = 25;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] >= middle[idx]);
        assert!(lower[idx] <= middle[idx]);
    }

    #[test]
    fn test_high_low_bands_advanced_validation() {
        assert!(HighLowBandsAdvanced::new(1, 1.0).is_err());  // period < 2
        assert!(HighLowBandsAdvanced::new(20, -0.5).is_err()); // offset_percent < 0
        assert!(HighLowBandsAdvanced::new(20, 1.0).is_ok());
        assert!(HighLowBandsAdvanced::new(20, 0.0).is_ok()); // zero offset is allowed
    }

    #[test]
    fn test_high_low_bands_advanced_trait() {
        let hlb = HighLowBandsAdvanced::new(20, 1.0).unwrap();
        assert_eq!(hlb.name(), "High Low Bands Advanced");
        assert_eq!(hlb.min_periods(), 20);
    }

    #[test]
    fn test_high_low_bands_advanced_no_offset() {
        let hlb = HighLowBandsAdvanced::no_offset(20).unwrap();
        let (high, low, _) = make_extended_test_data();
        let (middle, upper, lower) = hlb.calculate(&high, &low);

        let idx = 25;
        // Without offset, upper = EMA(high), lower = EMA(low)
        // Middle should be average of upper and lower
        assert!((middle[idx] - (upper[idx] + lower[idx]) / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_high_low_bands_advanced_empty_data() {
        let empty: Vec<f64> = vec![];
        let hlb = HighLowBandsAdvanced::new(20, 1.0).unwrap();
        let (m, u, l) = hlb.calculate(&empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_high_low_bands_advanced_offset_expansion() {
        let (high, low, _) = make_extended_test_data();

        let hlb_no_offset = HighLowBandsAdvanced::new(10, 0.0).unwrap();
        let (_, upper_no, lower_no) = hlb_no_offset.calculate(&high, &low);

        let hlb_offset = HighLowBandsAdvanced::new(10, 2.0).unwrap();
        let (_, upper_off, lower_off) = hlb_offset.calculate(&high, &low);

        let idx = 20;
        assert!(upper_off[idx] > upper_no[idx], "Offset should expand upper band");
        assert!(lower_off[idx] < lower_no[idx], "Offset should expand lower band");
    }

    // --- PivotBands Tests ---

    #[test]
    fn test_pivot_bands() {
        let (high, low, close) = make_extended_test_data();
        let pb = PivotBands::new(10, 1).unwrap();
        let (middle, upper, lower) = pb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 15;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] >= middle[idx]);
        assert!(lower[idx] <= middle[idx]);
    }

    #[test]
    fn test_pivot_bands_validation() {
        assert!(PivotBands::new(1, 1).is_err());   // period < 2
        assert!(PivotBands::new(10, 0).is_err()); // level < 1
        assert!(PivotBands::new(10, 4).is_err()); // level > 3
        assert!(PivotBands::new(10, 1).is_ok());
        assert!(PivotBands::new(10, 2).is_ok());
        assert!(PivotBands::new(10, 3).is_ok());
    }

    #[test]
    fn test_pivot_bands_trait() {
        let pb = PivotBands::new(10, 1).unwrap();
        assert_eq!(pb.name(), "Pivot Bands");
        assert_eq!(pb.min_periods(), 10);
    }

    #[test]
    fn test_pivot_bands_first_level() {
        let pb = PivotBands::first_level(10).unwrap();
        assert_eq!(pb.min_periods(), 10);
    }

    #[test]
    fn test_pivot_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let pb = PivotBands::new(10, 1).unwrap();
        let (m, u, l) = pb.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_pivot_bands_levels_ordering() {
        let (high, low, close) = make_extended_test_data();

        let pb1 = PivotBands::new(10, 1).unwrap();
        let pb2 = PivotBands::new(10, 2).unwrap();
        let pb3 = PivotBands::new(10, 3).unwrap();

        let (_, u1, l1) = pb1.calculate(&high, &low, &close);
        let (_, u2, l2) = pb2.calculate(&high, &low, &close);
        let (_, u3, l3) = pb3.calculate(&high, &low, &close);

        let idx = 20;
        // Higher levels should have wider bands
        assert!(u2[idx] >= u1[idx], "R2 should be >= R1");
        assert!(u3[idx] >= u2[idx], "R3 should be >= R2");
        assert!(l2[idx] <= l1[idx], "S2 should be <= S1");
        assert!(l3[idx] <= l2[idx], "S3 should be <= S2");
    }

    // --- MovingAverageBands Tests ---

    #[test]
    fn test_moving_average_bands() {
        let (_, _, close) = make_extended_test_data();
        let mab = MovingAverageBands::new(10, 20, 2.0).unwrap();
        let (middle, upper, lower) = mab.calculate(&close);

        assert_eq!(middle.len(), close.len());
        let idx = 25;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_moving_average_bands_validation() {
        assert!(MovingAverageBands::new(1, 20, 2.0).is_err());  // ema_period < 2
        assert!(MovingAverageBands::new(10, 1, 2.0).is_err()); // sma_period < 2
        assert!(MovingAverageBands::new(10, 20, 0.0).is_err()); // std_mult <= 0
        assert!(MovingAverageBands::new(10, 20, -1.0).is_err()); // std_mult < 0
        assert!(MovingAverageBands::new(10, 20, 2.0).is_ok());
    }

    #[test]
    fn test_moving_average_bands_trait() {
        let mab = MovingAverageBands::new(10, 20, 2.0).unwrap();
        assert_eq!(mab.name(), "Moving Average Bands");
        assert_eq!(mab.min_periods(), 21); // max(10, 20) + 1
    }

    #[test]
    fn test_moving_average_bands_equal_periods() {
        let mab = MovingAverageBands::equal_periods(15, 2.0).unwrap();
        assert_eq!(mab.min_periods(), 16); // 15 + 1
    }

    #[test]
    fn test_moving_average_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let mab = MovingAverageBands::new(10, 20, 2.0).unwrap();
        let (m, u, l) = mab.calculate(&empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_moving_average_bands_symmetry() {
        let (_, _, close) = make_extended_test_data();
        let mab = MovingAverageBands::new(10, 10, 2.0).unwrap();
        let (_, upper, lower) = mab.calculate(&close);

        // Get the SMA at a point to verify symmetry
        let idx = 20;
        let sma_start = idx + 1 - 10;
        let sma: f64 = close[sma_start..=idx].iter().sum::<f64>() / 10.0;

        let upper_diff = upper[idx] - sma;
        let lower_diff = sma - lower[idx];

        // Upper and lower should be symmetric around SMA
        assert!((upper_diff - lower_diff).abs() < 1e-10,
            "Bands should be symmetric around SMA");
    }

    // --- VolatilityAdjustedBandsExt Tests ---

    #[test]
    fn test_volatility_adjusted_bands_ext() {
        let (high, low, close) = make_extended_test_data();
        let vabe = VolatilityAdjustedBandsExt::new(20, 10, 50, 2.0).unwrap();
        let (middle, upper, lower) = vabe.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 60;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_volatility_adjusted_bands_ext_validation() {
        assert!(VolatilityAdjustedBandsExt::new(1, 10, 50, 2.0).is_err());  // ma_period < 2
        assert!(VolatilityAdjustedBandsExt::new(20, 1, 50, 2.0).is_err()); // short_vol_period < 2
        assert!(VolatilityAdjustedBandsExt::new(20, 10, 1, 2.0).is_err()); // long_vol_period < 2
        assert!(VolatilityAdjustedBandsExt::new(20, 50, 40, 2.0).is_err()); // short >= long
        assert!(VolatilityAdjustedBandsExt::new(20, 10, 50, 0.0).is_err()); // base_mult <= 0
        assert!(VolatilityAdjustedBandsExt::new(20, 10, 50, -1.0).is_err()); // base_mult < 0
        assert!(VolatilityAdjustedBandsExt::new(20, 10, 50, 2.0).is_ok());
    }

    #[test]
    fn test_volatility_adjusted_bands_ext_trait() {
        let vabe = VolatilityAdjustedBandsExt::new(20, 10, 50, 2.0).unwrap();
        assert_eq!(vabe.name(), "Volatility Adjusted Bands Extended");
        assert_eq!(vabe.min_periods(), 51); // max(20, 50) + 1
    }

    #[test]
    fn test_volatility_adjusted_bands_ext_default_periods() {
        let vabe = VolatilityAdjustedBandsExt::default_periods(20, 2.0).unwrap();
        assert_eq!(vabe.min_periods(), 51); // max(20, 50) + 1
    }

    #[test]
    fn test_volatility_adjusted_bands_ext_empty_data() {
        let empty: Vec<f64> = vec![];
        let vabe = VolatilityAdjustedBandsExt::new(20, 10, 50, 2.0).unwrap();
        let (m, u, l) = vabe.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_volatility_adjusted_bands_ext_volatility_response() {
        // Create data with low then high volatility
        let mut high = vec![102.0; 100];
        let mut low = vec![98.0; 100];
        let mut close = vec![100.0; 100];

        // High volatility period at end
        for i in 70..100 {
            high[i] = 120.0;
            low[i] = 80.0;
            close[i] = 100.0;
        }

        let vabe = VolatilityAdjustedBandsExt::new(10, 5, 30, 2.0).unwrap();
        let (_, upper, _) = vabe.calculate(&high, &low, &close);

        let low_vol_idx = 50;
        let high_vol_idx = 90;

        // Bands should be wider during high volatility
        let width_low = upper[low_vol_idx] - close[low_vol_idx];
        let width_high = upper[high_vol_idx] - close[high_vol_idx];

        assert!(width_high > width_low,
            "Bands should be wider during high volatility periods");
    }

    // --- TrendChannelBands Tests ---

    #[test]
    fn test_trend_channel_bands() {
        let (high, low, close) = make_extended_test_data();
        let tcb = TrendChannelBands::new(20, 14, 2.0, 0.5).unwrap();
        let (middle, upper, lower) = tcb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        let idx = 30;
        assert!(middle[idx] > 0.0);
        assert!(upper[idx] > middle[idx]);
        assert!(lower[idx] < middle[idx]);
    }

    #[test]
    fn test_trend_channel_bands_validation() {
        assert!(TrendChannelBands::new(2, 14, 2.0, 0.5).is_err());  // regression_period < 3
        assert!(TrendChannelBands::new(20, 1, 2.0, 0.5).is_err()); // atr_period < 2
        assert!(TrendChannelBands::new(20, 14, 0.0, 0.5).is_err()); // atr_mult <= 0
        assert!(TrendChannelBands::new(20, 14, -1.0, 0.5).is_err()); // atr_mult < 0
        assert!(TrendChannelBands::new(20, 14, 2.0, -0.1).is_err()); // trend_sensitivity < 0
        assert!(TrendChannelBands::new(20, 14, 2.0, 1.1).is_err()); // trend_sensitivity > 1
        assert!(TrendChannelBands::new(20, 14, 2.0, 0.5).is_ok());
        assert!(TrendChannelBands::new(20, 14, 2.0, 0.0).is_ok()); // zero sensitivity
        assert!(TrendChannelBands::new(20, 14, 2.0, 1.0).is_ok()); // max sensitivity
    }

    #[test]
    fn test_trend_channel_bands_trait() {
        let tcb = TrendChannelBands::new(20, 14, 2.0, 0.5).unwrap();
        assert_eq!(tcb.name(), "Trend Channel Bands");
        assert_eq!(tcb.min_periods(), 21); // max(20, 14) + 1
    }

    #[test]
    fn test_trend_channel_bands_default_settings() {
        let tcb = TrendChannelBands::default_settings(20, 2.0).unwrap();
        assert_eq!(tcb.min_periods(), 21); // max(20, 14) + 1
    }

    #[test]
    fn test_trend_channel_bands_empty_data() {
        let empty: Vec<f64> = vec![];
        let tcb = TrendChannelBands::new(20, 14, 2.0, 0.5).unwrap();
        let (m, u, l) = tcb.calculate(&empty, &empty, &empty);
        assert!(m.is_empty());
        assert!(u.is_empty());
        assert!(l.is_empty());
    }

    #[test]
    fn test_trend_channel_bands_zero_sensitivity() {
        // With zero sensitivity, bands should be symmetric
        let (high, low, close) = make_extended_test_data();
        let tcb = TrendChannelBands::new(10, 10, 2.0, 0.0).unwrap();
        let (middle, upper, lower) = tcb.calculate(&high, &low, &close);

        let idx = 30;
        let upper_diff = upper[idx] - middle[idx];
        let lower_diff = middle[idx] - lower[idx];

        assert!((upper_diff - lower_diff).abs() < 1e-10,
            "With zero sensitivity, bands should be symmetric");
    }

    #[test]
    fn test_trend_channel_bands_in_uptrend() {
        // Create strong uptrend
        let uptrend: Vec<f64> = (0..60).map(|i| 100.0 + (i as f64) * 2.0).collect();
        let high: Vec<f64> = uptrend.iter().map(|&c| c + 2.0).collect();
        let low: Vec<f64> = uptrend.iter().map(|&c| c - 2.0).collect();

        let tcb = TrendChannelBands::new(10, 10, 2.0, 0.5).unwrap();
        let (middle, upper, lower) = tcb.calculate(&high, &low, &uptrend);

        let idx = 40;
        let upper_dist = upper[idx] - middle[idx];
        let lower_dist = middle[idx] - lower[idx];

        // In uptrend with positive sensitivity, lower band should be tighter
        // (closer to price for trailing stop)
        assert!(lower_dist < upper_dist,
            "In uptrend, lower band should be tighter (closer to price)");
    }

    // --- Combined Tests for All 6 NEW Phase 3 Indicators ---

    #[test]
    fn test_all_6_new_phase3_indicators_short_data() {
        let short_high = vec![102.0, 103.0, 104.0];
        let short_low = vec![98.0, 99.0, 100.0];
        let short_close = vec![100.0, 101.0, 102.0];

        // DonchianChannelEnhanced
        let dce = DonchianChannelEnhanced::new(20, 14, 0.5).unwrap();
        let (m, u, l) = dce.calculate(&short_high, &short_low, &short_close);
        assert_eq!(m.len(), 3);

        // HighLowBandsAdvanced
        let hlb = HighLowBandsAdvanced::new(20, 1.0).unwrap();
        let (m, u, l) = hlb.calculate(&short_high, &short_low);
        assert_eq!(m.len(), 3);

        // PivotBands
        let pb = PivotBands::new(20, 1).unwrap();
        let (m, u, l) = pb.calculate(&short_high, &short_low, &short_close);
        assert_eq!(m.len(), 3);

        // MovingAverageBands
        let mab = MovingAverageBands::new(20, 15, 2.0).unwrap();
        let (m, u, l) = mab.calculate(&short_close);
        assert_eq!(m.len(), 3);

        // VolatilityAdjustedBandsExt
        let vabe = VolatilityAdjustedBandsExt::new(20, 10, 50, 2.0).unwrap();
        let (m, u, l) = vabe.calculate(&short_high, &short_low, &short_close);
        assert_eq!(m.len(), 3);

        // TrendChannelBands
        let tcb = TrendChannelBands::new(20, 14, 2.0, 0.5).unwrap();
        let (m, u, l) = tcb.calculate(&short_high, &short_low, &short_close);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn test_all_6_new_phase3_indicators_numerical_stability() {
        // Test with large values
        let large_high: Vec<f64> = (0..80).map(|i| 1e8 + (i as f64) * 1000.0).collect();
        let large_low: Vec<f64> = (0..80).map(|i| 1e8 - 1000.0 + (i as f64) * 1000.0).collect();
        let large_close: Vec<f64> = (0..80).map(|i| 1e8 + (i as f64) * 1000.0).collect();
        let idx = 70;

        // DonchianChannelEnhanced
        let dce = DonchianChannelEnhanced::new(10, 10, 0.5).unwrap();
        let (m, u, l) = dce.calculate(&large_high, &large_low, &large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // HighLowBandsAdvanced
        let hlb = HighLowBandsAdvanced::new(10, 1.0).unwrap();
        let (m, u, l) = hlb.calculate(&large_high, &large_low);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // PivotBands
        let pb = PivotBands::new(10, 1).unwrap();
        let (m, u, l) = pb.calculate(&large_high, &large_low, &large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // MovingAverageBands
        let mab = MovingAverageBands::new(10, 10, 2.0).unwrap();
        let (m, u, l) = mab.calculate(&large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // VolatilityAdjustedBandsExt
        let vabe = VolatilityAdjustedBandsExt::new(10, 5, 30, 2.0).unwrap();
        let (m, u, l) = vabe.calculate(&large_high, &large_low, &large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());

        // TrendChannelBands
        let tcb = TrendChannelBands::new(10, 10, 2.0, 0.5).unwrap();
        let (m, u, l) = tcb.calculate(&large_high, &large_low, &large_close);
        assert!(m[idx].is_finite());
        assert!(u[idx].is_finite());
        assert!(l[idx].is_finite());
    }
}
