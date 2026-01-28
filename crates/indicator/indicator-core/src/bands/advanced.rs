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
}
