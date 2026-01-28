//! Advanced Volatility Indicators
//!
//! Additional volatility analysis indicators for trend, momentum,
//! relative comparison, skew, implied volatility proxy, and persistence.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Volatility Trend - Tracks volatility trend direction.
///
/// Uses a smoothed volatility measure and its moving average
/// to determine if volatility is trending up or down.
#[derive(Debug, Clone)]
pub struct VolatilityTrend {
    volatility_period: usize,
    trend_period: usize,
}

impl VolatilityTrend {
    /// Create a new VolatilityTrend indicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for calculating rolling volatility
    /// * `trend_period` - Period for smoothing the trend
    pub fn new(volatility_period: usize, trend_period: usize) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if trend_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "trend_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { volatility_period, trend_period })
    }

    /// Calculate volatility trend direction.
    /// Positive values indicate rising volatility, negative indicates falling.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_period = self.volatility_period + self.trend_period;

        if n < total_period + 1 {
            return vec![0.0; n];
        }

        // Calculate rolling volatility
        let mut volatility = vec![0.0; n];
        for i in self.volatility_period..n {
            let start = i.saturating_sub(self.volatility_period);
            let returns: Vec<f64> = ((start + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 1e-10 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 2 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let var: f64 = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                volatility[i] = var.sqrt() * (252.0_f64).sqrt();
            }
        }

        // Calculate trend as difference from moving average
        let mut result = vec![0.0; n];
        for i in total_period..n {
            let start = i.saturating_sub(self.trend_period);
            let avg_vol: f64 = volatility[start..i].iter().sum::<f64>() / self.trend_period as f64;

            if avg_vol > 1e-10 {
                // Normalized difference: positive = vol above MA, negative = vol below MA
                result[i] = (volatility[i] - avg_vol) / avg_vol * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for VolatilityTrend {
    fn name(&self) -> &str {
        "Volatility Trend"
    }

    fn min_periods(&self) -> usize {
        self.volatility_period + self.trend_period + 1
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

/// Volatility Momentum - Rate of change in volatility.
///
/// Measures how quickly volatility is changing, similar to
/// momentum indicators but applied to volatility itself.
#[derive(Debug, Clone)]
pub struct VolatilityMomentum {
    volatility_period: usize,
    momentum_period: usize,
}

impl VolatilityMomentum {
    /// Create a new VolatilityMomentum indicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for calculating rolling volatility
    /// * `momentum_period` - Period for calculating rate of change
    pub fn new(volatility_period: usize, momentum_period: usize) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if momentum_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { volatility_period, momentum_period })
    }

    /// Calculate volatility momentum (rate of change in volatility).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_period = self.volatility_period + self.momentum_period;

        if n < total_period + 1 {
            return vec![0.0; n];
        }

        // Calculate rolling volatility
        let mut volatility = vec![0.0; n];
        for i in self.volatility_period..n {
            let start = i.saturating_sub(self.volatility_period);
            let returns: Vec<f64> = ((start + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 1e-10 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 2 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let var: f64 = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                volatility[i] = var.sqrt() * (252.0_f64).sqrt();
            }
        }

        // Calculate rate of change
        let mut result = vec![0.0; n];
        for i in total_period..n {
            let prev_vol = volatility[i - self.momentum_period];
            if prev_vol > 1e-10 {
                result[i] = (volatility[i] - prev_vol) / prev_vol * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for VolatilityMomentum {
    fn name(&self) -> &str {
        "Volatility Momentum"
    }

    fn min_periods(&self) -> usize {
        self.volatility_period + self.momentum_period + 1
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

/// Relative Volatility - Compares current volatility to historical volatility.
///
/// Shows how current volatility compares to a longer historical average,
/// expressed as a ratio.
#[derive(Debug, Clone)]
pub struct RelativeVolatility {
    short_period: usize,
    long_period: usize,
}

impl RelativeVolatility {
    /// Create a new RelativeVolatility indicator.
    ///
    /// # Arguments
    /// * `short_period` - Period for current volatility
    /// * `long_period` - Period for historical volatility comparison
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

    /// Calculate relative volatility ratio.
    /// Values > 1 indicate elevated volatility, < 1 indicates subdued volatility.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n < self.long_period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term volatility
            let short_start = i.saturating_sub(self.short_period);
            let short_returns: Vec<f64> = ((short_start + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 1e-10 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            let short_vol = if short_returns.len() >= 2 {
                let mean: f64 = short_returns.iter().sum::<f64>() / short_returns.len() as f64;
                let var: f64 = short_returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / short_returns.len() as f64;
                var.sqrt()
            } else {
                0.0
            };

            // Long-term volatility
            let long_start = i.saturating_sub(self.long_period);
            let long_returns: Vec<f64> = ((long_start + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 1e-10 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            let long_vol = if long_returns.len() >= 2 {
                let mean: f64 = long_returns.iter().sum::<f64>() / long_returns.len() as f64;
                let var: f64 = long_returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / long_returns.len() as f64;
                var.sqrt()
            } else {
                0.0
            };

            if long_vol > 1e-10 {
                result[i] = short_vol / long_vol;
            }
        }
        result
    }
}

impl TechnicalIndicator for RelativeVolatility {
    fn name(&self) -> &str {
        "Relative Volatility"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
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

/// Price Volatility Skew - Measures asymmetry in price movements.
///
/// Calculates the skewness of returns to identify whether
/// large moves tend to be positive or negative.
/// Named PriceVolatilitySkew to distinguish from VolatilitySkew in vix_derived.
#[derive(Debug, Clone)]
pub struct PriceVolatilitySkew {
    period: usize,
}

impl PriceVolatilitySkew {
    /// Create a new PriceVolatilitySkew indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for skewness calculation
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volatility skew using upside vs downside volatility.
    /// Positive values indicate more upside volatility (bullish),
    /// negative indicates more downside volatility (bearish).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len()).min(close.len());

        if n < self.period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate upside and downside moves
            let mut upside_moves = Vec::new();
            let mut downside_moves = Vec::new();

            for j in (start + 1)..=i {
                let ret = if close[j - 1] > 1e-10 {
                    (close[j] / close[j - 1]).ln()
                } else {
                    0.0
                };

                if ret > 0.0 {
                    upside_moves.push(ret);
                } else if ret < 0.0 {
                    downside_moves.push(ret.abs());
                }
            }

            // Calculate upside volatility
            let upside_vol = if upside_moves.len() >= 2 {
                let mean: f64 = upside_moves.iter().sum::<f64>() / upside_moves.len() as f64;
                let var: f64 = upside_moves.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / upside_moves.len() as f64;
                var.sqrt()
            } else {
                0.0
            };

            // Calculate downside volatility
            let downside_vol = if downside_moves.len() >= 2 {
                let mean: f64 = downside_moves.iter().sum::<f64>() / downside_moves.len() as f64;
                let var: f64 = downside_moves.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / downside_moves.len() as f64;
                var.sqrt()
            } else {
                0.0
            };

            // Skew ratio: positive = more upside vol, negative = more downside vol
            let total_vol = upside_vol + downside_vol;
            if total_vol > 1e-10 {
                result[i] = (upside_vol - downside_vol) / total_vol * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for PriceVolatilitySkew {
    fn name(&self) -> &str {
        "Price Volatility Skew"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Implied Volatility Proxy - Price-based implied volatility approximation.
///
/// Uses ATR and historical volatility to approximate implied volatility
/// when actual options data is not available.
#[derive(Debug, Clone)]
pub struct ImpliedVolatilityProxy {
    atr_period: usize,
    vol_period: usize,
    blend_factor: f64,
}

impl ImpliedVolatilityProxy {
    /// Create a new ImpliedVolatilityProxy indicator.
    ///
    /// # Arguments
    /// * `atr_period` - Period for ATR calculation
    /// * `vol_period` - Period for historical volatility
    /// * `blend_factor` - Weight for ATR vs HV (0.0-1.0, higher = more ATR weight)
    pub fn new(atr_period: usize, vol_period: usize, blend_factor: f64) -> Result<Self> {
        if atr_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if vol_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "vol_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&blend_factor) {
            return Err(IndicatorError::InvalidParameter {
                name: "blend_factor".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self { atr_period, vol_period, blend_factor })
    }

    /// Calculate implied volatility proxy.
    /// Returns annualized volatility estimate.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len()).min(close.len());
        let total_period = self.atr_period.max(self.vol_period);

        if n < total_period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        for i in total_period..n {
            // Calculate ATR-based volatility
            let atr_start = i.saturating_sub(self.atr_period);
            let mut atr_sum = 0.0;
            let mut atr_count = 0;

            for j in (atr_start + 1)..=i {
                let tr = (high[j] - low[j])
                    .max((high[j] - close[j - 1]).abs())
                    .max((low[j] - close[j - 1]).abs());
                atr_sum += tr;
                atr_count += 1;
            }

            let atr = if atr_count > 0 { atr_sum / atr_count as f64 } else { 0.0 };
            let atr_vol = if close[i] > 1e-10 {
                (atr / close[i]) * (252.0_f64).sqrt() * 100.0
            } else {
                0.0
            };

            // Calculate historical volatility
            let vol_start = i.saturating_sub(self.vol_period);
            let returns: Vec<f64> = ((vol_start + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 1e-10 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            let hist_vol = if returns.len() >= 2 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let var: f64 = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                var.sqrt() * (252.0_f64).sqrt() * 100.0
            } else {
                0.0
            };

            // Blend ATR and HV for IV proxy
            result[i] = self.blend_factor * atr_vol + (1.0 - self.blend_factor) * hist_vol;
        }
        result
    }
}

impl TechnicalIndicator for ImpliedVolatilityProxy {
    fn name(&self) -> &str {
        "Implied Volatility Proxy"
    }

    fn min_periods(&self) -> usize {
        self.atr_period.max(self.vol_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Volatility Persistence - Measures volatility clustering.
///
/// Uses autocorrelation of volatility to measure how persistent
/// volatility regimes are (high volatility tends to follow high volatility).
#[derive(Debug, Clone)]
pub struct VolatilityPersistence {
    volatility_period: usize,
    lag_period: usize,
    correlation_period: usize,
}

impl VolatilityPersistence {
    /// Create a new VolatilityPersistence indicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for calculating rolling volatility
    /// * `lag_period` - Lag for autocorrelation
    /// * `correlation_period` - Period for correlation calculation
    pub fn new(volatility_period: usize, lag_period: usize, correlation_period: usize) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if lag_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "lag_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if correlation_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "correlation_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { volatility_period, lag_period, correlation_period })
    }

    /// Calculate volatility persistence (autocorrelation).
    /// Values close to 1 indicate high persistence, close to 0 indicates random.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_period = self.volatility_period + self.lag_period + self.correlation_period;

        if n < total_period + 1 {
            return vec![0.0; n];
        }

        // Calculate rolling volatility
        let mut volatility = vec![0.0; n];
        for i in self.volatility_period..n {
            let start = i.saturating_sub(self.volatility_period);
            let returns: Vec<f64> = ((start + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 1e-10 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 2 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let var: f64 = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                volatility[i] = var.sqrt();
            }
        }

        // Calculate autocorrelation
        let mut result = vec![0.0; n];

        for i in total_period..n {
            let start = i.saturating_sub(self.correlation_period);

            // Current volatility series
            let vol_current: Vec<f64> = (start..=i)
                .map(|j| volatility[j])
                .collect();

            // Lagged volatility series
            let vol_lagged: Vec<f64> = (start..=i)
                .map(|j| volatility[j.saturating_sub(self.lag_period)])
                .collect();

            if vol_current.len() >= 2 && vol_lagged.len() >= 2 {
                let mean_current: f64 = vol_current.iter().sum::<f64>() / vol_current.len() as f64;
                let mean_lagged: f64 = vol_lagged.iter().sum::<f64>() / vol_lagged.len() as f64;

                // Covariance
                let cov: f64 = vol_current.iter().zip(vol_lagged.iter())
                    .map(|(c, l)| (c - mean_current) * (l - mean_lagged))
                    .sum::<f64>() / vol_current.len() as f64;

                // Standard deviations
                let std_current: f64 = (vol_current.iter()
                    .map(|v| (v - mean_current).powi(2))
                    .sum::<f64>() / vol_current.len() as f64).sqrt();

                let std_lagged: f64 = (vol_lagged.iter()
                    .map(|v| (v - mean_lagged).powi(2))
                    .sum::<f64>() / vol_lagged.len() as f64).sqrt();

                // Correlation
                if std_current > 1e-10 && std_lagged > 1e-10 {
                    result[i] = cov / (std_current * std_lagged);
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for VolatilityPersistence {
    fn name(&self) -> &str {
        "Volatility Persistence"
    }

    fn min_periods(&self) -> usize {
        self.volatility_period + self.lag_period + self.correlation_period + 1
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..100)
            .map(|i| 102.0 + (i as f64 * 0.15).sin() * 3.0 + i as f64 * 0.1)
            .collect();
        let low: Vec<f64> = (0..100)
            .map(|i| 98.0 + (i as f64 * 0.15).sin() * 3.0 + i as f64 * 0.1)
            .collect();
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.15).sin() * 3.0 + i as f64 * 0.1)
            .collect();
        (high, low, close)
    }

    #[test]
    fn test_volatility_trend() {
        let (_, _, close) = make_test_data();
        let vt = VolatilityTrend::new(10, 5).unwrap();
        let result = vt.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Check that values are computed after warmup period
        assert!(result[20] != 0.0 || result[25] != 0.0);
    }

    #[test]
    fn test_volatility_trend_invalid_params() {
        assert!(VolatilityTrend::new(2, 5).is_err());
        assert!(VolatilityTrend::new(10, 0).is_err());
    }

    #[test]
    fn test_volatility_momentum() {
        let (_, _, close) = make_test_data();
        let vm = VolatilityMomentum::new(10, 5).unwrap();
        let result = vm.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_volatility_momentum_invalid_params() {
        assert!(VolatilityMomentum::new(2, 5).is_err());
        assert!(VolatilityMomentum::new(10, 0).is_err());
    }

    #[test]
    fn test_relative_volatility() {
        let (_, _, close) = make_test_data();
        let rv = RelativeVolatility::new(10, 30).unwrap();
        let result = rv.calculate(&close);

        assert_eq!(result.len(), close.len());
        // After warmup, values should be positive ratios
        for i in 35..close.len() {
            assert!(result[i] >= 0.0, "Value at {} should be non-negative", i);
        }
    }

    #[test]
    fn test_relative_volatility_invalid_params() {
        assert!(RelativeVolatility::new(2, 30).is_err());
        assert!(RelativeVolatility::new(10, 10).is_err()); // long must be > short
        assert!(RelativeVolatility::new(10, 5).is_err());  // long must be > short
    }

    #[test]
    fn test_price_volatility_skew() {
        let (high, low, close) = make_test_data();
        let vs = PriceVolatilitySkew::new(20).unwrap();
        let result = vs.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Skew should be between -100 and 100
        for i in 25..close.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0,
                "Skew at {} should be between -100 and 100, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_price_volatility_skew_invalid_params() {
        assert!(PriceVolatilitySkew::new(5).is_err());
    }

    #[test]
    fn test_implied_volatility_proxy() {
        let (high, low, close) = make_test_data();
        let ivp = ImpliedVolatilityProxy::new(14, 20, 0.5).unwrap();
        let result = ivp.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // IV proxy should be positive after warmup
        for i in 25..close.len() {
            assert!(result[i] >= 0.0, "IV proxy at {} should be non-negative", i);
        }
    }

    #[test]
    fn test_implied_volatility_proxy_invalid_params() {
        assert!(ImpliedVolatilityProxy::new(2, 20, 0.5).is_err());
        assert!(ImpliedVolatilityProxy::new(14, 2, 0.5).is_err());
        assert!(ImpliedVolatilityProxy::new(14, 20, -0.1).is_err());
        assert!(ImpliedVolatilityProxy::new(14, 20, 1.5).is_err());
    }

    #[test]
    fn test_volatility_persistence() {
        let (_, _, close) = make_test_data();
        let vp = VolatilityPersistence::new(10, 1, 20).unwrap();
        let result = vp.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Persistence (correlation) should be between -1 and 1
        for i in 35..close.len() {
            assert!(result[i] >= -1.0 && result[i] <= 1.0,
                "Persistence at {} should be between -1 and 1, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_volatility_persistence_invalid_params() {
        assert!(VolatilityPersistence::new(2, 1, 20).is_err());
        assert!(VolatilityPersistence::new(10, 0, 20).is_err());
        assert!(VolatilityPersistence::new(10, 1, 5).is_err());
    }
}
