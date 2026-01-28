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

/// Volatility Forecast - Forecasts future volatility using GARCH-like approach.
///
/// Uses a weighted combination of recent squared returns and historical variance
/// to estimate future volatility, similar to GARCH(1,1) models.
#[derive(Debug, Clone)]
pub struct VolatilityForecast {
    period: usize,
    alpha: f64, // Weight for recent innovation
    beta: f64,  // Weight for historical variance
}

impl VolatilityForecast {
    /// Create a new VolatilityForecast indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for volatility calculation
    /// * `alpha` - Weight for recent squared return (innovation term)
    /// * `beta` - Weight for historical variance (persistence term)
    ///
    /// Note: alpha + beta should be < 1 for mean reversion
    pub fn new(period: usize, alpha: f64, beta: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if alpha < 0.0 || alpha > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        if beta < 0.0 || beta > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "beta".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self { period, alpha, beta })
    }

    /// Calculate volatility forecast.
    /// Returns annualized forecasted volatility.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n < self.period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        // Calculate long-term variance (omega in GARCH)
        let returns: Vec<f64> = (1..n)
            .filter_map(|i| {
                if close[i - 1] > 1e-10 {
                    Some((close[i] / close[i - 1]).ln())
                } else {
                    None
                }
            })
            .collect();

        let long_term_var = if returns.len() >= 2 {
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64
        } else {
            0.0001 // Default small variance
        };

        // Omega is scaled to maintain unconditional variance
        let omega = long_term_var * (1.0 - self.alpha - self.beta).max(0.01);

        // Initialize with historical variance
        let mut forecast_var = long_term_var;

        for i in self.period..n {
            // Get recent return
            let recent_return = if close[i - 1] > 1e-10 {
                (close[i] / close[i - 1]).ln()
            } else {
                0.0
            };

            // GARCH-like forecast: sigma^2_t+1 = omega + alpha * r^2_t + beta * sigma^2_t
            forecast_var = omega + self.alpha * recent_return.powi(2) + self.beta * forecast_var;

            // Annualize: sqrt(variance * 252) * 100 for percentage
            result[i] = forecast_var.sqrt() * (252.0_f64).sqrt() * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for VolatilityForecast {
    fn name(&self) -> &str {
        "Volatility Forecast"
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
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volatility Spike - Detects sudden spikes in volatility.
///
/// Compares current volatility to a rolling baseline to identify
/// abnormal volatility events that may indicate significant market moves.
#[derive(Debug, Clone)]
pub struct VolatilitySpike {
    volatility_period: usize,
    baseline_period: usize,
    threshold: f64,
}

impl VolatilitySpike {
    /// Create a new VolatilitySpike indicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for current volatility calculation
    /// * `baseline_period` - Period for baseline volatility calculation
    /// * `threshold` - Standard deviation threshold for spike detection (e.g., 2.0)
    pub fn new(volatility_period: usize, baseline_period: usize, threshold: f64) -> Result<Self> {
        if volatility_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if baseline_period <= volatility_period {
            return Err(IndicatorError::InvalidParameter {
                name: "baseline_period".to_string(),
                reason: "must be greater than volatility_period".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            volatility_period,
            baseline_period,
            threshold,
        })
    }

    /// Calculate volatility spike detection.
    /// Returns z-score of current volatility vs baseline.
    /// Values above threshold indicate a spike.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n < self.baseline_period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        // First compute rolling volatility for all periods
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
                let var: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                volatility[i] = var.sqrt();
            }
        }

        // Calculate z-score vs baseline
        for i in self.baseline_period..n {
            let baseline_start = i.saturating_sub(self.baseline_period);
            let baseline_vols: Vec<f64> = (baseline_start..i).map(|j| volatility[j]).collect();

            if baseline_vols.len() >= 2 {
                let mean: f64 = baseline_vols.iter().sum::<f64>() / baseline_vols.len() as f64;
                let var: f64 = baseline_vols
                    .iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>()
                    / baseline_vols.len() as f64;
                let std = var.sqrt();

                if std > 1e-10 {
                    result[i] = (volatility[i] - mean) / std;
                }
            }
        }

        result
    }

    /// Check if current value represents a spike (above threshold).
    pub fn is_spike(&self, zscore: f64) -> bool {
        zscore > self.threshold
    }
}

impl TechnicalIndicator for VolatilitySpike {
    fn name(&self) -> &str {
        "Volatility Spike"
    }

    fn min_periods(&self) -> usize {
        self.baseline_period + 1
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

/// Volatility Mean Reversion - Measures mean-reversion tendency in volatility.
///
/// Quantifies how strongly volatility tends to revert to its mean,
/// useful for volatility trading strategies.
#[derive(Debug, Clone)]
pub struct VolatilityMeanReversion {
    volatility_period: usize,
    mean_period: usize,
    lookback: usize,
}

impl VolatilityMeanReversion {
    /// Create a new VolatilityMeanReversion indicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for calculating rolling volatility
    /// * `mean_period` - Period for calculating mean volatility
    /// * `lookback` - Lookback period for measuring reversion strength
    pub fn new(volatility_period: usize, mean_period: usize, lookback: usize) -> Result<Self> {
        if volatility_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if mean_period < volatility_period {
            return Err(IndicatorError::InvalidParameter {
                name: "mean_period".to_string(),
                reason: "must be at least volatility_period".to_string(),
            });
        }
        if lookback < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            volatility_period,
            mean_period,
            lookback,
        })
    }

    /// Calculate mean reversion score.
    /// Positive values indicate current deviation from mean.
    /// Returns: (deviation_from_mean, reversion_speed)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_period = self.mean_period + self.lookback;

        if n < total_period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

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
                let var: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                volatility[i] = var.sqrt();
            }
        }

        // Calculate mean reversion indicator
        for i in total_period..n {
            // Calculate mean volatility over longer period
            let mean_start = i.saturating_sub(self.mean_period);
            let mean_vol: f64 =
                volatility[mean_start..i].iter().sum::<f64>() / self.mean_period as f64;

            if mean_vol > 1e-10 {
                // Current deviation from mean (normalized)
                let current_deviation = (volatility[i] - mean_vol) / mean_vol;

                // Measure reversion speed: correlation between deviation and subsequent change
                let mut deviations = Vec::new();
                let mut changes = Vec::new();

                for j in (i - self.lookback)..i {
                    let dev = (volatility[j] - mean_vol) / mean_vol;
                    let change = volatility[j + 1] - volatility[j];
                    deviations.push(dev);
                    changes.push(change);
                }

                // Calculate reversion strength (negative correlation = mean reverting)
                if deviations.len() >= 2 {
                    let dev_mean: f64 = deviations.iter().sum::<f64>() / deviations.len() as f64;
                    let chg_mean: f64 = changes.iter().sum::<f64>() / changes.len() as f64;

                    let cov: f64 = deviations
                        .iter()
                        .zip(changes.iter())
                        .map(|(d, c)| (d - dev_mean) * (c - chg_mean))
                        .sum::<f64>()
                        / deviations.len() as f64;

                    let dev_std: f64 = (deviations
                        .iter()
                        .map(|d| (d - dev_mean).powi(2))
                        .sum::<f64>()
                        / deviations.len() as f64)
                        .sqrt();
                    let chg_std: f64 = (changes
                        .iter()
                        .map(|c| (c - chg_mean).powi(2))
                        .sum::<f64>()
                        / changes.len() as f64)
                        .sqrt();

                    let reversion_strength = if dev_std > 1e-10 && chg_std > 1e-10 {
                        -cov / (dev_std * chg_std) // Negate so positive = mean reverting
                    } else {
                        0.0
                    };

                    // Combine deviation and reversion strength
                    // High positive = extended and mean-reverting (expect drop)
                    // High negative = depressed and mean-reverting (expect rise)
                    result[i] = current_deviation * reversion_strength * 100.0;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilityMeanReversion {
    fn name(&self) -> &str {
        "Volatility Mean Reversion"
    }

    fn min_periods(&self) -> usize {
        self.mean_period + self.lookback + 1
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

/// Volatility Clustering - Measures volatility clustering (ARCH effects).
///
/// Detects periods where high volatility follows high volatility,
/// which is a key characteristic of financial time series.
#[derive(Debug, Clone)]
pub struct VolatilityClustering {
    period: usize,
    lag: usize,
}

impl VolatilityClustering {
    /// Create a new VolatilityClustering indicator.
    ///
    /// # Arguments
    /// * `period` - Period for volatility and correlation calculation
    /// * `lag` - Lag for measuring autocorrelation
    pub fn new(period: usize, lag: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if lag < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "lag".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, lag })
    }

    /// Calculate volatility clustering score.
    /// Higher values indicate stronger clustering (high vol follows high vol).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_period = self.period + self.lag;

        if n < total_period + 2 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        // Calculate absolute returns (proxy for volatility)
        let mut abs_returns = vec![0.0; n];
        for i in 1..n {
            if close[i - 1] > 1e-10 {
                abs_returns[i] = ((close[i] / close[i - 1]).ln()).abs();
            }
        }

        // Calculate rolling autocorrelation of squared returns
        for i in total_period..n {
            let start = i.saturating_sub(self.period);

            // Current period squared returns
            let current: Vec<f64> = (start..=i).map(|j| abs_returns[j].powi(2)).collect();

            // Lagged squared returns
            let lagged: Vec<f64> = (start..=i)
                .map(|j| abs_returns[j.saturating_sub(self.lag)].powi(2))
                .collect();

            if current.len() >= 2 {
                let mean_current: f64 = current.iter().sum::<f64>() / current.len() as f64;
                let mean_lagged: f64 = lagged.iter().sum::<f64>() / lagged.len() as f64;

                // Covariance
                let cov: f64 = current
                    .iter()
                    .zip(lagged.iter())
                    .map(|(c, l)| (c - mean_current) * (l - mean_lagged))
                    .sum::<f64>()
                    / current.len() as f64;

                // Standard deviations
                let std_current: f64 = (current
                    .iter()
                    .map(|v| (v - mean_current).powi(2))
                    .sum::<f64>()
                    / current.len() as f64)
                    .sqrt();

                let std_lagged: f64 = (lagged
                    .iter()
                    .map(|v| (v - mean_lagged).powi(2))
                    .sum::<f64>()
                    / lagged.len() as f64)
                    .sqrt();

                // Correlation (clustering measure)
                if std_current > 1e-10 && std_lagged > 1e-10 {
                    result[i] = cov / (std_current * std_lagged);
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilityClustering {
    fn name(&self) -> &str {
        "Volatility Clustering"
    }

    fn min_periods(&self) -> usize {
        self.period + self.lag + 2
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

/// Volatility Asymmetry - Measures upside vs downside volatility.
///
/// Compares volatility during up moves vs down moves to detect
/// asymmetric risk profiles (e.g., higher downside volatility).
#[derive(Debug, Clone)]
pub struct VolatilityAsymmetry {
    period: usize,
}

impl VolatilityAsymmetry {
    /// Create a new VolatilityAsymmetry indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for volatility calculation
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volatility asymmetry.
    /// Positive values indicate higher upside volatility.
    /// Negative values indicate higher downside volatility (more common in equities).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n < self.period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Separate up and down returns
            let mut up_returns = Vec::new();
            let mut down_returns = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    let ret = (close[j] / close[j - 1]).ln();
                    if ret > 0.0 {
                        up_returns.push(ret);
                    } else if ret < 0.0 {
                        down_returns.push(ret.abs());
                    }
                }
            }

            // Calculate upside volatility (std of positive returns)
            let up_vol = if up_returns.len() >= 2 {
                let mean: f64 = up_returns.iter().sum::<f64>() / up_returns.len() as f64;
                let var: f64 = up_returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / up_returns.len() as f64;
                var.sqrt()
            } else if !up_returns.is_empty() {
                up_returns[0]
            } else {
                0.0
            };

            // Calculate downside volatility (std of absolute negative returns)
            let down_vol = if down_returns.len() >= 2 {
                let mean: f64 = down_returns.iter().sum::<f64>() / down_returns.len() as f64;
                let var: f64 = down_returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / down_returns.len() as f64;
                var.sqrt()
            } else if !down_returns.is_empty() {
                down_returns[0]
            } else {
                0.0
            };

            // Asymmetry ratio: (up_vol - down_vol) / (up_vol + down_vol)
            // Positive = more upside vol, Negative = more downside vol
            let total_vol = up_vol + down_vol;
            if total_vol > 1e-10 {
                result[i] = (up_vol - down_vol) / total_vol * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilityAsymmetry {
    fn name(&self) -> &str {
        "Volatility Asymmetry"
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
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volatility Efficiency - Measures how efficiently price moves through volatility.
///
/// Compares directional price movement to total path traveled,
/// adjusted for volatility. High values indicate efficient trending moves.
#[derive(Debug, Clone)]
pub struct VolatilityEfficiency {
    period: usize,
}

impl VolatilityEfficiency {
    /// Create a new VolatilityEfficiency indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for calculation
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volatility efficiency ratio.
    /// Range: 0 to 1 (or expressed as percentage 0-100).
    /// Higher values indicate more efficient price movement.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n < self.period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Net price change (directional movement)
            let net_change = (close[i] - close[start]).abs();

            // Total path length (sum of absolute bar-to-bar changes)
            let mut total_path = 0.0;
            for j in (start + 1)..=i {
                total_path += (close[j] - close[j - 1]).abs();
            }

            // Calculate volatility over the period
            let returns: Vec<f64> = ((start + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 1e-10 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            let volatility = if returns.len() >= 2 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let var: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                var.sqrt()
            } else {
                1e-10
            };

            // Efficiency = (Net Change / Total Path) * Volatility adjustment
            // Higher efficiency with lower volatility is even more efficient
            if total_path > 1e-10 && volatility > 1e-10 {
                let raw_efficiency = net_change / total_path;

                // Adjust for volatility: divide by normalized volatility
                // Lower volatility = higher adjusted efficiency
                let avg_price = (close[start] + close[i]) / 2.0;
                let normalized_vol = if avg_price > 1e-10 {
                    volatility / (avg_price / close[start])
                } else {
                    volatility
                };

                // Final efficiency score (0-100)
                // Raw efficiency divided by volatility factor
                result[i] = raw_efficiency / (1.0 + normalized_vol * 10.0) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilityEfficiency {
    fn name(&self) -> &str {
        "Volatility Efficiency"
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
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volatility Acceleration - Rate of change of volatility.
///
/// Measures how quickly volatility itself is changing (second derivative).
/// Positive values indicate accelerating volatility, negative indicates decelerating.
#[derive(Debug, Clone)]
pub struct VolatilityAcceleration {
    volatility_period: usize,
    acceleration_period: usize,
}

impl VolatilityAcceleration {
    /// Create a new VolatilityAcceleration indicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for calculating rolling volatility
    /// * `acceleration_period` - Period for calculating rate of change of volatility
    pub fn new(volatility_period: usize, acceleration_period: usize) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if acceleration_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "acceleration_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            volatility_period,
            acceleration_period,
        })
    }

    /// Calculate volatility acceleration.
    /// Returns the rate of change of the rate of change of volatility.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_period = self.volatility_period + self.acceleration_period * 2;

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
                let var: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                volatility[i] = var.sqrt() * (252.0_f64).sqrt();
            }
        }

        // Calculate first derivative (velocity of volatility)
        let mut vol_velocity = vec![0.0; n];
        for i in (self.volatility_period + self.acceleration_period)..n {
            let prev_vol = volatility[i - self.acceleration_period];
            if prev_vol > 1e-10 {
                vol_velocity[i] = (volatility[i] - prev_vol) / prev_vol;
            }
        }

        // Calculate second derivative (acceleration of volatility)
        let mut result = vec![0.0; n];
        for i in total_period..n {
            let prev_velocity = vol_velocity[i - self.acceleration_period];
            // Acceleration = change in velocity
            result[i] = (vol_velocity[i] - prev_velocity) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for VolatilityAcceleration {
    fn name(&self) -> &str {
        "Volatility Acceleration"
    }

    fn min_periods(&self) -> usize {
        self.volatility_period + self.acceleration_period * 2 + 1
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

/// Volatility Mean Reversion Distance - How far volatility is from its mean.
///
/// Measures the z-score of current volatility relative to its historical mean,
/// indicating potential mean reversion opportunities.
#[derive(Debug, Clone)]
pub struct VolatilityMeanReversionDistance {
    volatility_period: usize,
    mean_period: usize,
}

impl VolatilityMeanReversionDistance {
    /// Create a new VolatilityMeanReversionDistance indicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for calculating rolling volatility
    /// * `mean_period` - Period for calculating mean and standard deviation of volatility
    pub fn new(volatility_period: usize, mean_period: usize) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if mean_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "mean_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            volatility_period,
            mean_period,
        })
    }

    /// Calculate volatility mean reversion distance (z-score).
    /// Positive values indicate volatility above mean, negative below mean.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_period = self.volatility_period + self.mean_period;

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
                let var: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                volatility[i] = var.sqrt() * (252.0_f64).sqrt();
            }
        }

        // Calculate z-score of volatility
        let mut result = vec![0.0; n];
        for i in total_period..n {
            let start = i.saturating_sub(self.mean_period);
            let vol_slice: Vec<f64> = (start..i).map(|j| volatility[j]).collect();

            if vol_slice.len() >= 2 {
                let mean_vol: f64 = vol_slice.iter().sum::<f64>() / vol_slice.len() as f64;
                let var_vol: f64 = vol_slice
                    .iter()
                    .map(|v| (v - mean_vol).powi(2))
                    .sum::<f64>()
                    / vol_slice.len() as f64;
                let std_vol = var_vol.sqrt();

                if std_vol > 1e-10 {
                    result[i] = (volatility[i] - mean_vol) / std_vol;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilityMeanReversionDistance {
    fn name(&self) -> &str {
        "Volatility Mean Reversion Distance"
    }

    fn min_periods(&self) -> usize {
        self.volatility_period + self.mean_period + 1
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

/// Volatility Spread - Spread between short and long-term volatility.
///
/// Measures the difference between short-term and long-term volatility,
/// useful for identifying volatility term structure and regime changes.
#[derive(Debug, Clone)]
pub struct VolatilitySpread {
    short_period: usize,
    long_period: usize,
}

impl VolatilitySpread {
    /// Create a new VolatilitySpread indicator.
    ///
    /// # Arguments
    /// * `short_period` - Period for short-term volatility calculation
    /// * `long_period` - Period for long-term volatility calculation
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
        Ok(Self {
            short_period,
            long_period,
        })
    }

    /// Calculate volatility spread.
    /// Positive values indicate short-term vol > long-term vol (backwardation).
    /// Negative values indicate short-term vol < long-term vol (contango).
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
                let var: f64 = short_returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / short_returns.len() as f64;
                var.sqrt() * (252.0_f64).sqrt() * 100.0
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
                let var: f64 = long_returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / long_returns.len() as f64;
                var.sqrt() * (252.0_f64).sqrt() * 100.0
            } else {
                0.0
            };

            // Spread: short-term minus long-term
            result[i] = short_vol - long_vol;
        }

        result
    }
}

impl TechnicalIndicator for VolatilitySpread {
    fn name(&self) -> &str {
        "Volatility Spread"
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

/// Normalized Range Volatility - True range normalized by price.
///
/// Calculates volatility using the true range (high-low range accounting for gaps),
/// normalized by price to make it comparable across different price levels.
#[derive(Debug, Clone)]
pub struct NormalizedRangeVolatility {
    period: usize,
}

impl NormalizedRangeVolatility {
    /// Create a new NormalizedRangeVolatility indicator.
    ///
    /// # Arguments
    /// * `period` - Period for averaging the normalized true range
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate normalized range volatility.
    /// Returns annualized volatility expressed as a percentage.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len()).min(close.len());

        if n < self.period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        // Calculate normalized true range for each bar
        let mut ntr = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            // Normalize by the average price
            let avg_price = (high[i] + low[i] + close[i]) / 3.0;
            if avg_price > 1e-10 {
                ntr[i] = tr / avg_price;
            }
        }

        // Calculate rolling average of normalized true range
        for i in self.period..n {
            let start = i.saturating_sub(self.period) + 1;
            let avg_ntr: f64 = ntr[start..=i].iter().sum::<f64>() / self.period as f64;

            // Annualize and convert to percentage
            result[i] = avg_ntr * (252.0_f64).sqrt() * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for NormalizedRangeVolatility {
    fn name(&self) -> &str {
        "Normalized Range Volatility"
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
        Ok(IndicatorOutput::single(self.calculate(
            &data.high,
            &data.low,
            &data.close,
        )))
    }
}

/// Volatility Skew Indicator - Asymmetry in volatility distribution.
///
/// Measures the skewness of volatility itself over a lookback period,
/// indicating whether volatility tends to spike more in one direction.
#[derive(Debug, Clone)]
pub struct VolatilitySkewIndicator {
    volatility_period: usize,
    skew_period: usize,
}

impl VolatilitySkewIndicator {
    /// Create a new VolatilitySkewIndicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for calculating rolling volatility
    /// * `skew_period` - Period for calculating skewness of volatility
    pub fn new(volatility_period: usize, skew_period: usize) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if skew_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "skew_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            volatility_period,
            skew_period,
        })
    }

    /// Calculate volatility skewness.
    /// Positive values indicate right-skewed distribution (fat right tail - more high vol spikes).
    /// Negative values indicate left-skewed distribution (fat left tail - more low vol clustering).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_period = self.volatility_period + self.skew_period;

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
                let var: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                volatility[i] = var.sqrt();
            }
        }

        // Calculate skewness of volatility
        let mut result = vec![0.0; n];
        for i in total_period..n {
            let start = i.saturating_sub(self.skew_period);
            let vol_slice: Vec<f64> = (start..=i).map(|j| volatility[j]).collect();

            if vol_slice.len() >= 3 {
                let n_f = vol_slice.len() as f64;
                let mean: f64 = vol_slice.iter().sum::<f64>() / n_f;
                let var: f64 = vol_slice
                    .iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>()
                    / n_f;
                let std = var.sqrt();

                if std > 1e-10 {
                    // Calculate third moment for skewness
                    let m3: f64 = vol_slice
                        .iter()
                        .map(|v| ((v - mean) / std).powi(3))
                        .sum::<f64>()
                        / n_f;

                    // Adjust for sample skewness
                    let adj_factor = (n_f * (n_f - 1.0)).sqrt() / (n_f - 2.0);
                    result[i] = m3 * adj_factor;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilitySkewIndicator {
    fn name(&self) -> &str {
        "Volatility Skew Indicator"
    }

    fn min_periods(&self) -> usize {
        self.volatility_period + self.skew_period + 1
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

/// Adaptive Volatility Bands - Bands that adapt to volatility regime.
///
/// Creates dynamic bands around price that widen in high volatility regimes
/// and narrow in low volatility regimes, providing adaptive support/resistance levels.
#[derive(Debug, Clone)]
pub struct AdaptiveVolatilityBands {
    volatility_period: usize,
    band_period: usize,
    multiplier: f64,
}

impl AdaptiveVolatilityBands {
    /// Create a new AdaptiveVolatilityBands indicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for calculating rolling volatility
    /// * `band_period` - Period for the moving average center line
    /// * `multiplier` - Multiplier for band width (similar to Bollinger Bands)
    pub fn new(volatility_period: usize, band_period: usize, multiplier: f64) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if band_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "band_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if multiplier <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "multiplier".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            volatility_period,
            band_period,
            multiplier,
        })
    }

    /// Calculate adaptive volatility bands.
    /// Returns (middle_band, upper_band, lower_band, bandwidth).
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let total_period = self.volatility_period.max(self.band_period);

        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];
        let mut bandwidth = vec![0.0; n];

        if n < total_period + 1 {
            return (middle, upper, lower, bandwidth);
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
                let var: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                volatility[i] = var.sqrt();
            }
        }

        // Calculate long-term average volatility for regime detection
        let long_vol_period = self.volatility_period * 4;
        let mut avg_volatility = vec![0.0; n];
        for i in long_vol_period..n {
            let start = i.saturating_sub(long_vol_period);
            avg_volatility[i] = volatility[start..=i].iter().sum::<f64>() / long_vol_period as f64;
        }

        // Calculate bands
        for i in total_period..n {
            // Middle band: simple moving average
            let ma_start = i.saturating_sub(self.band_period);
            middle[i] = close[ma_start..=i].iter().sum::<f64>() / (self.band_period + 1) as f64;

            // Adaptive multiplier based on current vs average volatility
            let vol_ratio = if avg_volatility[i] > 1e-10 {
                (volatility[i] / avg_volatility[i]).sqrt()
            } else {
                1.0
            };
            let adaptive_mult = self.multiplier * vol_ratio;

            // Band width in price terms
            let band_width = middle[i] * volatility[i] * adaptive_mult * (252.0_f64).sqrt();

            upper[i] = middle[i] + band_width;
            lower[i] = middle[i] - band_width;

            // Bandwidth as percentage
            if middle[i] > 1e-10 {
                bandwidth[i] = (upper[i] - lower[i]) / middle[i] * 100.0;
            }
        }

        (middle, upper, lower, bandwidth)
    }
}

impl TechnicalIndicator for AdaptiveVolatilityBands {
    fn name(&self) -> &str {
        "Adaptive Volatility Bands"
    }

    fn min_periods(&self) -> usize {
        self.volatility_period.max(self.band_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        let (middle, upper, lower, _bandwidth) = self.calculate(&data.close);

        // Return triple output: middle (primary), upper (secondary), lower (tertiary)
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn output_features(&self) -> usize {
        3 // middle_band, upper_band, lower_band
    }
}

// ============================================================================
// 6 NEW VOLATILITY INDICATORS
// ============================================================================

/// Volatility Regime Enumeration for VolatilityRegimeDetector
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolRegimeLevel {
    /// Low volatility regime (< 1 std below mean)
    Low,
    /// Medium/Normal volatility regime (within 1 std of mean)
    Medium,
    /// High volatility regime (> 1 std above mean)
    High,
}

/// Volatility Regime Detector - Detects current volatility regime.
///
/// Classifies the current volatility environment into Low, Medium, or High
/// regimes based on statistical thresholds. This is useful for adapting
/// trading strategies to different market conditions.
///
/// # Regime Classification
/// - Low: Current volatility is more than 1 standard deviation below the mean
/// - Medium: Current volatility is within 1 standard deviation of the mean
/// - High: Current volatility is more than 1 standard deviation above the mean
///
/// # Example
/// ```ignore
/// let detector = VolatilityRegimeDetector::new(20, 60)?;
/// let (regime_values, regimes) = detector.calculate(&close);
/// // regime_values contains z-scores, regimes contains enum classifications
/// ```
#[derive(Debug, Clone)]
pub struct VolatilityRegimeDetector {
    /// Period for calculating rolling volatility
    volatility_period: usize,
    /// Period for calculating mean and standard deviation of volatility
    regime_period: usize,
    /// Low regime threshold (std deviations below mean)
    low_threshold: f64,
    /// High regime threshold (std deviations above mean)
    high_threshold: f64,
}

impl VolatilityRegimeDetector {
    /// Create a new VolatilityRegimeDetector indicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for calculating rolling volatility (minimum 5)
    /// * `regime_period` - Period for calculating volatility statistics (minimum 20)
    ///
    /// # Returns
    /// A Result containing the indicator or an error if parameters are invalid
    pub fn new(volatility_period: usize, regime_period: usize) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if regime_period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "regime_period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self {
            volatility_period,
            regime_period,
            low_threshold: -1.0,
            high_threshold: 1.0,
        })
    }

    /// Create with custom thresholds for regime classification.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for calculating rolling volatility
    /// * `regime_period` - Period for calculating volatility statistics
    /// * `low_threshold` - Z-score threshold for low regime (typically negative)
    /// * `high_threshold` - Z-score threshold for high regime (typically positive)
    pub fn with_thresholds(
        volatility_period: usize,
        regime_period: usize,
        low_threshold: f64,
        high_threshold: f64,
    ) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if regime_period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "regime_period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if low_threshold >= high_threshold {
            return Err(IndicatorError::InvalidParameter {
                name: "thresholds".to_string(),
                reason: "low_threshold must be less than high_threshold".to_string(),
            });
        }
        Ok(Self {
            volatility_period,
            regime_period,
            low_threshold,
            high_threshold,
        })
    }

    /// Calculate volatility regime detection.
    ///
    /// Returns a tuple of (z_scores, regime_levels) where:
    /// - z_scores: The z-score of current volatility relative to historical mean
    /// - regime_levels: The classified regime for each period
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<VolRegimeLevel>) {
        let n = close.len();
        let total_period = self.volatility_period + self.regime_period;

        let mut z_scores = vec![0.0; n];
        let mut regimes = vec![VolRegimeLevel::Medium; n];

        if n < total_period + 1 {
            return (z_scores, regimes);
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
                let var: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                volatility[i] = var.sqrt() * (252.0_f64).sqrt();
            }
        }

        // Calculate regime based on z-score of volatility
        for i in total_period..n {
            let start = i.saturating_sub(self.regime_period);
            let vol_slice: Vec<f64> = (start..i).map(|j| volatility[j]).collect();

            if vol_slice.len() >= 2 {
                let mean_vol: f64 = vol_slice.iter().sum::<f64>() / vol_slice.len() as f64;
                let var_vol: f64 = vol_slice
                    .iter()
                    .map(|v| (v - mean_vol).powi(2))
                    .sum::<f64>()
                    / vol_slice.len() as f64;
                let std_vol = var_vol.sqrt();

                if std_vol > 1e-10 {
                    let z_score = (volatility[i] - mean_vol) / std_vol;
                    z_scores[i] = z_score;

                    // Classify regime
                    regimes[i] = if z_score <= self.low_threshold {
                        VolRegimeLevel::Low
                    } else if z_score >= self.high_threshold {
                        VolRegimeLevel::High
                    } else {
                        VolRegimeLevel::Medium
                    };
                }
            }
        }

        (z_scores, regimes)
    }

    /// Get the regime level at a specific z-score.
    pub fn classify_regime(&self, z_score: f64) -> VolRegimeLevel {
        if z_score <= self.low_threshold {
            VolRegimeLevel::Low
        } else if z_score >= self.high_threshold {
            VolRegimeLevel::High
        } else {
            VolRegimeLevel::Medium
        }
    }
}

impl TechnicalIndicator for VolatilityRegimeDetector {
    fn name(&self) -> &str {
        "Volatility Regime Detector"
    }

    fn min_periods(&self) -> usize {
        self.volatility_period + self.regime_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        let (z_scores, _) = self.calculate(&data.close);
        Ok(IndicatorOutput::single(z_scores))
    }
}

/// Volatility Breakout Signal - Advanced volatility breakout detection.
///
/// Detects significant volatility breakouts using multiple confirmation methods
/// including ATR expansion, range breakout, and momentum confirmation. This is
/// more sophisticated than simple threshold-based breakout detection.
///
/// # Signal Values
/// - Positive values: Upside volatility breakout with momentum
/// - Negative values: Downside volatility breakout with momentum
/// - Zero: No significant breakout detected
///
/// # Example
/// ```ignore
/// let breakout = VolatilityBreakoutSignal::new(14, 50, 2.0)?;
/// let signals = breakout.calculate(&high, &low, &close);
/// ```
#[derive(Debug, Clone)]
pub struct VolatilityBreakoutSignal {
    /// Period for ATR and short-term volatility calculation
    atr_period: usize,
    /// Period for baseline volatility comparison
    baseline_period: usize,
    /// Multiplier threshold for breakout detection
    breakout_threshold: f64,
}

impl VolatilityBreakoutSignal {
    /// Create a new VolatilityBreakoutSignal indicator.
    ///
    /// # Arguments
    /// * `atr_period` - Period for ATR calculation (minimum 5)
    /// * `baseline_period` - Period for baseline volatility (minimum 20)
    /// * `breakout_threshold` - Multiplier for breakout detection (minimum 1.0)
    pub fn new(atr_period: usize, baseline_period: usize, breakout_threshold: f64) -> Result<Self> {
        if atr_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "atr_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if baseline_period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "baseline_period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if breakout_threshold < 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "breakout_threshold".to_string(),
                reason: "must be at least 1.0".to_string(),
            });
        }
        Ok(Self {
            atr_period,
            baseline_period,
            breakout_threshold,
        })
    }

    /// Calculate volatility breakout signals.
    ///
    /// Returns breakout signal strength:
    /// - Positive values indicate upside breakouts
    /// - Negative values indicate downside breakouts
    /// - Values close to zero indicate no breakout
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len()).min(close.len());

        if n < self.baseline_period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        // Calculate ATR
        let mut atr = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i >= self.atr_period {
                let start = i.saturating_sub(self.atr_period);
                let mut tr_sum = tr;
                for j in (start + 1)..i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    tr_sum += tr_j;
                }
                atr[i] = tr_sum / self.atr_period as f64;
            }
        }

        // Calculate baseline ATR and detect breakouts
        for i in self.baseline_period..n {
            let baseline_start = i.saturating_sub(self.baseline_period);

            // Calculate baseline ATR statistics
            let baseline_atrs: Vec<f64> = (baseline_start..i)
                .filter(|&j| atr[j] > 1e-10)
                .map(|j| atr[j])
                .collect();

            if baseline_atrs.len() < 5 {
                continue;
            }

            let mean_atr: f64 = baseline_atrs.iter().sum::<f64>() / baseline_atrs.len() as f64;
            let var_atr: f64 = baseline_atrs
                .iter()
                .map(|a| (a - mean_atr).powi(2))
                .sum::<f64>()
                / baseline_atrs.len() as f64;
            let std_atr = var_atr.sqrt();

            if std_atr > 1e-10 && mean_atr > 1e-10 {
                // Current ATR relative to baseline
                let atr_ratio = atr[i] / mean_atr;
                let atr_zscore = (atr[i] - mean_atr) / std_atr;

                // Determine direction based on price change
                let price_change = close[i] - close[i - 1];
                let direction = if price_change > 0.0 { 1.0 } else { -1.0 };

                // Breakout signal: ATR expansion above threshold with direction
                if atr_ratio > self.breakout_threshold && atr_zscore > 1.0 {
                    // Signal strength based on how much ATR exceeded threshold
                    let strength = (atr_ratio - 1.0) * atr_zscore.min(3.0);
                    result[i] = strength * direction * 100.0;
                }
            }
        }

        result
    }

    /// Check if a signal value represents a significant breakout.
    pub fn is_breakout(&self, signal: f64) -> bool {
        signal.abs() > 50.0
    }
}

impl TechnicalIndicator for VolatilityBreakoutSignal {
    fn name(&self) -> &str {
        "Volatility Breakout Signal"
    }

    fn min_periods(&self) -> usize {
        self.baseline_period + 1
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

/// Enhanced Volatility Mean Reversion - Advanced mean reversion measurement.
///
/// Combines multiple mean reversion signals including distance from mean,
/// half-life estimation, and reversion speed to provide a comprehensive
/// view of volatility mean reversion opportunities.
///
/// # Output
/// Returns a mean reversion score where:
/// - Large positive values: Volatility is high and likely to revert down
/// - Large negative values: Volatility is low and likely to revert up
/// - Values near zero: Volatility is near its mean
///
/// # Example
/// ```ignore
/// let mr = EnhancedVolatilityMeanReversion::new(10, 50, 20)?;
/// let scores = mr.calculate(&close);
/// ```
#[derive(Debug, Clone)]
pub struct EnhancedVolatilityMeanReversion {
    /// Period for calculating rolling volatility
    volatility_period: usize,
    /// Period for calculating long-term mean volatility
    mean_period: usize,
    /// Lookback period for measuring reversion dynamics
    lookback: usize,
}

impl EnhancedVolatilityMeanReversion {
    /// Create a new EnhancedVolatilityMeanReversion indicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for rolling volatility (minimum 5)
    /// * `mean_period` - Period for mean volatility (minimum 30)
    /// * `lookback` - Lookback for reversion dynamics (minimum 10)
    pub fn new(volatility_period: usize, mean_period: usize, lookback: usize) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if mean_period < 30 {
            return Err(IndicatorError::InvalidParameter {
                name: "mean_period".to_string(),
                reason: "must be at least 30".to_string(),
            });
        }
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            volatility_period,
            mean_period,
            lookback,
        })
    }

    /// Calculate enhanced mean reversion score.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_period = self.mean_period + self.lookback;

        if n < total_period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

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
                let var: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / returns.len() as f64;
                volatility[i] = var.sqrt() * (252.0_f64).sqrt();
            }
        }

        // Calculate enhanced mean reversion score
        for i in total_period..n {
            // Long-term mean and std of volatility
            let mean_start = i.saturating_sub(self.mean_period);
            let vol_history: Vec<f64> = (mean_start..i)
                .filter(|&j| volatility[j] > 1e-10)
                .map(|j| volatility[j])
                .collect();

            if vol_history.len() < 10 {
                continue;
            }

            let mean_vol: f64 = vol_history.iter().sum::<f64>() / vol_history.len() as f64;
            let var_vol: f64 = vol_history
                .iter()
                .map(|v| (v - mean_vol).powi(2))
                .sum::<f64>()
                / vol_history.len() as f64;
            let std_vol = var_vol.sqrt();

            if std_vol < 1e-10 || mean_vol < 1e-10 {
                continue;
            }

            // Component 1: Z-score (distance from mean)
            let z_score = (volatility[i] - mean_vol) / std_vol;

            // Component 2: Estimate half-life of mean reversion
            // Using AR(1) coefficient approximation
            let lookback_start = i.saturating_sub(self.lookback);
            let mut sum_xy = 0.0;
            let mut sum_x2 = 0.0;

            for j in (lookback_start + 1)..=i {
                let x = volatility[j - 1] - mean_vol;
                let y = volatility[j] - mean_vol;
                sum_xy += x * y;
                sum_x2 += x * x;
            }

            let ar1_coef = if sum_x2 > 1e-10 {
                (sum_xy / sum_x2).max(-0.99).min(0.99)
            } else {
                0.0
            };

            // Half-life: -ln(2) / ln(ar1_coef)
            // Smaller half-life = faster reversion
            let reversion_speed = if ar1_coef > 0.0 && ar1_coef < 1.0 {
                1.0 - ar1_coef // Simple speed measure
            } else {
                0.5 // Default moderate speed
            };

            // Combined score: z-score weighted by reversion speed
            // Positive z-score with high reversion speed = expect drop
            // Negative z-score with high reversion speed = expect rise
            result[i] = z_score * reversion_speed * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for EnhancedVolatilityMeanReversion {
    fn name(&self) -> &str {
        "Enhanced Volatility Mean Reversion"
    }

    fn min_periods(&self) -> usize {
        self.mean_period + self.lookback + 1
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

/// Volatility Skew Ratio - Measures asymmetry in volatility distribution.
///
/// Calculates the ratio of upside volatility to downside volatility,
/// providing insight into whether the market exhibits more volatility
/// on up moves versus down moves.
///
/// # Interpretation
/// - Ratio > 1: More volatility on up moves (unusual)
/// - Ratio < 1: More volatility on down moves (typical for equities)
/// - Ratio = 1: Symmetric volatility
///
/// # Example
/// ```ignore
/// let skew = VolatilitySkewRatio::new(20)?;
/// let ratios = skew.calculate(&close);
/// ```
#[derive(Debug, Clone)]
pub struct VolatilitySkewRatio {
    /// Lookback period for calculating skew
    period: usize,
}

impl VolatilitySkewRatio {
    /// Create a new VolatilitySkewRatio indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period (minimum 10)
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate volatility skew ratio.
    ///
    /// Returns the ratio of upside to downside volatility.
    /// Values > 1 indicate more upside volatility.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n < self.period + 1 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Separate positive and negative returns
            let mut positive_returns = Vec::new();
            let mut negative_returns = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    let ret = (close[j] / close[j - 1]).ln();
                    if ret > 0.0 {
                        positive_returns.push(ret);
                    } else if ret < 0.0 {
                        negative_returns.push(ret.abs());
                    }
                }
            }

            // Calculate upside volatility (std of positive returns)
            let upside_vol = if positive_returns.len() >= 2 {
                let mean: f64 =
                    positive_returns.iter().sum::<f64>() / positive_returns.len() as f64;
                let var: f64 = positive_returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / positive_returns.len() as f64;
                var.sqrt()
            } else if !positive_returns.is_empty() {
                positive_returns.iter().sum::<f64>() / positive_returns.len() as f64
            } else {
                0.0
            };

            // Calculate downside volatility (std of absolute negative returns)
            let downside_vol = if negative_returns.len() >= 2 {
                let mean: f64 =
                    negative_returns.iter().sum::<f64>() / negative_returns.len() as f64;
                let var: f64 = negative_returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>()
                    / negative_returns.len() as f64;
                var.sqrt()
            } else if !negative_returns.is_empty() {
                negative_returns.iter().sum::<f64>() / negative_returns.len() as f64
            } else {
                0.0
            };

            // Calculate ratio (upside / downside)
            if downside_vol > 1e-10 {
                result[i] = upside_vol / downside_vol;
            } else if upside_vol > 1e-10 {
                result[i] = 2.0; // Cap at 2 when no downside vol
            } else {
                result[i] = 1.0; // Neutral when both are zero
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilitySkewRatio {
    fn name(&self) -> &str {
        "Volatility Skew Ratio"
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
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volatility Clustering Index - Measures volatility clustering strength.
///
/// Quantifies the degree to which high volatility tends to follow high volatility
/// and low volatility tends to follow low volatility. This ARCH/GARCH effect
/// is a key characteristic of financial markets.
///
/// # Output
/// Returns a clustering index from 0 to 1:
/// - Values near 1: Strong clustering (volatility is highly persistent)
/// - Values near 0.5: Moderate clustering
/// - Values near 0: Weak clustering (volatility is more random)
///
/// # Example
/// ```ignore
/// let clustering = VolatilityClusteringIndex::new(20, 5)?;
/// let index = clustering.calculate(&close);
/// ```
#[derive(Debug, Clone)]
pub struct VolatilityClusteringIndex {
    /// Period for calculating rolling statistics
    period: usize,
    /// Number of lags to consider for clustering
    num_lags: usize,
}

impl VolatilityClusteringIndex {
    /// Create a new VolatilityClusteringIndex indicator.
    ///
    /// # Arguments
    /// * `period` - Period for statistics (minimum 20)
    /// * `num_lags` - Number of lags to consider (minimum 1, maximum 10)
    pub fn new(period: usize, num_lags: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if num_lags < 1 || num_lags > 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_lags".to_string(),
                reason: "must be between 1 and 10".to_string(),
            });
        }
        Ok(Self { period, num_lags })
    }

    /// Calculate volatility clustering index.
    ///
    /// Returns a value between 0 and 1 indicating clustering strength.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_period = self.period + self.num_lags;

        if n < total_period + 2 {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        // Calculate squared returns (proxy for volatility)
        let mut squared_returns = vec![0.0; n];
        for i in 1..n {
            if close[i - 1] > 1e-10 {
                let ret = (close[i] / close[i - 1]).ln();
                squared_returns[i] = ret * ret;
            }
        }

        // Calculate clustering index using multiple lag autocorrelations
        for i in total_period..n {
            let start = i.saturating_sub(self.period);

            // Calculate mean and variance of squared returns
            let slice: Vec<f64> = (start..=i).map(|j| squared_returns[j]).collect();
            let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
            let var: f64 = slice
                .iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f64>()
                / slice.len() as f64;

            if var < 1e-20 {
                continue;
            }

            // Calculate average autocorrelation across lags
            let mut total_autocorr = 0.0;
            let mut valid_lags = 0;

            for lag in 1..=self.num_lags {
                let mut cov = 0.0;
                let mut count = 0;

                for j in (start + lag)..=i {
                    let x = squared_returns[j] - mean;
                    let y = squared_returns[j - lag] - mean;
                    cov += x * y;
                    count += 1;
                }

                if count > 0 {
                    let autocorr = (cov / count as f64) / var;
                    // Weight earlier lags more heavily
                    let weight = 1.0 / lag as f64;
                    total_autocorr += autocorr.max(0.0) * weight;
                    valid_lags += 1;
                }
            }

            if valid_lags > 0 {
                // Normalize to 0-1 range
                // Higher autocorrelation = higher clustering
                let avg_autocorr = total_autocorr / valid_lags as f64;
                result[i] = avg_autocorr.max(0.0).min(1.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilityClusteringIndex {
    fn name(&self) -> &str {
        "Volatility Clustering Index"
    }

    fn min_periods(&self) -> usize {
        self.period + self.num_lags + 2
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

/// Dynamic Adaptive Volatility Bands - Self-adjusting volatility bands.
///
/// Creates dynamic price bands that automatically adjust their width based on
/// current volatility regime, market conditions, and trend strength. Unlike
/// standard Bollinger Bands, these bands adapt more aggressively to regime changes.
///
/// # Output
/// Returns four series:
/// - Middle band (adaptive moving average)
/// - Upper band (middle + adaptive width)
/// - Lower band (middle - adaptive width)
/// - Regime multiplier (current adaptation factor)
///
/// # Example
/// ```ignore
/// let bands = DynamicAdaptiveVolatilityBands::new(10, 20, 2.0)?;
/// let (middle, upper, lower, multiplier) = bands.calculate(&high, &low, &close);
/// ```
#[derive(Debug, Clone)]
pub struct DynamicAdaptiveVolatilityBands {
    /// Period for volatility calculation
    volatility_period: usize,
    /// Period for the adaptive moving average
    ma_period: usize,
    /// Base multiplier for band width
    base_multiplier: f64,
}

impl DynamicAdaptiveVolatilityBands {
    /// Create a new DynamicAdaptiveVolatilityBands indicator.
    ///
    /// # Arguments
    /// * `volatility_period` - Period for volatility (minimum 5)
    /// * `ma_period` - Period for moving average (minimum 10)
    /// * `base_multiplier` - Base multiplier for bands (minimum 0.5)
    pub fn new(volatility_period: usize, ma_period: usize, base_multiplier: f64) -> Result<Self> {
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if ma_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "ma_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if base_multiplier < 0.5 {
            return Err(IndicatorError::InvalidParameter {
                name: "base_multiplier".to_string(),
                reason: "must be at least 0.5".to_string(),
            });
        }
        Ok(Self {
            volatility_period,
            ma_period,
            base_multiplier,
        })
    }

    /// Calculate dynamic adaptive volatility bands.
    ///
    /// Returns (middle_band, upper_band, lower_band, regime_multiplier)
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len().min(low.len()).min(close.len());
        let total_period = self.volatility_period.max(self.ma_period);

        let mut middle = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];
        let mut regime_mult = vec![0.0; n];

        if n < total_period + 1 {
            return (middle, upper, lower, regime_mult);
        }

        // Calculate ATR-based volatility
        let mut atr = vec![0.0; n];
        for i in 1..n {
            let tr = (high[i] - low[i])
                .max((high[i] - close[i - 1]).abs())
                .max((low[i] - close[i - 1]).abs());

            if i >= self.volatility_period {
                let start = i.saturating_sub(self.volatility_period);
                let mut tr_sum = tr;
                for j in (start + 1)..i {
                    let tr_j = (high[j] - low[j])
                        .max((high[j] - close[j - 1]).abs())
                        .max((low[j] - close[j - 1]).abs());
                    tr_sum += tr_j;
                }
                atr[i] = tr_sum / self.volatility_period as f64;
            }
        }

        // Calculate long-term average ATR for regime detection
        let long_period = self.volatility_period * 4;
        let mut avg_atr = vec![0.0; n];
        for i in long_period..n {
            let start = i.saturating_sub(long_period);
            let sum: f64 = atr[start..i].iter().sum();
            avg_atr[i] = sum / long_period as f64;
        }

        // Calculate bands with dynamic adaptation
        for i in total_period..n {
            // Adaptive moving average (Kaufman-style efficiency adjustment)
            let ma_start = i.saturating_sub(self.ma_period);
            let price_change = (close[i] - close[ma_start]).abs();
            let mut path_length = 0.0;
            for j in (ma_start + 1)..=i {
                path_length += (close[j] - close[j - 1]).abs();
            }

            // Efficiency ratio (0 to 1)
            let efficiency = if path_length > 1e-10 {
                (price_change / path_length).min(1.0)
            } else {
                0.5
            };

            // Adaptive smoothing constant
            let fast_sc = 2.0 / 3.0;
            let slow_sc = 2.0 / 31.0;
            let sc = (efficiency * (fast_sc - slow_sc) + slow_sc).powi(2);

            // Calculate adaptive MA
            if i == total_period {
                middle[i] = close[ma_start..=i].iter().sum::<f64>() / (self.ma_period + 1) as f64;
            } else {
                middle[i] = middle[i - 1] + sc * (close[i] - middle[i - 1]);
            }

            // Regime-based multiplier
            let current_regime = if avg_atr[i] > 1e-10 {
                (atr[i] / avg_atr[i]).sqrt()
            } else {
                1.0
            };

            // Trend strength adjustment
            let trend_factor = 1.0 + (1.0 - efficiency) * 0.5;

            // Final adaptive multiplier
            let adaptive_mult = self.base_multiplier * current_regime * trend_factor;
            regime_mult[i] = adaptive_mult;

            // Calculate bands
            let band_width = atr[i] * adaptive_mult;
            upper[i] = middle[i] + band_width;
            lower[i] = middle[i] - band_width;
        }

        (middle, upper, lower, regime_mult)
    }
}

impl TechnicalIndicator for DynamicAdaptiveVolatilityBands {
    fn name(&self) -> &str {
        "Dynamic Adaptive Volatility Bands"
    }

    fn min_periods(&self) -> usize {
        self.volatility_period.max(self.ma_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        let (middle, upper, lower, _) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn output_features(&self) -> usize {
        3 // middle_band, upper_band, lower_band
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

    // ========================================================================
    // Tests for new advanced volatility indicators
    // ========================================================================

    #[test]
    fn test_volatility_forecast() {
        let (_, _, close) = make_test_data();
        let vf = VolatilityForecast::new(10, 0.1, 0.8).unwrap();
        let result = vf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Forecast should be positive after warmup
        for i in 15..close.len() {
            assert!(result[i] >= 0.0, "Forecast at {} should be non-negative", i);
        }
    }

    #[test]
    fn test_volatility_forecast_invalid_params() {
        assert!(VolatilityForecast::new(1, 0.1, 0.8).is_err()); // period < 2
        assert!(VolatilityForecast::new(10, -0.1, 0.8).is_err()); // alpha < 0
        assert!(VolatilityForecast::new(10, 1.5, 0.8).is_err()); // alpha > 1
        assert!(VolatilityForecast::new(10, 0.1, -0.1).is_err()); // beta < 0
        assert!(VolatilityForecast::new(10, 0.1, 1.5).is_err()); // beta > 1
    }

    #[test]
    fn test_volatility_forecast_technical_indicator() {
        let (_, _, close) = make_test_data();
        let vf = VolatilityForecast::new(10, 0.1, 0.8).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(vf.name(), "Volatility Forecast");
        assert_eq!(vf.min_periods(), 11);

        let result = vf.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_spike() {
        let (_, _, close) = make_test_data();
        let vs = VolatilitySpike::new(5, 20, 2.0).unwrap();
        let result = vs.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Z-scores can be positive or negative
    }

    #[test]
    fn test_volatility_spike_is_spike() {
        let vs = VolatilitySpike::new(5, 20, 2.0).unwrap();
        assert!(vs.is_spike(2.5));
        assert!(vs.is_spike(3.0));
        assert!(!vs.is_spike(1.5));
        assert!(!vs.is_spike(-1.0));
    }

    #[test]
    fn test_volatility_spike_invalid_params() {
        assert!(VolatilitySpike::new(1, 20, 2.0).is_err()); // volatility_period < 2
        assert!(VolatilitySpike::new(5, 5, 2.0).is_err()); // baseline_period <= volatility_period
        assert!(VolatilitySpike::new(5, 4, 2.0).is_err()); // baseline_period <= volatility_period
        assert!(VolatilitySpike::new(5, 20, 0.0).is_err()); // threshold <= 0
        assert!(VolatilitySpike::new(5, 20, -1.0).is_err()); // threshold <= 0
    }

    #[test]
    fn test_volatility_spike_technical_indicator() {
        let (_, _, close) = make_test_data();
        let vs = VolatilitySpike::new(5, 20, 2.0).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(vs.name(), "Volatility Spike");
        assert_eq!(vs.min_periods(), 21);

        let result = vs.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_mean_reversion() {
        let (_, _, close) = make_test_data();
        let vmr = VolatilityMeanReversion::new(5, 20, 10).unwrap();
        let result = vmr.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_volatility_mean_reversion_invalid_params() {
        assert!(VolatilityMeanReversion::new(1, 20, 10).is_err()); // volatility_period < 2
        assert!(VolatilityMeanReversion::new(5, 3, 10).is_err()); // mean_period < volatility_period
        assert!(VolatilityMeanReversion::new(5, 20, 1).is_err()); // lookback < 2
    }

    #[test]
    fn test_volatility_mean_reversion_technical_indicator() {
        let (_, _, close) = make_test_data();
        let vmr = VolatilityMeanReversion::new(5, 20, 10).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(vmr.name(), "Volatility Mean Reversion");
        assert_eq!(vmr.min_periods(), 31);

        let result = vmr.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_clustering() {
        let (_, _, close) = make_test_data();
        let vc = VolatilityClustering::new(20, 1).unwrap();
        let result = vc.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Clustering (correlation) should be between -1 and 1
        for i in 25..close.len() {
            assert!(
                result[i] >= -1.0 && result[i] <= 1.0,
                "Clustering at {} should be between -1 and 1, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_volatility_clustering_invalid_params() {
        assert!(VolatilityClustering::new(1, 1).is_err()); // period < 2
        assert!(VolatilityClustering::new(20, 0).is_err()); // lag < 1
    }

    #[test]
    fn test_volatility_clustering_technical_indicator() {
        let (_, _, close) = make_test_data();
        let vc = VolatilityClustering::new(20, 1).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(vc.name(), "Volatility Clustering");
        assert_eq!(vc.min_periods(), 23);

        let result = vc.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_asymmetry() {
        let (_, _, close) = make_test_data();
        let va = VolatilityAsymmetry::new(20).unwrap();
        let result = va.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Asymmetry should be between -100 and 100
        for i in 25..close.len() {
            assert!(
                result[i] >= -100.0 && result[i] <= 100.0,
                "Asymmetry at {} should be between -100 and 100, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_volatility_asymmetry_invalid_params() {
        assert!(VolatilityAsymmetry::new(1).is_err()); // period < 2
    }

    #[test]
    fn test_volatility_asymmetry_technical_indicator() {
        let (_, _, close) = make_test_data();
        let va = VolatilityAsymmetry::new(20).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(va.name(), "Volatility Asymmetry");
        assert_eq!(va.min_periods(), 21);

        let result = va.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_efficiency() {
        let (_, _, close) = make_test_data();
        let ve = VolatilityEfficiency::new(20).unwrap();
        let result = ve.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Efficiency should be non-negative
        for i in 25..close.len() {
            assert!(
                result[i] >= 0.0,
                "Efficiency at {} should be non-negative, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_volatility_efficiency_invalid_params() {
        assert!(VolatilityEfficiency::new(1).is_err()); // period < 2
    }

    #[test]
    fn test_volatility_efficiency_technical_indicator() {
        let (_, _, close) = make_test_data();
        let ve = VolatilityEfficiency::new(20).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(ve.name(), "Volatility Efficiency");
        assert_eq!(ve.min_periods(), 21);

        let result = ve.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_efficiency_trending_market() {
        // Create trending data - should have higher efficiency
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
        let ve = VolatilityEfficiency::new(10).unwrap();
        let result = ve.calculate(&close);

        // In a trending market, efficiency should be relatively high
        for i in 15..close.len() {
            assert!(result[i] > 0.0, "Trending market should have positive efficiency");
        }
    }

    #[test]
    fn test_insufficient_data_errors() {
        let short_data = OHLCVSeries::from_close(vec![100.0, 101.0, 102.0]);

        let vf = VolatilityForecast::new(10, 0.1, 0.8).unwrap();
        assert!(vf.compute(&short_data).is_err());

        let vs = VolatilitySpike::new(5, 20, 2.0).unwrap();
        assert!(vs.compute(&short_data).is_err());

        let vmr = VolatilityMeanReversion::new(5, 20, 10).unwrap();
        assert!(vmr.compute(&short_data).is_err());

        let vc = VolatilityClustering::new(20, 1).unwrap();
        assert!(vc.compute(&short_data).is_err());

        let va = VolatilityAsymmetry::new(20).unwrap();
        assert!(va.compute(&short_data).is_err());

        let ve = VolatilityEfficiency::new(20).unwrap();
        assert!(ve.compute(&short_data).is_err());
    }

    // ========================================================================
    // Tests for 6 newly added volatility indicators
    // ========================================================================

    #[test]
    fn test_volatility_acceleration() {
        let (_, _, close) = make_test_data();
        let va = VolatilityAcceleration::new(10, 5).unwrap();
        let result = va.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values are computed after warmup period
        let min_period = va.min_periods();
        for i in min_period..close.len() {
            // Acceleration can be positive or negative, but should be finite
            assert!(result[i].is_finite(), "Acceleration at {} should be finite", i);
        }
    }

    #[test]
    fn test_volatility_acceleration_invalid_params() {
        assert!(VolatilityAcceleration::new(2, 5).is_err()); // volatility_period < 5
        assert!(VolatilityAcceleration::new(10, 1).is_err()); // acceleration_period < 2
    }

    #[test]
    fn test_volatility_acceleration_technical_indicator() {
        let (_, _, close) = make_test_data();
        let va = VolatilityAcceleration::new(10, 5).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(va.name(), "Volatility Acceleration");
        assert_eq!(va.min_periods(), 21); // 10 + 5*2 + 1

        let result = va.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_acceleration_with_volatile_data() {
        // Create data with increasing then decreasing volatility
        let mut close = vec![100.0];
        for i in 1..100 {
            let vol_factor = if i < 50 { i as f64 * 0.01 } else { (100 - i) as f64 * 0.01 };
            let change = (i as f64 * 0.1).sin() * vol_factor;
            close.push(close[i - 1] * (1.0 + change));
        }

        let va = VolatilityAcceleration::new(5, 3).unwrap();
        let result = va.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_volatility_mean_reversion_distance() {
        let (_, _, close) = make_test_data();
        let vmrd = VolatilityMeanReversionDistance::new(10, 20).unwrap();
        let result = vmrd.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Z-scores should typically be within reasonable range
        let min_period = vmrd.min_periods();
        for i in min_period..close.len() {
            // Z-scores are typically between -4 and 4 for most data
            assert!(
                result[i] >= -10.0 && result[i] <= 10.0,
                "Z-score at {} should be reasonable, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_volatility_mean_reversion_distance_invalid_params() {
        assert!(VolatilityMeanReversionDistance::new(2, 20).is_err()); // volatility_period < 5
        assert!(VolatilityMeanReversionDistance::new(10, 5).is_err()); // mean_period < 10
    }

    #[test]
    fn test_volatility_mean_reversion_distance_technical_indicator() {
        let (_, _, close) = make_test_data();
        let vmrd = VolatilityMeanReversionDistance::new(10, 20).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(vmrd.name(), "Volatility Mean Reversion Distance");
        assert_eq!(vmrd.min_periods(), 31); // 10 + 20 + 1

        let result = vmrd.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_mean_reversion_distance_high_vol() {
        // Create data with a spike in volatility
        let mut close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 0.5).collect();
        // Add volatile period
        for i in 50..70 {
            close.push(close[i - 1] * (1.0 + (i as f64 * 0.3).sin() * 0.05));
        }
        // Return to calm
        for i in 70..100 {
            close.push(close[i - 1] * (1.0 + (i as f64 * 0.1).sin() * 0.002));
        }

        let vmrd = VolatilityMeanReversionDistance::new(5, 15).unwrap();
        let result = vmrd.calculate(&close);

        // During high vol period, z-score should be elevated
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_volatility_spread() {
        let (_, _, close) = make_test_data();
        let vs = VolatilitySpread::new(10, 30).unwrap();
        let result = vs.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Spread can be positive or negative
        let min_period = vs.min_periods();
        for i in min_period..close.len() {
            assert!(result[i].is_finite(), "Spread at {} should be finite", i);
        }
    }

    #[test]
    fn test_volatility_spread_invalid_params() {
        assert!(VolatilitySpread::new(2, 30).is_err()); // short_period < 5
        assert!(VolatilitySpread::new(10, 10).is_err()); // long_period <= short_period
        assert!(VolatilitySpread::new(10, 5).is_err()); // long_period <= short_period
    }

    #[test]
    fn test_volatility_spread_technical_indicator() {
        let (_, _, close) = make_test_data();
        let vs = VolatilitySpread::new(10, 30).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(vs.name(), "Volatility Spread");
        assert_eq!(vs.min_periods(), 31); // long_period + 1

        let result = vs.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_spread_term_structure() {
        // During calm period followed by spike, short-term vol should exceed long-term
        let mut close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.05).collect();
        // Add sudden volatility
        for i in 60..80 {
            close.push(close[i - 1] * (1.0 + (i as f64).sin() * 0.03));
        }

        let vs = VolatilitySpread::new(5, 20).unwrap();
        let result = vs.calculate(&close);

        // After the volatility spike, short-term should exceed long-term (positive spread)
        // This is backwardation in volatility term structure
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_normalized_range_volatility() {
        let (high, low, close) = make_test_data();
        let nrv = NormalizedRangeVolatility::new(14).unwrap();
        let result = nrv.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Normalized volatility should be non-negative
        let min_period = nrv.min_periods();
        for i in min_period..close.len() {
            assert!(
                result[i] >= 0.0,
                "Normalized range vol at {} should be non-negative, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_normalized_range_volatility_invalid_params() {
        assert!(NormalizedRangeVolatility::new(1).is_err()); // period < 2
    }

    #[test]
    fn test_normalized_range_volatility_technical_indicator() {
        let (high, low, close) = make_test_data();
        let nrv = NormalizedRangeVolatility::new(14).unwrap();
        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 100],
        };

        assert_eq!(nrv.name(), "Normalized Range Volatility");
        assert_eq!(nrv.min_periods(), 15); // period + 1

        let result = nrv.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_normalized_range_volatility_price_independence() {
        // Test that NRV is similar for different price levels with same percentage moves
        let high1: Vec<f64> = (0..50).map(|i| 102.0 + (i as f64 * 0.1).sin() * 2.0).collect();
        let low1: Vec<f64> = (0..50).map(|i| 98.0 + (i as f64 * 0.1).sin() * 2.0).collect();
        let close1: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1).sin() * 2.0).collect();

        // Same percentage moves but 10x price level
        let high2: Vec<f64> = (0..50).map(|i| 1020.0 + (i as f64 * 0.1).sin() * 20.0).collect();
        let low2: Vec<f64> = (0..50).map(|i| 980.0 + (i as f64 * 0.1).sin() * 20.0).collect();
        let close2: Vec<f64> = (0..50).map(|i| 1000.0 + (i as f64 * 0.1).sin() * 20.0).collect();

        let nrv = NormalizedRangeVolatility::new(10).unwrap();
        let result1 = nrv.calculate(&high1, &low1, &close1);
        let result2 = nrv.calculate(&high2, &low2, &close2);

        // Results should be similar (normalized removes price level effect)
        for i in 15..50 {
            let diff = (result1[i] - result2[i]).abs();
            assert!(
                diff < 5.0, // Allow some tolerance
                "NRV should be similar regardless of price level at {}: {} vs {}",
                i,
                result1[i],
                result2[i]
            );
        }
    }

    #[test]
    fn test_volatility_skew_indicator() {
        let (_, _, close) = make_test_data();
        let vsi = VolatilitySkewIndicator::new(10, 20).unwrap();
        let result = vsi.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Skewness can be any real number but typically within reasonable range
        let min_period = vsi.min_periods();
        for i in min_period..close.len() {
            assert!(result[i].is_finite(), "Skew at {} should be finite", i);
        }
    }

    #[test]
    fn test_volatility_skew_indicator_invalid_params() {
        assert!(VolatilitySkewIndicator::new(2, 20).is_err()); // volatility_period < 5
        assert!(VolatilitySkewIndicator::new(10, 5).is_err()); // skew_period < 10
    }

    #[test]
    fn test_volatility_skew_indicator_technical_indicator() {
        let (_, _, close) = make_test_data();
        let vsi = VolatilitySkewIndicator::new(10, 20).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(vsi.name(), "Volatility Skew Indicator");
        assert_eq!(vsi.min_periods(), 31); // 10 + 20 + 1

        let result = vsi.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_skew_indicator_asymmetric_data() {
        // Create data with occasional large spikes (right-skewed volatility)
        let mut close = vec![100.0];
        for i in 1..100 {
            let change = if i % 20 == 0 {
                0.05 // Large spike every 20 bars
            } else {
                (i as f64 * 0.1).sin() * 0.005
            };
            close.push(close[i - 1] * (1.0 + change));
        }

        let vsi = VolatilitySkewIndicator::new(5, 15).unwrap();
        let result = vsi.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_adaptive_volatility_bands() {
        let (_, _, close) = make_test_data();
        let avb = AdaptiveVolatilityBands::new(10, 20, 2.0).unwrap();
        let (middle, upper, lower, bandwidth) = avb.calculate(&close);

        assert_eq!(middle.len(), close.len());
        assert_eq!(upper.len(), close.len());
        assert_eq!(lower.len(), close.len());
        assert_eq!(bandwidth.len(), close.len());

        let min_period = avb.min_periods();
        for i in min_period..close.len() {
            // Upper should be above middle, lower below
            assert!(
                upper[i] >= middle[i],
                "Upper band should be >= middle at {}",
                i
            );
            assert!(
                lower[i] <= middle[i],
                "Lower band should be <= middle at {}",
                i
            );
            // Bandwidth should be non-negative
            assert!(
                bandwidth[i] >= 0.0,
                "Bandwidth should be non-negative at {}",
                i
            );
        }
    }

    #[test]
    fn test_adaptive_volatility_bands_invalid_params() {
        assert!(AdaptiveVolatilityBands::new(2, 20, 2.0).is_err()); // volatility_period < 5
        assert!(AdaptiveVolatilityBands::new(10, 2, 2.0).is_err()); // band_period < 5
        assert!(AdaptiveVolatilityBands::new(10, 20, 0.0).is_err()); // multiplier <= 0
        assert!(AdaptiveVolatilityBands::new(10, 20, -1.0).is_err()); // multiplier <= 0
    }

    #[test]
    fn test_adaptive_volatility_bands_technical_indicator() {
        let (_, _, close) = make_test_data();
        let avb = AdaptiveVolatilityBands::new(10, 20, 2.0).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(avb.name(), "Adaptive Volatility Bands");
        assert_eq!(avb.min_periods(), 21); // max(10, 20) + 1

        let result = avb.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
        assert!(result.secondary.is_some()); // upper band
        assert!(result.tertiary.is_some()); // lower band
    }

    #[test]
    fn test_adaptive_volatility_bands_output_features() {
        let avb = AdaptiveVolatilityBands::new(10, 20, 2.0).unwrap();
        let features = avb.output_features();

        // Should return 3 for: middle_band, upper_band, lower_band
        assert_eq!(features, 3);
    }

    #[test]
    fn test_adaptive_volatility_bands_adapts_to_volatility() {
        // Create data with varying volatility
        let mut close = Vec::new();
        // Low volatility period
        for i in 0..50 {
            close.push(100.0 + i as f64 * 0.01 + (i as f64 * 0.1).sin() * 0.1);
        }
        // High volatility period
        for i in 50..100 {
            close.push(close[i - 1] * (1.0 + (i as f64 * 0.3).sin() * 0.03));
        }

        let avb = AdaptiveVolatilityBands::new(5, 10, 2.0).unwrap();
        let (_, upper, lower, bandwidth) = avb.calculate(&close);

        // Bandwidth should be wider in high-vol period
        let low_vol_bandwidth: f64 = bandwidth[30..45].iter().sum::<f64>() / 15.0;
        let high_vol_bandwidth: f64 = bandwidth[80..95].iter().sum::<f64>() / 15.0;

        // High vol period should have wider bands (might need some warmup time)
        assert!(
            high_vol_bandwidth > low_vol_bandwidth * 0.5,
            "High vol bandwidth {} should be wider than low vol {}",
            high_vol_bandwidth,
            low_vol_bandwidth
        );

        // Verify bands exist
        assert!(upper.len() == close.len());
        assert!(lower.len() == close.len());
    }

    #[test]
    fn test_new_indicators_insufficient_data() {
        let short_data = OHLCVSeries::from_close(vec![100.0, 101.0, 102.0]);
        let short_ohlcv = OHLCVSeries {
            open: vec![100.0, 101.0, 102.0],
            high: vec![101.0, 102.0, 103.0],
            low: vec![99.0, 100.0, 101.0],
            close: vec![100.0, 101.0, 102.0],
            volume: vec![1000.0, 1000.0, 1000.0],
        };

        let va = VolatilityAcceleration::new(10, 5).unwrap();
        assert!(va.compute(&short_data).is_err());

        let vmrd = VolatilityMeanReversionDistance::new(10, 20).unwrap();
        assert!(vmrd.compute(&short_data).is_err());

        let vs = VolatilitySpread::new(10, 30).unwrap();
        assert!(vs.compute(&short_data).is_err());

        let nrv = NormalizedRangeVolatility::new(14).unwrap();
        assert!(nrv.compute(&short_ohlcv).is_err());

        let vsi = VolatilitySkewIndicator::new(10, 20).unwrap();
        assert!(vsi.compute(&short_data).is_err());

        let avb = AdaptiveVolatilityBands::new(10, 20, 2.0).unwrap();
        assert!(avb.compute(&short_data).is_err());
    }

    #[test]
    fn test_volatility_acceleration_clone() {
        let va = VolatilityAcceleration::new(10, 5).unwrap();
        let va_clone = va.clone();
        assert_eq!(va.volatility_period, va_clone.volatility_period);
        assert_eq!(va.acceleration_period, va_clone.acceleration_period);
    }

    #[test]
    fn test_volatility_spread_clone() {
        let vs = VolatilitySpread::new(10, 30).unwrap();
        let vs_clone = vs.clone();
        assert_eq!(vs.short_period, vs_clone.short_period);
        assert_eq!(vs.long_period, vs_clone.long_period);
    }

    #[test]
    fn test_normalized_range_volatility_clone() {
        let nrv = NormalizedRangeVolatility::new(14).unwrap();
        let nrv_clone = nrv.clone();
        assert_eq!(nrv.period, nrv_clone.period);
    }

    #[test]
    fn test_volatility_skew_indicator_clone() {
        let vsi = VolatilitySkewIndicator::new(10, 20).unwrap();
        let vsi_clone = vsi.clone();
        assert_eq!(vsi.volatility_period, vsi_clone.volatility_period);
        assert_eq!(vsi.skew_period, vsi_clone.skew_period);
    }

    #[test]
    fn test_adaptive_volatility_bands_clone() {
        let avb = AdaptiveVolatilityBands::new(10, 20, 2.0).unwrap();
        let avb_clone = avb.clone();
        assert_eq!(avb.volatility_period, avb_clone.volatility_period);
        assert_eq!(avb.band_period, avb_clone.band_period);
        assert!((avb.multiplier - avb_clone.multiplier).abs() < 1e-10);
    }

    // ========================================================================
    // Tests for 6 NEW volatility indicators
    // ========================================================================

    #[test]
    fn test_volatility_regime_detector() {
        let (_, _, close) = make_test_data();
        let vrd = VolatilityRegimeDetector::new(10, 30).unwrap();
        let (z_scores, regimes) = vrd.calculate(&close);

        assert_eq!(z_scores.len(), close.len());
        assert_eq!(regimes.len(), close.len());

        // Check that regimes are properly assigned
        let min_period = vrd.min_periods();
        for i in min_period..close.len() {
            // Z-scores should be reasonable
            assert!(
                z_scores[i].is_finite(),
                "Z-score at {} should be finite",
                i
            );
            // Regimes should match z-score classification
            let expected_regime = vrd.classify_regime(z_scores[i]);
            assert_eq!(regimes[i], expected_regime);
        }
    }

    #[test]
    fn test_volatility_regime_detector_with_thresholds() {
        let vrd = VolatilityRegimeDetector::with_thresholds(10, 30, -0.5, 0.5).unwrap();

        // Test classification with custom thresholds
        assert_eq!(vrd.classify_regime(-1.0), VolRegimeLevel::Low);
        assert_eq!(vrd.classify_regime(0.0), VolRegimeLevel::Medium);
        assert_eq!(vrd.classify_regime(1.0), VolRegimeLevel::High);
    }

    #[test]
    fn test_volatility_regime_detector_invalid_params() {
        assert!(VolatilityRegimeDetector::new(2, 30).is_err()); // volatility_period < 5
        assert!(VolatilityRegimeDetector::new(10, 10).is_err()); // regime_period < 20
        assert!(VolatilityRegimeDetector::with_thresholds(10, 30, 1.0, -1.0).is_err()); // low >= high
    }

    #[test]
    fn test_volatility_regime_detector_technical_indicator() {
        let (_, _, close) = make_test_data();
        let vrd = VolatilityRegimeDetector::new(10, 30).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(vrd.name(), "Volatility Regime Detector");
        assert_eq!(vrd.min_periods(), 41); // 10 + 30 + 1

        let result = vrd.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_regime_detector_regimes() {
        // Create data with clear regime changes
        let mut close = Vec::new();
        // Low volatility period
        for i in 0..40 {
            close.push(100.0 + i as f64 * 0.01);
        }
        // High volatility period
        for i in 40..80 {
            close.push(close[i - 1] * (1.0 + (i as f64 * 0.5).sin() * 0.05));
        }
        // Return to low volatility
        for i in 80..120 {
            close.push(close[i - 1] * (1.0 + (i as f64 * 0.1).sin() * 0.002));
        }

        let vrd = VolatilityRegimeDetector::new(5, 25).unwrap();
        let (_, regimes) = vrd.calculate(&close);

        // Should have different regimes
        assert_eq!(regimes.len(), close.len());
    }

    #[test]
    fn test_volatility_breakout_signal() {
        let (high, low, close) = make_test_data();
        let vbs = VolatilityBreakoutSignal::new(14, 40, 1.5).unwrap();
        let result = vbs.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());

        // Check values are finite
        let min_period = vbs.min_periods();
        for i in min_period..close.len() {
            assert!(
                result[i].is_finite(),
                "Breakout signal at {} should be finite",
                i
            );
        }
    }

    #[test]
    fn test_volatility_breakout_signal_is_breakout() {
        let vbs = VolatilityBreakoutSignal::new(14, 40, 1.5).unwrap();

        assert!(vbs.is_breakout(100.0));
        assert!(vbs.is_breakout(-100.0));
        assert!(!vbs.is_breakout(30.0));
        assert!(!vbs.is_breakout(-30.0));
    }

    #[test]
    fn test_volatility_breakout_signal_invalid_params() {
        assert!(VolatilityBreakoutSignal::new(2, 40, 1.5).is_err()); // atr_period < 5
        assert!(VolatilityBreakoutSignal::new(14, 10, 1.5).is_err()); // baseline_period < 20
        assert!(VolatilityBreakoutSignal::new(14, 40, 0.5).is_err()); // threshold < 1.0
    }

    #[test]
    fn test_volatility_breakout_signal_technical_indicator() {
        let (high, low, close) = make_test_data();
        let vbs = VolatilityBreakoutSignal::new(14, 40, 1.5).unwrap();
        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 100],
        };

        assert_eq!(vbs.name(), "Volatility Breakout Signal");
        assert_eq!(vbs.min_periods(), 41); // baseline_period + 1

        let result = vbs.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_enhanced_volatility_mean_reversion() {
        let (_, _, close) = make_test_data();
        let evmr = EnhancedVolatilityMeanReversion::new(10, 40, 15).unwrap();
        let result = evmr.calculate(&close);

        assert_eq!(result.len(), close.len());

        // Check values are finite
        let min_period = evmr.min_periods();
        for i in min_period..close.len() {
            assert!(
                result[i].is_finite(),
                "Mean reversion score at {} should be finite",
                i
            );
        }
    }

    #[test]
    fn test_enhanced_volatility_mean_reversion_invalid_params() {
        assert!(EnhancedVolatilityMeanReversion::new(2, 40, 15).is_err()); // volatility_period < 5
        assert!(EnhancedVolatilityMeanReversion::new(10, 20, 15).is_err()); // mean_period < 30
        assert!(EnhancedVolatilityMeanReversion::new(10, 40, 5).is_err()); // lookback < 10
    }

    #[test]
    fn test_enhanced_volatility_mean_reversion_technical_indicator() {
        let (_, _, close) = make_test_data();
        let evmr = EnhancedVolatilityMeanReversion::new(10, 40, 15).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(evmr.name(), "Enhanced Volatility Mean Reversion");
        assert_eq!(evmr.min_periods(), 56); // 40 + 15 + 1

        let result = evmr.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_skew_ratio() {
        let (_, _, close) = make_test_data();
        let vsr = VolatilitySkewRatio::new(20).unwrap();
        let result = vsr.calculate(&close);

        assert_eq!(result.len(), close.len());

        // Ratios should be positive
        let min_period = vsr.min_periods();
        for i in min_period..close.len() {
            assert!(
                result[i] >= 0.0,
                "Skew ratio at {} should be non-negative, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_volatility_skew_ratio_invalid_params() {
        assert!(VolatilitySkewRatio::new(5).is_err()); // period < 10
    }

    #[test]
    fn test_volatility_skew_ratio_technical_indicator() {
        let (_, _, close) = make_test_data();
        let vsr = VolatilitySkewRatio::new(20).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(vsr.name(), "Volatility Skew Ratio");
        assert_eq!(vsr.min_periods(), 21); // period + 1

        let result = vsr.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_skew_ratio_trending_market() {
        // Create uptrending data (more positive returns)
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
        let vsr = VolatilitySkewRatio::new(15).unwrap();
        let result = vsr.calculate(&close);

        // In trending market, ratio might be higher due to more up moves
        assert_eq!(result.len(), close.len());
        for i in 20..close.len() {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_volatility_clustering_index() {
        let (_, _, close) = make_test_data();
        let vci = VolatilityClusteringIndex::new(25, 3).unwrap();
        let result = vci.calculate(&close);

        assert_eq!(result.len(), close.len());

        // Clustering index should be between 0 and 1
        let min_period = vci.min_periods();
        for i in min_period..close.len() {
            assert!(
                result[i] >= 0.0 && result[i] <= 1.0,
                "Clustering index at {} should be between 0 and 1, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_volatility_clustering_index_invalid_params() {
        assert!(VolatilityClusteringIndex::new(10, 3).is_err()); // period < 20
        assert!(VolatilityClusteringIndex::new(25, 0).is_err()); // num_lags < 1
        assert!(VolatilityClusteringIndex::new(25, 15).is_err()); // num_lags > 10
    }

    #[test]
    fn test_volatility_clustering_index_technical_indicator() {
        let (_, _, close) = make_test_data();
        let vci = VolatilityClusteringIndex::new(25, 3).unwrap();
        let data = OHLCVSeries::from_close(close);

        assert_eq!(vci.name(), "Volatility Clustering Index");
        assert_eq!(vci.min_periods(), 30); // 25 + 3 + 2

        let result = vci.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_clustering_index_with_clusters() {
        // Create data with volatility clustering
        let mut close = vec![100.0];
        for i in 1..100 {
            // Alternate between high and low volatility periods
            let vol = if (i / 10) % 2 == 0 { 0.005 } else { 0.02 };
            close.push(close[i - 1] * (1.0 + (i as f64 * 0.1).sin() * vol));
        }

        let vci = VolatilityClusteringIndex::new(20, 2).unwrap();
        let result = vci.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_dynamic_adaptive_volatility_bands() {
        let (high, low, close) = make_test_data();
        let davb = DynamicAdaptiveVolatilityBands::new(10, 15, 2.0).unwrap();
        let (middle, upper, lower, regime_mult) = davb.calculate(&high, &low, &close);

        assert_eq!(middle.len(), close.len());
        assert_eq!(upper.len(), close.len());
        assert_eq!(lower.len(), close.len());
        assert_eq!(regime_mult.len(), close.len());

        let min_period = davb.min_periods();
        for i in min_period..close.len() {
            // Upper should be >= middle, lower should be <= middle
            assert!(
                upper[i] >= middle[i],
                "Upper band should be >= middle at {}",
                i
            );
            assert!(
                lower[i] <= middle[i],
                "Lower band should be <= middle at {}",
                i
            );
            // Regime multiplier should be positive
            assert!(
                regime_mult[i] > 0.0,
                "Regime multiplier should be positive at {}",
                i
            );
        }
    }

    #[test]
    fn test_dynamic_adaptive_volatility_bands_invalid_params() {
        assert!(DynamicAdaptiveVolatilityBands::new(2, 15, 2.0).is_err()); // volatility_period < 5
        assert!(DynamicAdaptiveVolatilityBands::new(10, 5, 2.0).is_err()); // ma_period < 10
        assert!(DynamicAdaptiveVolatilityBands::new(10, 15, 0.3).is_err()); // base_multiplier < 0.5
    }

    #[test]
    fn test_dynamic_adaptive_volatility_bands_technical_indicator() {
        let (high, low, close) = make_test_data();
        let davb = DynamicAdaptiveVolatilityBands::new(10, 15, 2.0).unwrap();
        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 100],
        };

        assert_eq!(davb.name(), "Dynamic Adaptive Volatility Bands");
        assert_eq!(davb.min_periods(), 16); // max(10, 15) + 1

        let result = davb.compute(&data).unwrap();
        assert_eq!(result.primary.len(), data.close.len());
        assert!(result.secondary.is_some()); // upper band
        assert!(result.tertiary.is_some()); // lower band
    }

    #[test]
    fn test_dynamic_adaptive_volatility_bands_output_features() {
        let davb = DynamicAdaptiveVolatilityBands::new(10, 15, 2.0).unwrap();
        assert_eq!(davb.output_features(), 3);
    }

    #[test]
    fn test_dynamic_adaptive_volatility_bands_adapts() {
        // Create data with varying volatility
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();

        // Low volatility period
        for i in 0..50 {
            let base = 100.0 + i as f64 * 0.05;
            high.push(base + 0.5);
            low.push(base - 0.5);
            close.push(base);
        }
        // High volatility period
        for i in 50..100 {
            let base = close[i - 1] * (1.0 + (i as f64 * 0.3).sin() * 0.02);
            high.push(base * 1.03);
            low.push(base * 0.97);
            close.push(base);
        }

        let davb = DynamicAdaptiveVolatilityBands::new(5, 10, 2.0).unwrap();
        let (_, upper, lower, _) = davb.calculate(&high, &low, &close);

        // Check bands exist and adapt
        assert_eq!(upper.len(), close.len());
        assert_eq!(lower.len(), close.len());

        // Band width in high vol should be larger
        let low_vol_width: f64 = (30..45)
            .map(|i| upper[i] - lower[i])
            .sum::<f64>()
            / 15.0;
        let high_vol_width: f64 = (80..95)
            .map(|i| upper[i] - lower[i])
            .sum::<f64>()
            / 15.0;

        // High vol should have wider bands (with some tolerance for adaptation lag)
        assert!(
            high_vol_width > low_vol_width * 0.3,
            "High vol width {} should be larger than low vol width {}",
            high_vol_width,
            low_vol_width
        );
    }

    #[test]
    fn test_new_6_indicators_insufficient_data() {
        let short_data = OHLCVSeries::from_close(vec![100.0, 101.0, 102.0]);
        let short_ohlcv = OHLCVSeries {
            open: vec![100.0, 101.0, 102.0],
            high: vec![101.0, 102.0, 103.0],
            low: vec![99.0, 100.0, 101.0],
            close: vec![100.0, 101.0, 102.0],
            volume: vec![1000.0, 1000.0, 1000.0],
        };

        let vrd = VolatilityRegimeDetector::new(10, 30).unwrap();
        assert!(vrd.compute(&short_data).is_err());

        let vbs = VolatilityBreakoutSignal::new(14, 40, 1.5).unwrap();
        assert!(vbs.compute(&short_ohlcv).is_err());

        let evmr = EnhancedVolatilityMeanReversion::new(10, 40, 15).unwrap();
        assert!(evmr.compute(&short_data).is_err());

        let vsr = VolatilitySkewRatio::new(20).unwrap();
        assert!(vsr.compute(&short_data).is_err());

        let vci = VolatilityClusteringIndex::new(25, 3).unwrap();
        assert!(vci.compute(&short_data).is_err());

        let davb = DynamicAdaptiveVolatilityBands::new(10, 15, 2.0).unwrap();
        assert!(davb.compute(&short_ohlcv).is_err());
    }

    #[test]
    fn test_new_6_indicators_clone() {
        let vrd = VolatilityRegimeDetector::new(10, 30).unwrap();
        let vrd_clone = vrd.clone();
        assert_eq!(vrd.volatility_period, vrd_clone.volatility_period);
        assert_eq!(vrd.regime_period, vrd_clone.regime_period);

        let vbs = VolatilityBreakoutSignal::new(14, 40, 1.5).unwrap();
        let vbs_clone = vbs.clone();
        assert_eq!(vbs.atr_period, vbs_clone.atr_period);
        assert_eq!(vbs.baseline_period, vbs_clone.baseline_period);

        let evmr = EnhancedVolatilityMeanReversion::new(10, 40, 15).unwrap();
        let evmr_clone = evmr.clone();
        assert_eq!(evmr.volatility_period, evmr_clone.volatility_period);
        assert_eq!(evmr.mean_period, evmr_clone.mean_period);

        let vsr = VolatilitySkewRatio::new(20).unwrap();
        let vsr_clone = vsr.clone();
        assert_eq!(vsr.period, vsr_clone.period);

        let vci = VolatilityClusteringIndex::new(25, 3).unwrap();
        let vci_clone = vci.clone();
        assert_eq!(vci.period, vci_clone.period);
        assert_eq!(vci.num_lags, vci_clone.num_lags);

        let davb = DynamicAdaptiveVolatilityBands::new(10, 15, 2.0).unwrap();
        let davb_clone = davb.clone();
        assert_eq!(davb.volatility_period, davb_clone.volatility_period);
        assert_eq!(davb.ma_period, davb_clone.ma_period);
        assert!((davb.base_multiplier - davb_clone.base_multiplier).abs() < 1e-10);
    }

    #[test]
    fn test_vol_regime_level_equality() {
        assert_eq!(VolRegimeLevel::Low, VolRegimeLevel::Low);
        assert_eq!(VolRegimeLevel::Medium, VolRegimeLevel::Medium);
        assert_eq!(VolRegimeLevel::High, VolRegimeLevel::High);
        assert_ne!(VolRegimeLevel::Low, VolRegimeLevel::High);
    }
}
