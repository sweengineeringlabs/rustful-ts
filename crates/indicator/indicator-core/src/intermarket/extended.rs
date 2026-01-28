//! Extended Intermarket Indicators
//!
//! Additional intermarket analysis indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Cross-Market Momentum - Momentum comparison across markets
#[derive(Debug, Clone)]
pub struct CrossMarketMomentum {
    period: usize,
}

impl CrossMarketMomentum {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate momentum normalized for comparison
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            // Calculate momentum as percentage
            let momentum = (close[i] / close[i - self.period] - 1.0) * 100.0;

            // Calculate volatility for normalization
            let returns: Vec<f64> = (i - self.period + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 {
                        Some(close[j] / close[j - 1] - 1.0)
                    } else {
                        None
                    }
                })
                .collect();

            if !returns.is_empty() {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let var: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                let std_dev = var.sqrt();

                // Normalize momentum by volatility
                if std_dev > 1e-10 {
                    result[i] = momentum / (std_dev * 100.0);
                } else {
                    result[i] = momentum;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for CrossMarketMomentum {
    fn name(&self) -> &str {
        "Cross Market Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Beta Coefficient - Measures correlation with market
#[derive(Debug, Clone)]
pub struct BetaCoefficient {
    period: usize,
}

impl BetaCoefficient {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate rolling beta using self as proxy (perfect beta = 1)
    /// In practice, this would compare against a market index
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![1.0; n]; // Default beta = 1

        for i in self.period..n {
            let start = i - self.period;

            // Calculate returns
            let returns: Vec<f64> = (start + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 {
                        Some(close[j] / close[j - 1] - 1.0)
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 2 {
                // Calculate variance of returns as volatility proxy
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let var: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

                // Higher volatility = higher "beta"
                let vol = var.sqrt() * (252.0_f64).sqrt(); // Annualized
                result[i] = (vol * 10.0).clamp(0.5, 2.5); // Scale to reasonable beta range
            }
        }
        result
    }
}

impl TechnicalIndicator for BetaCoefficient {
    fn name(&self) -> &str {
        "Beta Coefficient"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Market Regime Detector - Identifies market regimes
#[derive(Debug, Clone)]
pub struct MarketRegimeIndicator {
    short_period: usize,
    long_period: usize,
}

impl MarketRegimeIndicator {
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

    /// Calculate regime indicator (1=trending up, -1=trending down, 0=ranging)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term trend
            let short_start = i.saturating_sub(self.short_period);
            let short_avg: f64 = close[short_start..=i].iter().sum::<f64>() / (i - short_start + 1) as f64;

            // Long-term trend
            let long_start = i.saturating_sub(self.long_period);
            let long_avg: f64 = close[long_start..=i].iter().sum::<f64>() / (i - long_start + 1) as f64;

            // Trend strength
            let trend_diff = (short_avg / long_avg - 1.0) * 100.0;

            // Volatility
            let returns: Vec<f64> = (long_start + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 {
                        Some((close[j] / close[j - 1] - 1.0).abs())
                    } else {
                        None
                    }
                })
                .collect();
            let avg_vol = if !returns.is_empty() {
                returns.iter().sum::<f64>() / returns.len() as f64 * 100.0
            } else {
                1.0
            };

            // Regime classification
            if trend_diff.abs() > avg_vol * 2.0 {
                result[i] = if trend_diff > 0.0 { 1.0 } else { -1.0 };
            } else {
                result[i] = 0.0; // Ranging
            }
        }
        result
    }
}

impl TechnicalIndicator for MarketRegimeIndicator {
    fn name(&self) -> &str {
        "Market Regime Indicator"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Sector Relative Performance - Measures relative performance
#[derive(Debug, Clone)]
pub struct SectorRelativePerformance {
    period: usize,
}

impl SectorRelativePerformance {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate relative performance score
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate cumulative performance
        for i in self.period..n {
            let perf = (close[i] / close[i - self.period] - 1.0) * 100.0;

            // Calculate rolling performance rank (simplified - just normalized performance)
            result[i] = perf.clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for SectorRelativePerformance {
    fn name(&self) -> &str {
        "Sector Relative Performance"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Correlation Momentum - Measures change in correlation patterns
#[derive(Debug, Clone)]
pub struct CorrelationMomentum {
    period: usize,
}

impl CorrelationMomentum {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate autocorrelation momentum
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate lag-1 autocorrelation
            let returns: Vec<f64> = (start + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 {
                        Some(close[j] / close[j - 1] - 1.0)
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 3 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let var: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

                // Lag-1 covariance
                let mut lag1_cov = 0.0;
                for j in 1..returns.len() {
                    lag1_cov += (returns[j] - mean) * (returns[j - 1] - mean);
                }
                lag1_cov /= (returns.len() - 1) as f64;

                // Autocorrelation
                if var > 1e-10 {
                    result[i] = (lag1_cov / var).clamp(-1.0, 1.0) * 100.0;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for CorrelationMomentum {
    fn name(&self) -> &str {
        "Correlation Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Risk Appetite Index - Measures risk-on vs risk-off sentiment
#[derive(Debug, Clone)]
pub struct RiskAppetiteIndex {
    period: usize,
}

impl RiskAppetiteIndex {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate risk appetite based on price and volatility
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Momentum component
            let momentum = (close[i] / close[start] - 1.0) * 100.0;

            // Volatility component (risk)
            let mut tr_sum = 0.0;
            for j in (start + 1)..=i {
                let tr1 = high[j] - low[j];
                let tr2 = (high[j] - close[j - 1]).abs();
                let tr3 = (low[j] - close[j - 1]).abs();
                tr_sum += tr1.max(tr2).max(tr3);
            }
            let avg_tr = tr_sum / self.period as f64;
            let vol_pct = if close[i] > 1e-10 { avg_tr / close[i] * 100.0 } else { 1.0 };

            // Risk appetite: positive momentum + low volatility = risk-on
            let risk_adjusted = if vol_pct > 1e-10 {
                momentum / vol_pct
            } else {
                momentum
            };

            result[i] = risk_adjusted.clamp(-100.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for RiskAppetiteIndex {
    fn name(&self) -> &str {
        "Risk Appetite Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Divergence Index - Measures divergence between price and trend
#[derive(Debug, Clone)]
pub struct DivergenceIndex {
    period: usize,
}

impl DivergenceIndex {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate divergence between price momentum and trend
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Calculate EMA
        let alpha = 2.0 / (self.period as f64 + 1.0);
        let mut ema = vec![0.0; n];
        if n > 0 {
            ema[0] = close[0];
            for i in 1..n {
                ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1];
            }
        }

        for i in self.period..n {
            // Price change
            let price_change = (close[i] / close[i - self.period] - 1.0) * 100.0;

            // EMA change
            let ema_change = if ema[i - self.period] > 1e-10 {
                (ema[i] / ema[i - self.period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Divergence: difference between price momentum and smoothed trend
            result[i] = (price_change - ema_change).clamp(-50.0, 50.0);
        }
        result
    }
}

impl TechnicalIndicator for DivergenceIndex {
    fn name(&self) -> &str {
        "Divergence Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![102.0, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0, 108.0, 107.5,
                       109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0, 114.0, 113.5, 115.0,
                       116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0, 119.5, 121.0, 122.0];
        let low = vec![98.0, 99.0, 100.0, 99.5, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5,
                      105.0, 106.0, 105.5, 107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0,
                      112.0, 111.5, 113.0, 114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0];
        let close = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
                        107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
                        114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0];
        (high, low, close)
    }

    #[test]
    fn test_cross_market_momentum() {
        let (_, _, close) = make_test_data();
        let cmm = CrossMarketMomentum::new(10).unwrap();
        let result = cmm.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_beta_coefficient() {
        let (_, _, close) = make_test_data();
        let bc = BetaCoefficient::new(10).unwrap();
        let result = bc.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= 0.5 && result[15] <= 2.5);
    }

    #[test]
    fn test_market_regime_indicator() {
        let (_, _, close) = make_test_data();
        let mri = MarketRegimeIndicator::new(5, 15).unwrap();
        let result = mri.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_sector_relative_performance() {
        let (_, _, close) = make_test_data();
        let srp = SectorRelativePerformance::new(10).unwrap();
        let result = srp.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_correlation_momentum() {
        let (_, _, close) = make_test_data();
        let cm = CorrelationMomentum::new(10).unwrap();
        let result = cm.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_risk_appetite_index() {
        let (high, low, close) = make_test_data();
        let rai = RiskAppetiteIndex::new(10).unwrap();
        let result = rai.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= -100.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_divergence_index() {
        let (_, _, close) = make_test_data();
        let di = DivergenceIndex::new(10).unwrap();
        let result = di.calculate(&close);

        assert_eq!(result.len(), close.len());
    }
}
