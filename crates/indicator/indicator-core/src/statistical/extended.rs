//! Extended Statistical Indicators
//!
//! Additional statistical indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Rolling Variance - Rolling variance of returns
#[derive(Debug, Clone)]
pub struct RollingVariance {
    period: usize,
}

impl RollingVariance {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate rolling variance of log returns
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate log returns
            let returns: Vec<f64> = (start + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 && close[j] > 0.0 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if !returns.is_empty() {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                result[i] = variance * 10000.0; // Scale to basis points squared
            }
        }
        result
    }
}

impl TechnicalIndicator for RollingVariance {
    fn name(&self) -> &str {
        "Rolling Variance"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Rolling Skewness - Rolling skewness of returns
#[derive(Debug, Clone)]
pub struct RollingSkewness {
    period: usize,
}

impl RollingSkewness {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate rolling skewness
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate returns
            let returns: Vec<f64> = (start + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 && close[j] > 0.0 {
                        Some(close[j] / close[j - 1] - 1.0)
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 3 {
                let n_r = returns.len() as f64;
                let mean: f64 = returns.iter().sum::<f64>() / n_r;
                let m2: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n_r;
                let m3: f64 = returns.iter().map(|r| (r - mean).powi(3)).sum::<f64>() / n_r;

                if m2 > 1e-10 {
                    let std_dev = m2.sqrt();
                    result[i] = m3 / std_dev.powi(3);
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for RollingSkewness {
    fn name(&self) -> &str {
        "Rolling Skewness"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Rolling Kurtosis - Rolling kurtosis of returns
#[derive(Debug, Clone)]
pub struct RollingKurtosis {
    period: usize,
}

impl RollingKurtosis {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate rolling kurtosis (excess kurtosis)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate returns
            let returns: Vec<f64> = (start + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 && close[j] > 0.0 {
                        Some(close[j] / close[j - 1] - 1.0)
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 4 {
                let n_r = returns.len() as f64;
                let mean: f64 = returns.iter().sum::<f64>() / n_r;
                let m2: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n_r;
                let m4: f64 = returns.iter().map(|r| (r - mean).powi(4)).sum::<f64>() / n_r;

                if m2 > 1e-10 {
                    // Excess kurtosis (normal = 0)
                    result[i] = m4 / m2.powi(2) - 3.0;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for RollingKurtosis {
    fn name(&self) -> &str {
        "Rolling Kurtosis"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Price Distribution - Measures where price falls in distribution
#[derive(Debug, Clone)]
pub struct PriceDistribution {
    period: usize,
}

impl PriceDistribution {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate percentile rank of current price in distribution
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Count how many values are below current
            let below_count = close[start..i]
                .iter()
                .filter(|&&v| v < close[i])
                .count();

            // Percentile rank (0-100)
            result[i] = (below_count as f64 / self.period as f64) * 100.0;
        }
        result
    }
}

impl TechnicalIndicator for PriceDistribution {
    fn name(&self) -> &str {
        "Price Distribution"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Return Distribution - Statistics about return distribution
#[derive(Debug, Clone)]
pub struct ReturnDistribution {
    period: usize,
}

impl ReturnDistribution {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate return distribution score (positive = fat tails, negative = thin tails)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate returns
            let returns: Vec<f64> = (start + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 && close[j] > 0.0 {
                        Some(close[j] / close[j - 1] - 1.0)
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 5 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let std_dev: f64 = (returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64).sqrt();

                if std_dev > 1e-10 {
                    // Count extreme returns (beyond 2 std dev)
                    let extreme_count = returns.iter()
                        .filter(|&&r| (r - mean).abs() > 2.0 * std_dev)
                        .count();

                    // Expected under normal: ~4.55%
                    let expected_extreme = 0.0455 * returns.len() as f64;
                    let extreme_ratio = extreme_count as f64 / expected_extreme.max(0.1);

                    // Score: ratio of actual to expected extreme events
                    result[i] = (extreme_ratio - 1.0) * 100.0;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for ReturnDistribution {
    fn name(&self) -> &str {
        "Return Distribution"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Tail Risk Indicator - Measures left tail risk
#[derive(Debug, Clone)]
pub struct TailRiskIndicator {
    period: usize,
    confidence: f64,
}

impl TailRiskIndicator {
    pub fn new(period: usize, confidence: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "confidence".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { period, confidence })
    }

    /// Calculate tail risk (expected shortfall proxy)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Calculate returns
            let mut returns: Vec<f64> = (start + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 && close[j] > 0.0 {
                        Some(close[j] / close[j - 1] - 1.0)
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 5 {
                // Sort returns ascending
                returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                // VaR at confidence level
                let var_idx = ((1.0 - self.confidence) * returns.len() as f64).floor() as usize;
                let var_idx = var_idx.min(returns.len() - 1);

                // Expected shortfall: average of returns below VaR
                if var_idx > 0 {
                    let tail_returns = &returns[..=var_idx];
                    let es = tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
                    result[i] = es * 100.0; // As percentage
                } else {
                    result[i] = returns[0] * 100.0;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for TailRiskIndicator {
    fn name(&self) -> &str {
        "Tail Risk Indicator"
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

    fn make_test_data() -> Vec<f64> {
        vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
             107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
             114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0]
    }

    #[test]
    fn test_rolling_variance() {
        let close = make_test_data();
        let rv = RollingVariance::new(10).unwrap();
        let result = rv.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= 0.0);
    }

    #[test]
    fn test_rolling_skewness() {
        let close = make_test_data();
        let rs = RollingSkewness::new(10).unwrap();
        let result = rs.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_rolling_kurtosis() {
        let close = make_test_data();
        let rk = RollingKurtosis::new(10).unwrap();
        let result = rk.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_price_distribution() {
        let close = make_test_data();
        let pd = PriceDistribution::new(10).unwrap();
        let result = pd.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[15] >= 0.0 && result[15] <= 100.0);
    }

    #[test]
    fn test_return_distribution() {
        let close = make_test_data();
        let rd = ReturnDistribution::new(10).unwrap();
        let result = rd.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_tail_risk_indicator() {
        let close = make_test_data();
        let tri = TailRiskIndicator::new(20, 0.95).unwrap();
        let result = tri.calculate(&close);

        assert_eq!(result.len(), close.len());
    }
}
