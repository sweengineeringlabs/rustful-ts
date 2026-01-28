//! Extended Risk Metrics
//!
//! Additional risk and performance indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Sterling Ratio - Risk-adjusted return using average drawdown
#[derive(Debug, Clone)]
pub struct SterlingRatio {
    period: usize,
    risk_free_rate: f64,
}

impl SterlingRatio {
    pub fn new(period: usize, risk_free_rate: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period, risk_free_rate })
    }

    /// Calculate Sterling Ratio
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let returns: Vec<f64> = (start + 1..=i)
                .map(|j| close[j] / close[j - 1] - 1.0)
                .collect();

            let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let annualized_return = avg_return * 252.0;
            let excess_return = annualized_return - self.risk_free_rate;

            // Calculate average drawdown
            let mut peak = close[start];
            let mut drawdowns = Vec::new();
            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                let dd = (peak - close[j]) / peak;
                if dd > 0.0 {
                    drawdowns.push(dd);
                }
            }

            let avg_dd = if drawdowns.is_empty() {
                0.0
            } else {
                drawdowns.iter().sum::<f64>() / drawdowns.len() as f64
            };

            if avg_dd > 1e-10 {
                result[i] = excess_return / avg_dd;
            }
        }
        result
    }
}

impl TechnicalIndicator for SterlingRatio {
    fn name(&self) -> &str {
        "Sterling Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Burke Ratio - Modified Sterling using squared drawdowns
#[derive(Debug, Clone)]
pub struct BurkeRatio {
    period: usize,
    risk_free_rate: f64,
}

impl BurkeRatio {
    pub fn new(period: usize, risk_free_rate: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period, risk_free_rate })
    }

    /// Calculate Burke Ratio
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let returns: Vec<f64> = (start + 1..=i)
                .map(|j| close[j] / close[j - 1] - 1.0)
                .collect();

            let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let annualized_return = avg_return * 252.0;
            let excess_return = annualized_return - self.risk_free_rate;

            // Calculate root mean square of drawdowns
            let mut peak = close[start];
            let mut squared_dd_sum = 0.0;
            let mut dd_count = 0;
            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                let dd = (peak - close[j]) / peak;
                if dd > 0.0 {
                    squared_dd_sum += dd * dd;
                    dd_count += 1;
                }
            }

            if dd_count > 0 {
                let rms_dd = (squared_dd_sum / dd_count as f64).sqrt();
                if rms_dd > 1e-10 {
                    result[i] = excess_return / rms_dd;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for BurkeRatio {
    fn name(&self) -> &str {
        "Burke Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Ulcer Performance Index - Return per unit of ulcer index
#[derive(Debug, Clone)]
pub struct UlcerPerformanceIndex {
    period: usize,
    risk_free_rate: f64,
}

impl UlcerPerformanceIndex {
    pub fn new(period: usize, risk_free_rate: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period, risk_free_rate })
    }

    /// Calculate UPI (Martin Ratio)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate return
            let period_return = (close[i] / close[start] - 1.0) * (252.0 / self.period as f64);
            let excess_return = period_return - self.risk_free_rate;

            // Calculate Ulcer Index
            let max_high = close[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let squared_dd_sum: f64 = close[start..=i]
                .iter()
                .map(|&c| {
                    let pct_dd = 100.0 * (max_high - c) / max_high;
                    pct_dd * pct_dd
                })
                .sum();

            let ulcer_index = (squared_dd_sum / (self.period as f64 + 1.0)).sqrt();

            if ulcer_index > 1e-10 {
                result[i] = excess_return / (ulcer_index / 100.0);
            }
        }
        result
    }
}

impl TechnicalIndicator for UlcerPerformanceIndex {
    fn name(&self) -> &str {
        "Ulcer Performance Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Pain Index - Average drawdown depth
#[derive(Debug, Clone)]
pub struct PainIndex {
    period: usize,
}

impl PainIndex {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Pain Index (average drawdown)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Find running peak and calculate average drawdown
            let mut peak = close[start];
            let mut dd_sum = 0.0;

            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                dd_sum += (peak - close[j]) / peak;
            }

            result[i] = dd_sum / (self.period as f64 + 1.0) * 100.0;
        }
        result
    }
}

impl TechnicalIndicator for PainIndex {
    fn name(&self) -> &str {
        "Pain Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Recovery Factor - Total return divided by max drawdown
#[derive(Debug, Clone)]
pub struct RecoveryFactor {
    period: usize,
}

impl RecoveryFactor {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Recovery Factor
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate total return
            let total_return = (close[i] / close[start] - 1.0) * 100.0;

            // Calculate max drawdown
            let mut peak = close[start];
            let mut max_dd = 0.0;

            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                let dd = (peak - close[j]) / peak;
                if dd > max_dd {
                    max_dd = dd;
                }
            }

            if max_dd > 1e-10 {
                result[i] = total_return / (max_dd * 100.0);
            }
        }
        result
    }
}

impl TechnicalIndicator for RecoveryFactor {
    fn name(&self) -> &str {
        "Recovery Factor"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Tail Ratio - Ratio of positive to negative tail risk
#[derive(Debug, Clone)]
pub struct TailRatio {
    period: usize,
    percentile: f64,
}

impl TailRatio {
    pub fn new(period: usize, percentile: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if percentile <= 0.0 || percentile >= 50.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "percentile".to_string(),
                reason: "must be between 0 and 50".to_string(),
            });
        }
        Ok(Self { period, percentile })
    }

    /// Calculate Tail Ratio
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let mut returns: Vec<f64> = (start + 1..=i)
                .map(|j| close[j] / close[j - 1] - 1.0)
                .collect();

            returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n_returns = returns.len();
            let tail_idx = ((self.percentile / 100.0) * n_returns as f64) as usize;

            if tail_idx > 0 && tail_idx < n_returns {
                // Left tail (negative returns) - average of worst percentile
                let left_tail: f64 = returns[..tail_idx].iter().sum::<f64>() / tail_idx as f64;

                // Right tail (positive returns) - average of best percentile
                let right_tail: f64 = returns[(n_returns - tail_idx)..].iter().sum::<f64>() / tail_idx as f64;

                if left_tail.abs() > 1e-10 {
                    result[i] = right_tail / left_tail.abs();
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for TailRatio {
    fn name(&self) -> &str {
        "Tail Ratio"
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
        vec![100.0, 102.0, 101.0, 104.0, 103.0, 106.0, 105.0, 108.0, 107.0, 110.0,
             109.0, 112.0, 111.0, 114.0, 113.0, 116.0, 115.0, 118.0, 117.0, 120.0,
             119.0, 122.0, 121.0, 124.0, 123.0, 126.0, 125.0, 128.0, 127.0, 130.0]
    }

    #[test]
    fn test_sterling_ratio() {
        let close = make_test_data();
        let sr = SterlingRatio::new(20, 0.02).unwrap();
        let result = sr.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[25] > 0.0); // Positive ratio in uptrend
    }

    #[test]
    fn test_burke_ratio() {
        let close = make_test_data();
        let br = BurkeRatio::new(20, 0.02).unwrap();
        let result = br.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[25] > 0.0); // Positive ratio in uptrend
    }

    #[test]
    fn test_ulcer_performance_index() {
        let close = make_test_data();
        let upi = UlcerPerformanceIndex::new(20, 0.02).unwrap();
        let result = upi.calculate(&close);

        assert_eq!(result.len(), close.len());
        // UPI should be positive in uptrend
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_pain_index() {
        let close = make_test_data();
        let pi = PainIndex::new(10).unwrap();
        let result = pi.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Pain index is non-negative
        assert!(result[15] >= 0.0);
    }

    #[test]
    fn test_recovery_factor() {
        let close = make_test_data();
        let rf = RecoveryFactor::new(20).unwrap();
        let result = rf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Positive recovery factor in uptrend
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_tail_ratio() {
        let close = make_test_data();
        let tr = TailRatio::new(20, 10.0).unwrap();
        let result = tr.calculate(&close);

        assert_eq!(result.len(), close.len());
    }
}
