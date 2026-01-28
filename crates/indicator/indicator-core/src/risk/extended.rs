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

/// Conditional Drawdown - Conditional drawdown measure (expected drawdown given in drawdown)
#[derive(Debug, Clone)]
pub struct ConditionalDrawdown {
    period: usize,
    threshold: f64,
}

impl ConditionalDrawdown {
    /// Create a new Conditional Drawdown indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    /// * `threshold` - Drawdown threshold for conditional calculation (e.g., 0.05 for 5%)
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if threshold <= 0.0 || threshold >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0 and 1 (exclusive)".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate Conditional Drawdown values
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate drawdowns for each point in window
            let mut peak = close[start];
            let mut drawdowns = Vec::new();

            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                let dd = (peak - close[j]) / peak;
                drawdowns.push(dd);
            }

            // Filter drawdowns exceeding threshold and calculate conditional mean
            let conditional_dds: Vec<f64> = drawdowns
                .iter()
                .filter(|&&dd| dd >= self.threshold)
                .cloned()
                .collect();

            if !conditional_dds.is_empty() {
                result[i] = conditional_dds.iter().sum::<f64>() / conditional_dds.len() as f64;
            }
        }
        result
    }
}

impl TechnicalIndicator for ConditionalDrawdown {
    fn name(&self) -> &str {
        "Conditional Drawdown"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Risk-Adjusted Return - Risk-adjusted return metric (return per unit of risk)
#[derive(Debug, Clone)]
pub struct RiskAdjustedReturn {
    period: usize,
    risk_free_rate: f64,
}

impl RiskAdjustedReturn {
    /// Create a new Risk-Adjusted Return indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    /// * `risk_free_rate` - Annualized risk-free rate (e.g., 0.02 for 2%)
    pub fn new(period: usize, risk_free_rate: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if risk_free_rate < 0.0 || risk_free_rate > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "risk_free_rate".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { period, risk_free_rate })
    }

    /// Calculate Risk-Adjusted Return values
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let returns: Vec<f64> = (start + 1..=i)
                .map(|j| close[j] / close[j - 1] - 1.0)
                .collect();

            if returns.is_empty() {
                continue;
            }

            let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;

            // Calculate volatility (standard deviation of returns)
            let variance = returns
                .iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>() / (returns.len() - 1).max(1) as f64;
            let volatility = variance.sqrt();

            // Annualize
            let annualized_return = avg_return * 252.0;
            let annualized_vol = volatility * (252.0_f64).sqrt();
            let excess_return = annualized_return - self.risk_free_rate;

            if annualized_vol > 1e-10 {
                result[i] = excess_return / annualized_vol;
            }
        }
        result
    }
}

impl TechnicalIndicator for RiskAdjustedReturn {
    fn name(&self) -> &str {
        "Risk-Adjusted Return"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Return Variance - Rolling variance of returns
#[derive(Debug, Clone)]
pub struct ReturnVariance {
    period: usize,
    annualize: bool,
}

impl ReturnVariance {
    /// Create a new Return Variance indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    /// * `annualize` - Whether to annualize the variance
    pub fn new(period: usize, annualize: bool) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, annualize })
    }

    /// Calculate Return Variance values
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let returns: Vec<f64> = (start + 1..=i)
                .map(|j| close[j] / close[j - 1] - 1.0)
                .collect();

            if returns.len() < 2 {
                continue;
            }

            let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;

            // Calculate variance
            let variance = returns
                .iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64;

            result[i] = if self.annualize {
                variance * 252.0
            } else {
                variance
            };
        }
        result
    }
}

impl TechnicalIndicator for ReturnVariance {
    fn name(&self) -> &str {
        "Return Variance"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Drawdown Duration - Duration of drawdowns in periods
#[derive(Debug, Clone)]
pub struct DrawdownDuration {
    period: usize,
}

impl DrawdownDuration {
    /// Create a new Drawdown Duration indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Drawdown Duration values (average duration of drawdown periods)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Track drawdown periods
            let mut peak = close[start];
            let mut current_dd_duration = 0;
            let mut dd_durations = Vec::new();
            let mut in_drawdown = false;

            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                    if in_drawdown && current_dd_duration > 0 {
                        dd_durations.push(current_dd_duration);
                    }
                    current_dd_duration = 0;
                    in_drawdown = false;
                } else if close[j] < peak {
                    in_drawdown = true;
                    current_dd_duration += 1;
                }
            }

            // Include final drawdown if still in one
            if in_drawdown && current_dd_duration > 0 {
                dd_durations.push(current_dd_duration);
            }

            // Calculate average drawdown duration
            if !dd_durations.is_empty() {
                result[i] = dd_durations.iter().sum::<i32>() as f64 / dd_durations.len() as f64;
            }
        }
        result
    }

    /// Calculate current drawdown duration at each point
    pub fn calculate_current(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        let mut peak = close[0];
        let mut current_duration = 0;

        for i in 0..n {
            if close[i] >= peak {
                peak = close[i];
                current_duration = 0;
            } else {
                current_duration += 1;
            }
            result[i] = current_duration as f64;
        }
        result
    }
}

impl TechnicalIndicator for DrawdownDuration {
    fn name(&self) -> &str {
        "Drawdown Duration"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Recovery Ratio - Ratio of recovery to drawdown (how quickly recoveries happen)
#[derive(Debug, Clone)]
pub struct RecoveryRatio {
    period: usize,
}

impl RecoveryRatio {
    /// Create a new Recovery Ratio indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Recovery Ratio values
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Track drawdowns and recoveries
            let mut peak = close[start];
            let mut trough = close[start];
            let mut in_drawdown = false;
            let mut dd_depths = Vec::new();
            let mut recovery_depths = Vec::new();
            let mut current_dd_depth = 0.0;

            for j in start..=i {
                if close[j] > peak {
                    // New peak - if we were in drawdown, record recovery
                    if in_drawdown {
                        let recovery = (close[j] - trough) / trough;
                        recovery_depths.push(recovery);
                        dd_depths.push(current_dd_depth);
                    }
                    peak = close[j];
                    trough = close[j];
                    in_drawdown = false;
                    current_dd_depth = 0.0;
                } else if close[j] < trough {
                    trough = close[j];
                    current_dd_depth = (peak - trough) / peak;
                    in_drawdown = true;
                }
            }

            // Calculate ratio of average recovery to average drawdown
            if !dd_depths.is_empty() && !recovery_depths.is_empty() {
                let avg_dd = dd_depths.iter().sum::<f64>() / dd_depths.len() as f64;
                let avg_recovery = recovery_depths.iter().sum::<f64>() / recovery_depths.len() as f64;

                if avg_dd > 1e-10 {
                    result[i] = avg_recovery / avg_dd;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for RecoveryRatio {
    fn name(&self) -> &str {
        "Recovery Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volatility Risk Ratio - Ratio of volatility to expected returns
#[derive(Debug, Clone)]
pub struct VolatilityRiskRatio {
    period: usize,
}

impl VolatilityRiskRatio {
    /// Create a new Volatility Risk Ratio indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Volatility Risk Ratio values
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let returns: Vec<f64> = (start + 1..=i)
                .map(|j| close[j] / close[j - 1] - 1.0)
                .collect();

            if returns.len() < 2 {
                continue;
            }

            let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;

            // Calculate volatility (standard deviation of returns)
            let variance = returns
                .iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64;
            let volatility = variance.sqrt();

            // Ratio of volatility to absolute expected return
            // High ratio means more risk per unit of expected return
            if avg_return.abs() > 1e-10 {
                result[i] = volatility / avg_return.abs();
            }
        }
        result
    }
}

impl TechnicalIndicator for VolatilityRiskRatio {
    fn name(&self) -> &str {
        "Volatility Risk Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    fn make_volatile_test_data() -> Vec<f64> {
        vec![100.0, 102.0, 99.0, 104.0, 101.0, 106.0, 103.0, 108.0, 105.0, 110.0,
             107.0, 112.0, 109.0, 114.0, 111.0, 116.0, 113.0, 118.0, 115.0, 120.0,
             117.0, 122.0, 119.0, 124.0, 121.0, 126.0, 123.0, 128.0, 125.0, 130.0]
    }

    #[test]
    fn test_conditional_drawdown() {
        let close = make_volatile_test_data();
        let cdd = ConditionalDrawdown::new(20, 0.01).unwrap();
        let result = cdd.calculate(&close);

        assert_eq!(result.len(), close.len());
        // First period values should be zero
        assert_eq!(result[0], 0.0);
        // Later values may have conditional drawdown
        assert!(result[25] >= 0.0);
    }

    #[test]
    fn test_conditional_drawdown_invalid_params() {
        let result = ConditionalDrawdown::new(5, 0.01);
        assert!(result.is_err());

        let result = ConditionalDrawdown::new(20, 0.0);
        assert!(result.is_err());

        let result = ConditionalDrawdown::new(20, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_risk_adjusted_return() {
        let close = make_volatile_test_data();
        let rar = RiskAdjustedReturn::new(20, 0.02).unwrap();
        let result = rar.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Positive trend should give positive risk-adjusted return
        assert!(result[25] > 0.0);
    }

    #[test]
    fn test_risk_adjusted_return_invalid_params() {
        let result = RiskAdjustedReturn::new(2, 0.02);
        assert!(result.is_err());

        let result = RiskAdjustedReturn::new(20, -0.01);
        assert!(result.is_err());

        let result = RiskAdjustedReturn::new(20, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_return_variance() {
        let close = make_volatile_test_data();
        let rv = ReturnVariance::new(20, false).unwrap();
        let result = rv.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Variance should be non-negative
        assert!(result[25] >= 0.0);
    }

    #[test]
    fn test_return_variance_annualized() {
        let close = make_volatile_test_data();
        let rv_daily = ReturnVariance::new(20, false).unwrap();
        let rv_annual = ReturnVariance::new(20, true).unwrap();
        let daily_result = rv_daily.calculate(&close);
        let annual_result = rv_annual.calculate(&close);

        // Annualized variance should be ~252 times daily variance
        let ratio = annual_result[25] / daily_result[25];
        assert!((ratio - 252.0).abs() < 0.01);
    }

    #[test]
    fn test_return_variance_invalid_params() {
        let result = ReturnVariance::new(1, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_drawdown_duration() {
        let close = make_volatile_test_data();
        let dd = DrawdownDuration::new(20).unwrap();
        let result = dd.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Duration should be non-negative
        assert!(result[25] >= 0.0);
    }

    #[test]
    fn test_drawdown_duration_current() {
        let close = vec![100.0, 110.0, 105.0, 103.0, 108.0, 115.0];
        let dd = DrawdownDuration::new(5).unwrap();
        let result = dd.calculate_current(&close);

        assert_eq!(result.len(), close.len());
        // At index 2, we're 1 period into drawdown from peak at 110
        assert_eq!(result[2], 1.0);
        // At index 3, we're 2 periods into drawdown
        assert_eq!(result[3], 2.0);
        // At index 5, new peak so duration resets
        assert_eq!(result[5], 0.0);
    }

    #[test]
    fn test_drawdown_duration_invalid_params() {
        let result = DrawdownDuration::new(2);
        assert!(result.is_err());
    }

    #[test]
    fn test_recovery_ratio() {
        let close = make_volatile_test_data();
        let rr = RecoveryRatio::new(20).unwrap();
        let result = rr.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Recovery ratio should be non-negative
        assert!(result[25] >= 0.0);
    }

    #[test]
    fn test_recovery_ratio_invalid_params() {
        let result = RecoveryRatio::new(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_volatility_risk_ratio() {
        let close = make_volatile_test_data();
        let vrr = VolatilityRiskRatio::new(20).unwrap();
        let result = vrr.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Ratio should be non-negative
        assert!(result[25] >= 0.0);
    }

    #[test]
    fn test_volatility_risk_ratio_invalid_params() {
        let result = VolatilityRiskRatio::new(2);
        assert!(result.is_err());
    }

    #[test]
    fn test_indicator_names() {
        let cdd = ConditionalDrawdown::new(20, 0.01).unwrap();
        assert_eq!(cdd.name(), "Conditional Drawdown");

        let rar = RiskAdjustedReturn::new(20, 0.02).unwrap();
        assert_eq!(rar.name(), "Risk-Adjusted Return");

        let rv = ReturnVariance::new(20, false).unwrap();
        assert_eq!(rv.name(), "Return Variance");

        let dd = DrawdownDuration::new(20).unwrap();
        assert_eq!(dd.name(), "Drawdown Duration");

        let rr = RecoveryRatio::new(20).unwrap();
        assert_eq!(rr.name(), "Recovery Ratio");

        let vrr = VolatilityRiskRatio::new(20).unwrap();
        assert_eq!(vrr.name(), "Volatility Risk Ratio");
    }

    #[test]
    fn test_min_periods() {
        let cdd = ConditionalDrawdown::new(20, 0.01).unwrap();
        assert_eq!(cdd.min_periods(), 21);

        let rar = RiskAdjustedReturn::new(15, 0.02).unwrap();
        assert_eq!(rar.min_periods(), 16);

        let rv = ReturnVariance::new(10, false).unwrap();
        assert_eq!(rv.min_periods(), 11);

        let dd = DrawdownDuration::new(25).unwrap();
        assert_eq!(dd.min_periods(), 26);

        let rr = RecoveryRatio::new(30).unwrap();
        assert_eq!(rr.min_periods(), 31);

        let vrr = VolatilityRiskRatio::new(12).unwrap();
        assert_eq!(vrr.min_periods(), 13);
    }
}
