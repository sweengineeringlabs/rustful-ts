//! Advanced Statistical Indicators
//!
//! Advanced statistical measures for market analysis.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Rolling Covariance - Rolling covariance between price and lagged price
#[derive(Debug, Clone)]
pub struct RollingCovariance {
    period: usize,
    lag: usize,
}

impl RollingCovariance {
    pub fn new(period: usize, lag: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if lag < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "lag".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if lag >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "lag".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        Ok(Self { period, lag })
    }

    /// Calculate rolling covariance between price and lagged price
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let required = self.period + self.lag;
        let mut result = vec![0.0; n];

        if n < required {
            return result;
        }

        for i in (required - 1)..n {
            let start = i + 1 - self.period;
            let series_current = &close[start..=i];
            let series_lagged = &close[(start - self.lag)..=(i - self.lag)];

            // Calculate means
            let mean_current: f64 = series_current.iter().sum::<f64>() / self.period as f64;
            let mean_lagged: f64 = series_lagged.iter().sum::<f64>() / self.period as f64;

            // Calculate covariance
            let covariance: f64 = series_current.iter()
                .zip(series_lagged.iter())
                .map(|(c, l)| (c - mean_current) * (l - mean_lagged))
                .sum::<f64>() / self.period as f64;

            result[i] = covariance;
        }
        result
    }
}

impl TechnicalIndicator for RollingCovariance {
    fn name(&self) -> &str {
        "Rolling Covariance"
    }

    fn min_periods(&self) -> usize {
        self.period + self.lag
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Serial Correlation - Measures autocorrelation in returns
#[derive(Debug, Clone)]
pub struct SerialCorrelation {
    period: usize,
    lag: usize,
}

impl SerialCorrelation {
    pub fn new(period: usize, lag: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if lag < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "lag".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if lag >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "lag".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        Ok(Self { period, lag })
    }

    /// Calculate serial correlation (autocorrelation) in returns
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let required = self.period + self.lag + 1;
        let mut result = vec![0.0; n];

        if n < required {
            return result;
        }

        // First calculate all returns
        let returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    close[i] / close[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        for i in required..n {
            let return_idx = i - 1; // Index in returns array
            let start = return_idx + 1 - self.period;

            if start < self.lag {
                continue;
            }

            let series_current = &returns[start..=return_idx];
            let series_lagged = &returns[(start - self.lag)..=(return_idx - self.lag)];

            // Calculate means
            let mean_current: f64 = series_current.iter().sum::<f64>() / self.period as f64;
            let mean_lagged: f64 = series_lagged.iter().sum::<f64>() / self.period as f64;

            // Calculate covariance and variances
            let mut cov = 0.0;
            let mut var_current = 0.0;
            let mut var_lagged = 0.0;

            for (c, l) in series_current.iter().zip(series_lagged.iter()) {
                let dc = c - mean_current;
                let dl = l - mean_lagged;
                cov += dc * dl;
                var_current += dc * dc;
                var_lagged += dl * dl;
            }

            let denominator = (var_current * var_lagged).sqrt();
            if denominator > 1e-10 {
                result[i] = cov / denominator;
            }
        }
        result
    }
}

impl TechnicalIndicator for SerialCorrelation {
    fn name(&self) -> &str {
        "Serial Correlation"
    }

    fn min_periods(&self) -> usize {
        self.period + self.lag + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Runs Test - Statistical runs test for randomness
///
/// Counts the number of runs (consecutive sequences of same sign) in returns
/// and compares to expected value under randomness.
#[derive(Debug, Clone)]
pub struct RunsTest {
    period: usize,
}

impl RunsTest {
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate runs test Z-score
    ///
    /// Returns Z-score: positive = more runs than expected (mean reverting)
    ///                  negative = fewer runs than expected (trending)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate all returns first
        let returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    close[i] / close[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        for i in self.period..n {
            let return_idx = i - 1;
            let start = return_idx + 1 - self.period;
            let window = &returns[start..=return_idx];

            // Count positive and negative returns
            let n_pos = window.iter().filter(|&&r| r > 0.0).count();
            let n_neg = window.iter().filter(|&&r| r < 0.0).count();
            let n_total = n_pos + n_neg;

            if n_pos == 0 || n_neg == 0 || n_total < 10 {
                continue;
            }

            // Count runs (consecutive sequences of same sign)
            let mut runs = 1;
            let signs: Vec<bool> = window.iter()
                .filter(|&&r| r != 0.0)
                .map(|&r| r > 0.0)
                .collect();

            for j in 1..signs.len() {
                if signs[j] != signs[j - 1] {
                    runs += 1;
                }
            }

            // Expected number of runs and standard deviation
            let n_pos_f = n_pos as f64;
            let n_neg_f = n_neg as f64;
            let n_total_f = n_total as f64;

            let expected_runs = (2.0 * n_pos_f * n_neg_f) / n_total_f + 1.0;
            let variance = (2.0 * n_pos_f * n_neg_f * (2.0 * n_pos_f * n_neg_f - n_total_f))
                / (n_total_f * n_total_f * (n_total_f - 1.0));

            if variance > 0.0 {
                let std_dev = variance.sqrt();
                result[i] = (runs as f64 - expected_runs) / std_dev;
            }
        }
        result
    }
}

impl TechnicalIndicator for RunsTest {
    fn name(&self) -> &str {
        "Runs Test"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Mean Reversion Strength - Measures mean reversion tendency
///
/// Uses variance ratio test to detect mean reversion:
/// - Value < 1: Mean reverting behavior
/// - Value = 1: Random walk
/// - Value > 1: Trending/momentum behavior
#[derive(Debug, Clone)]
pub struct MeanReversionStrength {
    period: usize,
    ratio_period: usize,
}

impl MeanReversionStrength {
    pub fn new(period: usize, ratio_period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if ratio_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "ratio_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if ratio_period > period / 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "ratio_period".to_string(),
                reason: "must be at most period/2".to_string(),
            });
        }
        Ok(Self { period, ratio_period })
    }

    /// Calculate variance ratio (mean reversion strength)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate all log returns
        let log_returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    (close[i] / close[i - 1]).ln()
                } else {
                    0.0
                }
            })
            .collect();

        for i in self.period..n {
            let return_idx = i - 1;
            let start = return_idx + 1 - self.period;
            let window = &log_returns[start..=return_idx];

            // Calculate variance of 1-period returns
            let mean_1: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let var_1: f64 = window.iter()
                .map(|r| (r - mean_1).powi(2))
                .sum::<f64>() / window.len() as f64;

            if var_1 < 1e-10 {
                continue;
            }

            // Calculate variance of k-period returns
            let k = self.ratio_period;
            let mut k_returns = Vec::new();
            for j in (k - 1)..window.len() {
                let k_return: f64 = window[(j + 1 - k)..=j].iter().sum();
                k_returns.push(k_return);
            }

            if k_returns.is_empty() {
                continue;
            }

            let mean_k: f64 = k_returns.iter().sum::<f64>() / k_returns.len() as f64;
            let var_k: f64 = k_returns.iter()
                .map(|r| (r - mean_k).powi(2))
                .sum::<f64>() / k_returns.len() as f64;

            // Variance ratio: Var(k-period) / (k * Var(1-period))
            // Under random walk hypothesis, this should equal 1
            let variance_ratio = var_k / (k as f64 * var_1);
            result[i] = variance_ratio;
        }
        result
    }
}

impl TechnicalIndicator for MeanReversionStrength {
    fn name(&self) -> &str {
        "Mean Reversion Strength"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Distribution Moments Output - Contains all four moments
#[derive(Debug, Clone)]
pub struct DistributionMomentsOutput {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub skewness: Vec<f64>,
    pub kurtosis: Vec<f64>,
}

/// Distribution Moments - Higher moments of return distribution
///
/// Calculates the first four moments of the return distribution:
/// - Mean (first moment)
/// - Variance (second central moment)
/// - Skewness (third standardized moment)
/// - Kurtosis (fourth standardized moment, excess kurtosis)
#[derive(Debug, Clone)]
pub struct DistributionMoments {
    period: usize,
}

impl DistributionMoments {
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate all four distribution moments
    pub fn calculate(&self, close: &[f64]) -> DistributionMomentsOutput {
        let n = close.len();
        let mut mean_out = vec![0.0; n];
        let mut variance_out = vec![0.0; n];
        let mut skewness_out = vec![0.0; n];
        let mut kurtosis_out = vec![0.0; n];

        if n < self.period + 1 {
            return DistributionMomentsOutput {
                mean: mean_out,
                variance: variance_out,
                skewness: skewness_out,
                kurtosis: kurtosis_out,
            };
        }

        // Calculate all returns
        let returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    close[i] / close[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        for i in self.period..n {
            let return_idx = i - 1;
            let start = return_idx + 1 - self.period;
            let window = &returns[start..=return_idx];
            let n_w = window.len() as f64;

            // First moment: Mean
            let mean: f64 = window.iter().sum::<f64>() / n_w;
            mean_out[i] = mean * 100.0; // As percentage

            // Second central moment: Variance
            let m2: f64 = window.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / n_w;
            variance_out[i] = m2 * 10000.0; // Scale to basis points squared

            if m2 < 1e-10 {
                continue;
            }

            let std_dev = m2.sqrt();

            // Third standardized moment: Skewness
            let m3: f64 = window.iter()
                .map(|r| (r - mean).powi(3))
                .sum::<f64>() / n_w;
            skewness_out[i] = m3 / std_dev.powi(3);

            // Fourth standardized moment: Excess Kurtosis
            let m4: f64 = window.iter()
                .map(|r| (r - mean).powi(4))
                .sum::<f64>() / n_w;
            kurtosis_out[i] = m4 / m2.powi(2) - 3.0;
        }

        DistributionMomentsOutput {
            mean: mean_out,
            variance: variance_out,
            skewness: skewness_out,
            kurtosis: kurtosis_out,
        }
    }
}

impl TechnicalIndicator for DistributionMoments {
    fn name(&self) -> &str {
        "Distribution Moments"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let moments = self.calculate(&data.close);
        // Returns: primary=skewness, secondary=kurtosis, tertiary=variance
        // (mean is less useful as higher moment, available separately)
        Ok(IndicatorOutput::triple(moments.skewness, moments.kurtosis, moments.variance))
    }
}

/// Outlier Detector - Statistical outlier detection
///
/// Detects outliers in returns using multiple methods:
/// - Z-score method (> threshold standard deviations from mean)
/// - IQR method (outside 1.5 * IQR from quartiles)
/// - Modified Z-score using MAD (Median Absolute Deviation)
#[derive(Debug, Clone)]
pub struct OutlierDetector {
    period: usize,
    z_threshold: f64,
}

impl OutlierDetector {
    pub fn new(period: usize, z_threshold: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if z_threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "z_threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, z_threshold })
    }

    /// Calculate outlier score
    ///
    /// Returns a composite outlier score:
    /// - 0: Not an outlier
    /// - 1-3: Outlier detected by 1-3 methods
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate all returns
        let returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    close[i] / close[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        for i in self.period..n {
            let return_idx = i - 1;
            let start = return_idx + 1 - self.period;
            let window: Vec<f64> = returns[start..=return_idx].to_vec();
            let current_return = returns[return_idx];
            let n_w = window.len() as f64;

            let mut outlier_score = 0.0;

            // Method 1: Z-score
            let mean: f64 = window.iter().sum::<f64>() / n_w;
            let variance: f64 = window.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / n_w;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                let z_score = (current_return - mean).abs() / std_dev;
                if z_score > self.z_threshold {
                    outlier_score += 1.0;
                }
            }

            // Method 2: IQR
            let mut sorted = window.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let q1_idx = (sorted.len() as f64 * 0.25) as usize;
            let q3_idx = (sorted.len() as f64 * 0.75) as usize;
            let q1 = sorted[q1_idx.min(sorted.len() - 1)];
            let q3 = sorted[q3_idx.min(sorted.len() - 1)];
            let iqr = q3 - q1;

            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;

            if current_return < lower_bound || current_return > upper_bound {
                outlier_score += 1.0;
            }

            // Method 3: Modified Z-score using MAD
            let median_idx = sorted.len() / 2;
            let median = if sorted.len() % 2 == 0 {
                (sorted[median_idx - 1] + sorted[median_idx]) / 2.0
            } else {
                sorted[median_idx]
            };

            let mut abs_deviations: Vec<f64> = window.iter()
                .map(|r| (r - median).abs())
                .collect();
            abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mad_idx = abs_deviations.len() / 2;
            let mad = if abs_deviations.len() % 2 == 0 {
                (abs_deviations[mad_idx - 1] + abs_deviations[mad_idx]) / 2.0
            } else {
                abs_deviations[mad_idx]
            };

            // Modified Z-score: 0.6745 is the consistency constant for MAD
            if mad > 1e-10 {
                let modified_z = 0.6745 * (current_return - median).abs() / mad;
                if modified_z > self.z_threshold {
                    outlier_score += 1.0;
                }
            }

            result[i] = outlier_score;
        }
        result
    }
}

impl TechnicalIndicator for OutlierDetector {
    fn name(&self) -> &str {
        "Outlier Detector"
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

    fn make_ohlcv_series(close: Vec<f64>) -> OHLCVSeries {
        let n = close.len();
        OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|c| c * 1.01).collect(),
            low: close.iter().map(|c| c * 0.99).collect(),
            close,
            volume: vec![1000.0; n],
        }
    }

    #[test]
    fn test_rolling_covariance() {
        let close = make_test_data();
        let rc = RollingCovariance::new(10, 1).unwrap();
        let result = rc.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Trending data should have positive covariance with lagged version
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_rolling_covariance_validation() {
        assert!(RollingCovariance::new(3, 1).is_err()); // period too small
        assert!(RollingCovariance::new(10, 0).is_err()); // lag too small
        assert!(RollingCovariance::new(10, 10).is_err()); // lag >= period
    }

    #[test]
    fn test_rolling_covariance_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let rc = RollingCovariance::new(10, 1).unwrap();
        let output = rc.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_serial_correlation() {
        let close = make_test_data();
        let sc = SerialCorrelation::new(10, 1).unwrap();
        let result = sc.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be between -1 and 1
        for &val in result.iter().skip(12) {
            if val != 0.0 {
                assert!(val >= -1.0 && val <= 1.0);
            }
        }
    }

    #[test]
    fn test_serial_correlation_validation() {
        assert!(SerialCorrelation::new(5, 1).is_err()); // period too small
        assert!(SerialCorrelation::new(10, 0).is_err()); // lag too small
        assert!(SerialCorrelation::new(10, 10).is_err()); // lag >= period
    }

    #[test]
    fn test_serial_correlation_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let sc = SerialCorrelation::new(10, 1).unwrap();
        let output = sc.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_runs_test() {
        let close = make_test_data();
        let rt = RunsTest::new(20).unwrap();
        let result = rt.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_runs_test_validation() {
        assert!(RunsTest::new(10).is_err()); // period too small
    }

    #[test]
    fn test_runs_test_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let rt = RunsTest::new(20).unwrap();
        let output = rt.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_mean_reversion_strength() {
        let close = make_test_data();
        let mrs = MeanReversionStrength::new(20, 5).unwrap();
        let result = mrs.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be positive
        for &val in result.iter().skip(21) {
            if val != 0.0 {
                assert!(val > 0.0);
            }
        }
    }

    #[test]
    fn test_mean_reversion_strength_validation() {
        assert!(MeanReversionStrength::new(10, 2).is_err()); // period too small
        assert!(MeanReversionStrength::new(20, 1).is_err()); // ratio_period too small
        assert!(MeanReversionStrength::new(20, 15).is_err()); // ratio_period > period/2
    }

    #[test]
    fn test_mean_reversion_strength_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let mrs = MeanReversionStrength::new(20, 5).unwrap();
        let output = mrs.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_distribution_moments() {
        let close = make_test_data();
        let dm = DistributionMoments::new(20).unwrap();
        let result = dm.calculate(&close);

        assert_eq!(result.mean.len(), close.len());
        assert_eq!(result.variance.len(), close.len());
        assert_eq!(result.skewness.len(), close.len());
        assert_eq!(result.kurtosis.len(), close.len());

        // Variance should be non-negative
        for &val in result.variance.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_distribution_moments_validation() {
        assert!(DistributionMoments::new(10).is_err()); // period too small
    }

    #[test]
    fn test_distribution_moments_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let dm = DistributionMoments::new(20).unwrap();
        let output = dm.compute(&data).unwrap();

        // Should have triple output: skewness (primary), kurtosis (secondary), variance (tertiary)
        assert!(!output.primary.is_empty());
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_outlier_detector() {
        let mut close = make_test_data();
        // Add an outlier
        close[25] = close[24] * 1.10; // 10% jump

        let od = OutlierDetector::new(20, 2.5).unwrap();
        let result = od.calculate(&close);

        assert_eq!(result.len(), close.len());
        // The outlier should be detected
        assert!(result[25] >= 1.0);
    }

    #[test]
    fn test_outlier_detector_validation() {
        assert!(OutlierDetector::new(10, 2.5).is_err()); // period too small
        assert!(OutlierDetector::new(20, 0.0).is_err()); // z_threshold not positive
        assert!(OutlierDetector::new(20, -1.0).is_err()); // z_threshold negative
    }

    #[test]
    fn test_outlier_detector_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let od = OutlierDetector::new(20, 2.5).unwrap();
        let output = od.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_outlier_detector_no_outliers() {
        let close = make_test_data();
        let od = OutlierDetector::new(20, 2.5).unwrap();
        let result = od.calculate(&close);

        // For smooth trending data, most points should not be outliers
        let outlier_count: usize = result.iter().filter(|&&v| v >= 1.0).count();
        assert!(outlier_count < close.len() / 2);
    }
}
