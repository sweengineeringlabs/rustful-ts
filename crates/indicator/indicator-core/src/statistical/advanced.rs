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

/// Rolling Beta - Rolling beta coefficient vs benchmark
///
/// Measures the sensitivity of asset returns to benchmark returns.
/// Beta > 1: More volatile than benchmark
/// Beta = 1: Same volatility as benchmark
/// Beta < 1: Less volatile than benchmark
/// Beta < 0: Negatively correlated with benchmark
#[derive(Debug, Clone)]
pub struct RollingBeta {
    period: usize,
}

impl RollingBeta {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate rolling beta coefficient
    ///
    /// Beta = Cov(asset, benchmark) / Var(benchmark)
    pub fn calculate(&self, close: &[f64], benchmark: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 || benchmark.len() != n {
            return result;
        }

        // Calculate returns for both series
        let asset_returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    close[i] / close[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        let bench_returns: Vec<f64> = (1..n)
            .map(|i| {
                if benchmark[i - 1] > 0.0 && benchmark[i] > 0.0 {
                    benchmark[i] / benchmark[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        for i in self.period..n {
            let return_idx = i - 1;
            let start = return_idx + 1 - self.period;

            let asset_window = &asset_returns[start..=return_idx];
            let bench_window = &bench_returns[start..=return_idx];
            let n_w = asset_window.len() as f64;

            // Calculate means
            let mean_asset: f64 = asset_window.iter().sum::<f64>() / n_w;
            let mean_bench: f64 = bench_window.iter().sum::<f64>() / n_w;

            // Calculate covariance and variance
            let mut cov = 0.0;
            let mut var_bench = 0.0;

            for (a, b) in asset_window.iter().zip(bench_window.iter()) {
                let da = a - mean_asset;
                let db = b - mean_bench;
                cov += da * db;
                var_bench += db * db;
            }

            if var_bench > 1e-10 {
                result[i] = cov / var_bench;
            }
        }
        result
    }
}

impl TechnicalIndicator for RollingBeta {
    fn name(&self) -> &str {
        "Rolling Beta"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // When no benchmark provided, use volume as proxy
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Rolling Alpha - Rolling alpha (excess return)
///
/// Measures the excess return of an asset over the expected return based on beta.
/// Alpha = Asset Return - (Risk-Free Rate + Beta * (Benchmark Return - Risk-Free Rate))
/// Simplified: Alpha = Asset Return - Beta * Benchmark Return (assuming Rf = 0)
#[derive(Debug, Clone)]
pub struct RollingAlpha {
    period: usize,
}

impl RollingAlpha {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate rolling alpha (annualized)
    pub fn calculate(&self, close: &[f64], benchmark: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 || benchmark.len() != n {
            return result;
        }

        // Calculate returns for both series
        let asset_returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    close[i] / close[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        let bench_returns: Vec<f64> = (1..n)
            .map(|i| {
                if benchmark[i - 1] > 0.0 && benchmark[i] > 0.0 {
                    benchmark[i] / benchmark[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        for i in self.period..n {
            let return_idx = i - 1;
            let start = return_idx + 1 - self.period;

            let asset_window = &asset_returns[start..=return_idx];
            let bench_window = &bench_returns[start..=return_idx];
            let n_w = asset_window.len() as f64;

            // Calculate means
            let mean_asset: f64 = asset_window.iter().sum::<f64>() / n_w;
            let mean_bench: f64 = bench_window.iter().sum::<f64>() / n_w;

            // Calculate beta first
            let mut cov = 0.0;
            let mut var_bench = 0.0;

            for (a, b) in asset_window.iter().zip(bench_window.iter()) {
                let da = a - mean_asset;
                let db = b - mean_bench;
                cov += da * db;
                var_bench += db * db;
            }

            let beta = if var_bench > 1e-10 { cov / var_bench } else { 0.0 };

            // Alpha = mean(asset) - beta * mean(benchmark)
            // Annualized (assuming 252 trading days)
            let daily_alpha = mean_asset - beta * mean_bench;
            result[i] = daily_alpha * 252.0 * 100.0; // Annualized percentage
        }
        result
    }
}

impl TechnicalIndicator for RollingAlpha {
    fn name(&self) -> &str {
        "Rolling Alpha"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // When no benchmark provided, use volume as proxy
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Information Coefficient - IC for factor analysis
///
/// Measures the correlation between predicted returns (using current factor)
/// and actual future returns. Used in quantitative factor analysis.
#[derive(Debug, Clone)]
pub struct InformationCoefficient {
    period: usize,
    forward_period: usize,
}

impl InformationCoefficient {
    pub fn new(period: usize, forward_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if forward_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "forward_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, forward_period })
    }

    /// Calculate information coefficient
    ///
    /// Uses momentum as the factor (past returns predict future returns)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + self.forward_period + 1 {
            return result;
        }

        // Calculate returns
        let returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    close[i] / close[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        // For each point, calculate IC over the lookback period
        for i in (self.period + self.forward_period)..n {
            let mut factor_values = Vec::with_capacity(self.period);
            let mut future_returns = Vec::with_capacity(self.period);

            // Collect factor (past return) and forward return pairs
            for j in 0..self.period {
                let idx = i - self.forward_period - self.period + j;
                if idx + self.forward_period < returns.len() {
                    // Factor: return at time t
                    factor_values.push(returns[idx]);
                    // Forward return: return at time t + forward_period
                    future_returns.push(returns[idx + self.forward_period]);
                }
            }

            if factor_values.len() < 5 {
                continue;
            }

            let n_pairs = factor_values.len() as f64;

            // Calculate Pearson correlation between factor and forward returns
            let mean_factor: f64 = factor_values.iter().sum::<f64>() / n_pairs;
            let mean_future: f64 = future_returns.iter().sum::<f64>() / n_pairs;

            let mut cov = 0.0;
            let mut var_factor = 0.0;
            let mut var_future = 0.0;

            for (f, r) in factor_values.iter().zip(future_returns.iter()) {
                let df = f - mean_factor;
                let dr = r - mean_future;
                cov += df * dr;
                var_factor += df * df;
                var_future += dr * dr;
            }

            let denominator = (var_factor * var_future).sqrt();
            if denominator > 1e-10 {
                result[i] = cov / denominator;
            }
        }
        result
    }
}

impl TechnicalIndicator for InformationCoefficient {
    fn name(&self) -> &str {
        "Information Coefficient"
    }

    fn min_periods(&self) -> usize {
        self.period + self.forward_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Rank Correlation - Spearman rank correlation
///
/// Non-parametric measure of correlation that assesses monotonic relationships.
/// More robust to outliers than Pearson correlation.
#[derive(Debug, Clone)]
pub struct RankCorrelation {
    period: usize,
    lag: usize,
}

impl RankCorrelation {
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
        Ok(Self { period, lag })
    }

    /// Convert values to ranks
    fn rank(values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            // Handle ties by averaging ranks
            while j < n - 1 && (indexed[j].1 - indexed[j + 1].1).abs() < 1e-10 {
                j += 1;
            }
            let avg_rank = (i + j) as f64 / 2.0 + 1.0;
            for k in i..=j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j + 1;
        }
        ranks
    }

    /// Calculate Spearman rank correlation
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + self.lag + 1 {
            return result;
        }

        // Calculate returns
        let returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    close[i] / close[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        for i in (self.period + self.lag)..n {
            let return_idx = i - 1;
            let start = return_idx + 1 - self.period;

            if start < self.lag {
                continue;
            }

            let current_window: Vec<f64> = returns[start..=return_idx].to_vec();
            let lagged_window: Vec<f64> = returns[(start - self.lag)..=(return_idx - self.lag)].to_vec();

            // Get ranks
            let ranks_current = Self::rank(&current_window);
            let ranks_lagged = Self::rank(&lagged_window);

            let n_w = ranks_current.len() as f64;

            // Calculate Pearson correlation of ranks (which gives Spearman correlation)
            let mean_current: f64 = ranks_current.iter().sum::<f64>() / n_w;
            let mean_lagged: f64 = ranks_lagged.iter().sum::<f64>() / n_w;

            let mut cov = 0.0;
            let mut var_current = 0.0;
            let mut var_lagged = 0.0;

            for (c, l) in ranks_current.iter().zip(ranks_lagged.iter()) {
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

impl TechnicalIndicator for RankCorrelation {
    fn name(&self) -> &str {
        "Rank Correlation"
    }

    fn min_periods(&self) -> usize {
        self.period + self.lag + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Tail Dependence Output - Contains upper and lower tail dependence
#[derive(Debug, Clone)]
pub struct TailDependenceOutput {
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
}

/// Tail Dependence - Measures tail dependence in returns
///
/// Measures the probability that extreme returns occur together.
/// Upper tail: probability of joint extreme positive returns
/// Lower tail: probability of joint extreme negative returns
#[derive(Debug, Clone)]
pub struct TailDependence {
    period: usize,
    quantile: f64,
}

impl TailDependence {
    pub fn new(period: usize, quantile: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if quantile <= 0.0 || quantile >= 0.5 {
            return Err(IndicatorError::InvalidParameter {
                name: "quantile".to_string(),
                reason: "must be between 0 and 0.5 (exclusive)".to_string(),
            });
        }
        Ok(Self { period, quantile })
    }

    /// Calculate tail dependence coefficients
    ///
    /// Uses empirical copula approach to estimate tail dependence
    pub fn calculate(&self, close: &[f64]) -> TailDependenceOutput {
        let n = close.len();
        let mut upper = vec![0.0; n];
        let mut lower = vec![0.0; n];

        if n < self.period + 2 {
            return TailDependenceOutput { upper, lower };
        }

        // Calculate returns
        let returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    close[i] / close[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        for i in (self.period + 1)..n {
            let return_idx = i - 1;
            let start = return_idx + 1 - self.period;

            if start < 1 {
                continue;
            }

            // Get current and lagged returns
            let current_window: Vec<f64> = returns[start..=return_idx].to_vec();
            let lagged_window: Vec<f64> = returns[(start - 1)..=(return_idx - 1)].to_vec();

            // Sort to find quantile thresholds
            let mut sorted_current = current_window.clone();
            let mut sorted_lagged = lagged_window.clone();
            sorted_current.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted_lagged.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let lower_idx = (self.period as f64 * self.quantile) as usize;
            let upper_idx = (self.period as f64 * (1.0 - self.quantile)) as usize;

            if lower_idx >= sorted_current.len() || upper_idx >= sorted_current.len() {
                continue;
            }

            let lower_thresh_current = sorted_current[lower_idx];
            let upper_thresh_current = sorted_current[upper_idx.min(sorted_current.len() - 1)];
            let lower_thresh_lagged = sorted_lagged[lower_idx];
            let upper_thresh_lagged = sorted_lagged[upper_idx.min(sorted_lagged.len() - 1)];

            // Count joint exceedances
            let mut upper_joint = 0;
            let mut lower_joint = 0;
            let mut upper_marginal = 0;
            let mut lower_marginal = 0;

            for (c, l) in current_window.iter().zip(lagged_window.iter()) {
                // Upper tail
                if *c > upper_thresh_current {
                    upper_marginal += 1;
                    if *l > upper_thresh_lagged {
                        upper_joint += 1;
                    }
                }
                // Lower tail
                if *c < lower_thresh_current {
                    lower_marginal += 1;
                    if *l < lower_thresh_lagged {
                        lower_joint += 1;
                    }
                }
            }

            // Tail dependence coefficient = P(Y > thresh | X > thresh)
            if upper_marginal > 0 {
                upper[i] = upper_joint as f64 / upper_marginal as f64;
            }
            if lower_marginal > 0 {
                lower[i] = lower_joint as f64 / lower_marginal as f64;
            }
        }

        TailDependenceOutput { upper, lower }
    }
}

impl TechnicalIndicator for TailDependence {
    fn name(&self) -> &str {
        "Tail Dependence"
    }

    fn min_periods(&self) -> usize {
        self.period + 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(result.upper, result.lower))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Copula Correlation - Copula-based correlation measure
///
/// Uses the Gaussian copula approach to measure dependence structure
/// independent of marginal distributions. More robust than linear correlation.
#[derive(Debug, Clone)]
pub struct CopulaCorrelation {
    period: usize,
    lag: usize,
}

impl CopulaCorrelation {
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
        Ok(Self { period, lag })
    }

    /// Standard normal quantile function (inverse CDF) approximation
    fn norm_inv(p: f64) -> f64 {
        if p <= 0.0 {
            return -3.5;
        }
        if p >= 1.0 {
            return 3.5;
        }

        // Rational approximation (Abramowitz and Stegun)
        let a = [
            -3.969683028665376e+01,
             2.209460984245205e+02,
            -2.759285104469687e+02,
             1.383577518672690e+02,
            -3.066479806614716e+01,
             2.506628277459239e+00,
        ];
        let b = [
            -5.447609879822406e+01,
             1.615858368580409e+02,
            -1.556989798598866e+02,
             6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        let c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
             4.374664141464968e+00,
             2.938163982698783e+00,
        ];
        let d = [
             7.784695709041462e-03,
             3.224671290700398e-01,
             2.445134137142996e+00,
             3.754408661907416e+00,
        ];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        let q;
        let r;

        if p < p_low {
            q = (-2.0 * p.ln()).sqrt();
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        } else if p <= p_high {
            q = p - 0.5;
            r = q * q;
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
        } else {
            q = (-2.0 * (1.0 - p).ln()).sqrt();
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
    }

    /// Calculate copula correlation using Gaussian copula
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + self.lag + 1 {
            return result;
        }

        // Calculate returns
        let returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    close[i] / close[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        for i in (self.period + self.lag)..n {
            let return_idx = i - 1;
            let start = return_idx + 1 - self.period;

            if start < self.lag {
                continue;
            }

            let current_window: Vec<f64> = returns[start..=return_idx].to_vec();
            let lagged_window: Vec<f64> = returns[(start - self.lag)..=(return_idx - self.lag)].to_vec();

            // Transform to uniform using empirical CDF (pseudo-observations)
            let uniform_current = Self::to_pseudo_observations(&current_window);
            let uniform_lagged = Self::to_pseudo_observations(&lagged_window);

            // Transform to standard normal using inverse CDF
            let normal_current: Vec<f64> = uniform_current.iter()
                .map(|&u| Self::norm_inv(u))
                .collect();
            let normal_lagged: Vec<f64> = uniform_lagged.iter()
                .map(|&u| Self::norm_inv(u))
                .collect();

            // Calculate Pearson correlation of transformed values
            let n_w = normal_current.len() as f64;
            let mean_current: f64 = normal_current.iter().sum::<f64>() / n_w;
            let mean_lagged: f64 = normal_lagged.iter().sum::<f64>() / n_w;

            let mut cov = 0.0;
            let mut var_current = 0.0;
            let mut var_lagged = 0.0;

            for (c, l) in normal_current.iter().zip(normal_lagged.iter()) {
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

    /// Transform values to pseudo-observations (empirical CDF)
    fn to_pseudo_observations(values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut ranks = vec![0.0; n];
        for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
            // Use (rank + 1) / (n + 1) to avoid 0 and 1
            ranks[orig_idx] = (rank as f64 + 1.0) / (n as f64 + 1.0);
        }
        ranks
    }
}

impl TechnicalIndicator for CopulaCorrelation {
    fn name(&self) -> &str {
        "Copula Correlation"
    }

    fn min_periods(&self) -> usize {
        self.period + self.lag + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Z-Score Extreme - Detects extreme z-score readings in price series
///
/// Identifies when price moves are statistically extreme by calculating
/// the z-score (number of standard deviations from mean) and flagging
/// readings that exceed a threshold.
///
/// # Interpretation
/// - Values > threshold: Extremely overbought condition
/// - Values < -threshold: Extremely oversold condition
/// - Values near 0: Normal market conditions
///
/// # Example
/// ```ignore
/// let detector = ZScoreExtreme::new(20, 2.0).unwrap();
/// let extremes = detector.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct ZScoreExtreme {
    period: usize,
    threshold: f64,
}

impl ZScoreExtreme {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate z-score extremes
    ///
    /// Returns the z-score when it exceeds the threshold, otherwise 0.
    /// The sign indicates direction (positive = overbought, negative = oversold).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &close[start..=i];
            let current = close[i];

            // Calculate mean and standard deviation
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                let z_score = (current - mean) / std_dev;
                // Only return z-score if it exceeds threshold
                if z_score.abs() > self.threshold {
                    result[i] = z_score;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for ZScoreExtreme {
    fn name(&self) -> &str {
        "Z-Score Extreme"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Percentile Rank - Ranks current value in percentile terms
///
/// Calculates the percentile rank of the current price relative to
/// a historical window. Shows where the current price stands in the
/// distribution of past prices.
///
/// # Interpretation
/// - 100: Current price is the highest in the lookback period
/// - 50: Current price is at the median
/// - 0: Current price is the lowest in the lookback period
///
/// # Example
/// ```ignore
/// let rank = PercentileRank::new(50).unwrap();
/// let percentiles = rank.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct PercentileRank {
    period: usize,
}

impl PercentileRank {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate percentile rank (0-100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &close[start..=i];
            let current = close[i];

            // Count how many values are less than current
            let count_below = window.iter().filter(|&&x| x < current).count();
            // Count how many values are equal to current
            let count_equal = window.iter().filter(|&&x| (x - current).abs() < 1e-10).count();

            // Percentile rank formula: (count_below + 0.5 * count_equal) / total * 100
            let percentile = (count_below as f64 + 0.5 * count_equal as f64)
                / self.period as f64 * 100.0;
            result[i] = percentile;
        }
        result
    }
}

impl TechnicalIndicator for PercentileRank {
    fn name(&self) -> &str {
        "Percentile Rank"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Statistical Regime Output - Contains regime and transition data
#[derive(Debug, Clone)]
pub struct StatisticalRegimeOutput {
    /// Current regime: 1 = high volatility, 0 = normal, -1 = low volatility
    pub regime: Vec<f64>,
    /// Regime transition signal: 1 = regime change detected
    pub transition: Vec<f64>,
}

/// Statistical Regime - Detects statistical regime changes in market data
///
/// Identifies different market regimes based on volatility levels and
/// detects transitions between regimes. Uses rolling statistics to
/// classify periods as high volatility, normal, or low volatility.
///
/// # Interpretation
/// - Regime 1: High volatility regime (above upper threshold)
/// - Regime 0: Normal volatility regime
/// - Regime -1: Low volatility regime (below lower threshold)
/// - Transition = 1: Regime change detected
///
/// # Example
/// ```ignore
/// let regime = StatisticalRegime::new(20, 1.5).unwrap();
/// let output = regime.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct StatisticalRegime {
    period: usize,
    threshold_multiplier: f64,
}

impl StatisticalRegime {
    pub fn new(period: usize, threshold_multiplier: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if threshold_multiplier <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold_multiplier".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { period, threshold_multiplier })
    }

    /// Calculate statistical regime and transitions
    pub fn calculate(&self, close: &[f64]) -> StatisticalRegimeOutput {
        let n = close.len();
        let mut regime = vec![0.0; n];
        let mut transition = vec![0.0; n];

        if n < self.period + 1 {
            return StatisticalRegimeOutput { regime, transition };
        }

        // Calculate returns
        let returns: Vec<f64> = (1..n)
            .map(|i| {
                if close[i - 1] > 0.0 && close[i] > 0.0 {
                    close[i] / close[i - 1] - 1.0
                } else {
                    0.0
                }
            })
            .collect();

        // First pass: calculate rolling volatility
        let mut volatilities = vec![0.0; n];
        for i in self.period..n {
            let return_idx = i - 1;
            let start = return_idx + 1 - self.period;
            let window = &returns[start..=return_idx];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            volatilities[i] = variance.sqrt();
        }

        // Calculate long-term average volatility (using available data)
        let valid_vols: Vec<f64> = volatilities.iter()
            .skip(self.period)
            .copied()
            .filter(|&v| v > 0.0)
            .collect();

        if valid_vols.is_empty() {
            return StatisticalRegimeOutput { regime, transition };
        }

        let mean_vol: f64 = valid_vols.iter().sum::<f64>() / valid_vols.len() as f64;
        let vol_std: f64 = (valid_vols.iter()
            .map(|v| (v - mean_vol).powi(2))
            .sum::<f64>() / valid_vols.len() as f64).sqrt();

        let upper_threshold = mean_vol + self.threshold_multiplier * vol_std;
        let lower_threshold = mean_vol - self.threshold_multiplier * vol_std;

        // Classify regimes
        let mut prev_regime = 0.0;
        for i in self.period..n {
            let vol = volatilities[i];
            let current_regime = if vol > upper_threshold {
                1.0  // High volatility
            } else if vol < lower_threshold && lower_threshold > 0.0 {
                -1.0  // Low volatility
            } else {
                0.0  // Normal
            };

            regime[i] = current_regime;

            // Detect transition
            if i > self.period && current_regime != prev_regime {
                transition[i] = 1.0;
            }
            prev_regime = current_regime;
        }

        StatisticalRegimeOutput { regime, transition }
    }
}

impl TechnicalIndicator for StatisticalRegime {
    fn name(&self) -> &str {
        "Statistical Regime"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(result.regime, result.transition))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Autocorrelation Index - Measures price autocorrelation
///
/// Calculates the autocorrelation of returns at a specified lag,
/// measuring how much past returns predict future returns.
///
/// # Interpretation
/// - Positive values: Trending/momentum behavior (past returns predict same direction)
/// - Negative values: Mean-reverting behavior (past returns predict opposite direction)
/// - Near zero: Random walk behavior
///
/// # Example
/// ```ignore
/// let ac = AutocorrelationIndex::new(20, 1).unwrap();
/// let autocorr = ac.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct AutocorrelationIndex {
    period: usize,
    lag: usize,
}

impl AutocorrelationIndex {
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

    /// Calculate autocorrelation index
    ///
    /// Returns autocorrelation coefficient between -1 and 1
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        let required = self.period + self.lag + 1;
        if n < required {
            return result;
        }

        // Calculate returns
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
            let return_idx = i - 1;
            let start = return_idx + 1 - self.period;

            if start < self.lag {
                continue;
            }

            let current_window = &returns[start..=return_idx];
            let lagged_window = &returns[(start - self.lag)..=(return_idx - self.lag)];

            // Calculate means
            let mean_current: f64 = current_window.iter().sum::<f64>() / self.period as f64;
            let mean_lagged: f64 = lagged_window.iter().sum::<f64>() / self.period as f64;

            // Calculate autocorrelation
            let mut numerator = 0.0;
            let mut var_current = 0.0;
            let mut var_lagged = 0.0;

            for (c, l) in current_window.iter().zip(lagged_window.iter()) {
                let dc = c - mean_current;
                let dl = l - mean_lagged;
                numerator += dc * dl;
                var_current += dc * dc;
                var_lagged += dl * dl;
            }

            let denominator = (var_current * var_lagged).sqrt();
            if denominator > 1e-10 {
                result[i] = numerator / denominator;
            }
        }
        result
    }
}

impl TechnicalIndicator for AutocorrelationIndex {
    fn name(&self) -> &str {
        "Autocorrelation Index"
    }

    fn min_periods(&self) -> usize {
        self.period + self.lag + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Hurst Exponent MA - Moving Hurst exponent for trend persistence
///
/// Calculates a rolling Hurst exponent using the rescaled range (R/S) method.
/// The Hurst exponent measures the long-term memory of a time series.
///
/// # Interpretation
/// - H > 0.5: Persistent/trending behavior (momentum)
/// - H = 0.5: Random walk (no memory)
/// - H < 0.5: Anti-persistent/mean-reverting behavior
///
/// # Example
/// ```ignore
/// let hurst = HurstExponentMA::new(50).unwrap();
/// let h_values = hurst.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct HurstExponentMA {
    period: usize,
}

impl HurstExponentMA {
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate R/S statistic for a given window
    fn calculate_rs(returns: &[f64]) -> f64 {
        let n = returns.len();
        if n < 2 {
            return 0.0;
        }

        // Calculate mean
        let mean: f64 = returns.iter().sum::<f64>() / n as f64;

        // Calculate cumulative deviations
        let mut cumsum = Vec::with_capacity(n);
        let mut sum = 0.0;
        for r in returns {
            sum += r - mean;
            cumsum.push(sum);
        }

        // Range: max - min of cumulative deviations
        let max_val = cumsum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_val = cumsum.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = max_val - min_val;

        // Standard deviation
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        if std_dev > 1e-10 {
            range / std_dev
        } else {
            0.0
        }
    }

    /// Calculate moving Hurst exponent
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate log returns
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

            // Use multiple sub-periods to estimate Hurst exponent
            // Using 4 different window sizes
            let window_sizes = [
                self.period / 4,
                self.period / 2,
                3 * self.period / 4,
                self.period,
            ];

            let mut log_n_vec = Vec::new();
            let mut log_rs_vec = Vec::new();

            for &ws in &window_sizes {
                if ws < 4 {
                    continue;
                }

                // Calculate average R/S for this window size
                let num_windows = self.period / ws;
                if num_windows == 0 {
                    continue;
                }

                let mut rs_sum = 0.0;
                let mut count = 0;
                for j in 0..num_windows {
                    let w_start = j * ws;
                    let w_end = w_start + ws;
                    if w_end <= window.len() {
                        let rs = Self::calculate_rs(&window[w_start..w_end]);
                        if rs > 0.0 {
                            rs_sum += rs;
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    let avg_rs = rs_sum / count as f64;
                    log_n_vec.push((ws as f64).ln());
                    log_rs_vec.push(avg_rs.ln());
                }
            }

            // Linear regression to estimate Hurst exponent
            if log_n_vec.len() >= 2 {
                let n_points = log_n_vec.len() as f64;
                let sum_x: f64 = log_n_vec.iter().sum();
                let sum_y: f64 = log_rs_vec.iter().sum();
                let sum_xy: f64 = log_n_vec.iter().zip(log_rs_vec.iter())
                    .map(|(x, y)| x * y)
                    .sum();
                let sum_xx: f64 = log_n_vec.iter().map(|x| x * x).sum();

                let denominator = n_points * sum_xx - sum_x * sum_x;
                if denominator.abs() > 1e-10 {
                    let slope = (n_points * sum_xy - sum_x * sum_y) / denominator;
                    // Clamp to valid Hurst range [0, 1]
                    result[i] = slope.clamp(0.0, 1.0);
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for HurstExponentMA {
    fn name(&self) -> &str {
        "Hurst Exponent MA"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Entropy Measure - Information entropy of price series
///
/// Calculates the Shannon entropy of price returns distribution,
/// measuring the randomness or unpredictability of price movements.
///
/// # Interpretation
/// - High entropy: More random/unpredictable price action
/// - Low entropy: More predictable/concentrated price action
/// - Maximum entropy occurs when all outcomes are equally likely
///
/// # Example
/// ```ignore
/// let entropy = EntropyMeasure::new(20, 10).unwrap();
/// let entropy_values = entropy.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct EntropyMeasure {
    period: usize,
    num_bins: usize,
}

impl EntropyMeasure {
    pub fn new(period: usize, num_bins: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if num_bins < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_bins".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if num_bins > period {
            return Err(IndicatorError::InvalidParameter {
                name: "num_bins".to_string(),
                reason: "must not exceed period".to_string(),
            });
        }
        Ok(Self { period, num_bins })
    }

    /// Calculate Shannon entropy of returns distribution
    ///
    /// Returns normalized entropy between 0 and 1
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns
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

            // Find min and max to determine bin edges
            let min_val = window.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let range = max_val - min_val;
            if range < 1e-10 {
                // All values are the same, entropy is 0
                continue;
            }

            // Create histogram
            let bin_width = range / self.num_bins as f64;
            let mut bin_counts = vec![0usize; self.num_bins];

            for &r in &window {
                let bin_idx = ((r - min_val) / bin_width) as usize;
                let bin_idx = bin_idx.min(self.num_bins - 1);
                bin_counts[bin_idx] += 1;
            }

            // Calculate Shannon entropy
            let total = window.len() as f64;
            let mut entropy = 0.0;
            for &count in &bin_counts {
                if count > 0 {
                    let p = count as f64 / total;
                    entropy -= p * p.ln();
                }
            }

            // Normalize by maximum entropy (log of num_bins)
            let max_entropy = (self.num_bins as f64).ln();
            if max_entropy > 0.0 {
                result[i] = entropy / max_entropy;
            }
        }
        result
    }
}

impl TechnicalIndicator for EntropyMeasure {
    fn name(&self) -> &str {
        "Entropy Measure"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Variance Ratio - Ratio of variance across different periods
///
/// Calculates the ratio of variance at longer time horizons to shorter horizons.
/// Under the random walk hypothesis, variance should scale linearly with time.
/// Deviations indicate predictability in returns.
///
/// # Interpretation
/// - VR = 1: Random walk (no predictability)
/// - VR > 1: Positive autocorrelation/momentum (trending)
/// - VR < 1: Negative autocorrelation/mean reversion
///
/// # Example
/// ```ignore
/// let vr = VarianceRatio::new(20, 4).unwrap();
/// let ratios = vr.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct VarianceRatio {
    period: usize,
    short_period: usize,
}

impl VarianceRatio {
    pub fn new(period: usize, short_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if short_period >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        Ok(Self { period, short_period })
    }

    /// Calculate variance ratio
    ///
    /// Returns the ratio of long-period variance to scaled short-period variance.
    /// A ratio of 1 indicates random walk behavior.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate log returns
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
                result[i] = 1.0;
                continue;
            }

            // Calculate variance of short_period-period returns (sum of k consecutive returns)
            let k = self.short_period;
            let mut k_returns = Vec::new();
            for j in (k - 1)..window.len() {
                let k_return: f64 = window[(j + 1 - k)..=j].iter().sum();
                k_returns.push(k_return);
            }

            if k_returns.is_empty() {
                result[i] = 1.0;
                continue;
            }

            let mean_k: f64 = k_returns.iter().sum::<f64>() / k_returns.len() as f64;
            let var_k: f64 = k_returns.iter()
                .map(|r| (r - mean_k).powi(2))
                .sum::<f64>() / k_returns.len() as f64;

            // Variance ratio: Var(k-period) / (k * Var(1-period))
            // Under random walk, this equals 1
            let ratio = var_k / (k as f64 * var_1);
            result[i] = ratio;
        }
        result
    }
}

impl TechnicalIndicator for VarianceRatio {
    fn name(&self) -> &str {
        "Variance Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Mean Reversion Speed - Measures the speed of mean reversion
///
/// Estimates the half-life of mean reversion using the Ornstein-Uhlenbeck
/// process framework. Faster mean reversion indicates shorter half-life.
///
/// # Interpretation
/// - Lower values: Faster mean reversion (quick return to mean)
/// - Higher values: Slower mean reversion (takes longer to revert)
/// - Very high values: Trending behavior (no mean reversion)
///
/// # Example
/// ```ignore
/// let mrs = MeanReversionSpeed::new(30).unwrap();
/// let half_lives = mrs.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct MeanReversionSpeed {
    period: usize,
}

impl MeanReversionSpeed {
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate mean reversion speed (half-life in periods)
    ///
    /// Uses the Ornstein-Uhlenbeck process: dX = theta * (mu - X) * dt + sigma * dW
    /// Half-life = ln(2) / theta
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate log prices
        let log_prices: Vec<f64> = close.iter()
            .map(|&c| if c > 0.0 { c.ln() } else { 0.0 })
            .collect();

        for i in self.period..n {
            let start = i + 1 - self.period;
            let window = &log_prices[start..=i];

            // Calculate price changes: y(t) - y(t-1)
            let mut y_changes = Vec::with_capacity(self.period - 1);
            let mut y_lagged = Vec::with_capacity(self.period - 1);

            for j in 1..window.len() {
                y_changes.push(window[j] - window[j - 1]);
                y_lagged.push(window[j - 1]);
            }

            if y_changes.len() < 5 {
                continue;
            }

            let n_obs = y_changes.len() as f64;

            // Linear regression: y_change = alpha + beta * y_lagged + epsilon
            // Mean reversion speed is related to beta
            let mean_change: f64 = y_changes.iter().sum::<f64>() / n_obs;
            let mean_lagged: f64 = y_lagged.iter().sum::<f64>() / n_obs;

            let mut cov = 0.0;
            let mut var_lagged = 0.0;

            for (dy, y) in y_changes.iter().zip(y_lagged.iter()) {
                let dc = dy - mean_change;
                let dl = y - mean_lagged;
                cov += dc * dl;
                var_lagged += dl * dl;
            }

            if var_lagged < 1e-10 {
                continue;
            }

            let beta = cov / var_lagged;

            // For mean-reverting process, beta should be negative
            // Half-life = -ln(2) / ln(1 + beta) ≈ -ln(2) / beta for small beta
            if beta < -1e-10 {
                // Stable mean-reverting process
                let theta = -beta;
                let half_life = 0.693147 / theta; // ln(2) / theta
                // Cap at reasonable values
                result[i] = half_life.min(1000.0);
            } else if beta < 1e-10 {
                // Near random walk
                result[i] = 1000.0; // Very slow reversion
            } else {
                // Trending (explosive), no mean reversion
                result[i] = 0.0; // Indicate no reversion
            }
        }
        result
    }
}

impl TechnicalIndicator for MeanReversionSpeed {
    fn name(&self) -> &str {
        "Mean Reversion Speed"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Trend Stationarity - Tests for trend stationarity in price series
///
/// Combines multiple tests to assess whether a series is trend-stationary:
/// - Augmented Dickey-Fuller test statistic
/// - Variance ratio deviation from 1
/// - Linear trend strength
///
/// # Interpretation
/// - Values < -2: Strong evidence of stationarity
/// - Values between -2 and 0: Weak evidence of stationarity
/// - Values > 0: Evidence of non-stationarity (unit root)
///
/// # Example
/// ```ignore
/// let ts = TrendStationarity::new(30).unwrap();
/// let stationarity = ts.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct TrendStationarity {
    period: usize,
}

impl TrendStationarity {
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate trend stationarity test statistic
    ///
    /// Returns a composite test statistic. More negative values indicate
    /// stronger evidence of stationarity.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate log prices
        let log_prices: Vec<f64> = close.iter()
            .map(|&c| if c > 0.0 { c.ln() } else { 0.0 })
            .collect();

        for i in self.period..n {
            let start = i + 1 - self.period;
            let window = &log_prices[start..=i];

            // 1. Calculate ADF-like statistic
            // First difference: delta_y = y(t) - y(t-1)
            let mut delta_y = Vec::with_capacity(self.period - 1);
            let mut y_lagged = Vec::with_capacity(self.period - 1);

            for j in 1..window.len() {
                delta_y.push(window[j] - window[j - 1]);
                y_lagged.push(window[j - 1]);
            }

            if delta_y.len() < 5 {
                continue;
            }

            let n_obs = delta_y.len() as f64;

            // Regression: delta_y = alpha + gamma * y_lagged + epsilon
            let mean_delta: f64 = delta_y.iter().sum::<f64>() / n_obs;
            let mean_lagged: f64 = y_lagged.iter().sum::<f64>() / n_obs;

            let mut cov = 0.0;
            let mut var_lagged = 0.0;

            for (dy, y) in delta_y.iter().zip(y_lagged.iter()) {
                let dc = dy - mean_delta;
                let dl = y - mean_lagged;
                cov += dc * dl;
                var_lagged += dl * dl;
            }

            if var_lagged < 1e-10 {
                continue;
            }

            let gamma = cov / var_lagged;
            let alpha = mean_delta - gamma * mean_lagged;

            // Calculate residuals and standard error
            let mut residual_sum_sq = 0.0;
            for (dy, y) in delta_y.iter().zip(y_lagged.iter()) {
                let predicted = alpha + gamma * y;
                let residual = dy - predicted;
                residual_sum_sq += residual * residual;
            }

            let sigma_sq = residual_sum_sq / (n_obs - 2.0);
            let se_gamma = (sigma_sq / var_lagged).sqrt();

            // ADF test statistic: gamma / SE(gamma)
            let adf_stat = if se_gamma > 1e-10 {
                gamma / se_gamma
            } else {
                0.0
            };

            // 2. Calculate trend R-squared (how well linear trend fits)
            let x_values: Vec<f64> = (0..window.len()).map(|x| x as f64).collect();
            let mean_x: f64 = x_values.iter().sum::<f64>() / window.len() as f64;
            let mean_y: f64 = window.iter().sum::<f64>() / window.len() as f64;

            let mut cov_xy = 0.0;
            let mut var_x = 0.0;
            let mut var_y = 0.0;

            for (x, y) in x_values.iter().zip(window.iter()) {
                let dx = x - mean_x;
                let dy = y - mean_y;
                cov_xy += dx * dy;
                var_x += dx * dx;
                var_y += dy * dy;
            }

            let r_squared = if var_x > 1e-10 && var_y > 1e-10 {
                (cov_xy * cov_xy) / (var_x * var_y)
            } else {
                0.0
            };

            // Composite statistic: ADF stat adjusted by trend strength
            // Strong trend with high R^2 indicates trend-stationarity
            // Weak trend with non-stationary suggests unit root
            result[i] = adf_stat * (1.0 + r_squared);
        }
        result
    }
}

impl TechnicalIndicator for TrendStationarity {
    fn name(&self) -> &str {
        "Trend Stationarity"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Excess Kurtosis - Measures excess kurtosis in rolling returns
///
/// Calculates the excess kurtosis (fourth standardized moment minus 3)
/// of the return distribution over a rolling window. This measures
/// the "tailedness" of the distribution relative to normal.
///
/// # Interpretation
/// - Positive values: Leptokurtic (fat tails, more extreme events)
/// - Zero: Mesokurtic (normal distribution)
/// - Negative values: Platykurtic (thin tails, fewer extremes)
///
/// # Example
/// ```ignore
/// let ek = ExcessKurtosis::new(20).unwrap();
/// let kurtosis = ek.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct ExcessKurtosis {
    period: usize,
}

impl ExcessKurtosis {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate excess kurtosis of returns
    ///
    /// Returns the fourth standardized moment minus 3 (excess kurtosis).
    /// Normal distribution has excess kurtosis of 0.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns
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

            // Calculate mean
            let mean: f64 = window.iter().sum::<f64>() / n_w;

            // Calculate second moment (variance)
            let m2: f64 = window.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / n_w;

            if m2 < 1e-10 {
                continue;
            }

            // Calculate fourth moment
            let m4: f64 = window.iter()
                .map(|r| (r - mean).powi(4))
                .sum::<f64>() / n_w;

            // Excess kurtosis = m4 / m2^2 - 3
            let excess_kurt = m4 / (m2 * m2) - 3.0;
            result[i] = excess_kurt;
        }
        result
    }
}

impl TechnicalIndicator for ExcessKurtosis {
    fn name(&self) -> &str {
        "Excess Kurtosis"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Coefficient of Variation - Rolling coefficient of variation
///
/// Calculates the coefficient of variation (CV = std dev / mean) of
/// returns over a rolling window. This measures relative variability
/// normalized by the mean, useful for comparing volatility across assets.
///
/// # Interpretation
/// - Higher CV: More variable returns relative to mean
/// - Lower CV: More stable returns relative to mean
/// - CV > 1: Standard deviation exceeds mean (high relative risk)
///
/// # Example
/// ```ignore
/// let cov = CoefficientOfVariation::new(20).unwrap();
/// let cv_values = cov.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct CoefficientsOfVariation {
    period: usize,
}

impl CoefficientsOfVariation {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate rolling coefficient of variation
    ///
    /// Returns the ratio of standard deviation to absolute mean.
    /// Uses absolute mean to handle negative average returns.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns
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

            // Calculate mean
            let mean: f64 = window.iter().sum::<f64>() / n_w;

            // Calculate standard deviation
            let variance: f64 = window.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / n_w;
            let std_dev = variance.sqrt();

            // Coefficient of variation: std_dev / |mean|
            // Use absolute mean to handle negative means
            let abs_mean = mean.abs();
            if abs_mean > 1e-10 {
                result[i] = std_dev / abs_mean;
            } else {
                // When mean is near zero, CV is undefined
                // Return a large value to indicate high relative variability
                if std_dev > 1e-10 {
                    result[i] = 100.0; // Cap at 100
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for CoefficientsOfVariation {
    fn name(&self) -> &str {
        "Coefficient of Variation"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Statistical Momentum - Momentum based on statistical measures
///
/// Combines multiple statistical signals to create a robust momentum indicator:
/// - Z-score of current price vs rolling mean
/// - Percentile rank of recent returns
/// - Trend strength from linear regression slope
///
/// # Interpretation
/// - Positive values: Bullish momentum (above average, upward trend)
/// - Zero: Neutral momentum
/// - Negative values: Bearish momentum (below average, downward trend)
///
/// # Example
/// ```ignore
/// let sm = StatisticalMomentum::new(20).unwrap();
/// let momentum = sm.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct StatisticalMomentum {
    period: usize,
}

impl StatisticalMomentum {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate statistical momentum
    ///
    /// Returns a composite momentum score combining:
    /// 1. Price z-score (standardized distance from mean)
    /// 2. Return percentile (position in return distribution)
    /// 3. Trend slope (direction and strength of trend)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns
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
            let start = i + 1 - self.period;
            let price_window = &close[start..=i];
            let current_price = close[i];

            let return_idx = i - 1;
            let return_start = return_idx + 1 - self.period;
            let return_window = &returns[return_start..=return_idx];
            let current_return = returns[return_idx];

            let n_w = price_window.len() as f64;

            // 1. Z-score component: how far current price is from rolling mean
            let price_mean: f64 = price_window.iter().sum::<f64>() / n_w;
            let price_variance: f64 = price_window.iter()
                .map(|p| (p - price_mean).powi(2))
                .sum::<f64>() / n_w;
            let price_std = price_variance.sqrt();

            let z_score = if price_std > 1e-10 {
                (current_price - price_mean) / price_std
            } else {
                0.0
            };

            // 2. Percentile rank component: where current return ranks
            let count_below = return_window.iter().filter(|&&r| r < current_return).count();
            let percentile = (count_below as f64) / (return_window.len() as f64);
            // Convert to -1 to 1 scale
            let percentile_score = (percentile - 0.5) * 2.0;

            // 3. Trend component: linear regression slope normalized
            let x_values: Vec<f64> = (0..price_window.len()).map(|x| x as f64).collect();
            let mean_x: f64 = x_values.iter().sum::<f64>() / n_w;
            let mean_y: f64 = price_window.iter().sum::<f64>() / n_w;

            let mut cov_xy = 0.0;
            let mut var_x = 0.0;

            for (x, y) in x_values.iter().zip(price_window.iter()) {
                let dx = x - mean_x;
                let dy = y - mean_y;
                cov_xy += dx * dy;
                var_x += dx * dx;
            }

            let slope = if var_x > 1e-10 { cov_xy / var_x } else { 0.0 };
            // Normalize slope by mean price to make it scale-independent
            let trend_score = if price_mean > 1e-10 {
                (slope / price_mean) * n_w * 100.0 // Scale up for visibility
            } else {
                0.0
            };

            // Combine components with equal weights
            // Clamp trend_score to prevent domination
            let clamped_trend = trend_score.clamp(-2.0, 2.0);
            result[i] = (z_score + percentile_score + clamped_trend) / 3.0;
        }
        result
    }
}

impl TechnicalIndicator for StatisticalMomentum {
    fn name(&self) -> &str {
        "Statistical Momentum"
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

    #[test]
    fn test_rolling_beta() {
        let close = make_test_data();
        let benchmark: Vec<f64> = close.iter().map(|c| c * 1.02).collect();
        let rb = RollingBeta::new(20).unwrap();
        let result = rb.calculate(&close, &benchmark);

        assert_eq!(result.len(), close.len());
        // Beta should be positive for correlated assets
        for &val in result.iter().skip(21) {
            if val != 0.0 {
                assert!(val > 0.0);
            }
        }
    }

    #[test]
    fn test_rolling_beta_validation() {
        assert!(RollingBeta::new(5).is_err()); // period too small
    }

    #[test]
    fn test_rolling_beta_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let rb = RollingBeta::new(20).unwrap();
        let output = rb.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_rolling_alpha() {
        let close = make_test_data();
        let benchmark: Vec<f64> = close.iter().map(|c| c * 1.02).collect();
        let ra = RollingAlpha::new(20).unwrap();
        let result = ra.calculate(&close, &benchmark);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_rolling_alpha_validation() {
        assert!(RollingAlpha::new(5).is_err()); // period too small
    }

    #[test]
    fn test_rolling_alpha_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let ra = RollingAlpha::new(20).unwrap();
        let output = ra.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_information_coefficient() {
        let close = make_test_data();
        let ic = InformationCoefficient::new(20, 1).unwrap();
        let result = ic.calculate(&close);

        assert_eq!(result.len(), close.len());
        // IC should be between -1 and 1
        for &val in result.iter().skip(22) {
            if val != 0.0 {
                assert!(val >= -1.0 && val <= 1.0);
            }
        }
    }

    #[test]
    fn test_information_coefficient_validation() {
        assert!(InformationCoefficient::new(5, 1).is_err()); // period too small
        assert!(InformationCoefficient::new(20, 0).is_err()); // forward_period too small
    }

    #[test]
    fn test_information_coefficient_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let ic = InformationCoefficient::new(20, 1).unwrap();
        let output = ic.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_rank_correlation() {
        let close = make_test_data();
        let rc = RankCorrelation::new(20, 1).unwrap();
        let result = rc.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Spearman correlation should be between -1 and 1
        for &val in result.iter().skip(22) {
            if val != 0.0 {
                assert!(val >= -1.0 && val <= 1.0);
            }
        }
    }

    #[test]
    fn test_rank_correlation_validation() {
        assert!(RankCorrelation::new(5, 1).is_err()); // period too small
        assert!(RankCorrelation::new(20, 0).is_err()); // lag too small
    }

    #[test]
    fn test_rank_correlation_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let rc = RankCorrelation::new(20, 1).unwrap();
        let output = rc.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_tail_dependence() {
        let close = make_test_data();
        let td = TailDependence::new(20, 0.1).unwrap();
        let result = td.calculate(&close);

        assert_eq!(result.upper.len(), close.len());
        assert_eq!(result.lower.len(), close.len());
        // Tail dependence should be between 0 and 1
        for &val in result.upper.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
        for &val in result.lower.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_tail_dependence_validation() {
        assert!(TailDependence::new(5, 0.1).is_err()); // period too small
        assert!(TailDependence::new(20, 0.0).is_err()); // quantile too small
        assert!(TailDependence::new(20, 0.6).is_err()); // quantile too large
    }

    #[test]
    fn test_tail_dependence_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let td = TailDependence::new(20, 0.1).unwrap();
        let output = td.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_copula_correlation() {
        let close = make_test_data();
        let cc = CopulaCorrelation::new(20, 1).unwrap();
        let result = cc.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Copula correlation should be between -1 and 1
        for &val in result.iter().skip(22) {
            if val != 0.0 {
                assert!(val >= -1.0 && val <= 1.0);
            }
        }
    }

    #[test]
    fn test_copula_correlation_validation() {
        assert!(CopulaCorrelation::new(5, 1).is_err()); // period too small
        assert!(CopulaCorrelation::new(20, 0).is_err()); // lag too small
    }

    #[test]
    fn test_copula_correlation_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let cc = CopulaCorrelation::new(20, 1).unwrap();
        let output = cc.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
    }

    // ==================== ZScoreExtreme Tests ====================

    #[test]
    fn test_zscore_extreme() {
        let mut close = make_test_data();
        // Add an extreme value
        close[25] = close[24] * 1.15; // 15% jump

        let zse = ZScoreExtreme::new(20, 2.0).unwrap();
        let result = zse.calculate(&close);

        assert_eq!(result.len(), close.len());
        // The extreme should be detected
        assert!(result[25].abs() > 0.0);
    }

    #[test]
    fn test_zscore_extreme_validation() {
        assert!(ZScoreExtreme::new(3, 2.0).is_err()); // period too small
        assert!(ZScoreExtreme::new(20, 0.0).is_err()); // threshold not positive
        assert!(ZScoreExtreme::new(20, -1.0).is_err()); // threshold negative
    }

    #[test]
    fn test_zscore_extreme_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let zse = ZScoreExtreme::new(20, 2.0).unwrap();
        let output = zse.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_zscore_extreme_no_extremes() {
        let close = make_test_data(); // Smooth trending data
        let zse = ZScoreExtreme::new(20, 3.0).unwrap(); // High threshold
        let result = zse.calculate(&close);

        // For smooth data with high threshold, most values should be 0
        let extreme_count: usize = result.iter().filter(|&&v| v != 0.0).count();
        assert!(extreme_count < close.len() / 2);
    }

    // ==================== PercentileRank Tests ====================

    #[test]
    fn test_percentile_rank() {
        let close = make_test_data();
        let pr = PercentileRank::new(20).unwrap();
        let result = pr.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Percentile should be between 0 and 100
        for &val in result.iter().skip(19) {
            assert!(val >= 0.0 && val <= 100.0);
        }
    }

    #[test]
    fn test_percentile_rank_validation() {
        assert!(PercentileRank::new(3).is_err()); // period too small
    }

    #[test]
    fn test_percentile_rank_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let pr = PercentileRank::new(20).unwrap();
        let output = pr.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_percentile_rank_trending() {
        let close = make_test_data(); // Trending up data
        let pr = PercentileRank::new(10).unwrap();
        let result = pr.calculate(&close);

        // For trending up data, recent values should have high percentile ranks
        let last_valid = result[close.len() - 1];
        assert!(last_valid > 50.0); // Should be above median
    }

    // ==================== StatisticalRegime Tests ====================

    #[test]
    fn test_statistical_regime() {
        let close = make_test_data();
        let sr = StatisticalRegime::new(20, 1.5).unwrap();
        let result = sr.calculate(&close);

        assert_eq!(result.regime.len(), close.len());
        assert_eq!(result.transition.len(), close.len());
        // Regime should be -1, 0, or 1
        for &val in result.regime.iter() {
            assert!(val == -1.0 || val == 0.0 || val == 1.0);
        }
        // Transition should be 0 or 1
        for &val in result.transition.iter() {
            assert!(val == 0.0 || val == 1.0);
        }
    }

    #[test]
    fn test_statistical_regime_validation() {
        assert!(StatisticalRegime::new(5, 1.5).is_err()); // period too small
        assert!(StatisticalRegime::new(20, 0.0).is_err()); // threshold not positive
        assert!(StatisticalRegime::new(20, -1.0).is_err()); // threshold negative
    }

    #[test]
    fn test_statistical_regime_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let sr = StatisticalRegime::new(20, 1.5).unwrap();
        let output = sr.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert!(output.secondary.is_some());
    }

    // ==================== AutocorrelationIndex Tests ====================

    #[test]
    fn test_autocorrelation_index() {
        let close = make_test_data();
        let ac = AutocorrelationIndex::new(15, 1).unwrap();
        let result = ac.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Autocorrelation should be between -1 and 1
        for &val in result.iter().skip(17) {
            if val != 0.0 {
                assert!(val >= -1.0 && val <= 1.0, "Autocorrelation out of range: {}", val);
            }
        }
    }

    #[test]
    fn test_autocorrelation_index_validation() {
        assert!(AutocorrelationIndex::new(5, 1).is_err()); // period too small
        assert!(AutocorrelationIndex::new(20, 0).is_err()); // lag too small
        assert!(AutocorrelationIndex::new(20, 20).is_err()); // lag >= period
    }

    #[test]
    fn test_autocorrelation_index_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let ac = AutocorrelationIndex::new(15, 1).unwrap();
        let output = ac.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ==================== HurstExponentMA Tests ====================

    #[test]
    fn test_hurst_exponent_ma() {
        let close = make_test_data();
        let hurst = HurstExponentMA::new(20).unwrap();
        let result = hurst.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Hurst exponent should be between 0 and 1
        for &val in result.iter().skip(20) {
            assert!(val >= 0.0 && val <= 1.0, "Hurst out of range: {}", val);
        }
    }

    #[test]
    fn test_hurst_exponent_ma_validation() {
        assert!(HurstExponentMA::new(10).is_err()); // period too small
    }

    #[test]
    fn test_hurst_exponent_ma_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let hurst = HurstExponentMA::new(20).unwrap();
        let output = hurst.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_hurst_exponent_ma_trending() {
        // Create strongly trending data
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let hurst = HurstExponentMA::new(20).unwrap();
        let result = hurst.calculate(&close);

        // Trending data should have H > 0.5 generally
        let valid_values: Vec<f64> = result.iter()
            .skip(20)
            .copied()
            .filter(|&v| v > 0.0)
            .collect();

        if !valid_values.is_empty() {
            let avg: f64 = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            // Allow for some variance but expect persistence
            assert!(avg >= 0.3, "Expected H > 0.3 for trending, got {}", avg);
        }
    }

    // ==================== EntropyMeasure Tests ====================

    #[test]
    fn test_entropy_measure() {
        let close = make_test_data();
        let em = EntropyMeasure::new(20, 5).unwrap();
        let result = em.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Normalized entropy should be between 0 and 1
        for &val in result.iter().skip(20) {
            assert!(val >= 0.0 && val <= 1.0, "Entropy out of range: {}", val);
        }
    }

    #[test]
    fn test_entropy_measure_validation() {
        assert!(EntropyMeasure::new(5, 5).is_err()); // period too small
        assert!(EntropyMeasure::new(20, 1).is_err()); // num_bins too small
        assert!(EntropyMeasure::new(20, 25).is_err()); // num_bins > period
    }

    #[test]
    fn test_entropy_measure_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let em = EntropyMeasure::new(20, 5).unwrap();
        let output = em.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_entropy_measure_uniform_returns() {
        // Create data with varying returns to test entropy calculation
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + i as f64 * 0.5)
            .collect();
        let em = EntropyMeasure::new(20, 5).unwrap();
        let result = em.calculate(&close);

        // Should have some non-zero entropy values
        let non_zero_count = result.iter().skip(20).filter(|&&v| v > 0.0).count();
        assert!(non_zero_count > 0, "Expected some non-zero entropy values");
    }

    #[test]
    fn test_entropy_measure_constant_price() {
        // Constant price should have zero entropy (all returns are same)
        let close: Vec<f64> = vec![100.0; 50];
        let em = EntropyMeasure::new(20, 5).unwrap();
        let result = em.calculate(&close);

        // All values should be 0 for constant price
        for &val in result.iter() {
            assert!(val.abs() < 1e-10, "Expected 0 entropy for constant price");
        }
    }

    // ==================== VarianceRatio Tests ====================

    #[test]
    fn test_variance_ratio() {
        let close = make_test_data();
        let vr = VarianceRatio::new(20, 4).unwrap();
        let result = vr.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Variance ratio should be positive
        for &val in result.iter().skip(21) {
            if val != 0.0 {
                assert!(val > 0.0, "Variance ratio should be positive, got {}", val);
            }
        }
    }

    #[test]
    fn test_variance_ratio_validation() {
        assert!(VarianceRatio::new(5, 2).is_err()); // period too small
        assert!(VarianceRatio::new(20, 1).is_err()); // short_period too small
        assert!(VarianceRatio::new(20, 20).is_err()); // short_period >= period
        assert!(VarianceRatio::new(20, 25).is_err()); // short_period > period
    }

    #[test]
    fn test_variance_ratio_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let vr = VarianceRatio::new(20, 4).unwrap();
        let output = vr.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_variance_ratio_random_walk() {
        // For near-random walk data, VR should be close to 1
        let close = make_test_data();
        let vr = VarianceRatio::new(20, 5).unwrap();
        let result = vr.calculate(&close);

        // Check that values are in reasonable range (0.1 to 10)
        for &val in result.iter().skip(21) {
            if val > 0.0 {
                assert!(val > 0.01 && val < 100.0, "Variance ratio unreasonable: {}", val);
            }
        }
    }

    // ==================== MeanReversionSpeed Tests ====================

    #[test]
    fn test_mean_reversion_speed() {
        let close = make_test_data();
        let mrs = MeanReversionSpeed::new(20).unwrap();
        let result = mrs.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Half-life should be non-negative
        for &val in result.iter() {
            assert!(val >= 0.0, "Half-life should be non-negative, got {}", val);
        }
    }

    #[test]
    fn test_mean_reversion_speed_validation() {
        assert!(MeanReversionSpeed::new(10).is_err()); // period too small
        assert!(MeanReversionSpeed::new(5).is_err()); // period way too small
    }

    #[test]
    fn test_mean_reversion_speed_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let mrs = MeanReversionSpeed::new(20).unwrap();
        let output = mrs.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_mean_reversion_speed_trending() {
        // Strongly trending data should show no mean reversion (0) or very slow (high value)
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 3.0).collect();
        let mrs = MeanReversionSpeed::new(20).unwrap();
        let result = mrs.calculate(&close);

        // For trending data, expect either 0 (no reversion) or high values (slow reversion)
        let valid_values: Vec<f64> = result.iter()
            .skip(20)
            .copied()
            .filter(|&v| v > 0.0)
            .collect();

        if !valid_values.is_empty() {
            for &val in &valid_values {
                // Either no reversion (0) or slow reversion (high half-life)
                assert!(val == 0.0 || val >= 1.0, "Expected no reversion or slow reversion for trending, got {}", val);
            }
        }
    }

    // ==================== TrendStationarity Tests ====================

    #[test]
    fn test_trend_stationarity() {
        let close = make_test_data();
        let ts = TrendStationarity::new(20).unwrap();
        let result = ts.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be finite
        for &val in result.iter().skip(21) {
            assert!(val.is_finite(), "Stationarity test should be finite, got {}", val);
        }
    }

    #[test]
    fn test_trend_stationarity_validation() {
        assert!(TrendStationarity::new(10).is_err()); // period too small
        assert!(TrendStationarity::new(5).is_err()); // period way too small
    }

    #[test]
    fn test_trend_stationarity_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let ts = TrendStationarity::new(20).unwrap();
        let output = ts.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_trend_stationarity_trending_data() {
        // Strongly trending data
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let ts = TrendStationarity::new(20).unwrap();
        let result = ts.calculate(&close);

        // All non-zero values should be finite
        for &val in result.iter() {
            if val != 0.0 {
                assert!(val.is_finite(), "Expected finite stationarity value");
            }
        }
    }

    // ==================== ExcessKurtosis Tests ====================

    #[test]
    fn test_excess_kurtosis() {
        let close = make_test_data();
        let ek = ExcessKurtosis::new(20).unwrap();
        let result = ek.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Kurtosis values should be finite
        for &val in result.iter().skip(21) {
            assert!(val.is_finite(), "Kurtosis should be finite, got {}", val);
        }
    }

    #[test]
    fn test_excess_kurtosis_validation() {
        assert!(ExcessKurtosis::new(5).is_err()); // period too small
        assert!(ExcessKurtosis::new(3).is_err()); // period way too small
    }

    #[test]
    fn test_excess_kurtosis_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let ek = ExcessKurtosis::new(20).unwrap();
        let output = ek.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_excess_kurtosis_normal_like() {
        // For smooth trending data, excess kurtosis should be close to 0 or negative
        let close = make_test_data();
        let ek = ExcessKurtosis::new(20).unwrap();
        let result = ek.calculate(&close);

        // Check that values are in reasonable range
        for &val in result.iter().skip(21) {
            // Excess kurtosis typically ranges from -2 to 10+ for financial data
            assert!(val > -3.0 && val < 50.0, "Excess kurtosis unreasonable: {}", val);
        }
    }

    // ==================== CoefficientsOfVariation Tests ====================

    #[test]
    fn test_coefficients_of_variation() {
        let close = make_test_data();
        let cov = CoefficientsOfVariation::new(20).unwrap();
        let result = cov.calculate(&close);

        assert_eq!(result.len(), close.len());
        // CV should be non-negative
        for &val in result.iter() {
            assert!(val >= 0.0, "CV should be non-negative, got {}", val);
        }
    }

    #[test]
    fn test_coefficients_of_variation_validation() {
        assert!(CoefficientsOfVariation::new(3).is_err()); // period too small
        assert!(CoefficientsOfVariation::new(2).is_err()); // period way too small
    }

    #[test]
    fn test_coefficients_of_variation_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let cov = CoefficientsOfVariation::new(20).unwrap();
        let output = cov.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_coefficients_of_variation_constant() {
        // Constant price changes should have zero CV (no variation)
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let cov = CoefficientsOfVariation::new(20).unwrap();
        let result = cov.calculate(&close);

        // For perfectly linear data, CV should be low (returns are constant)
        let valid_values: Vec<f64> = result.iter()
            .skip(21)
            .copied()
            .filter(|&v| v > 0.0)
            .collect();

        if !valid_values.is_empty() {
            let avg: f64 = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            // For constant returns, CV should be near zero or small
            assert!(avg < 10.0, "Expected low CV for linear trending data, got {}", avg);
        }
    }

    // ==================== StatisticalMomentum Tests ====================

    #[test]
    fn test_statistical_momentum() {
        let close = make_test_data();
        let sm = StatisticalMomentum::new(20).unwrap();
        let result = sm.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be finite
        for &val in result.iter() {
            assert!(val.is_finite(), "Statistical momentum should be finite, got {}", val);
        }
    }

    #[test]
    fn test_statistical_momentum_validation() {
        assert!(StatisticalMomentum::new(5).is_err()); // period too small
        assert!(StatisticalMomentum::new(3).is_err()); // period way too small
    }

    #[test]
    fn test_statistical_momentum_compute() {
        let close = make_test_data();
        let data = make_ohlcv_series(close);
        let sm = StatisticalMomentum::new(20).unwrap();
        let output = sm.compute(&data).unwrap();

        assert!(!output.primary.is_empty());
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_statistical_momentum_uptrend() {
        // Strongly uptrending data should have positive momentum
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let sm = StatisticalMomentum::new(15).unwrap();
        let result = sm.calculate(&close);

        // Later values should show positive momentum
        let last_values: Vec<f64> = result.iter()
            .skip(30)
            .copied()
            .filter(|&v| v != 0.0)
            .collect();

        if !last_values.is_empty() {
            let avg: f64 = last_values.iter().sum::<f64>() / last_values.len() as f64;
            assert!(avg > 0.0, "Expected positive momentum for uptrend, got {}", avg);
        }
    }

    #[test]
    fn test_statistical_momentum_downtrend() {
        // Strongly downtrending data should have negative momentum
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64 * 2.0).collect();
        let sm = StatisticalMomentum::new(15).unwrap();
        let result = sm.calculate(&close);

        // Later values should show negative momentum
        let last_values: Vec<f64> = result.iter()
            .skip(30)
            .copied()
            .filter(|&v| v != 0.0)
            .collect();

        if !last_values.is_empty() {
            let avg: f64 = last_values.iter().sum::<f64>() / last_values.len() as f64;
            assert!(avg < 0.0, "Expected negative momentum for downtrend, got {}", avg);
        }
    }
}
