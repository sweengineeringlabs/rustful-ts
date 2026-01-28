//! Stationarity and Distribution Tests
//!
//! Tests for unit roots, stationarity, and distribution fitting.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Kolmogorov-Smirnov Test - Tests if data follows normal distribution.
///
/// Returns the D statistic (maximum deviation from expected CDF).
/// Lower values indicate better fit to normality.
#[derive(Debug, Clone)]
pub struct KolmogorovSmirnov {
    period: usize,
}

impl KolmogorovSmirnov {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Standard normal CDF approximation.
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation.
    fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Calculate rolling KS statistic.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 3 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let mut window: Vec<f64> = data[start..=i].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Standardize
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            if std_dev.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            // Calculate KS statistic
            let mut d_max = 0.0f64;

            for (j, &x) in window.iter().enumerate() {
                let z = (x - mean) / std_dev;
                let f_expected = Self::normal_cdf(z);
                let f_observed = (j + 1) as f64 / self.period as f64;
                let f_observed_prev = j as f64 / self.period as f64;

                d_max = d_max.max((f_observed - f_expected).abs());
                d_max = d_max.max((f_observed_prev - f_expected).abs());
            }

            result.push(d_max);
        }

        result
    }
}

impl TechnicalIndicator for KolmogorovSmirnov {
    fn name(&self) -> &str {
        "Kolmogorov-Smirnov"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Anderson-Darling Test - Tests for normality with emphasis on tails.
///
/// More sensitive to tail deviations than KS test.
#[derive(Debug, Clone)]
pub struct AndersonDarling {
    period: usize,
}

impl AndersonDarling {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling Anderson-Darling statistic.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 3 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let mut window: Vec<f64> = data[start..=i].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            if std_dev.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            // Calculate AD statistic
            let n_f = self.period as f64;
            let mut s = 0.0;

            for (j, &x) in window.iter().enumerate() {
                let z = (x - mean) / std_dev;
                let f = KolmogorovSmirnov::normal_cdf(z);
                let f = f.clamp(1e-10, 1.0 - 1e-10);

                let f_rev = KolmogorovSmirnov::normal_cdf((window[self.period - 1 - j] - mean) / std_dev);
                let f_rev = f_rev.clamp(1e-10, 1.0 - 1e-10);

                s += (2.0 * (j + 1) as f64 - 1.0) * (f.ln() + (1.0 - f_rev).ln());
            }

            let ad = -n_f - s / n_f;
            result.push(ad.max(0.0));
        }

        result
    }
}

impl TechnicalIndicator for AndersonDarling {
    fn name(&self) -> &str {
        "Anderson-Darling"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Augmented Dickey-Fuller Test (simplified) - Tests for unit root.
///
/// Tests H0: series has a unit root (non-stationary)
/// More negative values suggest stationarity.
#[derive(Debug, Clone)]
pub struct AugmentedDickeyFuller {
    period: usize,
    lags: usize,
}

impl AugmentedDickeyFuller {
    pub fn new(period: usize) -> Self {
        Self { period, lags: 1 }
    }

    pub fn with_lags(period: usize, lags: usize) -> Self {
        Self { period, lags: lags.max(1) }
    }

    /// Calculate rolling ADF statistic.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < self.lags + 3 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Calculate first differences
            let diffs: Vec<f64> = (1..window.len())
                .map(|j| window[j] - window[j - 1])
                .collect();

            if diffs.len() < self.lags + 2 {
                result.push(f64::NAN);
                continue;
            }

            // Simple OLS: diff(y) = alpha + beta * y_{t-1} + error
            // Test statistic for beta
            let y_lag: Vec<f64> = window[..window.len() - 1].to_vec();
            let y_diff: Vec<f64> = diffs.clone();

            let n_reg = y_diff.len().min(y_lag.len());
            if n_reg < 3 {
                result.push(f64::NAN);
                continue;
            }

            let y_lag = &y_lag[..n_reg];
            let y_diff = &y_diff[..n_reg];

            // Calculate means
            let mean_lag: f64 = y_lag.iter().sum::<f64>() / n_reg as f64;
            let mean_diff: f64 = y_diff.iter().sum::<f64>() / n_reg as f64;

            // Calculate regression coefficient
            let mut ss_xy = 0.0;
            let mut ss_xx = 0.0;

            for j in 0..n_reg {
                ss_xy += (y_lag[j] - mean_lag) * (y_diff[j] - mean_diff);
                ss_xx += (y_lag[j] - mean_lag).powi(2);
            }

            if ss_xx.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            let beta = ss_xy / ss_xx;
            let alpha = mean_diff - beta * mean_lag;

            // Calculate residuals and standard error
            let residuals: Vec<f64> = (0..n_reg)
                .map(|j| y_diff[j] - alpha - beta * y_lag[j])
                .collect();

            let sse: f64 = residuals.iter().map(|r| r * r).sum();
            let mse = sse / (n_reg - 2) as f64;
            let se_beta = (mse / ss_xx).sqrt();

            if se_beta.abs() < 1e-10 {
                result.push(f64::NAN);
            } else {
                let t_stat = beta / se_beta;
                result.push(t_stat);
            }
        }

        result
    }
}

impl TechnicalIndicator for AugmentedDickeyFuller {
    fn name(&self) -> &str {
        "Augmented Dickey-Fuller"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// KPSS Test (simplified) - Tests for stationarity.
///
/// Tests H0: series is stationary
/// Lower values suggest stationarity.
#[derive(Debug, Clone)]
pub struct KPSS {
    period: usize,
}

impl KPSS {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling KPSS statistic.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 4 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            // Calculate residuals from mean
            let residuals: Vec<f64> = window.iter().map(|x| x - mean).collect();

            // Calculate partial sums
            let mut partial_sums = Vec::with_capacity(self.period);
            let mut cumsum = 0.0;
            for r in &residuals {
                cumsum += r;
                partial_sums.push(cumsum);
            }

            // Calculate variance (long-run variance approximation)
            let variance: f64 = residuals.iter().map(|r| r * r).sum::<f64>() / self.period as f64;

            if variance.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            // KPSS statistic
            let ss_partial: f64 = partial_sums.iter().map(|s| s * s).sum();
            let kpss = ss_partial / (self.period as f64).powi(2) / variance;

            result.push(kpss);
        }

        result
    }
}

impl TechnicalIndicator for KPSS {
    fn name(&self) -> &str {
        "KPSS"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Phillips-Perron Test (simplified) - Unit root test robust to heteroskedasticity.
///
/// Similar to ADF but with non-parametric correction for serial correlation.
#[derive(Debug, Clone)]
pub struct PhillipsPerron {
    period: usize,
}

impl PhillipsPerron {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling Phillips-Perron statistic.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 4 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Simple AR(1) regression: y_t = alpha + rho * y_{t-1} + e_t
            let y: Vec<f64> = window[1..].to_vec();
            let y_lag: Vec<f64> = window[..window.len() - 1].to_vec();
            let n_reg = y.len();

            if n_reg < 3 {
                result.push(f64::NAN);
                continue;
            }

            let mean_y: f64 = y.iter().sum::<f64>() / n_reg as f64;
            let mean_lag: f64 = y_lag.iter().sum::<f64>() / n_reg as f64;

            let mut ss_xy = 0.0;
            let mut ss_xx = 0.0;

            for j in 0..n_reg {
                ss_xy += (y_lag[j] - mean_lag) * (y[j] - mean_y);
                ss_xx += (y_lag[j] - mean_lag).powi(2);
            }

            if ss_xx.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            let rho = ss_xy / ss_xx;

            // Calculate residuals
            let alpha = mean_y - rho * mean_lag;
            let residuals: Vec<f64> = (0..n_reg)
                .map(|j| y[j] - alpha - rho * y_lag[j])
                .collect();

            let s2: f64 = residuals.iter().map(|r| r * r).sum::<f64>() / n_reg as f64;
            let se_rho = (s2 / ss_xx).sqrt();

            if se_rho.abs() < 1e-10 {
                result.push(f64::NAN);
            } else {
                // PP statistic (simplified, without full Newey-West correction)
                let t_rho = (rho - 1.0) / se_rho;
                result.push(t_rho);
            }
        }

        result
    }
}

impl TechnicalIndicator for PhillipsPerron {
    fn name(&self) -> &str {
        "Phillips-Perron"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// TRIN Moving Averages - Smoothed versions of Arms Index.
#[derive(Debug, Clone)]
pub struct TRINMovingAverage {
    period: usize,
    smoothing: usize,
}

impl TRINMovingAverage {
    pub fn new(smoothing: usize) -> Self {
        Self { period: smoothing, smoothing }
    }

    /// TRIN-5: 5-day moving average of TRIN
    pub fn trin5() -> Self {
        Self { period: 5, smoothing: 5 }
    }

    /// TRIN-10: 10-day moving average of TRIN
    pub fn trin10() -> Self {
        Self { period: 10, smoothing: 10 }
    }

    /// Calculate TRIN from breadth data.
    /// Requires: advances, declines, advancing_volume, declining_volume
    pub fn calculate_trin(
        advances: &[f64],
        declines: &[f64],
        adv_volume: &[f64],
        dec_volume: &[f64],
    ) -> Vec<f64> {
        let n = advances.len();
        let mut trin = Vec::with_capacity(n);

        for i in 0..n {
            let ad_ratio = if declines[i].abs() > 1e-10 {
                advances[i] / declines[i]
            } else {
                f64::NAN
            };

            let vol_ratio = if dec_volume[i].abs() > 1e-10 {
                adv_volume[i] / dec_volume[i]
            } else {
                f64::NAN
            };

            if ad_ratio.is_nan() || vol_ratio.is_nan() || vol_ratio.abs() < 1e-10 {
                trin.push(f64::NAN);
            } else {
                trin.push(ad_ratio / vol_ratio);
            }
        }

        trin
    }

    /// Calculate smoothed TRIN.
    pub fn calculate_smoothed(&self, trin: &[f64]) -> Vec<f64> {
        let n = trin.len();
        if n < self.smoothing {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.smoothing - 1];

        for i in (self.smoothing - 1)..n {
            let start = i + 1 - self.smoothing;
            let window = &trin[start..=i];

            let valid: Vec<f64> = window.iter().filter(|x| !x.is_nan()).cloned().collect();
            if valid.is_empty() {
                result.push(f64::NAN);
            } else {
                let sum: f64 = valid.iter().sum();
                result.push(sum / valid.len() as f64);
            }
        }

        result
    }
}

impl TechnicalIndicator for TRINMovingAverage {
    fn name(&self) -> &str {
        "TRIN Moving Average"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // TRIN requires breadth data, not regular OHLCV
        // This returns NaN for standard OHLCV data
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        Ok(IndicatorOutput::single(vec![f64::NAN; data.close.len()]))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kolmogorov_smirnov() {
        // Normal-ish data
        let data: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).sin()).collect();
        let ks = KolmogorovSmirnov::new(20);
        let result = ks.calculate(&data);

        assert_eq!(result.len(), 30);
        // D should be between 0 and 1
        for val in &result[19..] {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 1.0);
            }
        }
    }

    #[test]
    fn test_anderson_darling() {
        let data: Vec<f64> = (0..30).map(|i| (i as f64 * 0.2).sin()).collect();
        let ad = AndersonDarling::new(15);
        let result = ad.calculate(&data);

        assert_eq!(result.len(), 30);
        // AD should be non-negative
        for val in &result[14..] {
            if !val.is_nan() {
                assert!(*val >= 0.0);
            }
        }
    }

    #[test]
    fn test_adf() {
        // Stationary data (oscillating)
        let stationary: Vec<f64> = (0..50).map(|i| (i as f64 * 0.5).sin()).collect();
        let adf = AugmentedDickeyFuller::new(20);
        let result = adf.calculate(&stationary);

        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_kpss() {
        let data: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64 * 0.3).sin()).collect();
        let kpss = KPSS::new(20);
        let result = kpss.calculate(&data);

        assert_eq!(result.len(), 40);
        // KPSS should be non-negative
        for val in &result[19..] {
            if !val.is_nan() {
                assert!(*val >= 0.0);
            }
        }
    }

    #[test]
    fn test_phillips_perron() {
        let data: Vec<f64> = (0..40).map(|i| (i as f64 * 0.4).sin()).collect();
        let pp = PhillipsPerron::new(20);
        let result = pp.calculate(&data);

        assert_eq!(result.len(), 40);
    }

    #[test]
    fn test_trin_calculation() {
        let advances = vec![1500.0, 1600.0, 1400.0, 1700.0, 1550.0];
        let declines = vec![1000.0, 900.0, 1100.0, 800.0, 950.0];
        let adv_vol = vec![2000.0, 2200.0, 1800.0, 2500.0, 2100.0];
        let dec_vol = vec![1500.0, 1400.0, 1600.0, 1200.0, 1450.0];

        let trin = TRINMovingAverage::calculate_trin(&advances, &declines, &adv_vol, &dec_vol);

        assert_eq!(trin.len(), 5);
        // TRIN values should be reasonable
        for val in &trin {
            if !val.is_nan() {
                assert!(*val > 0.0 && *val < 10.0);
            }
        }
    }
}
