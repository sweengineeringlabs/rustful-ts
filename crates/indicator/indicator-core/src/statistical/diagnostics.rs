//! Model Diagnostic Statistics
//!
//! Statistical tests for model validation and residual analysis.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// F-Statistic - Tests overall significance of regression.
///
/// F = (R² / k) / ((1 - R²) / (n - k - 1))
///
/// Where k is number of predictors.
#[derive(Debug, Clone)]
pub struct FStatistic {
    period: usize,
    predictors: usize,
}

impl FStatistic {
    pub fn new(period: usize) -> Self {
        Self { period, predictors: 1 }
    }

    pub fn with_predictors(period: usize, predictors: usize) -> Self {
        Self { period, predictors }
    }

    /// Calculate rolling F-statistic.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period <= self.predictors + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Calculate R² first
            let x_mean = (self.period - 1) as f64 / 2.0;
            let y_mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            let mut ss_xy = 0.0;
            let mut ss_xx = 0.0;
            let mut ss_tot = 0.0;

            for (j, &y) in window.iter().enumerate() {
                let x = j as f64;
                ss_xy += (x - x_mean) * (y - y_mean);
                ss_xx += (x - x_mean).powi(2);
                ss_tot += (y - y_mean).powi(2);
            }

            if ss_xx.abs() < 1e-10 || ss_tot.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            let slope = ss_xy / ss_xx;
            let intercept = y_mean - slope * x_mean;

            let ss_res: f64 = window.iter()
                .enumerate()
                .map(|(j, &y)| {
                    let y_pred = intercept + slope * j as f64;
                    (y - y_pred).powi(2)
                })
                .sum();

            let r_squared = 1.0 - (ss_res / ss_tot);
            let k = self.predictors as f64;
            let n_f = self.period as f64;

            let denominator = (1.0 - r_squared) / (n_f - k - 1.0);
            if denominator.abs() < 1e-10 {
                result.push(f64::NAN);
            } else {
                let f = (r_squared / k) / denominator;
                result.push(f.max(0.0));
            }
        }

        result
    }
}

impl TechnicalIndicator for FStatistic {
    fn name(&self) -> &str {
        "F-Statistic"
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

/// AIC - Akaike Information Criterion.
///
/// AIC = 2k - 2ln(L) ≈ n * ln(RSS/n) + 2k
///
/// Lower values indicate better model fit with penalty for complexity.
#[derive(Debug, Clone)]
pub struct AIC {
    period: usize,
    predictors: usize,
}

impl AIC {
    pub fn new(period: usize) -> Self {
        Self { period, predictors: 1 }
    }

    pub fn with_predictors(period: usize, predictors: usize) -> Self {
        Self { period, predictors }
    }

    /// Calculate rolling AIC.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 3 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Linear regression
            let x_mean = (self.period - 1) as f64 / 2.0;
            let y_mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            let mut ss_xy = 0.0;
            let mut ss_xx = 0.0;

            for (j, &y) in window.iter().enumerate() {
                let x = j as f64;
                ss_xy += (x - x_mean) * (y - y_mean);
                ss_xx += (x - x_mean).powi(2);
            }

            if ss_xx.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            let slope = ss_xy / ss_xx;
            let intercept = y_mean - slope * x_mean;

            // Calculate RSS
            let rss: f64 = window.iter()
                .enumerate()
                .map(|(j, &y)| {
                    let y_pred = intercept + slope * j as f64;
                    (y - y_pred).powi(2)
                })
                .sum();

            let n_f = self.period as f64;
            let k = (self.predictors + 1) as f64; // +1 for intercept

            let aic = n_f * (rss / n_f).ln() + 2.0 * k;
            result.push(aic);
        }

        result
    }
}

impl TechnicalIndicator for AIC {
    fn name(&self) -> &str {
        "AIC"
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

/// BIC - Bayesian Information Criterion.
///
/// BIC = k * ln(n) - 2ln(L) ≈ n * ln(RSS/n) + k * ln(n)
///
/// Similar to AIC but with stronger penalty for model complexity.
#[derive(Debug, Clone)]
pub struct BIC {
    period: usize,
    predictors: usize,
}

impl BIC {
    pub fn new(period: usize) -> Self {
        Self { period, predictors: 1 }
    }

    pub fn with_predictors(period: usize, predictors: usize) -> Self {
        Self { period, predictors }
    }

    /// Calculate rolling BIC.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 3 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Linear regression
            let x_mean = (self.period - 1) as f64 / 2.0;
            let y_mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            let mut ss_xy = 0.0;
            let mut ss_xx = 0.0;

            for (j, &y) in window.iter().enumerate() {
                let x = j as f64;
                ss_xy += (x - x_mean) * (y - y_mean);
                ss_xx += (x - x_mean).powi(2);
            }

            if ss_xx.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            let slope = ss_xy / ss_xx;
            let intercept = y_mean - slope * x_mean;

            let rss: f64 = window.iter()
                .enumerate()
                .map(|(j, &y)| {
                    let y_pred = intercept + slope * j as f64;
                    (y - y_pred).powi(2)
                })
                .sum();

            let n_f = self.period as f64;
            let k = (self.predictors + 1) as f64;

            let bic = n_f * (rss / n_f).ln() + k * n_f.ln();
            result.push(bic);
        }

        result
    }
}

impl TechnicalIndicator for BIC {
    fn name(&self) -> &str {
        "BIC"
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

/// Durbin-Watson Statistic - Tests for autocorrelation in residuals.
///
/// DW = Σ(e_t - e_{t-1})² / Σe_t²
///
/// Values close to 2 indicate no autocorrelation.
/// < 2: positive autocorrelation
/// > 2: negative autocorrelation
#[derive(Debug, Clone)]
pub struct DurbinWatson {
    period: usize,
}

impl DurbinWatson {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling Durbin-Watson statistic.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 3 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Calculate residuals from linear regression
            let x_mean = (self.period - 1) as f64 / 2.0;
            let y_mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            let mut ss_xy = 0.0;
            let mut ss_xx = 0.0;

            for (j, &y) in window.iter().enumerate() {
                let x = j as f64;
                ss_xy += (x - x_mean) * (y - y_mean);
                ss_xx += (x - x_mean).powi(2);
            }

            if ss_xx.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            let slope = ss_xy / ss_xx;
            let intercept = y_mean - slope * x_mean;

            // Calculate residuals
            let residuals: Vec<f64> = window.iter()
                .enumerate()
                .map(|(j, &y)| y - (intercept + slope * j as f64))
                .collect();

            // Calculate DW statistic
            let mut sum_diff_sq = 0.0;
            let mut sum_sq = 0.0;

            for j in 1..residuals.len() {
                sum_diff_sq += (residuals[j] - residuals[j - 1]).powi(2);
            }

            for r in &residuals {
                sum_sq += r.powi(2);
            }

            if sum_sq.abs() < 1e-10 {
                result.push(f64::NAN);
            } else {
                let dw = sum_diff_sq / sum_sq;
                result.push(dw);
            }
        }

        result
    }
}

impl TechnicalIndicator for DurbinWatson {
    fn name(&self) -> &str {
        "Durbin-Watson"
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

/// Jarque-Bera Test - Tests for normality of residuals.
///
/// JB = (n/6) * (S² + (K-3)²/4)
///
/// Where S is skewness and K is kurtosis.
/// Higher values indicate departure from normality.
#[derive(Debug, Clone)]
pub struct JarqueBera {
    period: usize,
}

impl JarqueBera {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling Jarque-Bera statistic.
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

            let m2: f64 = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.period as f64;
            let m3: f64 = window.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / self.period as f64;
            let m4: f64 = window.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / self.period as f64;

            if m2.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            let std_dev = m2.sqrt();
            let skewness = m3 / std_dev.powi(3);
            let kurtosis = m4 / m2.powi(2);

            let n_f = self.period as f64;
            let jb = (n_f / 6.0) * (skewness.powi(2) + (kurtosis - 3.0).powi(2) / 4.0);
            result.push(jb);
        }

        result
    }
}

impl TechnicalIndicator for JarqueBera {
    fn name(&self) -> &str {
        "Jarque-Bera"
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

/// Shapiro-Wilk Test Approximation - Tests for normality.
///
/// Returns an approximation of the W statistic.
/// Values close to 1 indicate normality.
#[derive(Debug, Clone)]
pub struct ShapiroWilk {
    period: usize,
}

impl ShapiroWilk {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling Shapiro-Wilk approximation.
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
            let ss: f64 = window.iter().map(|x| (x - mean).powi(2)).sum();

            if ss.abs() < 1e-10 {
                result.push(f64::NAN);
                continue;
            }

            // Simplified approximation using ordered statistics
            // W ≈ (Σ a_i * x_(i))² / SS
            // Using simplified coefficients
            let m = self.period / 2;
            let mut b = 0.0;

            for j in 0..m {
                // Approximate coefficients
                let a = (2.0 * (j + 1) as f64 - 1.0) / (2.0 * self.period as f64);
                let z = Self::inv_normal(a);
                b += z * (window[self.period - 1 - j] - window[j]);
            }

            let w = (b * b) / ss;
            result.push(w.min(1.0));
        }

        result
    }

    /// Simple approximation of inverse normal CDF.
    fn inv_normal(p: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let p = p.clamp(0.001, 0.999);
        let t = (-2.0 * (if p <= 0.5 { p } else { 1.0 - p }).ln()).sqrt();

        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

        if p <= 0.5 { -z } else { z }
    }
}

impl TechnicalIndicator for ShapiroWilk {
    fn name(&self) -> &str {
        "Shapiro-Wilk"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f_statistic() {
        let linear_data: Vec<f64> = (0..20).map(|x| 100.0 + x as f64 * 2.0).collect();
        let f_stat = FStatistic::new(10);
        let result = f_stat.calculate(&linear_data);

        assert_eq!(result.len(), 20);
        // F should be high for perfect linear fit
        if !result[19].is_nan() {
            assert!(result[19] > 100.0);
        }
    }

    #[test]
    fn test_aic() {
        let data: Vec<f64> = (0..20).map(|x| 100.0 + x as f64 * 0.5 + (x as f64).sin()).collect();
        let aic = AIC::new(10);
        let result = aic.calculate(&data);

        assert_eq!(result.len(), 20);
        // AIC should return finite values
        for val in &result[9..] {
            if !val.is_nan() {
                assert!(val.is_finite());
            }
        }
    }

    #[test]
    fn test_bic() {
        let data: Vec<f64> = (0..20).map(|x| 100.0 + x as f64 * 0.5).collect();
        let bic = BIC::new(10);
        let result = bic.calculate(&data);

        assert_eq!(result.len(), 20);
    }

    #[test]
    fn test_durbin_watson() {
        // Random-looking data should have DW close to 2
        let data = vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.6, 2.4, 1.7, 2.3,
                       1.9, 2.1, 1.4, 2.6, 1.3, 2.7, 1.2, 2.8, 1.1, 2.9];
        let dw = DurbinWatson::new(10);
        let result = dw.calculate(&data);

        assert_eq!(result.len(), 20);
        // DW should be between 0 and 4
        for val in &result[9..] {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 4.0);
            }
        }
    }

    #[test]
    fn test_jarque_bera() {
        // Normal-ish data should have low JB
        let data: Vec<f64> = (0..30).map(|i| (i as f64 * 0.5).sin()).collect();
        let jb = JarqueBera::new(20);
        let result = jb.calculate(&data);

        assert_eq!(result.len(), 30);
        // JB should be non-negative
        for val in &result[19..] {
            if !val.is_nan() {
                assert!(*val >= 0.0);
            }
        }
    }

    #[test]
    fn test_shapiro_wilk() {
        let data: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin()).collect();
        let sw = ShapiroWilk::new(10);
        let result = sw.calculate(&data);

        assert_eq!(result.len(), 20);
        // W should be between 0 and 1
        for val in &result[9..] {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 1.0);
            }
        }
    }
}
