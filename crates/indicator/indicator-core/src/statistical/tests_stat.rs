//! Statistical Tests and Confidence Measures
//!
//! Hypothesis testing and confidence interval calculations for time series analysis.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Mean Absolute Deviation (MAD) - Average absolute deviation from mean.
///
/// MAD = mean(|x - mean(x)|)
///
/// More robust to outliers than standard deviation.
#[derive(Debug, Clone)]
pub struct MAD {
    period: usize,
}

impl MAD {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling MAD.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let mad: f64 = window.iter()
                .map(|x| (x - mean).abs())
                .sum::<f64>() / self.period as f64;

            result.push(mad);
        }

        result
    }
}

impl TechnicalIndicator for MAD {
    fn name(&self) -> &str {
        "MAD"
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

/// T-Statistic - Test whether mean differs from a reference value.
///
/// t = (sample_mean - reference) / (sample_std / sqrt(n))
///
/// Used to test statistical significance of price changes.
#[derive(Debug, Clone)]
pub struct TStatistic {
    period: usize,
    reference: f64,
}

impl TStatistic {
    pub fn new(period: usize) -> Self {
        Self { period, reference: 0.0 }
    }

    pub fn with_reference(period: usize, reference: f64) -> Self {
        Self { period, reference }
    }

    /// Calculate rolling t-statistic.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 2 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (self.period - 1) as f64;
            let std_dev = variance.sqrt();

            if std_dev.abs() < 1e-10 {
                result.push(f64::NAN);
            } else {
                let se = std_dev / (self.period as f64).sqrt();
                let t = (mean - self.reference) / se;
                result.push(t);
            }
        }

        result
    }
}

impl TechnicalIndicator for TStatistic {
    fn name(&self) -> &str {
        "T-Statistic"
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

/// P-Value approximation from t-statistic.
///
/// Uses approximation for two-tailed test.
/// For large samples, uses normal distribution approximation.
#[derive(Debug, Clone)]
pub struct PValue {
    period: usize,
    reference: f64,
}

impl PValue {
    pub fn new(period: usize) -> Self {
        Self { period, reference: 0.0 }
    }

    pub fn with_reference(period: usize, reference: f64) -> Self {
        Self { period, reference }
    }

    /// Approximate p-value from t-statistic using normal CDF approximation.
    fn approx_p_value(t: f64, _df: usize) -> f64 {
        // Use normal approximation for simplicity
        // For df > 30, t-distribution is very close to normal
        let x = t.abs();

        // Approximation of 2 * (1 - Φ(x)) where Φ is standard normal CDF
        // Using Abramowitz and Stegun approximation
        let b1 = 0.319381530;
        let b2 = -0.356563782;
        let b3 = 1.781477937;
        let b4 = -1.821255978;
        let b5 = 1.330274429;
        let p = 0.2316419;

        let t_val = 1.0 / (1.0 + p * x);
        let z = (-(x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let phi = 1.0 - z * (b1 * t_val + b2 * t_val.powi(2) + b3 * t_val.powi(3)
            + b4 * t_val.powi(4) + b5 * t_val.powi(5));

        2.0 * (1.0 - phi)
    }

    /// Calculate rolling p-values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let t_stat = TStatistic::with_reference(self.period, self.reference);
        let t_values = t_stat.calculate(data);
        let df = self.period.saturating_sub(1);

        t_values.iter().map(|&t| {
            if t.is_nan() {
                f64::NAN
            } else {
                Self::approx_p_value(t, df)
            }
        }).collect()
    }
}

impl TechnicalIndicator for PValue {
    fn name(&self) -> &str {
        "P-Value"
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

/// Confidence Interval - Upper and lower bounds for mean estimate.
///
/// CI = mean ± t_critical * (std / sqrt(n))
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    period: usize,
    confidence_level: f64, // e.g., 0.95 for 95%
}

/// Output for Confidence Interval.
#[derive(Debug, Clone)]
pub struct ConfidenceIntervalOutput {
    /// Lower bound of confidence interval
    pub lower: Vec<f64>,
    /// Sample mean
    pub mean: Vec<f64>,
    /// Upper bound of confidence interval
    pub upper: Vec<f64>,
}

impl ConfidenceInterval {
    pub fn new(period: usize) -> Self {
        Self { period, confidence_level: 0.95 }
    }

    pub fn with_level(period: usize, confidence_level: f64) -> Self {
        Self {
            period,
            confidence_level: confidence_level.clamp(0.50, 0.999),
        }
    }

    /// Get approximate z-score for confidence level (normal approximation).
    fn z_score(&self) -> f64 {
        // Common z-scores for confidence levels
        match (self.confidence_level * 100.0).round() as i32 {
            90 => 1.645,
            95 => 1.96,
            99 => 2.576,
            _ => {
                // Use approximation
                let alpha = 1.0 - self.confidence_level;
                // Inverse normal approximation (rough)
                if alpha <= 0.01 {
                    2.576
                } else if alpha <= 0.05 {
                    1.96
                } else {
                    1.645
                }
            }
        }
    }

    /// Calculate rolling confidence intervals.
    pub fn calculate(&self, data: &[f64]) -> ConfidenceIntervalOutput {
        let n = data.len();
        if n < self.period || self.period < 2 {
            return ConfidenceIntervalOutput {
                lower: vec![f64::NAN; n],
                mean: vec![f64::NAN; n],
                upper: vec![f64::NAN; n],
            };
        }

        let mut lower = vec![f64::NAN; self.period - 1];
        let mut mean_vec = vec![f64::NAN; self.period - 1];
        let mut upper = vec![f64::NAN; self.period - 1];

        let z = self.z_score();

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (self.period - 1) as f64;
            let std_dev = variance.sqrt();

            let se = std_dev / (self.period as f64).sqrt();
            let margin = z * se;

            mean_vec.push(mean);
            lower.push(mean - margin);
            upper.push(mean + margin);
        }

        ConfidenceIntervalOutput { lower, mean: mean_vec, upper }
    }
}

impl TechnicalIndicator for ConfidenceInterval {
    fn name(&self) -> &str {
        "Confidence Interval"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let output = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(output.lower, output.mean, output.upper))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3
    }
}

/// R-Squared - Coefficient of determination.
///
/// Measures how well a linear trend fits the data.
/// R² = 1 - (SS_res / SS_tot)
#[derive(Debug, Clone)]
pub struct RSquared {
    period: usize,
}

impl RSquared {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate rolling R-squared.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 2 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Calculate linear regression
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

            // Calculate SS_res
            let ss_res: f64 = window.iter()
                .enumerate()
                .map(|(j, &y)| {
                    let y_pred = intercept + slope * j as f64;
                    (y - y_pred).powi(2)
                })
                .sum();

            let r_squared = 1.0 - (ss_res / ss_tot);
            result.push(r_squared.clamp(0.0, 1.0));
        }

        result
    }
}

impl TechnicalIndicator for RSquared {
    fn name(&self) -> &str {
        "R-Squared"
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

/// Adjusted R-Squared - R² adjusted for number of predictors.
///
/// Adj R² = 1 - ((1 - R²)(n - 1) / (n - k - 1))
///
/// Where k is the number of predictors (1 for simple linear regression).
#[derive(Debug, Clone)]
pub struct AdjustedRSquared {
    period: usize,
    predictors: usize,
}

impl AdjustedRSquared {
    pub fn new(period: usize) -> Self {
        Self { period, predictors: 1 }
    }

    pub fn with_predictors(period: usize, predictors: usize) -> Self {
        Self { period, predictors }
    }

    /// Calculate rolling adjusted R-squared.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let r_squared_calc = RSquared::new(self.period);
        let r_squared = r_squared_calc.calculate(data);

        r_squared.iter().map(|&r2| {
            if r2.is_nan() {
                f64::NAN
            } else {
                let n = self.period as f64;
                let k = self.predictors as f64;
                if n - k - 1.0 <= 0.0 {
                    f64::NAN
                } else {
                    1.0 - ((1.0 - r2) * (n - 1.0) / (n - k - 1.0))
                }
            }
        }).collect()
    }
}

impl TechnicalIndicator for AdjustedRSquared {
    fn name(&self) -> &str {
        "Adjusted R-Squared"
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
    fn test_mad() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mad = MAD::new(5);
        let result = mad.calculate(&data);

        assert_eq!(result.len(), 5);
        // MAD of [1,2,3,4,5] with mean 3: |1-3| + |2-3| + |3-3| + |4-3| + |5-3| = 2+1+0+1+2 = 6/5 = 1.2
        assert!((result[4] - 1.2).abs() < 0.001);
    }

    #[test]
    fn test_t_statistic() {
        // Data with positive mean
        let data = vec![10.0, 11.0, 12.0, 9.0, 10.5, 11.5, 10.2];
        let t_stat = TStatistic::with_reference(5, 10.0);
        let result = t_stat.calculate(&data);

        assert_eq!(result.len(), 7);
        // Should have a positive t-value if mean > reference
    }

    #[test]
    fn test_p_value() {
        let data = vec![0.1, -0.05, 0.08, 0.12, -0.02, 0.09, 0.03];
        let p_val = PValue::new(5);
        let result = p_val.calculate(&data);

        assert_eq!(result.len(), 7);
        // P-values should be between 0 and 1
        for val in &result[4..] {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 1.0);
            }
        }
    }

    #[test]
    fn test_confidence_interval() {
        let data: Vec<f64> = (0..20).map(|x| 100.0 + x as f64 * 0.5).collect();
        let ci = ConfidenceInterval::new(10);
        let output = ci.calculate(&data);

        assert_eq!(output.lower.len(), 20);
        assert_eq!(output.mean.len(), 20);
        assert_eq!(output.upper.len(), 20);

        // Lower < Mean < Upper
        for i in 9..20 {
            if !output.lower[i].is_nan() {
                assert!(output.lower[i] < output.mean[i]);
                assert!(output.mean[i] < output.upper[i]);
            }
        }
    }

    #[test]
    fn test_r_squared() {
        // Perfect linear data should have R² ≈ 1
        let linear_data: Vec<f64> = (0..20).map(|x| 100.0 + x as f64 * 2.0).collect();
        let r2 = RSquared::new(10);
        let result = r2.calculate(&linear_data);

        assert_eq!(result.len(), 20);
        // R² should be close to 1 for perfect linear data
        assert!((result[19] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_adjusted_r_squared() {
        let linear_data: Vec<f64> = (0..20).map(|x| 100.0 + x as f64 * 2.0).collect();
        let adj_r2 = AdjustedRSquared::new(10);
        let result = adj_r2.calculate(&linear_data);

        assert_eq!(result.len(), 20);
        // Adjusted R² should also be close to 1 for perfect linear data
        assert!((result[19] - 1.0).abs() < 0.01);
    }
}
