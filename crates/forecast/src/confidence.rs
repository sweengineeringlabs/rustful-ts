//! Confidence intervals for forecasts

use serde::{Deserialize, Serialize};

/// Forecast with confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastWithConfidence {
    /// Point forecast
    pub forecast: Vec<f64>,
    /// Lower bound of confidence interval
    pub lower: Vec<f64>,
    /// Upper bound of confidence interval
    pub upper: Vec<f64>,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

impl ForecastWithConfidence {
    /// Create from point forecast and standard errors
    pub fn from_standard_errors(
        forecast: Vec<f64>,
        std_errors: &[f64],
        confidence_level: f64,
    ) -> Self {
        // Z-score for confidence level (approximate)
        let z = match confidence_level {
            x if x >= 0.99 => 2.576,
            x if x >= 0.95 => 1.96,
            x if x >= 0.90 => 1.645,
            x if x >= 0.80 => 1.282,
            _ => 1.96, // default to 95%
        };

        let lower = forecast
            .iter()
            .zip(std_errors.iter())
            .map(|(&f, &se)| f - z * se)
            .collect();

        let upper = forecast
            .iter()
            .zip(std_errors.iter())
            .map(|(&f, &se)| f + z * se)
            .collect();

        Self {
            forecast,
            lower,
            upper,
            confidence_level,
        }
    }

    /// Create confidence intervals based on historical residuals
    pub fn from_residuals(
        forecast: Vec<f64>,
        residuals: &[f64],
        confidence_level: f64,
    ) -> Self {
        // Calculate standard deviation of residuals
        let n = residuals.len() as f64;
        let mean = residuals.iter().sum::<f64>() / n;
        let variance = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        // Assume constant standard error for all forecast horizons
        let std_errors: Vec<f64> = (0..forecast.len())
            .map(|h| std_dev * ((h + 1) as f64).sqrt())
            .collect();

        Self::from_standard_errors(forecast, &std_errors, confidence_level)
    }
}

/// Calculate prediction intervals using bootstrap
pub fn bootstrap_prediction_interval(
    forecasts: &[Vec<f64>],
    confidence_level: f64,
) -> ForecastWithConfidence {
    if forecasts.is_empty() {
        return ForecastWithConfidence {
            forecast: vec![],
            lower: vec![],
            upper: vec![],
            confidence_level,
        };
    }

    let n_steps = forecasts[0].len();
    let n_samples = forecasts.len();

    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0) * n_samples as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_samples as f64).ceil() as usize;

    let mut forecast = Vec::with_capacity(n_steps);
    let mut lower = Vec::with_capacity(n_steps);
    let mut upper = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let mut values: Vec<f64> = forecasts.iter().map(|f| f[step]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        forecast.push(values.iter().sum::<f64>() / n_samples as f64);
        lower.push(values.get(lower_idx).copied().unwrap_or(values[0]));
        upper.push(values.get(upper_idx.min(n_samples - 1)).copied().unwrap_or(values[n_samples - 1]));
    }

    ForecastWithConfidence {
        forecast,
        lower,
        upper,
        confidence_level,
    }
}

