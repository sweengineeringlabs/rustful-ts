//! Seasonality detection implementations
//!
//! Provides autocorrelation-based seasonality detection.

use forecast_spi::SeasonalityDetector;

/// Autocorrelation-based seasonality detector
pub struct AutocorrelationDetector {
    /// Minimum autocorrelation threshold to consider significant
    threshold: f64,
}

impl AutocorrelationDetector {
    pub fn new() -> Self {
        Self { threshold: 0.3 }
    }

    pub fn with_threshold(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl Default for AutocorrelationDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl SeasonalityDetector for AutocorrelationDetector {
    fn detect(&self, data: &[f64], max_period: usize) -> Option<usize> {
        detect_seasonality_with_threshold(data, max_period, self.threshold)
    }
}

/// Detect seasonality period using autocorrelation with default threshold
pub fn detect_seasonality(data: &[f64], max_period: usize) -> Option<usize> {
    detect_seasonality_with_threshold(data, max_period, 0.3)
}

/// Detect seasonality period using autocorrelation with custom threshold
pub fn detect_seasonality_with_threshold(
    data: &[f64],
    max_period: usize,
    threshold: f64,
) -> Option<usize> {
    if data.len() < max_period * 2 {
        return None;
    }

    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();

    if var == 0.0 {
        return None;
    }

    let mut best_period = 0;
    let mut best_acf = 0.0;

    for lag in 2..=max_period.min(n / 2) {
        let acf: f64 = data
            .iter()
            .take(n - lag)
            .zip(data.iter().skip(lag))
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum::<f64>()
            / var;

        if acf > best_acf && acf > threshold {
            best_acf = acf;
            best_period = lag;
        }
    }

    if best_period > 0 {
        Some(best_period)
    } else {
        None
    }
}

/// Compute autocorrelation function for a time series
pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();

    if var == 0.0 {
        return vec![1.0; max_lag + 1];
    }

    (0..=max_lag.min(n - 1))
        .map(|lag| {
            if lag == 0 {
                1.0
            } else {
                data.iter()
                    .take(n - lag)
                    .zip(data.iter().skip(lag))
                    .map(|(a, b)| (a - mean) * (b - mean))
                    .sum::<f64>()
                    / var
            }
        })
        .collect()
}

/// Find all significant peaks in the autocorrelation function
pub fn find_seasonal_peaks(data: &[f64], max_lag: usize, threshold: f64) -> Vec<usize> {
    let acf = autocorrelation(data, max_lag);
    let mut peaks = Vec::new();

    for i in 2..acf.len().saturating_sub(1) {
        // Check if this is a local maximum above threshold
        if acf[i] > threshold && acf[i] > acf[i - 1] && acf[i] > acf[i + 1] {
            peaks.push(i);
        }
    }

    peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_seasonality() {
        // Create data with period 4
        let data: Vec<f64> = (0..40)
            .map(|i| (i % 4) as f64 * 10.0 + (i as f64) * 0.1)
            .collect();

        let period = detect_seasonality(&data, 10);
        assert!(period.is_some());
        // Period should be detected as 4 (or a multiple)
        if let Some(p) = period {
            assert!(p == 4 || p % 4 == 0);
        }
    }

    #[test]
    fn test_short_data() {
        let data = vec![1.0, 2.0, 3.0];
        let period = detect_seasonality(&data, 10);
        assert!(period.is_none());
    }

    #[test]
    fn test_constant_data() {
        let data = vec![5.0; 100];
        let period = detect_seasonality(&data, 10);
        assert!(period.is_none());
    }

    #[test]
    fn test_autocorrelation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let acf = autocorrelation(&data, 3);
        assert_eq!(acf[0], 1.0); // ACF at lag 0 is always 1
        assert!(acf.len() == 4);
    }
}
