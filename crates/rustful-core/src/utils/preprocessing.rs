//! Data preprocessing utilities for time series
//!
//! Functions for cleaning, transforming, and preparing time series data.

/// Normalize data to [0, 1] range (min-max scaling)
pub fn normalize(data: &[f64]) -> (Vec<f64>, f64, f64) {
    if data.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }

    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    if range.abs() < 1e-10 {
        return (vec![0.5; data.len()], min, max);
    }

    let normalized: Vec<f64> = data.iter().map(|x| (x - min) / range).collect();

    (normalized, min, max)
}

/// Denormalize data from [0, 1] range
pub fn denormalize(data: &[f64], min: f64, max: f64) -> Vec<f64> {
    let range = max - min;
    data.iter().map(|x| x * range + min).collect()
}

/// Standardize data to zero mean and unit variance (z-score)
pub fn standardize(data: &[f64]) -> (Vec<f64>, f64, f64) {
    if data.is_empty() {
        return (Vec::new(), 0.0, 1.0);
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev < 1e-10 {
        return (vec![0.0; data.len()], mean, 1.0);
    }

    let standardized: Vec<f64> = data.iter().map(|x| (x - mean) / std_dev).collect();

    (standardized, mean, std_dev)
}

/// Destandardize data
pub fn destandardize(data: &[f64], mean: f64, std_dev: f64) -> Vec<f64> {
    data.iter().map(|x| x * std_dev + mean).collect()
}

/// Apply log transformation (log1p for handling zeros)
pub fn log_transform(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| (x + 1.0).ln()).collect()
}

/// Inverse log transformation
pub fn inverse_log_transform(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.exp() - 1.0).collect()
}

/// Apply Box-Cox transformation
///
/// # Arguments
///
/// * `data` - Input data (must be positive)
/// * `lambda` - Transformation parameter (0 = log transform)
pub fn box_cox(data: &[f64], lambda: f64) -> Vec<f64> {
    if lambda.abs() < 1e-10 {
        data.iter().map(|x| x.ln()).collect()
    } else {
        data.iter().map(|x| (x.powf(lambda) - 1.0) / lambda).collect()
    }
}

/// Inverse Box-Cox transformation
pub fn inverse_box_cox(data: &[f64], lambda: f64) -> Vec<f64> {
    if lambda.abs() < 1e-10 {
        data.iter().map(|x| x.exp()).collect()
    } else {
        data.iter()
            .map(|x| (x * lambda + 1.0).powf(1.0 / lambda))
            .collect()
    }
}

/// Remove NaN and infinite values
pub fn clean_data(data: &[f64]) -> Vec<f64> {
    data.iter()
        .filter(|x| x.is_finite())
        .cloned()
        .collect()
}

/// Interpolate missing values (marked as NaN)
pub fn interpolate_linear(data: &[f64]) -> Vec<f64> {
    let mut result = data.to_vec();
    let n = result.len();

    for i in 0..n {
        if result[i].is_nan() {
            // Find previous valid value
            let prev_idx = (0..i).rev().find(|&j| !result[j].is_nan());
            // Find next valid value
            let next_idx = ((i + 1)..n).find(|&j| !result[j].is_nan());

            result[i] = match (prev_idx, next_idx) {
                (Some(p), Some(n_idx)) => {
                    let ratio = (i - p) as f64 / (n_idx - p) as f64;
                    result[p] + ratio * (result[n_idx] - result[p])
                }
                (Some(p), None) => result[p],
                (None, Some(n_idx)) => result[n_idx],
                (None, None) => 0.0,
            };
        }
    }

    result
}

/// Detect outliers using IQR method
///
/// Returns indices of outlier points
pub fn detect_outliers_iqr(data: &[f64], multiplier: f64) -> Vec<usize> {
    let mut sorted: Vec<f64> = data.iter().filter(|x| x.is_finite()).cloned().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if sorted.len() < 4 {
        return Vec::new();
    }

    let q1 = sorted[sorted.len() / 4];
    let q3 = sorted[3 * sorted.len() / 4];
    let iqr = q3 - q1;

    let lower_bound = q1 - multiplier * iqr;
    let upper_bound = q3 + multiplier * iqr;

    data.iter()
        .enumerate()
        .filter(|(_, x)| **x < lower_bound || **x > upper_bound)
        .map(|(i, _)| i)
        .collect()
}

/// Compute first-order differences
pub fn difference(data: &[f64], order: usize) -> Vec<f64> {
    let mut result = data.to_vec();
    for _ in 0..order {
        if result.len() <= 1 {
            return Vec::new();
        }
        result = result.windows(2).map(|w| w[1] - w[0]).collect();
    }
    result
}

/// Compute seasonal differences
pub fn seasonal_difference(data: &[f64], period: usize) -> Vec<f64> {
    if data.len() <= period {
        return Vec::new();
    }

    data.iter()
        .skip(period)
        .zip(data.iter())
        .map(|(curr, prev)| curr - prev)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let data = vec![0.0, 5.0, 10.0];
        let (normalized, min, max) = normalize(&data);

        assert!((min - 0.0).abs() < 1e-10);
        assert!((max - 10.0).abs() < 1e-10);
        assert!((normalized[0] - 0.0).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
        assert!((normalized[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_standardize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (standardized, mean, std_dev) = standardize(&data);

        assert!((mean - 3.0).abs() < 1e-10);
        assert!((standardized.iter().sum::<f64>()).abs() < 1e-10);

        let recovered = destandardize(&standardized, mean, std_dev);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-10);
        }
    }

    #[test]
    fn test_difference() {
        let data = vec![1.0, 3.0, 6.0, 10.0];
        let diff = difference(&data, 1);
        assert_eq!(diff, vec![2.0, 3.0, 4.0]);

        let diff2 = difference(&data, 2);
        assert_eq!(diff2, vec![1.0, 1.0]);
    }
}
