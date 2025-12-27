//! Seasonality detection

/// Detect seasonality period using autocorrelation
pub fn detect_seasonality(data: &[f64], max_period: usize) -> Option<usize> {
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

        if acf > best_acf && acf > 0.3 {
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
