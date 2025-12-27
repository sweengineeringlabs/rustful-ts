//! Time series decomposition

use serde::{Deserialize, Serialize};

/// Decomposed time series components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

/// Perform additive decomposition
pub fn decompose_additive(data: &[f64], period: usize) -> Decomposition {
    let n = data.len();
    if n < period * 2 {
        return Decomposition {
            trend: data.to_vec(),
            seasonal: vec![0.0; n],
            residual: vec![0.0; n],
        };
    }

    // Simple moving average for trend
    let mut trend = vec![0.0; n];
    let half = period / 2;
    for i in half..(n - half) {
        let sum: f64 = data[i - half..=i + half].iter().sum();
        trend[i] = sum / period as f64;
    }
    // Extend trend at edges
    for i in 0..half {
        trend[i] = trend[half];
    }
    for i in (n - half)..n {
        trend[i] = trend[n - half - 1];
    }

    // Detrend
    let detrended: Vec<f64> = data.iter().zip(trend.iter()).map(|(d, t)| d - t).collect();

    // Seasonal component (average by period position)
    let mut seasonal = vec![0.0; n];
    for pos in 0..period {
        let values: Vec<f64> = detrended.iter().skip(pos).step_by(period).copied().collect();
        let avg = values.iter().sum::<f64>() / values.len() as f64;
        for (i, s) in seasonal.iter_mut().skip(pos).step_by(period).enumerate() {
            if i < values.len() {
                *s = avg;
            }
        }
    }

    // Residual
    let residual: Vec<f64> = data
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((d, t), s)| d - t - s)
        .collect();

    Decomposition {
        trend,
        seasonal,
        residual,
    }
}
