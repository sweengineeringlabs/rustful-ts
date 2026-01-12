//! Time series decomposition implementations
//!
//! Provides additive and multiplicative decomposition methods.

use forecast_spi::{DecompositionResult, Decomposer};
use serde::{Deserialize, Serialize};

/// Decomposed time series components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

impl From<DecompositionResult> for Decomposition {
    fn from(result: DecompositionResult) -> Self {
        Self {
            trend: result.trend,
            seasonal: result.seasonal,
            residual: result.residual,
        }
    }
}

/// Additive decomposition: Y = T + S + R
pub struct AdditiveDecomposer;

impl AdditiveDecomposer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AdditiveDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl Decomposer for AdditiveDecomposer {
    fn decompose(&self, data: &[f64], period: usize) -> DecompositionResult {
        decompose_additive(data, period)
    }
}

/// Perform additive decomposition
pub fn decompose_additive(data: &[f64], period: usize) -> DecompositionResult {
    let n = data.len();
    if n < period * 2 {
        return DecompositionResult {
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

    DecompositionResult {
        trend,
        seasonal,
        residual,
    }
}

/// Multiplicative decomposition: Y = T * S * R
pub struct MultiplicativeDecomposer;

impl MultiplicativeDecomposer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MultiplicativeDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl Decomposer for MultiplicativeDecomposer {
    fn decompose(&self, data: &[f64], period: usize) -> DecompositionResult {
        decompose_multiplicative(data, period)
    }
}

/// Perform multiplicative decomposition
pub fn decompose_multiplicative(data: &[f64], period: usize) -> DecompositionResult {
    let n = data.len();
    if n < period * 2 || data.iter().any(|&x| x <= 0.0) {
        // Fall back to additive for non-positive data
        return decompose_additive(data, period);
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

    // Detrend (divide)
    let detrended: Vec<f64> = data
        .iter()
        .zip(trend.iter())
        .map(|(d, t)| if *t != 0.0 { d / t } else { 1.0 })
        .collect();

    // Seasonal component (average by period position)
    let mut seasonal = vec![1.0; n];
    for pos in 0..period {
        let values: Vec<f64> = detrended.iter().skip(pos).step_by(period).copied().collect();
        let avg = values.iter().sum::<f64>() / values.len() as f64;
        for (i, s) in seasonal.iter_mut().skip(pos).step_by(period).enumerate() {
            if i < values.len() {
                *s = avg;
            }
        }
    }

    // Residual (Y / T / S)
    let residual: Vec<f64> = data
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((d, t), s)| {
            let ts = t * s;
            if ts != 0.0 {
                d / ts
            } else {
                1.0
            }
        })
        .collect();

    DecompositionResult {
        trend,
        seasonal,
        residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_additive_decomposition() {
        // Create simple seasonal data
        let data: Vec<f64> = (0..24)
            .map(|i| 100.0 + (i as f64) * 2.0 + (i % 4) as f64 * 10.0)
            .collect();

        let result = decompose_additive(&data, 4);
        assert_eq!(result.trend.len(), data.len());
        assert_eq!(result.seasonal.len(), data.len());
        assert_eq!(result.residual.len(), data.len());
    }

    #[test]
    fn test_short_data() {
        let data = vec![1.0, 2.0, 3.0];
        let result = decompose_additive(&data, 4);
        assert_eq!(result.trend, data);
        assert_eq!(result.seasonal, vec![0.0, 0.0, 0.0]);
    }
}
