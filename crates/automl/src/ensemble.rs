//! Ensemble methods

use serde::{Deserialize, Serialize};

/// Ensemble combination method
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EnsembleMethod {
    Average,
    WeightedAverage,
    Median,
}

/// Combine predictions using specified method
pub fn combine_predictions(predictions: &[Vec<f64>], method: EnsembleMethod, weights: Option<&[f64]>) -> Vec<f64> {
    if predictions.is_empty() {
        return Vec::new();
    }
    let n_steps = predictions[0].len();

    match method {
        EnsembleMethod::Average => {
            (0..n_steps)
                .map(|i| {
                    let sum: f64 = predictions.iter().map(|p| p[i]).sum();
                    sum / predictions.len() as f64
                })
                .collect()
        }
        EnsembleMethod::WeightedAverage => {
            let default_weights = vec![1.0 / predictions.len() as f64; predictions.len()];
            let w = weights.unwrap_or(&default_weights);
            (0..n_steps)
                .map(|i| {
                    predictions.iter().zip(w.iter()).map(|(p, &wt)| p[i] * wt).sum()
                })
                .collect()
        }
        EnsembleMethod::Median => {
            (0..n_steps)
                .map(|i| {
                    let mut vals: Vec<f64> = predictions.iter().map(|p| p[i]).collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = vals.len() / 2;
                    if vals.len() % 2 == 0 {
                        (vals[mid - 1] + vals[mid]) / 2.0
                    } else {
                        vals[mid]
                    }
                })
                .collect()
        }
    }
}
