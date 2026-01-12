//! Ensemble methods for combining predictions

use automl_api::EnsembleMethod;
use automl_spi::EnsembleCombiner;

/// Ensemble combiner using simple averaging
#[derive(Debug, Clone, Default)]
pub struct EnsembleAverager;

impl EnsembleCombiner for EnsembleAverager {
    fn combine(&self, predictions: &[Vec<f64>], _weights: Option<&[f64]>) -> Vec<f64> {
        if predictions.is_empty() {
            return Vec::new();
        }
        let n_steps = predictions[0].len();

        (0..n_steps)
            .map(|i| {
                let sum: f64 = predictions.iter().map(|p| p[i]).sum();
                sum / predictions.len() as f64
            })
            .collect()
    }
}

/// Ensemble combiner using weighted averaging
#[derive(Debug, Clone, Default)]
pub struct EnsembleWeighted;

impl EnsembleCombiner for EnsembleWeighted {
    fn combine(&self, predictions: &[Vec<f64>], weights: Option<&[f64]>) -> Vec<f64> {
        if predictions.is_empty() {
            return Vec::new();
        }
        let n_steps = predictions[0].len();

        let default_weights = vec![1.0 / predictions.len() as f64; predictions.len()];
        let w = weights.unwrap_or(&default_weights);

        (0..n_steps)
            .map(|i| predictions.iter().zip(w.iter()).map(|(p, &wt)| p[i] * wt).sum())
            .collect()
    }
}

/// Ensemble combiner using median
#[derive(Debug, Clone, Default)]
pub struct EnsembleMedian;

impl EnsembleCombiner for EnsembleMedian {
    fn combine(&self, predictions: &[Vec<f64>], _weights: Option<&[f64]>) -> Vec<f64> {
        if predictions.is_empty() {
            return Vec::new();
        }
        let n_steps = predictions[0].len();

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

/// Combine predictions using specified method
pub fn combine_predictions(
    predictions: &[Vec<f64>],
    method: EnsembleMethod,
    weights: Option<&[f64]>,
) -> Vec<f64> {
    match method {
        EnsembleMethod::Average => EnsembleAverager.combine(predictions, weights),
        EnsembleMethod::WeightedAverage => EnsembleWeighted.combine(predictions, weights),
        EnsembleMethod::Median => EnsembleMedian.combine(predictions, weights),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_average() {
        let predictions = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]];
        let result = combine_predictions(&predictions, EnsembleMethod::Average, None);
        assert_eq!(result, vec![1.5, 3.0, 4.5]);
    }

    #[test]
    fn test_weighted_average() {
        let predictions = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]];
        let weights = vec![0.25, 0.75];
        let result =
            combine_predictions(&predictions, EnsembleMethod::WeightedAverage, Some(&weights));
        // 0.25*1 + 0.75*2 = 1.75, 0.25*2 + 0.75*4 = 3.5, 0.25*3 + 0.75*6 = 5.25
        assert_eq!(result, vec![1.75, 3.5, 5.25]);
    }

    #[test]
    fn test_median() {
        let predictions = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![5.0, 5.0]];
        let result = combine_predictions(&predictions, EnsembleMethod::Median, None);
        assert_eq!(result, vec![2.0, 4.0]); // middle values
    }

    #[test]
    fn test_empty_predictions() {
        let predictions: Vec<Vec<f64>> = vec![];
        let result = combine_predictions(&predictions, EnsembleMethod::Average, None);
        assert!(result.is_empty());
    }
}
