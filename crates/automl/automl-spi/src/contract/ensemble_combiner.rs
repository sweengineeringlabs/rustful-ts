//! Ensemble combination trait for AutoML.

/// Trait for ensemble combiners.
pub trait EnsembleCombiner {
    /// Combine multiple predictions into a single prediction.
    fn combine(&self, predictions: &[Vec<f64>], weights: Option<&[f64]>) -> Vec<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Mock Implementations ==========

    /// A simple average combiner.
    struct MeanCombiner;

    impl EnsembleCombiner for MeanCombiner {
        fn combine(&self, predictions: &[Vec<f64>], weights: Option<&[f64]>) -> Vec<f64> {
            if predictions.is_empty() {
                return vec![];
            }

            let len = predictions[0].len();
            if len == 0 {
                return vec![];
            }

            match weights {
                Some(w) => {
                    // Weighted average
                    let total_weight: f64 = w.iter().sum();
                    if total_weight == 0.0 {
                        return vec![0.0; len];
                    }

                    (0..len)
                        .map(|i| {
                            predictions
                                .iter()
                                .zip(w.iter())
                                .map(|(pred, weight)| pred.get(i).unwrap_or(&0.0) * weight)
                                .sum::<f64>()
                                / total_weight
                        })
                        .collect()
                }
                None => {
                    // Simple average
                    let n = predictions.len() as f64;
                    (0..len)
                        .map(|i| {
                            predictions
                                .iter()
                                .map(|pred| pred.get(i).unwrap_or(&0.0))
                                .sum::<f64>()
                                / n
                        })
                        .collect()
                }
            }
        }
    }

    /// A median combiner.
    struct MedianCombiner;

    impl EnsembleCombiner for MedianCombiner {
        fn combine(&self, predictions: &[Vec<f64>], _weights: Option<&[f64]>) -> Vec<f64> {
            if predictions.is_empty() {
                return vec![];
            }

            let len = predictions[0].len();
            if len == 0 {
                return vec![];
            }

            (0..len)
                .map(|i| {
                    let mut values: Vec<f64> = predictions
                        .iter()
                        .filter_map(|pred| pred.get(i).copied())
                        .collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    let mid = values.len() / 2;
                    if values.len() % 2 == 0 {
                        (values[mid - 1] + values[mid]) / 2.0
                    } else {
                        values[mid]
                    }
                })
                .collect()
        }
    }

    /// A combiner that selects the best prediction (lowest variance).
    struct BestModelCombiner;

    impl EnsembleCombiner for BestModelCombiner {
        fn combine(&self, predictions: &[Vec<f64>], weights: Option<&[f64]>) -> Vec<f64> {
            if predictions.is_empty() {
                return vec![];
            }

            // If weights provided, select the one with highest weight
            if let Some(w) = weights {
                let best_idx = w
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                return predictions.get(best_idx).cloned().unwrap_or_default();
            }

            // Otherwise, select first prediction (mock behavior)
            predictions[0].clone()
        }
    }

    /// A stacking combiner that uses a meta-model.
    struct StackingCombiner {
        meta_weights: Vec<f64>,
    }

    impl EnsembleCombiner for StackingCombiner {
        fn combine(&self, predictions: &[Vec<f64>], _weights: Option<&[f64]>) -> Vec<f64> {
            if predictions.is_empty() {
                return vec![];
            }

            let len = predictions[0].len();
            if len == 0 {
                return vec![];
            }

            // Use meta_weights to combine predictions
            let weights = &self.meta_weights;
            let total_weight: f64 = weights.iter().sum();

            if total_weight == 0.0 {
                return vec![0.0; len];
            }

            (0..len)
                .map(|i| {
                    predictions
                        .iter()
                        .zip(weights.iter())
                        .map(|(pred, w)| pred.get(i).unwrap_or(&0.0) * w)
                        .sum::<f64>()
                        / total_weight
                })
                .collect()
        }
    }

    /// A trimmed mean combiner that excludes extremes.
    struct TrimmedMeanCombiner {
        trim_percent: f64,
    }

    impl EnsembleCombiner for TrimmedMeanCombiner {
        fn combine(&self, predictions: &[Vec<f64>], _weights: Option<&[f64]>) -> Vec<f64> {
            if predictions.is_empty() {
                return vec![];
            }

            let len = predictions[0].len();
            if len == 0 {
                return vec![];
            }

            let n = predictions.len();
            let trim_count = ((n as f64 * self.trim_percent) / 2.0).ceil() as usize;

            if trim_count * 2 >= n {
                // Not enough predictions to trim, use simple mean
                return MeanCombiner.combine(predictions, None);
            }

            (0..len)
                .map(|i| {
                    let mut values: Vec<f64> = predictions
                        .iter()
                        .filter_map(|pred| pred.get(i).copied())
                        .collect();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    let trimmed: Vec<f64> = values[trim_count..n - trim_count].to_vec();
                    if trimmed.is_empty() {
                        0.0
                    } else {
                        trimmed.iter().sum::<f64>() / trimmed.len() as f64
                    }
                })
                .collect()
        }
    }

    // ========== Basic Functionality Tests ==========

    #[test]
    fn test_mean_combiner_simple() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]];
        let result = combiner.combine(&predictions, None);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
        assert!((result[1] - 3.0).abs() < f64::EPSILON);
        assert!((result[2] - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mean_combiner_weighted() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![3.0, 1.0]; // 75% weight on first, 25% on second
        let result = combiner.combine(&predictions, Some(&weights));

        assert_eq!(result.len(), 2);
        // (1.0 * 3 + 3.0 * 1) / 4 = 6 / 4 = 1.5
        assert!((result[0] - 1.5).abs() < f64::EPSILON);
        // (2.0 * 3 + 4.0 * 1) / 4 = 10 / 4 = 2.5
        assert!((result[1] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_median_combiner_odd() {
        let combiner = MedianCombiner;
        let predictions = vec![vec![1.0], vec![5.0], vec![3.0]];
        let result = combiner.combine(&predictions, None);

        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_median_combiner_even() {
        let combiner = MedianCombiner;
        let predictions = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let result = combiner.combine(&predictions, None);

        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_best_model_combiner_with_weights() {
        let combiner = BestModelCombiner;
        let predictions = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let weights = vec![0.2, 0.5, 0.3]; // Second model has highest weight
        let result = combiner.combine(&predictions, Some(&weights));

        assert_eq!(result, vec![3.0, 4.0]);
    }

    #[test]
    fn test_best_model_combiner_without_weights() {
        let combiner = BestModelCombiner;
        let predictions = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = combiner.combine(&predictions, None);

        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn test_stacking_combiner() {
        let combiner = StackingCombiner {
            meta_weights: vec![0.6, 0.4],
        };
        let predictions = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = combiner.combine(&predictions, None);

        assert_eq!(result.len(), 2);
        // (1.0 * 0.6 + 3.0 * 0.4) / 1.0 = 1.8
        assert!((result[0] - 1.8).abs() < 1e-10);
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_empty_predictions() {
        let combiner = MeanCombiner;
        let predictions: Vec<Vec<f64>> = vec![];
        let result = combiner.combine(&predictions, None);

        assert!(result.is_empty());
    }

    #[test]
    fn test_empty_inner_predictions() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![], vec![]];
        let result = combiner.combine(&predictions, None);

        assert!(result.is_empty());
    }

    #[test]
    fn test_single_prediction() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![1.0, 2.0, 3.0]];
        let result = combiner.combine(&predictions, None);

        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_single_value_predictions() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![1.0], vec![2.0], vec![3.0]];
        let result = combiner.combine(&predictions, None);

        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_zero_weights() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![0.0, 0.0];
        let result = combiner.combine(&predictions, Some(&weights));

        // Should return zeros when total weight is 0
        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.0).abs() < f64::EPSILON);
        assert!((result[1] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unequal_weight_sum() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![10.0], vec![20.0]];
        let weights = vec![1.0, 3.0]; // Sum is 4, not 1
        let result = combiner.combine(&predictions, Some(&weights));

        // (10 * 1 + 20 * 3) / 4 = 70 / 4 = 17.5
        assert!((result[0] - 17.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_negative_predictions() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![-1.0, -2.0], vec![-3.0, -4.0]];
        let result = combiner.combine(&predictions, None);

        assert!((result[0] - (-2.0)).abs() < f64::EPSILON);
        assert!((result[1] - (-3.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mixed_sign_predictions() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![-1.0, 2.0], vec![3.0, -4.0]];
        let result = combiner.combine(&predictions, None);

        assert!((result[0] - 1.0).abs() < f64::EPSILON);
        assert!((result[1] - (-1.0)).abs() < f64::EPSILON);
    }

    // ========== Trimmed Mean Tests ==========

    #[test]
    fn test_trimmed_mean_combiner() {
        let combiner = TrimmedMeanCombiner { trim_percent: 0.2 };
        let predictions = vec![
            vec![1.0],  // Will be trimmed (min)
            vec![5.0],
            vec![6.0],
            vec![7.0],
            vec![100.0], // Will be trimmed (max)
        ];
        let result = combiner.combine(&predictions, None);

        // After trimming 1 from each end: mean of [5, 6, 7] = 6
        assert!((result[0] - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trimmed_mean_not_enough_predictions() {
        let combiner = TrimmedMeanCombiner { trim_percent: 0.5 };
        let predictions = vec![vec![1.0], vec![2.0]];
        let result = combiner.combine(&predictions, None);

        // Falls back to simple mean
        assert!((result[0] - 1.5).abs() < f64::EPSILON);
    }

    // ========== Trait Object Tests ==========

    #[test]
    fn test_combiner_as_trait_object() {
        let combiner: Box<dyn EnsembleCombiner> = Box::new(MeanCombiner);
        let predictions = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = combiner.combine(&predictions, None);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multiple_combiners_as_trait_objects() {
        let combiners: Vec<Box<dyn EnsembleCombiner>> = vec![
            Box::new(MeanCombiner),
            Box::new(MedianCombiner),
            Box::new(BestModelCombiner),
        ];

        let predictions = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];

        for combiner in combiners {
            let result = combiner.combine(&predictions, None);
            assert_eq!(result.len(), 3);
        }
    }

    // ========== Large Dataset Tests ==========

    #[test]
    fn test_many_predictions() {
        let combiner = MeanCombiner;
        let predictions: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        let result = combiner.combine(&predictions, None);

        // Mean of 0..99 = 49.5
        assert!((result[0] - 49.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_long_predictions() {
        let combiner = MeanCombiner;
        let predictions = vec![
            (0..1000).map(|i| i as f64).collect(),
            (0..1000).map(|i| (i + 1) as f64).collect(),
        ];
        let result = combiner.combine(&predictions, None);

        assert_eq!(result.len(), 1000);
        // First element: mean of 0 and 1 = 0.5
        assert!((result[0] - 0.5).abs() < f64::EPSILON);
        // Last element: mean of 999 and 1000 = 999.5
        assert!((result[999] - 999.5).abs() < f64::EPSILON);
    }

    // ========== Weights Edge Cases ==========

    #[test]
    fn test_single_weight() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![5.0]];
        let weights = vec![1.0];
        let result = combiner.combine(&predictions, Some(&weights));

        assert!((result[0] - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_very_small_weights() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![1.0], vec![2.0]];
        let weights = vec![1e-10, 1e-10];
        let result = combiner.combine(&predictions, Some(&weights));

        // Should still produce valid result
        assert!((result[0] - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_very_large_weights() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![1.0], vec![2.0]];
        let weights = vec![1e10, 1e10];
        let result = combiner.combine(&predictions, Some(&weights));

        assert!((result[0] - 1.5).abs() < f64::EPSILON);
    }

    // ========== Specific Value Tests ==========

    #[test]
    fn test_all_same_predictions() {
        let combiner = MeanCombiner;
        let predictions = vec![vec![5.0, 5.0], vec![5.0, 5.0], vec![5.0, 5.0]];
        let result = combiner.combine(&predictions, None);

        assert_eq!(result, vec![5.0, 5.0]);
    }

    #[test]
    fn test_median_with_duplicates() {
        let combiner = MedianCombiner;
        let predictions = vec![vec![1.0], vec![1.0], vec![1.0], vec![10.0], vec![10.0]];
        let result = combiner.combine(&predictions, None);

        // Sorted: [1, 1, 1, 10, 10], median is middle element = 1
        assert!((result[0] - 1.0).abs() < f64::EPSILON);
    }

    // ========== Generic Function Tests ==========

    #[test]
    fn test_generic_combine_function() {
        fn combine_with<C: EnsembleCombiner>(
            combiner: &C,
            predictions: &[Vec<f64>],
        ) -> Vec<f64> {
            combiner.combine(predictions, None)
        }

        let mean = MeanCombiner;
        let predictions = vec![vec![1.0], vec![3.0]];
        let result = combine_with(&mean, &predictions);

        assert!((result[0] - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_combiner_chain() {
        // Combine predictions with mean, then median
        let predictions = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];

        let mean_combiner = MeanCombiner;
        let mean_result = mean_combiner.combine(&predictions, None);
        // mean_result = [4.0, 5.0]

        let median_combiner = MedianCombiner;
        let median_result = median_combiner.combine(&predictions, None);
        // For each position, median of [1,3,5,7] = 4 and [2,4,6,8] = 5

        assert_eq!(mean_result.len(), median_result.len());
        assert!((mean_result[0] - 4.0).abs() < f64::EPSILON);
        assert!((median_result[0] - 4.0).abs() < f64::EPSILON);
    }
}
