//! Model selection result types for AutoML.

use super::SelectedModel;

/// Result of model selection.
#[derive(Debug, Clone)]
pub struct ModelSelectionResult {
    /// The best model type selected.
    pub best_model: SelectedModel,
    /// Score of the best model (lower is better).
    pub score: f64,
    /// All evaluated models with their scores.
    pub all_scores: Vec<(SelectedModel, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Construction Tests ==========

    #[test]
    fn test_model_selection_result_construction() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.5 },
            score: 0.123,
            all_scores: vec![],
        };
        assert!((result.score - 0.123).abs() < f64::EPSILON);
        assert!(result.all_scores.is_empty());
    }

    #[test]
    fn test_model_selection_result_with_all_scores() {
        let all_scores = vec![
            (SelectedModel::SES { alpha: 0.5 }, 0.123),
            (SelectedModel::Holt { alpha: 0.3, beta: 0.2 }, 0.456),
            (SelectedModel::LinearRegression, 0.789),
        ];
        let result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.5 },
            score: 0.123,
            all_scores,
        };
        assert_eq!(result.all_scores.len(), 3);
    }

    #[test]
    fn test_model_selection_result_best_model_access() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::Arima { p: 1, d: 1, q: 1 },
            score: 0.05,
            all_scores: vec![],
        };
        if let SelectedModel::Arima { p, d, q } = result.best_model {
            assert_eq!(p, 1);
            assert_eq!(d, 1);
            assert_eq!(q, 1);
        } else {
            panic!("Expected Arima variant");
        }
    }

    // ========== Score Tests ==========

    #[test]
    fn test_model_selection_result_zero_score() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::LinearRegression,
            score: 0.0,
            all_scores: vec![],
        };
        assert!((result.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_selection_result_negative_score() {
        // Some metrics can be negative (e.g., log-likelihood)
        let result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.5 },
            score: -100.5,
            all_scores: vec![],
        };
        assert!(result.score < 0.0);
    }

    #[test]
    fn test_model_selection_result_large_score() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.5 },
            score: 1e10,
            all_scores: vec![],
        };
        assert!(result.score > 1e9);
    }

    #[test]
    fn test_model_selection_result_small_score() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.5 },
            score: 1e-10,
            all_scores: vec![],
        };
        assert!(result.score < 1e-9);
        assert!(result.score > 0.0);
    }

    // ========== All Scores Tests ==========

    #[test]
    fn test_model_selection_result_all_scores_ordering() {
        let all_scores = vec![
            (SelectedModel::SES { alpha: 0.5 }, 0.1),
            (SelectedModel::Holt { alpha: 0.3, beta: 0.2 }, 0.2),
            (SelectedModel::LinearRegression, 0.3),
        ];
        let result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.5 },
            score: 0.1,
            all_scores,
        };

        // Verify best_model has the lowest score
        for (_, score) in &result.all_scores {
            assert!(result.score <= *score);
        }
    }

    #[test]
    fn test_model_selection_result_all_scores_contains_best() {
        let all_scores = vec![
            (SelectedModel::SES { alpha: 0.5 }, 0.1),
            (SelectedModel::LinearRegression, 0.3),
        ];
        let result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.5 },
            score: 0.1,
            all_scores,
        };

        // Verify best model's score matches
        let best_in_all = result
            .all_scores
            .iter()
            .find(|(m, _)| matches!(m, SelectedModel::SES { .. }));
        assert!(best_in_all.is_some());
        assert!((best_in_all.unwrap().1 - result.score).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_selection_result_all_model_variants() {
        let all_scores = vec![
            (SelectedModel::Arima { p: 1, d: 1, q: 1 }, 0.1),
            (SelectedModel::SES { alpha: 0.5 }, 0.2),
            (SelectedModel::Holt { alpha: 0.3, beta: 0.2 }, 0.3),
            (
                SelectedModel::HoltWinters {
                    alpha: 0.3,
                    beta: 0.2,
                    gamma: 0.1,
                    period: 12,
                },
                0.4,
            ),
            (SelectedModel::LinearRegression, 0.5),
            (SelectedModel::KNN { k: 5, window: 10 }, 0.6),
        ];
        let result = ModelSelectionResult {
            best_model: SelectedModel::Arima { p: 1, d: 1, q: 1 },
            score: 0.1,
            all_scores,
        };

        assert_eq!(result.all_scores.len(), 6);
    }

    // ========== Clone Tests ==========

    #[test]
    fn test_model_selection_result_clone() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.5 },
            score: 0.123,
            all_scores: vec![
                (SelectedModel::SES { alpha: 0.5 }, 0.123),
                (SelectedModel::LinearRegression, 0.456),
            ],
        };
        let cloned = result.clone();

        assert_eq!(result.best_model.to_string(), cloned.best_model.to_string());
        assert!((result.score - cloned.score).abs() < f64::EPSILON);
        assert_eq!(result.all_scores.len(), cloned.all_scores.len());
    }

    #[test]
    fn test_model_selection_result_clone_independence() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.5 },
            score: 0.123,
            all_scores: vec![(SelectedModel::SES { alpha: 0.5 }, 0.123)],
        };
        let cloned = result.clone();

        // Verify cloned data is independent (can be used separately)
        assert_eq!(cloned.all_scores.len(), 1);
    }

    // ========== Debug Tests ==========

    #[test]
    fn test_model_selection_result_debug() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.5 },
            score: 0.123,
            all_scores: vec![],
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("ModelSelectionResult"));
        assert!(debug_str.contains("best_model"));
        assert!(debug_str.contains("score"));
        assert!(debug_str.contains("all_scores"));
    }

    #[test]
    fn test_model_selection_result_debug_with_data() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::Arima { p: 1, d: 2, q: 3 },
            score: 0.5,
            all_scores: vec![
                (SelectedModel::Arima { p: 1, d: 2, q: 3 }, 0.5),
                (SelectedModel::SES { alpha: 0.3 }, 0.8),
            ],
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("Arima"));
        assert!(debug_str.contains("SES"));
    }

    // ========== Field Accessibility Tests ==========

    #[test]
    fn test_model_selection_result_field_mutability() {
        let mut result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.5 },
            score: 0.123,
            all_scores: vec![],
        };

        // Modify fields (they are public)
        result.score = 0.456;
        result.best_model = SelectedModel::LinearRegression;
        result.all_scores.push((SelectedModel::LinearRegression, 0.456));

        assert!((result.score - 0.456).abs() < f64::EPSILON);
        assert!(matches!(result.best_model, SelectedModel::LinearRegression));
        assert_eq!(result.all_scores.len(), 1);
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_model_selection_result_empty_all_scores() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::LinearRegression,
            score: 0.5,
            all_scores: vec![],
        };
        assert!(result.all_scores.is_empty());
    }

    #[test]
    fn test_model_selection_result_single_model() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::LinearRegression,
            score: 0.5,
            all_scores: vec![(SelectedModel::LinearRegression, 0.5)],
        };
        assert_eq!(result.all_scores.len(), 1);
    }

    #[test]
    fn test_model_selection_result_many_models() {
        let mut all_scores = Vec::new();
        for i in 0..100 {
            all_scores.push((SelectedModel::SES { alpha: i as f64 / 100.0 }, i as f64));
        }
        let result = ModelSelectionResult {
            best_model: SelectedModel::SES { alpha: 0.0 },
            score: 0.0,
            all_scores,
        };
        assert_eq!(result.all_scores.len(), 100);
    }

    #[test]
    fn test_model_selection_result_nan_score() {
        // Although NaN is typically invalid, the struct should handle it
        let result = ModelSelectionResult {
            best_model: SelectedModel::LinearRegression,
            score: f64::NAN,
            all_scores: vec![],
        };
        assert!(result.score.is_nan());
    }

    #[test]
    fn test_model_selection_result_infinity_score() {
        let result = ModelSelectionResult {
            best_model: SelectedModel::LinearRegression,
            score: f64::INFINITY,
            all_scores: vec![],
        };
        assert!(result.score.is_infinite());
    }
}
