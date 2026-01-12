//! Model selection trait for AutoML.

use crate::error::AutoMLError;
use crate::model::ModelSelectionResult;

/// Result type for model selector operations.
pub type Result<T> = std::result::Result<T, AutoMLError>;

/// Trait for model selection strategies.
pub trait ModelSelector {
    /// Select the best model for the given data.
    fn select(&self, data: &[f64], horizon: usize) -> Result<ModelSelectionResult>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SelectedModel;

    // ========== Mock Implementations ==========

    /// A mock model selector that always returns SES.
    struct AlwaysSESSelector {
        alpha: f64,
    }

    impl ModelSelector for AlwaysSESSelector {
        fn select(&self, data: &[f64], _horizon: usize) -> Result<ModelSelectionResult> {
            if data.len() < 2 {
                return Err(AutoMLError::InsufficientData {
                    required: 2,
                    actual: data.len(),
                });
            }
            Ok(ModelSelectionResult {
                best_model: SelectedModel::SES { alpha: self.alpha },
                score: 0.1,
                all_scores: vec![(SelectedModel::SES { alpha: self.alpha }, 0.1)],
            })
        }
    }

    /// A mock model selector that tries multiple models.
    struct MultiModelSelector;

    impl ModelSelector for MultiModelSelector {
        fn select(&self, data: &[f64], _horizon: usize) -> Result<ModelSelectionResult> {
            if data.len() < 10 {
                return Err(AutoMLError::InsufficientData {
                    required: 10,
                    actual: data.len(),
                });
            }

            let all_scores = vec![
                (SelectedModel::SES { alpha: 0.3 }, 0.5),
                (SelectedModel::Holt { alpha: 0.3, beta: 0.2 }, 0.3),
                (SelectedModel::LinearRegression, 0.4),
            ];

            Ok(ModelSelectionResult {
                best_model: SelectedModel::Holt { alpha: 0.3, beta: 0.2 },
                score: 0.3,
                all_scores,
            })
        }
    }

    /// A mock model selector that always fails.
    struct FailingSelector {
        error_type: &'static str,
    }

    impl ModelSelector for FailingSelector {
        fn select(&self, _data: &[f64], _horizon: usize) -> Result<ModelSelectionResult> {
            match self.error_type {
                "fit" => Err(AutoMLError::FitError("Mock fit error".to_string())),
                "convergence" => Err(AutoMLError::ConvergenceFailure { iterations: 100 }),
                "no_models" => Err(AutoMLError::NoValidModels("All models failed".to_string())),
                "numerical" => Err(AutoMLError::NumericalError("Division by zero".to_string())),
                _ => Err(AutoMLError::FitError("Unknown error".to_string())),
            }
        }
    }

    /// A mock model selector that uses horizon information.
    struct HorizonAwareSelector;

    impl ModelSelector for HorizonAwareSelector {
        fn select(&self, data: &[f64], horizon: usize) -> Result<ModelSelectionResult> {
            if data.len() < 5 {
                return Err(AutoMLError::InsufficientData {
                    required: 5,
                    actual: data.len(),
                });
            }

            // Use different models based on horizon
            let best_model = if horizon <= 1 {
                SelectedModel::SES { alpha: 0.5 }
            } else if horizon <= 5 {
                SelectedModel::Holt { alpha: 0.3, beta: 0.2 }
            } else {
                SelectedModel::Arima { p: 1, d: 1, q: 1 }
            };

            Ok(ModelSelectionResult {
                best_model,
                score: 0.2,
                all_scores: vec![],
            })
        }
    }

    // ========== Trait Implementation Tests ==========

    #[test]
    fn test_model_selector_trait_object() {
        let selector: Box<dyn ModelSelector> = Box::new(AlwaysSESSelector { alpha: 0.5 });
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = selector.select(&data, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_selector_multiple_implementors() {
        let selectors: Vec<Box<dyn ModelSelector>> = vec![
            Box::new(AlwaysSESSelector { alpha: 0.5 }),
            Box::new(MultiModelSelector),
            Box::new(HorizonAwareSelector),
        ];

        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();

        for selector in selectors {
            let result = selector.select(&data, 5);
            assert!(result.is_ok());
        }
    }

    // ========== Success Cases ==========

    #[test]
    fn test_always_ses_selector_success() {
        let selector = AlwaysSESSelector { alpha: 0.7 };
        let data = vec![1.0, 2.0, 3.0];
        let result = selector.select(&data, 1).unwrap();

        assert!(matches!(result.best_model, SelectedModel::SES { alpha } if (alpha - 0.7).abs() < f64::EPSILON));
        assert!((result.score - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multi_model_selector_success() {
        let selector = MultiModelSelector;
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = selector.select(&data, 5).unwrap();

        assert!(matches!(result.best_model, SelectedModel::Holt { .. }));
        assert_eq!(result.all_scores.len(), 3);
        assert!((result.score - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_horizon_aware_selector_short_horizon() {
        let selector = HorizonAwareSelector;
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let result = selector.select(&data, 1).unwrap();

        assert!(matches!(result.best_model, SelectedModel::SES { .. }));
    }

    #[test]
    fn test_horizon_aware_selector_medium_horizon() {
        let selector = HorizonAwareSelector;
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let result = selector.select(&data, 3).unwrap();

        assert!(matches!(result.best_model, SelectedModel::Holt { .. }));
    }

    #[test]
    fn test_horizon_aware_selector_long_horizon() {
        let selector = HorizonAwareSelector;
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let result = selector.select(&data, 10).unwrap();

        assert!(matches!(result.best_model, SelectedModel::Arima { .. }));
    }

    // ========== Error Cases ==========

    #[test]
    fn test_insufficient_data_error() {
        let selector = AlwaysSESSelector { alpha: 0.5 };
        let data = vec![1.0]; // Only 1 data point
        let result = selector.select(&data, 1);

        assert!(result.is_err());
        if let Err(AutoMLError::InsufficientData { required, actual }) = result {
            assert_eq!(required, 2);
            assert_eq!(actual, 1);
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    #[test]
    fn test_empty_data_error() {
        let selector = AlwaysSESSelector { alpha: 0.5 };
        let data: Vec<f64> = vec![];
        let result = selector.select(&data, 1);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(AutoMLError::InsufficientData { actual: 0, .. })
        ));
    }

    #[test]
    fn test_fit_error() {
        let selector = FailingSelector { error_type: "fit" };
        let data = vec![1.0, 2.0, 3.0];
        let result = selector.select(&data, 1);

        assert!(matches!(result, Err(AutoMLError::FitError(_))));
    }

    #[test]
    fn test_convergence_failure_error() {
        let selector = FailingSelector {
            error_type: "convergence",
        };
        let data = vec![1.0, 2.0, 3.0];
        let result = selector.select(&data, 1);

        assert!(matches!(
            result,
            Err(AutoMLError::ConvergenceFailure { iterations: 100 })
        ));
    }

    #[test]
    fn test_no_valid_models_error() {
        let selector = FailingSelector {
            error_type: "no_models",
        };
        let data = vec![1.0, 2.0, 3.0];
        let result = selector.select(&data, 1);

        assert!(matches!(result, Err(AutoMLError::NoValidModels(_))));
    }

    #[test]
    fn test_numerical_error() {
        let selector = FailingSelector {
            error_type: "numerical",
        };
        let data = vec![1.0, 2.0, 3.0];
        let result = selector.select(&data, 1);

        assert!(matches!(result, Err(AutoMLError::NumericalError(_))));
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_select_with_zero_horizon() {
        let selector = HorizonAwareSelector;
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let result = selector.select(&data, 0);

        assert!(result.is_ok());
    }

    #[test]
    fn test_select_with_large_horizon() {
        let selector = HorizonAwareSelector;
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let result = selector.select(&data, 1000);

        assert!(result.is_ok());
        assert!(matches!(result.unwrap().best_model, SelectedModel::Arima { .. }));
    }

    #[test]
    fn test_select_with_exact_minimum_data() {
        let selector = AlwaysSESSelector { alpha: 0.5 };
        let data = vec![1.0, 2.0]; // Exactly 2 data points (minimum)
        let result = selector.select(&data, 1);

        assert!(result.is_ok());
    }

    #[test]
    fn test_select_with_large_dataset() {
        let selector = MultiModelSelector;
        let data: Vec<f64> = (1..=10000).map(|x| x as f64).collect();
        let result = selector.select(&data, 5);

        assert!(result.is_ok());
    }

    #[test]
    fn test_select_with_negative_values() {
        let selector = AlwaysSESSelector { alpha: 0.5 };
        let data = vec![-5.0, -3.0, -1.0, 1.0, 3.0];
        let result = selector.select(&data, 1);

        assert!(result.is_ok());
    }

    #[test]
    fn test_select_with_nan_values() {
        let selector = AlwaysSESSelector { alpha: 0.5 };
        let data = vec![1.0, f64::NAN, 3.0];
        // The mock doesn't validate NaN, but real implementations should handle this
        let result = selector.select(&data, 1);
        assert!(result.is_ok()); // Mock passes, real impl might fail
    }

    // ========== Result Type Tests ==========

    #[test]
    fn test_result_type_is_std_result() {
        fn assert_result_type<T, E>(_: std::result::Result<T, E>) {}
        let selector = AlwaysSESSelector { alpha: 0.5 };
        let data = vec![1.0, 2.0];
        let result = selector.select(&data, 1);
        assert_result_type(result);
    }

    #[test]
    fn test_result_can_be_unwrapped() {
        let selector = AlwaysSESSelector { alpha: 0.5 };
        let data = vec![1.0, 2.0, 3.0];
        let result = selector.select(&data, 1).unwrap();
        assert!(matches!(result.best_model, SelectedModel::SES { .. }));
    }

    #[test]
    fn test_result_can_use_question_mark() {
        fn inner() -> Result<()> {
            let selector = AlwaysSESSelector { alpha: 0.5 };
            let data = vec![1.0, 2.0, 3.0];
            let _result = selector.select(&data, 1)?;
            Ok(())
        }
        assert!(inner().is_ok());
    }
}
