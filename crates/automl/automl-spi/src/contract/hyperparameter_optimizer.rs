//! Hyperparameter optimization trait for AutoML.

use crate::error::AutoMLError;

/// Result type for hyperparameter optimizer operations.
pub type Result<T> = std::result::Result<T, AutoMLError>;

/// Trait for hyperparameter optimizers.
pub trait HyperparameterOptimizer {
    /// The type of parameters being optimized.
    type Params;

    /// Optimize parameters for the given data.
    fn optimize(&self, data: &[f64], horizon: usize) -> Result<(Self::Params, f64)>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Parameter Types for Mock Implementations ==========

    /// Simple exponential smoothing parameters.
    #[derive(Debug, Clone, PartialEq)]
    struct SESParams {
        alpha: f64,
    }

    /// Holt's linear trend parameters.
    #[derive(Debug, Clone, PartialEq)]
    struct HoltParams {
        alpha: f64,
        beta: f64,
    }

    /// ARIMA parameters.
    #[derive(Debug, Clone, PartialEq)]
    struct ArimaParams {
        p: usize,
        d: usize,
        q: usize,
    }

    /// Complex nested parameters.
    #[derive(Debug, Clone)]
    struct ComplexParams {
        learning_rate: f64,
        regularization: f64,
        layers: Vec<usize>,
    }

    // ========== Mock Implementations ==========

    /// A mock optimizer for SES parameters.
    struct SESOptimizer {
        initial_alpha: f64,
    }

    impl HyperparameterOptimizer for SESOptimizer {
        type Params = SESParams;

        fn optimize(&self, data: &[f64], _horizon: usize) -> Result<(Self::Params, f64)> {
            if data.len() < 3 {
                return Err(AutoMLError::InsufficientData {
                    required: 3,
                    actual: data.len(),
                });
            }

            // Simple mock: use initial alpha adjusted by data length
            let optimized_alpha = (self.initial_alpha + data.len() as f64 * 0.01).min(1.0);
            let score = 0.1 + (1.0 - optimized_alpha) * 0.1;

            Ok((SESParams { alpha: optimized_alpha }, score))
        }
    }

    /// A mock optimizer for Holt parameters.
    struct HoltOptimizer;

    impl HyperparameterOptimizer for HoltOptimizer {
        type Params = HoltParams;

        fn optimize(&self, data: &[f64], horizon: usize) -> Result<(Self::Params, f64)> {
            if data.len() < 5 {
                return Err(AutoMLError::InsufficientData {
                    required: 5,
                    actual: data.len(),
                });
            }

            // Adjust parameters based on horizon
            let alpha = if horizon > 5 { 0.3 } else { 0.5 };
            let beta = if horizon > 10 { 0.1 } else { 0.2 };
            let score = 0.05 * horizon as f64;

            Ok((HoltParams { alpha, beta }, score))
        }
    }

    /// A mock optimizer for ARIMA parameters.
    struct ArimaOptimizer {
        max_order: usize,
    }

    impl HyperparameterOptimizer for ArimaOptimizer {
        type Params = ArimaParams;

        fn optimize(&self, data: &[f64], _horizon: usize) -> Result<(Self::Params, f64)> {
            if data.len() < 20 {
                return Err(AutoMLError::InsufficientData {
                    required: 20,
                    actual: data.len(),
                });
            }

            // Mock grid search result
            let p = self.max_order.min(2);
            let d = 1;
            let q = self.max_order.min(2);
            let score = 0.15;

            Ok((ArimaParams { p, d, q }, score))
        }
    }

    /// A mock optimizer that always fails.
    struct FailingOptimizer {
        error_type: &'static str,
    }

    impl HyperparameterOptimizer for FailingOptimizer {
        type Params = SESParams;

        fn optimize(&self, _data: &[f64], _horizon: usize) -> Result<(Self::Params, f64)> {
            match self.error_type {
                "convergence" => Err(AutoMLError::ConvergenceFailure { iterations: 500 }),
                "numerical" => Err(AutoMLError::NumericalError("Gradient exploded".to_string())),
                "invalid" => Err(AutoMLError::InvalidParameter {
                    name: "alpha".to_string(),
                    reason: "outside valid range".to_string(),
                }),
                _ => Err(AutoMLError::FitError("Optimization failed".to_string())),
            }
        }
    }

    /// A mock optimizer with complex parameters.
    struct ComplexOptimizer;

    impl HyperparameterOptimizer for ComplexOptimizer {
        type Params = ComplexParams;

        fn optimize(&self, data: &[f64], _horizon: usize) -> Result<(Self::Params, f64)> {
            if data.is_empty() {
                return Err(AutoMLError::InsufficientData {
                    required: 1,
                    actual: 0,
                });
            }

            Ok((
                ComplexParams {
                    learning_rate: 0.001,
                    regularization: 0.01,
                    layers: vec![64, 32, 16],
                },
                0.05,
            ))
        }
    }

    // ========== Trait Tests ==========

    #[test]
    fn test_ses_optimizer_success() {
        let optimizer = SESOptimizer { initial_alpha: 0.3 };
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let result = optimizer.optimize(&data, 1);

        assert!(result.is_ok());
        let (params, score) = result.unwrap();
        assert!(params.alpha > 0.3);
        assert!(params.alpha <= 1.0);
        assert!(score >= 0.0);
    }

    #[test]
    fn test_holt_optimizer_success() {
        let optimizer = HoltOptimizer;
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = optimizer.optimize(&data, 5);

        assert!(result.is_ok());
        let (params, _score) = result.unwrap();
        assert!((params.alpha - 0.5).abs() < f64::EPSILON);
        assert!((params.beta - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_arima_optimizer_success() {
        let optimizer = ArimaOptimizer { max_order: 3 };
        let data: Vec<f64> = (1..=50).map(|x| x as f64).collect();
        let result = optimizer.optimize(&data, 10);

        assert!(result.is_ok());
        let (params, _score) = result.unwrap();
        assert_eq!(params.p, 2);
        assert_eq!(params.d, 1);
        assert_eq!(params.q, 2);
    }

    #[test]
    fn test_complex_optimizer_success() {
        let optimizer = ComplexOptimizer;
        let data = vec![1.0, 2.0, 3.0];
        let result = optimizer.optimize(&data, 1);

        assert!(result.is_ok());
        let (params, score) = result.unwrap();
        assert!((params.learning_rate - 0.001).abs() < f64::EPSILON);
        assert!((params.regularization - 0.01).abs() < f64::EPSILON);
        assert_eq!(params.layers, vec![64, 32, 16]);
        assert!((score - 0.05).abs() < f64::EPSILON);
    }

    // ========== Error Cases ==========

    #[test]
    fn test_ses_optimizer_insufficient_data() {
        let optimizer = SESOptimizer { initial_alpha: 0.5 };
        let data = vec![1.0, 2.0]; // Only 2 points, need 3
        let result = optimizer.optimize(&data, 1);

        assert!(result.is_err());
        if let Err(AutoMLError::InsufficientData { required, actual }) = result {
            assert_eq!(required, 3);
            assert_eq!(actual, 2);
        } else {
            panic!("Expected InsufficientData error");
        }
    }

    #[test]
    fn test_holt_optimizer_insufficient_data() {
        let optimizer = HoltOptimizer;
        let data = vec![1.0, 2.0, 3.0, 4.0]; // Only 4 points, need 5
        let result = optimizer.optimize(&data, 1);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(AutoMLError::InsufficientData {
                required: 5,
                actual: 4
            })
        ));
    }

    #[test]
    fn test_arima_optimizer_insufficient_data() {
        let optimizer = ArimaOptimizer { max_order: 2 };
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect(); // Only 10 points, need 20
        let result = optimizer.optimize(&data, 1);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(AutoMLError::InsufficientData {
                required: 20,
                actual: 10
            })
        ));
    }

    #[test]
    fn test_failing_optimizer_convergence() {
        let optimizer = FailingOptimizer {
            error_type: "convergence",
        };
        let data = vec![1.0, 2.0, 3.0];
        let result = optimizer.optimize(&data, 1);

        assert!(matches!(
            result,
            Err(AutoMLError::ConvergenceFailure { iterations: 500 })
        ));
    }

    #[test]
    fn test_failing_optimizer_numerical() {
        let optimizer = FailingOptimizer {
            error_type: "numerical",
        };
        let data = vec![1.0, 2.0, 3.0];
        let result = optimizer.optimize(&data, 1);

        assert!(matches!(result, Err(AutoMLError::NumericalError(_))));
    }

    #[test]
    fn test_failing_optimizer_invalid_parameter() {
        let optimizer = FailingOptimizer {
            error_type: "invalid",
        };
        let data = vec![1.0, 2.0, 3.0];
        let result = optimizer.optimize(&data, 1);

        assert!(matches!(result, Err(AutoMLError::InvalidParameter { .. })));
    }

    // ========== Associated Type Tests ==========

    #[test]
    fn test_associated_type_ses() {
        fn check_params_type<O: HyperparameterOptimizer<Params = SESParams>>(_: &O) {}
        let optimizer = SESOptimizer { initial_alpha: 0.5 };
        check_params_type(&optimizer);
    }

    #[test]
    fn test_associated_type_holt() {
        fn check_params_type<O: HyperparameterOptimizer<Params = HoltParams>>(_: &O) {}
        let optimizer = HoltOptimizer;
        check_params_type(&optimizer);
    }

    #[test]
    fn test_associated_type_arima() {
        fn check_params_type<O: HyperparameterOptimizer<Params = ArimaParams>>(_: &O) {}
        let optimizer = ArimaOptimizer { max_order: 2 };
        check_params_type(&optimizer);
    }

    // ========== Horizon Sensitivity Tests ==========

    #[test]
    fn test_holt_optimizer_short_horizon() {
        let optimizer = HoltOptimizer;
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = optimizer.optimize(&data, 3).unwrap();

        assert!((result.0.alpha - 0.5).abs() < f64::EPSILON);
        assert!((result.0.beta - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_holt_optimizer_long_horizon() {
        let optimizer = HoltOptimizer;
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = optimizer.optimize(&data, 15).unwrap();

        assert!((result.0.alpha - 0.3).abs() < f64::EPSILON);
        assert!((result.0.beta - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_holt_optimizer_score_increases_with_horizon() {
        let optimizer = HoltOptimizer;
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();

        let result_short = optimizer.optimize(&data, 2).unwrap();
        let result_long = optimizer.optimize(&data, 10).unwrap();

        assert!(result_long.1 > result_short.1);
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_empty_data() {
        let optimizer = ComplexOptimizer;
        let data: Vec<f64> = vec![];
        let result = optimizer.optimize(&data, 1);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(AutoMLError::InsufficientData { actual: 0, .. })
        ));
    }

    #[test]
    fn test_zero_horizon() {
        let optimizer = HoltOptimizer;
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = optimizer.optimize(&data, 0);

        assert!(result.is_ok());
    }

    #[test]
    fn test_large_horizon() {
        let optimizer = HoltOptimizer;
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let result = optimizer.optimize(&data, 1000);

        assert!(result.is_ok());
    }

    #[test]
    fn test_exact_minimum_data() {
        let optimizer = SESOptimizer { initial_alpha: 0.5 };
        let data = vec![1.0, 2.0, 3.0]; // Exactly 3 points (minimum)
        let result = optimizer.optimize(&data, 1);

        assert!(result.is_ok());
    }

    #[test]
    fn test_large_dataset() {
        let optimizer = ArimaOptimizer { max_order: 2 };
        let data: Vec<f64> = (1..=10000).map(|x| x as f64).collect();
        let result = optimizer.optimize(&data, 10);

        assert!(result.is_ok());
    }

    // ========== Result Structure Tests ==========

    #[test]
    fn test_result_contains_params_and_score() {
        let optimizer = SESOptimizer { initial_alpha: 0.5 };
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let (params, score) = optimizer.optimize(&data, 1).unwrap();

        // Verify tuple structure
        let _alpha = params.alpha;
        let _score_value = score;
    }

    #[test]
    fn test_score_is_finite() {
        let optimizer = HoltOptimizer;
        let data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
        let (_, score) = optimizer.optimize(&data, 5).unwrap();

        assert!(score.is_finite());
        assert!(!score.is_nan());
    }

    #[test]
    fn test_score_is_non_negative() {
        let optimizer = SESOptimizer { initial_alpha: 0.5 };
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let (_, score) = optimizer.optimize(&data, 1).unwrap();

        assert!(score >= 0.0);
    }

    // ========== Multiple Optimizers Tests ==========

    #[test]
    fn test_different_optimizers_same_data() {
        let data: Vec<f64> = (1..=50).map(|x| x as f64).collect();

        let ses_result = SESOptimizer { initial_alpha: 0.3 }.optimize(&data, 5);
        let holt_result = HoltOptimizer.optimize(&data, 5);
        let arima_result = ArimaOptimizer { max_order: 2 }.optimize(&data, 5);

        assert!(ses_result.is_ok());
        assert!(holt_result.is_ok());
        assert!(arima_result.is_ok());
    }

    #[test]
    fn test_same_optimizer_different_data() {
        let optimizer = SESOptimizer { initial_alpha: 0.3 };

        let data_small: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let data_large: Vec<f64> = (1..=100).map(|x| x as f64).collect();

        let result_small = optimizer.optimize(&data_small, 5).unwrap();
        let result_large = optimizer.optimize(&data_large, 5).unwrap();

        // Alpha should be higher for larger dataset due to our mock logic
        assert!(result_large.0.alpha > result_small.0.alpha);
    }

    // ========== Generic Function Tests ==========

    #[test]
    fn test_generic_function_with_optimizer() {
        fn run_optimization<O: HyperparameterOptimizer>(
            optimizer: &O,
            data: &[f64],
            horizon: usize,
        ) -> Result<f64> {
            let (_, score) = optimizer.optimize(data, horizon)?;
            Ok(score)
        }

        let ses = SESOptimizer { initial_alpha: 0.5 };
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let score = run_optimization(&ses, &data, 5);

        assert!(score.is_ok());
    }
}
