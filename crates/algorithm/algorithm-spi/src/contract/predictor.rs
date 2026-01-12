//! Predictor traits for time series algorithms
//!
//! Defines the core trait interfaces that all prediction algorithms must implement.

use crate::error::Result;

/// Common trait for all time series predictors
///
/// This trait defines the core interface that all prediction algorithms
/// must implement. It follows a fit-predict pattern common in statistical
/// and machine learning libraries.
///
/// # Example
///
/// ```rust,ignore
/// use algorithm_spi::Predictor;
///
/// fn forecast<P: Predictor>(predictor: &mut P, data: &[f64], horizon: usize) -> algorithm_spi::Result<Vec<f64>> {
///     predictor.fit(data)?;
///     predictor.predict(horizon)
/// }
/// ```
pub trait Predictor {
    /// Fit the model to historical data
    ///
    /// # Arguments
    ///
    /// * `data` - Historical time series data
    ///
    /// # Returns
    ///
    /// `Ok(())` if fitting succeeds, `Err(TsError)` otherwise
    fn fit(&mut self, data: &[f64]) -> Result<()>;

    /// Predict future values
    ///
    /// # Arguments
    ///
    /// * `steps` - Number of future time steps to predict
    ///
    /// # Returns
    ///
    /// Vector of predicted values, or an error if prediction fails
    fn predict(&self, steps: usize) -> Result<Vec<f64>>;

    /// Check if the model has been fitted
    ///
    /// # Returns
    ///
    /// `true` if the model has been successfully fitted, `false` otherwise
    fn is_fitted(&self) -> bool;
}

/// Trait for models that support incremental updates
///
/// This trait extends [`Predictor`] for algorithms that can efficiently
/// incorporate new data without complete retraining. This is useful for
/// streaming or online learning scenarios.
///
/// # Example
///
/// ```rust,ignore
/// use algorithm_spi::{Predictor, IncrementalPredictor};
///
/// fn update_and_forecast<P: IncrementalPredictor>(
///     predictor: &mut P,
///     new_data: &[f64],
///     horizon: usize
/// ) -> algorithm_spi::Result<Vec<f64>> {
///     predictor.update(new_data)?;
///     predictor.predict(horizon)
/// }
/// ```
pub trait IncrementalPredictor: Predictor {
    /// Update the model with new data point(s)
    ///
    /// # Arguments
    ///
    /// * `data` - New observations to incorporate into the model
    ///
    /// # Returns
    ///
    /// `Ok(())` if update succeeds, `Err(TsError)` otherwise
    fn update(&mut self, data: &[f64]) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TsError;

    // ==========================================================================
    // Mock Implementations for Testing Trait Definitions
    // ==========================================================================

    /// A simple mock predictor that calculates the mean of fitted data
    struct MockMeanPredictor {
        mean: Option<f64>,
        min_data_points: usize,
    }

    impl MockMeanPredictor {
        fn new(min_data_points: usize) -> Self {
            Self {
                mean: None,
                min_data_points,
            }
        }
    }

    impl Predictor for MockMeanPredictor {
        fn fit(&mut self, data: &[f64]) -> Result<()> {
            if data.len() < self.min_data_points {
                return Err(TsError::InsufficientData {
                    required: self.min_data_points,
                    actual: data.len(),
                });
            }

            // Check for NaN values
            if data.iter().any(|x| x.is_nan()) {
                return Err(TsError::InvalidData("Data contains NaN values".to_string()));
            }

            let sum: f64 = data.iter().sum();
            self.mean = Some(sum / data.len() as f64);
            Ok(())
        }

        fn predict(&self, steps: usize) -> Result<Vec<f64>> {
            match self.mean {
                Some(mean) => Ok(vec![mean; steps]),
                None => Err(TsError::NotFitted),
            }
        }

        fn is_fitted(&self) -> bool {
            self.mean.is_some()
        }
    }

    /// A mock incremental predictor that supports updates
    struct MockIncrementalPredictor {
        values: Vec<f64>,
        min_data_points: usize,
    }

    impl MockIncrementalPredictor {
        fn new(min_data_points: usize) -> Self {
            Self {
                values: Vec::new(),
                min_data_points,
            }
        }

        fn current_mean(&self) -> Option<f64> {
            if self.values.is_empty() {
                None
            } else {
                Some(self.values.iter().sum::<f64>() / self.values.len() as f64)
            }
        }
    }

    impl Predictor for MockIncrementalPredictor {
        fn fit(&mut self, data: &[f64]) -> Result<()> {
            if data.len() < self.min_data_points {
                return Err(TsError::InsufficientData {
                    required: self.min_data_points,
                    actual: data.len(),
                });
            }

            self.values = data.to_vec();
            Ok(())
        }

        fn predict(&self, steps: usize) -> Result<Vec<f64>> {
            match self.current_mean() {
                Some(mean) => Ok(vec![mean; steps]),
                None => Err(TsError::NotFitted),
            }
        }

        fn is_fitted(&self) -> bool {
            !self.values.is_empty()
        }
    }

    impl IncrementalPredictor for MockIncrementalPredictor {
        fn update(&mut self, data: &[f64]) -> Result<()> {
            if self.values.is_empty() {
                return Err(TsError::NotFitted);
            }

            self.values.extend_from_slice(data);
            Ok(())
        }
    }

    /// A mock predictor that always fails during fitting (for error testing)
    struct MockFailingPredictor {
        error_type: FailureType,
    }

    enum FailureType {
        ConvergenceFailure,
        NumericalError,
        InvalidParameter,
    }

    impl Predictor for MockFailingPredictor {
        fn fit(&mut self, _data: &[f64]) -> Result<()> {
            match self.error_type {
                FailureType::ConvergenceFailure => Err(TsError::ConvergenceFailure { iterations: 100 }),
                FailureType::NumericalError => {
                    Err(TsError::NumericalError("Matrix is singular".to_string()))
                }
                FailureType::InvalidParameter => Err(TsError::InvalidParameter {
                    name: "smoothing_factor".to_string(),
                    reason: "must be positive".to_string(),
                }),
            }
        }

        fn predict(&self, _steps: usize) -> Result<Vec<f64>> {
            Err(TsError::NotFitted)
        }

        fn is_fitted(&self) -> bool {
            false
        }
    }

    // ==========================================================================
    // Predictor Trait Tests
    // ==========================================================================

    #[test]
    fn test_predictor_fit_success() {
        let mut predictor = MockMeanPredictor::new(3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = predictor.fit(&data);

        assert!(result.is_ok());
        assert!(predictor.is_fitted());
    }

    #[test]
    fn test_predictor_fit_insufficient_data() {
        let mut predictor = MockMeanPredictor::new(5);
        let data = vec![1.0, 2.0];

        let result = predictor.fit(&data);

        assert!(result.is_err());
        match result.unwrap_err() {
            TsError::InsufficientData { required, actual } => {
                assert_eq!(required, 5);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected InsufficientData error"),
        }
        assert!(!predictor.is_fitted());
    }

    #[test]
    fn test_predictor_fit_invalid_data() {
        let mut predictor = MockMeanPredictor::new(1);
        let data = vec![1.0, f64::NAN, 3.0];

        let result = predictor.fit(&data);

        assert!(result.is_err());
        match result.unwrap_err() {
            TsError::InvalidData(msg) => {
                assert!(msg.contains("NaN"));
            }
            _ => panic!("Expected InvalidData error"),
        }
    }

    #[test]
    fn test_predictor_predict_success() {
        let mut predictor = MockMeanPredictor::new(1);
        predictor.fit(&[2.0, 4.0, 6.0]).unwrap();

        let predictions = predictor.predict(3).unwrap();

        assert_eq!(predictions.len(), 3);
        assert!((predictions[0] - 4.0).abs() < 1e-10);
        assert!((predictions[1] - 4.0).abs() < 1e-10);
        assert!((predictions[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_predictor_predict_not_fitted() {
        let predictor = MockMeanPredictor::new(1);

        let result = predictor.predict(5);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TsError::NotFitted);
    }

    #[test]
    fn test_predictor_predict_zero_steps() {
        let mut predictor = MockMeanPredictor::new(1);
        predictor.fit(&[1.0, 2.0, 3.0]).unwrap();

        let predictions = predictor.predict(0).unwrap();

        assert!(predictions.is_empty());
    }

    #[test]
    fn test_predictor_is_fitted_before_fit() {
        let predictor = MockMeanPredictor::new(1);
        assert!(!predictor.is_fitted());
    }

    #[test]
    fn test_predictor_is_fitted_after_fit() {
        let mut predictor = MockMeanPredictor::new(1);
        predictor.fit(&[1.0, 2.0]).unwrap();
        assert!(predictor.is_fitted());
    }

    #[test]
    fn test_predictor_refit() {
        let mut predictor = MockMeanPredictor::new(1);

        // First fit
        predictor.fit(&[2.0, 4.0]).unwrap();
        let first_predictions = predictor.predict(1).unwrap();
        assert!((first_predictions[0] - 3.0).abs() < 1e-10);

        // Refit with different data
        predictor.fit(&[10.0, 20.0]).unwrap();
        let second_predictions = predictor.predict(1).unwrap();
        assert!((second_predictions[0] - 15.0).abs() < 1e-10);
    }

    // ==========================================================================
    // IncrementalPredictor Trait Tests
    // ==========================================================================

    #[test]
    fn test_incremental_predictor_update_success() {
        let mut predictor = MockIncrementalPredictor::new(2);

        // Initial fit
        predictor.fit(&[2.0, 4.0]).unwrap();
        let initial_predictions = predictor.predict(1).unwrap();
        assert!((initial_predictions[0] - 3.0).abs() < 1e-10);

        // Update with new data
        predictor.update(&[6.0]).unwrap();
        let updated_predictions = predictor.predict(1).unwrap();
        assert!((updated_predictions[0] - 4.0).abs() < 1e-10); // (2+4+6)/3 = 4
    }

    #[test]
    fn test_incremental_predictor_update_not_fitted() {
        let mut predictor = MockIncrementalPredictor::new(2);

        let result = predictor.update(&[1.0, 2.0]);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TsError::NotFitted);
    }

    #[test]
    fn test_incremental_predictor_multiple_updates() {
        let mut predictor = MockIncrementalPredictor::new(1);
        predictor.fit(&[10.0]).unwrap();

        // Multiple sequential updates
        predictor.update(&[20.0]).unwrap();
        predictor.update(&[30.0]).unwrap();

        let predictions = predictor.predict(1).unwrap();
        assert!((predictions[0] - 20.0).abs() < 1e-10); // (10+20+30)/3 = 20
    }

    #[test]
    fn test_incremental_predictor_update_empty_data() {
        let mut predictor = MockIncrementalPredictor::new(1);
        predictor.fit(&[5.0]).unwrap();

        // Update with empty data should succeed but not change anything
        predictor.update(&[]).unwrap();

        let predictions = predictor.predict(1).unwrap();
        assert!((predictions[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_incremental_predictor_inherits_predictor() {
        // Test that IncrementalPredictor can be used where Predictor is expected
        fn use_predictor<P: Predictor>(predictor: &mut P, data: &[f64]) -> Result<Vec<f64>> {
            predictor.fit(data)?;
            predictor.predict(3)
        }

        let mut predictor = MockIncrementalPredictor::new(1);
        let result = use_predictor(&mut predictor, &[1.0, 2.0, 3.0]);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    // ==========================================================================
    // Trait Object Tests (Dynamic Dispatch)
    // ==========================================================================

    #[test]
    fn test_predictor_as_trait_object() {
        let mut predictor: Box<dyn Predictor> = Box::new(MockMeanPredictor::new(1));

        predictor.fit(&[1.0, 2.0, 3.0]).unwrap();
        let predictions = predictor.predict(2).unwrap();

        assert_eq!(predictions.len(), 2);
        assert!(predictor.is_fitted());
    }

    #[test]
    fn test_incremental_predictor_as_trait_object() {
        let mut predictor: Box<dyn IncrementalPredictor> =
            Box::new(MockIncrementalPredictor::new(1));

        predictor.fit(&[1.0, 2.0]).unwrap();
        predictor.update(&[3.0]).unwrap();
        let predictions = predictor.predict(1).unwrap();

        assert!((predictions[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_predictor_types_in_collection() {
        let predictors: Vec<Box<dyn Predictor>> = vec![
            Box::new(MockMeanPredictor::new(1)),
            Box::new(MockIncrementalPredictor::new(1)),
        ];

        for mut predictor in predictors {
            let result = predictor.fit(&[1.0, 2.0, 3.0]);
            assert!(result.is_ok());
            assert!(predictor.is_fitted());
        }
    }

    // ==========================================================================
    // Error Handling Tests
    // ==========================================================================

    #[test]
    fn test_predictor_convergence_failure() {
        let mut predictor = MockFailingPredictor {
            error_type: FailureType::ConvergenceFailure,
        };

        let result = predictor.fit(&[1.0, 2.0]);

        assert!(result.is_err());
        match result.unwrap_err() {
            TsError::ConvergenceFailure { iterations } => {
                assert_eq!(iterations, 100);
            }
            _ => panic!("Expected ConvergenceFailure error"),
        }
    }

    #[test]
    fn test_predictor_numerical_error() {
        let mut predictor = MockFailingPredictor {
            error_type: FailureType::NumericalError,
        };

        let result = predictor.fit(&[1.0, 2.0]);

        assert!(result.is_err());
        match result.unwrap_err() {
            TsError::NumericalError(msg) => {
                assert!(msg.contains("singular"));
            }
            _ => panic!("Expected NumericalError error"),
        }
    }

    #[test]
    fn test_predictor_invalid_parameter() {
        let mut predictor = MockFailingPredictor {
            error_type: FailureType::InvalidParameter,
        };

        let result = predictor.fit(&[1.0, 2.0]);

        assert!(result.is_err());
        match result.unwrap_err() {
            TsError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smoothing_factor");
                assert!(reason.contains("positive"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    // ==========================================================================
    // Generic Function Tests
    // ==========================================================================

    #[test]
    fn test_generic_forecast_function() {
        fn forecast<P: Predictor>(
            predictor: &mut P,
            data: &[f64],
            horizon: usize,
        ) -> Result<Vec<f64>> {
            predictor.fit(data)?;
            predictor.predict(horizon)
        }

        let mut predictor = MockMeanPredictor::new(1);
        let result = forecast(&mut predictor, &[1.0, 2.0, 3.0], 5);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 5);
    }

    #[test]
    fn test_generic_update_and_forecast_function() {
        fn update_and_forecast<P: IncrementalPredictor>(
            predictor: &mut P,
            initial_data: &[f64],
            new_data: &[f64],
            horizon: usize,
        ) -> Result<Vec<f64>> {
            predictor.fit(initial_data)?;
            predictor.update(new_data)?;
            predictor.predict(horizon)
        }

        let mut predictor = MockIncrementalPredictor::new(1);
        let result = update_and_forecast(&mut predictor, &[1.0, 2.0], &[3.0, 4.0], 3);

        assert!(result.is_ok());
        let predictions = result.unwrap();
        assert_eq!(predictions.len(), 3);
        assert!((predictions[0] - 2.5).abs() < 1e-10); // (1+2+3+4)/4 = 2.5
    }

    // ==========================================================================
    // Edge Case Tests
    // ==========================================================================

    #[test]
    fn test_predictor_with_single_data_point() {
        let mut predictor = MockMeanPredictor::new(1);
        predictor.fit(&[42.0]).unwrap();

        let predictions = predictor.predict(3).unwrap();

        for pred in predictions {
            assert!((pred - 42.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_predictor_with_large_horizon() {
        let mut predictor = MockMeanPredictor::new(1);
        predictor.fit(&[1.0, 2.0, 3.0]).unwrap();

        let predictions = predictor.predict(10000).unwrap();

        assert_eq!(predictions.len(), 10000);
    }

    #[test]
    fn test_predictor_with_negative_values() {
        let mut predictor = MockMeanPredictor::new(1);
        predictor.fit(&[-5.0, -3.0, -1.0]).unwrap();

        let predictions = predictor.predict(1).unwrap();

        assert!((predictions[0] - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_predictor_with_infinity() {
        let mut predictor = MockMeanPredictor::new(1);
        // Note: Our mock doesn't check for infinity, so this will succeed
        // but produce inf in predictions
        predictor.fit(&[f64::INFINITY, 1.0]).unwrap();

        let predictions = predictor.predict(1).unwrap();

        assert!(predictions[0].is_infinite());
    }

    #[test]
    fn test_predictor_with_very_small_values() {
        let mut predictor = MockMeanPredictor::new(1);
        predictor.fit(&[1e-300, 2e-300, 3e-300]).unwrap();

        let predictions = predictor.predict(1).unwrap();

        assert!((predictions[0] - 2e-300).abs() < 1e-310);
    }

    #[test]
    fn test_predictor_with_very_large_values() {
        let mut predictor = MockMeanPredictor::new(1);
        predictor.fit(&[1e300, 2e300, 3e300]).unwrap();

        let predictions = predictor.predict(1).unwrap();

        assert!((predictions[0] - 2e300).abs() / 2e300 < 1e-10);
    }
}
