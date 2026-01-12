//! Pipeline step trait for composable preprocessing

use crate::error::Result;

/// Pipeline step trait for composable preprocessing
pub trait PipelineStep: Send + Sync {
    /// Fit the step to data (learn parameters)
    fn fit(&mut self, data: &[f64]);

    /// Transform data forward
    fn transform(&self, data: &[f64]) -> Result<Vec<f64>>;

    /// Inverse transform (undo the transformation)
    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>>;

    /// Name of this step
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ForecastError;

    /// Mock implementation: Identity transform (pass-through)
    struct IdentityStep {
        name: String,
        fitted: bool,
    }

    impl IdentityStep {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                fitted: false,
            }
        }
    }

    impl PipelineStep for IdentityStep {
        fn fit(&mut self, _data: &[f64]) {
            self.fitted = true;
        }

        fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
            if !self.fitted {
                return Err(Box::new(ForecastError::NotFitted));
            }
            Ok(data.to_vec())
        }

        fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
            if !self.fitted {
                return Err(Box::new(ForecastError::NotFitted));
            }
            Ok(data.to_vec())
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    /// Mock implementation: Scaling transform
    struct ScalingStep {
        scale: f64,
        offset: f64,
        fitted: bool,
    }

    impl ScalingStep {
        fn new() -> Self {
            Self {
                scale: 1.0,
                offset: 0.0,
                fitted: false,
            }
        }
    }

    impl PipelineStep for ScalingStep {
        fn fit(&mut self, data: &[f64]) {
            if !data.is_empty() {
                let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = max - min;
                if range > 0.0 {
                    self.scale = 1.0 / range;
                    self.offset = min;
                }
            }
            self.fitted = true;
        }

        fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
            if !self.fitted {
                return Err(Box::new(ForecastError::NotFitted));
            }
            Ok(data.iter().map(|&x| (x - self.offset) * self.scale).collect())
        }

        fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
            if !self.fitted {
                return Err(Box::new(ForecastError::NotFitted));
            }
            Ok(data.iter().map(|&x| x / self.scale + self.offset).collect())
        }

        fn name(&self) -> &str {
            "scaling"
        }
    }

    /// Mock implementation: Differencing transform
    struct DifferencingStep {
        first_value: Option<f64>,
        fitted: bool,
    }

    impl DifferencingStep {
        fn new() -> Self {
            Self {
                first_value: None,
                fitted: false,
            }
        }
    }

    impl PipelineStep for DifferencingStep {
        fn fit(&mut self, data: &[f64]) {
            self.first_value = data.first().copied();
            self.fitted = true;
        }

        fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
            if !self.fitted {
                return Err(Box::new(ForecastError::NotFitted));
            }
            if data.len() < 2 {
                return Err(Box::new(ForecastError::InsufficientData {
                    required: 2,
                    actual: data.len(),
                }));
            }
            Ok(data.windows(2).map(|w| w[1] - w[0]).collect())
        }

        fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
            if !self.fitted {
                return Err(Box::new(ForecastError::NotFitted));
            }
            let first = self.first_value.ok_or_else(|| {
                Box::new(ForecastError::NumericalError(
                    "No first value stored".to_string(),
                )) as Box<dyn std::error::Error + Send + Sync>
            })?;
            let mut result = vec![first];
            for &diff in data {
                result.push(result.last().unwrap() + diff);
            }
            Ok(result)
        }

        fn name(&self) -> &str {
            "differencing"
        }
    }

    #[test]
    fn test_identity_step_creation() {
        let step = IdentityStep::new("test_identity");
        assert_eq!(step.name(), "test_identity");
        assert!(!step.fitted);
    }

    #[test]
    fn test_identity_step_fit() {
        let mut step = IdentityStep::new("identity");
        let data = vec![1.0, 2.0, 3.0];
        step.fit(&data);
        assert!(step.fitted);
    }

    #[test]
    fn test_identity_step_transform_without_fit() {
        let step = IdentityStep::new("identity");
        let data = vec![1.0, 2.0, 3.0];
        let result = step.transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_identity_step_transform_after_fit() {
        let mut step = IdentityStep::new("identity");
        let data = vec![1.0, 2.0, 3.0];
        step.fit(&data);
        let result = step.transform(&data).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_identity_step_inverse_transform() {
        let mut step = IdentityStep::new("identity");
        let data = vec![1.0, 2.0, 3.0];
        step.fit(&data);
        let transformed = step.transform(&data).unwrap();
        let inverse = step.inverse_transform(&transformed).unwrap();
        assert_eq!(inverse, data);
    }

    #[test]
    fn test_identity_step_empty_data() {
        let mut step = IdentityStep::new("identity");
        let data: Vec<f64> = vec![];
        step.fit(&data);
        let result = step.transform(&data).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_scaling_step_fit_and_transform() {
        let mut step = ScalingStep::new();
        let data = vec![0.0, 50.0, 100.0];
        step.fit(&data);

        let transformed = step.transform(&data).unwrap();
        assert!((transformed[0] - 0.0).abs() < 1e-10);
        assert!((transformed[1] - 0.5).abs() < 1e-10);
        assert!((transformed[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling_step_inverse_transform() {
        let mut step = ScalingStep::new();
        let data = vec![0.0, 50.0, 100.0];
        step.fit(&data);

        let transformed = step.transform(&data).unwrap();
        let inverse = step.inverse_transform(&transformed).unwrap();

        for i in 0..data.len() {
            assert!((inverse[i] - data[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_scaling_step_name() {
        let step = ScalingStep::new();
        assert_eq!(step.name(), "scaling");
    }

    #[test]
    fn test_differencing_step_transform() {
        let mut step = DifferencingStep::new();
        let data = vec![1.0, 3.0, 6.0, 10.0];
        step.fit(&data);

        let transformed = step.transform(&data).unwrap();
        assert_eq!(transformed, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_differencing_step_inverse_transform() {
        let mut step = DifferencingStep::new();
        let data = vec![1.0, 3.0, 6.0, 10.0];
        step.fit(&data);

        let transformed = step.transform(&data).unwrap();
        let inverse = step.inverse_transform(&transformed).unwrap();

        assert_eq!(inverse, data);
    }

    #[test]
    fn test_differencing_step_insufficient_data() {
        let mut step = DifferencingStep::new();
        let data = vec![1.0];
        step.fit(&data);

        let result = step.transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_step_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<IdentityStep>();
        assert_send::<ScalingStep>();
        assert_send::<DifferencingStep>();
    }

    #[test]
    fn test_pipeline_step_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<IdentityStep>();
        assert_sync::<ScalingStep>();
        assert_sync::<DifferencingStep>();
    }

    #[test]
    fn test_pipeline_step_as_trait_object() {
        let mut step: Box<dyn PipelineStep> = Box::new(IdentityStep::new("boxed"));
        let data = vec![1.0, 2.0, 3.0];
        step.fit(&data);
        let result = step.transform(&data).unwrap();
        assert_eq!(result, data);
        assert_eq!(step.name(), "boxed");
    }

    #[test]
    fn test_pipeline_step_multiple_fits() {
        let mut step = ScalingStep::new();

        // First fit with one range
        let data1 = vec![0.0, 100.0];
        step.fit(&data1);
        let t1 = step.transform(&[50.0]).unwrap();
        assert!((t1[0] - 0.5).abs() < 1e-10);

        // Second fit with different range
        let data2 = vec![0.0, 200.0];
        step.fit(&data2);
        let t2 = step.transform(&[50.0]).unwrap();
        assert!((t2[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_step_special_values() {
        let mut step = IdentityStep::new("special");
        let data = vec![f64::INFINITY, f64::NEG_INFINITY, 0.0];
        step.fit(&data);
        let result = step.transform(&data).unwrap();
        assert!(result[0].is_infinite());
        assert!(result[1].is_infinite());
        assert_eq!(result[2], 0.0);
    }

    #[test]
    fn test_pipeline_step_roundtrip() {
        let mut step = ScalingStep::new();
        let original_data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        step.fit(&original_data);

        let transformed = step.transform(&original_data).unwrap();
        let restored = step.inverse_transform(&transformed).unwrap();

        for i in 0..original_data.len() {
            assert!(
                (original_data[i] - restored[i]).abs() < 1e-10,
                "Mismatch at index {}: {} vs {}",
                i,
                original_data[i],
                restored[i]
            );
        }
    }

    #[test]
    fn test_pipeline_step_with_negative_values() {
        let mut step = ScalingStep::new();
        let data = vec![-100.0, -50.0, 0.0, 50.0, 100.0];
        step.fit(&data);

        let transformed = step.transform(&data).unwrap();
        assert!((transformed[0] - 0.0).abs() < 1e-10);
        assert!((transformed[4] - 1.0).abs() < 1e-10);

        let inverse = step.inverse_transform(&transformed).unwrap();
        for i in 0..data.len() {
            assert!((inverse[i] - data[i]).abs() < 1e-10);
        }
    }
}
