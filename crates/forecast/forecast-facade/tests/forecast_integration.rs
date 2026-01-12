//! Integration tests for rustful-forecast

use forecast_facade::{Pipeline, NormalizeStep, DifferenceStep, StandardizeStep, PipelineStep};

fn sample_data() -> Vec<f64> {
    vec![100.0, 102.0, 105.0, 103.0, 108.0, 110.0, 107.0, 112.0, 115.0, 113.0]
}

#[test]
fn test_normalize_step() {
    let data = sample_data();
    let mut step = NormalizeStep::new();
    step.fit(&data);

    let normalized = step.transform(&data).unwrap();

    // All values should be in [0, 1]
    for val in &normalized {
        assert!(*val >= 0.0 && *val <= 1.0);
    }

    // Min should be 0, max should be 1
    let min = normalized.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = normalized.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!((min - 0.0).abs() < 0.001);
    assert!((max - 1.0).abs() < 0.001);
}

#[test]
fn test_normalize_inverse() {
    let data = sample_data();
    let mut step = NormalizeStep::new();
    step.fit(&data);

    let normalized = step.transform(&data).unwrap();
    let recovered = step.inverse_transform(&normalized).unwrap();

    // Should recover original data
    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig - rec).abs() < 0.001);
    }
}

#[test]
fn test_standardize_step() {
    let data = sample_data();
    let mut step = StandardizeStep::new();
    step.fit(&data);

    let standardized = step.transform(&data).unwrap();

    // Mean should be approximately 0
    let mean: f64 = standardized.iter().sum::<f64>() / standardized.len() as f64;
    assert!(mean.abs() < 0.001);

    // Std should be approximately 1
    let variance: f64 = standardized.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / standardized.len() as f64;
    let std = variance.sqrt();
    assert!((std - 1.0).abs() < 0.001);
}

#[test]
fn test_standardize_inverse() {
    let data = sample_data();
    let mut step = StandardizeStep::new();
    step.fit(&data);

    let standardized = step.transform(&data).unwrap();
    let recovered = step.inverse_transform(&standardized).unwrap();

    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig - rec).abs() < 0.001);
    }
}

#[test]
fn test_difference_step() {
    let data = sample_data();
    let step = DifferenceStep::new(1);

    let differenced = step.transform(&data).unwrap();

    // First-order differences should have len - 1 elements
    assert_eq!(differenced.len(), data.len() - 1);

    // Check first difference manually
    assert!((differenced[0] - (data[1] - data[0])).abs() < 0.001);
}

#[test]
fn test_pipeline_single_step() {
    let data = sample_data();
    let mut pipeline = Pipeline::new();
    pipeline.add_step(Box::new(NormalizeStep::new()));

    let transformed = pipeline.fit_transform(&data).unwrap();

    // Should be normalized
    for val in &transformed {
        assert!(*val >= 0.0 && *val <= 1.0);
    }
}

#[test]
fn test_pipeline_multiple_steps() {
    let data = sample_data();
    let mut pipeline = Pipeline::new();
    pipeline.add_step(Box::new(NormalizeStep::new()));
    pipeline.add_step(Box::new(DifferenceStep::new(1)));

    let transformed = pipeline.fit_transform(&data).unwrap();

    // Should have len - 1 elements (due to differencing)
    assert_eq!(transformed.len(), data.len() - 1);
}

#[test]
fn test_pipeline_inverse_transform() {
    let data = sample_data();
    let mut pipeline = Pipeline::new();
    pipeline.add_step(Box::new(StandardizeStep::new()));

    let transformed = pipeline.fit_transform(&data).unwrap();
    let recovered = pipeline.inverse_transform(&transformed).unwrap();

    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig - rec).abs() < 0.001);
    }
}

#[test]
fn test_step_names() {
    let normalize = NormalizeStep::new();
    let standardize = StandardizeStep::new();
    let difference = DifferenceStep::new(1);

    assert_eq!(normalize.name(), "normalize");
    assert_eq!(standardize.name(), "standardize");
    assert_eq!(difference.name(), "difference");
}

#[test]
fn test_pipeline_default() {
    let pipeline = Pipeline::default();
    let data = sample_data();

    // Empty pipeline should return data unchanged
    let mut p = pipeline;
    let result = p.fit_transform(&data).unwrap();
    assert_eq!(result.len(), data.len());
}
