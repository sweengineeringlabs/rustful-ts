//! End-to-end tests for rustful-forecast crate
//!
//! Tests complete pipeline workflows using only this crate's API.

use forecast_facade::{Pipeline, NormalizeStep, StandardizeStep, DifferenceStep, PipelineStep};

fn sample_data() -> Vec<f64> {
    (0..50).map(|i| 100.0 + i as f64 * 2.0 + (i as f64 * 0.5).sin() * 5.0).collect()
}

#[test]
fn e2e_normalization_pipeline() {
    let data = sample_data();

    let mut pipeline = Pipeline::new();
    pipeline.add_step(Box::new(NormalizeStep::new()));

    let transformed = pipeline.fit_transform(&data).unwrap();

    // Check normalized range
    let min = transformed.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = transformed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    assert!((min - 0.0).abs() < 0.001);
    assert!((max - 1.0).abs() < 0.001);

    // Inverse transform should recover original
    let recovered = pipeline.inverse_transform(&transformed).unwrap();

    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig - rec).abs() < 0.001,
            "Mismatch: {} vs {}", orig, rec);
    }
}

#[test]
fn e2e_standardization_pipeline() {
    let data = sample_data();

    let mut pipeline = Pipeline::new();
    pipeline.add_step(Box::new(StandardizeStep::new()));

    let transformed = pipeline.fit_transform(&data).unwrap();

    // Check standardized properties
    let mean: f64 = transformed.iter().sum::<f64>() / transformed.len() as f64;
    let variance: f64 = transformed.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / transformed.len() as f64;
    let std = variance.sqrt();

    assert!(mean.abs() < 0.001, "Mean should be ~0, got {}", mean);
    assert!((std - 1.0).abs() < 0.001, "Std should be ~1, got {}", std);

    // Inverse transform
    let recovered = pipeline.inverse_transform(&transformed).unwrap();

    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig - rec).abs() < 0.001);
    }
}

#[test]
fn e2e_differencing_pipeline() {
    let data = sample_data();

    let step = DifferenceStep::new(1);
    let differenced = step.transform(&data).unwrap();

    assert_eq!(differenced.len(), data.len() - 1);

    // First difference should match manual calculation
    for i in 0..differenced.len() {
        let expected = data[i + 1] - data[i];
        assert!((differenced[i] - expected).abs() < 0.001);
    }
}

#[test]
fn e2e_multi_step_pipeline() {
    let data = sample_data();

    let mut pipeline = Pipeline::new();
    pipeline.add_step(Box::new(NormalizeStep::new()));
    pipeline.add_step(Box::new(StandardizeStep::new()));

    let transformed = pipeline.fit_transform(&data).unwrap();
    assert_eq!(transformed.len(), data.len());

    // Should be standardized after both steps
    let mean: f64 = transformed.iter().sum::<f64>() / transformed.len() as f64;
    assert!(mean.abs() < 0.001);

    // Inverse should recover original
    let recovered = pipeline.inverse_transform(&transformed).unwrap();

    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig - rec).abs() < 0.01,
            "Mismatch after multi-step inverse: {} vs {}", orig, rec);
    }
}

#[test]
fn e2e_pipeline_with_differencing() {
    let data = sample_data();

    let mut pipeline = Pipeline::new();
    pipeline.add_step(Box::new(NormalizeStep::new()));
    pipeline.add_step(Box::new(DifferenceStep::new(1)));

    let transformed = pipeline.fit_transform(&data).unwrap();

    // Length reduced by 1 due to differencing
    assert_eq!(transformed.len(), data.len() - 1);
}

#[test]
fn e2e_empty_pipeline() {
    let data = sample_data();

    let mut pipeline = Pipeline::default();
    let transformed = pipeline.fit_transform(&data).unwrap();

    // Empty pipeline should return data unchanged
    assert_eq!(transformed.len(), data.len());
    for (orig, trans) in data.iter().zip(transformed.iter()) {
        assert!((orig - trans).abs() < 0.001);
    }
}

#[test]
fn e2e_step_names() {
    let normalize = NormalizeStep::new();
    let standardize = StandardizeStep::new();
    let difference = DifferenceStep::new(1);

    assert_eq!(normalize.name(), "normalize");
    assert_eq!(standardize.name(), "standardize");
    assert_eq!(difference.name(), "difference");
}

#[test]
fn e2e_pipeline_reuse() {
    let data1 = sample_data();
    let data2: Vec<f64> = (0..30).map(|i| 50.0 + i as f64).collect();

    let mut pipeline = Pipeline::new();
    pipeline.add_step(Box::new(NormalizeStep::new()));

    // Fit on first dataset
    let _ = pipeline.fit_transform(&data1).unwrap();

    // Transform second dataset (uses first dataset's parameters)
    let transformed2 = pipeline.fit_transform(&data2).unwrap();
    assert_eq!(transformed2.len(), data2.len());
}

#[test]
fn e2e_higher_order_differencing() {
    let data: Vec<f64> = (0..20).map(|i| (i as f64).powi(2)).collect();

    // First order difference of x^2 = 2x + 1 (linear)
    let diff1 = DifferenceStep::new(1);
    let first_diff = diff1.transform(&data).unwrap();
    assert_eq!(first_diff.len(), 19);

    // Second order difference should be approximately constant
    let diff2 = DifferenceStep::new(1);
    let second_diff = diff2.transform(&first_diff).unwrap();
    assert_eq!(second_diff.len(), 18);

    // Second differences should be ~2
    for &d in &second_diff {
        assert!((d - 2.0).abs() < 0.001);
    }
}
