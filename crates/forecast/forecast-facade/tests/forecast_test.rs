//! Unit tests for rustful-forecast crate
//!
//! Extracted from source modules for better organization.

use forecast_facade::ForecastWithConfidence;

// ============================================================================
// Confidence Interval Tests
// ============================================================================

#[test]
fn test_from_standard_errors() {
    let forecast = vec![10.0, 11.0, 12.0];
    let std_errors = vec![1.0, 1.5, 2.0];

    let result = ForecastWithConfidence::from_standard_errors(forecast, &std_errors, 0.95);

    assert_eq!(result.forecast.len(), 3);
    assert_eq!(result.lower.len(), 3);
    assert_eq!(result.upper.len(), 3);

    // Lower should be less than forecast
    for (l, f) in result.lower.iter().zip(result.forecast.iter()) {
        assert!(l < f);
    }

    // Upper should be greater than forecast
    for (u, f) in result.upper.iter().zip(result.forecast.iter()) {
        assert!(u > f);
    }
}

#[test]
fn test_from_residuals() {
    let forecast = vec![10.0, 11.0, 12.0];
    let residuals = vec![0.1, -0.2, 0.15, -0.1, 0.05, -0.15, 0.2, -0.05];

    let result = ForecastWithConfidence::from_residuals(forecast, &residuals, 0.95);

    assert_eq!(result.forecast.len(), 3);
    // Intervals should widen with forecast horizon
    let width1 = result.upper[0] - result.lower[0];
    let width2 = result.upper[1] - result.lower[1];
    let width3 = result.upper[2] - result.lower[2];

    assert!(width2 > width1);
    assert!(width3 > width2);
}
