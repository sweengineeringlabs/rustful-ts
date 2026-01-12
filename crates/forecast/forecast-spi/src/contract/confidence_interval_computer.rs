//! Trait for confidence interval computation

use crate::model::ConfidenceInterval;

/// Trait for confidence interval computation
pub trait ConfidenceIntervalComputer: Send + Sync {
    /// Compute confidence intervals for forecasts
    fn compute(
        &self,
        forecast: &[f64],
        residuals: &[f64],
        confidence_level: f64,
    ) -> ConfidenceInterval;
}
