//! Confidence interval model

/// Confidence interval result
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    /// Point forecast
    pub forecast: Vec<f64>,
    /// Lower bound of confidence interval
    pub lower: Vec<f64>,
    /// Upper bound of confidence interval
    pub upper: Vec<f64>,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}
