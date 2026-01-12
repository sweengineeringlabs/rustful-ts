//! Decomposition result model

/// Result of time series decomposition
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Trend component
    pub trend: Vec<f64>,
    /// Seasonal component
    pub seasonal: Vec<f64>,
    /// Residual component
    pub residual: Vec<f64>,
}
