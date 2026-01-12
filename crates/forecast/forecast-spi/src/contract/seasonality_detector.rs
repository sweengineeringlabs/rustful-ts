//! Trait for seasonality detection

/// Trait for seasonality detection
pub trait SeasonalityDetector: Send + Sync {
    /// Detect the dominant seasonality period in the data
    fn detect(&self, data: &[f64], max_period: usize) -> Option<usize>;
}
