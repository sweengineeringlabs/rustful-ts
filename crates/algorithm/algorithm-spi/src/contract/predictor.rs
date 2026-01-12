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
