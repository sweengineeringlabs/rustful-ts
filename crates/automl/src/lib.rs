//! # rustful-automl
//!
//! Automatic machine learning for time series forecasting.
//! Provides model selection, hyperparameter optimization, and ensemble methods.

mod model_selection;
mod hyperopt;
mod ensemble;

pub use model_selection::*;
pub use ensemble::*;
