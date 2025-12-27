//! # automl
//!
//! Automatic machine learning for time series forecasting.
//! Provides model selection, hyperparameter optimization, and ensemble methods.

mod ensemble;
mod hyperopt;
mod model_selection;

pub use ensemble::*;
pub use hyperopt::GridSearch;
pub use model_selection::*;
