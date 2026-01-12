//! Model module containing AutoML data structures.
//!
//! This module defines data structures for automatic machine learning:
//! - [`SelectedModel`] - Represents a selected model with parameters
//! - [`ModelSelectionResult`] - Result of model selection process
//! - [`ModelType`] - Alias for backward compatibility

mod model_selection_result;
mod selected_model;

pub use model_selection_result::ModelSelectionResult;
pub use selected_model::{ModelType, SelectedModel};
