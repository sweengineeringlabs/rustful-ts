//! Model selection result types for AutoML.

use super::SelectedModel;

/// Result of model selection.
#[derive(Debug, Clone)]
pub struct ModelSelectionResult {
    /// The best model type selected.
    pub best_model: SelectedModel,
    /// Score of the best model (lower is better).
    pub score: f64,
    /// All evaluated models with their scores.
    pub all_scores: Vec<(SelectedModel, f64)>,
}
