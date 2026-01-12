//! AutoML Facade
//!
//! High-level API for automatic machine learning. Re-exports all public types
//! from the automl stack for convenient usage.
//!
//! # Example
//!
//! ```ignore
//! use automl_facade::prelude::*;
//!
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, /* ... more data ... */];
//! let automl = AutoML::with_defaults();
//! let result = automl.select(&data, 5)?;
//! println!("Best model: {}", result.best_model);
//! ```

// Re-export everything from core (which includes API and SPI)
pub use automl_core::*;

/// Prelude module for convenient imports
pub mod prelude {
    // Traits
    pub use automl_spi::{EnsembleCombiner, HyperparameterOptimizer, ModelSelector};

    // Core types
    pub use automl_api::{AutoMLConfig, EnsembleMethod, GridSearchConfig, OptimizationMetric};

    // Error types
    pub use automl_spi::{AutoMLError, ModelSelectionResult, ModelType, Result, SelectedModel};

    // Implementations
    pub use automl_core::{
        combine_predictions, AutoML, EnsembleAverager, EnsembleMedian, EnsembleWeighted, GridSearch,
    };
}
