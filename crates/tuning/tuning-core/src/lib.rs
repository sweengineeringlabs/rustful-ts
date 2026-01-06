//! Hyperparameter Tuning Core
//!
//! Optimization algorithms and validation strategies.

pub mod objective;
pub mod validation;
pub mod optimizer;
pub mod genetic;
pub mod bayesian;

pub use objective::*;
pub use validation::*;
pub use optimizer::*;

// Re-export SPI types
pub use tuning_spi::{
    TuningError, Result, Objective, ValidationStrategy, OptimizationMethod,
    SignalCombination, OptimizedParams, OptimizationResult,
    ObjectiveFunction, Validator, Optimizer, ParameterSpace, ValidationSplit,
};

// Re-export API types
pub use tuning_api::{
    OptimizerConfig, OptimizerBuilder, GeneticConfig, BayesianConfig, WalkForwardConfig,
};
