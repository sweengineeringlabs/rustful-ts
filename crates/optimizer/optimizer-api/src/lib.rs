//! Optimizer API
//!
//! **WARNING: This is an internal crate. Do not depend on it directly.**
//! **Use `optimizer-facade` instead for a stable public API.**
//!
//! Configuration types and builders for optimization.

mod evaluator;

use serde::{Deserialize, Serialize};
use optimizer_spi::{Objective, ValidationStrategy, OptimizationMethod, SignalCombination};

pub use evaluator::{Evaluator, IndicatorSpec};

// ============================================================================
// Optimizer Configuration
// ============================================================================

/// Optimizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub objective: Objective,
    pub method: OptimizationMethod,
    pub validation: ValidationStrategy,
    pub signal_combination: SignalCombination,
    pub top_n: usize,
    pub verbose: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            objective: Objective::default(),
            method: OptimizationMethod::default(),
            validation: ValidationStrategy::default(),
            signal_combination: SignalCombination::default(),
            top_n: 10,
            verbose: false,
        }
    }
}

impl OptimizerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_objective(mut self, objective: Objective) -> Self {
        self.objective = objective;
        self
    }

    pub fn with_method(mut self, method: OptimizationMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_validation(mut self, validation: ValidationStrategy) -> Self {
        self.validation = validation;
        self
    }

    pub fn with_signal_combination(mut self, combination: SignalCombination) -> Self {
        self.signal_combination = combination;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

// ============================================================================
// Genetic Algorithm Configuration
// ============================================================================

/// Genetic algorithm configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticConfig {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elite_count: usize,
    pub tournament_size: usize,
}

impl Default for GeneticConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elite_count: 5,
            tournament_size: 3,
        }
    }
}

// ============================================================================
// Bayesian Optimization Configuration
// ============================================================================

/// Bayesian optimization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianConfig {
    pub iterations: usize,
    pub exploration_factor: f64,
    pub initial_samples: usize,
}

impl Default for BayesianConfig {
    fn default() -> Self {
        Self {
            iterations: 50,
            exploration_factor: 2.576,  // 99% confidence
            initial_samples: 10,
        }
    }
}

/// Acquisition function for Bayesian optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcquisitionFunction {
    /// Upper Confidence Bound.
    UCB,
    /// Expected Improvement.
    ExpectedImprovement,
    /// Probability of Improvement.
    ProbabilityOfImprovement,
}

impl Default for AcquisitionFunction {
    fn default() -> Self {
        AcquisitionFunction::UCB
    }
}

// ============================================================================
// Walk-Forward Configuration
// ============================================================================

/// Walk-forward validation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardConfig {
    pub windows: usize,
    pub train_ratio: f64,
    pub anchored: bool,  // Whether training window is anchored to start
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            windows: 5,
            train_ratio: 0.8,
            anchored: false,
        }
    }
}
