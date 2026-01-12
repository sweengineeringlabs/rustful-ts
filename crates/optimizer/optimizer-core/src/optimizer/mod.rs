//! Optimizer implementations.

mod grid;
mod genetic;
mod bayesian;

pub use grid::*;
pub use genetic::*;
pub use bayesian::*;

use optimizer_spi::{Optimizer, OptimizationMethod, OptimizationResult, ObjectiveFunction, Validator, Result};

/// Create optimizer from method.
pub fn create_optimizer(method: &OptimizationMethod) -> Box<dyn Optimizer> {
    match method {
        OptimizationMethod::GridSearch => Box::new(GridSearchOptimizer::new()),
        OptimizationMethod::ParallelGrid => Box::new(GridSearchOptimizer::new()), // TODO: parallel
        OptimizationMethod::RandomSearch { iterations } => {
            Box::new(GridSearchOptimizer::new()) // TODO: random
        }
        OptimizationMethod::GeneticAlgorithm { population, generations, mutation_rate, crossover_rate } => {
            Box::new(GeneticOptimizer::new(*population, *generations, *mutation_rate, *crossover_rate))
        }
        OptimizationMethod::Bayesian { iterations } => {
            Box::new(BayesianOptimizer::new(*iterations))
        }
    }
}
