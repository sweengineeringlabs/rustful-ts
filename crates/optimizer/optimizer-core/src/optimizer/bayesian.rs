//! Bayesian optimization implementation.

use optimizer_spi::{
    Optimizer, ObjectiveFunction, Validator, OptimizationResult,
    Result, OptimizerError, OptimizationMethod,
};

/// Bayesian optimizer with surrogate model.
#[derive(Debug, Clone)]
pub struct BayesianOptimizer {
    iterations: usize,
    exploration_factor: f64,
    initial_samples: usize,
}

impl BayesianOptimizer {
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            exploration_factor: 2.576,
            initial_samples: 10,
        }
    }

    pub fn with_exploration(mut self, factor: f64) -> Self {
        self.exploration_factor = factor;
        self
    }

    pub fn with_initial_samples(mut self, samples: usize) -> Self {
        self.initial_samples = samples;
        self
    }
}

impl Optimizer for BayesianOptimizer {
    fn optimize(
        &self,
        _data: &[f64],
        _objective: &dyn ObjectiveFunction,
        _validator: &dyn Validator,
    ) -> Result<OptimizationResult> {
        Err(OptimizerError::OptimizationFailed(
            "Bayesian optimization requires Gaussian Process implementation. \
             Use GridSearch or GeneticAlgorithm instead.".into()
        ))
    }

    fn method(&self) -> OptimizationMethod {
        OptimizationMethod::Bayesian { iterations: self.iterations }
    }
}
