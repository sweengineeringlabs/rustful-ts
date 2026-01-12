//! Parallel optimizer runner - runs multiple optimization algorithms concurrently.

use optimizer_spi::{
    OptimizationResult, OptimizedParams, Result, OptimizerError, ValidationSplit,
    IndicatorEvaluator, MarketData, ObjectiveFunction,
};
use rayon::prelude::*;
use std::sync::Arc;

use super::{
    IndicatorGridSearch, GridSearchConfig,
    IndicatorGeneticOptimizer, GeneticConfig,
    IndicatorBayesianOptimizer, BayesianConfig,
};

/// Which optimizer produced the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerType {
    GridSearch,
    Genetic,
    Bayesian,
}

impl std::fmt::Display for OptimizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerType::GridSearch => write!(f, "GridSearch"),
            OptimizerType::Genetic => write!(f, "Genetic"),
            OptimizerType::Bayesian => write!(f, "Bayesian"),
        }
    }
}

/// Result from parallel optimizer run.
#[derive(Debug, Clone)]
pub struct ParallelOptimizationResult {
    /// Best parameters found across all optimizers.
    pub best_params: Vec<OptimizedParams>,
    /// Best score achieved.
    pub best_score: f64,
    /// Out-of-sample score (if validation was used).
    pub oos_score: Option<f64>,
    /// Which optimizer produced the best result.
    pub best_optimizer: OptimizerType,
    /// Results from each optimizer that was run.
    pub optimizer_results: Vec<(OptimizerType, OptimizationResult)>,
    /// Total evaluations across all optimizers.
    pub total_evaluations: usize,
}

/// Configuration for an optimizer to run.
#[derive(Debug, Clone)]
enum OptimizerConfig {
    Grid(GridSearchConfig),
    Genetic(GeneticConfig),
    Bayesian(BayesianConfig),
}

/// Parallel optimizer runner that executes multiple optimization algorithms concurrently.
///
/// # Example
///
/// ```rust,ignore
/// use optimizer_facade::optimizer::*;
///
/// let result = ParallelOptimizerRunner::new()
///     .add_grid(GridSearchConfig::default())
///     .add_genetic(GeneticConfig { generations: 50, ..Default::default() })
///     .add_bayesian(BayesianConfig { iterations: 30, ..Default::default() })
///     .run(&evaluator, &data, &objective, &[])?;
///
/// println!("Best: {:?} from {}", result.best_params, result.best_optimizer);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ParallelOptimizerRunner {
    optimizers: Vec<OptimizerConfig>,
}

impl ParallelOptimizerRunner {
    /// Create a new parallel optimizer runner.
    pub fn new() -> Self {
        Self {
            optimizers: Vec::new(),
        }
    }

    /// Add all three optimizers with default configurations.
    pub fn all_default() -> Self {
        Self::new()
            .add_grid(GridSearchConfig::default())
            .add_genetic(GeneticConfig::default())
            .add_bayesian(BayesianConfig::default())
    }

    /// Add a grid search optimizer.
    pub fn add_grid(mut self, config: GridSearchConfig) -> Self {
        self.optimizers.push(OptimizerConfig::Grid(config));
        self
    }

    /// Add a genetic algorithm optimizer.
    pub fn add_genetic(mut self, config: GeneticConfig) -> Self {
        self.optimizers.push(OptimizerConfig::Genetic(config));
        self
    }

    /// Add a Bayesian optimizer.
    pub fn add_bayesian(mut self, config: BayesianConfig) -> Self {
        self.optimizers.push(OptimizerConfig::Bayesian(config));
        self
    }

    /// Run all configured optimizers in parallel.
    pub fn run(
        &self,
        evaluator: &dyn IndicatorEvaluator,
        data: &MarketData,
        objective: &dyn ObjectiveFunction,
        validation_splits: &[ValidationSplit],
    ) -> Result<ParallelOptimizationResult> {
        if self.optimizers.is_empty() {
            return Err(OptimizerError::InvalidConfig(
                "No optimizers configured. Use add_grid(), add_genetic(), or add_bayesian()".into()
            ));
        }

        // Run all optimizers in parallel
        let results: Vec<(OptimizerType, Result<OptimizationResult>)> = self.optimizers
            .par_iter()
            .map(|config| {
                match config {
                    OptimizerConfig::Grid(cfg) => {
                        let opt = IndicatorGridSearch::with_config(cfg.clone());
                        (OptimizerType::GridSearch, opt.optimize(evaluator, data, objective, validation_splits))
                    }
                    OptimizerConfig::Genetic(cfg) => {
                        let opt = IndicatorGeneticOptimizer::with_config(cfg.clone());
                        (OptimizerType::Genetic, opt.optimize(evaluator, data, objective, validation_splits))
                    }
                    OptimizerConfig::Bayesian(cfg) => {
                        let opt = IndicatorBayesianOptimizer::with_config(cfg.clone());
                        (OptimizerType::Bayesian, opt.optimize(evaluator, data, objective, validation_splits))
                    }
                }
            })
            .collect();

        // Collect successful results
        let mut successful: Vec<(OptimizerType, OptimizationResult)> = Vec::new();
        let mut errors: Vec<(OptimizerType, OptimizerError)> = Vec::new();

        for (opt_type, result) in results {
            match result {
                Ok(r) => successful.push((opt_type, r)),
                Err(e) => errors.push((opt_type, e)),
            }
        }

        if successful.is_empty() {
            // All failed - return the first error
            let (opt_type, err) = errors.into_iter().next().unwrap();
            return Err(OptimizerError::OptimizationFailed(
                format!("{} failed: {}", opt_type, err)
            ));
        }

        // Find the best result
        let (best_optimizer, best_result) = successful
            .iter()
            .max_by(|a, b| a.1.best_score.partial_cmp(&b.1.best_score).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let total_evaluations: usize = successful.iter().map(|(_, r)| r.evaluations).sum();

        Ok(ParallelOptimizationResult {
            best_params: best_result.best_params.clone(),
            best_score: best_result.best_score,
            oos_score: best_result.oos_score,
            best_optimizer: *best_optimizer,
            optimizer_results: successful,
            total_evaluations,
        })
    }

    /// Run optimizers and select best based on out-of-sample score (if available).
    pub fn run_select_by_oos(
        &self,
        evaluator: &dyn IndicatorEvaluator,
        data: &MarketData,
        objective: &dyn ObjectiveFunction,
        validation_splits: &[ValidationSplit],
    ) -> Result<ParallelOptimizationResult> {
        if validation_splits.is_empty() {
            // No validation - fall back to regular run
            return self.run(evaluator, data, objective, validation_splits);
        }

        let mut result = self.run(evaluator, data, objective, validation_splits)?;

        // Re-select best based on OOS score
        if let Some((best_opt, best_res)) = result.optimizer_results
            .iter()
            .filter(|(_, r)| r.oos_score.is_some())
            .max_by(|a, b| {
                a.1.oos_score.unwrap().partial_cmp(&b.1.oos_score.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            result.best_params = best_res.best_params.clone();
            result.best_score = best_res.best_score;
            result.oos_score = best_res.oos_score;
            result.best_optimizer = *best_opt;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimizer_spi::{ParamRange, Timeframe};
    use crate::evaluators::RSIEvaluator;
    use crate::objective::SharpeRatio;
    use crate::datasource::FixtureDataSource;
    use optimizer_spi::DataSource;

    fn create_test_data() -> MarketData {
        let mut data = MarketData::new("TEST", Timeframe::D1);
        for i in 0..200 {
            let trend = i as f64 * 0.1;
            let noise = (i as f64 * 0.5).sin() * 5.0;
            data.close.push(100.0 + trend + noise);
            data.open.push(100.0 + trend + noise - 0.5);
            data.high.push(100.0 + trend + noise + 2.0);
            data.low.push(100.0 + trend + noise - 2.0);
            data.volume.push(1000.0);
            data.timestamps.push(i as i64);
        }
        data
    }

    #[test]
    fn test_parallel_runner_all_optimizers() {
        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
        let objective = SharpeRatio::new();
        let data = create_test_data();

        let result = ParallelOptimizerRunner::new()
            .add_grid(GridSearchConfig { parallel: false, top_n: 5 })
            .add_genetic(GeneticConfig {
                population_size: 10,
                generations: 5,
                parallel: false,
                ..Default::default()
            })
            .add_bayesian(BayesianConfig {
                iterations: 10,
                initial_samples: 5,
                ..Default::default()
            })
            .run(&evaluator, &data, &objective, &[])
            .unwrap();

        assert_eq!(result.optimizer_results.len(), 3);
        assert!(result.total_evaluations > 0);
        println!("Best from {}: {:?} -> {:.4}",
            result.best_optimizer,
            result.best_params,
            result.best_score);

        for (opt_type, res) in &result.optimizer_results {
            println!("  {}: {:.4} ({} evals)", opt_type, res.best_score, res.evaluations);
        }
    }

    #[test]
    fn test_parallel_runner_with_validation() {
        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
        let objective = SharpeRatio::new();
        let data = create_test_data();

        let splits = vec![
            ValidationSplit {
                train_start: 0,
                train_end: 140,
                test_start: 140,
                test_end: 200,
            },
        ];

        let result = ParallelOptimizerRunner::new()
            .add_grid(GridSearchConfig::default())
            .add_genetic(GeneticConfig {
                population_size: 10,
                generations: 5,
                ..Default::default()
            })
            .run_select_by_oos(&evaluator, &data, &objective, &splits)
            .unwrap();

        assert!(result.oos_score.is_some());
        println!("Best by OOS from {}: IS={:.4}, OOS={:.4}",
            result.best_optimizer,
            result.best_score,
            result.oos_score.unwrap());
    }

    #[test]
    fn test_parallel_runner_all_default() {
        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
        let objective = SharpeRatio::new();
        let data = create_test_data();

        // This will take longer due to default configs
        let result = ParallelOptimizerRunner::new()
            .add_grid(GridSearchConfig { parallel: true, top_n: 5 })
            .add_genetic(GeneticConfig {
                population_size: 15,
                generations: 10,
                ..Default::default()
            })
            .add_bayesian(BayesianConfig {
                iterations: 15,
                initial_samples: 5,
                ..Default::default()
            })
            .run(&evaluator, &data, &objective, &[])
            .unwrap();

        assert_eq!(result.optimizer_results.len(), 3);
        println!("All default - Best: {} with score {:.4}", result.best_optimizer, result.best_score);
    }

    #[test]
    fn test_parallel_runner_single_optimizer() {
        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
        let objective = SharpeRatio::new();
        let data = create_test_data();

        let result = ParallelOptimizerRunner::new()
            .add_genetic(GeneticConfig {
                population_size: 10,
                generations: 5,
                ..Default::default()
            })
            .run(&evaluator, &data, &objective, &[])
            .unwrap();

        assert_eq!(result.optimizer_results.len(), 1);
        assert_eq!(result.best_optimizer, OptimizerType::Genetic);
    }

    #[test]
    fn test_parallel_runner_empty_error() {
        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
        let objective = SharpeRatio::new();
        let data = create_test_data();

        let result = ParallelOptimizerRunner::new()
            .run(&evaluator, &data, &objective, &[]);

        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_runner_with_real_data() {
        let ds = FixtureDataSource::new();
        let data = match ds.load("SPY", Timeframe::D1) {
            Ok(d) => d,
            Err(_) => return, // Skip if no fixture data
        };

        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
        let objective = SharpeRatio::new();

        let result = ParallelOptimizerRunner::new()
            .add_grid(GridSearchConfig::default())
            .add_genetic(GeneticConfig {
                population_size: 15,
                generations: 10,
                ..Default::default()
            })
            .add_bayesian(BayesianConfig {
                iterations: 15,
                initial_samples: 5,
                ..Default::default()
            })
            .run(&evaluator, &data, &objective, &[])
            .unwrap();

        println!("SPY Real Data - Best from {}: {:?} -> {:.4}",
            result.best_optimizer,
            result.best_params,
            result.best_score);
    }
}
