//! # Optimizer Facade
//!
//! **This is the only supported entry point for the optimizer module.**
//!
//! Do not depend on `optimizer-spi`, `optimizer-api`, or `optimizer-core` directly.
//! All public types are re-exported here with a stable API.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use optimizer_facade::prelude::*;
//!
//! let result = Evaluator::new()
//!     .symbol("SPY")
//!     .timeframe(Timeframe::D1)
//!     .indicator(IndicatorSpec::rsi(10, 20, 5))
//!     .objective(Objective::SharpeRatio)
//!     .train_test(0.7)
//!     .run()?;
//!
//! println!("Best RSI period: {:?}", result.indicator_results[0].best_params);
//! println!("Sharpe ratio: {:.4}", result.indicator_results[0].best_score);
//! ```
//!
//! ## Module Organization
//!
//! - [`prelude`] - Common imports for typical usage
//! - [`evaluator`] - High-level Evaluator API
//! - [`indicators`] - Indicator evaluators (RSI, MACD, etc.)
//! - [`objectives`] - Objective functions (Sharpe, Sortino, etc.)
//! - [`validation`] - Validation strategies (train/test, walk-forward, etc.)
//! - [`data`] - Data sources and market data types
//! - [`types`] - Core types (Timeframe, Signal, params, etc.)

// ============================================================================
// Prelude - Common imports for typical usage
// ============================================================================

/// Prelude module with common imports.
///
/// ```rust,ignore
/// use optimizer_facade::prelude::*;
/// ```
pub mod prelude {
    // High-level API
    pub use crate::evaluator::{Evaluator, IndicatorSpec, EvaluatorExt, EvaluatorResult};

    // Core types
    pub use crate::types::{
        Objective, Timeframe, Signal,
        OptimizationResult, OptimizedParams,
        ParamRange, FloatParamRange,
    };

    // Data
    pub use crate::data::FixtureDataSource;

    // Error handling
    pub use crate::OptimizerError;
    pub use crate::Result;
}

// ============================================================================
// Evaluator - High-level API
// ============================================================================

/// High-level Evaluator API for indicator optimization.
pub mod evaluator {
    pub use optimizer_api::{Evaluator, IndicatorSpec};
    pub use optimizer_core::runner::{EvaluatorRunner, EvaluatorResult, EvaluatorExt};
}

// ============================================================================
// Indicators - Indicator evaluators
// ============================================================================

/// Indicator evaluators for optimization.
pub mod indicators {
    pub use optimizer_core::evaluators::{
        RSIEvaluator,
        SMAEvaluator,
        EMAEvaluator,
        MACDEvaluator,
        BollingerEvaluator,
        StochasticEvaluator,
        ATREvaluator,
        create_evaluator,
        EvaluatorType,
    };

    // Trait for custom evaluators
    pub use optimizer_spi::IndicatorEvaluator;
}

// ============================================================================
// Objectives - Objective functions
// ============================================================================

/// Objective functions for optimization.
pub mod objectives {
    pub use optimizer_core::objective::{
        SharpeRatio,
        SortinoRatio,
        DirectionalAccuracy,
        TotalReturn,
        MaxDrawdown,
        ProfitFactor,
        create_objective,
    };

    // Trait for custom objectives
    pub use optimizer_spi::ObjectiveFunction;
}

// ============================================================================
// Validation - Validation strategies
// ============================================================================

/// Validation strategies to prevent overfitting.
pub mod validation {
    pub use optimizer_core::validator::{
        TrainTestValidator,
        WalkForwardValidator,
        KFoldValidator,
        TimeSeriesCVValidator,
        NoValidator,
        create_validator,
    };

    // Trait for custom validators
    pub use optimizer_spi::{Validator, ValidationSplit, ValidationStrategy};
}

// ============================================================================
// Optimizer - Optimization algorithms
// ============================================================================

/// Optimization algorithms.
pub mod optimizer {
    // Grid search
    pub use optimizer_core::optimizer::{
        GridSearchOptimizer,
        IndicatorGridSearch,
        GridSearchConfig,
    };

    // Genetic algorithm
    pub use optimizer_core::optimizer::{
        GeneticOptimizer,
        IndicatorGeneticOptimizer,
        GeneticConfig,
    };

    // Bayesian optimization
    pub use optimizer_core::optimizer::{
        BayesianOptimizer,
        IndicatorBayesianOptimizer,
        BayesianConfig,
    };

    // Parallel optimizer runner
    pub use optimizer_core::optimizer::{
        ParallelOptimizerRunner,
        ParallelOptimizationResult,
        OptimizerType,
    };

    pub use optimizer_core::optimizer::create_optimizer;

    // Trait for custom optimizers
    pub use optimizer_spi::{Optimizer, OptimizationMethod};
}

// ============================================================================
// Data - Data sources and market data
// ============================================================================

/// Data sources and market data types.
pub mod data {
    pub use optimizer_core::datasource::FixtureDataSource;
    pub use optimizer_spi::{DataSource, MarketData, Timeframe};
}

// ============================================================================
// Types - Core types
// ============================================================================

/// Core types used throughout the optimizer.
pub mod types {
    // Enums
    pub use optimizer_spi::{
        Objective,
        Timeframe,
        Signal,
        SignalCombination,
    };

    // Parameter ranges
    pub use optimizer_spi::{ParamRange, FloatParamRange};

    // Results
    pub use optimizer_spi::{
        OptimizationResult,
        OptimizedParams,
        EvaluationResult,
        IndicatorParams,
    };

    // Traits
    pub use optimizer_spi::ParameterSpace;
}

// ============================================================================
// Config - Configuration types
// ============================================================================

/// Configuration types for optimization.
pub mod config {
    pub use optimizer_api::{
        OptimizerConfig,
        GeneticConfig,
        BayesianConfig,
        WalkForwardConfig,
        AcquisitionFunction,
    };
}

// ============================================================================
// Strategy - Strategy building (advanced)
// ============================================================================

/// Strategy building for backtesting (advanced usage).
pub mod strategy {
    pub use optimizer_core::strategy::{
        EntryCondition,
        ExitCondition,
        PositionSizing,
        StrategyRule,
        StrategyMetrics,
        StrategyResult,
        StrategyTrade,
        StrategyBuilder,
    };
}

// ============================================================================
// Top-level exports - Error types and Result
// ============================================================================

/// Optimizer error type.
pub use optimizer_spi::OptimizerError;

/// Result type for optimizer operations.
pub use optimizer_spi::Result;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_prelude_imports() {
        // Verify all prelude imports are accessible
        let _tf = Timeframe::D1;
        let _obj = Objective::SharpeRatio;
        let _range = ParamRange::new(10, 20, 5);
    }

    #[test]
    fn test_evaluator_api() {
        let evaluator = Evaluator::new()
            .symbol("SPY")
            .timeframe(Timeframe::D1)
            .indicator(IndicatorSpec::rsi(10, 20, 5))
            .objective(Objective::SharpeRatio);

        assert!(evaluator.validate().is_ok());
    }

    #[test]
    fn test_full_workflow_via_facade() {
        let result = Evaluator::new()
            .symbol("SPY")
            .timeframe(Timeframe::D1)
            .indicator(IndicatorSpec::rsi(10, 20, 5))
            .objective(Objective::SharpeRatio)
            .run()
            .unwrap();

        assert_eq!(result.symbol, "SPY");
        assert!(!result.indicator_results.is_empty());
        println!("Facade test: Best RSI = {:?}, Sharpe = {:.4}",
            result.indicator_results[0].best_params,
            result.indicator_results[0].best_score);
    }

    #[test]
    fn test_indicators_module() {
        use super::indicators::*;

        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
        assert_eq!(evaluator.name(), "RSI");
    }

    #[test]
    fn test_objectives_module() {
        use super::objectives::*;

        let sharpe = SharpeRatio::new();
        let signals = vec![1.0, -1.0, 1.0];
        let returns = vec![0.01, -0.02, 0.015];
        let score = sharpe.compute(&signals, &returns);
        assert!(score.is_finite());
    }

    #[test]
    fn test_validation_module() {
        use super::validation::*;

        let validator = TrainTestValidator::new(0.7);
        let splits = validator.splits(100).unwrap();
        assert_eq!(splits.len(), 1);
    }

    #[test]
    fn test_data_module() {
        use super::data::*;

        let ds = FixtureDataSource::new();
        let symbols = ds.symbols();
        assert!(symbols.contains(&"SPY".to_string()));
    }

    #[test]
    fn test_genetic_optimizer() {
        use super::optimizer::*;
        use super::indicators::*;
        use super::objectives::*;
        use super::data::*;
        use super::types::ParamRange;

        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
        let objective = SharpeRatio::new();
        let data = FixtureDataSource::new().load("SPY", Timeframe::D1).unwrap();

        let config = GeneticConfig {
            population_size: 10,
            generations: 5,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elite_count: 2,
            tournament_size: 3,
            parallel: false,
        };

        let optimizer = IndicatorGeneticOptimizer::with_config(config);
        let result = optimizer.optimize(&evaluator, &data, &objective, &[]).unwrap();

        assert!(result.evaluations > 0);
        println!("Genetic via facade: {:?} -> {:.4}", result.best_params, result.best_score);
    }

    #[test]
    fn test_bayesian_optimizer() {
        use super::optimizer::*;
        use super::indicators::*;
        use super::objectives::*;
        use super::data::*;
        use super::types::ParamRange;

        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
        let objective = SharpeRatio::new();
        let data = FixtureDataSource::new().load("SPY", Timeframe::D1).unwrap();

        let config = BayesianConfig {
            iterations: 10,
            initial_samples: 5,
            exploration_factor: 2.0,
            length_scale: 0.5,
            noise_variance: 0.01,
        };

        let optimizer = IndicatorBayesianOptimizer::with_config(config);
        let result = optimizer.optimize(&evaluator, &data, &objective, &[]).unwrap();

        assert!(result.evaluations > 0);
        println!("Bayesian via facade: {:?} -> {:.4}", result.best_params, result.best_score);
    }

    #[test]
    fn test_parallel_optimizer_runner() {
        use super::optimizer::*;
        use super::indicators::*;
        use super::objectives::*;
        use super::data::*;
        use super::types::ParamRange;

        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
        let objective = SharpeRatio::new();
        let data = FixtureDataSource::new().load("SPY", Timeframe::D1).unwrap();

        let result = ParallelOptimizerRunner::new()
            .add_grid(GridSearchConfig { parallel: true, top_n: 5 })
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
        println!("Parallel runner - Best from {}: {:?} -> {:.4}",
            result.best_optimizer,
            result.best_params,
            result.best_score);

        for (opt_type, res) in &result.optimizer_results {
            println!("  {}: {:.4} ({} evals)", opt_type, res.best_score, res.evaluations);
        }
    }
}
