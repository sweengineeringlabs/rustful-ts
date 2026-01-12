//! Optimizer Core
//!
//! **WARNING: This is an internal crate. Do not depend on it directly.**
//! **Use `optimizer-facade` instead for a stable public API.**
//!
//! Implementation of optimization algorithms for technical indicators.

pub mod objective;
pub mod optimizer;
pub mod validator;
pub mod indicator;
pub mod strategy;
pub mod evaluators;
pub mod datasource;
pub mod runner;

// Re-export SPI types
pub use optimizer_spi::{
    OptimizerError, Result, Objective, ObjectiveFunction,
    ValidationStrategy, ValidationSplit, Validator,
    OptimizationMethod, SignalCombination,
    OptimizationResult, OptimizedParams,
    ParamRange, FloatParamRange, ParameterSpace, Optimizer,
};

// Re-export API types
pub use optimizer_api::{
    OptimizerConfig, GeneticConfig, BayesianConfig, WalkForwardConfig,
    AcquisitionFunction,
};

// Re-export objective implementations
pub use objective::{
    SharpeRatio, SortinoRatio, DirectionalAccuracy, TotalReturn,
    MaxDrawdown, ProfitFactor, create_objective,
};

// Re-export validator implementations
pub use validator::{
    TrainTestValidator, WalkForwardValidator, KFoldValidator,
    TimeSeriesCVValidator, NoValidator, create_validator,
};

// Re-export optimizer implementations
pub use optimizer::{
    GridSearchOptimizer, GeneticOptimizer, BayesianOptimizer,
    IndicatorGridSearch, GridSearchConfig,
    create_optimizer,
};

// Re-export indicator optimization
pub use indicator::{
    IndicatorConfig, IndicatorOptimizer, IndicatorOptResult,
};

// Re-export strategy builder
pub use strategy::{
    EntryCondition, ExitCondition, PositionSizing,
    StrategyRule, StrategyMetrics, StrategyResult, StrategyTrade,
    StrategyBuilder,
};

// Re-export evaluators
pub use evaluators::{
    RSIEvaluator, SMAEvaluator, EMAEvaluator, MACDEvaluator,
    BollingerEvaluator, StochasticEvaluator, ATREvaluator,
    create_evaluator, EvaluatorType,
};

// Re-export SPI indicator types
pub use optimizer_spi::{
    Timeframe, MarketData, DataSource, IndicatorEvaluator,
    IndicatorParams, Signal, EvaluationResult,
};

// Re-export data sources
pub use datasource::FixtureDataSource;

// Re-export runner
pub use runner::{EvaluatorRunner, EvaluatorResult, EvaluatorExt};

// Re-export Evaluator API types
pub use optimizer_api::{Evaluator, IndicatorSpec};
