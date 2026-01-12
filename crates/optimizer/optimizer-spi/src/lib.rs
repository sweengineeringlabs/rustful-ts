//! Optimizer Service Provider Interface
//!
//! Defines traits for optimization, validation, and objective functions.
//! This is the extension point for custom optimizers, validators, and objectives.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Optimizer errors.
#[derive(Debug, Error)]
pub enum OptimizerError {
    #[error("Insufficient data: required {required}, got {got}")]
    InsufficientData { required: usize, got: usize },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Indicator error: {0}")]
    IndicatorError(String),
}

pub type Result<T> = std::result::Result<T, OptimizerError>;

// ============================================================================
// Parameter Ranges
// ============================================================================

/// Defines a range for a single integer parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamRange {
    pub min: usize,
    pub max: usize,
    pub step: usize,
}

impl ParamRange {
    pub fn new(min: usize, max: usize, step: usize) -> Self {
        Self { min, max, step }
    }

    /// Get all values in this range.
    pub fn values(&self) -> Vec<usize> {
        (self.min..=self.max).step_by(self.step.max(1)).collect()
    }

    /// Number of discrete values in this range.
    pub fn count(&self) -> usize {
        if self.step == 0 {
            return 1;
        }
        (self.max - self.min) / self.step + 1
    }
}

/// Defines a range for a floating-point parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatParamRange {
    pub min: f64,
    pub max: f64,
    pub step: f64,
}

impl FloatParamRange {
    pub fn new(min: f64, max: f64, step: f64) -> Self {
        Self { min, max, step }
    }

    /// Get all values in this range.
    pub fn values(&self) -> Vec<f64> {
        let mut result = Vec::new();
        let mut val = self.min;
        while val <= self.max + 1e-10 {
            result.push(val);
            val += self.step;
        }
        result
    }

    /// Number of discrete values in this range.
    pub fn count(&self) -> usize {
        if self.step <= 0.0 {
            return 1;
        }
        ((self.max - self.min) / self.step + 1.0) as usize
    }
}

// ============================================================================
// Objective Functions
// ============================================================================

/// Optimization objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Objective {
    /// Directional accuracy (% correct up/down predictions).
    DirectionalAccuracy,
    /// Sharpe ratio of indicator-based signals.
    SharpeRatio,
    /// Total return from indicator signals.
    TotalReturn,
    /// Information coefficient (correlation with future returns).
    InformationCoefficient,
    /// Maximum drawdown (minimize).
    MaxDrawdown,
    /// Sortino ratio (downside risk-adjusted returns).
    SortinoRatio,
    /// Profit factor (gross profit / gross loss).
    ProfitFactor,
    /// Win rate (percentage of winning trades).
    WinRate,
}

impl Default for Objective {
    fn default() -> Self {
        Objective::SharpeRatio
    }
}

impl Objective {
    /// Whether higher values are better for this objective.
    pub fn is_maximize(&self) -> bool {
        match self {
            Objective::DirectionalAccuracy => true,
            Objective::SharpeRatio => true,
            Objective::TotalReturn => true,
            Objective::InformationCoefficient => true,
            Objective::MaxDrawdown => false, // Minimize
            Objective::SortinoRatio => true,
            Objective::ProfitFactor => true,
            Objective::WinRate => true,
        }
    }
}

/// Trait for objective function computation.
pub trait ObjectiveFunction: Send + Sync {
    /// Compute objective value from signals and returns.
    fn compute(&self, signals: &[f64], returns: &[f64]) -> f64;

    /// Objective type.
    fn objective_type(&self) -> Objective;
}

// ============================================================================
// Validation Strategies
// ============================================================================

/// Validation strategy to prevent overfitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStrategy {
    /// No validation (in-sample only).
    None,
    /// Simple train/test split.
    TrainTest { train_ratio: f64 },
    /// Walk-forward validation.
    WalkForward { windows: usize, train_ratio: f64 },
    /// K-fold cross-validation.
    KFold { folds: usize },
    /// Time series cross-validation (expanding window).
    TimeSeriesCV { n_splits: usize, test_size: usize },
}

impl Default for ValidationStrategy {
    fn default() -> Self {
        ValidationStrategy::TrainTest { train_ratio: 0.7 }
    }
}

/// Result of a validation split.
#[derive(Debug, Clone)]
pub struct ValidationSplit {
    pub train_start: usize,
    pub train_end: usize,
    pub test_start: usize,
    pub test_end: usize,
}

/// Trait for validation strategy implementation.
pub trait Validator: Send + Sync {
    /// Generate validation splits.
    fn splits(&self, data_len: usize) -> Result<Vec<ValidationSplit>>;

    /// Strategy type.
    fn strategy(&self) -> ValidationStrategy;
}

// ============================================================================
// Optimization Methods
// ============================================================================

/// Optimization method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationMethod {
    /// Exhaustive grid search.
    GridSearch,
    /// Multi-threaded grid search.
    ParallelGrid,
    /// Random sampling.
    RandomSearch { iterations: usize },
    /// Genetic algorithm.
    GeneticAlgorithm {
        population: usize,
        generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    },
    /// Bayesian optimization with surrogate model.
    Bayesian { iterations: usize },
}

impl Default for OptimizationMethod {
    fn default() -> Self {
        OptimizationMethod::GridSearch
    }
}

// ============================================================================
// Signal Combination
// ============================================================================

/// How to combine signals from multiple indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalCombination {
    /// Use only the first indicator's signal.
    FirstOnly,
    /// All indicators must agree (AND logic).
    Unanimous,
    /// Majority vote (>50% must agree).
    Majority,
    /// Average of all indicator signals.
    Average,
    /// Weighted combination with custom weights.
    Weighted(Vec<f64>),
    /// Primary indicator with secondary confirmation.
    Confirmation,
}

impl Default for SignalCombination {
    fn default() -> Self {
        SignalCombination::FirstOnly
    }
}

// ============================================================================
// Optimization Results
// ============================================================================

/// Optimized parameter for a single indicator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedParams {
    /// Indicator type name.
    pub indicator_type: String,
    /// Parameter name-value pairs.
    pub params: Vec<(String, f64)>,
}

impl OptimizedParams {
    pub fn new(indicator_type: &str) -> Self {
        Self {
            indicator_type: indicator_type.to_string(),
            params: Vec::new(),
        }
    }

    pub fn with_param(mut self, name: &str, value: f64) -> Self {
        self.params.push((name.to_string(), value));
        self
    }

    pub fn get_param(&self, name: &str) -> Option<f64> {
        self.params.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
    }
}

/// Complete optimization result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Best parameters found for each indicator.
    pub best_params: Vec<OptimizedParams>,
    /// Best objective score achieved (in-sample).
    pub best_score: f64,
    /// Out-of-sample score (if validation used).
    pub oos_score: Option<f64>,
    /// Number of evaluations performed.
    pub evaluations: usize,
    /// Robustness ratio (OOS/IS) if walk-forward used.
    pub robustness: Option<f64>,
    /// Top N results for analysis.
    pub top_results: Vec<(Vec<OptimizedParams>, f64)>,
}

impl Default for OptimizationResult {
    fn default() -> Self {
        Self {
            best_params: Vec::new(),
            best_score: f64::NEG_INFINITY,
            oos_score: None,
            evaluations: 0,
            robustness: None,
            top_results: Vec::new(),
        }
    }
}

// ============================================================================
// Core Traits
// ============================================================================

/// Parameter space for optimization.
pub trait ParameterSpace: Send + Sync {
    /// Total number of parameter combinations.
    fn combinations(&self) -> usize;

    /// Get parameter values at given index.
    fn get_params(&self, index: usize) -> Vec<f64>;

    /// Random parameter sample.
    fn random_sample(&self) -> Vec<f64>;
}

/// Optimizer trait.
pub trait Optimizer: Send + Sync {
    /// Run optimization.
    fn optimize(
        &self,
        data: &[f64],
        objective: &dyn ObjectiveFunction,
        validator: &dyn Validator,
    ) -> Result<OptimizationResult>;

    /// Optimization method type.
    fn method(&self) -> OptimizationMethod;
}
