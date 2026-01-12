//! Indicator-specific optimization.
//!
//! This module contains the indicator optimizer that knows about
//! technical indicator types and their parameter spaces.

use optimizer_spi::{ParamRange, FloatParamRange, OptimizationResult, Objective, SignalCombination};
use optimizer_api::OptimizerConfig;
use serde::{Deserialize, Serialize};

// ============================================================================
// Indicator Configuration
// ============================================================================

/// Configuration for an indicator to optimize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndicatorConfig {
    /// Simple Moving Average
    SMA { period: ParamRange },
    /// Exponential Moving Average
    EMA { period: ParamRange },
    /// Relative Strength Index
    RSI { period: ParamRange },
    /// Moving Average Convergence Divergence
    MACD {
        fast: ParamRange,
        slow: ParamRange,
        signal: ParamRange,
    },
    /// Bollinger Bands
    Bollinger {
        period: ParamRange,
        std_dev: FloatParamRange,
    },
    /// Rate of Change
    ROC { period: ParamRange },
    /// Standard Deviation
    StdDev { period: ParamRange },
    /// Average True Range
    ATR { period: ParamRange },
    /// Stochastic Oscillator
    Stochastic {
        k_period: ParamRange,
        d_period: ParamRange,
    },
}

impl IndicatorConfig {
    /// Total parameter combinations for this indicator.
    pub fn combinations(&self) -> usize {
        match self {
            IndicatorConfig::SMA { period } |
            IndicatorConfig::EMA { period } |
            IndicatorConfig::RSI { period } |
            IndicatorConfig::ROC { period } |
            IndicatorConfig::StdDev { period } |
            IndicatorConfig::ATR { period } => period.count(),

            IndicatorConfig::MACD { fast, slow, signal } => {
                fast.count() * slow.count() * signal.count()
            }

            IndicatorConfig::Bollinger { period, std_dev } => {
                period.count() * std_dev.count()
            }

            IndicatorConfig::Stochastic { k_period, d_period } => {
                k_period.count() * d_period.count()
            }
        }
    }
}

// ============================================================================
// Indicator Optimizer
// ============================================================================

/// Builder for indicator parameter optimization.
#[derive(Clone)]
pub struct IndicatorOptimizer {
    /// Indicators to optimize
    pub indicators: Vec<IndicatorConfig>,
    /// Optimization objective
    pub objective: Objective,
    /// Optimization method
    pub method: optimizer_spi::OptimizationMethod,
    /// Validation strategy
    pub validation: optimizer_spi::ValidationStrategy,
    /// Signal combination mode
    pub signal_combination: SignalCombination,
    /// Number of top results to return
    pub top_n: usize,
    /// Verbose output
    pub verbose: bool,
}

impl IndicatorOptimizer {
    /// Create a new optimizer with default settings.
    pub fn new() -> Self {
        Self {
            indicators: Vec::new(),
            objective: Objective::default(),
            method: optimizer_spi::OptimizationMethod::default(),
            validation: optimizer_spi::ValidationStrategy::default(),
            signal_combination: SignalCombination::default(),
            top_n: 10,
            verbose: false,
        }
    }

    /// Add SMA with period range.
    pub fn add_sma_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorConfig::SMA {
            period: ParamRange::new(min, max, step),
        });
        self
    }

    /// Add EMA with period range.
    pub fn add_ema_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorConfig::EMA {
            period: ParamRange::new(min, max, step),
        });
        self
    }

    /// Add RSI with period range.
    pub fn add_rsi_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorConfig::RSI {
            period: ParamRange::new(min, max, step),
        });
        self
    }

    /// Set optimization objective.
    pub fn objective(mut self, obj: Objective) -> Self {
        self.objective = obj;
        self
    }

    /// Set optimization method.
    pub fn method(mut self, method: optimizer_spi::OptimizationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set validation strategy.
    pub fn validation(mut self, val: optimizer_spi::ValidationStrategy) -> Self {
        self.validation = val;
        self
    }

    /// Total parameter combinations.
    pub fn total_combinations(&self) -> usize {
        self.indicators.iter().map(|i| i.combinations()).product()
    }

    /// Run optimization.
    pub fn optimize(&self, prices: &[f64]) -> IndicatorOptResult {
        // Placeholder - full implementation would use grid/genetic/bayesian
        IndicatorOptResult {
            best_score: 0.0,
            oos_score: None,
            evaluations: 0,
            robustness: None,
        }
    }
}

impl Default for IndicatorOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Results
// ============================================================================

/// Indicator optimization result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorOptResult {
    pub best_score: f64,
    pub oos_score: Option<f64>,
    pub evaluations: usize,
    pub robustness: Option<f64>,
}

impl Default for IndicatorOptResult {
    fn default() -> Self {
        Self {
            best_score: f64::NEG_INFINITY,
            oos_score: None,
            evaluations: 0,
            robustness: None,
        }
    }
}
