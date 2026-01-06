//! Optimizer builder.

use indicator_api::{ParamRange, FloatParamRange, IndicatorType};
use tuning_spi::{Objective, ValidationStrategy, OptimizationMethod, SignalCombination};
use crate::OptimizerConfig;

/// Builder for indicator parameter optimization.
#[derive(Debug, Clone)]
pub struct OptimizerBuilder {
    indicators: Vec<IndicatorType>,
    config: OptimizerConfig,
}

impl OptimizerBuilder {
    pub fn new() -> Self {
        Self {
            indicators: Vec::new(),
            config: OptimizerConfig::default(),
        }
    }

    // ========== Indicator Range Methods ==========

    /// Add SMA with period range.
    pub fn add_sma_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorType::SMA {
            period: ParamRange::new(min, max, step),
        });
        self
    }

    /// Add EMA with period range.
    pub fn add_ema_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorType::EMA {
            period: ParamRange::new(min, max, step),
        });
        self
    }

    /// Add RSI with period range.
    pub fn add_rsi_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorType::RSI {
            period: ParamRange::new(min, max, step),
        });
        self
    }

    /// Add MACD with parameter ranges.
    pub fn add_macd_range(
        mut self,
        fast: (usize, usize, usize),
        slow: (usize, usize, usize),
        signal: (usize, usize, usize),
    ) -> Self {
        self.indicators.push(IndicatorType::MACD {
            fast: ParamRange::new(fast.0, fast.1, fast.2),
            slow: ParamRange::new(slow.0, slow.1, slow.2),
            signal: ParamRange::new(signal.0, signal.1, signal.2),
        });
        self
    }

    /// Add Bollinger Bands with parameter ranges.
    pub fn add_bollinger_range(
        mut self,
        period: (usize, usize, usize),
        std_dev: (f64, f64, f64),
    ) -> Self {
        self.indicators.push(IndicatorType::Bollinger {
            period: ParamRange::new(period.0, period.1, period.2),
            std_dev: FloatParamRange::new(std_dev.0, std_dev.1, std_dev.2),
        });
        self
    }

    /// Add ATR with period range.
    pub fn add_atr_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorType::ATR {
            period: ParamRange::new(min, max, step),
        });
        self
    }

    /// Add Stochastic with parameter ranges.
    pub fn add_stochastic_range(
        mut self,
        k_period: (usize, usize, usize),
        d_period: (usize, usize, usize),
    ) -> Self {
        self.indicators.push(IndicatorType::Stochastic {
            k_period: ParamRange::new(k_period.0, k_period.1, k_period.2),
            d_period: ParamRange::new(d_period.0, d_period.1, d_period.2),
        });
        self
    }

    // ========== Configuration Methods ==========

    /// Set optimization objective.
    pub fn objective(mut self, objective: Objective) -> Self {
        self.config.objective = objective;
        self
    }

    /// Set optimization method.
    pub fn method(mut self, method: OptimizationMethod) -> Self {
        self.config.method = method;
        self
    }

    /// Set validation strategy.
    pub fn validation(mut self, validation: ValidationStrategy) -> Self {
        self.config.validation = validation;
        self
    }

    /// Set signal combination strategy.
    pub fn signal_combination(mut self, combination: SignalCombination) -> Self {
        self.config.signal_combination = combination;
        self
    }

    /// Set number of top results to keep.
    pub fn top_n(mut self, n: usize) -> Self {
        self.config.top_n = n;
        self
    }

    /// Enable verbose output.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    // ========== Convenience Methods ==========

    /// Use grid search optimization.
    pub fn grid_search(self) -> Self {
        self.method(OptimizationMethod::GridSearch)
    }

    /// Use random search optimization.
    pub fn random_search(self, iterations: usize) -> Self {
        self.method(OptimizationMethod::RandomSearch { iterations })
    }

    /// Use genetic algorithm optimization.
    pub fn genetic(
        self,
        population: usize,
        generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> Self {
        self.method(OptimizationMethod::GeneticAlgorithm {
            population,
            generations,
            mutation_rate,
            crossover_rate,
        })
    }

    /// Use Bayesian optimization.
    pub fn bayesian(self, iterations: usize) -> Self {
        self.method(OptimizationMethod::Bayesian { iterations })
    }

    /// Use walk-forward validation.
    pub fn walk_forward(self, windows: usize, train_ratio: f64) -> Self {
        self.validation(ValidationStrategy::WalkForward { windows, train_ratio })
    }

    /// Use K-fold cross-validation.
    pub fn k_fold(self, folds: usize) -> Self {
        self.validation(ValidationStrategy::KFold { folds })
    }

    /// Use train/test split.
    pub fn train_test(self, train_ratio: f64) -> Self {
        self.validation(ValidationStrategy::TrainTest { train_ratio })
    }

    // ========== Build ==========

    /// Get indicators.
    pub fn indicators(&self) -> &[IndicatorType] {
        &self.indicators
    }

    /// Get configuration.
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Total parameter combinations.
    pub fn total_combinations(&self) -> usize {
        self.indicators.iter()
            .map(|i| i.combinations())
            .product()
    }
}

impl Default for OptimizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
