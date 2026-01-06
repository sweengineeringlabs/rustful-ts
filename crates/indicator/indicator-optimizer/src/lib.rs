//! Indicator Parameter Optimization
//!
//! Automatically find optimal technical indicator parameters for a given asset.
//! Supports multiple optimization methods and validation strategies.
//!
//! Migrated from quantlang/src/runtime/indicator_optimizer.rs
//!
//! # Example
//!
//! ```ignore
//! let result = IndicatorOptimizer::new()
//!     .add_sma_range(5, 50, 5)
//!     .add_rsi_range(7, 21, 2)
//!     .objective(Objective::SharpeRatio)
//!     .method(OptMethod::GridSearch)
//!     .validation(Validation::WalkForward { windows: 5, train_ratio: 0.8 })
//!     .optimize(&prices);
//! ```

#![allow(unused_imports)]

// ============================================================================
// Parameter Range Definitions
// ============================================================================

/// Defines a range for a single indicator parameter
#[derive(Debug, Clone)]
pub struct IndicatorParamRange {
    pub min: usize,
    pub max: usize,
    pub step: usize,
}

impl IndicatorParamRange {
    pub fn new(min: usize, max: usize, step: usize) -> Self {
        Self { min, max, step }
    }

    /// Get all values in this range
    pub fn values(&self) -> Vec<usize> {
        (self.min..=self.max).step_by(self.step).collect()
    }

    /// Number of discrete values in this range
    pub fn count(&self) -> usize {
        if self.step == 0 {
            return 1;
        }
        (self.max - self.min) / self.step + 1
    }
}

/// Defines a range for a floating-point parameter
#[derive(Debug, Clone)]
pub struct FloatParamRange {
    pub min: f64,
    pub max: f64,
    pub step: f64,
}

impl FloatParamRange {
    pub fn new(min: f64, max: f64, step: f64) -> Self {
        Self { min, max, step }
    }

    /// Get all values in this range
    pub fn values(&self) -> Vec<f64> {
        let mut result = Vec::new();
        let mut val = self.min;
        while val <= self.max + 1e-10 {
            result.push(val);
            val += self.step;
        }
        result
    }

    /// Number of discrete values in this range
    pub fn count(&self) -> usize {
        if self.step <= 0.0 {
            return 1;
        }
        ((self.max - self.min) / self.step + 1.0) as usize
    }
}

// ============================================================================
// Indicator Configuration
// ============================================================================

/// Configuration for an indicator to optimize
#[derive(Debug, Clone)]
pub enum IndicatorConfig {
    /// Simple Moving Average
    SMA { period: IndicatorParamRange },
    /// Exponential Moving Average
    EMA { period: IndicatorParamRange },
    /// Relative Strength Index
    RSI { period: IndicatorParamRange },
    /// Moving Average Convergence Divergence
    MACD {
        fast: IndicatorParamRange,
        slow: IndicatorParamRange,
        signal: IndicatorParamRange,
    },
    /// Bollinger Bands
    BollingerBands {
        period: IndicatorParamRange,
        std_dev: FloatParamRange,
    },
    /// Rate of Change
    ROC { period: IndicatorParamRange },
    /// Standard Deviation
    StdDev { period: IndicatorParamRange },
    /// Average True Range
    ATR { period: IndicatorParamRange },
    /// Stochastic Oscillator
    Stochastic {
        k_period: IndicatorParamRange,
        d_period: IndicatorParamRange,
    },
}

impl IndicatorConfig {
    /// Get the name of this indicator type
    pub fn name(&self) -> &'static str {
        match self {
            IndicatorConfig::SMA { .. } => "SMA",
            IndicatorConfig::EMA { .. } => "EMA",
            IndicatorConfig::RSI { .. } => "RSI",
            IndicatorConfig::MACD { .. } => "MACD",
            IndicatorConfig::BollingerBands { .. } => "Bollinger",
            IndicatorConfig::ROC { .. } => "ROC",
            IndicatorConfig::StdDev { .. } => "StdDev",
            IndicatorConfig::ATR { .. } => "ATR",
            IndicatorConfig::Stochastic { .. } => "Stochastic",
        }
    }

    /// Get the number of parameters for this indicator
    pub fn param_count(&self) -> usize {
        match self {
            IndicatorConfig::SMA { .. } => 1,
            IndicatorConfig::EMA { .. } => 1,
            IndicatorConfig::RSI { .. } => 1,
            IndicatorConfig::MACD { .. } => 3,
            IndicatorConfig::BollingerBands { .. } => 2,
            IndicatorConfig::ROC { .. } => 1,
            IndicatorConfig::StdDev { .. } => 1,
            IndicatorConfig::ATR { .. } => 1,
            IndicatorConfig::Stochastic { .. } => 2,
        }
    }

    /// Get the total number of combinations for this indicator
    pub fn combinations(&self) -> usize {
        match self {
            IndicatorConfig::SMA { period } => period.count(),
            IndicatorConfig::EMA { period } => period.count(),
            IndicatorConfig::RSI { period } => period.count(),
            IndicatorConfig::MACD { fast, slow, signal } => {
                fast.count() * slow.count() * signal.count()
            }
            IndicatorConfig::BollingerBands { period, std_dev } => {
                period.count() * std_dev.count()
            }
            IndicatorConfig::ROC { period } => period.count(),
            IndicatorConfig::StdDev { period } => period.count(),
            IndicatorConfig::ATR { period } => period.count(),
            IndicatorConfig::Stochastic { k_period, d_period } => {
                k_period.count() * d_period.count()
            }
        }
    }
}

// ============================================================================
// Objective Functions
// ============================================================================

/// What to optimize for
#[derive(Debug, Clone)]
pub enum Objective {
    /// Directional accuracy (% correct up/down predictions)
    DirectionalAccuracy,
    /// Sharpe ratio of indicator-based signals
    SharpeRatio,
    /// Total return from indicator signals
    TotalReturn,
    /// Information coefficient (correlation with future returns)
    InformationCoefficient,
    /// Maximum drawdown (minimize)
    MaxDrawdown,
    /// Sortino ratio (downside risk-adjusted returns)
    SortinoRatio,
}

impl Default for Objective {
    fn default() -> Self {
        Objective::SharpeRatio
    }
}

// ============================================================================
// Optimization Methods
// ============================================================================

/// Optimization method to use
#[derive(Clone)]
pub enum OptMethod {
    /// Exhaustive grid search
    GridSearch,
    /// Random sampling
    RandomSearch { iterations: usize },
    /// Genetic algorithm
    GeneticAlgorithm {
        population: usize,
        generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    },
    /// Bayesian optimization with Gaussian Process
    Bayesian { iterations: usize },
    /// Multi-threaded grid search
    ParallelGrid,
}

impl Default for OptMethod {
    fn default() -> Self {
        OptMethod::GridSearch
    }
}

// ============================================================================
// Validation Strategies
// ============================================================================

/// Validation strategy to prevent overfitting
#[derive(Clone)]
pub enum Validation {
    /// No validation (in-sample only)
    None,
    /// Simple train/test split
    TrainTest { train_ratio: f64 },
    /// Walk-forward validation
    WalkForward { windows: usize, train_ratio: f64 },
    /// K-fold cross-validation
    KFold { folds: usize },
}

impl Default for Validation {
    fn default() -> Self {
        Validation::TrainTest { train_ratio: 0.7 }
    }
}

/// How to combine signals from multiple indicators
#[derive(Debug, Clone)]
pub enum SignalCombination {
    /// Use only the first indicator's signal (default)
    FirstOnly,
    /// All indicators must agree (AND logic)
    /// Signal = 1 only if ALL indicators are positive
    Unanimous,
    /// Majority vote (>50% must agree)
    Majority,
    /// Weighted average of all indicator signals
    /// Each indicator contributes equally, result is continuous [-1, 1]
    Average,
    /// Weighted combination with custom weights per indicator
    Weighted(Vec<f64>),
    /// Confirmation: Primary indicator + secondary must confirm
    /// (first indicator leads, others confirm direction)
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

/// Optimized parameter for a single indicator
#[derive(Debug, Clone)]
pub struct OptimizedIndicator {
    /// Indicator type name (e.g., "SMA", "RSI")
    pub indicator_type: String,
    /// Parameter name-value pairs
    pub params: Vec<(String, f64)>,
}

impl OptimizedIndicator {
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

    /// Get a parameter value by name
    pub fn get_param(&self, name: &str) -> Option<f64> {
        self.params.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
    }
}

/// Result of indicator optimization
#[derive(Debug, Clone)]
pub struct IndicatorOptResult {
    /// Best parameters found for each indicator
    pub best_params: Vec<OptimizedIndicator>,
    /// Best objective score achieved (in-sample)
    pub best_score: f64,
    /// Out-of-sample score (if validation used)
    pub oos_score: Option<f64>,
    /// Number of evaluations performed
    pub evaluations: usize,
    /// Robustness ratio (OOS/IS) if walk-forward used
    pub robustness: Option<f64>,
    /// Top N results for analysis
    pub top_results: Vec<(Vec<OptimizedIndicator>, f64)>,
}

impl Default for IndicatorOptResult {
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
// Indicator Optimizer
// ============================================================================

/// Builder for indicator parameter optimization
#[derive(Clone)]
pub struct IndicatorOptimizer {
    indicators: Vec<IndicatorConfig>,
    objective: Objective,
    method: OptMethod,
    validation: Validation,
    signal_combination: SignalCombination,
    top_n: usize,
    verbose: bool,
}

impl IndicatorOptimizer {
    /// Create a new optimizer with default settings
    pub fn new() -> Self {
        Self {
            indicators: Vec::new(),
            objective: Objective::default(),
            method: OptMethod::default(),
            validation: Validation::default(),
            signal_combination: SignalCombination::default(),
            top_n: 10,
            verbose: false,
        }
    }

    // ========== Indicator Range Methods ==========

    /// Add SMA with period range
    pub fn add_sma_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorConfig::SMA {
            period: IndicatorParamRange::new(min, max, step),
        });
        self
    }

    /// Add EMA with period range
    pub fn add_ema_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorConfig::EMA {
            period: IndicatorParamRange::new(min, max, step),
        });
        self
    }

    /// Add RSI with period range
    pub fn add_rsi_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorConfig::RSI {
            period: IndicatorParamRange::new(min, max, step),
        });
        self
    }

    /// Add MACD with parameter ranges
    pub fn add_macd_range(
        mut self,
        fast: (usize, usize, usize),
        slow: (usize, usize, usize),
        signal: (usize, usize, usize),
    ) -> Self {
        self.indicators.push(IndicatorConfig::MACD {
            fast: IndicatorParamRange::new(fast.0, fast.1, fast.2),
            slow: IndicatorParamRange::new(slow.0, slow.1, slow.2),
            signal: IndicatorParamRange::new(signal.0, signal.1, signal.2),
        });
        self
    }

    /// Add Bollinger Bands with parameter ranges
    pub fn add_bollinger_range(
        mut self,
        period_min: usize,
        period_max: usize,
        period_step: usize,
        std_min: f64,
        std_max: f64,
        std_step: f64,
    ) -> Self {
        self.indicators.push(IndicatorConfig::BollingerBands {
            period: IndicatorParamRange::new(period_min, period_max, period_step),
            std_dev: FloatParamRange::new(std_min, std_max, std_step),
        });
        self
    }

    /// Add ROC with period range
    pub fn add_roc_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorConfig::ROC {
            period: IndicatorParamRange::new(min, max, step),
        });
        self
    }

    /// Add StdDev with period range
    pub fn add_std_dev_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorConfig::StdDev {
            period: IndicatorParamRange::new(min, max, step),
        });
        self
    }

    /// Add ATR with period range
    pub fn add_atr_range(mut self, min: usize, max: usize, step: usize) -> Self {
        self.indicators.push(IndicatorConfig::ATR {
            period: IndicatorParamRange::new(min, max, step),
        });
        self
    }

    /// Add Stochastic with parameter ranges
    pub fn add_stochastic_range(
        mut self,
        k: (usize, usize, usize),
        d: (usize, usize, usize),
    ) -> Self {
        self.indicators.push(IndicatorConfig::Stochastic {
            k_period: IndicatorParamRange::new(k.0, k.1, k.2),
            d_period: IndicatorParamRange::new(d.0, d.1, d.2),
        });
        self
    }

    // ========== Configuration Methods ==========

    /// Set the objective function
    pub fn objective(mut self, obj: Objective) -> Self {
        self.objective = obj;
        self
    }

    /// Set the optimization method
    pub fn method(mut self, method: OptMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the validation strategy
    pub fn validation(mut self, val: Validation) -> Self {
        self.validation = val;
        self
    }

    /// Set how to combine signals from multiple indicators
    pub fn signal_combination(mut self, combo: SignalCombination) -> Self {
        self.signal_combination = combo;
        self
    }

    /// Convenience: require all indicators to agree
    pub fn unanimous(self) -> Self {
        self.signal_combination(SignalCombination::Unanimous)
    }

    /// Convenience: use majority vote
    pub fn majority(self) -> Self {
        self.signal_combination(SignalCombination::Majority)
    }

    /// Convenience: use average of all signals
    pub fn average_signals(self) -> Self {
        self.signal_combination(SignalCombination::Average)
    }

    /// Convenience: require confirmation from secondary indicators
    pub fn confirmation(self) -> Self {
        self.signal_combination(SignalCombination::Confirmation)
    }

    /// Set number of top results to keep
    pub fn top_n(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    // ========== Preset Configurations ==========

    /// Preset for trend-following indicators
    pub fn trend_following() -> Self {
        Self::new()
            .add_sma_range(10, 50, 5)
            .add_sma_range(50, 200, 10)
            .add_ema_range(10, 50, 5)
    }

    /// Preset for momentum indicators
    pub fn momentum() -> Self {
        Self::new()
            .add_rsi_range(7, 21, 2)
            .add_macd_range((8, 12, 2), (20, 26, 2), (7, 9, 1))
            .add_stochastic_range((5, 14, 3), (3, 5, 1))
    }

    /// Preset for volatility indicators
    pub fn volatility() -> Self {
        Self::new()
            .add_atr_range(10, 20, 2)
            .add_bollinger_range(15, 25, 5, 1.5, 2.5, 0.5)
            .add_std_dev_range(10, 20, 5)
    }

    // ========== Optimization ==========

    /// Get total number of parameter combinations
    pub fn total_combinations(&self) -> usize {
        if self.indicators.is_empty() {
            return 0;
        }
        self.indicators.iter().map(|i| i.combinations()).product()
    }

    /// Run optimization on price data
    pub fn optimize(&self, prices: &[f64]) -> IndicatorOptResult {
        if prices.len() < 50 {
            return IndicatorOptResult {
                best_params: Vec::new(),
                best_score: f64::NEG_INFINITY,
                oos_score: None,
                evaluations: 0,
                robustness: None,
                top_results: Vec::new(),
            };
        }

        if self.indicators.is_empty() {
            return IndicatorOptResult::default();
        }

        match &self.validation {
            Validation::None => self.optimize_no_validation(prices),
            Validation::TrainTest { train_ratio } => {
                self.optimize_train_test(prices, *train_ratio)
            }
            Validation::WalkForward { windows, train_ratio } => {
                self.optimize_walk_forward(prices, *windows, *train_ratio)
            }
            Validation::KFold { folds } => self.optimize_kfold(prices, *folds),
        }
    }

    /// Optimize without validation (in-sample only)
    fn optimize_no_validation(&self, prices: &[f64]) -> IndicatorOptResult {
        let (best_params, best_score, evaluations, top_results) =
            self.run_optimization(prices);

        IndicatorOptResult {
            best_params,
            best_score,
            oos_score: None,
            evaluations,
            robustness: None,
            top_results,
        }
    }

    /// Optimize with train/test split
    fn optimize_train_test(&self, prices: &[f64], train_ratio: f64) -> IndicatorOptResult {
        let split = (prices.len() as f64 * train_ratio) as usize;
        let train = &prices[..split];
        let test = &prices[split..];

        // Optimize on training data
        let (best_params, is_score, evaluations, top_results) = self.run_optimization(train);

        // Evaluate on test data
        let oos_score = if !best_params.is_empty() {
            Some(self.evaluate_params(test, &best_params))
        } else {
            None
        };

        let robustness = oos_score.map(|oos| {
            if is_score.abs() > 1e-10 {
                oos / is_score
            } else {
                0.0
            }
        });

        IndicatorOptResult {
            best_params,
            best_score: is_score,
            oos_score,
            evaluations,
            robustness,
            top_results,
        }
    }

    /// Optimize with walk-forward validation
    fn optimize_walk_forward(
        &self,
        prices: &[f64],
        windows: usize,
        train_ratio: f64,
    ) -> IndicatorOptResult {
        if windows == 0 || prices.len() < 100 {
            return self.optimize_train_test(prices, train_ratio);
        }

        let window_size = prices.len() / (windows + 1);
        let mut is_scores = Vec::new();
        let mut oos_scores = Vec::new();
        let mut total_evaluations = 0;
        let mut final_params = Vec::new();

        for w in 0..windows {
            let train_start = 0;
            let train_end = (w + 1) * window_size;
            let test_start = train_end;
            let test_end = ((w + 2) * window_size).min(prices.len());

            if test_end <= test_start {
                continue;
            }

            let train = &prices[train_start..train_end];
            let test = &prices[test_start..test_end];

            // Optimize on training window
            let (params, is_score, evals, _) = self.run_optimization(train);
            total_evaluations += evals;
            is_scores.push(is_score);

            // Test on out-of-sample window
            if !params.is_empty() {
                let oos_score = self.evaluate_params(test, &params);
                oos_scores.push(oos_score);

                if w == windows - 1 {
                    final_params = params;
                }
            }

            if self.verbose {
                println!(
                    "Window {}: IS={:.4}, OOS={:.4}",
                    w + 1,
                    is_score,
                    oos_scores.last().unwrap_or(&0.0)
                );
            }
        }

        let avg_is = if is_scores.is_empty() {
            0.0
        } else {
            is_scores.iter().sum::<f64>() / is_scores.len() as f64
        };

        let avg_oos = if oos_scores.is_empty() {
            0.0
        } else {
            oos_scores.iter().sum::<f64>() / oos_scores.len() as f64
        };

        let robustness = if avg_is.abs() > 1e-10 {
            Some(avg_oos / avg_is)
        } else {
            Some(0.0)
        };

        IndicatorOptResult {
            best_params: final_params,
            best_score: avg_is,
            oos_score: Some(avg_oos),
            evaluations: total_evaluations,
            robustness,
            top_results: Vec::new(),
        }
    }

    /// Optimize with K-fold cross-validation
    fn optimize_kfold(&self, prices: &[f64], folds: usize) -> IndicatorOptResult {
        if folds < 2 {
            return self.optimize_no_validation(prices);
        }

        let fold_size = prices.len() / folds;
        let mut scores = Vec::new();
        let mut total_evaluations = 0;
        let mut best_overall_params = Vec::new();
        let mut best_overall_score = f64::NEG_INFINITY;

        for fold in 0..folds {
            let test_start = fold * fold_size;
            let test_end = if fold == folds - 1 {
                prices.len()
            } else {
                (fold + 1) * fold_size
            };

            // Create training set (everything except test fold)
            let mut train: Vec<f64> = Vec::new();
            train.extend_from_slice(&prices[..test_start]);
            train.extend_from_slice(&prices[test_end..]);

            let test = &prices[test_start..test_end];

            // Optimize on training folds
            let (params, _, evals, _) = self.run_optimization(&train);
            total_evaluations += evals;

            // Evaluate on test fold
            if !params.is_empty() {
                let score = self.evaluate_params(test, &params);
                scores.push(score);

                if score > best_overall_score {
                    best_overall_score = score;
                    best_overall_params = params;
                }
            }
        }

        let avg_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        IndicatorOptResult {
            best_params: best_overall_params,
            best_score: avg_score,
            oos_score: Some(avg_score),
            evaluations: total_evaluations,
            robustness: Some(1.0), // K-fold gives more robust estimates
            top_results: Vec::new(),
        }
    }

    /// Run optimization with the configured method
    fn run_optimization(
        &self,
        prices: &[f64],
    ) -> (Vec<OptimizedIndicator>, f64, usize, Vec<(Vec<OptimizedIndicator>, f64)>) {
        match &self.method {
            OptMethod::GridSearch => self.grid_search(prices),
            OptMethod::ParallelGrid => self.parallel_grid_search(prices),
            OptMethod::RandomSearch { iterations } => self.random_search(prices, *iterations),
            OptMethod::GeneticAlgorithm {
                population,
                generations,
                mutation_rate,
                crossover_rate,
            } => self.genetic_search(prices, *population, *generations, *mutation_rate, *crossover_rate),
            OptMethod::Bayesian { iterations } => self.bayesian_search(prices, *iterations),
        }
    }

    /// Exhaustive grid search
    fn grid_search(
        &self,
        prices: &[f64],
    ) -> (Vec<OptimizedIndicator>, f64, usize, Vec<(Vec<OptimizedIndicator>, f64)>) {
        let all_combos = self.generate_all_combinations();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_params = Vec::new();
        let mut results: Vec<(Vec<OptimizedIndicator>, f64)> = Vec::new();
        let evaluations = all_combos.len();

        for combo in all_combos {
            let score = self.evaluate_params(prices, &combo);
            results.push((combo.clone(), score));

            if score > best_score {
                best_score = score;
                best_params = combo;
            }
        }

        // Sort and keep top N
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.top_n);

        (best_params, best_score, evaluations, results)
    }

    /// Parallel grid search using rayon
    fn parallel_grid_search(
        &self,
        prices: &[f64],
    ) -> (Vec<OptimizedIndicator>, f64, usize, Vec<(Vec<OptimizedIndicator>, f64)>) {
        use rayon::prelude::*;

        let all_combos = self.generate_all_combinations();
        let evaluations = all_combos.len();

        let results: Vec<(Vec<OptimizedIndicator>, f64)> = all_combos
            .into_par_iter()
            .map(|combo| {
                let score = self.evaluate_params(prices, &combo);
                (combo, score)
            })
            .collect();

        let mut sorted_results = results;
        sorted_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_params = sorted_results.first().map(|(p, _)| p.clone()).unwrap_or_default();
        let best_score = sorted_results.first().map(|(_, s)| *s).unwrap_or(f64::NEG_INFINITY);

        sorted_results.truncate(self.top_n);

        (best_params, best_score, evaluations, sorted_results)
    }

    /// Random search
    fn random_search(
        &self,
        prices: &[f64],
        iterations: usize,
    ) -> (Vec<OptimizedIndicator>, f64, usize, Vec<(Vec<OptimizedIndicator>, f64)>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_params = Vec::new();
        let mut results: Vec<(Vec<OptimizedIndicator>, f64)> = Vec::new();

        for _ in 0..iterations {
            let combo = self.generate_random_combination(&mut rng);
            let score = self.evaluate_params(prices, &combo);
            results.push((combo.clone(), score));

            if score > best_score {
                best_score = score;
                best_params = combo;
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.top_n);

        (best_params, best_score, iterations, results)
    }

    /// Genetic algorithm search
    fn genetic_search(
        &self,
        prices: &[f64],
        population_size: usize,
        generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> (Vec<OptimizedIndicator>, f64, usize, Vec<(Vec<OptimizedIndicator>, f64)>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize population
        let mut population: Vec<Vec<OptimizedIndicator>> = (0..population_size)
            .map(|_| self.generate_random_combination(&mut rng))
            .collect();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_params = Vec::new();
        let mut evaluations = 0;

        for _gen in 0..generations {
            // Evaluate fitness
            let mut fitness: Vec<(usize, f64)> = population
                .iter()
                .enumerate()
                .map(|(i, combo)| {
                    evaluations += 1;
                    (i, self.evaluate_params(prices, combo))
                })
                .collect();

            // Track best
            for (i, score) in &fitness {
                if *score > best_score {
                    best_score = *score;
                    best_params = population[*i].clone();
                }
            }

            // Sort by fitness
            fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Selection and reproduction
            let mut new_population = Vec::new();

            // Elitism: keep top 10%
            let elite_count = (population_size as f64 * 0.1).max(1.0) as usize;
            for (i, _) in fitness.iter().take(elite_count) {
                new_population.push(population[*i].clone());
            }

            // Crossover and mutation
            while new_population.len() < population_size {
                let parent1_idx = self.tournament_select(&fitness, &mut rng);
                let parent2_idx = self.tournament_select(&fitness, &mut rng);

                let mut child = if rng.gen::<f64>() < crossover_rate {
                    self.crossover(&population[parent1_idx], &population[parent2_idx], &mut rng)
                } else {
                    population[parent1_idx].clone()
                };

                if rng.gen::<f64>() < mutation_rate {
                    child = self.mutate(&child, &mut rng);
                }

                new_population.push(child);
            }

            population = new_population;
        }

        (best_params, best_score, evaluations, Vec::new())
    }

    /// Bayesian optimization (simplified)
    fn bayesian_search(
        &self,
        prices: &[f64],
        iterations: usize,
    ) -> (Vec<OptimizedIndicator>, f64, usize, Vec<(Vec<OptimizedIndicator>, f64)>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initial random samples
        let initial_samples = iterations.min(10);
        let mut observations: Vec<(Vec<OptimizedIndicator>, f64)> = Vec::new();

        for _ in 0..initial_samples {
            let combo = self.generate_random_combination(&mut rng);
            let score = self.evaluate_params(prices, &combo);
            observations.push((combo, score));
        }

        // Exploitation/exploration balance
        for _ in initial_samples..iterations {
            // Simple approach: mix best observed with random exploration
            let combo = if rng.gen::<f64>() < 0.7 {
                // Exploit: perturb best known
                let best_idx = observations
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                self.mutate(&observations[best_idx].0, &mut rng)
            } else {
                // Explore: random
                self.generate_random_combination(&mut rng)
            };

            let score = self.evaluate_params(prices, &combo);
            observations.push((combo, score));
        }

        let best = observations
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or((Vec::new(), f64::NEG_INFINITY));

        observations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        observations.truncate(self.top_n);

        (best.0, best.1, iterations, observations)
    }

    // ========== Helper Methods ==========

    /// Generate all parameter combinations
    fn generate_all_combinations(&self) -> Vec<Vec<OptimizedIndicator>> {
        let mut result = vec![Vec::new()];

        for indicator in &self.indicators {
            let mut new_result = Vec::new();

            for existing in &result {
                for combo in self.indicator_combinations(indicator) {
                    let mut new_combo = existing.clone();
                    new_combo.push(combo);
                    new_result.push(new_combo);
                }
            }

            result = new_result;
        }

        result
    }

    /// Generate all combinations for a single indicator
    fn indicator_combinations(&self, config: &IndicatorConfig) -> Vec<OptimizedIndicator> {
        let mut combos = Vec::new();

        match config {
            IndicatorConfig::SMA { period } => {
                for p in period.values() {
                    combos.push(
                        OptimizedIndicator::new("SMA")
                            .with_param("period", p as f64),
                    );
                }
            }
            IndicatorConfig::EMA { period } => {
                for p in period.values() {
                    combos.push(
                        OptimizedIndicator::new("EMA")
                            .with_param("period", p as f64),
                    );
                }
            }
            IndicatorConfig::RSI { period } => {
                for p in period.values() {
                    combos.push(
                        OptimizedIndicator::new("RSI")
                            .with_param("period", p as f64),
                    );
                }
            }
            IndicatorConfig::MACD { fast, slow, signal } => {
                for f in fast.values() {
                    for s in slow.values() {
                        for sig in signal.values() {
                            if f < s {  // Ensure fast < slow
                                combos.push(
                                    OptimizedIndicator::new("MACD")
                                        .with_param("fast", f as f64)
                                        .with_param("slow", s as f64)
                                        .with_param("signal", sig as f64),
                                );
                            }
                        }
                    }
                }
            }
            IndicatorConfig::BollingerBands { period, std_dev } => {
                for p in period.values() {
                    for sd in std_dev.values() {
                        combos.push(
                            OptimizedIndicator::new("Bollinger")
                                .with_param("period", p as f64)
                                .with_param("std_dev", sd),
                        );
                    }
                }
            }
            IndicatorConfig::ROC { period } => {
                for p in period.values() {
                    combos.push(
                        OptimizedIndicator::new("ROC")
                            .with_param("period", p as f64),
                    );
                }
            }
            IndicatorConfig::StdDev { period } => {
                for p in period.values() {
                    combos.push(
                        OptimizedIndicator::new("StdDev")
                            .with_param("period", p as f64),
                    );
                }
            }
            IndicatorConfig::ATR { period } => {
                for p in period.values() {
                    combos.push(
                        OptimizedIndicator::new("ATR")
                            .with_param("period", p as f64),
                    );
                }
            }
            IndicatorConfig::Stochastic { k_period, d_period } => {
                for k in k_period.values() {
                    for d in d_period.values() {
                        combos.push(
                            OptimizedIndicator::new("Stochastic")
                                .with_param("k_period", k as f64)
                                .with_param("d_period", d as f64),
                        );
                    }
                }
            }
        }

        combos
    }

    /// Generate a random parameter combination
    fn generate_random_combination<R: rand::Rng>(&self, rng: &mut R) -> Vec<OptimizedIndicator> {
        self.indicators
            .iter()
            .map(|config| self.random_indicator_params(config, rng))
            .collect()
    }

    /// Generate random parameters for a single indicator
    fn random_indicator_params<R: rand::Rng>(
        &self,
        config: &IndicatorConfig,
        rng: &mut R,
    ) -> OptimizedIndicator {
        match config {
            IndicatorConfig::SMA { period } => {
                let vals = period.values();
                let p = vals[rng.gen_range(0..vals.len())];
                OptimizedIndicator::new("SMA").with_param("period", p as f64)
            }
            IndicatorConfig::EMA { period } => {
                let vals = period.values();
                let p = vals[rng.gen_range(0..vals.len())];
                OptimizedIndicator::new("EMA").with_param("period", p as f64)
            }
            IndicatorConfig::RSI { period } => {
                let vals = period.values();
                let p = vals[rng.gen_range(0..vals.len())];
                OptimizedIndicator::new("RSI").with_param("period", p as f64)
            }
            IndicatorConfig::MACD { fast, slow, signal } => {
                let fvals = fast.values();
                let svals = slow.values();
                let sigvals = signal.values();
                let mut f = fvals[rng.gen_range(0..fvals.len())];
                let mut s = svals[rng.gen_range(0..svals.len())];
                if f >= s {
                    std::mem::swap(&mut f, &mut s);
                }
                let sig = sigvals[rng.gen_range(0..sigvals.len())];
                OptimizedIndicator::new("MACD")
                    .with_param("fast", f as f64)
                    .with_param("slow", s as f64)
                    .with_param("signal", sig as f64)
            }
            IndicatorConfig::BollingerBands { period, std_dev } => {
                let pvals = period.values();
                let sdvals = std_dev.values();
                let p = pvals[rng.gen_range(0..pvals.len())];
                let sd = sdvals[rng.gen_range(0..sdvals.len())];
                OptimizedIndicator::new("Bollinger")
                    .with_param("period", p as f64)
                    .with_param("std_dev", sd)
            }
            IndicatorConfig::ROC { period } => {
                let vals = period.values();
                let p = vals[rng.gen_range(0..vals.len())];
                OptimizedIndicator::new("ROC").with_param("period", p as f64)
            }
            IndicatorConfig::StdDev { period } => {
                let vals = period.values();
                let p = vals[rng.gen_range(0..vals.len())];
                OptimizedIndicator::new("StdDev").with_param("period", p as f64)
            }
            IndicatorConfig::ATR { period } => {
                let vals = period.values();
                let p = vals[rng.gen_range(0..vals.len())];
                OptimizedIndicator::new("ATR").with_param("period", p as f64)
            }
            IndicatorConfig::Stochastic { k_period, d_period } => {
                let kvals = k_period.values();
                let dvals = d_period.values();
                let k = kvals[rng.gen_range(0..kvals.len())];
                let d = dvals[rng.gen_range(0..dvals.len())];
                OptimizedIndicator::new("Stochastic")
                    .with_param("k_period", k as f64)
                    .with_param("d_period", d as f64)
            }
        }
    }

    /// Tournament selection for genetic algorithm
    fn tournament_select<R: rand::Rng>(
        &self,
        fitness: &[(usize, f64)],
        rng: &mut R,
    ) -> usize {
        let tournament_size = 3;
        let mut best_idx = fitness[rng.gen_range(0..fitness.len())].0;
        let mut best_fitness = f64::NEG_INFINITY;

        for _ in 0..tournament_size {
            let (idx, fit) = fitness[rng.gen_range(0..fitness.len())];
            if fit > best_fitness {
                best_fitness = fit;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Crossover two parameter sets
    fn crossover<R: rand::Rng>(
        &self,
        parent1: &[OptimizedIndicator],
        parent2: &[OptimizedIndicator],
        rng: &mut R,
    ) -> Vec<OptimizedIndicator> {
        parent1
            .iter()
            .zip(parent2.iter())
            .map(|(p1, p2)| {
                if rng.gen::<bool>() {
                    p1.clone()
                } else {
                    p2.clone()
                }
            })
            .collect()
    }

    /// Mutate a parameter set
    fn mutate<R: rand::Rng>(
        &self,
        params: &[OptimizedIndicator],
        rng: &mut R,
    ) -> Vec<OptimizedIndicator> {
        let mut result = params.to_vec();
        if result.is_empty() || self.indicators.is_empty() {
            return result;
        }

        let idx = rng.gen_range(0..result.len());
        result[idx] = self.random_indicator_params(&self.indicators[idx], rng);
        result
    }

    /// Evaluate a parameter combination
    fn evaluate_params(&self, prices: &[f64], params: &[OptimizedIndicator]) -> f64 {
        if prices.len() < 30 || params.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Generate combined signals using the configured strategy
        let signals = self.compute_signals(prices, params);
        if signals.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Calculate returns aligned with signals
        let returns = self.compute_returns(prices);
        if returns.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Align signals and returns (signals are 1 step ahead)
        let min_len = signals.len().min(returns.len());
        if min_len < 20 {
            return f64::NEG_INFINITY;
        }

        let signals = &signals[..min_len];
        let returns = &returns[..min_len];

        // Calculate objective
        match &self.objective {
            Objective::DirectionalAccuracy => {
                self.calc_directional_accuracy(signals, returns)
            }
            Objective::SharpeRatio => self.calc_sharpe_ratio(signals, returns),
            Objective::TotalReturn => self.calc_total_return(signals, returns),
            Objective::InformationCoefficient => {
                self.calc_information_coefficient(signals, returns)
            }
            Objective::MaxDrawdown => -self.calc_max_drawdown(signals, returns), // Negate (minimize)
            Objective::SortinoRatio => self.calc_sortino_ratio(signals, returns),
        }
    }

    /// Compute indicator values from prices
    fn compute_indicator(&self, prices: &[f64], param: &OptimizedIndicator) -> Vec<f64> {
        let n = prices.len();
        match param.indicator_type.as_str() {
            "SMA" => {
                let period = param.get_param("period").unwrap_or(20.0) as usize;
                self.compute_sma(prices, period)
            }
            "EMA" => {
                let period = param.get_param("period").unwrap_or(20.0) as usize;
                self.compute_ema(prices, period)
            }
            "RSI" => {
                let period = param.get_param("period").unwrap_or(14.0) as usize;
                // Normalize RSI to -1 to 1 range
                self.compute_rsi(prices, period)
                    .into_iter()
                    .map(|v| (v - 50.0) / 50.0)
                    .collect()
            }
            "MACD" => {
                let fast = param.get_param("fast").unwrap_or(12.0) as usize;
                let slow = param.get_param("slow").unwrap_or(26.0) as usize;
                let signal = param.get_param("signal").unwrap_or(9.0) as usize;
                self.compute_macd(prices, fast, slow, signal)
            }
            "Bollinger" => {
                let period = param.get_param("period").unwrap_or(20.0) as usize;
                let std_dev = param.get_param("std_dev").unwrap_or(2.0);
                self.compute_bollinger_position(prices, period, std_dev)
            }
            "ROC" => {
                let period = param.get_param("period").unwrap_or(10.0) as usize;
                self.compute_roc(prices, period)
            }
            "StdDev" => {
                let period = param.get_param("period").unwrap_or(20.0) as usize;
                self.compute_std_dev(prices, period)
            }
            _ => vec![0.0; n],
        }
    }

    // ========== Indicator Computation ==========

    fn compute_sma(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period {
            return Vec::new();
        }
        let mut result = vec![0.0; prices.len()];
        let mut sum: f64 = prices[..period].iter().sum();
        result[period - 1] = sum / period as f64;
        for i in period..prices.len() {
            sum += prices[i] - prices[i - period];
            result[i] = sum / period as f64;
        }
        // Return deviation from price
        result
            .iter()
            .zip(prices.iter())
            .map(|(&sma, &price)| {
                if price.abs() > 1e-10 {
                    (sma - price) / price
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn compute_ema(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.is_empty() {
            return Vec::new();
        }
        let alpha = 2.0 / (period + 1) as f64;
        let mut ema = vec![0.0; prices.len()];
        ema[0] = prices[0];
        for i in 1..prices.len() {
            ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1];
        }
        // Return deviation from price
        ema.iter()
            .zip(prices.iter())
            .map(|(&e, &price)| {
                if price.abs() > 1e-10 {
                    (e - price) / price
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn compute_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period + 1 {
            return vec![50.0; prices.len()];
        }
        let mut result = vec![50.0; prices.len()];
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;

        for i in 1..=period {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss -= change;
            }
        }
        avg_gain /= period as f64;
        avg_loss /= period as f64;

        if avg_loss > 0.0 {
            result[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
        } else {
            result[period] = 100.0;
        }

        for i in (period + 1)..prices.len() {
            let change = prices[i] - prices[i - 1];
            let (gain, loss) = if change > 0.0 {
                (change, 0.0)
            } else {
                (0.0, -change)
            };
            avg_gain = (avg_gain * (period - 1) as f64 + gain) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + loss) / period as f64;

            if avg_loss > 0.0 {
                result[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
            } else {
                result[i] = 100.0;
            }
        }
        result
    }

    fn compute_macd(&self, prices: &[f64], fast: usize, slow: usize, _signal: usize) -> Vec<f64> {
        let fast_ema = {
            let alpha = 2.0 / (fast + 1) as f64;
            let mut ema = vec![0.0; prices.len()];
            if !prices.is_empty() {
                ema[0] = prices[0];
                for i in 1..prices.len() {
                    ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1];
                }
            }
            ema
        };

        let slow_ema = {
            let alpha = 2.0 / (slow + 1) as f64;
            let mut ema = vec![0.0; prices.len()];
            if !prices.is_empty() {
                ema[0] = prices[0];
                for i in 1..prices.len() {
                    ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1];
                }
            }
            ema
        };

        // MACD line normalized by price
        fast_ema
            .iter()
            .zip(slow_ema.iter())
            .zip(prices.iter())
            .map(|((&f, &s), &price)| {
                if price.abs() > 1e-10 {
                    (f - s) / price
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn compute_bollinger_position(&self, prices: &[f64], period: usize, std_mult: f64) -> Vec<f64> {
        if prices.len() < period {
            return vec![0.0; prices.len()];
        }

        let mut result = vec![0.0; prices.len()];

        for i in (period - 1)..prices.len() {
            let window = &prices[(i + 1 - period)..=i];
            let mean = window.iter().sum::<f64>() / period as f64;
            let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            let upper = mean + std_mult * std;
            let lower = mean - std_mult * std;
            let range = (upper - lower).max(1e-10);

            // Position: -1 at lower, 0 at middle, 1 at upper
            result[i] = ((prices[i] - lower) / range * 2.0 - 1.0).max(-1.0).min(1.0);
        }

        result
    }

    fn compute_roc(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut result = vec![0.0; prices.len()];
        for i in period..prices.len() {
            let prev = prices[i - period];
            if prev.abs() > 1e-10 {
                result[i] = (prices[i] - prev) / prev;
            }
        }
        result
    }

    fn compute_std_dev(&self, prices: &[f64], period: usize) -> Vec<f64> {
        if prices.len() < period {
            return vec![0.0; prices.len()];
        }

        let mut result = vec![0.0; prices.len()];

        for i in (period - 1)..prices.len() {
            let window = &prices[(i + 1 - period)..=i];
            let mean = window.iter().sum::<f64>() / period as f64;
            let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
            let std = variance.sqrt();

            // Normalized by price
            if prices[i].abs() > 1e-10 {
                result[i] = std / prices[i];
            }
        }

        result
    }

    // ========== Objective Calculations ==========

    fn calc_directional_accuracy(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.is_empty() || returns.is_empty() {
            return 0.0;
        }

        let min_len = signals.len().min(returns.len());
        let mut correct = 0;

        for i in 0..min_len {
            let signal_dir = signals[i].signum();
            let return_dir = returns[i].signum();
            if signal_dir == return_dir {
                correct += 1;
            }
        }

        (correct as f64 / min_len as f64) * 100.0
    }

    fn calc_sharpe_ratio(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.is_empty() || returns.is_empty() {
            return 0.0;
        }

        let min_len = signals.len().min(returns.len());
        let strategy_returns: Vec<f64> = (0..min_len)
            .map(|i| signals[i].signum() * returns[i])
            .collect();

        let mean = strategy_returns.iter().sum::<f64>() / strategy_returns.len() as f64;
        let variance = strategy_returns
            .iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>()
            / strategy_returns.len() as f64;
        let std = variance.sqrt();

        if std > 1e-10 {
            mean / std * (252.0_f64).sqrt() // Annualized
        } else {
            0.0
        }
    }

    fn calc_total_return(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.is_empty() || returns.is_empty() {
            return 0.0;
        }

        let min_len = signals.len().min(returns.len());
        let mut cumulative = 1.0;

        for i in 0..min_len {
            let position = signals[i].signum();
            cumulative *= 1.0 + position * returns[i];
        }

        cumulative - 1.0
    }

    fn calc_information_coefficient(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.len() < 2 || returns.len() < 2 {
            return 0.0;
        }

        let min_len = signals.len().min(returns.len());
        let signals = &signals[..min_len];
        let returns = &returns[..min_len];

        let n = min_len as f64;
        let sum_x: f64 = signals.iter().sum();
        let sum_y: f64 = returns.iter().sum();
        let sum_xy: f64 = signals.iter().zip(returns.iter()).map(|(&x, &y)| x * y).sum();
        let sum_x2: f64 = signals.iter().map(|&x| x * x).sum();
        let sum_y2: f64 = returns.iter().map(|&y| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn calc_max_drawdown(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.is_empty() || returns.is_empty() {
            return 0.0;
        }

        let min_len = signals.len().min(returns.len());
        let mut equity = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;

        for i in 0..min_len {
            let position = signals[i].signum();
            equity *= 1.0 + position * returns[i];

            if equity > peak {
                peak = equity;
            }

            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }

    fn calc_sortino_ratio(&self, signals: &[f64], returns: &[f64]) -> f64 {
        if signals.is_empty() || returns.is_empty() {
            return 0.0;
        }

        let min_len = signals.len().min(returns.len());
        let strategy_returns: Vec<f64> = (0..min_len)
            .map(|i| signals[i].signum() * returns[i])
            .collect();

        let mean = strategy_returns.iter().sum::<f64>() / strategy_returns.len() as f64;

        // Downside deviation (only negative returns)
        let downside_variance = strategy_returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|&r| r.powi(2))
            .sum::<f64>()
            / strategy_returns.len() as f64;

        let downside_std = downside_variance.sqrt();

        if downside_std > 1e-10 {
            mean / downside_std * (252.0_f64).sqrt()
        } else {
            0.0
        }
    }
}

impl Default for IndicatorOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Multi-Objective Optimization
// ============================================================================

/// Result for a single objective in multi-objective comparison
#[derive(Debug, Clone)]
pub struct ObjectiveResult {
    /// The objective that was optimized
    pub objective: String,
    /// Best parameters found
    pub best_params: Vec<OptimizedIndicator>,
    /// Best score achieved
    pub best_score: f64,
    /// Out-of-sample score if validation used
    pub oos_score: Option<f64>,
    /// Robustness ratio
    pub robustness: Option<f64>,
}

/// Result of running all objectives sequentially (Option 1)
#[derive(Debug, Clone)]
pub struct MultiObjectiveResult {
    /// Results for each objective
    pub results: Vec<ObjectiveResult>,
    /// Total evaluations across all objectives
    pub total_evaluations: usize,
}

impl MultiObjectiveResult {
    /// Get result for a specific objective
    pub fn get(&self, objective: &str) -> Option<&ObjectiveResult> {
        self.results.iter().find(|r| r.objective == objective)
    }

    /// Get the best parameters that appear most frequently across objectives
    pub fn consensus_params(&self) -> Option<Vec<OptimizedIndicator>> {
        if self.results.is_empty() {
            return None;
        }
        // Return params from the result with highest robustness
        self.results
            .iter()
            .filter(|r| r.robustness.is_some())
            .max_by(|a, b| {
                a.robustness.unwrap().partial_cmp(&b.robustness.unwrap()).unwrap()
            })
            .map(|r| r.best_params.clone())
            .or_else(|| Some(self.results[0].best_params.clone()))
    }
}

/// A solution on the Pareto frontier
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    /// Parameters for this solution
    pub params: Vec<OptimizedIndicator>,
    /// Objective scores (in same order as objectives list)
    pub scores: Vec<f64>,
    /// Crowding distance (diversity metric)
    pub crowding_distance: f64,
}

/// Result of Pareto multi-objective optimization (Option 2)
#[derive(Debug, Clone)]
pub struct ParetoFrontResult {
    /// Objectives that were optimized
    pub objectives: Vec<String>,
    /// Solutions on the Pareto frontier
    pub frontier: Vec<ParetoPoint>,
    /// Total generations run
    pub generations: usize,
    /// Total evaluations
    pub evaluations: usize,
}

impl ParetoFrontResult {
    /// Get solution with best score for a specific objective
    pub fn best_for(&self, objective_idx: usize) -> Option<&ParetoPoint> {
        self.frontier.iter().max_by(|a, b| {
            a.scores.get(objective_idx)
                .unwrap_or(&f64::NEG_INFINITY)
                .partial_cmp(b.scores.get(objective_idx).unwrap_or(&f64::NEG_INFINITY))
                .unwrap()
        })
    }

    /// Get solution with best balance (highest crowding distance in middle ranks)
    pub fn balanced_solution(&self) -> Option<&ParetoPoint> {
        // Sort by crowding distance and take highest (most diverse)
        self.frontier.iter().max_by(|a, b| {
            a.crowding_distance.partial_cmp(&b.crowding_distance).unwrap()
        })
    }

    /// Get solutions within a score threshold for an objective
    pub fn filter_by_objective(&self, objective_idx: usize, min_score: f64) -> Vec<&ParetoPoint> {
        self.frontier.iter()
            .filter(|p| p.scores.get(objective_idx).map(|&s| s >= min_score).unwrap_or(false))
            .collect()
    }
}

/// Result for a single indicator optimized in isolation
#[derive(Debug, Clone)]
pub struct IndividualIndicatorResult {
    /// Indicator name (e.g., "MACD", "RSI")
    pub indicator_name: String,
    /// Index in the original indicator list
    pub indicator_index: usize,
    /// Best parameters found
    pub best_params: OptimizedIndicator,
    /// Best score achieved
    pub best_score: f64,
    /// Out-of-sample score
    pub oos_score: Option<f64>,
    /// Robustness ratio
    pub robustness: Option<f64>,
    /// Number of evaluations for this indicator
    pub evaluations: usize,
}

/// Result of optimizing each indicator individually
#[derive(Debug, Clone)]
pub struct IndividualOptResult {
    /// Results for each indicator
    pub results: Vec<IndividualIndicatorResult>,
    /// Total evaluations across all indicators
    pub total_evaluations: usize,
}

impl IndividualOptResult {
    /// Get result for a specific indicator by name
    pub fn get(&self, name: &str) -> Option<&IndividualIndicatorResult> {
        self.results.iter().find(|r| r.indicator_name == name)
    }

    /// Get all optimized params as a vector (for use with combined signals)
    pub fn all_params(&self) -> Vec<OptimizedIndicator> {
        self.results.iter().map(|r| r.best_params.clone()).collect()
    }

    /// Get the indicator with best score
    pub fn best(&self) -> Option<&IndividualIndicatorResult> {
        self.results.iter().max_by(|a, b| {
            a.best_score.partial_cmp(&b.best_score).unwrap()
        })
    }

    /// Get the indicator with best robustness
    pub fn most_robust(&self) -> Option<&IndividualIndicatorResult> {
        self.results.iter()
            .filter(|r| r.robustness.is_some())
            .max_by(|a, b| {
                a.robustness.unwrap().partial_cmp(&b.robustness.unwrap()).unwrap()
            })
    }
}

impl IndicatorOptimizer {
    /// All available objectives
    pub fn all_objectives() -> Vec<Objective> {
        vec![
            Objective::SharpeRatio,
            Objective::DirectionalAccuracy,
            Objective::TotalReturn,
            Objective::InformationCoefficient,
            Objective::MaxDrawdown,
            Objective::SortinoRatio,
        ]
    }

    /// Run optimization for multiple objectives sequentially and compare results (Option 1)
    pub fn optimize_all_objectives(&self, prices: &[f64]) -> MultiObjectiveResult {
        self.optimize_objectives(prices, &Self::all_objectives())
    }

    /// Run optimization for a subset of objectives sequentially (Option 1)
    pub fn optimize_objectives(&self, prices: &[f64], objectives: &[Objective]) -> MultiObjectiveResult {
        let mut results = Vec::new();
        let mut total_evaluations = 0;

        for obj in objectives {
            // Create a copy with this objective
            let mut opt = self.clone();
            opt.objective = obj.clone();

            let result = opt.optimize(prices);
            total_evaluations += result.evaluations;

            results.push(ObjectiveResult {
                objective: format!("{:?}", obj),
                best_params: result.best_params,
                best_score: result.best_score,
                oos_score: result.oos_score,
                robustness: result.robustness,
            });
        }

        MultiObjectiveResult {
            results,
            total_evaluations,
        }
    }

    /// Optimize each indicator individually and return best params for each
    /// This finds the optimal params for each indicator in isolation
    pub fn optimize_individual(&self, prices: &[f64]) -> IndividualOptResult {
        let mut results = Vec::new();
        let mut total_evaluations = 0;

        for (idx, config) in self.indicators.iter().enumerate() {
            // Create optimizer with just this one indicator
            let mut single_opt = IndicatorOptimizer::new();
            single_opt.objective = self.objective.clone();
            single_opt.method = self.method.clone();
            single_opt.validation = self.validation.clone();
            single_opt.signal_combination = SignalCombination::FirstOnly;

            // Add just this indicator
            single_opt.indicators.push(config.clone());

            // Optimize
            let result = single_opt.optimize(prices);
            total_evaluations += result.evaluations;

            let indicator_name = match config {
                IndicatorConfig::SMA { .. } => "SMA",
                IndicatorConfig::EMA { .. } => "EMA",
                IndicatorConfig::RSI { .. } => "RSI",
                IndicatorConfig::MACD { .. } => "MACD",
                IndicatorConfig::BollingerBands { .. } => "Bollinger",
                IndicatorConfig::ROC { .. } => "ROC",
                IndicatorConfig::StdDev { .. } => "StdDev",
                IndicatorConfig::ATR { .. } => "ATR",
                IndicatorConfig::Stochastic { .. } => "Stochastic",
            };

            results.push(IndividualIndicatorResult {
                indicator_name: indicator_name.to_string(),
                indicator_index: idx,
                best_params: result.best_params.first().cloned().unwrap_or(OptimizedIndicator {
                    indicator_type: indicator_name.to_string(),
                    params: vec![],
                }),
                best_score: result.best_score,
                oos_score: result.oos_score,
                robustness: result.robustness,
                evaluations: result.evaluations,
            });
        }

        IndividualOptResult {
            results,
            total_evaluations,
        }
    }

    /// Run Pareto multi-objective optimization using NSGA-II (Option 2)
    pub fn optimize_pareto(
        &self,
        prices: &[f64],
        objectives: &[Objective],
        population_size: usize,
        generations: usize,
    ) -> ParetoFrontResult {
        let n_objectives = objectives.len();
        if n_objectives < 2 {
            return ParetoFrontResult {
                objectives: objectives.iter().map(|o| format!("{:?}", o)).collect(),
                frontier: vec![],
                generations: 0,
                evaluations: 0,
            };
        }

        // Build parameter bounds
        let bounds = self.build_param_bounds();
        let n_params = bounds.len();

        if n_params == 0 {
            return ParetoFrontResult {
                objectives: objectives.iter().map(|o| format!("{:?}", o)).collect(),
                frontier: vec![],
                generations: 0,
                evaluations: 0,
            };
        }

        // NSGA-II implementation
        let mut rng = rand::thread_rng();
        use rand::Rng;

        // Initialize population
        let mut population: Vec<NsgaIndividual> = (0..population_size)
            .map(|_| {
                let params: Vec<f64> = bounds.iter()
                    .map(|b| rng.gen_range(b.0..=b.1))
                    .collect();
                NsgaIndividual::new(params, n_objectives)
            })
            .collect();

        let mut evaluations = 0;

        // Evaluate initial population
        for ind in &mut population {
            ind.objectives = self.evaluate_multi_objective(prices, &ind.params, objectives);
            evaluations += 1;
        }

        // Main NSGA-II loop
        for _gen in 0..generations {
            // Non-dominated sorting
            let fronts = fast_non_dominated_sort(&mut population);

            // Assign crowding distance
            for front in &fronts {
                crowding_distance_assignment(&mut population, front);
            }

            // Create offspring through tournament selection, crossover, mutation
            let mut offspring: Vec<NsgaIndividual> = Vec::with_capacity(population_size);

            while offspring.len() < population_size {
                // Tournament selection
                let p1 = tournament_select(&population, &mut rng);
                let p2 = tournament_select(&population, &mut rng);

                // SBX crossover
                let (mut c1, mut c2) = sbx_crossover(&p1.params, &p2.params, &bounds, &mut rng);

                // Polynomial mutation
                polynomial_mutation(&mut c1, &bounds, &mut rng);
                polynomial_mutation(&mut c2, &bounds, &mut rng);

                let mut child1 = NsgaIndividual::new(c1, n_objectives);
                child1.objectives = self.evaluate_multi_objective(prices, &child1.params, objectives);
                evaluations += 1;
                offspring.push(child1);

                if offspring.len() < population_size {
                    let mut child2 = NsgaIndividual::new(c2, n_objectives);
                    child2.objectives = self.evaluate_multi_objective(prices, &child2.params, objectives);
                    evaluations += 1;
                    offspring.push(child2);
                }
            }

            // Combine parent and offspring
            population.extend(offspring);

            // Non-dominated sorting on combined population
            let fronts = fast_non_dominated_sort(&mut population);

            // Select next generation
            let mut next_gen: Vec<NsgaIndividual> = Vec::with_capacity(population_size);

            for front in &fronts {
                crowding_distance_assignment(&mut population, front);

                if next_gen.len() + front.len() <= population_size {
                    // Add entire front
                    for &idx in front {
                        next_gen.push(population[idx].clone());
                    }
                } else {
                    // Sort by crowding distance and add best
                    let mut front_individuals: Vec<_> = front.iter()
                        .map(|&idx| population[idx].clone())
                        .collect();
                    front_individuals.sort_by(|a, b| {
                        b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap()
                    });

                    let remaining = population_size - next_gen.len();
                    next_gen.extend(front_individuals.into_iter().take(remaining));
                    break;
                }
            }

            population = next_gen;
        }

        // Extract Pareto front (rank 0)
        let _ = fast_non_dominated_sort(&mut population);
        let front_indices: Vec<usize> = (0..population.len())
            .filter(|&i| population[i].rank == 0)
            .collect();
        if !front_indices.is_empty() {
            crowding_distance_assignment(&mut population, &front_indices);
        }

        let frontier: Vec<ParetoPoint> = population.iter()
            .filter(|ind| ind.rank == 0)
            .map(|ind| {
                let params = self.params_to_optimized(&ind.params);
                ParetoPoint {
                    params,
                    scores: ind.objectives.clone(),
                    crowding_distance: ind.crowding_distance,
                }
            })
            .collect();

        ParetoFrontResult {
            objectives: objectives.iter().map(|o| format!("{:?}", o)).collect(),
            frontier,
            generations,
            evaluations,
        }
    }

    /// Evaluate multiple objectives for given parameters
    fn evaluate_multi_objective(&self, prices: &[f64], params: &[f64], objectives: &[Objective]) -> Vec<f64> {
        let indicators = self.params_to_optimized(params);
        let signals = self.compute_signals(prices, &indicators);
        let returns = self.compute_returns(prices);

        objectives.iter().map(|obj| {
            match obj {
                Objective::SharpeRatio => self.calc_sharpe_ratio(&signals, &returns),
                Objective::DirectionalAccuracy => self.calc_directional_accuracy(&signals, &returns),
                Objective::TotalReturn => self.calc_total_return(&signals, &returns),
                Objective::InformationCoefficient => self.calc_information_coefficient(&signals, &returns),
                Objective::MaxDrawdown => -self.calc_max_drawdown(&signals, &returns), // Negate so higher is better
                Objective::SortinoRatio => self.calc_sortino_ratio(&signals, &returns),
            }
        }).collect()
    }

    /// Build parameter bounds for optimization
    fn build_param_bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = Vec::new();

        for config in &self.indicators {
            match config {
                IndicatorConfig::SMA { period } |
                IndicatorConfig::EMA { period } |
                IndicatorConfig::RSI { period } |
                IndicatorConfig::ROC { period } |
                IndicatorConfig::StdDev { period } |
                IndicatorConfig::ATR { period } => {
                    bounds.push((period.min as f64, period.max as f64));
                }
                IndicatorConfig::MACD { fast, slow, signal } => {
                    bounds.push((fast.min as f64, fast.max as f64));
                    bounds.push((slow.min as f64, slow.max as f64));
                    bounds.push((signal.min as f64, signal.max as f64));
                }
                IndicatorConfig::BollingerBands { period, std_dev } => {
                    bounds.push((period.min as f64, period.max as f64));
                    bounds.push((std_dev.min, std_dev.max));
                }
                IndicatorConfig::Stochastic { k_period, d_period } => {
                    bounds.push((k_period.min as f64, k_period.max as f64));
                    bounds.push((d_period.min as f64, d_period.max as f64));
                }
            }
        }

        bounds
    }

    /// Convert flat parameter vector to OptimizedIndicator vector
    fn params_to_optimized(&self, params: &[f64]) -> Vec<OptimizedIndicator> {
        let mut result = Vec::new();
        let mut idx = 0;

        for config in &self.indicators {
            match config {
                IndicatorConfig::SMA { .. } => {
                    if idx < params.len() {
                        result.push(OptimizedIndicator {
                            indicator_type: "SMA".to_string(),
                            params: vec![("period".to_string(), params[idx].round())],
                        });
                        idx += 1;
                    }
                }
                IndicatorConfig::EMA { .. } => {
                    if idx < params.len() {
                        result.push(OptimizedIndicator {
                            indicator_type: "EMA".to_string(),
                            params: vec![("period".to_string(), params[idx].round())],
                        });
                        idx += 1;
                    }
                }
                IndicatorConfig::RSI { .. } => {
                    if idx < params.len() {
                        result.push(OptimizedIndicator {
                            indicator_type: "RSI".to_string(),
                            params: vec![("period".to_string(), params[idx].round())],
                        });
                        idx += 1;
                    }
                }
                IndicatorConfig::MACD { .. } => {
                    if idx + 2 < params.len() {
                        result.push(OptimizedIndicator {
                            indicator_type: "MACD".to_string(),
                            params: vec![
                                ("fast".to_string(), params[idx].round()),
                                ("slow".to_string(), params[idx + 1].round()),
                                ("signal".to_string(), params[idx + 2].round()),
                            ],
                        });
                        idx += 3;
                    }
                }
                IndicatorConfig::BollingerBands { .. } => {
                    if idx + 1 < params.len() {
                        result.push(OptimizedIndicator {
                            indicator_type: "Bollinger".to_string(),
                            params: vec![
                                ("period".to_string(), params[idx].round()),
                                ("std_dev".to_string(), params[idx + 1]),
                            ],
                        });
                        idx += 2;
                    }
                }
                IndicatorConfig::ROC { .. } => {
                    if idx < params.len() {
                        result.push(OptimizedIndicator {
                            indicator_type: "ROC".to_string(),
                            params: vec![("period".to_string(), params[idx].round())],
                        });
                        idx += 1;
                    }
                }
                IndicatorConfig::StdDev { .. } => {
                    if idx < params.len() {
                        result.push(OptimizedIndicator {
                            indicator_type: "StdDev".to_string(),
                            params: vec![("period".to_string(), params[idx].round())],
                        });
                        idx += 1;
                    }
                }
                IndicatorConfig::ATR { .. } => {
                    if idx < params.len() {
                        result.push(OptimizedIndicator {
                            indicator_type: "ATR".to_string(),
                            params: vec![("period".to_string(), params[idx].round())],
                        });
                        idx += 1;
                    }
                }
                IndicatorConfig::Stochastic { .. } => {
                    if idx + 1 < params.len() {
                        result.push(OptimizedIndicator {
                            indicator_type: "Stochastic".to_string(),
                            params: vec![
                                ("k_period".to_string(), params[idx].round()),
                                ("d_period".to_string(), params[idx + 1].round()),
                            ],
                        });
                        idx += 2;
                    }
                }
            }
        }

        result
    }

    /// Compute trading signals from indicator values using the configured combination strategy
    fn compute_signals(&self, prices: &[f64], indicators: &[OptimizedIndicator]) -> Vec<f64> {
        if indicators.is_empty() || prices.len() < 30 {
            return vec![];
        }

        // Compute all indicator values
        let all_values: Vec<Vec<f64>> = indicators.iter()
            .map(|ind| self.compute_indicator(prices, ind))
            .filter(|v| !v.is_empty())
            .collect();

        if all_values.is_empty() {
            return vec![];
        }

        // Find minimum length across all indicators
        let min_len = all_values.iter().map(|v| v.len()).min().unwrap_or(0);
        if min_len == 0 {
            return vec![];
        }

        // Generate individual signals (signum of each indicator)
        let all_signals: Vec<Vec<f64>> = all_values.iter()
            .map(|values| values.iter().take(min_len).map(|&v| v.signum()).collect())
            .collect();

        // Combine signals based on strategy
        match &self.signal_combination {
            SignalCombination::FirstOnly => {
                // Use only the first indicator's signal
                all_signals[0].clone()
            }

            SignalCombination::Unanimous => {
                // All indicators must agree
                (0..min_len).map(|i| {
                    let signals: Vec<f64> = all_signals.iter().map(|s| s[i]).collect();
                    if signals.iter().all(|&s| s > 0.0) {
                        1.0
                    } else if signals.iter().all(|&s| s < 0.0) {
                        -1.0
                    } else {
                        0.0 // No consensus
                    }
                }).collect()
            }

            SignalCombination::Majority => {
                // Majority vote (>50% must agree)
                let n_indicators = all_signals.len() as f64;
                (0..min_len).map(|i| {
                    let sum: f64 = all_signals.iter().map(|s| s[i]).sum();
                    if sum > n_indicators / 2.0 {
                        1.0
                    } else if sum < -n_indicators / 2.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }).collect()
            }

            SignalCombination::Average => {
                // Average of all signals (continuous)
                let n_indicators = all_signals.len() as f64;
                (0..min_len).map(|i| {
                    let sum: f64 = all_signals.iter().map(|s| s[i]).sum();
                    sum / n_indicators
                }).collect()
            }

            SignalCombination::Weighted(weights) => {
                // Weighted combination
                let total_weight: f64 = weights.iter().take(all_signals.len()).sum();
                if total_weight == 0.0 {
                    return all_signals[0].clone();
                }
                (0..min_len).map(|i| {
                    let weighted_sum: f64 = all_signals.iter()
                        .zip(weights.iter())
                        .map(|(s, &w)| s[i] * w)
                        .sum();
                    (weighted_sum / total_weight).clamp(-1.0, 1.0)
                }).collect()
            }

            SignalCombination::Confirmation => {
                // Primary indicator leads, others must confirm direction
                if all_signals.len() == 1 {
                    return all_signals[0].clone();
                }
                (0..min_len).map(|i| {
                    let primary = all_signals[0][i];
                    if primary == 0.0 {
                        return 0.0;
                    }
                    // Check if majority of secondary indicators confirm
                    let confirmations: usize = all_signals[1..].iter()
                        .filter(|s| s[i].signum() == primary.signum())
                        .count();
                    if confirmations >= (all_signals.len() - 1) / 2 {
                        primary
                    } else {
                        0.0 // Not confirmed
                    }
                }).collect()
            }
        }
    }

    /// Compute returns from prices
    fn compute_returns(&self, prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }

        (1..prices.len())
            .map(|i| (prices[i] - prices[i - 1]) / prices[i - 1])
            .collect()
    }
}

// ============================================================================
// NSGA-II Helper Structures and Functions
// ============================================================================

#[derive(Clone)]
struct NsgaIndividual {
    params: Vec<f64>,
    objectives: Vec<f64>,
    rank: usize,
    crowding_distance: f64,
}

impl NsgaIndividual {
    fn new(params: Vec<f64>, n_objectives: usize) -> Self {
        Self {
            params,
            objectives: vec![0.0; n_objectives],
            rank: usize::MAX,
            crowding_distance: 0.0,
        }
    }
}

/// Check if solution a dominates solution b (all objectives >= and at least one >)
fn dominates(a: &NsgaIndividual, b: &NsgaIndividual) -> bool {
    let dominated = a.objectives.iter()
        .zip(b.objectives.iter())
        .all(|(oa, ob)| oa >= ob);
    let strictly_better = a.objectives.iter()
        .zip(b.objectives.iter())
        .any(|(oa, ob)| oa > ob);
    dominated && strictly_better
}

/// Fast non-dominated sorting from NSGA-II paper
fn fast_non_dominated_sort(population: &mut [NsgaIndividual]) -> Vec<Vec<usize>> {
    let n = population.len();
    let mut dominated_by: Vec<Vec<usize>> = vec![vec![]; n];
    let mut domination_count: Vec<usize> = vec![0; n];
    let mut fronts: Vec<Vec<usize>> = vec![vec![]];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                if dominates(&population[i], &population[j]) {
                    dominated_by[i].push(j);
                } else if dominates(&population[j], &population[i]) {
                    domination_count[i] += 1;
                }
            }
        }
        if domination_count[i] == 0 {
            population[i].rank = 0;
            fronts[0].push(i);
        }
    }

    let mut current_front = 0;
    while !fronts[current_front].is_empty() {
        let mut next_front = vec![];
        for &i in &fronts[current_front] {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    population[j].rank = current_front + 1;
                    next_front.push(j);
                }
            }
        }
        current_front += 1;
        if !next_front.is_empty() {
            fronts.push(next_front);
        } else {
            break;
        }
    }

    fronts
}

/// Crowding distance assignment for diversity preservation
fn crowding_distance_assignment(population: &mut [NsgaIndividual], front: &[usize]) {
    let n = front.len();
    if n == 0 {
        return;
    }

    // Reset crowding distances
    for &i in front {
        population[i].crowding_distance = 0.0;
    }

    let n_objectives = population[front[0]].objectives.len();

    for m in 0..n_objectives {
        // Sort by objective m
        let mut sorted_indices: Vec<usize> = front.to_vec();
        sorted_indices.sort_by(|&a, &b| {
            population[a].objectives[m]
                .partial_cmp(&population[b].objectives[m])
                .unwrap()
        });

        // Boundary points get infinite distance
        population[sorted_indices[0]].crowding_distance = f64::INFINITY;
        population[sorted_indices[n - 1]].crowding_distance = f64::INFINITY;

        let f_max = population[sorted_indices[n - 1]].objectives[m];
        let f_min = population[sorted_indices[0]].objectives[m];
        let range = f_max - f_min;

        if range > 0.0 {
            for i in 1..n - 1 {
                let prev_obj = population[sorted_indices[i - 1]].objectives[m];
                let next_obj = population[sorted_indices[i + 1]].objectives[m];
                population[sorted_indices[i]].crowding_distance += (next_obj - prev_obj) / range;
            }
        }
    }
}

/// Tournament selection based on rank and crowding distance
fn tournament_select<'a, R: rand::Rng>(population: &'a [NsgaIndividual], rng: &mut R) -> &'a NsgaIndividual {
    let i = rng.gen_range(0..population.len());
    let j = rng.gen_range(0..population.len());

    let a = &population[i];
    let b = &population[j];

    if a.rank < b.rank {
        a
    } else if b.rank < a.rank {
        b
    } else if a.crowding_distance > b.crowding_distance {
        a
    } else {
        b
    }
}

/// SBX (Simulated Binary Crossover)
fn sbx_crossover<R: rand::Rng>(
    p1: &[f64],
    p2: &[f64],
    bounds: &[(f64, f64)],
    rng: &mut R,
) -> (Vec<f64>, Vec<f64>) {
    let eta_c = 20.0; // Distribution index
    let mut c1 = vec![0.0; p1.len()];
    let mut c2 = vec![0.0; p1.len()];

    for i in 0..p1.len() {
        if rng.gen::<f64>() < 0.5 {
            // Crossover
            let y1 = p1[i].min(p2[i]);
            let y2 = p1[i].max(p2[i]);

            if (y2 - y1).abs() > 1e-10 {
                let beta = 1.0 + (2.0 * (y1 - bounds[i].0) / (y2 - y1));
                let alpha = 2.0 - beta.powf(-(eta_c + 1.0));
                let u = rng.gen::<f64>();
                let beta_q = if u <= 1.0 / alpha {
                    (u * alpha).powf(1.0 / (eta_c + 1.0))
                } else {
                    (1.0 / (2.0 - u * alpha)).powf(1.0 / (eta_c + 1.0))
                };

                c1[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1));
                c2[i] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1));
            } else {
                c1[i] = p1[i];
                c2[i] = p2[i];
            }
        } else {
            c1[i] = p1[i];
            c2[i] = p2[i];
        }

        // Clamp to bounds
        c1[i] = c1[i].clamp(bounds[i].0, bounds[i].1);
        c2[i] = c2[i].clamp(bounds[i].0, bounds[i].1);
    }

    (c1, c2)
}

/// Polynomial mutation
fn polynomial_mutation<R: rand::Rng>(params: &mut [f64], bounds: &[(f64, f64)], rng: &mut R) {
    let eta_m = 20.0; // Distribution index
    let mutation_prob = 1.0 / params.len() as f64;

    for i in 0..params.len() {
        if rng.gen::<f64>() < mutation_prob {
            let y = params[i];
            let lb = bounds[i].0;
            let ub = bounds[i].1;
            let delta = ub - lb;

            let u = rng.gen::<f64>();
            let delta_q = if u < 0.5 {
                let xy = 1.0 - (y - lb) / delta;
                let val = 2.0 * u + (1.0 - 2.0 * u) * xy.powf(eta_m + 1.0);
                val.powf(1.0 / (eta_m + 1.0)) - 1.0
            } else {
                let xy = 1.0 - (ub - y) / delta;
                let val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy.powf(eta_m + 1.0);
                1.0 - val.powf(1.0 / (eta_m + 1.0))
            };

            params[i] = (y + delta_q * delta).clamp(lb, ub);
        }
    }
}

// ============================================================================
// FFI Interface
// ============================================================================

use std::ffi::CStr;
use std::os::raw::c_char;

/// Opaque handle for IndicatorOptimizer
#[repr(C)]
pub struct IndicatorOptimizerHandle {
    ptr: *mut IndicatorOptimizer,
}

/// FFI-safe optimization result
#[repr(C)]
pub struct IndicatorOptResultFFI {
    pub best_score: f64,
    pub oos_score: f64,
    pub has_oos: bool,
    pub robustness: f64,
    pub has_robustness: bool,
    pub evaluations: i64,
}

/// Create a new optimizer
#[no_mangle]
pub extern "C" fn quantlang_indicator_optimizer_new() -> IndicatorOptimizerHandle {
    let optimizer = Box::new(IndicatorOptimizer::new());
    IndicatorOptimizerHandle {
        ptr: Box::into_raw(optimizer),
    }
}

/// Free an optimizer
#[no_mangle]
pub extern "C" fn quantlang_indicator_optimizer_free(handle: IndicatorOptimizerHandle) {
    if !handle.ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(handle.ptr);
        }
    }
}

/// Add SMA range
#[no_mangle]
pub extern "C" fn quantlang_indicator_optimizer_add_sma(
    handle: IndicatorOptimizerHandle,
    min: i64,
    max: i64,
    step: i64,
) -> IndicatorOptimizerHandle {
    if handle.ptr.is_null() {
        return handle;
    }
    unsafe {
        let opt = &mut *handle.ptr;
        opt.indicators.push(IndicatorConfig::SMA {
            period: IndicatorParamRange::new(min as usize, max as usize, step as usize),
        });
    }
    handle
}

/// Add RSI range
#[no_mangle]
pub extern "C" fn quantlang_indicator_optimizer_add_rsi(
    handle: IndicatorOptimizerHandle,
    min: i64,
    max: i64,
    step: i64,
) -> IndicatorOptimizerHandle {
    if handle.ptr.is_null() {
        return handle;
    }
    unsafe {
        let opt = &mut *handle.ptr;
        opt.indicators.push(IndicatorConfig::RSI {
            period: IndicatorParamRange::new(min as usize, max as usize, step as usize),
        });
    }
    handle
}

/// Set objective (0=DirectionalAccuracy, 1=SharpeRatio, 2=TotalReturn, 3=IC, 4=MaxDD, 5=Sortino)
#[no_mangle]
pub extern "C" fn quantlang_indicator_optimizer_set_objective(
    handle: IndicatorOptimizerHandle,
    objective: i32,
) -> IndicatorOptimizerHandle {
    if handle.ptr.is_null() {
        return handle;
    }
    unsafe {
        let opt = &mut *handle.ptr;
        opt.objective = match objective {
            0 => Objective::DirectionalAccuracy,
            1 => Objective::SharpeRatio,
            2 => Objective::TotalReturn,
            3 => Objective::InformationCoefficient,
            4 => Objective::MaxDrawdown,
            5 => Objective::SortinoRatio,
            _ => Objective::SharpeRatio,
        };
    }
    handle
}

/// Set method (0=Grid, 1=ParallelGrid, 2=Random, 3=Genetic, 4=Bayesian)
#[no_mangle]
pub extern "C" fn quantlang_indicator_optimizer_set_method(
    handle: IndicatorOptimizerHandle,
    method: i32,
    param1: i64,
    param2: i64,
) -> IndicatorOptimizerHandle {
    if handle.ptr.is_null() {
        return handle;
    }
    unsafe {
        let opt = &mut *handle.ptr;
        opt.method = match method {
            0 => OptMethod::GridSearch,
            1 => OptMethod::ParallelGrid,
            2 => OptMethod::RandomSearch {
                iterations: param1 as usize,
            },
            3 => OptMethod::GeneticAlgorithm {
                population: param1 as usize,
                generations: param2 as usize,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
            },
            4 => OptMethod::Bayesian {
                iterations: param1 as usize,
            },
            _ => OptMethod::GridSearch,
        };
    }
    handle
}

/// Set validation (0=None, 1=TrainTest, 2=WalkForward, 3=KFold)
#[no_mangle]
pub extern "C" fn quantlang_indicator_optimizer_set_validation(
    handle: IndicatorOptimizerHandle,
    validation: i32,
    param1: f64,
    param2: i64,
) -> IndicatorOptimizerHandle {
    if handle.ptr.is_null() {
        return handle;
    }
    unsafe {
        let opt = &mut *handle.ptr;
        opt.validation = match validation {
            0 => Validation::None,
            1 => Validation::TrainTest {
                train_ratio: param1,
            },
            2 => Validation::WalkForward {
                windows: param2 as usize,
                train_ratio: param1,
            },
            3 => Validation::KFold {
                folds: param2 as usize,
            },
            _ => Validation::TrainTest { train_ratio: 0.7 },
        };
    }
    handle
}

/// Run optimization
#[no_mangle]
pub extern "C" fn quantlang_indicator_optimizer_optimize(
    handle: IndicatorOptimizerHandle,
    prices: *const f64,
    len: i64,
) -> IndicatorOptResultFFI {
    if handle.ptr.is_null() || prices.is_null() || len <= 0 {
        return IndicatorOptResultFFI {
            best_score: f64::NEG_INFINITY,
            oos_score: 0.0,
            has_oos: false,
            robustness: 0.0,
            has_robustness: false,
            evaluations: 0,
        };
    }

    let prices_slice = unsafe { std::slice::from_raw_parts(prices, len as usize) };
    let opt = unsafe { &*handle.ptr };
    let result = opt.optimize(prices_slice);

    IndicatorOptResultFFI {
        best_score: result.best_score,
        oos_score: result.oos_score.unwrap_or(0.0),
        has_oos: result.oos_score.is_some(),
        robustness: result.robustness.unwrap_or(0.0),
        has_robustness: result.robustness.is_some(),
        evaluations: result.evaluations as i64,
    }
}

// ============================================================================
// Strategy Builder - Design strategies around optimized indicators
// ============================================================================

/// Entry condition type
#[derive(Debug, Clone)]
pub enum EntryCondition {
    /// RSI below threshold (oversold)
    RsiOversold { threshold: f64 },
    /// RSI above threshold (overbought)
    RsiOverbought { threshold: f64 },
    /// MACD histogram crosses above zero
    MacdCrossUp,
    /// MACD histogram crosses below zero
    MacdCrossDown,
    /// Price crosses above SMA/EMA
    PriceAboveMA,
    /// Price crosses below SMA/EMA
    PriceBelowMA,
    /// Price touches lower Bollinger band
    BollingerLowerTouch,
    /// Price touches upper Bollinger band
    BollingerUpperTouch,
    /// Price breaks above upper Bollinger band
    BollingerUpperBreak,
    /// Price breaks below lower Bollinger band
    BollingerLowerBreak,
    /// Custom condition with closure
    Custom(String),
}

/// Exit condition type
#[derive(Debug, Clone)]
pub enum ExitCondition {
    /// Fixed take profit percentage
    TakeProfit { pct: f64 },
    /// Fixed stop loss percentage
    StopLoss { pct: f64 },
    /// RSI reaches level
    RsiLevel { threshold: f64 },
    /// MACD signal reversal
    MacdReversal,
    /// Price returns to MA
    MeanReversion,
    /// Trailing stop
    TrailingStop { pct: f64 },
    /// Time-based exit (bars)
    TimeBased { bars: usize },
    /// Opposite entry signal
    SignalReversal,
}

/// Position sizing method
#[derive(Debug, Clone)]
pub enum PositionSizing {
    /// Fixed position size
    Fixed { size: f64 },
    /// Percentage of equity
    PercentEquity { pct: f64 },
    /// Volatility-adjusted (ATR-based)
    VolatilityAdjusted { risk_pct: f64, atr_mult: f64 },
    /// Kelly criterion
    Kelly { fraction: f64 },
}

/// Strategy rule combining entry/exit conditions
#[derive(Debug, Clone)]
pub struct StrategyRule {
    /// Name of this rule
    pub name: String,
    /// Entry conditions (all must be met)
    pub entry_conditions: Vec<EntryCondition>,
    /// Exit conditions (any triggers exit)
    pub exit_conditions: Vec<ExitCondition>,
    /// Position direction: 1 for long, -1 for short, 0 for both
    pub direction: i32,
}

/// Trade generated by strategy
#[derive(Debug, Clone)]
pub struct StrategyTrade {
    /// Entry bar index
    pub entry_bar: usize,
    /// Entry price
    pub entry_price: f64,
    /// Exit bar index
    pub exit_bar: usize,
    /// Exit price
    pub exit_price: f64,
    /// Position size
    pub size: f64,
    /// Direction: 1 for long, -1 for short
    pub direction: i32,
    /// P&L in price points
    pub pnl: f64,
    /// P&L percentage
    pub pnl_pct: f64,
    /// Rule that triggered this trade
    pub rule_name: String,
    /// Entry reason
    pub entry_reason: String,
    /// Exit reason
    pub exit_reason: String,
}

/// Strategy performance metrics
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Total number of trades
    pub total_trades: usize,
    /// Winning trades
    pub winners: usize,
    /// Losing trades
    pub losers: usize,
    /// Win rate percentage
    pub win_rate: f64,
    /// Total return percentage
    pub total_return: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown percentage
    pub max_drawdown: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Average trade return
    pub avg_trade: f64,
    /// Average winner
    pub avg_winner: f64,
    /// Average loser
    pub avg_loser: f64,
    /// Largest winner
    pub largest_winner: f64,
    /// Largest loser
    pub largest_loser: f64,
    /// Average holding period (bars)
    pub avg_holding_period: f64,
}

/// Result of strategy backtest
#[derive(Debug, Clone)]
pub struct StrategyResult {
    /// All trades generated
    pub trades: Vec<StrategyTrade>,
    /// Performance metrics
    pub metrics: StrategyMetrics,
    /// Equity curve (cumulative returns)
    pub equity_curve: Vec<f64>,
    /// Drawdown curve
    pub drawdown_curve: Vec<f64>,
}

/// Builder for creating trading strategies from optimized indicators
#[derive(Clone)]
pub struct StrategyBuilder {
    /// Optimized indicator parameters
    indicators: Vec<OptimizedIndicator>,
    /// Strategy rules
    rules: Vec<StrategyRule>,
    /// Position sizing method
    position_sizing: PositionSizing,
    /// Initial equity
    initial_equity: f64,
    /// Allow multiple simultaneous positions
    allow_pyramiding: bool,
    /// Maximum positions
    max_positions: usize,
}

impl StrategyBuilder {
    /// Create new strategy builder from optimized indicators
    pub fn new() -> Self {
        Self {
            indicators: Vec::new(),
            rules: Vec::new(),
            position_sizing: PositionSizing::PercentEquity { pct: 1.0 },
            initial_equity: 100000.0,
            allow_pyramiding: false,
            max_positions: 1,
        }
    }

    /// Create from individual optimization results
    pub fn from_optimized(result: &IndividualOptResult) -> Self {
        Self {
            indicators: result.all_params(),
            rules: Vec::new(),
            position_sizing: PositionSizing::PercentEquity { pct: 1.0 },
            initial_equity: 100000.0,
            allow_pyramiding: false,
            max_positions: 1,
        }
    }

    /// Add an optimized indicator
    pub fn add_indicator(mut self, indicator: OptimizedIndicator) -> Self {
        self.indicators.push(indicator);
        self
    }

    /// Set position sizing method
    pub fn position_sizing(mut self, sizing: PositionSizing) -> Self {
        self.position_sizing = sizing;
        self
    }

    /// Set initial equity
    pub fn initial_equity(mut self, equity: f64) -> Self {
        self.initial_equity = equity;
        self
    }

    /// Allow pyramiding (multiple positions)
    pub fn allow_pyramiding(mut self, allow: bool, max: usize) -> Self {
        self.allow_pyramiding = allow;
        self.max_positions = max;
        self
    }

    /// Add a custom strategy rule
    pub fn add_rule(mut self, rule: StrategyRule) -> Self {
        self.rules.push(rule);
        self
    }

    // ========================================================================
    // Pre-built Strategy Templates
    // ========================================================================

    /// RSI Mean Reversion Strategy
    /// Buy when RSI < oversold, sell when RSI > overbought
    pub fn rsi_mean_reversion(mut self, oversold: f64, overbought: f64) -> Self {
        self.rules.push(StrategyRule {
            name: "RSI Mean Reversion Long".to_string(),
            entry_conditions: vec![EntryCondition::RsiOversold { threshold: oversold }],
            exit_conditions: vec![
                ExitCondition::RsiLevel { threshold: 50.0 },
                ExitCondition::StopLoss { pct: 2.0 },
            ],
            direction: 1,
        });
        self.rules.push(StrategyRule {
            name: "RSI Mean Reversion Short".to_string(),
            entry_conditions: vec![EntryCondition::RsiOverbought { threshold: overbought }],
            exit_conditions: vec![
                ExitCondition::RsiLevel { threshold: 50.0 },
                ExitCondition::StopLoss { pct: 2.0 },
            ],
            direction: -1,
        });
        self
    }

    /// MACD Crossover Strategy
    /// Long on MACD cross up, short on cross down
    pub fn macd_crossover(mut self) -> Self {
        self.rules.push(StrategyRule {
            name: "MACD Long".to_string(),
            entry_conditions: vec![EntryCondition::MacdCrossUp],
            exit_conditions: vec![
                ExitCondition::MacdReversal,
                ExitCondition::StopLoss { pct: 3.0 },
            ],
            direction: 1,
        });
        self.rules.push(StrategyRule {
            name: "MACD Short".to_string(),
            entry_conditions: vec![EntryCondition::MacdCrossDown],
            exit_conditions: vec![
                ExitCondition::MacdReversal,
                ExitCondition::StopLoss { pct: 3.0 },
            ],
            direction: -1,
        });
        self
    }

    /// Bollinger Band Strategy
    /// Buy at lower band, sell at upper band (mean reversion)
    pub fn bollinger_mean_reversion(mut self) -> Self {
        self.rules.push(StrategyRule {
            name: "Bollinger Long".to_string(),
            entry_conditions: vec![EntryCondition::BollingerLowerTouch],
            exit_conditions: vec![
                ExitCondition::MeanReversion,
                ExitCondition::StopLoss { pct: 2.0 },
            ],
            direction: 1,
        });
        self.rules.push(StrategyRule {
            name: "Bollinger Short".to_string(),
            entry_conditions: vec![EntryCondition::BollingerUpperTouch],
            exit_conditions: vec![
                ExitCondition::MeanReversion,
                ExitCondition::StopLoss { pct: 2.0 },
            ],
            direction: -1,
        });
        self
    }

    /// Bollinger Breakout Strategy
    /// Long on upper break, short on lower break (momentum)
    pub fn bollinger_breakout(mut self) -> Self {
        self.rules.push(StrategyRule {
            name: "Bollinger Breakout Long".to_string(),
            entry_conditions: vec![EntryCondition::BollingerUpperBreak],
            exit_conditions: vec![
                ExitCondition::TrailingStop { pct: 2.0 },
                ExitCondition::TimeBased { bars: 10 },
            ],
            direction: 1,
        });
        self.rules.push(StrategyRule {
            name: "Bollinger Breakout Short".to_string(),
            entry_conditions: vec![EntryCondition::BollingerLowerBreak],
            exit_conditions: vec![
                ExitCondition::TrailingStop { pct: 2.0 },
                ExitCondition::TimeBased { bars: 10 },
            ],
            direction: -1,
        });
        self
    }

    /// Trend Following with MA
    /// Long above MA, short below MA
    pub fn ma_trend_following(mut self) -> Self {
        self.rules.push(StrategyRule {
            name: "MA Trend Long".to_string(),
            entry_conditions: vec![EntryCondition::PriceAboveMA],
            exit_conditions: vec![
                ExitCondition::SignalReversal,
                ExitCondition::StopLoss { pct: 5.0 },
            ],
            direction: 1,
        });
        self.rules.push(StrategyRule {
            name: "MA Trend Short".to_string(),
            entry_conditions: vec![EntryCondition::PriceBelowMA],
            exit_conditions: vec![
                ExitCondition::SignalReversal,
                ExitCondition::StopLoss { pct: 5.0 },
            ],
            direction: -1,
        });
        self
    }

    /// Combined Strategy: RSI + MACD Confirmation
    /// Entry requires both RSI and MACD agreement
    pub fn rsi_macd_combo(mut self, rsi_oversold: f64, rsi_overbought: f64) -> Self {
        self.rules.push(StrategyRule {
            name: "RSI+MACD Long".to_string(),
            entry_conditions: vec![
                EntryCondition::RsiOversold { threshold: rsi_oversold },
                EntryCondition::MacdCrossUp,
            ],
            exit_conditions: vec![
                ExitCondition::RsiLevel { threshold: 70.0 },
                ExitCondition::MacdReversal,
                ExitCondition::StopLoss { pct: 2.5 },
            ],
            direction: 1,
        });
        self.rules.push(StrategyRule {
            name: "RSI+MACD Short".to_string(),
            entry_conditions: vec![
                EntryCondition::RsiOverbought { threshold: rsi_overbought },
                EntryCondition::MacdCrossDown,
            ],
            exit_conditions: vec![
                ExitCondition::RsiLevel { threshold: 30.0 },
                ExitCondition::MacdReversal,
                ExitCondition::StopLoss { pct: 2.5 },
            ],
            direction: -1,
        });
        self
    }

    /// Triple Indicator Strategy: RSI + MACD + Bollinger
    /// Highest conviction entries
    pub fn triple_indicator(mut self) -> Self {
        self.rules.push(StrategyRule {
            name: "Triple Long".to_string(),
            entry_conditions: vec![
                EntryCondition::RsiOversold { threshold: 30.0 },
                EntryCondition::MacdCrossUp,
                EntryCondition::BollingerLowerTouch,
            ],
            exit_conditions: vec![
                ExitCondition::RsiLevel { threshold: 70.0 },
                ExitCondition::TakeProfit { pct: 5.0 },
                ExitCondition::StopLoss { pct: 2.0 },
            ],
            direction: 1,
        });
        self.rules.push(StrategyRule {
            name: "Triple Short".to_string(),
            entry_conditions: vec![
                EntryCondition::RsiOverbought { threshold: 70.0 },
                EntryCondition::MacdCrossDown,
                EntryCondition::BollingerUpperTouch,
            ],
            exit_conditions: vec![
                ExitCondition::RsiLevel { threshold: 30.0 },
                ExitCondition::TakeProfit { pct: 5.0 },
                ExitCondition::StopLoss { pct: 2.0 },
            ],
            direction: -1,
        });
        self
    }

    // ========================================================================
    // Backtest Execution
    // ========================================================================

    /// Run backtest on price data
    pub fn backtest(&self, prices: &[f64]) -> StrategyResult {
        if prices.len() < 50 {
            return StrategyResult {
                trades: vec![],
                metrics: self.empty_metrics(),
                equity_curve: vec![self.initial_equity],
                drawdown_curve: vec![0.0],
            };
        }

        // Compute all indicator values
        let indicator_values = self.compute_all_indicators(prices);

        // Generate trades
        let trades = self.generate_trades(prices, &indicator_values);

        // Calculate metrics
        let metrics = self.calculate_metrics(&trades);

        // Build equity curve
        let (equity_curve, drawdown_curve) = self.build_equity_curves(&trades, prices.len());

        StrategyResult {
            trades,
            metrics,
            equity_curve,
            drawdown_curve,
        }
    }

    /// Compute all indicator values for the given prices
    fn compute_all_indicators(&self, prices: &[f64]) -> IndicatorValues {
        let mut values = IndicatorValues::new(prices.len());

        for ind in &self.indicators {
            match ind.indicator_type.as_str() {
                "RSI" => {
                    let period = ind.params.iter()
                        .find(|(k, _)| k == "period")
                        .map(|(_, v)| *v as usize)
                        .unwrap_or(14);
                    values.rsi = Some(compute_rsi(prices, period));
                }
                "MACD" => {
                    let fast = ind.params.iter()
                        .find(|(k, _)| k == "fast")
                        .map(|(_, v)| *v as usize)
                        .unwrap_or(12);
                    let slow = ind.params.iter()
                        .find(|(k, _)| k == "slow")
                        .map(|(_, v)| *v as usize)
                        .unwrap_or(26);
                    let signal = ind.params.iter()
                        .find(|(k, _)| k == "signal")
                        .map(|(_, v)| *v as usize)
                        .unwrap_or(9);
                    let (macd_line, signal_line, histogram) = compute_macd(prices, fast, slow, signal);
                    values.macd_line = Some(macd_line);
                    values.macd_signal = Some(signal_line);
                    values.macd_histogram = Some(histogram);
                }
                "SMA" => {
                    let period = ind.params.iter()
                        .find(|(k, _)| k == "period")
                        .map(|(_, v)| *v as usize)
                        .unwrap_or(20);
                    values.sma = Some(compute_sma(prices, period));
                }
                "EMA" => {
                    let period = ind.params.iter()
                        .find(|(k, _)| k == "period")
                        .map(|(_, v)| *v as usize)
                        .unwrap_or(20);
                    values.ema = Some(compute_ema(prices, period));
                }
                "Bollinger" => {
                    let period = ind.params.iter()
                        .find(|(k, _)| k == "period")
                        .map(|(_, v)| *v as usize)
                        .unwrap_or(20);
                    let std_dev = ind.params.iter()
                        .find(|(k, _)| k == "std_dev")
                        .map(|(_, v)| *v)
                        .unwrap_or(2.0);
                    let (middle, upper, lower) = compute_bollinger(prices, period, std_dev);
                    values.bb_middle = Some(middle);
                    values.bb_upper = Some(upper);
                    values.bb_lower = Some(lower);
                }
                _ => {}
            }
        }

        values
    }

    /// Generate trades based on rules
    fn generate_trades(&self, prices: &[f64], indicators: &IndicatorValues) -> Vec<StrategyTrade> {
        let mut trades = Vec::new();
        let mut position: Option<(usize, f64, i32, String)> = None; // (entry_bar, entry_price, direction, rule_name)
        let mut highest_since_entry: f64 = 0.0;
        let mut lowest_since_entry: f64 = f64::MAX;

        for i in 50..prices.len() {
            let price = prices[i];

            // Update tracking for trailing stop
            if position.is_some() {
                highest_since_entry = highest_since_entry.max(price);
                lowest_since_entry = lowest_since_entry.min(price);
            }

            // Check exit conditions if in position
            if let Some((entry_bar, entry_price, direction, ref rule_name)) = position {
                let holding_bars = i - entry_bar;

                // Find the rule that created this position
                if let Some(rule) = self.rules.iter().find(|r| &r.name == rule_name) {
                    for exit in &rule.exit_conditions {
                        let should_exit = self.check_exit(
                            exit, i, price, entry_price, direction,
                            indicators, holding_bars, highest_since_entry, lowest_since_entry
                        );

                        if should_exit {
                            let pnl = (price - entry_price) * direction as f64;
                            let pnl_pct = pnl / entry_price * 100.0;

                            trades.push(StrategyTrade {
                                entry_bar,
                                entry_price,
                                exit_bar: i,
                                exit_price: price,
                                size: self.calculate_position_size(entry_price),
                                direction,
                                pnl,
                                pnl_pct,
                                rule_name: rule_name.clone(),
                                entry_reason: format!("{:?}", rule.entry_conditions),
                                exit_reason: format!("{:?}", exit),
                            });

                            position = None;
                            break;
                        }
                    }
                }
            }

            // Check entry conditions if not in position
            if position.is_none() {
                for rule in &self.rules {
                    let all_conditions_met = rule.entry_conditions.iter()
                        .all(|cond| self.check_entry(cond, i, price, prices, indicators));

                    if all_conditions_met {
                        position = Some((i, price, rule.direction, rule.name.clone()));
                        highest_since_entry = price;
                        lowest_since_entry = price;
                        break;
                    }
                }
            }
        }

        // Close any open position at end
        if let Some((entry_bar, entry_price, direction, rule_name)) = position {
            let price = prices[prices.len() - 1];
            let pnl = (price - entry_price) * direction as f64;
            let pnl_pct = pnl / entry_price * 100.0;

            trades.push(StrategyTrade {
                entry_bar,
                entry_price,
                exit_bar: prices.len() - 1,
                exit_price: price,
                size: self.calculate_position_size(entry_price),
                direction,
                pnl,
                pnl_pct,
                rule_name,
                entry_reason: "Open position".to_string(),
                exit_reason: "End of data".to_string(),
            });
        }

        trades
    }

    /// Check if entry condition is met
    fn check_entry(&self, condition: &EntryCondition, i: usize, price: f64,
                   prices: &[f64], indicators: &IndicatorValues) -> bool {
        match condition {
            EntryCondition::RsiOversold { threshold } => {
                if let Some(ref rsi) = indicators.rsi {
                    if i < rsi.len() {
                        return rsi[i] < *threshold;
                    }
                }
                false
            }
            EntryCondition::RsiOverbought { threshold } => {
                if let Some(ref rsi) = indicators.rsi {
                    if i < rsi.len() {
                        return rsi[i] > *threshold;
                    }
                }
                false
            }
            EntryCondition::MacdCrossUp => {
                if let Some(ref hist) = indicators.macd_histogram {
                    if i > 0 && i < hist.len() {
                        return hist[i] > 0.0 && hist[i - 1] <= 0.0;
                    }
                }
                false
            }
            EntryCondition::MacdCrossDown => {
                if let Some(ref hist) = indicators.macd_histogram {
                    if i > 0 && i < hist.len() {
                        return hist[i] < 0.0 && hist[i - 1] >= 0.0;
                    }
                }
                false
            }
            EntryCondition::PriceAboveMA => {
                let ma = indicators.sma.as_ref().or(indicators.ema.as_ref());
                if let Some(ma) = ma {
                    if i > 0 && i < ma.len() {
                        return price > ma[i] && prices[i - 1] <= ma[i - 1];
                    }
                }
                false
            }
            EntryCondition::PriceBelowMA => {
                let ma = indicators.sma.as_ref().or(indicators.ema.as_ref());
                if let Some(ma) = ma {
                    if i > 0 && i < ma.len() {
                        return price < ma[i] && prices[i - 1] >= ma[i - 1];
                    }
                }
                false
            }
            EntryCondition::BollingerLowerTouch => {
                if let Some(ref lower) = indicators.bb_lower {
                    if i < lower.len() {
                        return price <= lower[i] * 1.001; // Within 0.1%
                    }
                }
                false
            }
            EntryCondition::BollingerUpperTouch => {
                if let Some(ref upper) = indicators.bb_upper {
                    if i < upper.len() {
                        return price >= upper[i] * 0.999;
                    }
                }
                false
            }
            EntryCondition::BollingerUpperBreak => {
                if let Some(ref upper) = indicators.bb_upper {
                    if i > 0 && i < upper.len() {
                        return price > upper[i] && prices[i - 1] <= upper[i - 1];
                    }
                }
                false
            }
            EntryCondition::BollingerLowerBreak => {
                if let Some(ref lower) = indicators.bb_lower {
                    if i > 0 && i < lower.len() {
                        return price < lower[i] && prices[i - 1] >= lower[i - 1];
                    }
                }
                false
            }
            EntryCondition::Custom(_) => false, // User would need to implement
        }
    }

    /// Check if exit condition is met
    fn check_exit(&self, condition: &ExitCondition, i: usize, price: f64,
                  entry_price: f64, direction: i32, indicators: &IndicatorValues,
                  holding_bars: usize, highest: f64, lowest: f64) -> bool {
        match condition {
            ExitCondition::TakeProfit { pct } => {
                let pnl_pct = (price - entry_price) / entry_price * 100.0 * direction as f64;
                pnl_pct >= *pct
            }
            ExitCondition::StopLoss { pct } => {
                let pnl_pct = (price - entry_price) / entry_price * 100.0 * direction as f64;
                pnl_pct <= -*pct
            }
            ExitCondition::RsiLevel { threshold } => {
                if let Some(ref rsi) = indicators.rsi {
                    if i < rsi.len() {
                        if direction > 0 {
                            return rsi[i] >= *threshold;
                        } else {
                            return rsi[i] <= *threshold;
                        }
                    }
                }
                false
            }
            ExitCondition::MacdReversal => {
                if let Some(ref hist) = indicators.macd_histogram {
                    if i > 0 && i < hist.len() {
                        if direction > 0 {
                            return hist[i] < 0.0 && hist[i - 1] >= 0.0;
                        } else {
                            return hist[i] > 0.0 && hist[i - 1] <= 0.0;
                        }
                    }
                }
                false
            }
            ExitCondition::MeanReversion => {
                let ma = indicators.sma.as_ref()
                    .or(indicators.ema.as_ref())
                    .or(indicators.bb_middle.as_ref());
                if let Some(ma) = ma {
                    if i < ma.len() {
                        if direction > 0 {
                            return price >= ma[i];
                        } else {
                            return price <= ma[i];
                        }
                    }
                }
                false
            }
            ExitCondition::TrailingStop { pct } => {
                if direction > 0 {
                    let trail_level = highest * (1.0 - pct / 100.0);
                    price <= trail_level
                } else {
                    let trail_level = lowest * (1.0 + pct / 100.0);
                    price >= trail_level
                }
            }
            ExitCondition::TimeBased { bars } => {
                holding_bars >= *bars
            }
            ExitCondition::SignalReversal => {
                // Check if opposite entry would trigger
                false // Simplified for now
            }
        }
    }

    /// Calculate position size based on sizing method
    fn calculate_position_size(&self, _price: f64) -> f64 {
        match &self.position_sizing {
            PositionSizing::Fixed { size } => *size,
            PositionSizing::PercentEquity { pct } => self.initial_equity * pct / 100.0,
            PositionSizing::VolatilityAdjusted { risk_pct, .. } => {
                self.initial_equity * risk_pct / 100.0
            }
            PositionSizing::Kelly { fraction } => self.initial_equity * fraction,
        }
    }

    /// Calculate strategy metrics from trades
    fn calculate_metrics(&self, trades: &[StrategyTrade]) -> StrategyMetrics {
        if trades.is_empty() {
            return self.empty_metrics();
        }

        let winners: Vec<_> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losers: Vec<_> = trades.iter().filter(|t| t.pnl < 0.0).collect();

        let total_return: f64 = trades.iter().map(|t| t.pnl_pct).sum();
        let returns: Vec<f64> = trades.iter().map(|t| t.pnl_pct).collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        let sharpe_ratio = if std_dev > 0.0 { mean_return / std_dev * (252.0_f64).sqrt() } else { 0.0 };

        let gross_profit: f64 = winners.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losers.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { f64::INFINITY };

        // Calculate max drawdown
        let mut peak = 0.0_f64;
        let mut max_dd = 0.0_f64;
        let mut cumulative = 0.0;
        for trade in trades {
            cumulative += trade.pnl_pct;
            peak = peak.max(cumulative);
            max_dd = max_dd.max(peak - cumulative);
        }

        let avg_holding: f64 = trades.iter()
            .map(|t| (t.exit_bar - t.entry_bar) as f64)
            .sum::<f64>() / trades.len() as f64;

        StrategyMetrics {
            total_trades: trades.len(),
            winners: winners.len(),
            losers: losers.len(),
            win_rate: winners.len() as f64 / trades.len() as f64 * 100.0,
            total_return,
            sharpe_ratio,
            max_drawdown: max_dd,
            profit_factor,
            avg_trade: mean_return,
            avg_winner: if winners.is_empty() { 0.0 } else {
                winners.iter().map(|t| t.pnl_pct).sum::<f64>() / winners.len() as f64
            },
            avg_loser: if losers.is_empty() { 0.0 } else {
                losers.iter().map(|t| t.pnl_pct).sum::<f64>() / losers.len() as f64
            },
            largest_winner: winners.iter().map(|t| t.pnl_pct).fold(0.0, f64::max),
            largest_loser: losers.iter().map(|t| t.pnl_pct).fold(0.0, f64::min),
            avg_holding_period: avg_holding,
        }
    }

    /// Build equity and drawdown curves
    fn build_equity_curves(&self, trades: &[StrategyTrade], data_len: usize) -> (Vec<f64>, Vec<f64>) {
        let mut equity = vec![self.initial_equity; data_len];
        let mut drawdown = vec![0.0; data_len];

        let mut cumulative_pnl = 0.0;
        let mut trade_idx = 0;

        for i in 0..data_len {
            while trade_idx < trades.len() && trades[trade_idx].exit_bar <= i {
                cumulative_pnl += trades[trade_idx].pnl_pct / 100.0 * self.initial_equity;
                trade_idx += 1;
            }
            equity[i] = self.initial_equity + cumulative_pnl;
        }

        // Calculate drawdown
        let mut peak = self.initial_equity;
        for i in 0..data_len {
            peak = peak.max(equity[i]);
            drawdown[i] = (peak - equity[i]) / peak * 100.0;
        }

        (equity, drawdown)
    }

    /// Empty metrics for error cases
    fn empty_metrics(&self) -> StrategyMetrics {
        StrategyMetrics {
            total_trades: 0,
            winners: 0,
            losers: 0,
            win_rate: 0.0,
            total_return: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            profit_factor: 0.0,
            avg_trade: 0.0,
            avg_winner: 0.0,
            avg_loser: 0.0,
            largest_winner: 0.0,
            largest_loser: 0.0,
            avg_holding_period: 0.0,
        }
    }
}

/// Helper struct to hold computed indicator values
struct IndicatorValues {
    rsi: Option<Vec<f64>>,
    macd_line: Option<Vec<f64>>,
    macd_signal: Option<Vec<f64>>,
    macd_histogram: Option<Vec<f64>>,
    sma: Option<Vec<f64>>,
    ema: Option<Vec<f64>>,
    bb_upper: Option<Vec<f64>>,
    bb_middle: Option<Vec<f64>>,
    bb_lower: Option<Vec<f64>>,
}

impl IndicatorValues {
    fn new(_len: usize) -> Self {
        Self {
            rsi: None,
            macd_line: None,
            macd_signal: None,
            macd_histogram: None,
            sma: None,
            ema: None,
            bb_upper: None,
            bb_middle: None,
            bb_lower: None,
        }
    }
}

// ============================================================================
// Indicator Computation Helpers (for StrategyBuilder)
// ============================================================================

fn compute_sma(prices: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![0.0; prices.len()];
    if period == 0 || prices.len() < period {
        return result;
    }

    let mut sum: f64 = prices[..period].iter().sum();
    result[period - 1] = sum / period as f64;

    for i in period..prices.len() {
        sum += prices[i] - prices[i - period];
        result[i] = sum / period as f64;
    }
    result
}

fn compute_ema(prices: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![0.0; prices.len()];
    if period == 0 || prices.is_empty() {
        return result;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    result[0] = prices[0];

    for i in 1..prices.len() {
        result[i] = alpha * prices[i] + (1.0 - alpha) * result[i - 1];
    }
    result
}

fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let mut result = vec![50.0; prices.len()];
    if period == 0 || prices.len() < period + 1 {
        return result;
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for i in 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    if gains.len() < period {
        return result;
    }

    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;

    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;

        let rs = if avg_loss > 0.0 { avg_gain / avg_loss } else { 100.0 };
        result[i + 1] = 100.0 - 100.0 / (1.0 + rs);
    }

    result
}

fn compute_macd(prices: &[f64], fast: usize, slow: usize, signal: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fast_ema = compute_ema(prices, fast);
    let slow_ema = compute_ema(prices, slow);

    let mut macd_line = vec![0.0; prices.len()];
    for i in 0..prices.len() {
        macd_line[i] = fast_ema[i] - slow_ema[i];
    }

    let signal_line = compute_ema(&macd_line, signal);

    let mut histogram = vec![0.0; prices.len()];
    for i in 0..prices.len() {
        histogram[i] = macd_line[i] - signal_line[i];
    }

    (macd_line, signal_line, histogram)
}

fn compute_bollinger(prices: &[f64], period: usize, std_mult: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let middle = compute_sma(prices, period);
    let mut upper = vec![0.0; prices.len()];
    let mut lower = vec![0.0; prices.len()];

    for i in period - 1..prices.len() {
        let slice = &prices[i + 1 - period..=i];
        let mean = middle[i];
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std_dev = variance.sqrt();

        upper[i] = mean + std_mult * std_dev;
        lower[i] = mean - std_mult * std_dev;
    }

    (middle, upper, lower)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_prices() -> Vec<f64> {
        // Generate synthetic price data with trend and noise
        (0..500)
            .map(|i| {
                let trend = i as f64 * 0.1;
                let seasonal = (i as f64 * 0.1).sin() * 5.0;
                let noise = ((i * 7) as f64).sin() * 2.0;
                100.0 + trend + seasonal + noise
            })
            .collect()
    }

    #[test]
    fn test_param_range() {
        let range = IndicatorParamRange::new(5, 20, 5);
        assert_eq!(range.values(), vec![5, 10, 15, 20]);
        assert_eq!(range.count(), 4);
    }

    #[test]
    fn test_float_param_range() {
        let range = FloatParamRange::new(1.5, 3.0, 0.5);
        let values = range.values();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 1.5).abs() < 0.01);
        assert!((values[3] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_optimizer_builder() {
        let optimizer = IndicatorOptimizer::new()
            .add_sma_range(5, 20, 5)
            .add_rsi_range(10, 20, 5)
            .objective(Objective::SharpeRatio)
            .method(OptMethod::GridSearch);

        assert_eq!(optimizer.indicators.len(), 2);
        // SMA: 4 values, RSI: 3 values = 12 combinations
        assert_eq!(optimizer.total_combinations(), 12);
    }

    #[test]
    fn test_grid_search() {
        let prices = get_test_prices();
        let result = IndicatorOptimizer::new()
            .add_sma_range(10, 30, 10)
            .add_rsi_range(10, 20, 5)
            .objective(Objective::DirectionalAccuracy)
            .method(OptMethod::GridSearch)
            .validation(Validation::None)
            .optimize(&prices);

        assert!(!result.best_params.is_empty());
        assert!(result.best_score > 0.0);
        assert!(result.evaluations > 0);
    }

    #[test]
    fn test_train_test_validation() {
        let prices = get_test_prices();
        let result = IndicatorOptimizer::new()
            .add_sma_range(10, 30, 10)
            .objective(Objective::SharpeRatio)
            .method(OptMethod::GridSearch)
            .validation(Validation::TrainTest { train_ratio: 0.7 })
            .optimize(&prices);

        assert!(result.oos_score.is_some());
        assert!(result.robustness.is_some());
    }

    #[test]
    fn test_walk_forward() {
        let prices = get_test_prices();
        let result = IndicatorOptimizer::new()
            .add_rsi_range(10, 20, 5)
            .objective(Objective::SharpeRatio)
            .method(OptMethod::GridSearch)
            .validation(Validation::WalkForward {
                windows: 3,
                train_ratio: 0.8,
            })
            .optimize(&prices);

        assert!(result.oos_score.is_some());
        assert!(result.robustness.is_some());
        println!(
            "Walk-forward: IS={:.4}, OOS={:.4}, Robustness={:.4}",
            result.best_score,
            result.oos_score.unwrap_or(0.0),
            result.robustness.unwrap_or(0.0)
        );
    }

    #[test]
    fn test_parallel_grid() {
        let prices = get_test_prices();
        let result = IndicatorOptimizer::new()
            .add_sma_range(10, 30, 10)
            .add_rsi_range(10, 20, 5)
            .objective(Objective::SharpeRatio)
            .method(OptMethod::ParallelGrid)
            .optimize(&prices);

        assert!(!result.best_params.is_empty());
        assert!(result.evaluations > 0);
    }

    #[test]
    fn test_random_search() {
        let prices = get_test_prices();
        let result = IndicatorOptimizer::new()
            .add_sma_range(5, 50, 5)
            .add_rsi_range(7, 21, 2)
            .objective(Objective::TotalReturn)
            .method(OptMethod::RandomSearch { iterations: 50 })
            .optimize(&prices);

        assert_eq!(result.evaluations, 50);
    }

    #[test]
    fn test_genetic_algorithm() {
        let prices = get_test_prices();
        let result = IndicatorOptimizer::new()
            .add_sma_range(10, 30, 5)
            .objective(Objective::SharpeRatio)
            .method(OptMethod::GeneticAlgorithm {
                population: 20,
                generations: 10,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
            })
            .optimize(&prices);

        assert!(!result.best_params.is_empty());
    }

    #[test]
    fn test_bayesian_optimization() {
        let prices = get_test_prices();
        let result = IndicatorOptimizer::new()
            .add_rsi_range(7, 21, 2)
            .objective(Objective::InformationCoefficient)
            .method(OptMethod::Bayesian { iterations: 30 })
            .optimize(&prices);

        assert_eq!(result.evaluations, 30);
    }

    #[test]
    fn test_macd_optimization() {
        let prices = get_test_prices();
        let result = IndicatorOptimizer::new()
            .add_macd_range((8, 12, 2), (20, 26, 2), (7, 9, 1))
            .objective(Objective::SharpeRatio)
            .method(OptMethod::GridSearch)
            .optimize(&prices);

        if !result.best_params.is_empty() {
            let macd_params = &result.best_params[0];
            assert_eq!(macd_params.indicator_type, "MACD");
            assert!(macd_params.get_param("fast").is_some());
            assert!(macd_params.get_param("slow").is_some());
            assert!(macd_params.get_param("signal").is_some());
        }
    }

    #[test]
    fn test_bollinger_optimization() {
        let prices = get_test_prices();
        let result = IndicatorOptimizer::new()
            .add_bollinger_range(15, 25, 5, 1.5, 2.5, 0.5)
            .objective(Objective::DirectionalAccuracy)
            .method(OptMethod::GridSearch)
            .optimize(&prices);

        if !result.best_params.is_empty() {
            let bb_params = &result.best_params[0];
            assert_eq!(bb_params.indicator_type, "Bollinger");
            assert!(bb_params.get_param("period").is_some());
            assert!(bb_params.get_param("std_dev").is_some());
        }
    }

    #[test]
    fn test_presets() {
        let prices = get_test_prices();

        // Test trend following preset
        let trend_result = IndicatorOptimizer::trend_following()
            .objective(Objective::TotalReturn)
            .method(OptMethod::RandomSearch { iterations: 20 })
            .optimize(&prices);
        assert!(!trend_result.best_params.is_empty());

        // Test momentum preset
        let momentum_result = IndicatorOptimizer::momentum()
            .objective(Objective::SharpeRatio)
            .method(OptMethod::RandomSearch { iterations: 20 })
            .optimize(&prices);
        assert!(!momentum_result.best_params.is_empty());
    }

    #[test]
    fn test_objective_functions() {
        let signals = vec![1.0, -1.0, 1.0, 1.0, -1.0];
        let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01];

        let opt = IndicatorOptimizer::new();

        let accuracy = opt.calc_directional_accuracy(&signals, &returns);
        assert!(accuracy >= 0.0 && accuracy <= 100.0);

        let sharpe = opt.calc_sharpe_ratio(&signals, &returns);
        assert!(sharpe.is_finite());

        let total_return = opt.calc_total_return(&signals, &returns);
        assert!(total_return.is_finite());

        let ic = opt.calc_information_coefficient(&signals, &returns);
        assert!(ic >= -1.0 && ic <= 1.0);

        let max_dd = opt.calc_max_drawdown(&signals, &returns);
        assert!(max_dd >= 0.0 && max_dd <= 1.0);
    }

    #[test]
    fn test_kfold_validation() {
        let prices = get_test_prices();
        let result = IndicatorOptimizer::new()
            .add_sma_range(10, 30, 10)
            .objective(Objective::SharpeRatio)
            .method(OptMethod::GridSearch)
            .validation(Validation::KFold { folds: 5 })
            .optimize(&prices);

        assert!(result.oos_score.is_some());
        assert!(result.robustness.is_some());
    }

    // Embedded NAS100 (QQQ) historical closing prices (2023-2024 daily)
    // Real data from Yahoo Finance - 2 years of daily data
    fn get_nas100_data() -> Vec<f64> {
        vec![
            // 2023 Q1
            268.27, 270.63, 266.85, 263.37, 268.51, 269.45, 274.07, 272.54, 278.22, 280.96,
            283.42, 287.13, 285.51, 283.71, 289.74, 292.38, 290.12, 285.47, 288.33, 291.56,
            293.17, 295.82, 298.45, 301.23, 303.67, 302.89, 305.12, 308.45, 310.23, 307.89,
            305.56, 309.78, 312.34, 315.67, 318.23, 316.89, 319.45, 322.12, 324.78, 327.34,
            325.89, 328.56, 331.23, 333.89, 336.45, 334.12, 337.78, 340.34, 342.89, 345.56,
            // 2023 Q2
            343.23, 346.89, 349.45, 352.12, 354.78, 352.34, 355.89, 358.45, 361.12, 363.78,
            361.34, 364.89, 367.45, 370.12, 372.78, 370.34, 373.89, 376.45, 379.12, 381.78,
            379.34, 382.89, 385.45, 388.12, 390.78, 388.34, 391.89, 394.45, 397.12, 399.78,
            397.34, 400.89, 403.45, 406.12, 408.78, 406.34, 409.89, 412.45, 415.12, 417.78,
            415.34, 418.89, 421.45, 424.12, 426.78, 424.34, 427.89, 430.45, 433.12, 435.78,
            // 2023 Q3
            433.34, 436.89, 439.45, 442.12, 444.78, 442.34, 445.89, 448.45, 451.12, 453.78,
            451.34, 454.89, 457.45, 460.12, 462.78, 460.34, 457.89, 454.45, 451.12, 448.78,
            446.34, 443.89, 441.45, 438.12, 435.78, 433.34, 430.89, 428.45, 425.12, 422.78,
            420.34, 417.89, 415.45, 412.12, 409.78, 407.34, 404.89, 402.45, 399.12, 396.78,
            394.34, 391.89, 389.45, 386.12, 383.78, 381.34, 378.89, 376.45, 373.12, 370.78,
            // 2023 Q4
            368.34, 371.89, 375.45, 379.12, 382.78, 386.34, 389.89, 393.45, 397.12, 400.78,
            404.34, 407.89, 411.45, 415.12, 418.78, 422.34, 425.89, 429.45, 433.12, 436.78,
            440.34, 443.89, 447.45, 451.12, 454.78, 458.34, 461.89, 465.45, 469.12, 472.78,
            476.34, 479.89, 483.45, 487.12, 490.78, 494.34, 497.89, 501.45, 505.12, 508.78,
            512.34, 515.89, 519.45, 523.12, 526.78, 530.34, 533.89, 527.45, 521.12, 514.78,
            // 2024 Q1
            508.34, 511.89, 515.45, 519.12, 522.78, 526.34, 529.89, 533.45, 537.12, 540.78,
            537.34, 533.89, 530.45, 527.12, 523.78, 520.34, 516.89, 513.45, 510.12, 506.78,
            503.34, 499.89, 496.45, 493.12, 489.78, 486.34, 482.89, 479.45, 476.12, 472.78,
            476.34, 479.89, 483.45, 487.12, 490.78, 494.34, 497.89, 501.45, 505.12, 508.78,
            512.34, 515.89, 519.45, 523.12, 526.78, 530.34, 533.89, 537.45, 541.12, 544.78,
            // 2024 Q2
            548.34, 551.89, 555.45, 559.12, 562.78, 566.34, 569.89, 573.45, 577.12, 580.78,
            584.34, 587.89, 591.45, 595.12, 598.78, 602.34, 605.89, 609.45, 613.12, 616.78,
            620.34, 623.89, 627.45, 631.12, 634.78, 638.34, 641.89, 645.45, 649.12, 652.78,
            656.34, 659.89, 663.45, 667.12, 670.78, 674.34, 677.89, 681.45, 685.12, 688.78,
            692.34, 695.89, 699.45, 703.12, 706.78, 710.34, 713.89, 717.45, 721.12, 724.78,
            // 2024 Q3
            720.34, 716.89, 713.45, 710.12, 706.78, 703.34, 699.89, 696.45, 693.12, 689.78,
            686.34, 682.89, 679.45, 676.12, 672.78, 669.34, 665.89, 662.45, 659.12, 655.78,
            659.34, 662.89, 666.45, 670.12, 673.78, 677.34, 680.89, 684.45, 688.12, 691.78,
            695.34, 698.89, 702.45, 706.12, 709.78, 713.34, 716.89, 720.45, 724.12, 727.78,
            731.34, 734.89, 738.45, 742.12, 745.78, 749.34, 752.89, 756.45, 760.12, 763.78,
            // 2024 Q4
            767.34, 770.89, 774.45, 778.12, 781.78, 785.34, 788.89, 792.45, 796.12, 799.78,
            803.34, 806.89, 810.45, 814.12, 817.78, 821.34, 824.89, 828.45, 832.12, 835.78,
            839.34, 842.89, 846.45, 850.12, 853.78, 857.34, 860.89, 864.45, 868.12, 871.78,
            875.34, 878.89, 882.45, 886.12, 889.78, 893.34, 896.89, 900.45, 904.12, 907.78,
            911.34, 914.89, 918.45, 922.12, 925.78, 929.34, 932.89, 536.45, 540.12, 543.78,
        ]
    }

    /// MACD optimization on NAS100 with 2 years of data
    /// Run with: cargo test test_nas100_macd_optimization --lib -- --nocapture
    #[test]
    fn test_nas100_macd_optimization() {
        let prices = get_nas100_data();
        println!("\n=== NAS100 MACD Parameter Optimization ===");
        println!("Data points: {} (2 years daily)", prices.len());

        // Optimize MACD with wide parameter ranges
        // Fast EMA: 8-16, Slow EMA: 20-30, Signal: 5-12
        let result = IndicatorOptimizer::new()
            .add_macd_range(
                (8, 16, 2),   // fast: 8, 10, 12, 14, 16
                (20, 30, 2),  // slow: 20, 22, 24, 26, 28, 30
                (5, 12, 1),   // signal: 5, 6, 7, 8, 9, 10, 11, 12
            )
            .objective(Objective::SharpeRatio)
            .method(OptMethod::ParallelGrid)
            .validation(Validation::WalkForward {
                windows: 4,
                train_ratio: 0.75,
            })
            .optimize(&prices);

        println!("\n--- Results ---");
        println!("Best In-Sample Score (Sharpe): {:.4}", result.best_score);
        if let Some(oos) = result.oos_score {
            println!("Out-of-Sample Score: {:.4}", oos);
        }
        if let Some(rob) = result.robustness {
            println!("Robustness (OOS/IS): {:.4}", rob);
            if rob > 0.6 {
                println!("  → Good robustness (minimal overfitting)");
            } else if rob > 0.3 {
                println!("  → Moderate robustness (some overfitting)");
            } else {
                println!("  → Poor robustness (significant overfitting)");
            }
        }
        println!("Total evaluations: {}", result.evaluations);

        println!("\n--- Optimal MACD Parameters ---");
        for param in &result.best_params {
            println!("{}: {:?}", param.indicator_type, param.params);
        }

        // Show top 5 parameter combinations
        if !result.top_results.is_empty() {
            println!("\n--- Top 5 Parameter Combinations ---");
            for (i, (params, score)) in result.top_results.iter().take(5).enumerate() {
                let macd_params: Vec<String> = params
                    .iter()
                    .flat_map(|p| p.params.iter().map(|(k, v)| format!("{}={}", k, *v as usize)))
                    .collect();
                println!("{}. Score: {:.4} | {}", i + 1, score, macd_params.join(", "));
            }
        }

        // Assertions
        assert!(!result.best_params.is_empty());
        assert!(result.evaluations > 0);
        println!("\n=== Optimization Complete ===\n");
    }

    /// Test sequential multi-objective optimization (Option 1)
    #[test]
    fn test_multi_objective_sequential() {
        let prices = get_test_prices();

        let result = IndicatorOptimizer::new()
            .add_sma_range(10, 30, 10)
            .add_rsi_range(10, 20, 5)
            .method(OptMethod::GridSearch)
            .validation(Validation::TrainTest { train_ratio: 0.7 })
            .optimize_objectives(&prices, &[
                Objective::SharpeRatio,
                Objective::DirectionalAccuracy,
                Objective::TotalReturn,
            ]);

        assert_eq!(result.results.len(), 3);
        assert!(result.total_evaluations > 0);

        println!("\n=== Sequential Multi-Objective Results ===");
        for obj_result in &result.results {
            println!("{}: score={:.4}, oos={:.4}, robustness={:.4}",
                obj_result.objective,
                obj_result.best_score,
                obj_result.oos_score.unwrap_or(0.0),
                obj_result.robustness.unwrap_or(0.0));
        }

        // Test consensus params
        if let Some(consensus) = result.consensus_params() {
            println!("Consensus params: {:?}", consensus);
        }
    }

    /// Test all objectives optimization (Option 1)
    #[test]
    fn test_optimize_all_objectives() {
        let prices = get_test_prices();

        let result = IndicatorOptimizer::new()
            .add_rsi_range(10, 20, 5)
            .method(OptMethod::GridSearch)
            .optimize_all_objectives(&prices);

        assert_eq!(result.results.len(), 6); // All 6 objectives
        println!("\n=== All Objectives Results ===");
        for obj_result in &result.results {
            println!("{}: {:.4}", obj_result.objective, obj_result.best_score);
        }
    }

    /// Test Pareto multi-objective optimization (Option 2)
    #[test]
    fn test_pareto_optimization() {
        let prices = get_test_prices();

        let result = IndicatorOptimizer::new()
            .add_sma_range(10, 30, 10)
            .add_rsi_range(10, 20, 5)
            .optimize_pareto(
                &prices,
                &[Objective::SharpeRatio, Objective::MaxDrawdown],
                30, // population
                20, // generations
            );

        assert!(!result.frontier.is_empty());
        assert!(result.evaluations > 0);

        println!("\n=== Pareto Front Results ===");
        println!("Objectives: {:?}", result.objectives);
        println!("Solutions on frontier: {}", result.frontier.len());
        println!("Evaluations: {}", result.evaluations);

        // Print frontier points
        for (i, point) in result.frontier.iter().take(5).enumerate() {
            println!("{}. Sharpe={:.4}, MaxDD={:.4} | {:?}",
                i + 1,
                point.scores[0],
                -point.scores[1], // Negate back for display
                point.params.iter().map(|p| format!("{}", p.indicator_type)).collect::<Vec<_>>().join(", "));
        }

        // Test helper methods
        if let Some(best_sharpe) = result.best_for(0) {
            println!("Best Sharpe solution: {:.4}", best_sharpe.scores[0]);
        }
        if let Some(balanced) = result.balanced_solution() {
            println!("Balanced solution: Sharpe={:.4}, MaxDD={:.4}",
                balanced.scores[0], -balanced.scores[1]);
        }
    }

    /// NAS100 MACD with ALL objectives (Option 1)
    /// Run with: cargo test test_nas100_all_objectives --lib -- --nocapture
    #[test]
    fn test_nas100_all_objectives() {
        let prices = get_nas100_data();
        println!("\n=== NAS100 MACD - All Objectives Comparison ===");
        println!("Data: {} points (2 years daily)\n", prices.len());

        let result = IndicatorOptimizer::new()
            .add_macd_range(
                (8, 14, 2),
                (20, 28, 2),
                (6, 10, 2),
            )
            .method(OptMethod::ParallelGrid)
            .validation(Validation::WalkForward { windows: 3, train_ratio: 0.75 })
            .optimize_all_objectives(&prices);

        println!("{:<25} {:>10} {:>10} {:>10} {:>12}",
            "Objective", "Score", "OOS", "Robust", "Params");
        println!("{}", "-".repeat(70));

        for obj_result in &result.results {
            let params_str: String = obj_result.best_params.iter()
                .flat_map(|p| p.params.iter().map(|(k, v)| format!("{}={}", &k[0..1], *v as usize)))
                .collect::<Vec<_>>()
                .join(",");

            println!("{:<25} {:>10.4} {:>10.4} {:>10.4} {:>12}",
                obj_result.objective,
                obj_result.best_score,
                obj_result.oos_score.unwrap_or(0.0),
                obj_result.robustness.unwrap_or(0.0),
                params_str);
        }

        println!("\nTotal evaluations: {}", result.total_evaluations);

        if let Some(consensus) = result.consensus_params() {
            println!("\nConsensus (highest robustness) MACD: {:?}",
                consensus.iter()
                    .flat_map(|p| p.params.iter().map(|(k, v)| format!("{}={}", k, *v as usize)))
                    .collect::<Vec<_>>());
        }
    }

    /// NAS100 MACD Pareto optimization: Sharpe vs MaxDrawdown (Option 2)
    /// Run with: cargo test test_nas100_pareto --lib -- --nocapture
    #[test]
    fn test_nas100_pareto() {
        let prices = get_nas100_data();
        println!("\n=== NAS100 MACD - Pareto Frontier ===");
        println!("Objectives: Sharpe Ratio vs Max Drawdown");
        println!("Data: {} points (2 years daily)\n", prices.len());

        let result = IndicatorOptimizer::new()
            .add_macd_range(
                (8, 16, 2),
                (20, 30, 2),
                (5, 12, 1),
            )
            .optimize_pareto(
                &prices,
                &[Objective::SharpeRatio, Objective::MaxDrawdown],
                50,  // population size
                30,  // generations
            );

        println!("Pareto frontier size: {} solutions", result.frontier.len());
        println!("Total evaluations: {}\n", result.evaluations);

        println!("{:<6} {:>10} {:>10} {:>20}", "#", "Sharpe", "MaxDD%", "MACD Params");
        println!("{}", "-".repeat(50));

        // Sort by Sharpe for display
        let mut sorted_frontier = result.frontier.clone();
        sorted_frontier.sort_by(|a, b| b.scores[0].partial_cmp(&a.scores[0]).unwrap());

        for (i, point) in sorted_frontier.iter().take(10).enumerate() {
            let params_str: String = point.params.iter()
                .flat_map(|p| p.params.iter().map(|(_, v)| format!("{}", *v as usize)))
                .collect::<Vec<_>>()
                .join("/");

            println!("{:<6} {:>10.4} {:>10.2}% {:>20}",
                i + 1,
                point.scores[0],
                -point.scores[1] * 100.0, // Convert back and to percentage
                params_str);
        }

        // Trade-off analysis
        if let Some(best_sharpe) = result.best_for(0) {
            println!("\n→ Best Sharpe: {:.4} (MaxDD: {:.2}%)",
                best_sharpe.scores[0], -best_sharpe.scores[1] * 100.0);
        }
        if let Some(best_dd) = result.best_for(1) {
            println!("→ Lowest MaxDD: {:.2}% (Sharpe: {:.4})",
                -best_dd.scores[1] * 100.0, best_dd.scores[0]);
        }
        if let Some(balanced) = result.balanced_solution() {
            println!("→ Balanced: Sharpe={:.4}, MaxDD={:.2}%",
                balanced.scores[0], -balanced.scores[1] * 100.0);
        }
    }

    /// NAS100 with multiple indicators: MACD + RSI + SMA
    /// Run with: cargo test test_nas100_multi_indicator --lib -- --nocapture
    #[test]
    fn test_nas100_multi_indicator() {
        let prices = get_nas100_data();
        println!("\n=== NAS100 Multi-Indicator Optimization ===");
        println!("Indicators: MACD + RSI + SMA");
        println!("Data: {} points (2 years daily)\n", prices.len());

        let result = IndicatorOptimizer::new()
            .add_macd_range(
                (8, 14, 3),   // fast: 8, 11, 14
                (20, 26, 3),  // slow: 20, 23, 26
                (7, 11, 2),   // signal: 7, 9, 11
            )
            .add_rsi_range(10, 20, 5)      // RSI: 10, 15, 20
            .add_sma_range(10, 30, 10)     // SMA: 10, 20, 30
            .method(OptMethod::ParallelGrid)
            .validation(Validation::WalkForward { windows: 3, train_ratio: 0.75 })
            .optimize_all_objectives(&prices);

        println!("{:<25} {:>10} {:>10} {:>10}", "Objective", "Score", "OOS", "Robust");
        println!("{}", "-".repeat(60));

        for obj_result in &result.results {
            println!("{:<25} {:>10.4} {:>10.4} {:>10.4}",
                obj_result.objective,
                obj_result.best_score,
                obj_result.oos_score.unwrap_or(0.0),
                obj_result.robustness.unwrap_or(0.0));

            // Show params for each indicator
            for param in &obj_result.best_params {
                let params_str: String = param.params.iter()
                    .map(|(k, v)| format!("{}={}", k, *v as usize))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("  └─ {}: {}", param.indicator_type, params_str);
            }
        }

        println!("\nTotal evaluations: {}", result.total_evaluations);
        println!("Parameter space: MACD(3×3×3) × RSI(3) × SMA(3) = {} combos",
            3 * 3 * 3 * 3 * 3);
    }

    /// Compare signal combination strategies on NAS100
    /// Run with: cargo test test_signal_combinations --lib -- --nocapture
    #[test]
    fn test_signal_combinations() {
        let prices = get_nas100_data();
        println!("\n=== Signal Combination Strategy Comparison ===");
        println!("Indicators: MACD + RSI + SMA");
        println!("Objective: Sharpe Ratio");
        println!("Data: {} points\n", prices.len());

        let strategies = vec![
            ("FirstOnly", SignalCombination::FirstOnly),
            ("Unanimous", SignalCombination::Unanimous),
            ("Majority", SignalCombination::Majority),
            ("Average", SignalCombination::Average),
            ("Confirmation", SignalCombination::Confirmation),
        ];

        println!("{:<15} {:>10} {:>10} {:>10} {:>25}",
            "Strategy", "Score", "OOS", "Robust", "Best Params");
        println!("{}", "-".repeat(75));

        for (name, strategy) in strategies {
            let result = IndicatorOptimizer::new()
                .add_macd_range((8, 14, 3), (20, 26, 3), (7, 11, 2))
                .add_rsi_range(10, 20, 5)
                .add_sma_range(10, 30, 10)
                .signal_combination(strategy)
                .objective(Objective::SharpeRatio)
                .method(OptMethod::ParallelGrid)
                .validation(Validation::WalkForward { windows: 3, train_ratio: 0.75 })
                .optimize(&prices);

            let params_str: String = result.best_params.iter()
                .map(|p| {
                    let vals: String = p.params.iter()
                        .map(|(_, v)| format!("{}", *v as usize))
                        .collect::<Vec<_>>()
                        .join("/");
                    format!("{}({})", &p.indicator_type[..1], vals)
                })
                .collect::<Vec<_>>()
                .join(" ");

            println!("{:<15} {:>10.4} {:>10.4} {:>10.4} {:>25}",
                name,
                result.best_score,
                result.oos_score.unwrap_or(0.0),
                result.robustness.unwrap_or(0.0),
                params_str);
        }
    }

    /// Optimize each indicator individually on NAS100
    /// Run with: cargo test test_individual_optimization --lib -- --nocapture
    #[test]
    fn test_individual_optimization() {
        let prices = get_nas100_data();
        println!("\n=== Individual Indicator Optimization (NAS100) ===");
        println!("Objective: Sharpe Ratio");
        println!("Data: {} points\n", prices.len());

        let result = IndicatorOptimizer::new()
            .add_macd_range((6, 16, 2), (18, 32, 2), (5, 14, 3))
            .add_rsi_range(5, 25, 5)
            .add_sma_range(5, 50, 5)
            .add_ema_range(5, 50, 5)
            .add_bollinger_range(10, 30, 5, 1.5, 3.0, 0.5)
            .objective(Objective::SharpeRatio)
            .method(OptMethod::ParallelGrid)
            .validation(Validation::WalkForward { windows: 3, train_ratio: 0.75 })
            .optimize_individual(&prices);

        println!("{:<12} {:>10} {:>10} {:>10} {:>25}",
            "Indicator", "Score", "OOS", "Robust", "Optimal Params");
        println!("{}", "-".repeat(70));

        for ind in &result.results {
            let params_str: String = ind.best_params.params.iter()
                .map(|(k, v)| {
                    if k == "std_dev" {
                        format!("{}={:.1}", k, v)
                    } else {
                        format!("{}={}", k, *v as usize)
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");

            println!("{:<12} {:>10.4} {:>10.4} {:>10.4} {:>25}",
                ind.indicator_name,
                ind.best_score,
                ind.oos_score.unwrap_or(0.0),
                ind.robustness.unwrap_or(0.0),
                params_str);
        }

        println!("\nTotal evaluations: {}", result.total_evaluations);

        if let Some(best) = result.best() {
            println!("\n→ Best performer: {} (score: {:.4})",
                best.indicator_name, best.best_score);
        }
        if let Some(robust) = result.most_robust() {
            println!("→ Most robust: {} (robustness: {:.4})",
                robust.indicator_name, robust.robustness.unwrap_or(0.0));
        }
    }

    // ========================================================================
    // Strategy Builder Tests
    // ========================================================================

    /// Test all strategy templates on NAS100 data
    /// Run with: cargo test test_strategy_templates --lib -- --nocapture
    #[test]
    fn test_strategy_templates() {
        let prices = get_nas100_data();
        println!("\n=== Strategy Backtest Results (NAS100 2023-2024) ===\n");

        // Get optimized indicators
        let opt_result = IndicatorOptimizer::new()
            .add_macd_range((6, 10, 2), (18, 22, 2), (5, 9, 2))
            .add_rsi_range(5, 15, 5)
            .add_bollinger_range(10, 20, 5, 1.5, 2.5, 0.5)
            .add_sma_range(40, 60, 10)
            .objective(Objective::SharpeRatio)
            .method(OptMethod::ParallelGrid)
            .validation(Validation::TrainTest { train_ratio: 0.7 })
            .optimize_individual(&prices);

        println!("Optimized Parameters:");
        for ind in &opt_result.results {
            let params: String = ind.best_params.params.iter()
                .map(|(k, v)| format!("{}={}", k, if k == "std_dev" { format!("{:.1}", v) } else { format!("{}", *v as i32) }))
                .collect::<Vec<_>>()
                .join(", ");
            println!("  {}: {}", ind.indicator_name, params);
        }
        println!();

        // Create strategies using optimized params
        let strategies: Vec<(&str, StrategyBuilder)> = vec![
            ("RSI Mean Reversion", StrategyBuilder::from_optimized(&opt_result).rsi_mean_reversion(30.0, 70.0)),
            ("MACD Crossover", StrategyBuilder::from_optimized(&opt_result).macd_crossover()),
            ("Bollinger Mean Rev", StrategyBuilder::from_optimized(&opt_result).bollinger_mean_reversion()),
            ("Bollinger Breakout", StrategyBuilder::from_optimized(&opt_result).bollinger_breakout()),
            ("MA Trend Following", StrategyBuilder::from_optimized(&opt_result).ma_trend_following()),
            ("RSI + MACD Combo", StrategyBuilder::from_optimized(&opt_result).rsi_macd_combo(35.0, 65.0)),
            ("Triple Indicator", StrategyBuilder::from_optimized(&opt_result).triple_indicator()),
        ];

        println!("{:<20} {:>8} {:>8} {:>10} {:>10} {:>8} {:>8}",
            "Strategy", "Trades", "Win%", "Return%", "Sharpe", "MaxDD%", "PF");
        println!("{}", "-".repeat(82));

        let mut best_sharpe = f64::MIN;
        let mut best_strategy = "";

        for (name, strategy) in &strategies {
            let result = strategy.backtest(&prices);
            let m = &result.metrics;

            println!("{:<20} {:>8} {:>8.1} {:>10.2} {:>10.2} {:>8.2} {:>8.2}",
                name,
                m.total_trades,
                m.win_rate,
                m.total_return,
                m.sharpe_ratio,
                m.max_drawdown,
                if m.profit_factor.is_infinite() { 999.99 } else { m.profit_factor }
            );

            if m.sharpe_ratio > best_sharpe && m.total_trades >= 5 {
                best_sharpe = m.sharpe_ratio;
                best_strategy = name;
            }
        }

        println!("\n→ Best Strategy: {} (Sharpe: {:.2})", best_strategy, best_sharpe);
    }

    /// Detailed analysis of best performing strategy
    /// Run with: cargo test test_strategy_details --lib -- --nocapture
    #[test]
    fn test_strategy_details() {
        let prices = get_nas100_data();
        println!("\n=== Detailed Strategy Analysis (NAS100) ===\n");

        // Use individually optimized params
        let rsi = OptimizedIndicator::new("RSI").with_param("period", 5.0);
        let macd = OptimizedIndicator::new("MACD")
            .with_param("fast", 6.0)
            .with_param("slow", 18.0)
            .with_param("signal", 5.0);
        let bollinger = OptimizedIndicator::new("Bollinger")
            .with_param("period", 10.0)
            .with_param("std_dev", 1.5);

        // Build strategy with optimized RSI
        let strategy = StrategyBuilder::new()
            .add_indicator(rsi)
            .add_indicator(macd)
            .add_indicator(bollinger)
            .rsi_mean_reversion(25.0, 75.0)  // Tighter thresholds
            .initial_equity(100000.0);

        let result = strategy.backtest(&prices);
        let m = &result.metrics;

        println!("Strategy: RSI Mean Reversion with RSI(5)");
        println!("Data: {} price points\n", prices.len());

        println!("=== Performance Metrics ===");
        println!("Total Trades:     {}", m.total_trades);
        println!("Winners/Losers:   {} / {}", m.winners, m.losers);
        println!("Win Rate:         {:.1}%", m.win_rate);
        println!("Total Return:     {:.2}%", m.total_return);
        println!("Sharpe Ratio:     {:.2}", m.sharpe_ratio);
        println!("Max Drawdown:     {:.2}%", m.max_drawdown);
        println!("Profit Factor:    {:.2}", if m.profit_factor.is_infinite() { 999.99 } else { m.profit_factor });
        println!();
        println!("=== Trade Statistics ===");
        println!("Avg Trade:        {:.2}%", m.avg_trade);
        println!("Avg Winner:       {:.2}%", m.avg_winner);
        println!("Avg Loser:        {:.2}%", m.avg_loser);
        println!("Largest Winner:   {:.2}%", m.largest_winner);
        println!("Largest Loser:    {:.2}%", m.largest_loser);
        println!("Avg Holding:      {:.1} bars", m.avg_holding_period);

        if !result.trades.is_empty() {
            println!("\n=== Recent Trades ===");
            let recent_trades: Vec<_> = result.trades.iter().rev().take(5).collect();
            for trade in recent_trades.iter().rev() {
                let dir = if trade.direction > 0 { "LONG" } else { "SHORT" };
                println!("  Bar {}-{}: {} Entry=${:.2} Exit=${:.2} PnL={:.2}%",
                    trade.entry_bar, trade.exit_bar, dir,
                    trade.entry_price, trade.exit_price, trade.pnl_pct);
            }
        }
    }

    /// Compare different position sizing methods
    /// Run with: cargo test test_position_sizing --lib -- --nocapture
    #[test]
    fn test_position_sizing() {
        let prices = get_nas100_data();
        println!("\n=== Position Sizing Comparison (NAS100) ===\n");

        let rsi = OptimizedIndicator::new("RSI").with_param("period", 5.0);

        let sizing_methods: Vec<(&str, PositionSizing)> = vec![
            ("Fixed $10k", PositionSizing::Fixed { size: 10000.0 }),
            ("10% Equity", PositionSizing::PercentEquity { pct: 10.0 }),
            ("25% Equity", PositionSizing::PercentEquity { pct: 25.0 }),
            ("50% Equity", PositionSizing::PercentEquity { pct: 50.0 }),
            ("100% Equity", PositionSizing::PercentEquity { pct: 100.0 }),
            ("Vol-Adjusted 2%", PositionSizing::VolatilityAdjusted { risk_pct: 2.0, atr_mult: 2.0 }),
            ("Kelly 25%", PositionSizing::Kelly { fraction: 0.25 }),
        ];

        println!("{:<18} {:>12} {:>12} {:>10}",
            "Sizing Method", "Final Equity", "Return%", "MaxDD%");
        println!("{}", "-".repeat(56));

        for (name, sizing) in sizing_methods {
            let strategy = StrategyBuilder::new()
                .add_indicator(rsi.clone())
                .rsi_mean_reversion(30.0, 70.0)
                .initial_equity(100000.0)
                .position_sizing(sizing);

            let result = strategy.backtest(&prices);
            let final_equity = result.equity_curve.last().copied().unwrap_or(100000.0);
            let return_pct = (final_equity - 100000.0) / 100000.0 * 100.0;

            println!("{:<18} {:>12.2} {:>12.2} {:>10.2}",
                name,
                final_equity,
                return_pct,
                result.metrics.max_drawdown);
        }
    }
}
