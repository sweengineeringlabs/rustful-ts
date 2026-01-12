//! Evaluator Runner - executes Evaluator configurations.

use optimizer_api::{Evaluator, IndicatorSpec};
use optimizer_spi::{
    DataSource, IndicatorEvaluator, Objective, OptimizationResult,
    OptimizerError, ParamRange, FloatParamRange, Result, ValidationStrategy,
};

use crate::datasource::FixtureDataSource;
use crate::evaluators::{
    RSIEvaluator, SMAEvaluator, EMAEvaluator, MACDEvaluator,
    BollingerEvaluator, StochasticEvaluator, ATREvaluator,
};
use crate::objective::create_objective;
use crate::optimizer::{IndicatorGridSearch, GridSearchConfig};
use crate::validator::{
    TrainTestValidator, WalkForwardValidator, KFoldValidator,
    TimeSeriesCVValidator, NoValidator,
};

/// Result of running an Evaluator.
#[derive(Debug, Clone)]
pub struct EvaluatorResult {
    /// Symbol that was optimized.
    pub symbol: String,
    /// Timeframe used.
    pub timeframe: optimizer_spi::Timeframe,
    /// Results for each indicator (in order they were added).
    pub indicator_results: Vec<OptimizationResult>,
}

/// Runner that executes Evaluator configurations.
pub struct EvaluatorRunner<D: DataSource = FixtureDataSource> {
    data_source: D,
}

impl EvaluatorRunner<FixtureDataSource> {
    /// Create runner with default FixtureDataSource.
    pub fn new() -> Self {
        Self {
            data_source: FixtureDataSource::new(),
        }
    }
}

impl<D: DataSource> EvaluatorRunner<D> {
    /// Create runner with custom data source.
    pub fn with_data_source(data_source: D) -> Self {
        Self { data_source }
    }

    /// Run the evaluator configuration.
    pub fn run(&self, evaluator: &Evaluator) -> Result<EvaluatorResult> {
        // Validate configuration
        evaluator.validate()?;

        let symbol = evaluator.get_symbol()
            .ok_or_else(|| OptimizerError::InvalidConfig("Symbol required".to_string()))?;
        let timeframe = evaluator.get_timeframe();

        // Load market data
        let data = self.data_source.load(symbol, timeframe)?;

        if data.is_empty() {
            return Err(OptimizerError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        // Create objective function
        let objective = create_objective(evaluator.get_objective());

        // Create validation splits
        let splits = create_validation_splits(evaluator.get_validation(), data.len())?;

        // Create grid search optimizer
        let grid_config = GridSearchConfig {
            parallel: evaluator.is_parallel(),
            top_n: evaluator.get_top_n(),
        };
        let grid = IndicatorGridSearch::with_config(grid_config);

        // Run optimization for each indicator
        let mut indicator_results = Vec::new();

        for spec in evaluator.get_indicators() {
            let ind_evaluator = spec_to_evaluator(spec)?;
            let result = grid.optimize(ind_evaluator.as_ref(), &data, objective.as_ref(), &splits)?;
            indicator_results.push(result);
        }

        Ok(EvaluatorResult {
            symbol: symbol.to_string(),
            timeframe,
            indicator_results,
        })
    }
}

impl Default for EvaluatorRunner<FixtureDataSource> {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert IndicatorSpec to a boxed IndicatorEvaluator.
fn spec_to_evaluator(spec: &IndicatorSpec) -> Result<Box<dyn IndicatorEvaluator>> {
    match spec {
        IndicatorSpec::RSI {
            period_min, period_max, step,
            overbought, oversold,
        } => {
            let evaluator = RSIEvaluator::new(ParamRange::new(*period_min, *period_max, *step))
                .with_thresholds(*overbought, *oversold);
            Ok(Box::new(evaluator))
        }

        IndicatorSpec::SMA {
            fast_min, fast_max,
            slow_min, slow_max,
            step,
        } => {
            let evaluator = SMAEvaluator::new(
                ParamRange::new(*fast_min, *fast_max, *step),
                ParamRange::new(*slow_min, *slow_max, *step),
            );
            Ok(Box::new(evaluator))
        }

        IndicatorSpec::EMA {
            fast_min, fast_max,
            slow_min, slow_max,
            step,
        } => {
            let evaluator = EMAEvaluator::new(
                ParamRange::new(*fast_min, *fast_max, *step),
                ParamRange::new(*slow_min, *slow_max, *step),
            );
            Ok(Box::new(evaluator))
        }

        IndicatorSpec::MACD {
            fast_min, fast_max,
            slow_min, slow_max,
            signal_min, signal_max,
            step,
        } => {
            let evaluator = MACDEvaluator::new(
                ParamRange::new(*fast_min, *fast_max, *step),
                ParamRange::new(*slow_min, *slow_max, *step),
                ParamRange::new(*signal_min, *signal_max, *step),
            );
            Ok(Box::new(evaluator))
        }

        IndicatorSpec::Bollinger {
            period_min, period_max, period_step,
            std_dev_min, std_dev_max, std_dev_step,
        } => {
            let evaluator = BollingerEvaluator::new(
                ParamRange::new(*period_min, *period_max, *period_step),
                FloatParamRange::new(*std_dev_min, *std_dev_max, *std_dev_step),
            );
            Ok(Box::new(evaluator))
        }

        IndicatorSpec::Stochastic {
            k_min, k_max,
            d_min, d_max,
            step,
            overbought, oversold,
        } => {
            let evaluator = StochasticEvaluator::new(
                ParamRange::new(*k_min, *k_max, *step),
                ParamRange::new(*d_min, *d_max, *step),
            ).with_thresholds(*overbought, *oversold);
            Ok(Box::new(evaluator))
        }

        IndicatorSpec::ATR {
            period_min, period_max, step,
        } => {
            let evaluator = ATREvaluator::new(ParamRange::new(*period_min, *period_max, *step));
            Ok(Box::new(evaluator))
        }
    }
}

/// Create validation splits from strategy.
fn create_validation_splits(
    strategy: &ValidationStrategy,
    data_len: usize,
) -> Result<Vec<optimizer_spi::ValidationSplit>> {
    use optimizer_spi::Validator;

    match strategy {
        ValidationStrategy::None => Ok(vec![]),

        ValidationStrategy::TrainTest { train_ratio } => {
            let validator = TrainTestValidator::new(*train_ratio);
            validator.splits(data_len)
        }

        ValidationStrategy::WalkForward { windows, train_ratio } => {
            let validator = WalkForwardValidator::new(*windows, *train_ratio);
            validator.splits(data_len)
        }

        ValidationStrategy::KFold { folds } => {
            let validator = KFoldValidator::new(*folds);
            validator.splits(data_len)
        }

        ValidationStrategy::TimeSeriesCV { n_splits, test_size } => {
            let validator = TimeSeriesCVValidator::new(*n_splits, *test_size);
            validator.splits(data_len)
        }
    }
}

// ============================================================================
// Extension trait for Evaluator
// ============================================================================

/// Extension trait to add `run()` method directly to Evaluator.
pub trait EvaluatorExt {
    /// Run the evaluator using the default FixtureDataSource.
    fn run(&self) -> Result<EvaluatorResult>;

    /// Run the evaluator with a custom data source.
    fn run_with<D: DataSource>(&self, data_source: &D) -> Result<EvaluatorResult>;
}

impl EvaluatorExt for Evaluator {
    fn run(&self) -> Result<EvaluatorResult> {
        let runner = EvaluatorRunner::new();
        runner.run(self)
    }

    fn run_with<D: DataSource>(&self, data_source: &D) -> Result<EvaluatorResult> {
        // Validate configuration
        self.validate()?;

        let symbol = self.get_symbol()
            .ok_or_else(|| OptimizerError::InvalidConfig("Symbol required".to_string()))?;
        let timeframe = self.get_timeframe();

        // Load market data
        let data = data_source.load(symbol, timeframe)?;

        if data.is_empty() {
            return Err(OptimizerError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        // Create objective function
        let objective = create_objective(self.get_objective());

        // Create validation splits
        let splits = create_validation_splits(self.get_validation(), data.len())?;

        // Create grid search optimizer
        let grid_config = GridSearchConfig {
            parallel: self.is_parallel(),
            top_n: self.get_top_n(),
        };
        let grid = IndicatorGridSearch::with_config(grid_config);

        // Run optimization for each indicator
        let mut indicator_results = Vec::new();

        for spec in self.get_indicators() {
            let ind_evaluator = spec_to_evaluator(spec)?;
            let result = grid.optimize(ind_evaluator.as_ref(), &data, objective.as_ref(), &splits)?;
            indicator_results.push(result);
        }

        Ok(EvaluatorResult {
            symbol: symbol.to_string(),
            timeframe,
            indicator_results,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimizer_spi::{Objective, Timeframe};

    #[test]
    fn test_evaluator_runner_single_indicator() {
        let evaluator = Evaluator::new()
            .symbol("SPY")
            .timeframe(Timeframe::D1)
            .indicator(IndicatorSpec::rsi(10, 20, 5))
            .objective(Objective::SharpeRatio);

        let runner = EvaluatorRunner::new();
        let result = runner.run(&evaluator).unwrap();

        assert_eq!(result.symbol, "SPY");
        assert_eq!(result.timeframe, Timeframe::D1);
        assert_eq!(result.indicator_results.len(), 1);

        let rsi_result = &result.indicator_results[0];
        println!("RSI best params: {:?}", rsi_result.best_params);
        println!("RSI best score: {:.4}", rsi_result.best_score);
    }

    #[test]
    fn test_evaluator_runner_multiple_indicators() {
        let evaluator = Evaluator::new()
            .symbol("AAPL")
            .timeframe(Timeframe::D1)
            .indicator(IndicatorSpec::rsi(10, 20, 5))
            .indicator(IndicatorSpec::macd(8, 14, 20, 28, 7, 11, 2))
            .objective(Objective::SharpeRatio)
            .train_test(0.7);

        let runner = EvaluatorRunner::new();
        let result = runner.run(&evaluator).unwrap();

        assert_eq!(result.indicator_results.len(), 2);

        println!("AAPL Multi-Indicator Results:");
        for (i, res) in result.indicator_results.iter().enumerate() {
            println!("  Indicator {}: {:?} -> {:.4}",
                i + 1, res.best_params, res.best_score);
            println!("    OOS: {:?}, Robustness: {:?}",
                res.oos_score, res.robustness);
        }
    }

    #[test]
    fn test_evaluator_runner_with_walk_forward() {
        let evaluator = Evaluator::new()
            .symbol("SPY")
            .timeframe(Timeframe::D1)
            .indicator(IndicatorSpec::rsi(10, 20, 5))
            .objective(Objective::SharpeRatio)
            .walk_forward(5);

        let runner = EvaluatorRunner::new();
        let result = runner.run(&evaluator).unwrap();

        let rsi_result = &result.indicator_results[0];
        assert!(rsi_result.oos_score.is_some());
        println!("Walk-forward RSI: IS={:.4}, OOS={:.4}",
            rsi_result.best_score,
            rsi_result.oos_score.unwrap());
    }

    #[test]
    fn test_evaluator_runner_h4() {
        let evaluator = Evaluator::new()
            .symbol("SPY")
            .timeframe(Timeframe::H4)
            .indicator(IndicatorSpec::rsi(10, 20, 5))
            .objective(Objective::SharpeRatio);

        let runner = EvaluatorRunner::new();
        let result = runner.run(&evaluator).unwrap();

        assert_eq!(result.timeframe, Timeframe::H4);
        println!("H4 RSI: {:?}", result.indicator_results[0].best_params);
    }

    #[test]
    fn test_evaluator_runner_bollinger() {
        let evaluator = Evaluator::new()
            .symbol("EURUSD")
            .timeframe(Timeframe::D1)
            .indicator(IndicatorSpec::bollinger(15, 25, 5, 1.5, 2.5, 0.5))
            .objective(Objective::SortinoRatio);

        let runner = EvaluatorRunner::new();
        let result = runner.run(&evaluator).unwrap();

        println!("Bollinger EURUSD: {:?} -> {:.4}",
            result.indicator_results[0].best_params,
            result.indicator_results[0].best_score);
    }

    // Tests for extension trait (direct .run() on Evaluator)
    #[test]
    fn test_evaluator_ext_run() {
        use super::EvaluatorExt;

        let result = Evaluator::new()
            .symbol("SPY")
            .timeframe(Timeframe::D1)
            .indicator(IndicatorSpec::rsi(10, 20, 5))
            .objective(Objective::SharpeRatio)
            .run()
            .unwrap();

        assert_eq!(result.symbol, "SPY");
        assert_eq!(result.indicator_results.len(), 1);
        println!("Direct .run() test: {:?} -> {:.4}",
            result.indicator_results[0].best_params,
            result.indicator_results[0].best_score);
    }

    #[test]
    fn test_evaluator_ext_full_api() {
        use super::EvaluatorExt;

        // This is the intended one-liner API
        let result = Evaluator::new()
            .symbol("AAPL")
            .timeframe(Timeframe::H4)
            .indicator(IndicatorSpec::rsi(10, 20, 5))
            .indicator(IndicatorSpec::macd(8, 14, 20, 28, 7, 11, 2))
            .objective(Objective::SharpeRatio)
            .train_test(0.7)
            .parallel(true)
            .top(5)
            .run()
            .unwrap();

        println!("Full API test on AAPL H4:");
        for (i, res) in result.indicator_results.iter().enumerate() {
            println!("  Indicator {}: {:?}", i + 1, res.best_params);
            println!("    Score: {:.4}, OOS: {:?}", res.best_score, res.oos_score);
        }

        assert_eq!(result.indicator_results.len(), 2);
        assert!(result.indicator_results[0].oos_score.is_some());
    }
}
