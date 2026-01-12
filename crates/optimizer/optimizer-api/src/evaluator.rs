//! High-level Evaluator API for indicator parameter optimization.
//!
//! Example usage:
//! ```ignore
//! use optimizer_api::{Evaluator, IndicatorSpec, Objective, Timeframe};
//!
//! let results = Evaluator::new()
//!     .symbol("SPY")
//!     .timeframe(Timeframe::D1)
//!     .indicator(IndicatorSpec::RSI { period_min: 10, period_max: 20, step: 5 })
//!     .objective(Objective::SharpeRatio)
//!     .train_test(0.7)
//!     .run()?;
//! ```

use optimizer_spi::{
    Objective, ValidationStrategy, Timeframe, ParamRange, FloatParamRange,
    OptimizationResult, OptimizerError,
};
use serde::{Deserialize, Serialize};

/// Indicator specification with parameter ranges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndicatorSpec {
    /// RSI with period range.
    RSI {
        period_min: usize,
        period_max: usize,
        step: usize,
        overbought: f64,
        oversold: f64,
    },
    /// SMA crossover with fast/slow period ranges.
    SMA {
        fast_min: usize,
        fast_max: usize,
        slow_min: usize,
        slow_max: usize,
        step: usize,
    },
    /// EMA crossover with fast/slow period ranges.
    EMA {
        fast_min: usize,
        fast_max: usize,
        slow_min: usize,
        slow_max: usize,
        step: usize,
    },
    /// MACD with fast/slow/signal ranges.
    MACD {
        fast_min: usize,
        fast_max: usize,
        slow_min: usize,
        slow_max: usize,
        signal_min: usize,
        signal_max: usize,
        step: usize,
    },
    /// Bollinger Bands with period and std_dev ranges.
    Bollinger {
        period_min: usize,
        period_max: usize,
        period_step: usize,
        std_dev_min: f64,
        std_dev_max: f64,
        std_dev_step: f64,
    },
    /// Stochastic with K and D period ranges.
    Stochastic {
        k_min: usize,
        k_max: usize,
        d_min: usize,
        d_max: usize,
        step: usize,
        overbought: f64,
        oversold: f64,
    },
    /// ATR with period range.
    ATR {
        period_min: usize,
        period_max: usize,
        step: usize,
    },
}

impl IndicatorSpec {
    /// RSI with default thresholds.
    pub fn rsi(period_min: usize, period_max: usize, step: usize) -> Self {
        IndicatorSpec::RSI {
            period_min,
            period_max,
            step,
            overbought: 70.0,
            oversold: 30.0,
        }
    }

    /// SMA crossover.
    pub fn sma_crossover(
        fast_min: usize, fast_max: usize,
        slow_min: usize, slow_max: usize,
        step: usize,
    ) -> Self {
        IndicatorSpec::SMA {
            fast_min, fast_max, slow_min, slow_max, step,
        }
    }

    /// EMA crossover.
    pub fn ema_crossover(
        fast_min: usize, fast_max: usize,
        slow_min: usize, slow_max: usize,
        step: usize,
    ) -> Self {
        IndicatorSpec::EMA {
            fast_min, fast_max, slow_min, slow_max, step,
        }
    }

    /// MACD.
    pub fn macd(
        fast_min: usize, fast_max: usize,
        slow_min: usize, slow_max: usize,
        signal_min: usize, signal_max: usize,
        step: usize,
    ) -> Self {
        IndicatorSpec::MACD {
            fast_min, fast_max,
            slow_min, slow_max,
            signal_min, signal_max,
            step,
        }
    }

    /// Bollinger Bands.
    pub fn bollinger(
        period_min: usize, period_max: usize, period_step: usize,
        std_dev_min: f64, std_dev_max: f64, std_dev_step: f64,
    ) -> Self {
        IndicatorSpec::Bollinger {
            period_min, period_max, period_step,
            std_dev_min, std_dev_max, std_dev_step,
        }
    }

    /// Stochastic with default thresholds.
    pub fn stochastic(
        k_min: usize, k_max: usize,
        d_min: usize, d_max: usize,
        step: usize,
    ) -> Self {
        IndicatorSpec::Stochastic {
            k_min, k_max, d_min, d_max, step,
            overbought: 80.0,
            oversold: 20.0,
        }
    }

    /// ATR.
    pub fn atr(period_min: usize, period_max: usize, step: usize) -> Self {
        IndicatorSpec::ATR { period_min, period_max, step }
    }
}

/// Evaluator builder for indicator optimization.
#[derive(Debug, Clone)]
pub struct Evaluator {
    symbol: Option<String>,
    timeframe: Timeframe,
    indicators: Vec<IndicatorSpec>,
    objective: Objective,
    validation: ValidationStrategy,
    parallel: bool,
    top_n: usize,
}

impl Default for Evaluator {
    fn default() -> Self {
        Self {
            symbol: None,
            timeframe: Timeframe::D1,
            indicators: Vec::new(),
            objective: Objective::SharpeRatio,
            validation: ValidationStrategy::None,
            parallel: true,
            top_n: 10,
        }
    }
}

impl Evaluator {
    /// Create new Evaluator builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set symbol to optimize.
    pub fn symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }

    /// Set timeframe.
    pub fn timeframe(mut self, tf: Timeframe) -> Self {
        self.timeframe = tf;
        self
    }

    /// Add indicator to optimize.
    pub fn indicator(mut self, spec: IndicatorSpec) -> Self {
        self.indicators.push(spec);
        self
    }

    /// Set optimization objective.
    pub fn objective(mut self, obj: Objective) -> Self {
        self.objective = obj;
        self
    }

    /// Use train/test split validation.
    pub fn train_test(mut self, train_ratio: f64) -> Self {
        self.validation = ValidationStrategy::TrainTest { train_ratio };
        self
    }

    /// Use walk-forward validation.
    pub fn walk_forward(mut self, windows: usize) -> Self {
        self.validation = ValidationStrategy::WalkForward {
            windows,
            train_ratio: 0.8,
        };
        self
    }

    /// Use k-fold cross-validation.
    pub fn k_fold(mut self, folds: usize) -> Self {
        self.validation = ValidationStrategy::KFold { folds };
        self
    }

    /// Enable/disable parallel processing.
    pub fn parallel(mut self, enable: bool) -> Self {
        self.parallel = enable;
        self
    }

    /// Set number of top results to return.
    pub fn top(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    /// Get the symbol.
    pub fn get_symbol(&self) -> Option<&str> {
        self.symbol.as_deref()
    }

    /// Get the timeframe.
    pub fn get_timeframe(&self) -> Timeframe {
        self.timeframe
    }

    /// Get the indicators.
    pub fn get_indicators(&self) -> &[IndicatorSpec] {
        &self.indicators
    }

    /// Get the objective.
    pub fn get_objective(&self) -> Objective {
        self.objective
    }

    /// Get the validation strategy.
    pub fn get_validation(&self) -> &ValidationStrategy {
        &self.validation
    }

    /// Get parallel setting.
    pub fn is_parallel(&self) -> bool {
        self.parallel
    }

    /// Get top_n setting.
    pub fn get_top_n(&self) -> usize {
        self.top_n
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), OptimizerError> {
        if self.symbol.is_none() {
            return Err(OptimizerError::InvalidConfig("Symbol is required".to_string()));
        }
        if self.indicators.is_empty() {
            return Err(OptimizerError::InvalidConfig("At least one indicator is required".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluator_builder() {
        let eval = Evaluator::new()
            .symbol("SPY")
            .timeframe(Timeframe::D1)
            .indicator(IndicatorSpec::rsi(10, 20, 5))
            .objective(Objective::SharpeRatio)
            .train_test(0.7);

        assert_eq!(eval.get_symbol(), Some("SPY"));
        assert_eq!(eval.get_timeframe(), Timeframe::D1);
        assert_eq!(eval.get_indicators().len(), 1);
        eval.validate().unwrap();
    }

    #[test]
    fn test_multiple_indicators() {
        let eval = Evaluator::new()
            .symbol("AAPL")
            .timeframe(Timeframe::H4)
            .indicator(IndicatorSpec::rsi(10, 20, 5))
            .indicator(IndicatorSpec::macd(8, 12, 20, 26, 7, 11, 2))
            .walk_forward(5);

        assert_eq!(eval.get_indicators().len(), 2);
    }

    #[test]
    fn test_validation_required() {
        let eval = Evaluator::new();
        assert!(eval.validate().is_err());

        let eval = Evaluator::new().symbol("SPY");
        assert!(eval.validate().is_err());

        let eval = Evaluator::new()
            .symbol("SPY")
            .indicator(IndicatorSpec::rsi(10, 20, 5));
        assert!(eval.validate().is_ok());
    }
}
