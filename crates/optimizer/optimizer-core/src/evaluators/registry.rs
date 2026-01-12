//! Evaluator registry and factory.

use optimizer_spi::{IndicatorEvaluator, ParamRange, FloatParamRange};
use super::{
    RSIEvaluator, SMAEvaluator, EMAEvaluator, MACDEvaluator,
    BollingerEvaluator, StochasticEvaluator, ATREvaluator,
};

/// Available evaluator types.
#[derive(Debug, Clone)]
pub enum EvaluatorType {
    RSI {
        period: ParamRange,
        overbought: f64,
        oversold: f64,
    },
    SMA {
        fast: ParamRange,
        slow: ParamRange,
    },
    EMA {
        fast: ParamRange,
        slow: ParamRange,
    },
    MACD {
        fast: ParamRange,
        slow: ParamRange,
        signal: ParamRange,
    },
    Bollinger {
        period: ParamRange,
        std_dev: FloatParamRange,
    },
    Stochastic {
        k: ParamRange,
        d: ParamRange,
        overbought: f64,
        oversold: f64,
    },
    ATR {
        period: ParamRange,
    },
}

impl Default for EvaluatorType {
    fn default() -> Self {
        EvaluatorType::RSI {
            period: ParamRange::new(10, 20, 2),
            overbought: 70.0,
            oversold: 30.0,
        }
    }
}

/// Create an evaluator from its type specification.
pub fn create_evaluator(config: EvaluatorType) -> Box<dyn IndicatorEvaluator> {
    match config {
        EvaluatorType::RSI { period, overbought, oversold } => {
            Box::new(RSIEvaluator::new(period).with_thresholds(overbought, oversold))
        }
        EvaluatorType::SMA { fast, slow } => {
            Box::new(SMAEvaluator::new(fast, slow))
        }
        EvaluatorType::EMA { fast, slow } => {
            Box::new(EMAEvaluator::new(fast, slow))
        }
        EvaluatorType::MACD { fast, slow, signal } => {
            Box::new(MACDEvaluator::new(fast, slow, signal))
        }
        EvaluatorType::Bollinger { period, std_dev } => {
            Box::new(BollingerEvaluator::new(period, std_dev))
        }
        EvaluatorType::Stochastic { k, d, overbought, oversold } => {
            Box::new(StochasticEvaluator::new(k, d).with_thresholds(overbought, oversold))
        }
        EvaluatorType::ATR { period } => {
            Box::new(ATREvaluator::new(period))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_evaluator() {
        let config = EvaluatorType::RSI {
            period: ParamRange::new(10, 20, 5),
            overbought: 70.0,
            oversold: 30.0,
        };
        let evaluator = create_evaluator(config);
        assert_eq!(evaluator.name(), "RSI");

        let config = EvaluatorType::MACD {
            fast: ParamRange::new(8, 16, 4),
            slow: ParamRange::new(20, 30, 5),
            signal: ParamRange::new(7, 11, 2),
        };
        let evaluator = create_evaluator(config);
        assert_eq!(evaluator.name(), "MACD");
    }
}
