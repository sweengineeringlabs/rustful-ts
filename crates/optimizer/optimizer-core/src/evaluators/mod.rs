//! Indicator Evaluators for Optimization
//!
//! Each evaluator wraps an indicator and provides:
//! - Parameter space definition
//! - Signal generation based on indicator values
//! - Integration with the optimization framework

mod rsi_evaluator;
mod sma_evaluator;
mod ema_evaluator;
mod macd_evaluator;
mod bollinger_evaluator;
mod stochastic_evaluator;
mod atr_evaluator;
mod registry;
pub mod bulk_evaluators;

// Core evaluators
pub use rsi_evaluator::RSIEvaluator;
pub use sma_evaluator::SMAEvaluator;
pub use ema_evaluator::EMAEvaluator;
pub use macd_evaluator::MACDEvaluator;
pub use bollinger_evaluator::BollingerEvaluator;
pub use stochastic_evaluator::StochasticEvaluator;
pub use atr_evaluator::ATREvaluator;
pub use registry::{create_evaluator, EvaluatorType};

// Bulk evaluators (43 additional indicators)
pub use bulk_evaluators::*;
