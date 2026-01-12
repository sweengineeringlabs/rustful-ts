//! EMA Evaluator for optimization.
//! Uses dual EMA crossover strategy.

use optimizer_spi::{
    IndicatorEvaluator, IndicatorParams, EvaluationResult, MarketData, Signal,
    ParamRange, FloatParamRange, Result, OptimizerError,
};
use indicator_core::EMA;

/// EMA Evaluator with crossover signals.
/// Uses fast/slow EMA crossover to generate signals.
#[derive(Debug, Clone)]
pub struct EMAEvaluator {
    fast_range: ParamRange,
    slow_range: ParamRange,
}

impl EMAEvaluator {
    pub fn new(fast_range: ParamRange, slow_range: ParamRange) -> Self {
        Self { fast_range, slow_range }
    }
}

impl IndicatorEvaluator for EMAEvaluator {
    fn name(&self) -> &str {
        "EMA"
    }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let fast = params.get_usize("fast").ok_or_else(|| {
            OptimizerError::InvalidConfig("EMA requires 'fast' parameter".to_string())
        })?;
        let slow = params.get_usize("slow").ok_or_else(|| {
            OptimizerError::InvalidConfig("EMA requires 'slow' parameter".to_string())
        })?;

        if fast >= slow {
            return Err(OptimizerError::InvalidConfig(
                "EMA fast period must be less than slow period".to_string()
            ));
        }

        if data.close.len() < slow {
            return Err(OptimizerError::InsufficientData {
                required: slow,
                got: data.close.len(),
            });
        }

        let fast_ema = EMA::new(fast).calculate(&data.close);
        let slow_ema = EMA::new(slow).calculate(&data.close);

        // Generate crossover signals
        let n = data.close.len();
        let mut signals = vec![Signal::Hold; n];

        for i in 1..n {
            if fast_ema[i].is_nan() || slow_ema[i].is_nan() ||
               fast_ema[i-1].is_nan() || slow_ema[i-1].is_nan() {
                continue;
            }

            // Bullish crossover
            if fast_ema[i-1] <= slow_ema[i-1] && fast_ema[i] > slow_ema[i] {
                signals[i] = Signal::Buy;
            }
            // Bearish crossover
            else if fast_ema[i-1] >= slow_ema[i-1] && fast_ema[i] < slow_ema[i] {
                signals[i] = Signal::Sell;
            }
        }

        let values: Vec<f64> = fast_ema.iter()
            .zip(slow_ema.iter())
            .map(|(f, s)| {
                if f.is_nan() || s.is_nan() { f64::NAN } else { f - s }
            })
            .collect();

        let mut result = EvaluationResult::new(params.clone());
        result.indicator_values = values;
        result.signals = signals;

        Ok(result)
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("fast".to_string(), self.fast_range.clone()),
            ("slow".to_string(), self.slow_range.clone()),
        ]
    }
}
