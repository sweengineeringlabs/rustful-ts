//! SMA Evaluator for optimization.
//! Uses dual SMA crossover strategy.

use optimizer_spi::{
    IndicatorEvaluator, IndicatorParams, EvaluationResult, MarketData, Signal,
    ParamRange, FloatParamRange, Result, OptimizerError,
};
use indicator_core::SMA;

/// SMA Evaluator with crossover signals.
/// Uses fast/slow SMA crossover to generate signals.
#[derive(Debug, Clone)]
pub struct SMAEvaluator {
    fast_range: ParamRange,
    slow_range: ParamRange,
}

impl SMAEvaluator {
    pub fn new(fast_range: ParamRange, slow_range: ParamRange) -> Self {
        Self { fast_range, slow_range }
    }
}

impl IndicatorEvaluator for SMAEvaluator {
    fn name(&self) -> &str {
        "SMA"
    }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let fast = params.get_usize("fast").ok_or_else(|| {
            OptimizerError::InvalidConfig("SMA requires 'fast' parameter".to_string())
        })?;
        let slow = params.get_usize("slow").ok_or_else(|| {
            OptimizerError::InvalidConfig("SMA requires 'slow' parameter".to_string())
        })?;

        if fast >= slow {
            return Err(OptimizerError::InvalidConfig(
                "SMA fast period must be less than slow period".to_string()
            ));
        }

        if data.close.len() < slow {
            return Err(OptimizerError::InsufficientData {
                required: slow,
                got: data.close.len(),
            });
        }

        let fast_sma = SMA::new(fast).calculate(&data.close);
        let slow_sma = SMA::new(slow).calculate(&data.close);

        // Generate crossover signals
        let n = data.close.len();
        let mut signals = vec![Signal::Hold; n];

        for i in 1..n {
            if fast_sma[i].is_nan() || slow_sma[i].is_nan() ||
               fast_sma[i-1].is_nan() || slow_sma[i-1].is_nan() {
                continue;
            }

            // Bullish crossover: fast crosses above slow
            if fast_sma[i-1] <= slow_sma[i-1] && fast_sma[i] > slow_sma[i] {
                signals[i] = Signal::Buy;
            }
            // Bearish crossover: fast crosses below slow
            else if fast_sma[i-1] >= slow_sma[i-1] && fast_sma[i] < slow_sma[i] {
                signals[i] = Signal::Sell;
            }
        }

        // Return the fast-slow difference as indicator value
        let values: Vec<f64> = fast_sma.iter()
            .zip(slow_sma.iter())
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

#[cfg(test)]
mod tests {
    use super::*;
    use optimizer_spi::Timeframe;

    #[test]
    fn test_sma_evaluator() {
        let evaluator = SMAEvaluator::new(
            ParamRange::new(5, 15, 5),
            ParamRange::new(20, 50, 10),
        );

        let mut data = MarketData::new("TEST", Timeframe::D1);
        for i in 0..100 {
            data.close.push(100.0 + i as f64);
            data.open.push(100.0 + i as f64);
            data.high.push(110.0 + i as f64);
            data.low.push(90.0 + i as f64);
            data.volume.push(1000.0);
            data.timestamps.push(i as i64);
        }

        let params = IndicatorParams::new("SMA")
            .with_param("fast", 10.0)
            .with_param("slow", 30.0);
        let result = evaluator.evaluate(&params, &data).unwrap();

        assert_eq!(result.signals.len(), 100);
    }
}
