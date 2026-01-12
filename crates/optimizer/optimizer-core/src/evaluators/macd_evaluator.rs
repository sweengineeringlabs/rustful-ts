//! MACD Evaluator for optimization.

use optimizer_spi::{
    IndicatorEvaluator, IndicatorParams, EvaluationResult, MarketData, Signal,
    ParamRange, FloatParamRange, Result, OptimizerError,
};
use indicator_core::MACD;

/// MACD Evaluator with crossover signals.
#[derive(Debug, Clone)]
pub struct MACDEvaluator {
    fast_range: ParamRange,
    slow_range: ParamRange,
    signal_range: ParamRange,
}

impl MACDEvaluator {
    pub fn new(fast_range: ParamRange, slow_range: ParamRange, signal_range: ParamRange) -> Self {
        Self { fast_range, slow_range, signal_range }
    }
}

impl IndicatorEvaluator for MACDEvaluator {
    fn name(&self) -> &str {
        "MACD"
    }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let fast = params.get_usize("fast").ok_or_else(|| {
            OptimizerError::InvalidConfig("MACD requires 'fast' parameter".to_string())
        })?;
        let slow = params.get_usize("slow").ok_or_else(|| {
            OptimizerError::InvalidConfig("MACD requires 'slow' parameter".to_string())
        })?;
        let signal = params.get_usize("signal").ok_or_else(|| {
            OptimizerError::InvalidConfig("MACD requires 'signal' parameter".to_string())
        })?;

        if fast >= slow {
            return Err(OptimizerError::InvalidConfig(
                "MACD fast period must be less than slow period".to_string()
            ));
        }

        let min_required = slow + signal - 1;
        if data.close.len() < min_required {
            return Err(OptimizerError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let macd = MACD::new(fast, slow, signal);
        let (macd_line, signal_line, histogram) = macd.calculate(&data.close);

        // Generate crossover signals
        let n = data.close.len();
        let mut signals = vec![Signal::Hold; n];

        for i in 1..n {
            if macd_line[i].is_nan() || signal_line[i].is_nan() ||
               macd_line[i-1].is_nan() || signal_line[i-1].is_nan() {
                continue;
            }

            // Bullish crossover: MACD crosses above signal
            if macd_line[i-1] <= signal_line[i-1] && macd_line[i] > signal_line[i] {
                signals[i] = Signal::Buy;
            }
            // Bearish crossover: MACD crosses below signal
            else if macd_line[i-1] >= signal_line[i-1] && macd_line[i] < signal_line[i] {
                signals[i] = Signal::Sell;
            }
        }

        let mut result = EvaluationResult::new(params.clone());
        result.indicator_values = histogram; // Use histogram as primary value
        result.signals = signals;

        Ok(result)
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("fast".to_string(), self.fast_range.clone()),
            ("slow".to_string(), self.slow_range.clone()),
            ("signal".to_string(), self.signal_range.clone()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimizer_spi::Timeframe;

    #[test]
    fn test_macd_evaluator() {
        let evaluator = MACDEvaluator::new(
            ParamRange::new(8, 16, 4),
            ParamRange::new(20, 30, 5),
            ParamRange::new(7, 11, 2),
        );

        let mut data = MarketData::new("TEST", Timeframe::D1);
        for i in 0..100 {
            data.close.push(100.0 + (i as f64).sin() * 10.0);
            data.open.push(100.0 + (i as f64).sin() * 10.0);
            data.high.push(110.0 + (i as f64).sin() * 10.0);
            data.low.push(90.0 + (i as f64).sin() * 10.0);
            data.volume.push(1000.0);
            data.timestamps.push(i as i64);
        }

        let params = IndicatorParams::new("MACD")
            .with_param("fast", 12.0)
            .with_param("slow", 26.0)
            .with_param("signal", 9.0);
        let result = evaluator.evaluate(&params, &data).unwrap();

        assert_eq!(result.signals.len(), 100);
    }
}
