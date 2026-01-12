//! Stochastic Oscillator Evaluator for optimization.

use optimizer_spi::{
    IndicatorEvaluator, IndicatorParams, EvaluationResult, MarketData, Signal,
    ParamRange, FloatParamRange, Result, OptimizerError,
};
use indicator_core::Stochastic;

/// Stochastic Evaluator with overbought/oversold signals.
#[derive(Debug, Clone)]
pub struct StochasticEvaluator {
    k_range: ParamRange,
    d_range: ParamRange,
    overbought: f64,
    oversold: f64,
}

impl StochasticEvaluator {
    pub fn new(k_range: ParamRange, d_range: ParamRange) -> Self {
        Self {
            k_range,
            d_range,
            overbought: 80.0,
            oversold: 20.0,
        }
    }

    pub fn with_thresholds(mut self, overbought: f64, oversold: f64) -> Self {
        self.overbought = overbought;
        self.oversold = oversold;
        self
    }
}

impl IndicatorEvaluator for StochasticEvaluator {
    fn name(&self) -> &str {
        "Stochastic"
    }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let k_period = params.get_usize("k_period").ok_or_else(|| {
            OptimizerError::InvalidConfig("Stochastic requires 'k_period' parameter".to_string())
        })?;
        let d_period = params.get_usize("d_period").ok_or_else(|| {
            OptimizerError::InvalidConfig("Stochastic requires 'd_period' parameter".to_string())
        })?;

        let required = k_period + d_period - 1;
        if data.close.len() < required {
            return Err(OptimizerError::InsufficientData {
                required,
                got: data.close.len(),
            });
        }

        let stoch = Stochastic::new(k_period, d_period);
        let (k, d) = stoch.calculate(&data.high, &data.low, &data.close);

        // Generate signals based on %K overbought/oversold
        let signals: Vec<Signal> = k.iter().map(|&v| {
            if v.is_nan() {
                Signal::Hold
            } else if v >= self.overbought {
                Signal::Sell
            } else if v <= self.oversold {
                Signal::Buy
            } else {
                Signal::Hold
            }
        }).collect();

        let mut result = EvaluationResult::new(params.clone());
        result.indicator_values = k;
        result.signals = signals;

        Ok(result)
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("k_period".to_string(), self.k_range.clone()),
            ("d_period".to_string(), self.d_range.clone()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimizer_spi::Timeframe;

    #[test]
    fn test_stochastic_evaluator() {
        let evaluator = StochasticEvaluator::new(
            ParamRange::new(10, 20, 5),
            ParamRange::new(3, 5, 1),
        );

        let mut data = MarketData::new("TEST", Timeframe::D1);
        for i in 0..100 {
            let base = 100.0 + (i as f64).sin() * 10.0;
            data.close.push(base);
            data.open.push(base);
            data.high.push(base + 2.0);
            data.low.push(base - 2.0);
            data.volume.push(1000.0);
            data.timestamps.push(i as i64);
        }

        let params = IndicatorParams::new("Stochastic")
            .with_param("k_period", 14.0)
            .with_param("d_period", 3.0);
        let result = evaluator.evaluate(&params, &data).unwrap();

        assert_eq!(result.signals.len(), 100);
    }
}
