//! RSI Evaluator for optimization.

use optimizer_spi::{
    IndicatorEvaluator, IndicatorParams, EvaluationResult, MarketData, Signal,
    ParamRange, FloatParamRange, Result, OptimizerError,
};
use indicator_core::RSI;

/// RSI Evaluator with overbought/oversold signals.
#[derive(Debug, Clone)]
pub struct RSIEvaluator {
    period_range: ParamRange,
    overbought: f64,
    oversold: f64,
}

impl RSIEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self {
            period_range,
            overbought: 70.0,
            oversold: 30.0,
        }
    }

    pub fn with_thresholds(mut self, overbought: f64, oversold: f64) -> Self {
        self.overbought = overbought;
        self.oversold = oversold;
        self
    }
}

impl IndicatorEvaluator for RSIEvaluator {
    fn name(&self) -> &str {
        "RSI"
    }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get_usize("period").ok_or_else(|| {
            OptimizerError::InvalidConfig("RSI requires 'period' parameter".to_string())
        })?;

        if data.close.len() < period + 1 {
            return Err(OptimizerError::InsufficientData {
                required: period + 1,
                got: data.close.len(),
            });
        }

        let rsi = RSI::new(period);
        let values = rsi.calculate(&data.close);

        // Generate signals based on overbought/oversold
        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v.is_nan() {
                Signal::Hold
            } else if v >= self.overbought {
                Signal::Sell // Overbought = potential reversal down
            } else if v <= self.oversold {
                Signal::Buy // Oversold = potential reversal up
            } else {
                Signal::Hold
            }
        }).collect();

        let mut result = EvaluationResult::new(params.clone());
        result.indicator_values = values;
        result.signals = signals;

        Ok(result)
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }

    fn float_parameter_space(&self) -> Vec<(String, FloatParamRange)> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimizer_spi::Timeframe;

    #[test]
    fn test_rsi_evaluator() {
        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));

        // Create test market data
        let mut data = MarketData::new("TEST", Timeframe::D1);
        for i in 0..100 {
            data.close.push(100.0 + (i as f64).sin() * 10.0);
            data.open.push(100.0 + (i as f64).sin() * 10.0);
            data.high.push(110.0 + (i as f64).sin() * 10.0);
            data.low.push(90.0 + (i as f64).sin() * 10.0);
            data.volume.push(1000.0);
            data.timestamps.push(i as i64);
        }

        let params = IndicatorParams::new("RSI").with_param("period", 14.0);
        let result = evaluator.evaluate(&params, &data).unwrap();

        assert_eq!(result.indicator_values.len(), 100);
        assert_eq!(result.signals.len(), 100);
    }
}
