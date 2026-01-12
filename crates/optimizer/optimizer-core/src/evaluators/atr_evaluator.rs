//! ATR Evaluator for optimization.
//! ATR is used for volatility-based position sizing/stops, not direct signals.

use optimizer_spi::{
    IndicatorEvaluator, IndicatorParams, EvaluationResult, MarketData, Signal,
    ParamRange, FloatParamRange, Result, OptimizerError,
};
use indicator_core::ATR;

/// ATR Evaluator.
/// Generates volatility-based signals using ATR expansion/contraction.
#[derive(Debug, Clone)]
pub struct ATREvaluator {
    period_range: ParamRange,
    expansion_threshold: f64,
}

impl ATREvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self {
            period_range,
            expansion_threshold: 1.5, // Signal when ATR is 1.5x its average
        }
    }

    pub fn with_expansion_threshold(mut self, threshold: f64) -> Self {
        self.expansion_threshold = threshold;
        self
    }
}

impl IndicatorEvaluator for ATREvaluator {
    fn name(&self) -> &str {
        "ATR"
    }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get_usize("period").ok_or_else(|| {
            OptimizerError::InvalidConfig("ATR requires 'period' parameter".to_string())
        })?;

        if data.close.len() < period + 1 {
            return Err(OptimizerError::InsufficientData {
                required: period + 1,
                got: data.close.len(),
            });
        }

        let atr = ATR::new(period);
        let values = atr.calculate(&data.high, &data.low, &data.close);

        // Compute average ATR for volatility regime detection
        let valid_atr: Vec<f64> = values.iter().filter(|x| !x.is_nan()).cloned().collect();
        let avg_atr = if valid_atr.is_empty() {
            0.0
        } else {
            valid_atr.iter().sum::<f64>() / valid_atr.len() as f64
        };

        // ATR signals: expansion = potential breakout opportunity
        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v.is_nan() || avg_atr == 0.0 {
                Signal::Hold
            } else if v > avg_atr * self.expansion_threshold {
                Signal::Hold // High volatility = hold position (or could signal trend)
            } else {
                Signal::Hold // ATR is typically used for sizing, not direction
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimizer_spi::Timeframe;

    #[test]
    fn test_atr_evaluator() {
        let evaluator = ATREvaluator::new(ParamRange::new(10, 20, 5));

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

        let params = IndicatorParams::new("ATR").with_param("period", 14.0);
        let result = evaluator.evaluate(&params, &data).unwrap();

        assert_eq!(result.indicator_values.len(), 100);
    }
}
