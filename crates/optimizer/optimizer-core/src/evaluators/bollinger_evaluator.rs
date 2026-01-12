//! Bollinger Bands Evaluator for optimization.

use optimizer_spi::{
    IndicatorEvaluator, IndicatorParams, EvaluationResult, MarketData, Signal,
    ParamRange, FloatParamRange, Result, OptimizerError,
};
use indicator_core::BollingerBands;

/// Bollinger Bands Evaluator with band touch signals.
#[derive(Debug, Clone)]
pub struct BollingerEvaluator {
    period_range: ParamRange,
    std_dev_range: FloatParamRange,
}

impl BollingerEvaluator {
    pub fn new(period_range: ParamRange, std_dev_range: FloatParamRange) -> Self {
        Self { period_range, std_dev_range }
    }
}

impl IndicatorEvaluator for BollingerEvaluator {
    fn name(&self) -> &str {
        "Bollinger"
    }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get_usize("period").ok_or_else(|| {
            OptimizerError::InvalidConfig("Bollinger requires 'period' parameter".to_string())
        })?;
        let std_dev = params.get("std_dev").ok_or_else(|| {
            OptimizerError::InvalidConfig("Bollinger requires 'std_dev' parameter".to_string())
        })?;

        if data.close.len() < period {
            return Err(OptimizerError::InsufficientData {
                required: period,
                got: data.close.len(),
            });
        }

        let bb = BollingerBands::new(period, std_dev);
        let (middle, upper, lower) = bb.calculate(&data.close);

        // Generate signals based on band touches
        let signals: Vec<Signal> = data.close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() {
                    Signal::Hold
                } else if price <= l {
                    Signal::Buy // Price at lower band = oversold
                } else if price >= u {
                    Signal::Sell // Price at upper band = overbought
                } else {
                    Signal::Hold
                }
            })
            .collect();

        // Use %B as indicator value
        let values: Vec<f64> = data.close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() || (u - l).abs() < 1e-10 {
                    f64::NAN
                } else {
                    (price - l) / (u - l)
                }
            })
            .collect();

        let mut result = EvaluationResult::new(params.clone());
        result.indicator_values = values;
        result.signals = signals;

        Ok(result)
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }

    fn float_parameter_space(&self) -> Vec<(String, FloatParamRange)> {
        vec![("std_dev".to_string(), self.std_dev_range.clone())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimizer_spi::Timeframe;

    #[test]
    fn test_bollinger_evaluator() {
        let evaluator = BollingerEvaluator::new(
            ParamRange::new(15, 25, 5),
            FloatParamRange::new(1.5, 2.5, 0.5),
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

        let params = IndicatorParams::new("Bollinger")
            .with_param("period", 20.0)
            .with_param("std_dev", 2.0);
        let result = evaluator.evaluate(&params, &data).unwrap();

        assert_eq!(result.signals.len(), 100);
    }
}
