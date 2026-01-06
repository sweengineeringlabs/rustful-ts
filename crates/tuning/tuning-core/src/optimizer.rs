//! Grid search optimizer implementation.

use tuning_spi::{
    Optimizer, ObjectiveFunction, Validator, OptimizationResult,
    OptimizedParams, Result, TuningError, OptimizationMethod,
};
use indicator_api::IndicatorType;
use indicator_core::{SMA, EMA, RSI, MACD, BollingerBands, ATR, Stochastic};
use indicator_spi::{TechnicalIndicator, OHLCVSeries, IndicatorSignal, SignalIndicator};

/// Grid search optimizer.
#[derive(Debug, Clone)]
pub struct GridSearchOptimizer {
    indicators: Vec<IndicatorType>,
    top_n: usize,
    verbose: bool,
}

impl GridSearchOptimizer {
    pub fn new(indicators: Vec<IndicatorType>) -> Self {
        Self {
            indicators,
            top_n: 10,
            verbose: false,
        }
    }

    pub fn with_top_n(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Generate all parameter combinations.
    fn generate_combinations(&self) -> Vec<Vec<(String, Vec<f64>)>> {
        let mut all_combos = Vec::new();

        for indicator in &self.indicators {
            let combos = self.indicator_combinations(indicator);
            all_combos.push(combos);
        }

        // Cross-product of all indicators
        self.cross_product(&all_combos)
    }

    fn indicator_combinations(&self, indicator: &IndicatorType) -> Vec<(String, Vec<f64>)> {
        match indicator {
            IndicatorType::SMA { period } => {
                period.values().iter()
                    .map(|&p| ("SMA".to_string(), vec![p as f64]))
                    .collect()
            }
            IndicatorType::EMA { period } => {
                period.values().iter()
                    .map(|&p| ("EMA".to_string(), vec![p as f64]))
                    .collect()
            }
            IndicatorType::RSI { period } => {
                period.values().iter()
                    .map(|&p| ("RSI".to_string(), vec![p as f64]))
                    .collect()
            }
            IndicatorType::MACD { fast, slow, signal } => {
                let mut combos = Vec::new();
                for &f in &fast.values() {
                    for &s in &slow.values() {
                        for &sig in &signal.values() {
                            combos.push(("MACD".to_string(), vec![f as f64, s as f64, sig as f64]));
                        }
                    }
                }
                combos
            }
            IndicatorType::Bollinger { period, std_dev } => {
                let mut combos = Vec::new();
                for &p in &period.values() {
                    for &sd in &std_dev.values() {
                        combos.push(("Bollinger".to_string(), vec![p as f64, sd]));
                    }
                }
                combos
            }
            IndicatorType::ATR { period } => {
                period.values().iter()
                    .map(|&p| ("ATR".to_string(), vec![p as f64]))
                    .collect()
            }
            IndicatorType::Stochastic { k_period, d_period } => {
                let mut combos = Vec::new();
                for &k in &k_period.values() {
                    for &d in &d_period.values() {
                        combos.push(("Stochastic".to_string(), vec![k as f64, d as f64]));
                    }
                }
                combos
            }
            _ => Vec::new(),
        }
    }

    fn cross_product(&self, sets: &[Vec<(String, Vec<f64>)>]) -> Vec<Vec<(String, Vec<f64>)>> {
        if sets.is_empty() {
            return vec![vec![]];
        }

        if sets.len() == 1 {
            return sets[0].iter()
                .map(|x| vec![x.clone()])
                .collect();
        }

        let first = &sets[0];
        let rest = self.cross_product(&sets[1..]);

        let mut result = Vec::new();
        for item in first {
            for r in &rest {
                let mut combo = vec![item.clone()];
                combo.extend(r.clone());
                result.push(combo);
            }
        }

        result
    }

    /// Compute signals for given parameters.
    fn compute_signals(&self, params: &[(String, Vec<f64>)], data: &OHLCVSeries) -> Vec<f64> {
        let mut all_signals: Vec<Vec<f64>> = Vec::new();

        for (indicator_type, param_values) in params {
            let signals = match indicator_type.as_str() {
                "SMA" => self.sma_signals(param_values[0] as usize, data),
                "EMA" => self.ema_signals(param_values[0] as usize, data),
                "RSI" => self.rsi_signals(param_values[0] as usize, data),
                "MACD" => self.macd_signals(
                    param_values[0] as usize,
                    param_values[1] as usize,
                    param_values[2] as usize,
                    data,
                ),
                "Bollinger" => self.bollinger_signals(
                    param_values[0] as usize,
                    param_values[1],
                    data,
                ),
                "Stochastic" => self.stochastic_signals(
                    param_values[0] as usize,
                    param_values[1] as usize,
                    data,
                ),
                _ => vec![0.0; data.len()],
            };
            all_signals.push(signals);
        }

        // Combine signals (average for simplicity)
        if all_signals.is_empty() {
            return vec![0.0; data.len()];
        }

        let len = all_signals[0].len();
        (0..len)
            .map(|i| {
                let sum: f64 = all_signals.iter().map(|s| s[i]).sum();
                sum / all_signals.len() as f64
            })
            .collect()
    }

    fn sma_signals(&self, period: usize, data: &OHLCVSeries) -> Vec<f64> {
        let sma = SMA::new(period);
        let values = sma.calculate(&data.close);

        // Signal: price above SMA = bullish (1), below = bearish (-1)
        data.close.iter()
            .zip(values.iter())
            .map(|(&price, &sma_val)| {
                if sma_val.is_nan() {
                    0.0
                } else if price > sma_val {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }

    fn ema_signals(&self, period: usize, data: &OHLCVSeries) -> Vec<f64> {
        let ema = EMA::new(period);
        let values = ema.calculate(&data.close);

        data.close.iter()
            .zip(values.iter())
            .map(|(&price, &ema_val)| {
                if ema_val.is_nan() {
                    0.0
                } else if price > ema_val {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    }

    fn rsi_signals(&self, period: usize, data: &OHLCVSeries) -> Vec<f64> {
        let rsi = RSI::new(period);
        match rsi.signals(data) {
            Ok(signals) => signals.iter()
                .map(|s| s.to_numeric())
                .collect(),
            Err(_) => vec![0.0; data.len()],
        }
    }

    fn macd_signals(&self, fast: usize, slow: usize, signal: usize, data: &OHLCVSeries) -> Vec<f64> {
        let macd = MACD::new(fast, slow, signal);
        match macd.signals(data) {
            Ok(signals) => signals.iter()
                .map(|s| s.to_numeric())
                .collect(),
            Err(_) => vec![0.0; data.len()],
        }
    }

    fn bollinger_signals(&self, period: usize, std_dev: f64, data: &OHLCVSeries) -> Vec<f64> {
        let bb = BollingerBands::new(period, std_dev);
        match bb.signals(data) {
            Ok(signals) => signals.iter()
                .map(|s| s.to_numeric())
                .collect(),
            Err(_) => vec![0.0; data.len()],
        }
    }

    fn stochastic_signals(&self, k_period: usize, d_period: usize, data: &OHLCVSeries) -> Vec<f64> {
        let stoch = Stochastic::new(k_period, d_period);
        match stoch.signals(data) {
            Ok(signals) => signals.iter()
                .map(|s| s.to_numeric())
                .collect(),
            Err(_) => vec![0.0; data.len()],
        }
    }

    fn params_to_optimized(&self, params: &[(String, Vec<f64>)]) -> Vec<OptimizedParams> {
        params.iter().map(|(name, values)| {
            let mut opt = OptimizedParams::new(name);
            match name.as_str() {
                "SMA" | "EMA" | "RSI" | "ATR" => {
                    opt = opt.with_param("period", values[0]);
                }
                "MACD" => {
                    opt = opt.with_param("fast", values[0])
                        .with_param("slow", values[1])
                        .with_param("signal", values[2]);
                }
                "Bollinger" => {
                    opt = opt.with_param("period", values[0])
                        .with_param("std_dev", values[1]);
                }
                "Stochastic" => {
                    opt = opt.with_param("k_period", values[0])
                        .with_param("d_period", values[1]);
                }
                _ => {}
            }
            opt
        }).collect()
    }
}

impl Optimizer for GridSearchOptimizer {
    fn optimize(
        &self,
        data: &[f64],
        objective: &dyn ObjectiveFunction,
        validator: &dyn Validator,
    ) -> Result<OptimizationResult> {
        let combinations = self.generate_combinations();
        let splits = validator.splits(data.len())?;

        if combinations.is_empty() {
            return Err(TuningError::InvalidConfig("No parameter combinations".into()));
        }

        // Calculate returns
        let returns: Vec<f64> = (1..data.len())
            .map(|i| (data[i] - data[i - 1]) / data[i - 1])
            .collect();

        let ohlcv = OHLCVSeries::from_close(data.to_vec());

        let mut results: Vec<(Vec<(String, Vec<f64>)>, f64, Option<f64>)> = Vec::new();

        for combo in &combinations {
            let signals = self.compute_signals(combo, &ohlcv);

            let mut in_sample_scores = Vec::new();
            let mut out_sample_scores = Vec::new();

            for split in &splits {
                // In-sample evaluation
                let train_signals = &signals[split.train_start..split.train_end.min(signals.len())];
                let train_returns = &returns[split.train_start..split.train_end.min(returns.len())];

                if !train_signals.is_empty() && train_signals.len() == train_returns.len() {
                    let is_score = objective.compute(train_signals, train_returns);
                    if is_score.is_finite() {
                        in_sample_scores.push(is_score);
                    }
                }

                // Out-of-sample evaluation
                if split.test_start < split.test_end {
                    let test_signals = &signals[split.test_start..split.test_end.min(signals.len())];
                    let test_returns = &returns[split.test_start..split.test_end.min(returns.len())];

                    if !test_signals.is_empty() && test_signals.len() == test_returns.len() {
                        let oos_score = objective.compute(test_signals, test_returns);
                        if oos_score.is_finite() {
                            out_sample_scores.push(oos_score);
                        }
                    }
                }
            }

            if !in_sample_scores.is_empty() {
                let avg_is = in_sample_scores.iter().sum::<f64>() / in_sample_scores.len() as f64;
                let avg_oos = if out_sample_scores.is_empty() {
                    None
                } else {
                    Some(out_sample_scores.iter().sum::<f64>() / out_sample_scores.len() as f64)
                };
                results.push((combo.clone(), avg_is, avg_oos));
            }
        }

        if results.is_empty() {
            return Err(TuningError::OptimizationFailed("No valid results".into()));
        }

        // Sort by score
        let is_maximize = objective.objective_type().is_maximize();
        results.sort_by(|a, b| {
            if is_maximize {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        let best = &results[0];
        let top_results: Vec<_> = results.iter()
            .take(self.top_n)
            .map(|(params, score, _)| (self.params_to_optimized(params), *score))
            .collect();

        let robustness = best.2.map(|oos| {
            if best.1.abs() > 1e-10 {
                oos / best.1
            } else {
                1.0
            }
        });

        Ok(OptimizationResult {
            best_params: self.params_to_optimized(&best.0),
            best_score: best.1,
            oos_score: best.2,
            evaluations: combinations.len(),
            robustness,
            top_results,
        })
    }

    fn method(&self) -> OptimizationMethod {
        OptimizationMethod::GridSearch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SharpeRatio, TrainTestValidator};
    use indicator_api::ParamRange;

    #[test]
    fn test_grid_search_basic() {
        let indicators = vec![
            IndicatorType::SMA { period: ParamRange::new(5, 15, 5) },
        ];

        let optimizer = GridSearchOptimizer::new(indicators);
        let data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();

        let objective = SharpeRatio::new();
        let validator = TrainTestValidator::new(0.7);

        let result = optimizer.optimize(&data, &objective, &validator);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(!result.best_params.is_empty());
        assert!(result.evaluations > 0);
    }
}
