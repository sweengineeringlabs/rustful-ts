//! Grid search optimizer implementation.

use optimizer_spi::{
    Optimizer, ObjectiveFunction, Validator, OptimizationResult, OptimizedParams,
    Result, OptimizationMethod, OptimizerError, ValidationSplit,
    IndicatorEvaluator, IndicatorParams, MarketData, Signal, ParamRange, FloatParamRange,
};
use rayon::prelude::*;
use std::sync::Arc;

/// Grid search optimizer for indicators.
#[derive(Debug, Clone, Default)]
pub struct GridSearchOptimizer;

impl GridSearchOptimizer {
    pub fn new() -> Self {
        Self
    }
}

impl Optimizer for GridSearchOptimizer {
    fn optimize(
        &self,
        data: &[f64],
        objective: &dyn ObjectiveFunction,
        validator: &dyn Validator,
    ) -> Result<OptimizationResult> {
        // Basic implementation for compatibility
        let _splits = validator.splits(data.len())?;
        let mut result = OptimizationResult::default();
        result.evaluations = 1;
        result.best_score = 0.0;
        Ok(result)
    }

    fn method(&self) -> OptimizationMethod {
        OptimizationMethod::GridSearch
    }
}

/// Configuration for indicator grid search.
#[derive(Debug, Clone)]
pub struct GridSearchConfig {
    pub parallel: bool,
    pub top_n: usize,
}

impl Default for GridSearchConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            top_n: 10,
        }
    }
}

/// Indicator-aware grid search optimizer.
pub struct IndicatorGridSearch {
    config: GridSearchConfig,
}

impl IndicatorGridSearch {
    pub fn new() -> Self {
        Self {
            config: GridSearchConfig::default(),
        }
    }

    pub fn with_config(config: GridSearchConfig) -> Self {
        Self { config }
    }

    /// Run grid search optimization on an indicator.
    pub fn optimize(
        &self,
        evaluator: &dyn IndicatorEvaluator,
        data: &MarketData,
        objective: &dyn ObjectiveFunction,
        validation_splits: &[ValidationSplit],
    ) -> Result<OptimizationResult> {
        let param_space = evaluator.parameter_space();
        let float_param_space = evaluator.float_parameter_space();

        if param_space.is_empty() && float_param_space.is_empty() {
            return Err(OptimizerError::InvalidConfig(
                "Evaluator has no parameter space defined".to_string()
            ));
        }

        // Generate all parameter combinations
        let combinations = generate_combinations(&param_space, &float_param_space);

        if combinations.is_empty() {
            return Err(OptimizerError::InvalidConfig(
                "No parameter combinations to evaluate".to_string()
            ));
        }

        // Compute returns for objective calculation
        let returns = data.returns();
        if returns.is_empty() {
            return Err(OptimizerError::InsufficientData {
                required: 2,
                got: data.close.len(),
            });
        }

        let indicator_name = evaluator.name().to_string();

        // Evaluate all combinations (in parallel if configured)
        let results: Vec<(IndicatorParams, f64, Option<f64>)> = if self.config.parallel {
            combinations
                .par_iter()
                .filter_map(|params| {
                    self.evaluate_params(
                        evaluator,
                        data,
                        &returns,
                        &indicator_name,
                        params,
                        objective,
                        validation_splits,
                    ).ok()
                })
                .collect()
        } else {
            combinations
                .iter()
                .filter_map(|params| {
                    self.evaluate_params(
                        evaluator,
                        data,
                        &returns,
                        &indicator_name,
                        params,
                        objective,
                        validation_splits,
                    ).ok()
                })
                .collect()
        };

        if results.is_empty() {
            return Err(OptimizerError::OptimizationFailed(
                "No valid parameter combinations found".to_string()
            ));
        }

        // Find best result
        let (best_params, best_score, oos_score) = results
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .clone();

        // Collect top N results
        let mut sorted_results = results.clone();
        sorted_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_results: Vec<(Vec<OptimizedParams>, f64)> = sorted_results
            .into_iter()
            .take(self.config.top_n)
            .map(|(params, score, _)| {
                let opt_params = OptimizedParams {
                    indicator_type: params.indicator_name.clone(),
                    params: params.params.clone(),
                };
                (vec![opt_params], score)
            })
            .collect();

        // Convert best params to OptimizedParams
        let optimized = OptimizedParams {
            indicator_type: best_params.indicator_name.clone(),
            params: best_params.params.clone(),
        };

        // Calculate robustness if we have OOS score
        let robustness = oos_score.and_then(|oos| {
            if best_score.abs() > 1e-10 {
                Some(oos / best_score)
            } else {
                None
            }
        });

        Ok(OptimizationResult {
            best_params: vec![optimized],
            best_score,
            oos_score,
            evaluations: results.len(),
            robustness,
            top_results,
        })
    }

    /// Evaluate a single parameter combination.
    fn evaluate_params(
        &self,
        evaluator: &dyn IndicatorEvaluator,
        data: &MarketData,
        returns: &[f64],
        indicator_name: &str,
        params: &[(String, f64)],
        objective: &dyn ObjectiveFunction,
        validation_splits: &[ValidationSplit],
    ) -> Result<(IndicatorParams, f64, Option<f64>)> {
        let ind_params = params.iter().fold(
            IndicatorParams::new(indicator_name),
            |p, (name, val)| p.with_param(name, *val)
        );

        if validation_splits.is_empty() {
            // No validation - use full dataset
            let eval_result = evaluator.evaluate(&ind_params, data)?;
            let signals: Vec<f64> = eval_result.signals.iter()
                .map(|s| s.as_position())
                .collect();

            // Align signals with returns (signals[i] predicts returns[i])
            let aligned_signals = if signals.len() > returns.len() {
                signals[..returns.len()].to_vec()
            } else {
                signals
            };
            let aligned_returns = if returns.len() > aligned_signals.len() {
                returns[..aligned_signals.len()].to_vec()
            } else {
                returns.to_vec()
            };

            let score = objective.compute(&aligned_signals, &aligned_returns);
            return Ok((ind_params, score, None));
        }

        // With validation splits - compute in-sample and out-of-sample scores
        let mut is_scores = Vec::new();
        let mut oos_scores = Vec::new();

        for split in validation_splits {
            // Training data
            let train_data = data.slice(split.train_start, split.train_end);
            let train_returns = returns[split.train_start..split.train_end.min(returns.len())].to_vec();

            if let Ok(train_result) = evaluator.evaluate(&ind_params, &train_data) {
                let train_signals: Vec<f64> = train_result.signals.iter()
                    .map(|s| s.as_position())
                    .collect();

                let aligned_len = train_signals.len().min(train_returns.len());
                let is_score = objective.compute(
                    &train_signals[..aligned_len],
                    &train_returns[..aligned_len]
                );
                is_scores.push(is_score);
            }

            // Test data
            let test_data = data.slice(split.test_start, split.test_end);
            let test_returns = returns[split.test_start..split.test_end.min(returns.len())].to_vec();

            if let Ok(test_result) = evaluator.evaluate(&ind_params, &test_data) {
                let test_signals: Vec<f64> = test_result.signals.iter()
                    .map(|s| s.as_position())
                    .collect();

                let aligned_len = test_signals.len().min(test_returns.len());
                let oos_score = objective.compute(
                    &test_signals[..aligned_len],
                    &test_returns[..aligned_len]
                );
                oos_scores.push(oos_score);
            }
        }

        if is_scores.is_empty() {
            return Err(OptimizerError::OptimizationFailed(
                "No valid in-sample scores".to_string()
            ));
        }

        let avg_is = is_scores.iter().sum::<f64>() / is_scores.len() as f64;
        let avg_oos = if oos_scores.is_empty() {
            None
        } else {
            Some(oos_scores.iter().sum::<f64>() / oos_scores.len() as f64)
        };

        Ok((ind_params, avg_is, avg_oos))
    }
}

impl Default for IndicatorGridSearch {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate all parameter combinations from parameter space.
fn generate_combinations(
    param_space: &[(String, ParamRange)],
    float_param_space: &[(String, FloatParamRange)],
) -> Vec<Vec<(String, f64)>> {
    let mut all_params: Vec<(String, Vec<f64>)> = Vec::new();

    // Add integer parameters
    for (name, range) in param_space {
        let values: Vec<f64> = range.values().iter().map(|&v| v as f64).collect();
        all_params.push((name.clone(), values));
    }

    // Add float parameters
    for (name, range) in float_param_space {
        all_params.push((name.clone(), range.values()));
    }

    if all_params.is_empty() {
        return Vec::new();
    }

    // Generate Cartesian product
    let mut combinations: Vec<Vec<(String, f64)>> = vec![vec![]];

    for (name, values) in all_params {
        let mut new_combinations = Vec::new();
        for combo in &combinations {
            for val in &values {
                let mut new_combo = combo.clone();
                new_combo.push((name.clone(), *val));
                new_combinations.push(new_combo);
            }
        }
        combinations = new_combinations;
    }

    combinations
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimizer_spi::Timeframe;
    use crate::evaluators::RSIEvaluator;
    use crate::objective::SharpeRatio;

    #[test]
    fn test_generate_combinations() {
        let param_space = vec![
            ("period".to_string(), ParamRange::new(10, 20, 5)),
        ];
        let float_space = vec![];

        let combos = generate_combinations(&param_space, &float_space);
        assert_eq!(combos.len(), 3); // 10, 15, 20
    }

    #[test]
    fn test_generate_combinations_multi() {
        let param_space = vec![
            ("fast".to_string(), ParamRange::new(10, 20, 10)),
            ("slow".to_string(), ParamRange::new(30, 50, 10)),
        ];
        let float_space = vec![];

        let combos = generate_combinations(&param_space, &float_space);
        assert_eq!(combos.len(), 6); // 2 * 3
    }

    #[test]
    fn test_indicator_grid_search() {
        let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
        let objective = SharpeRatio::new();

        // Create test data
        let mut data = MarketData::new("TEST", Timeframe::D1);
        for i in 0..200 {
            data.close.push(100.0 + (i as f64).sin() * 10.0);
            data.open.push(100.0 + (i as f64).sin() * 10.0);
            data.high.push(110.0 + (i as f64).sin() * 10.0);
            data.low.push(90.0 + (i as f64).sin() * 10.0);
            data.volume.push(1000.0);
            data.timestamps.push(i as i64);
        }

        let grid = IndicatorGridSearch::new();
        let result = grid.optimize(&evaluator, &data, &objective, &[]).unwrap();

        assert!(result.evaluations >= 3);
        assert!(!result.best_params.is_empty());
    }
}
