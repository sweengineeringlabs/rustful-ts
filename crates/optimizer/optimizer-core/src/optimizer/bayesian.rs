//! Bayesian optimization implementation.

use optimizer_spi::{
    Optimizer, ObjectiveFunction, Validator, OptimizationResult, OptimizedParams,
    Result, OptimizerError, OptimizationMethod, ValidationSplit,
    IndicatorEvaluator, IndicatorParams, MarketData, ParamRange, FloatParamRange,
};
use rand::prelude::*;

/// Bayesian optimizer (placeholder for Optimizer trait).
#[derive(Debug, Clone)]
pub struct BayesianOptimizer {
    iterations: usize,
    exploration_factor: f64,
    initial_samples: usize,
}

impl BayesianOptimizer {
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            exploration_factor: 2.576,
            initial_samples: 10,
        }
    }

    pub fn with_exploration(mut self, factor: f64) -> Self {
        self.exploration_factor = factor;
        self
    }

    pub fn with_initial_samples(mut self, samples: usize) -> Self {
        self.initial_samples = samples;
        self
    }
}

impl Optimizer for BayesianOptimizer {
    fn optimize(
        &self,
        _data: &[f64],
        _objective: &dyn ObjectiveFunction,
        _validator: &dyn Validator,
    ) -> Result<OptimizationResult> {
        Err(OptimizerError::OptimizationFailed(
            "Use IndicatorBayesianOptimizer for indicator optimization".into()
        ))
    }

    fn method(&self) -> OptimizationMethod {
        OptimizationMethod::Bayesian { iterations: self.iterations }
    }
}

// ============================================================================
// Indicator-aware Bayesian Optimizer
// ============================================================================

/// Configuration for Bayesian optimization.
#[derive(Debug, Clone)]
pub struct BayesianConfig {
    pub iterations: usize,
    pub initial_samples: usize,
    pub exploration_factor: f64,
    pub length_scale: f64,
    pub noise_variance: f64,
}

impl Default for BayesianConfig {
    fn default() -> Self {
        Self {
            iterations: 50,
            initial_samples: 10,
            exploration_factor: 2.0,  // UCB beta parameter
            length_scale: 1.0,
            noise_variance: 0.01,
        }
    }
}

/// Observed data point for the surrogate model.
#[derive(Debug, Clone)]
struct Observation {
    params: Vec<f64>,
    fitness: f64,
    oos_fitness: Option<f64>,
}

/// Simple Gaussian Process surrogate model.
struct GaussianProcess {
    observations: Vec<Observation>,
    length_scale: f64,
    noise_variance: f64,
    param_bounds: Vec<(f64, f64)>,
}

impl GaussianProcess {
    fn new(param_bounds: Vec<(f64, f64)>, length_scale: f64, noise_variance: f64) -> Self {
        Self {
            observations: Vec::new(),
            length_scale,
            noise_variance,
            param_bounds,
        }
    }

    fn add_observation(&mut self, params: Vec<f64>, fitness: f64, oos_fitness: Option<f64>) {
        self.observations.push(Observation { params, fitness, oos_fitness });
    }

    /// RBF (squared exponential) kernel.
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let mut sq_dist = 0.0;
        for (a, b) in x1.iter().zip(x2.iter()) {
            sq_dist += (a - b).powi(2);
        }
        (-sq_dist / (2.0 * self.length_scale.powi(2))).exp()
    }

    /// Predict mean and variance at a point.
    fn predict(&self, x: &[f64]) -> (f64, f64) {
        if self.observations.is_empty() {
            return (0.0, 1.0);
        }

        let n = self.observations.len();

        // Compute kernel matrix K
        let mut k_matrix: Vec<f64> = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                k_matrix[i * n + j] = self.kernel(&self.observations[i].params, &self.observations[j].params);
                if i == j {
                    k_matrix[i * n + j] += self.noise_variance;
                }
            }
        }

        // Compute k_star (kernel between x and all observations)
        let k_star: Vec<f64> = self.observations.iter()
            .map(|obs| self.kernel(x, &obs.params))
            .collect();

        // k_star_star (kernel at x with itself)
        let k_star_star = self.kernel(x, x) + self.noise_variance;

        // Solve K * alpha = y for alpha (using simple Cholesky-like approach)
        // For simplicity, use direct inversion approximation
        let y: Vec<f64> = self.observations.iter().map(|o| o.fitness).collect();

        // Simple matrix inverse approximation for small matrices
        let (alpha, k_inv) = self.solve_system(&k_matrix, &y, n);

        // Mean: k_star^T * alpha
        let mean: f64 = k_star.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();

        // Variance: k_star_star - k_star^T * K^-1 * k_star
        let mut var_reduction = 0.0;
        for i in 0..n {
            for j in 0..n {
                var_reduction += k_star[i] * k_inv[i * n + j] * k_star[j];
            }
        }
        let variance = (k_star_star - var_reduction).max(1e-10);

        (mean, variance)
    }

    /// Simple system solver (for small matrices).
    fn solve_system(&self, k_matrix: &[f64], y: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        // For numerical stability, add small regularization and use simple Gauss-Jordan
        let mut a: Vec<f64> = k_matrix.to_vec();
        let mut inv: Vec<f64> = vec![0.0; n * n];

        // Initialize inverse as identity
        for i in 0..n {
            inv[i * n + i] = 1.0;
        }

        // Add regularization
        for i in 0..n {
            a[i * n + i] += 1e-6;
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if a[k * n + i].abs() > a[max_row * n + i].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            for j in 0..n {
                a.swap(i * n + j, max_row * n + j);
                inv.swap(i * n + j, max_row * n + j);
            }

            let pivot = a[i * n + i];
            if pivot.abs() < 1e-10 {
                continue;
            }

            // Scale row
            for j in 0..n {
                a[i * n + j] /= pivot;
                inv[i * n + j] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = a[k * n + i];
                    for j in 0..n {
                        a[k * n + j] -= factor * a[i * n + j];
                        inv[k * n + j] -= factor * inv[i * n + j];
                    }
                }
            }
        }

        // Compute alpha = K^-1 * y
        let mut alpha = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                alpha[i] += inv[i * n + j] * y[j];
            }
        }

        (alpha, inv)
    }

    /// Upper Confidence Bound acquisition function.
    fn ucb(&self, x: &[f64], beta: f64) -> f64 {
        let (mean, variance) = self.predict(x);
        mean + beta * variance.sqrt()
    }

    /// Find next point to sample using random search on acquisition function.
    fn suggest_next(&self, beta: f64, n_candidates: usize) -> Vec<f64> {
        let mut rng = thread_rng();
        let mut best_x = Vec::new();
        let mut best_acq = f64::NEG_INFINITY;

        for _ in 0..n_candidates {
            let x: Vec<f64> = self.param_bounds.iter()
                .map(|(min, max)| rng.gen_range(*min..=*max))
                .collect();

            let acq = self.ucb(&x, beta);
            if acq > best_acq {
                best_acq = acq;
                best_x = x;
            }
        }

        best_x
    }
}

/// Indicator-aware Bayesian optimizer.
pub struct IndicatorBayesianOptimizer {
    config: BayesianConfig,
}

impl IndicatorBayesianOptimizer {
    pub fn new() -> Self {
        Self {
            config: BayesianConfig::default(),
        }
    }

    pub fn with_config(config: BayesianConfig) -> Self {
        Self { config }
    }

    /// Run Bayesian optimization on an indicator.
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

        let returns = data.returns();
        if returns.is_empty() {
            return Err(OptimizerError::InsufficientData {
                required: 2,
                got: data.close.len(),
            });
        }

        let indicator_name = evaluator.name().to_string();

        // Build parameter info
        let param_info: Vec<(String, f64, f64, bool)> = self.build_param_info(&param_space, &float_param_space);
        let param_bounds: Vec<(f64, f64)> = param_info.iter().map(|(_, min, max, _)| (*min, *max)).collect();

        // Normalize bounds for GP
        let normalized_bounds: Vec<(f64, f64)> = param_bounds.iter().map(|_| (0.0, 1.0)).collect();

        // Create Gaussian Process
        let mut gp = GaussianProcess::new(normalized_bounds, self.config.length_scale, self.config.noise_variance);

        let mut rng = thread_rng();
        let mut best_fitness = f64::NEG_INFINITY;
        let mut best_params: Vec<f64> = Vec::new();
        let mut best_oos: Option<f64> = None;
        let mut all_observations: Vec<Observation> = Vec::new();

        // Initial random samples
        for _ in 0..self.config.initial_samples {
            let params: Vec<f64> = param_info.iter()
                .map(|(_, min, max, is_int)| {
                    let val = rng.gen_range(*min..=*max);
                    if *is_int { val.round() } else { val }
                })
                .collect();

            if let Ok((fitness, oos)) = self.evaluate_params(
                &params, &param_info, evaluator, data, &returns, &indicator_name, objective, validation_splits
            ) {
                // Normalize params for GP
                let normalized: Vec<f64> = params.iter().zip(param_bounds.iter())
                    .map(|(p, (min, max))| (p - min) / (max - min).max(1e-10))
                    .collect();

                gp.add_observation(normalized, fitness, oos);
                all_observations.push(Observation { params: params.clone(), fitness, oos_fitness: oos });

                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_params = params;
                    best_oos = oos;
                }
            }
        }

        // Bayesian optimization loop
        for _ in 0..self.config.iterations {
            // Get next point to sample
            let normalized_next = gp.suggest_next(self.config.exploration_factor, 1000);

            // Denormalize parameters
            let next_params: Vec<f64> = normalized_next.iter().zip(param_info.iter())
                .map(|(norm, (_, min, max, is_int))| {
                    let val = min + norm * (max - min);
                    if *is_int { val.round() } else { val }
                })
                .collect();

            if let Ok((fitness, oos)) = self.evaluate_params(
                &next_params, &param_info, evaluator, data, &returns, &indicator_name, objective, validation_splits
            ) {
                gp.add_observation(normalized_next, fitness, oos);
                all_observations.push(Observation { params: next_params.clone(), fitness, oos_fitness: oos });

                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_params = next_params;
                    best_oos = oos;
                }
            }
        }

        // Build result
        let optimized = OptimizedParams {
            indicator_type: indicator_name.clone(),
            params: param_info.iter().zip(best_params.iter())
                .map(|((name, _, _, _), val)| (name.clone(), *val))
                .collect(),
        };

        // Calculate robustness
        let robustness = best_oos.and_then(|oos| {
            if best_fitness.abs() > 1e-10 {
                Some(oos / best_fitness)
            } else {
                None
            }
        });

        // Get top N results
        all_observations.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        all_observations.dedup_by(|a, b| {
            a.params.iter().zip(b.params.iter()).all(|(x, y)| (x - y).abs() < 1e-10)
        });

        let top_results: Vec<(Vec<OptimizedParams>, f64)> = all_observations.iter()
            .take(10)
            .map(|obs| {
                let opt = OptimizedParams {
                    indicator_type: indicator_name.clone(),
                    params: param_info.iter().zip(obs.params.iter())
                        .map(|((name, _, _, _), val)| (name.clone(), *val))
                        .collect(),
                };
                (vec![opt], obs.fitness)
            })
            .collect();

        Ok(OptimizationResult {
            best_params: vec![optimized],
            best_score: best_fitness,
            oos_score: best_oos,
            evaluations: all_observations.len(),
            robustness,
            top_results,
        })
    }

    fn build_param_info(
        &self,
        param_space: &[(String, ParamRange)],
        float_param_space: &[(String, FloatParamRange)],
    ) -> Vec<(String, f64, f64, bool)> {
        let mut info = Vec::new();

        for (name, range) in param_space {
            info.push((name.clone(), range.min as f64, range.max as f64, true));
        }

        for (name, range) in float_param_space {
            info.push((name.clone(), range.min, range.max, false));
        }

        info
    }

    fn evaluate_params(
        &self,
        params: &[f64],
        param_info: &[(String, f64, f64, bool)],
        evaluator: &dyn IndicatorEvaluator,
        data: &MarketData,
        returns: &[f64],
        indicator_name: &str,
        objective: &dyn ObjectiveFunction,
        validation_splits: &[ValidationSplit],
    ) -> Result<(f64, Option<f64>)> {
        let ind_params = params.iter().zip(param_info.iter()).fold(
            IndicatorParams::new(indicator_name),
            |p, (val, (name, _, _, _))| p.with_param(name, *val)
        );

        if validation_splits.is_empty() {
            let eval_result = evaluator.evaluate(&ind_params, data)?;
            let signals: Vec<f64> = eval_result.signals.iter()
                .map(|s| s.as_position())
                .collect();

            let aligned_len = signals.len().min(returns.len());
            let score = objective.compute(&signals[..aligned_len], &returns[..aligned_len]);
            return Ok((score, None));
        }

        // With validation
        let mut is_scores = Vec::new();
        let mut oos_scores = Vec::new();

        for split in validation_splits {
            let train_data = data.slice(split.train_start, split.train_end);
            let train_returns = &returns[split.train_start..split.train_end.min(returns.len())];

            if let Ok(train_result) = evaluator.evaluate(&ind_params, &train_data) {
                let train_signals: Vec<f64> = train_result.signals.iter()
                    .map(|s| s.as_position())
                    .collect();
                let aligned_len = train_signals.len().min(train_returns.len());
                let is_score = objective.compute(&train_signals[..aligned_len], &train_returns[..aligned_len]);
                is_scores.push(is_score);
            }

            let test_data = data.slice(split.test_start, split.test_end);
            let test_returns = &returns[split.test_start..split.test_end.min(returns.len())];

            if let Ok(test_result) = evaluator.evaluate(&ind_params, &test_data) {
                let test_signals: Vec<f64> = test_result.signals.iter()
                    .map(|s| s.as_position())
                    .collect();
                let aligned_len = test_signals.len().min(test_returns.len());
                let oos_score = objective.compute(&test_signals[..aligned_len], &test_returns[..aligned_len]);
                oos_scores.push(oos_score);
            }
        }

        if is_scores.is_empty() {
            return Err(OptimizerError::OptimizationFailed("No valid in-sample scores".into()));
        }

        let avg_is = is_scores.iter().sum::<f64>() / is_scores.len() as f64;
        let avg_oos = if oos_scores.is_empty() {
            None
        } else {
            Some(oos_scores.iter().sum::<f64>() / oos_scores.len() as f64)
        };

        Ok((avg_is, avg_oos))
    }
}

impl Default for IndicatorBayesianOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use optimizer_spi::Timeframe;
    use crate::evaluators::RSIEvaluator;
    use crate::objective::SharpeRatio;

    #[test]
    fn test_bayesian_optimizer() {
        let evaluator = RSIEvaluator::new(ParamRange::new(5, 25, 1));
        let objective = SharpeRatio::new();

        let mut data = MarketData::new("TEST", Timeframe::D1);
        for i in 0..200 {
            let trend = i as f64 * 0.1;
            let noise = (i as f64 * 0.5).sin() * 5.0;
            data.close.push(100.0 + trend + noise);
            data.open.push(100.0 + trend + noise - 0.5);
            data.high.push(100.0 + trend + noise + 2.0);
            data.low.push(100.0 + trend + noise - 2.0);
            data.volume.push(1000.0);
            data.timestamps.push(i as i64);
        }

        let config = BayesianConfig {
            iterations: 15,
            initial_samples: 5,
            exploration_factor: 2.0,
            length_scale: 0.5,
            noise_variance: 0.01,
        };

        let optimizer = IndicatorBayesianOptimizer::with_config(config);
        let result = optimizer.optimize(&evaluator, &data, &objective, &[]).unwrap();

        assert!(result.evaluations > 0);
        assert!(!result.best_params.is_empty());
        println!("Bayesian best: {:?} -> {:.4}", result.best_params, result.best_score);
    }

    #[test]
    fn test_bayesian_with_validation() {
        let evaluator = RSIEvaluator::new(ParamRange::new(10, 25, 1));
        let objective = SharpeRatio::new();

        let mut data = MarketData::new("TEST", Timeframe::D1);
        for i in 0..300 {
            let trend = i as f64 * 0.05;
            let noise = (i as f64 * 0.3).sin() * 3.0;
            data.close.push(100.0 + trend + noise);
            data.open.push(100.0 + trend + noise);
            data.high.push(100.0 + trend + noise + 1.0);
            data.low.push(100.0 + trend + noise - 1.0);
            data.volume.push(1000.0);
            data.timestamps.push(i as i64);
        }

        let splits = vec![
            ValidationSplit {
                train_start: 0,
                train_end: 200,
                test_start: 200,
                test_end: 300,
            },
        ];

        let config = BayesianConfig {
            iterations: 10,
            initial_samples: 5,
            ..Default::default()
        };

        let optimizer = IndicatorBayesianOptimizer::with_config(config);
        let result = optimizer.optimize(&evaluator, &data, &objective, &splits).unwrap();

        assert!(result.oos_score.is_some());
        println!("Bayesian IS: {:.4}, OOS: {:.4}",
            result.best_score,
            result.oos_score.unwrap());
    }

    #[test]
    fn test_gaussian_process() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut gp = GaussianProcess::new(bounds, 0.5, 0.01);

        // Add some observations
        gp.add_observation(vec![0.0, 0.0], 0.5, None);
        gp.add_observation(vec![1.0, 1.0], 0.8, None);
        gp.add_observation(vec![0.5, 0.5], 1.0, None);

        // Predict at a point
        let (mean, var) = gp.predict(&[0.5, 0.5]);
        println!("GP predict at [0.5, 0.5]: mean={:.4}, var={:.4}", mean, var);

        // The mean should be close to 1.0 since we observed 1.0 at that point
        assert!((mean - 1.0).abs() < 0.2);
        assert!(var > 0.0);
    }
}
