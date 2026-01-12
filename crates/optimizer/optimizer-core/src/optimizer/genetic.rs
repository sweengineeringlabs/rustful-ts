//! Genetic algorithm optimizer implementation.

use optimizer_spi::{
    Optimizer, ObjectiveFunction, Validator, OptimizationResult, OptimizedParams,
    Result, OptimizerError, OptimizationMethod, ValidationSplit,
    IndicatorEvaluator, IndicatorParams, MarketData, ParamRange, FloatParamRange,
};
use rand::prelude::*;
use rayon::prelude::*;

/// Genetic algorithm optimizer (placeholder for Optimizer trait).
#[derive(Debug, Clone)]
pub struct GeneticOptimizer {
    population_size: usize,
    generations: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    elite_count: usize,
}

impl GeneticOptimizer {
    pub fn new(
        population_size: usize,
        generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> Self {
        Self {
            population_size,
            generations,
            mutation_rate,
            crossover_rate,
            elite_count: (population_size / 10).max(1),
        }
    }

    pub fn with_elite_count(mut self, count: usize) -> Self {
        self.elite_count = count;
        self
    }
}

impl Optimizer for GeneticOptimizer {
    fn optimize(
        &self,
        _data: &[f64],
        _objective: &dyn ObjectiveFunction,
        _validator: &dyn Validator,
    ) -> Result<OptimizationResult> {
        Err(OptimizerError::OptimizationFailed(
            "Use IndicatorGeneticOptimizer for indicator optimization".into()
        ))
    }

    fn method(&self) -> OptimizationMethod {
        OptimizationMethod::GeneticAlgorithm {
            population: self.population_size,
            generations: self.generations,
            mutation_rate: self.mutation_rate,
            crossover_rate: self.crossover_rate,
        }
    }
}

// ============================================================================
// Indicator-aware Genetic Optimizer
// ============================================================================

/// Configuration for genetic algorithm optimization.
#[derive(Debug, Clone)]
pub struct GeneticConfig {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elite_count: usize,
    pub tournament_size: usize,
    pub parallel: bool,
}

impl Default for GeneticConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elite_count: 5,
            tournament_size: 3,
            parallel: true,
        }
    }
}

/// Individual in the genetic population.
#[derive(Debug, Clone)]
struct Individual {
    /// Parameter values as (name, value, min, max, is_integer)
    genes: Vec<(String, f64, f64, f64, bool)>,
    fitness: f64,
    oos_fitness: Option<f64>,
}

impl Individual {
    fn to_params(&self, indicator_name: &str) -> IndicatorParams {
        self.genes.iter().fold(
            IndicatorParams::new(indicator_name),
            |p, (name, val, _, _, _)| p.with_param(name, *val)
        )
    }
}

/// Indicator-aware genetic algorithm optimizer.
pub struct IndicatorGeneticOptimizer {
    config: GeneticConfig,
}

impl IndicatorGeneticOptimizer {
    pub fn new() -> Self {
        Self {
            config: GeneticConfig::default(),
        }
    }

    pub fn with_config(config: GeneticConfig) -> Self {
        Self { config }
    }

    /// Run genetic algorithm optimization on an indicator.
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

        // Build parameter bounds
        let param_bounds: Vec<(String, f64, f64, bool)> = self.build_param_bounds(&param_space, &float_param_space);

        // Initialize population
        let mut population = self.initialize_population(&param_bounds);

        // Evaluate initial population
        self.evaluate_population(
            &mut population,
            evaluator,
            data,
            &returns,
            &indicator_name,
            objective,
            validation_splits,
        )?;

        let mut best_individual = population.iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap();

        let mut all_evaluated: Vec<Individual> = population.clone();
        let mut total_evaluations = population.len();

        // Evolution loop
        for _gen in 0..self.config.generations {
            // Sort by fitness (descending)
            population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

            // Keep elites
            let elites: Vec<Individual> = population.iter()
                .take(self.config.elite_count)
                .cloned()
                .collect();

            // Generate new population
            let mut new_population = elites.clone();

            while new_population.len() < self.config.population_size {
                // Tournament selection
                let parent1 = self.tournament_select(&population);
                let parent2 = self.tournament_select(&population);

                // Crossover
                let (mut child1, mut child2) = self.crossover(&parent1, &parent2, &param_bounds);

                // Mutation
                self.mutate(&mut child1, &param_bounds);
                self.mutate(&mut child2, &param_bounds);

                new_population.push(child1);
                if new_population.len() < self.config.population_size {
                    new_population.push(child2);
                }
            }

            population = new_population;

            // Evaluate new population (skip elites which already have fitness)
            self.evaluate_population(
                &mut population[self.config.elite_count..].to_vec().as_mut(),
                evaluator,
                data,
                &returns,
                &indicator_name,
                objective,
                validation_splits,
            )?;

            // Note: elites keep their fitness, non-elites were evaluated above

            total_evaluations += population.len() - self.config.elite_count;

            // Update best
            if let Some(gen_best) = population.iter()
                .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            {
                if gen_best.fitness > best_individual.fitness {
                    best_individual = gen_best.clone();
                }
            }

            all_evaluated.extend(population.clone());
        }

        // Build result
        let optimized = OptimizedParams {
            indicator_type: indicator_name.clone(),
            params: best_individual.genes.iter()
                .map(|(name, val, _, _, is_int)| {
                    if *is_int {
                        (name.clone(), val.round())
                    } else {
                        (name.clone(), *val)
                    }
                })
                .collect(),
        };

        // Calculate robustness
        let robustness = best_individual.oos_fitness.and_then(|oos| {
            if best_individual.fitness.abs() > 1e-10 {
                Some(oos / best_individual.fitness)
            } else {
                None
            }
        });

        // Get top N results
        all_evaluated.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        all_evaluated.dedup_by(|a, b| {
            a.genes.iter().zip(b.genes.iter()).all(|(x, y)| (x.1 - y.1).abs() < 1e-10)
        });

        let top_results: Vec<(Vec<OptimizedParams>, f64)> = all_evaluated.iter()
            .take(10)
            .map(|ind| {
                let opt = OptimizedParams {
                    indicator_type: indicator_name.clone(),
                    params: ind.genes.iter()
                        .map(|(name, val, _, _, is_int)| {
                            if *is_int { (name.clone(), val.round()) } else { (name.clone(), *val) }
                        })
                        .collect(),
                };
                (vec![opt], ind.fitness)
            })
            .collect();

        Ok(OptimizationResult {
            best_params: vec![optimized],
            best_score: best_individual.fitness,
            oos_score: best_individual.oos_fitness,
            evaluations: total_evaluations,
            robustness,
            top_results,
        })
    }

    fn build_param_bounds(
        &self,
        param_space: &[(String, ParamRange)],
        float_param_space: &[(String, FloatParamRange)],
    ) -> Vec<(String, f64, f64, bool)> {
        let mut bounds = Vec::new();

        for (name, range) in param_space {
            bounds.push((name.clone(), range.min as f64, range.max as f64, true));
        }

        for (name, range) in float_param_space {
            bounds.push((name.clone(), range.min, range.max, false));
        }

        bounds
    }

    fn initialize_population(&self, param_bounds: &[(String, f64, f64, bool)]) -> Vec<Individual> {
        let mut rng = thread_rng();
        let mut population = Vec::with_capacity(self.config.population_size);

        for _ in 0..self.config.population_size {
            let genes: Vec<(String, f64, f64, f64, bool)> = param_bounds.iter()
                .map(|(name, min, max, is_int)| {
                    let val = if *is_int {
                        rng.gen_range(*min..=*max).round()
                    } else {
                        rng.gen_range(*min..=*max)
                    };
                    (name.clone(), val, *min, *max, *is_int)
                })
                .collect();

            population.push(Individual {
                genes,
                fitness: f64::NEG_INFINITY,
                oos_fitness: None,
            });
        }

        population
    }

    fn evaluate_population(
        &self,
        population: &mut [Individual],
        evaluator: &dyn IndicatorEvaluator,
        data: &MarketData,
        returns: &[f64],
        indicator_name: &str,
        objective: &dyn ObjectiveFunction,
        validation_splits: &[ValidationSplit],
    ) -> Result<()> {
        if self.config.parallel {
            let results: Vec<(f64, Option<f64>)> = population.par_iter()
                .map(|ind| {
                    self.evaluate_individual(
                        ind, evaluator, data, returns, indicator_name, objective, validation_splits
                    ).unwrap_or((f64::NEG_INFINITY, None))
                })
                .collect();

            for (ind, (fitness, oos)) in population.iter_mut().zip(results.into_iter()) {
                ind.fitness = fitness;
                ind.oos_fitness = oos;
            }
        } else {
            for ind in population.iter_mut() {
                if let Ok((fitness, oos)) = self.evaluate_individual(
                    ind, evaluator, data, returns, indicator_name, objective, validation_splits
                ) {
                    ind.fitness = fitness;
                    ind.oos_fitness = oos;
                } else {
                    ind.fitness = f64::NEG_INFINITY;
                }
            }
        }

        Ok(())
    }

    fn evaluate_individual(
        &self,
        individual: &Individual,
        evaluator: &dyn IndicatorEvaluator,
        data: &MarketData,
        returns: &[f64],
        indicator_name: &str,
        objective: &dyn ObjectiveFunction,
        validation_splits: &[ValidationSplit],
    ) -> Result<(f64, Option<f64>)> {
        let ind_params = individual.to_params(indicator_name);

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

    fn tournament_select(&self, population: &[Individual]) -> Individual {
        let mut rng = thread_rng();
        let mut best: Option<&Individual> = None;

        for _ in 0..self.config.tournament_size {
            let idx = rng.gen_range(0..population.len());
            let candidate = &population[idx];

            if best.is_none() || candidate.fitness > best.unwrap().fitness {
                best = Some(candidate);
            }
        }

        best.unwrap().clone()
    }

    fn crossover(
        &self,
        parent1: &Individual,
        parent2: &Individual,
        _param_bounds: &[(String, f64, f64, bool)],
    ) -> (Individual, Individual) {
        let mut rng = thread_rng();

        if rng.gen::<f64>() > self.config.crossover_rate {
            return (parent1.clone(), parent2.clone());
        }

        // BLX-alpha crossover (blend crossover)
        let alpha = 0.5;
        let mut child1_genes = Vec::new();
        let mut child2_genes = Vec::new();

        for (g1, g2) in parent1.genes.iter().zip(parent2.genes.iter()) {
            let (name, v1, min, max, is_int) = g1;
            let v2 = g2.1;

            let d = (v1 - v2).abs();
            let low = (v1.min(v2) - alpha * d).max(*min);
            let high = (v1.max(v2) + alpha * d).min(*max);

            let c1_val = if *is_int {
                rng.gen_range(low..=high).round()
            } else {
                rng.gen_range(low..=high)
            };
            let c2_val = if *is_int {
                rng.gen_range(low..=high).round()
            } else {
                rng.gen_range(low..=high)
            };

            child1_genes.push((name.clone(), c1_val, *min, *max, *is_int));
            child2_genes.push((name.clone(), c2_val, *min, *max, *is_int));
        }

        (
            Individual { genes: child1_genes, fitness: f64::NEG_INFINITY, oos_fitness: None },
            Individual { genes: child2_genes, fitness: f64::NEG_INFINITY, oos_fitness: None },
        )
    }

    fn mutate(&self, individual: &mut Individual, _param_bounds: &[(String, f64, f64, bool)]) {
        let mut rng = thread_rng();

        for gene in individual.genes.iter_mut() {
            if rng.gen::<f64>() < self.config.mutation_rate {
                let (_, ref mut val, ref min, ref max, ref is_int) = gene;

                // Gaussian mutation
                let range = *max - *min;
                let sigma = range * 0.1; // 10% of range
                let delta: f64 = rng.sample(rand_distr::Normal::new(0.0, sigma).unwrap());

                *val = (*val + delta).clamp(*min, *max);
                if *is_int {
                    *val = val.round();
                }
            }
        }
    }
}

impl Default for IndicatorGeneticOptimizer {
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
    fn test_genetic_optimizer() {
        let evaluator = RSIEvaluator::new(ParamRange::new(5, 30, 1));
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

        let config = GeneticConfig {
            population_size: 20,
            generations: 10,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elite_count: 2,
            tournament_size: 3,
            parallel: false,
        };

        let optimizer = IndicatorGeneticOptimizer::with_config(config);
        let result = optimizer.optimize(&evaluator, &data, &objective, &[]).unwrap();

        assert!(result.evaluations > 0);
        assert!(!result.best_params.is_empty());
        println!("Genetic best: {:?} -> {:.4}", result.best_params, result.best_score);
    }

    #[test]
    fn test_genetic_with_validation() {
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

        let config = GeneticConfig {
            population_size: 15,
            generations: 5,
            ..Default::default()
        };

        let optimizer = IndicatorGeneticOptimizer::with_config(config);
        let result = optimizer.optimize(&evaluator, &data, &objective, &splits).unwrap();

        assert!(result.oos_score.is_some());
        println!("Genetic IS: {:.4}, OOS: {:.4}",
            result.best_score,
            result.oos_score.unwrap());
    }
}
