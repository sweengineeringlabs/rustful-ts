//! Genetic algorithm optimizer implementation.

use tuning_spi::{
    Optimizer, ObjectiveFunction, Validator, OptimizationResult,
    OptimizedParams, Result, TuningError, OptimizationMethod,
};
use indicator_api::IndicatorType;

/// Genetic algorithm optimizer.
#[derive(Debug, Clone)]
pub struct GeneticOptimizer {
    indicators: Vec<IndicatorType>,
    population_size: usize,
    generations: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    elite_count: usize,
}

impl GeneticOptimizer {
    pub fn new(
        indicators: Vec<IndicatorType>,
        population_size: usize,
        generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> Self {
        Self {
            indicators,
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

    /// Get parameter bounds for an indicator.
    fn get_bounds(&self, indicator: &IndicatorType) -> Vec<(f64, f64)> {
        match indicator {
            IndicatorType::SMA { period } |
            IndicatorType::EMA { period } |
            IndicatorType::RSI { period } |
            IndicatorType::ATR { period } |
            IndicatorType::ROC { period } |
            IndicatorType::StdDev { period } => {
                vec![(period.min as f64, period.max as f64)]
            }
            IndicatorType::MACD { fast, slow, signal } => {
                vec![
                    (fast.min as f64, fast.max as f64),
                    (slow.min as f64, slow.max as f64),
                    (signal.min as f64, signal.max as f64),
                ]
            }
            IndicatorType::Bollinger { period, std_dev } => {
                vec![
                    (period.min as f64, period.max as f64),
                    (std_dev.min, std_dev.max),
                ]
            }
            IndicatorType::Stochastic { k_period, d_period } => {
                vec![
                    (k_period.min as f64, k_period.max as f64),
                    (d_period.min as f64, d_period.max as f64),
                ]
            }
        }
    }

    /// Random individual within bounds.
    fn random_individual(&self) -> Vec<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::SystemTime;

        // Simple random using system time
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0) as u64;

        let mut hasher = DefaultHasher::new();

        let mut individual = Vec::new();
        for (idx, indicator) in self.indicators.iter().enumerate() {
            for (param_idx, (min, max)) in self.get_bounds(indicator).iter().enumerate() {
                // Generate pseudo-random value
                (seed, idx, param_idx).hash(&mut hasher);
                let hash = hasher.finish();
                let random = (hash as f64) / (u64::MAX as f64);
                individual.push(min + random * (max - min));
            }
        }
        individual
    }

    /// Mutate an individual.
    fn mutate(&self, individual: &mut Vec<f64>, _rate: f64) {
        // Simple mutation: perturb each gene slightly
        for gene in individual.iter_mut() {
            *gene *= 1.0 + (rand_simple() - 0.5) * 0.2;
        }
    }

    /// Crossover two parents.
    fn crossover(&self, parent1: &[f64], parent2: &[f64]) -> Vec<f64> {
        // Uniform crossover
        parent1.iter()
            .zip(parent2.iter())
            .map(|(g1, g2)| {
                if rand_simple() < 0.5 { *g1 } else { *g2 }
            })
            .collect()
    }

    /// Tournament selection.
    fn tournament_select<'a>(&self, population: &'a [(Vec<f64>, f64)], tournament_size: usize) -> &'a Vec<f64> {
        let mut best_idx = (rand_simple() * population.len() as f64) as usize % population.len();
        let mut best_fitness = population[best_idx].1;

        for _ in 1..tournament_size {
            let idx = (rand_simple() * population.len() as f64) as usize % population.len();
            if population[idx].1 > best_fitness {
                best_idx = idx;
                best_fitness = population[idx].1;
            }
        }

        &population[best_idx].0
    }

    fn genes_to_params(&self, genes: &[f64]) -> Vec<OptimizedParams> {
        let mut params = Vec::new();
        let mut gene_idx = 0;

        for indicator in &self.indicators {
            let opt = match indicator {
                IndicatorType::SMA { .. } => {
                    let p = genes[gene_idx].round() as usize;
                    gene_idx += 1;
                    OptimizedParams::new("SMA").with_param("period", p as f64)
                }
                IndicatorType::EMA { .. } => {
                    let p = genes[gene_idx].round() as usize;
                    gene_idx += 1;
                    OptimizedParams::new("EMA").with_param("period", p as f64)
                }
                IndicatorType::RSI { .. } => {
                    let p = genes[gene_idx].round() as usize;
                    gene_idx += 1;
                    OptimizedParams::new("RSI").with_param("period", p as f64)
                }
                IndicatorType::MACD { .. } => {
                    let f = genes[gene_idx].round();
                    let s = genes[gene_idx + 1].round();
                    let sig = genes[gene_idx + 2].round();
                    gene_idx += 3;
                    OptimizedParams::new("MACD")
                        .with_param("fast", f)
                        .with_param("slow", s)
                        .with_param("signal", sig)
                }
                IndicatorType::Bollinger { .. } => {
                    let p = genes[gene_idx].round();
                    let sd = genes[gene_idx + 1];
                    gene_idx += 2;
                    OptimizedParams::new("Bollinger")
                        .with_param("period", p)
                        .with_param("std_dev", sd)
                }
                IndicatorType::Stochastic { .. } => {
                    let k = genes[gene_idx].round();
                    let d = genes[gene_idx + 1].round();
                    gene_idx += 2;
                    OptimizedParams::new("Stochastic")
                        .with_param("k_period", k)
                        .with_param("d_period", d)
                }
                _ => {
                    gene_idx += 1;
                    OptimizedParams::new("Unknown")
                }
            };
            params.push(opt);
        }

        params
    }
}

/// Simple pseudo-random number generator.
fn rand_simple() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    static mut COUNTER: u64 = 0;

    unsafe {
        COUNTER = COUNTER.wrapping_add(1);
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0) as u64;

        let mut hasher = DefaultHasher::new();
        (seed, COUNTER).hash(&mut hasher);
        (hasher.finish() as f64) / (u64::MAX as f64)
    }
}

impl Optimizer for GeneticOptimizer {
    fn optimize(
        &self,
        _data: &[f64],
        _objective: &dyn ObjectiveFunction,
        _validator: &dyn Validator,
    ) -> Result<OptimizationResult> {
        // Simplified implementation - full version would integrate with indicator computation
        Err(TuningError::OptimizationFailed(
            "Genetic optimization requires full indicator integration".into()
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
