//! NAS100 optimization example - runs all optimizers in parallel.

use optimizer_core::{
    optimizer::{
        ParallelOptimizerRunner, GridSearchConfig, GeneticConfig, BayesianConfig,
    },
    evaluators::{RSIEvaluator, MACDEvaluator, BollingerEvaluator},
    objective::SharpeRatio,
    datasource::FixtureDataSource,
};
use optimizer_spi::{DataSource, ParamRange, FloatParamRange, Timeframe, ValidationSplit};
use std::time::Instant;

fn main() {
    println!("=== NAS100 Indicator Parameter Optimization ===\n");

    let ds = FixtureDataSource::new();
    let data = ds.load("NAS100", Timeframe::D1).expect("Failed to load NAS100 data");

    println!("Loaded {} data points for NAS100", data.close.len());
    println!("Date range: {} bars of daily data\n", data.close.len());

    let objective = SharpeRatio::new();

    // Use last 5 years for optimization with walk-forward validation
    // Assuming ~252 trading days/year, 5 years = ~1260 bars
    let five_years = 1260;
    let data_slice = if data.close.len() > five_years {
        data.slice(data.close.len() - five_years, data.close.len())
    } else {
        data.clone()
    };

    println!("Using last {} bars (~5 years) for optimization\n", data_slice.close.len());

    // Walk-forward validation: 70% train, 30% test
    let train_size = (data_slice.close.len() as f64 * 0.7) as usize;
    let splits = vec![
        ValidationSplit {
            train_start: 0,
            train_end: train_size,
            test_start: train_size,
            test_end: data_slice.close.len(),
        },
    ];

    // =========================================================================
    // RSI Optimization
    // =========================================================================
    println!("--- RSI Optimization ---");
    let rsi_evaluator = RSIEvaluator::new(ParamRange::new(5, 30, 1))
        .with_thresholds(70.0, 30.0);

    let start = Instant::now();
    let result = ParallelOptimizerRunner::new()
        .add_grid(GridSearchConfig { parallel: true, top_n: 5 })
        .add_genetic(GeneticConfig {
            population_size: 30,
            generations: 20,
            mutation_rate: 0.15,
            crossover_rate: 0.8,
            elite_count: 3,
            tournament_size: 3,
            parallel: true,
        })
        .add_bayesian(BayesianConfig {
            iterations: 25,
            initial_samples: 10,
            exploration_factor: 2.0,
            length_scale: 0.3,
            noise_variance: 0.01,
        })
        .run_select_by_oos(&rsi_evaluator, &data_slice, &objective, &splits)
        .expect("RSI optimization failed");

    println!("Best optimizer: {}", result.best_optimizer);
    println!("Best params: {:?}", result.best_params);
    println!("In-sample Sharpe: {:.4}", result.best_score);
    println!("Out-of-sample Sharpe: {:.4}", result.oos_score.unwrap_or(0.0));
    println!("Total evaluations: {}", result.total_evaluations);
    println!("Time: {:.2?}\n", start.elapsed());

    println!("All optimizer results:");
    for (opt_type, res) in &result.optimizer_results {
        println!("  {}: IS={:.4}, OOS={:.4} ({} evals)",
            opt_type, res.best_score, res.oos_score.unwrap_or(0.0), res.evaluations);
    }
    println!();

    // =========================================================================
    // MACD Optimization
    // =========================================================================
    println!("--- MACD Optimization ---");
    let macd_evaluator = MACDEvaluator::new(
        ParamRange::new(8, 14, 2),   // fast
        ParamRange::new(20, 30, 2),  // slow
        ParamRange::new(7, 11, 2),   // signal
    );

    let start = Instant::now();
    let result = ParallelOptimizerRunner::new()
        .add_grid(GridSearchConfig { parallel: true, top_n: 5 })
        .add_genetic(GeneticConfig {
            population_size: 30,
            generations: 15,
            parallel: true,
            ..Default::default()
        })
        .add_bayesian(BayesianConfig {
            iterations: 20,
            initial_samples: 10,
            ..Default::default()
        })
        .run_select_by_oos(&macd_evaluator, &data_slice, &objective, &splits)
        .expect("MACD optimization failed");

    println!("Best optimizer: {}", result.best_optimizer);
    println!("Best params: {:?}", result.best_params);
    println!("In-sample Sharpe: {:.4}", result.best_score);
    println!("Out-of-sample Sharpe: {:.4}", result.oos_score.unwrap_or(0.0));
    println!("Time: {:.2?}\n", start.elapsed());

    // =========================================================================
    // Bollinger Bands Optimization
    // =========================================================================
    println!("--- Bollinger Bands Optimization ---");
    let bb_evaluator = BollingerEvaluator::new(
        ParamRange::new(15, 30, 5),              // period
        FloatParamRange::new(1.5, 2.5, 0.25),    // std dev
    );

    let start = Instant::now();
    let result = ParallelOptimizerRunner::new()
        .add_grid(GridSearchConfig { parallel: true, top_n: 5 })
        .add_genetic(GeneticConfig {
            population_size: 25,
            generations: 15,
            parallel: true,
            ..Default::default()
        })
        .add_bayesian(BayesianConfig {
            iterations: 20,
            initial_samples: 8,
            ..Default::default()
        })
        .run_select_by_oos(&bb_evaluator, &data_slice, &objective, &splits)
        .expect("Bollinger optimization failed");

    println!("Best optimizer: {}", result.best_optimizer);
    println!("Best params: {:?}", result.best_params);
    println!("In-sample Sharpe: {:.4}", result.best_score);
    println!("Out-of-sample Sharpe: {:.4}", result.oos_score.unwrap_or(0.0));
    println!("Time: {:.2?}\n", start.elapsed());

    println!("=== Optimization Complete ===");
}
