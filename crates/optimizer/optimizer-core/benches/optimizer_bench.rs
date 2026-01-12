//! Benchmark suite for optimizer algorithms.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use optimizer_core::{
    optimizer::{IndicatorGridSearch, IndicatorGeneticOptimizer, IndicatorBayesianOptimizer, GridSearchConfig, GeneticConfig, BayesianConfig},
    evaluators::RSIEvaluator,
    objective::SharpeRatio,
    datasource::FixtureDataSource,
};
use optimizer_spi::{DataSource, ParamRange, Timeframe, MarketData};

fn create_test_data(size: usize) -> MarketData {
    let mut data = MarketData::new("BENCH", Timeframe::D1);
    for i in 0..size {
        let trend = i as f64 * 0.05;
        let noise = (i as f64 * 0.3).sin() * 3.0;
        let price = 100.0 + trend + noise;
        data.close.push(price);
        data.open.push(price - 0.5);
        data.high.push(price + 2.0);
        data.low.push(price - 2.0);
        data.volume.push(1000.0);
        data.timestamps.push(i as i64);
    }
    data
}

fn bench_grid_search(c: &mut Criterion) {
    let evaluator = RSIEvaluator::new(ParamRange::new(5, 30, 5)); // 6 values
    let objective = SharpeRatio::new();

    let mut group = c.benchmark_group("GridSearch");

    for data_size in [100, 500, 1000].iter() {
        let data = create_test_data(*data_size);

        // Sequential
        group.bench_with_input(
            BenchmarkId::new("sequential", data_size),
            &data,
            |b, data| {
                let config = GridSearchConfig {
                    parallel: false,
                    top_n: 5,
                };
                let optimizer = IndicatorGridSearch::with_config(config);
                b.iter(|| {
                    optimizer.optimize(black_box(&evaluator), black_box(data), black_box(&objective), &[])
                });
            },
        );

        // Parallel
        group.bench_with_input(
            BenchmarkId::new("parallel", data_size),
            &data,
            |b, data| {
                let config = GridSearchConfig {
                    parallel: true,
                    top_n: 5,
                };
                let optimizer = IndicatorGridSearch::with_config(config);
                b.iter(|| {
                    optimizer.optimize(black_box(&evaluator), black_box(data), black_box(&objective), &[])
                });
            },
        );
    }

    group.finish();
}

fn bench_genetic(c: &mut Criterion) {
    let evaluator = RSIEvaluator::new(ParamRange::new(5, 30, 1)); // 26 values
    let objective = SharpeRatio::new();
    let data = create_test_data(500);

    let mut group = c.benchmark_group("Genetic");

    // Different population sizes
    for pop_size in [10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("pop", pop_size),
            pop_size,
            |b, &pop_size| {
                let config = GeneticConfig {
                    population_size: pop_size,
                    generations: 10,
                    mutation_rate: 0.1,
                    crossover_rate: 0.8,
                    elite_count: 2,
                    tournament_size: 3,
                    parallel: false,
                };
                let optimizer = IndicatorGeneticOptimizer::with_config(config);
                b.iter(|| {
                    optimizer.optimize(black_box(&evaluator), black_box(&data), black_box(&objective), &[])
                });
            },
        );
    }

    // Different generation counts
    for gens in [5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("gens", gens),
            gens,
            |b, &gens| {
                let config = GeneticConfig {
                    population_size: 15,
                    generations: gens,
                    mutation_rate: 0.1,
                    crossover_rate: 0.8,
                    elite_count: 2,
                    tournament_size: 3,
                    parallel: false,
                };
                let optimizer = IndicatorGeneticOptimizer::with_config(config);
                b.iter(|| {
                    optimizer.optimize(black_box(&evaluator), black_box(&data), black_box(&objective), &[])
                });
            },
        );
    }

    group.finish();
}

fn bench_bayesian(c: &mut Criterion) {
    let evaluator = RSIEvaluator::new(ParamRange::new(5, 30, 1));
    let objective = SharpeRatio::new();
    let data = create_test_data(500);

    let mut group = c.benchmark_group("Bayesian");

    // Different iteration counts
    for iters in [5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("iters", iters),
            iters,
            |b, &iters| {
                let config = BayesianConfig {
                    iterations: iters,
                    initial_samples: 5,
                    exploration_factor: 2.0,
                    length_scale: 0.5,
                    noise_variance: 0.01,
                };
                let optimizer = IndicatorBayesianOptimizer::with_config(config);
                b.iter(|| {
                    optimizer.optimize(black_box(&evaluator), black_box(&data), black_box(&objective), &[])
                });
            },
        );
    }

    group.finish();
}

fn bench_optimizer_comparison(c: &mut Criterion) {
    let evaluator = RSIEvaluator::new(ParamRange::new(5, 25, 1)); // 21 values
    let objective = SharpeRatio::new();
    let data = create_test_data(500);

    let mut group = c.benchmark_group("Comparison");

    // Grid Search (exhaustive)
    group.bench_function("grid_exhaustive", |b| {
        let config = GridSearchConfig {
            parallel: true,
            top_n: 5,
        };
        let optimizer = IndicatorGridSearch::with_config(config);
        b.iter(|| {
            optimizer.optimize(black_box(&evaluator), black_box(&data), black_box(&objective), &[])
        });
    });

    // Genetic (50 evaluations = 10 pop * 5 gens)
    group.bench_function("genetic_50_evals", |b| {
        let config = GeneticConfig {
            population_size: 10,
            generations: 5,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elite_count: 2,
            tournament_size: 3,
            parallel: false,
        };
        let optimizer = IndicatorGeneticOptimizer::with_config(config);
        b.iter(|| {
            optimizer.optimize(black_box(&evaluator), black_box(&data), black_box(&objective), &[])
        });
    });

    // Bayesian (20 evaluations = 5 initial + 15 iterations)
    group.bench_function("bayesian_20_evals", |b| {
        let config = BayesianConfig {
            iterations: 15,
            initial_samples: 5,
            exploration_factor: 2.0,
            length_scale: 0.5,
            noise_variance: 0.01,
        };
        let optimizer = IndicatorBayesianOptimizer::with_config(config);
        b.iter(|| {
            optimizer.optimize(black_box(&evaluator), black_box(&data), black_box(&objective), &[])
        });
    });

    group.finish();
}

fn bench_with_real_data(c: &mut Criterion) {
    let ds = FixtureDataSource::new();

    // Skip if fixture data not available
    let data = match ds.load("SPY", Timeframe::D1) {
        Ok(d) => d,
        Err(_) => return,
    };

    let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
    let objective = SharpeRatio::new();

    let mut group = c.benchmark_group("RealData");

    group.bench_function("grid_spy_d1", |b| {
        let config = GridSearchConfig {
            parallel: true,
            top_n: 5,
        };
        let optimizer = IndicatorGridSearch::with_config(config);
        b.iter(|| {
            optimizer.optimize(black_box(&evaluator), black_box(&data), black_box(&objective), &[])
        });
    });

    group.bench_function("genetic_spy_d1", |b| {
        let config = GeneticConfig {
            population_size: 10,
            generations: 5,
            parallel: false,
            ..Default::default()
        };
        let optimizer = IndicatorGeneticOptimizer::with_config(config);
        b.iter(|| {
            optimizer.optimize(black_box(&evaluator), black_box(&data), black_box(&objective), &[])
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_grid_search,
    bench_genetic,
    bench_bayesian,
    bench_optimizer_comparison,
    bench_with_real_data,
);
criterion_main!(benches);
