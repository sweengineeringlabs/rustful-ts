//! Integration tests for the optimizer framework using real market data.

use optimizer_core::{
    // Evaluators
    RSIEvaluator, MACDEvaluator, SMAEvaluator, BollingerEvaluator,
    create_evaluator, EvaluatorType,
    // Optimizer
    IndicatorGridSearch, GridSearchConfig,
    // Objective
    SharpeRatio, SortinoRatio, TotalReturn, create_objective,
    // Data source
    FixtureDataSource,
    // Types
    Timeframe, MarketData, ParamRange, FloatParamRange,
    DataSource, Objective, ValidationStrategy,
};
use optimizer_core::validator::{TrainTestValidator, WalkForwardValidator, create_validator};
use optimizer_spi::Validator;

#[test]
fn test_rsi_optimization_spy_daily() {
    // Load real SPY daily data
    let ds = FixtureDataSource::new();
    let data = ds.load("SPY", Timeframe::D1).expect("Failed to load SPY data");

    println!("Loaded {} bars of SPY daily data", data.len());
    assert!(data.len() > 1000, "Should have plenty of historical data");

    // Create RSI evaluator with period range 10-30
    let evaluator = RSIEvaluator::new(ParamRange::new(10, 30, 5));

    // Create Sharpe ratio objective
    let objective = SharpeRatio::new();

    // Run grid search
    let grid = IndicatorGridSearch::new();
    let result = grid.optimize(&evaluator, &data, &objective, &[]).unwrap();

    println!("RSI Optimization Results:");
    println!("  Best params: {:?}", result.best_params);
    println!("  Best Sharpe: {:.4}", result.best_score);
    println!("  Evaluations: {}", result.evaluations);

    // Check we got results
    assert!(!result.best_params.is_empty());
    assert!(result.evaluations >= 5); // Should have tested multiple periods
}

#[test]
fn test_macd_optimization_aapl_daily() {
    let ds = FixtureDataSource::new();
    let data = ds.load("AAPL", Timeframe::D1).expect("Failed to load AAPL data");

    println!("Loaded {} bars of AAPL daily data", data.len());

    // MACD with ranges for fast, slow, signal
    let evaluator = MACDEvaluator::new(
        ParamRange::new(8, 14, 3),   // fast: 8, 11, 14
        ParamRange::new(20, 28, 4),  // slow: 20, 24, 28
        ParamRange::new(7, 11, 2),   // signal: 7, 9, 11
    );

    let objective = SharpeRatio::new();
    let grid = IndicatorGridSearch::new();
    let result = grid.optimize(&evaluator, &data, &objective, &[]).unwrap();

    println!("MACD Optimization Results:");
    println!("  Best params: {:?}", result.best_params);
    println!("  Best Sharpe: {:.4}", result.best_score);
    println!("  Evaluations: {}", result.evaluations);

    assert!(result.evaluations >= 9); // 3 * 3 * 3 = 27 minimum combinations
}

#[test]
fn test_sma_crossover_optimization_btc() {
    let ds = FixtureDataSource::new();
    let data = ds.load("BTC-USD", Timeframe::D1).expect("Failed to load BTC data");

    println!("Loaded {} bars of BTC daily data", data.len());

    // SMA crossover with fast and slow ranges
    let evaluator = SMAEvaluator::new(
        ParamRange::new(10, 30, 10),  // fast: 10, 20, 30
        ParamRange::new(40, 60, 10),  // slow: 40, 50, 60
    );

    let objective = TotalReturn::new();
    let grid = IndicatorGridSearch::new();
    let result = grid.optimize(&evaluator, &data, &objective, &[]).unwrap();

    println!("SMA Crossover Optimization Results:");
    println!("  Best params: {:?}", result.best_params);
    println!("  Best Total Return: {:.4}", result.best_score);
    println!("  Evaluations: {}", result.evaluations);

    assert!(result.evaluations >= 9);
}

#[test]
fn test_bollinger_optimization_eurusd() {
    let ds = FixtureDataSource::new();
    let data = ds.load("EURUSD", Timeframe::D1).expect("Failed to load EURUSD data");

    println!("Loaded {} bars of EURUSD daily data", data.len());

    // Bollinger Bands with period and std_dev ranges
    let evaluator = BollingerEvaluator::new(
        ParamRange::new(15, 25, 5),         // period: 15, 20, 25
        FloatParamRange::new(1.5, 2.5, 0.5), // std_dev: 1.5, 2.0, 2.5
    );

    let objective = SortinoRatio::new();
    let grid = IndicatorGridSearch::new();
    let result = grid.optimize(&evaluator, &data, &objective, &[]).unwrap();

    println!("Bollinger Optimization Results:");
    println!("  Best params: {:?}", result.best_params);
    println!("  Best Sortino: {:.4}", result.best_score);
    println!("  Evaluations: {}", result.evaluations);

    assert!(result.evaluations >= 9);
}

#[test]
fn test_with_train_test_validation() {
    let ds = FixtureDataSource::new();
    let data = ds.load("SPY", Timeframe::D1).expect("Failed to load SPY data");

    let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
    let objective = SharpeRatio::new();

    // Create train/test validation split (70/30)
    let validator = TrainTestValidator::new(0.7);
    let splits = validator.splits(data.len()).unwrap();

    let grid = IndicatorGridSearch::new();
    let result = grid.optimize(&evaluator, &data, &objective, &splits).unwrap();

    println!("RSI with Train/Test Validation:");
    println!("  Best params: {:?}", result.best_params);
    println!("  In-sample Sharpe: {:.4}", result.best_score);
    println!("  Out-of-sample Sharpe: {:?}", result.oos_score);
    println!("  Robustness (OOS/IS): {:?}", result.robustness);

    // Should have OOS score with validation
    assert!(result.oos_score.is_some());
}

#[test]
fn test_with_walk_forward_validation() {
    let ds = FixtureDataSource::new();
    let data = ds.load("NAS100", Timeframe::D1).expect("Failed to load NAS100 data");

    let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
    let objective = SharpeRatio::new();

    // Walk-forward validation with 5 windows
    let validator = WalkForwardValidator::new(5, 0.7);
    let splits = validator.splits(data.len()).unwrap();

    println!("Walk-forward splits: {} windows", splits.len());

    let grid = IndicatorGridSearch::new();
    let result = grid.optimize(&evaluator, &data, &objective, &splits).unwrap();

    println!("RSI with Walk-Forward Validation:");
    println!("  Best params: {:?}", result.best_params);
    println!("  In-sample Sharpe: {:.4}", result.best_score);
    println!("  Out-of-sample Sharpe: {:?}", result.oos_score);
    println!("  Robustness: {:?}", result.robustness);

    assert!(result.oos_score.is_some());
}

#[test]
fn test_h4_timeframe() {
    let ds = FixtureDataSource::new();
    let data = ds.load("SPY", Timeframe::H4).expect("Failed to load SPY H4 data");

    println!("Loaded {} bars of SPY H4 data (aggregated from H1)", data.len());
    assert_eq!(data.timeframe, Timeframe::H4);

    let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
    let objective = SharpeRatio::new();
    let grid = IndicatorGridSearch::new();
    let result = grid.optimize(&evaluator, &data, &objective, &[]).unwrap();

    println!("RSI H4 Optimization:");
    println!("  Best params: {:?}", result.best_params);
    println!("  Best Sharpe: {:.4}", result.best_score);
}

#[test]
fn test_evaluator_factory() {
    let ds = FixtureDataSource::new();
    let data = ds.load("GLD", Timeframe::D1).expect("Failed to load GLD data");

    // Use factory to create evaluator
    let evaluator = create_evaluator(EvaluatorType::RSI {
        period: ParamRange::new(12, 18, 3),
        overbought: 70.0,
        oversold: 30.0,
    });

    let objective = create_objective(Objective::SharpeRatio);
    let grid = IndicatorGridSearch::new();
    let result = grid.optimize(evaluator.as_ref(), &data, objective.as_ref(), &[]).unwrap();

    println!("Factory-created RSI on GLD:");
    println!("  Best params: {:?}", result.best_params);
    println!("  Best Sharpe: {:.4}", result.best_score);
}

#[test]
fn test_top_n_results() {
    let ds = FixtureDataSource::new();
    let data = ds.load("SPY", Timeframe::D1).expect("Failed to load SPY data");

    let evaluator = RSIEvaluator::new(ParamRange::new(5, 30, 5)); // More periods for more results
    let objective = SharpeRatio::new();

    let config = GridSearchConfig {
        parallel: true,
        top_n: 5,
    };
    let grid = IndicatorGridSearch::with_config(config);
    let result = grid.optimize(&evaluator, &data, &objective, &[]).unwrap();

    println!("Top {} RSI results:", result.top_results.len());
    for (i, (params, score)) in result.top_results.iter().enumerate() {
        println!("  {}. {:?} -> {:.4}", i + 1, params, score);
    }

    assert!(result.top_results.len() <= 5);
    // Results should be sorted by score (descending)
    for window in result.top_results.windows(2) {
        assert!(window[0].1 >= window[1].1);
    }
}

#[test]
fn test_multiple_symbols() {
    let ds = FixtureDataSource::new();
    let symbols = ["SPY", "AAPL", "GOOGL"];
    let evaluator = RSIEvaluator::new(ParamRange::new(10, 20, 5));
    let objective = SharpeRatio::new();
    let grid = IndicatorGridSearch::new();

    println!("Cross-symbol RSI optimization:");
    for symbol in &symbols {
        let data = ds.load(symbol, Timeframe::D1).expect(&format!("Failed to load {}", symbol));
        let result = grid.optimize(&evaluator, &data, &objective, &[]).unwrap();
        println!("  {}: period={}, sharpe={:.4}",
            symbol,
            result.best_params[0].get_param("period").unwrap_or(0.0),
            result.best_score
        );
    }
}
