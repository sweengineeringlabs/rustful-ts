# StrategyBuilder: Indicator-Based Trading Strategies

The `StrategyBuilder` in QuantLang provides a framework for designing, backtesting, and analyzing trading strategies built around optimized technical indicators.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Strategy Design Flow                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. Optimize Indicators          2. Build Strategy                      │
│   ┌─────────────────────┐         ┌─────────────────────┐               │
│   │ IndicatorOptimizer  │         │  StrategyBuilder    │               │
│   │                     │         │                     │               │
│   │ MACD(6, 18, 5)     │────────▶│ .from_optimized()   │               │
│   │ RSI(5)             │         │ .bollinger_breakout()│               │
│   │ Bollinger(10, 1.5) │         │ .position_sizing()   │               │
│   └─────────────────────┘         └──────────┬──────────┘               │
│                                              │                           │
│   3. Backtest                     4. Analyze Results                     │
│   ┌─────────────────────┐         ┌─────────────────────┐               │
│   │  .backtest(&prices) │         │  StrategyResult     │               │
│   │                     │────────▶│                     │               │
│   │  - Generate trades  │         │  - Trades list      │               │
│   │  - Apply rules      │         │  - Metrics          │               │
│   │  - Track positions  │         │  - Equity curve     │               │
│   └─────────────────────┘         └─────────────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## NAS100 Backtest Report (2023-2024)

### Optimized Indicator Parameters

Individual optimization on 400 daily price points yielded:

| Indicator | Optimal Parameters | Sharpe | Robustness |
|-----------|-------------------|--------|------------|
| RSI | period=5 | 12.58 | 1.09 |
| Bollinger | period=10, std_dev=1.5 | 12.56 | 1.09 |
| MACD | fast=6, slow=18, signal=5 | 8.90 | 0.86 |
| SMA | period=50 | -8.02 | 1.43 |
| EMA | period=50 | -8.47 | 1.26 |

### Strategy Performance Comparison

| Strategy | Trades | Win% | Return% | Sharpe | MaxDD% | Profit Factor |
|----------|--------|------|---------|--------|--------|---------------|
| **Bollinger Breakout** | 16 | 75.0 | 61.37 | **18.66** | 2.64 | 11.39 |
| MACD Crossover | 41 | 51.2 | 96.52 | 2.74 | 43.59 | 1.66 |
| RSI Mean Reversion | 83 | 12.0 | -109.72 | -4.30 | 152.45 | 0.51 |
| Bollinger Mean Rev | 69 | 11.6 | -49.30 | -1.88 | 93.17 | 0.82 |
| MA Trend Following | 4 | 0.0 | -17.46 | -39.97 | 17.46 | 0.00 |
| RSI + MACD Combo | 20 | 20.0 | -16.37 | -6.28 | 27.43 | 0.52 |
| Triple Indicator | 2 | 50.0 | -0.44 | -1.60 | 2.41 | 0.65 |

### Best Performing Strategy: Bollinger Breakout

```
Strategy Parameters:
  - Bollinger Bands: period=10, std_dev=1.5 (tight bands)
  - Entry: Price breaks above upper band
  - Exit: Trailing stop (2%) or time-based (10 bars)

Performance:
  - Sharpe Ratio: 18.66 (exceptional risk-adjusted returns)
  - Win Rate: 75% (12 of 16 trades profitable)
  - Max Drawdown: 2.64% (minimal capital at risk)
  - Profit Factor: 11.39 (gross profit / gross loss)
  - Total Return: 61.37%
```

### Key Insights

1. **Momentum strategies dominated** on NAS100's 2023-2024 uptrend
   - Bollinger Breakout captured trending moves
   - Mean reversion strategies underperformed in trending market

2. **Tight indicator parameters worked best**
   - RSI(5) vs traditional RSI(14)
   - Bollinger(10, 1.5) vs traditional (20, 2.0)
   - MACD(6, 18, 5) vs traditional (12, 26, 9)

3. **Fewer high-quality trades outperformed high-frequency trading**
   - Bollinger Breakout: 16 trades, 75% win rate
   - RSI Mean Reversion: 83 trades, 12% win rate

## API Reference

### Creating a Strategy

```rust
use quantlang::runtime::{
    StrategyBuilder, IndicatorOptimizer, OptimizedIndicator,
    PositionSizing, Objective, OptMethod, Validation
};

// Option 1: From optimized indicators
let opt_result = IndicatorOptimizer::new()
    .add_rsi_range(5, 25, 5)
    .add_bollinger_range(10, 30, 5, 1.5, 3.0, 0.5)
    .objective(Objective::SharpeRatio)
    .method(OptMethod::ParallelGrid)
    .optimize_individual(&prices);

let strategy = StrategyBuilder::from_optimized(&opt_result)
    .bollinger_breakout();

// Option 2: Manual indicator specification
let strategy = StrategyBuilder::new()
    .add_indicator(OptimizedIndicator::new("RSI").with_param("period", 5.0))
    .add_indicator(OptimizedIndicator::new("Bollinger")
        .with_param("period", 10.0)
        .with_param("std_dev", 1.5))
    .rsi_mean_reversion(30.0, 70.0);
```

### Pre-built Strategy Templates

| Method | Entry Condition | Exit Condition |
|--------|-----------------|----------------|
| `.rsi_mean_reversion(oversold, overbought)` | RSI < oversold | RSI = 50 or stop loss |
| `.macd_crossover()` | MACD histogram crosses zero | MACD reversal or stop loss |
| `.bollinger_mean_reversion()` | Price touches lower band | Price returns to middle band |
| `.bollinger_breakout()` | Price breaks upper band | Trailing stop or time exit |
| `.ma_trend_following()` | Price crosses above MA | Signal reversal or stop loss |
| `.rsi_macd_combo(oversold, overbought)` | RSI + MACD both signal | Either reverses |
| `.triple_indicator()` | RSI + MACD + Bollinger align | Any indicator exits |

### Position Sizing

```rust
// Fixed dollar amount
.position_sizing(PositionSizing::Fixed { size: 10000.0 })

// Percentage of equity
.position_sizing(PositionSizing::PercentEquity { pct: 25.0 })

// Volatility-adjusted (risk % of equity per ATR multiple)
.position_sizing(PositionSizing::VolatilityAdjusted {
    risk_pct: 2.0,
    atr_mult: 2.0
})

// Kelly criterion
.position_sizing(PositionSizing::Kelly { fraction: 0.25 })
```

### Running Backtest

```rust
let result = strategy.backtest(&prices);

// Access metrics
println!("Total Return: {:.2}%", result.metrics.total_return);
println!("Sharpe Ratio: {:.2}", result.metrics.sharpe_ratio);
println!("Max Drawdown: {:.2}%", result.metrics.max_drawdown);
println!("Win Rate: {:.1}%", result.metrics.win_rate);

// Analyze trades
for trade in &result.trades {
    println!("Bar {}-{}: {} PnL={:.2}%",
        trade.entry_bar, trade.exit_bar,
        if trade.direction > 0 { "LONG" } else { "SHORT" },
        trade.pnl_pct);
}

// Equity curve for visualization
let final_equity = result.equity_curve.last().unwrap();
```

### Custom Strategy Rules

```rust
use quantlang::runtime::{StrategyRule, EntryCondition, ExitCondition};

let custom_strategy = StrategyBuilder::new()
    .add_indicator(rsi)
    .add_indicator(macd)
    .add_rule(StrategyRule {
        name: "Custom Long".to_string(),
        entry_conditions: vec![
            EntryCondition::RsiOversold { threshold: 25.0 },
            EntryCondition::MacdCrossUp,
        ],
        exit_conditions: vec![
            ExitCondition::TakeProfit { pct: 5.0 },
            ExitCondition::StopLoss { pct: 2.0 },
            ExitCondition::TimeBased { bars: 20 },
        ],
        direction: 1, // Long only
    });
```

## Entry Conditions

| Condition | Description |
|-----------|-------------|
| `RsiOversold { threshold }` | RSI below threshold |
| `RsiOverbought { threshold }` | RSI above threshold |
| `MacdCrossUp` | MACD histogram crosses above zero |
| `MacdCrossDown` | MACD histogram crosses below zero |
| `PriceAboveMA` | Price crosses above SMA/EMA |
| `PriceBelowMA` | Price crosses below SMA/EMA |
| `BollingerLowerTouch` | Price touches lower band |
| `BollingerUpperTouch` | Price touches upper band |
| `BollingerUpperBreak` | Price breaks above upper band |
| `BollingerLowerBreak` | Price breaks below lower band |

## Exit Conditions

| Condition | Description |
|-----------|-------------|
| `TakeProfit { pct }` | Exit at specified profit % |
| `StopLoss { pct }` | Exit at specified loss % |
| `RsiLevel { threshold }` | Exit when RSI reaches level |
| `MacdReversal` | Exit on MACD signal reversal |
| `MeanReversion` | Exit when price returns to MA |
| `TrailingStop { pct }` | Trailing stop loss % |
| `TimeBased { bars }` | Exit after N bars |
| `SignalReversal` | Exit on opposite entry signal |

## Performance Metrics

| Metric | Description |
|--------|-------------|
| `total_trades` | Number of completed trades |
| `winners` / `losers` | Count of profitable/unprofitable trades |
| `win_rate` | Percentage of winning trades |
| `total_return` | Cumulative return percentage |
| `sharpe_ratio` | Annualized risk-adjusted return |
| `max_drawdown` | Largest peak-to-trough decline |
| `profit_factor` | Gross profit / gross loss |
| `avg_trade` | Average return per trade |
| `avg_winner` / `avg_loser` | Average winning/losing trade |
| `largest_winner` / `largest_loser` | Best/worst single trade |
| `avg_holding_period` | Average trade duration (bars) |

## Best Practices

### 1. Match Strategy to Market Regime

```
Trending Market (NAS100 2023-2024):
  ✓ Bollinger Breakout
  ✓ MACD Crossover
  ✗ RSI Mean Reversion

Ranging Market:
  ✓ RSI Mean Reversion
  ✓ Bollinger Mean Reversion
  ✗ Trend Following
```

### 2. Use Walk-Forward Validation

```rust
let result = IndicatorOptimizer::new()
    .add_bollinger_range(10, 30, 5, 1.5, 3.0, 0.5)
    .validation(Validation::WalkForward {
        windows: 5,
        train_ratio: 0.8
    })
    .optimize_individual(&prices);

// Check robustness ratio (OOS/IS score)
// > 0.7 is good, < 0.5 indicates overfitting
```

### 3. Consider Trade Frequency

- More trades = more transaction costs
- Fewer high-conviction trades often outperform
- Bollinger Breakout: 16 trades with 75% win rate
- RSI Mean Reversion: 83 trades with 12% win rate

### 4. Risk Management

```rust
// Limit position size based on volatility
.position_sizing(PositionSizing::VolatilityAdjusted {
    risk_pct: 1.0,  // Risk 1% of equity per trade
    atr_mult: 2.0   // Stop at 2x ATR
})

// Or use Kelly criterion with fraction
.position_sizing(PositionSizing::Kelly {
    fraction: 0.25  // Quarter-Kelly for safety
})
```

## Integration with IndicatorOptimizer

The StrategyBuilder integrates seamlessly with the indicator optimization pipeline:

```rust
// Step 1: Find optimal parameters
let opt_result = IndicatorOptimizer::new()
    .add_macd_range((6, 16, 2), (18, 32, 2), (5, 14, 3))
    .add_rsi_range(5, 25, 5)
    .add_bollinger_range(10, 30, 5, 1.5, 3.0, 0.5)
    .objective(Objective::SharpeRatio)
    .method(OptMethod::ParallelGrid)
    .validation(Validation::WalkForward { windows: 3, train_ratio: 0.75 })
    .optimize_individual(&prices);

// Step 2: Build strategy with optimized params
let strategy = StrategyBuilder::from_optimized(&opt_result)
    .bollinger_breakout()
    .initial_equity(100000.0)
    .position_sizing(PositionSizing::PercentEquity { pct: 25.0 });

// Step 3: Backtest
let result = strategy.backtest(&prices);

// Step 4: Analyze
if result.metrics.sharpe_ratio > 2.0 && result.metrics.max_drawdown < 10.0 {
    println!("Strategy meets criteria for deployment");
}
```

## Files

- **Implementation**: `src/runtime/indicator_optimizer.rs` (lines 2809-3833)
- **Exports**: `src/runtime/mod.rs`
- **Tests**: `cargo test test_strategy_templates --lib -- --nocapture`
