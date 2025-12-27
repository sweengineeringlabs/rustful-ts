# rustful-financial Overview

## WHAT: Financial Analytics Module

rustful-financial provides portfolio management, backtesting, and risk metrics for financial time series applications.

Key capabilities:
- **Portfolio Management** - Position tracking, P&L calculation
- **Backtesting** - Historical strategy testing
- **Signal Generation** - Trading signals (Buy/Sell/Hold)
- **Risk Metrics** - VaR, Sharpe ratio, Sortino ratio, max drawdown

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Rust | 1.75+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |

**Dependencies**: rustful-core (automatically resolved via Cargo)

### Build

```bash
cargo build -p rustful-financial
cargo test -p rustful-financial
```

## WHY: Trading Analytics

**Problems Solved**:
1. Separate libraries needed for forecasting and backtesting
2. Inconsistent risk metric implementations
3. No integration between prediction and trading signals

**When to Use**: Trading strategy development, portfolio analysis, risk management

**When NOT to Use**: Non-financial forecasting applications

## HOW: Usage Guide

### Portfolio Management

```rust
use rustful_financial::portfolio::{Portfolio, Position};

let mut portfolio = Portfolio::new(100_000.0);
portfolio.add_position(Position::new("AAPL", 100, 150.0));

let value = portfolio.total_value(&current_prices);
let pnl = portfolio.unrealized_pnl(&current_prices);
```

### Backtesting

```rust
use rustful_financial::backtest::{Backtester, BacktestConfig};
use rustful_financial::signals::SMACrossover;

let strategy = SMACrossover::new(20, 50);
let config = BacktestConfig::default();

let result = Backtester::run(&prices, strategy, config)?;
println!("Return: {:.2}%", result.total_return * 100.0);
```

### Risk Metrics

```rust
use rustful_financial::risk::{var_historical, sharpe_ratio, max_drawdown};

let var = var_historical(&returns, 0.95);
let sharpe = sharpe_ratio(&returns, 0.02);
let drawdown = max_drawdown(&equity_curve);
```

### Signal Generation

```rust
use rustful_financial::signals::{Signal, SignalGenerator, SMACrossover};

let strategy = SMACrossover::new(20, 50);
let signal = strategy.generate(&prices, index);

match signal {
    Signal::Buy => { /* enter long */ }
    Signal::Sell => { /* exit position */ }
    Signal::Hold => { /* maintain position */ }
}
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| rustful-core | Uses for price forecasting |
| rustful-server | Exposes via REST endpoints |
| rustful-cli | Available via CLI commands |

**Integration Points**:
- Combines predictions with signal generation
- Backtester uses forecasts for validation

## Examples and Tests

### Examples

**Location**: [`examples/`](../examples/)

- `basic.rs` - Simple portfolio and backtest

### Tests

**Location**: [`tests/`](../tests/)

- `integration.rs` - Public API tests

### Testing

```bash
cargo test -p rustful-financial
```

---

**Status**: Beta
**Roadmap**: See [backlog.md](../backlog.md) | [Framework Backlog](../../../doc/framework-backlog.md)
