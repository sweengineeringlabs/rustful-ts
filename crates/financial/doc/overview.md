# financial Overview

## Audience

Quantitative developers and traders who need portfolio management, backtesting, and risk analytics.

## WHAT

Financial analytics for time series:
- **Portfolio Management** - Position tracking, P&L calculation
- **Backtesting** - Historical strategy testing
- **Risk Metrics** - Sharpe ratio, Sortino ratio, VaR, max drawdown

## WHY

| Problem | Solution |
|---------|----------|
| Separate libraries needed for forecasting and backtesting | Integrated financial module |
| Inconsistent risk metric implementations | Standardized, tested implementations |
| No integration between prediction and trading signals | Combines forecasts with signal generation |

## HOW

```rust
use financial::{backtest, sharpe_ratio, max_drawdown};

// Run backtest
let result = backtest(&prices, &signals, initial_capital)?;
println!("Return: {:.2}%", result.total_return * 100.0);

// Risk metrics
let sharpe = sharpe_ratio(&returns, risk_free_rate);
let drawdown = max_drawdown(&equity_curve);
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../doc/4-development/developer-guide.md) | Build, test, contribute |
| [Backlog](../backlog.md) | Planned features |

---

**Status**: Beta
