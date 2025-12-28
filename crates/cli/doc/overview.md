# cli Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](../../../doc/4-development/developer-guide.md).

## Audience

Data analysts and DevOps engineers who need command-line access for scripting and automation.

## WHAT

Command-line interface for rustful-ts:
- **Forecasting** - CSV input, multiple models, file output
- **Anomaly Detection** - Batch detection with configurable methods
- **Backtesting** - Strategy testing from command line

## WHY

| Problem | Solution |
|---------|----------|
| Quick data exploration without coding | Simple CLI commands |
| Batch processing in scripts and pipelines | CSV input/output, composable commands |
| CI/CD integration for model validation | Exit codes and structured output |

## HOW

```bash
# Install
cargo install --path crates/cli

# Forecast
rustful forecast -i data.csv -s 10 --model arima

# Detect anomalies
rustful detect -i metrics.csv --method zscore --threshold 3

# Backtest
rustful backtest -i prices.csv --strategy sma_crossover
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../doc/4-development/developer-guide.md) | Build, test, contribute |
| [Backlog](../backlog.md) | Planned features |

---

**Status**: Beta
