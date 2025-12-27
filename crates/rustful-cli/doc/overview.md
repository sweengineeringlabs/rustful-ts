# rustful-cli Overview

## WHAT: Command-Line Interface

rustful-cli provides a command-line tool for forecasting, anomaly detection, backtesting, and running the API server.

Key capabilities:
- **Forecasting** - CSV input, multiple models, output to file or stdout
- **Anomaly Detection** - Batch detection with configurable methods
- **Backtesting** - Strategy testing from command line
- **Server** - Start REST API server

## WHY: Scripting and Automation

**Problems Solved**:
1. Quick data exploration without coding
2. Batch processing in scripts and pipelines
3. CI/CD integration for model validation

**When to Use**: Shell scripts, data exploration, quick experiments

**When NOT to Use**: Programmatic access (use library directly)

## HOW: Usage Guide

### Installation

```bash
cargo install --path crates/rustful-cli
```

### Forecasting

```bash
# Basic forecast
rustful forecast -i data.csv -s 10

# Specify model
rustful forecast -i data.csv -s 10 --model arima

# Auto model selection
rustful forecast -i data.csv -s 10 --model auto

# Output to file
rustful forecast -i data.csv -s 10 -o forecast.csv
```

### Anomaly Detection

```bash
# Z-score detection
rustful detect -i metrics.csv --method zscore --threshold 3

# IQR detection
rustful detect -i metrics.csv --method iqr --multiplier 1.5

# Output anomalies
rustful detect -i metrics.csv --method zscore -o anomalies.csv
```

### Backtesting

```bash
# SMA crossover strategy
rustful backtest -i prices.csv --strategy sma_crossover

# Custom parameters
rustful backtest -i prices.csv --strategy sma_crossover --short 20 --long 50
```

### Server

```bash
# Start API server
rustful serve --port 8080

# With host binding
rustful serve --host 0.0.0.0 --port 8080
```

### Command Reference

| Command | Description |
|---------|-------------|
| `forecast` | Generate time series forecasts |
| `detect` | Detect anomalies in data |
| `backtest` | Run trading strategy backtest |
| `serve` | Start REST API server |

### Options

```bash
rustful --help
rustful forecast --help
rustful detect --help
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| rustful-core | Uses for predictions |
| rustful-anomaly | Uses for detection |
| rustful-financial | Uses for backtesting |
| rustful-server | Starts via `serve` command |

**Integration Points**:
- CLI wrapper around all functionality
- CSV input/output for data processing

## Examples and Tests

### Building

```bash
cargo build -p rustful-cli --release
```

### Testing

```bash
cargo test -p rustful-cli
```

### Example Scripts

```bash
#!/bin/bash
# Forecast pipeline
rustful forecast -i sales.csv -s 30 --model auto -o forecast.csv

# Anomaly monitoring
rustful detect -i logs.csv --method zscore --threshold 3 -o alerts.csv
```

---

**Status**: Beta
**Roadmap**: See [framework-backlog.md](../../../docs/framework-backlog.md)
