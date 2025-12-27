# rustful-core Overview

## WHAT: Core Prediction Algorithms

rustful-core provides the foundational time series prediction algorithms used by all other crates in the rustful-ts workspace.

Key capabilities:
- **Statistical Methods** - ARIMA, Exponential Smoothing (SES, Holt, Holt-Winters)
- **Smoothing Methods** - Simple Moving Average, Weighted Moving Average
- **Regression Methods** - Linear Regression
- **Pattern Matching** - K-Nearest Neighbors for time series
- **Utilities** - Metrics (MAE, RMSE, MAPE), preprocessing, validation

## WHY: Performance Foundation

**Problems Solved**:
1. JavaScript time series libraries are slow for large datasets
2. No consistent interface across different prediction algorithms
3. Preprocessing and metrics scattered across implementations

**When to Use**: Building time series prediction applications requiring high performance

**When NOT to Use**: Simple averaging that doesn't need WASM overhead

## HOW: Usage Guide

### Basic Example

```rust
use rustful_core::algorithms::{Arima, Predictor};

let mut model = Arima::new(1, 1, 1)?;
model.fit(&data)?;
let forecast = model.predict(10)?;
```

### Predictor Trait

All algorithms implement the `Predictor` trait:

```rust
pub trait Predictor {
    fn fit(&mut self, data: &[f64]) -> Result<()>;
    fn predict(&self, steps: usize) -> Result<Vec<f64>>;
    fn is_fitted(&self) -> bool;
}
```

### Available Algorithms

| Algorithm | Struct | Best For |
|-----------|--------|----------|
| ARIMA | `Arima` | General-purpose |
| Simple Exponential Smoothing | `SES` | Flat data |
| Holt's Method | `Holt` | Data with trend |
| Holt-Winters | `HoltWinters` | Trend + seasonality |
| Simple Moving Average | `SMA` | Noise reduction |
| Linear Regression | `LinearRegression` | Linear trend |
| K-Nearest Neighbors | `TimeSeriesKNN` | Pattern matching |

### Metrics

```rust
use rustful_core::utils::metrics::{mae, rmse, mape};

let error = mae(&actual, &predicted);
let root_error = rmse(&actual, &predicted);
let percentage = mape(&actual, &predicted);
```

### Preprocessing

```rust
use rustful_core::utils::preprocessing::{normalize, standardize, difference};

let normalized = normalize(&data);
let standardized = standardize(&data);
let differenced = difference(&data, 1);
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| rustful-wasm | Exposes core via WASM bindings |
| rustful-financial | Uses for forecasting in backtests |
| rustful-automl | Compares core algorithms |
| rustful-forecast | Wraps in pipeline steps |

**Integration Points**:
- All domain modules depend on rustful-core
- Interface modules expose core functionality

## Examples and Tests

### Examples

**Location**: [`examples/`](../examples/)

- `basic.rs` - Simple ARIMA usage
- `comparison.rs` - Compare algorithms

### Tests

**Location**: [`tests/`](../tests/)

- `integration.rs` - Public API tests
- `yahoo_integration.rs` - Yahoo Finance integration tests

### Testing

```bash
cargo test -p rustful-core
cargo test -p rustful-core --features fetch  # With Yahoo Finance
```

---

**Status**: Stable
**Roadmap**: See [backlog.md](../backlog.md) | [Framework Backlog](../../../doc/framework-backlog.md)
