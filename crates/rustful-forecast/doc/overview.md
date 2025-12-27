# rustful-forecast Overview

## WHAT: Pipeline Infrastructure

rustful-forecast provides composable forecasting pipelines with preprocessing steps and model chaining.

Key capabilities:
- **Pipeline Steps** - Transform/inverse-transform pattern
- **Preprocessing** - Normalization, differencing, decomposition
- **Seasonality** - Detection and handling
- **Decomposition** - Trend, seasonal, residual separation

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Rust | 1.75+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |

**Dependencies**: rustful-core (automatically resolved via Cargo)

### Build

```bash
cargo build -p rustful-forecast
cargo test -p rustful-forecast
```

## WHY: Composable Forecasting

**Problems Solved**:
1. Manual preprocessing/postprocessing is error-prone
2. Inverse transformations often forgotten
3. Complex pipelines hard to maintain

**When to Use**: Multi-step preprocessing, seasonal decomposition, composable workflows

**When NOT to Use**: Simple single-model forecasting

## HOW: Usage Guide

### Pipeline Builder

```rust
use rustful_forecast::pipeline::{PipelineBuilder, NormalizeStep, DifferenceStep};
use rustful_core::algorithms::Arima;

let pipeline = PipelineBuilder::new()
    .add_step(NormalizeStep::new())
    .add_step(DifferenceStep::new(1))
    .with_model(Arima::new(1, 0, 1)?)
    .build();

pipeline.fit(&data)?;
let forecast = pipeline.predict(10)?;  // Auto inverse-transforms
```

### Pipeline Step Trait

```rust
pub trait PipelineStep {
    fn transform(&self, data: &[f64]) -> Result<Vec<f64>>;
    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>>;
}
```

### Available Steps

| Step | Purpose |
|------|---------|
| `NormalizeStep` | Scale to [0, 1] |
| `StandardizeStep` | Zero mean, unit variance |
| `DifferenceStep` | Remove trend via differencing |
| `LogStep` | Log transformation |

### Seasonality Detection

```rust
use rustful_forecast::seasonality::detect_period;

let period = detect_period(&data)?;
println!("Detected period: {}", period);
```

### Decomposition

```rust
use rustful_forecast::decomposition::stl_decompose;

let components = stl_decompose(&data, period)?;
// components.trend, components.seasonal, components.residual
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| rustful-core | Uses core algorithms as models |
| ts/src/pipeline | TypeScript mirrors this API |

**Integration Points**:
- PipelineBuilder wraps Predictor implementations
- TypeScript Pipeline class mirrors Rust API

## Examples and Tests

### Examples

**Location**: [`examples/`](../examples/)

- `basic.rs` - Simple pipeline usage

### Tests

**Location**: [`tests/`](../tests/)

- `integration.rs` - Pipeline tests

### Testing

```bash
cargo test -p rustful-forecast
```

---

**Status**: Beta
**Roadmap**: See [backlog.md](../backlog.md) | [Framework Backlog](../../../doc/framework-backlog.md)
