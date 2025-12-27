# rustful-automl Overview

## WHAT: Automatic Model Selection

rustful-automl provides automatic model selection, hyperparameter tuning, and ensemble methods.

Key capabilities:
- **Model Selection** - Compare algorithms automatically
- **Hyperparameter Search** - Grid search, random search
- **Ensembles** - Combine multiple models
- **Cross-Validation** - Time series aware validation

## WHY: Optimal Model Discovery

**Problems Solved**:
1. Manual model selection is time-consuming
2. Hyperparameter tuning requires expertise
3. Single models may underperform ensembles

**When to Use**: Unknown optimal algorithm, hyperparameter tuning, production model selection

**When NOT to Use**: Known best algorithm, simple prototyping

## HOW: Usage Guide

### Automatic Model Selection

```rust
use rustful_automl::selector::{AutoSelector, AutoMLConfig};

let config = AutoMLConfig {
    models: vec!["arima", "ses", "holt", "linear"],
    metric: "rmse",
    cross_validation: true,
};

let mut selector = AutoSelector::new(config);
selector.fit(&data)?;

println!("Best model: {}", selector.best_model());
let forecast = selector.predict(10)?;
```

### Hyperparameter Search

```rust
use rustful_automl::search::{GridSearch, SearchSpace};

let space = SearchSpace::new()
    .add_int("p", 0, 3)
    .add_int("d", 0, 2)
    .add_int("q", 0, 3);

let best_params = GridSearch::run(&data, "arima", space)?;
```

### Ensembles

```rust
use rustful_automl::ensemble::{EnsembleForecaster, combine_predictions};

let models = vec![arima, ses, holt];
let ensemble = EnsembleForecaster::new(models);

ensemble.fit(&data)?;
let forecast = ensemble.predict(10)?;  // Averages predictions
```

### Weighting Methods

| Method | Description |
|--------|-------------|
| Simple Average | Equal weights |
| Weighted Average | Based on validation error |
| Stacking | Meta-learner combination |

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| rustful-core | Evaluates core algorithms |
| rustful-forecast | Can include pipelines |
| ts/src/automl | TypeScript mirrors this API |

**Integration Points**:
- Compares Predictor implementations
- Uses metrics from rustful-core

## Examples and Tests

### Examples

**Location**: [`examples/`](../examples/)

- `basic.rs` - Simple model selection

### Tests

**Location**: [`tests/`](../tests/)

- `integration.rs` - Selection tests

### Testing

```bash
cargo test -p rustful-automl
```

---

**Status**: Beta
**Roadmap**: See [framework-backlog.md](../../../docs/framework-backlog.md)
