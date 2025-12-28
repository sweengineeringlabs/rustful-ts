# automl Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](../../../doc/4-development/developer-guide.md).

## Audience

Data scientists and developers who need automated model selection without manual trial-and-error.

## WHAT

Automatic machine learning for time series forecasting:
- **Model Selection** - Automatically find the best algorithm for your data
- **Hyperparameter Optimization** - Grid search across parameter spaces
- **Ensembles** - Combine multiple model predictions

## WHY

| Problem | Solution |
|---------|----------|
| Manual model comparison is tedious | AutoML tests all models automatically |
| Hyperparameter tuning requires expertise | Grid search finds optimal parameters |
| Single models may underperform | Ensemble methods combine strengths |

## HOW

```rust
use automl::{AutoML, combine_predictions, EnsembleMethod};

// Select best model automatically
let automl = AutoML::with_defaults();
let result = automl.select_best_model(&data, horizon)?;
println!("Best: {}", result.best_model);

// Combine predictions from multiple models
let combined = combine_predictions(&predictions, EnsembleMethod::Average, None);
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../doc/4-development/developer-guide.md) | Build, test, contribute |
| [Backlog](../backlog.md) | Planned features |

---

**Status**: Beta
