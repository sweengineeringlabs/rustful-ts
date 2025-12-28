# predictor-core Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](../../../../doc/4-development/developer-guide.md).

## Audience

Developers implementing or using time series prediction algorithms.

## WHAT

Core types and utilities for predictors:
- **Error types** - TsError enum with common error variants
- **Metrics** - MAE, MSE, RMSE, MAPE, R-squared
- **Preprocessing** - Normalize, standardize, difference
- **Validation** - Train-test split, cross-validation

## WHY

| Problem | Solution |
|---------|----------|
| Need consistent error handling | TsError with descriptive variants |
| Need accuracy metrics | Standard metrics module |
| Need data preprocessing | Preprocessing utilities |
| Need cross-validation | Validation utilities |

## HOW

```rust
use predictor_core::{Result, TsError};
use predictor_core::utils::metrics::rmse;
use predictor_core::utils::preprocessing::normalize;

let (normalized, min, max) = normalize(&data);
let error = rmse(&actual, &predicted);
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../../doc/4-development/developer-guide.md) | Build, test, contribute |

---

**Status**: Stable
