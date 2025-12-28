# predictor-facade Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](../../../../doc/4-development/developer-guide.md).

## Audience

Application developers who need time series prediction with a simple, unified API.

## WHAT

High-level facade for predictions:
- **All algorithms** - ARIMA, Holt-Winters, SES, KNN, Linear Regression
- **Utilities** - Metrics, preprocessing, validation
- **Single import** - One crate for all functionality

## WHY

| Problem | Solution |
|---------|----------|
| Multiple crates to import | Single facade re-exports all |
| Complex dependency management | Facade handles layering |
| Need quick start | Prelude module for common types |

## HOW

```rust
use predictor_facade::prelude::*;

let mut model = Arima::new(1, 1, 0)?;
model.fit(&data)?;
let forecast = model.predict(10)?;
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../../doc/4-development/developer-guide.md) | Build, test, contribute |

---

**Status**: Stable
