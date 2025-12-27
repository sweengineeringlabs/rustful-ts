# forecast Overview

## Audience

Data scientists who need composable preprocessing pipelines for complex forecasting workflows.

## WHAT

Pipeline infrastructure for forecasting:
- **Pipeline Steps** - Transform/inverse-transform pattern
- **Preprocessing** - Normalization, differencing, log transform
- **Decomposition** - Trend, seasonal, residual separation

## WHY

| Problem | Solution |
|---------|----------|
| Manual preprocessing/postprocessing is error-prone | Automatic inverse transforms |
| Complex pipelines hard to maintain | Composable step-based design |
| Seasonal patterns need special handling | Built-in decomposition and detection |

## HOW

```rust
use forecast::{Pipeline, NormalizeStep, DifferenceStep};
use algorithm::Arima;

let pipeline = Pipeline::new()
    .add_step(NormalizeStep::new())
    .add_step(DifferenceStep::new(1))
    .with_model(Arima::new(1, 0, 1)?);

pipeline.fit(&data)?;
let forecast = pipeline.predict(10)?;  // Auto inverse-transforms
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../doc/4-development/developer-guide.md) | Build, test, contribute |
| [Backlog](../backlog.md) | Planned features |

---

**Status**: Beta
