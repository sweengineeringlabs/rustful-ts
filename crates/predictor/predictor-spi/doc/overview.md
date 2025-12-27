# predictor-spi Overview

## Audience

Library developers who want to implement custom time series prediction algorithms.

## WHAT

Service Provider Interface for predictors:
- **Predictor trait** - Core interface: `fit()`, `predict()`, `is_fitted()`
- **IncrementalPredictor trait** - Extension for streaming: `update()`

## WHY

| Problem | Solution |
|---------|----------|
| Custom algorithms need standard interface | Implement Predictor trait |
| Framework needs to work with any predictor | Program to trait, not implementation |
| Streaming data needs incremental updates | IncrementalPredictor extends Predictor |

## HOW

```rust
use predictor_spi::{Predictor, Result};

pub struct MyPredictor { /* ... */ }

impl Predictor for MyPredictor {
    fn fit(&mut self, data: &[f64]) -> Result<()> { /* ... */ }
    fn predict(&self, steps: usize) -> Result<Vec<f64>> { /* ... */ }
    fn is_fitted(&self) -> bool { /* ... */ }
}
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../../doc/4-development/developer-guide.md) | Build, test, contribute |

---

**Status**: Stable
