# algorithm Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](../../../doc/4-development/developer-guide.md).

## Audience

Developers implementing or using time series prediction algorithms.

## WHAT

Core prediction algorithms for time series:
- **ARIMA** - AutoRegressive Integrated Moving Average
- **Holt-Winters** - Triple exponential smoothing with seasonality
- **SES/Holt** - Simple and double exponential smoothing
- **KNN** - K-Nearest Neighbors for time series
- **Linear Regression** - Trend-based forecasting

## WHY

| Problem | Solution |
|---------|----------|
| Need accurate forecasting methods | Multiple algorithm implementations |
| Different data patterns require different models | Algorithm variety for different use cases |
| Performance critical for large datasets | Rust implementation for speed |

## HOW

```rust
use algorithm::Arima;
use predictor_spi::Predictor;

let mut model = Arima::new(1, 1, 1)?;
model.fit(&data)?;
let forecast = model.predict(10)?;
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../doc/4-development/developer-guide.md) | Build, test, API reference |
| [Algorithms Guide](../../../doc/algorithms/README.md) | Algorithm theory and selection |

---

**Status**: Stable
