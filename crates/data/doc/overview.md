# data Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](../../../doc/4-development/developer-guide.md).

## Audience

Developers who need to fetch and manage time series data from external sources.

## WHAT

Data fetching and management utilities:
- **Yahoo Finance** - Fetch historical stock data
- **Data Validation** - Input validation and cleaning
- **Format Conversion** - Convert between data formats

## WHY

| Problem | Solution |
|---------|----------|
| Need real-world data for testing | Yahoo Finance integration |
| Raw data often has quality issues | Built-in validation and cleaning |
| Different systems use different formats | Format conversion utilities |

## HOW

```rust
use data::yahoo::fetch_stock_data;

let prices = fetch_stock_data("AAPL", "2024-01-01", "2024-12-31").await?;
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../doc/4-development/developer-guide.md) | Build, test, API reference |

---

**Status**: Stable
