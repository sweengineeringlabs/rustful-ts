# signal-facade Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](../../../../doc/4-development/developer-guide.md).

## Audience

Application developers who need trading signal generation with a simple, unified API.

## WHAT

High-level facade for trading signals:
- **Signals** - Buy, Sell, Hold
- **Generators** - SMA Crossover
- **Single import** - One crate for all functionality

## WHY

| Problem | Solution |
|---------|----------|
| Multiple crates to import | Single facade re-exports all |
| Complex dependency management | Facade handles layering |
| Need quick start | Prelude module for common types |

## HOW

```rust
use signal_facade::prelude::*;

let generator = SMACrossover::new(10, 20)?;
let signal = generator.generate(&prices);
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../../doc/4-development/developer-guide.md) | Build, test, contribute |

---

**Status**: Stable
