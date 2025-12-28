# pipeline-facade Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](../../../../doc/4-development/developer-guide.md).

## Audience

Application developers who need data transformation pipelines with a simple, unified API.

## WHAT

High-level facade for data transformation:
- **Steps** - Normalize, Standardize, Difference
- **Pipeline** - Composable step chains
- **Single import** - One crate for all functionality

## WHY

| Problem | Solution |
|---------|----------|
| Multiple crates to import | Single facade re-exports all |
| Complex dependency management | Facade handles layering |
| Need quick start | Prelude module for common types |

## HOW

```rust
use pipeline_facade::prelude::*;

let mut pipeline = Pipeline::new();
pipeline.add_step(Box::new(NormalizeStep::new()));
let transformed = pipeline.fit_transform(&data)?;
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../../doc/4-development/developer-guide.md) | Build, test, contribute |

---

**Status**: Stable
