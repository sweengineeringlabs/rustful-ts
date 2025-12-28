# detector-facade Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](../../../../doc/4-development/developer-guide.md).

## Audience

Application developers who need anomaly detection with a simple, unified API.

## WHAT

High-level facade for anomaly detection:
- **Detectors** - Z-Score, IQR
- **Monitoring** - Real-time streaming detection
- **Alerting** - Alert types and severity levels
- **Single import** - One crate for all functionality

## WHY

| Problem | Solution |
|---------|----------|
| Multiple crates to import | Single facade re-exports all |
| Complex dependency management | Facade handles layering |
| Need quick start | Prelude module for common types |

## HOW

```rust
use detector_facade::prelude::*;

let mut detector = ZScoreDetector::new(3.0)?;
detector.fit(&data)?;
let result = detector.detect(&new_data)?;
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../../doc/4-development/developer-guide.md) | Build, test, contribute |

---

**Status**: Stable
