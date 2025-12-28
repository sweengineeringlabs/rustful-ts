# wasm Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](../../../doc/4-development/developer-guide.md).

## Audience

TypeScript/JavaScript developers who need high-performance time series algorithms via WebAssembly.

## WHAT

WebAssembly bindings for rustful-ts algorithms:
- **Algorithm Wrappers** - WASM-compatible wrappers for all core algorithms
- **Type Conversions** - Automatic Float64Array ↔ Rust slice conversion
- **Error Handling** - Rust errors converted to JavaScript exceptions

## WHY

| Problem | Solution |
|---------|----------|
| JavaScript time series libraries are slow | Rust algorithms compiled to WASM (3-8x faster) |
| Native addons require compilation | WASM runs without native dependencies |
| Browser deployment needs portable format | WASM works in Node.js and browsers |

## HOW

```typescript
import { initWasm, Arima } from 'rustful-ts';

await initWasm();
const model = new Arima(1, 1, 1);
await model.fit(data);
const forecast = await model.predict(10);
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../doc/4-development/developer-guide.md) | Build, test, contribute |
| [Backlog](../backlog.md) | Planned features |

---

**Status**: Stable
