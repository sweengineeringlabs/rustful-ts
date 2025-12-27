# rustful-wasm Overview

## WHAT: WASM Bindings

rustful-wasm provides WebAssembly bindings for using rustful-ts algorithms in JavaScript and TypeScript environments.

Key capabilities:
- **Algorithm Wrappers** - WASM-compatible wrappers for all core algorithms
- **Type Conversions** - Automatic Float64Array <-> Rust slice conversion
- **Error Handling** - Rust errors converted to JavaScript exceptions

## WHY: JavaScript Performance

**Problems Solved**:
1. JavaScript time series libraries are slow
2. Native addons require compilation
3. Browser deployment needs portable format

**When to Use**: TypeScript/JavaScript applications needing high-performance forecasting

**When NOT to Use**: Pure Rust applications (use rustful-core directly)

## HOW: Usage Guide

### Building WASM

```bash
wasm-pack build --target bundler --out-dir ts/pkg
```

### Target Options

| Target | Use Case |
|--------|----------|
| `bundler` | Webpack, Rollup, Vite |
| `web` | Native browser ESM |
| `nodejs` | Node.js |

### Exposed Classes

```rust
#[wasm_bindgen]
pub struct WasmArima { /* ... */ }

#[wasm_bindgen]
impl WasmArima {
    #[wasm_bindgen(constructor)]
    pub fn new(p: usize, d: usize, q: usize) -> Result<WasmArima, JsValue>;
    pub fn fit(&mut self, data: &[f64]) -> Result<(), JsValue>;
    pub fn predict(&self, steps: usize) -> Result<Vec<f64>, JsValue>;
}
```

### Available Wrappers

| Rust Type | WASM Wrapper |
|-----------|--------------|
| `Arima` | `WasmArima` |
| `SES` | `WasmSES` |
| `Holt` | `WasmHolt` |
| `HoltWinters` | `WasmHoltWinters` |
| `SMA` | `WasmSMA` |
| `LinearRegression` | `WasmLinearRegression` |
| `TimeSeriesKNN` | `WasmKNN` |

### Utility Functions

```rust
#[wasm_bindgen]
pub fn compute_mae(actual: &[f64], predicted: &[f64]) -> f64;

#[wasm_bindgen]
pub fn normalize_data(data: &[f64]) -> Vec<f64>;
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| rustful-core | Wraps core algorithms |
| ts/src/ | TypeScript consumes WASM |

**Integration Points**:
- TypeScript imports WASM module
- wasm-loader.ts initializes WASM

## Examples and Tests

### Building

```bash
# Build for bundlers (Vite, Webpack)
wasm-pack build --target bundler --out-dir ts/pkg

# Build for Node.js
wasm-pack build --target nodejs --out-dir ts/pkg
```

### Testing

WASM is tested via TypeScript tests in `ts/tests/`.

---

**Status**: Stable
**Roadmap**: See [backlog.md](../backlog.md) | [Framework Backlog](../../../doc/framework-backlog.md)
