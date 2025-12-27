# Toolchain

Complete reference of tools used in rustful-ts, their purpose, and how they integrate.

## Overview

rustful-ts uses a dual-language toolchain: Rust for core algorithms and TypeScript for the JavaScript interface.

```
┌─────────────────────────────────────────────────────────────┐
│                     Development Flow                         │
├─────────────────────────────────────────────────────────────┤
│  Rust Source  ──►  Cargo  ──►  rustc  ──►  Native Binary    │
│       │                                                      │
│       └──►  wasm-pack  ──►  wasm-bindgen  ──►  .wasm + JS   │
│                                    │                         │
│                              wasm-opt (optimize)             │
│                                    │                         │
│  TypeScript  ──►  tsc  ──►  JavaScript  ◄──┘                │
└─────────────────────────────────────────────────────────────┘
```

## Rust Toolchain

### rustup

| | |
|---|---|
| **What** | Rust toolchain manager |
| **Version** | Latest stable |
| **Install** | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |

**Why we use it**: Manages Rust versions, components, and cross-compilation targets. Required to install the `wasm32-unknown-unknown` target.

**How we use it**:
```bash
rustup target add wasm32-unknown-unknown  # Add WASM target
rustup update                              # Update toolchain
rustup component add clippy rustfmt        # Add linting/formatting
```

### rustc

| | |
|---|---|
| **What** | Rust compiler |
| **Version** | 1.75+ |
| **Install** | Via rustup |

**Why we use it**: Compiles Rust source to native binaries and WASM. Provides memory safety, zero-cost abstractions, and predictable performance.

**How we use it**: Invoked automatically by Cargo. Direct usage rare.

### Cargo

| | |
|---|---|
| **What** | Rust package manager and build system |
| **Version** | Ships with rustc |
| **Install** | Via rustup |

**Why we use it**: Manages dependencies, builds workspace, runs tests, and handles the multi-crate structure.

**How we use it**:
```bash
cargo build -p rustful-core          # Build specific crate
cargo test -p rustful-core           # Test specific crate
cargo build --release                # Optimized build
cargo doc --open                     # Generate documentation
```

**Configuration**: `Cargo.toml` at workspace root defines:
- Workspace members (8 crates)
- Shared dependencies
- Release profile (LTO enabled)

## WASM Toolchain

### wasm-pack

| | |
|---|---|
| **What** | Rust-to-WASM build tool |
| **Version** | 0.13+ |
| **Install** | `cargo install wasm-pack` |

**Why we use it**: Compiles Rust to WASM, generates JS/TS bindings, creates npm-ready packages. Single command replaces complex manual build process.

**How we use it**:
```bash
# Node.js target (benchmarks, CLI)
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg

# Browser target
wasm-pack build crates/rustful-wasm --target web --out-dir ../../ts/pkg

# Bundler target (Vite, Webpack)
wasm-pack build crates/rustful-wasm --target bundler --out-dir ../../ts/pkg
```

**Output**:
- `rustful_wasm_bg.wasm` - WASM binary
- `rustful_wasm.js` - JS wrapper
- `rustful_wasm.d.ts` - TypeScript types

### wasm-bindgen

| | |
|---|---|
| **What** | Rust/JS interop library |
| **Version** | 0.2 |
| **Install** | Cargo dependency |

**Why we use it**: Generates glue code between Rust and JavaScript. Handles type conversion (e.g., `Float64Array` ↔ `&[f64]`), error propagation, and memory management.

**How we use it**: Annotations in Rust code:
```rust
#[wasm_bindgen]
pub struct WasmArima { /* ... */ }

#[wasm_bindgen]
impl WasmArima {
    #[wasm_bindgen(constructor)]
    pub fn new(p: usize, d: usize, q: usize) -> Result<WasmArima, JsValue> { /* ... */ }
}
```

### wasm-opt (Binaryen)

| | |
|---|---|
| **What** | WASM optimizer |
| **Version** | 117+ |
| **Install** | `apt install binaryen` or `brew install binaryen` |

**Why we use it**: Reduces WASM binary size by 30-40%, improves runtime performance. Runs automatically during `wasm-pack build --release`.

**How we use it**:
```bash
# Automatic (via wasm-pack --release)
wasm-pack build crates/rustful-wasm --release --target nodejs --out-dir ../../ts/pkg

# Manual optimization
wasm-opt -Oz ts/pkg/rustful_wasm_bg.wasm -o ts/pkg/rustful_wasm_bg.wasm
```

**Optimization levels**:
| Flag | Use Case |
|------|----------|
| `-O3` | Maximum speed |
| `-Os` | Balance size/speed |
| `-Oz` | Minimum size (edge/serverless) |

## TypeScript Toolchain

### Node.js

| | |
|---|---|
| **What** | JavaScript runtime |
| **Version** | 18+ |
| **Install** | `nvm install 18` or [nodejs.org](https://nodejs.org) |

**Why we use it**: Runs TypeScript tests, benchmarks, and development scripts. Required for npm package ecosystem.

**How we use it**:
```bash
node --version                              # Verify version
npx tsx benchmark/time-series-analytics.ts  # Run TypeScript directly
```

### npm

| | |
|---|---|
| **What** | Node.js package manager |
| **Version** | Ships with Node.js |
| **Install** | Via Node.js |

**Why we use it**: Manages TypeScript dependencies, runs scripts, publishes packages.

**How we use it**:
```bash
npm install           # Install dependencies
npm test              # Run tests
npm run build         # Build TypeScript + WASM
npm publish           # Publish to npm
```

### TypeScript

| | |
|---|---|
| **What** | Typed JavaScript superset |
| **Version** | 5.3+ |
| **Install** | `npm install -D typescript` |

**Why we use it**: Type safety catches errors at compile time, improves IDE support, documents API contracts.

**How we use it**:
```bash
npx tsc                    # Compile TypeScript
npx tsc --watch            # Watch mode
```

**Configuration**: `tsconfig.json` defines:
- Target: ES2020
- Module: CommonJS and ESM outputs
- Strict mode enabled

### Vitest

| | |
|---|---|
| **What** | Test runner |
| **Version** | 1.0+ |
| **Install** | `npm install -D vitest` |

**Why we use it**: Fast, Vite-native test runner with TypeScript support. Compatible with Jest API.

**How we use it**:
```bash
npm test              # Run tests once
npm run test:watch    # Watch mode
```

### ESLint

| | |
|---|---|
| **What** | JavaScript/TypeScript linter |
| **Version** | 8.55+ |
| **Install** | `npm install -D eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin` |

**Why we use it**: Enforces code style, catches common errors, maintains consistency.

**How we use it**:
```bash
npm run lint          # Check for issues
npm run lint -- --fix # Auto-fix issues
```

## Key Rust Dependencies

### nalgebra

| | |
|---|---|
| **What** | Linear algebra library |
| **Crate** | `nalgebra = "0.32"` |

**Why we use it**: Matrix operations, vector math for time series algorithms (ARIMA, regression).

**WASM note**: Must disable default features to avoid `getrandom` dependency:
```toml
nalgebra = { version = "0.32", default-features = false, features = ["std"] }
```

### serde

| | |
|---|---|
| **What** | Serialization framework |
| **Crate** | `serde = { version = "1.0", features = ["derive"] }` |

**Why we use it**: JSON serialization for REST API, configuration files, data persistence.

### axum

| | |
|---|---|
| **What** | Web framework |
| **Crate** | `axum = "0.7"` |

**Why we use it**: Powers rustful-server REST API. Async, type-safe, ergonomic.

### tokio

| | |
|---|---|
| **What** | Async runtime |
| **Crate** | `tokio = { version = "1", features = ["rt-multi-thread", "macros"] }` |

**Why we use it**: Required by axum. Provides async I/O for server and HTTP clients.

### clap

| | |
|---|---|
| **What** | CLI argument parser |
| **Crate** | `clap = { version = "4", features = ["derive"] }` |

**Why we use it**: Powers rustful-cli. Declarative argument parsing with derive macros.

## Version Matrix

| Tool | Minimum | Recommended | Notes |
|------|---------|-------------|-------|
| Rust | 1.75 | Latest stable | Edition 2021 |
| wasm-pack | 0.13 | Latest | |
| wasm-opt | 117 | Latest | Binaryen |
| Node.js | 18 | 20 LTS | |
| TypeScript | 5.3 | 5.4+ | |

## Verification

Run this script to verify your toolchain:

```bash
echo "=== Rust ===" && rustc --version && cargo --version
echo "=== WASM ===" && wasm-pack --version && wasm-opt --version
echo "=== Node ===" && node --version && npm --version
echo "=== TypeScript ===" && npx tsc --version
echo "=== Targets ===" && rustup target list --installed | grep wasm
```

Expected output (versions may differ):
```
=== Rust ===
rustc 1.75.0
cargo 1.75.0
=== WASM ===
wasm-pack 0.13.0
wasm-opt version 117
=== Node ===
v20.10.0
10.2.0
=== TypeScript ===
Version 5.3.0
=== Targets ===
wasm32-unknown-unknown
```
