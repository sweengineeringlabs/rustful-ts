# Setup Guide

How to set up the development environment for contributing to rustful-ts.

## Prerequisites

| Tool | Version | Install | Required For |
|------|---------|---------|--------------|
| Rust | 1.75+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` | All crates |
| wasm32 target | - | `rustup target add wasm32-unknown-unknown` | rustful-wasm |
| wasm-pack | 0.13+ | `cargo install wasm-pack` | rustful-wasm |
| wasm-opt | latest | Via binaryen: `apt install binaryen` or `brew install binaryen` | WASM optimization |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) | TypeScript package |
| npm/pnpm | latest | Comes with Node.js | TypeScript package |

### Optional Tools

| Tool | Version | Install | Used For |
|------|---------|---------|----------|
| Bun | latest | `curl -fsSL https://bun.sh/install \| bash` | Faster TS runtime |
| tsx | latest | `npm install -g tsx` | Running TS benchmarks |

## Verify Installation

```bash
rustc --version               # rustc 1.75.0 or higher
rustup target list --installed  # should include wasm32-unknown-unknown
wasm-pack --version           # wasm-pack 0.13.0 or higher
wasm-opt --version            # binaryen version 117 or higher
node --version                # v18.0.0 or higher
```

## Clone & Setup

```bash
# Clone the repository
git clone https://github.com/sweengineeringlabs/rustful-ts.git
cd rustful-ts

# Build WASM (for Node.js)
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg

# Install TypeScript dependencies
cd ts && npm install && cd ..

# Verify WASM build
ls ts/pkg/  # Should contain .wasm, .js, .d.ts files
```

## Build Targets

### WASM Target Options

| Target | Use Case | Command |
|--------|----------|---------|
| `nodejs` | Node.js, benchmarks, CLI | `wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg` |
| `web` | Browser bundles | `wasm-pack build crates/rustful-wasm --target web --out-dir ../../ts/pkg` |
| `bundler` | Webpack/Vite | `wasm-pack build crates/rustful-wasm --target bundler --out-dir ../../ts/pkg` |

### Release Build (Optimized)

```bash
# Build with release optimizations
wasm-pack build crates/rustful-wasm --target nodejs --release --out-dir ../../ts/pkg
```

## Build Commands

### Full Build

```bash
# Build all Rust crates
cargo build --workspace

# Build WASM + TypeScript
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg
cd ts && npm run build && cd ..
```

### TypeScript/WASM Only

```bash
# Check for errors (fast)
cargo check -p rustful-wasm

# Build WASM for Node.js
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg

# Build WASM for browsers
wasm-pack build crates/rustful-wasm --target web --out-dir ../../ts/pkg

# Build TypeScript
cd ts && npm run build
```

### Individual Crates

```bash
# Build specific crate
cargo build -p rustful-core
cargo build -p rustful-financial
cargo build -p rustful-anomaly
cargo build -p rustful-automl
cargo build -p rustful-forecast

# Test specific crate
cargo test -p rustful-core
```

## Run Examples

### Stock Forecast Example (Rust)

```bash
# Fetch real AAPL data from Yahoo Finance and forecast
cargo run --example stock_forecast --features fetch
```

This example:
- Fetches historical stock data from Yahoo Finance
- Compares Linear Regression, Holt, and ARIMA models
- Shows prediction accuracy metrics (MAE, RMSE, MAPE)

### TypeScript Examples

```bash
cd ts
npx ts-node examples/stock-forecast.ts
```

## Run Tests

### Rust Tests

```bash
cargo test                    # All tests
cargo test arima              # Tests matching "arima"
cargo test -- --nocapture     # Show println! output
cargo test --features fetch   # Include data fetch tests
```

### TypeScript Tests

```bash
cd ts
npm test              # Run all tests
npm run test:watch    # Watch mode

# With Bun
bun test
```

### Python Tests

```bash
# Install test dependencies
pip install pytest pytest-benchmark

# Run tests
pytest py/tests/

# Run with verbose output
pytest py/tests/ -v

# Run specific test
pytest py/tests/test_algorithms.py::TestArima -v
```

## Development Workflow

### Modifying Rust Code

```bash
# 1. Edit files in crates/
vim crates/rustful-core/src/algorithms/arima.rs

# 2. Check for errors
cargo check -p rustful-core

# 3. Run tests
cargo test -p rustful-core

# 4. Rebuild WASM
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg

# 5. Test from TypeScript
cd ts && npm test
```

### Modifying TypeScript Code

```bash
# 1. Edit files in ts/src/
vim ts/src/algorithms/arima.ts

# 2. Build
cd ts && npm run build

# 3. Test
npm test
```

### Adding a New Algorithm

1. **Rust implementation**: `crates/rustful-core/src/algorithms/new_algo.rs`
2. **Add to module**: Update `crates/rustful-core/src/algorithms/mod.rs`
3. **WASM bindings**: Add to `crates/rustful-wasm/src/lib.rs`
4. **Rebuild WASM**: `wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg`
5. **TypeScript wrapper**: `ts/src/algorithms/new-algo.ts`
6. **Export**: Update `ts/src/index.ts`
7. **Tests**: Add tests in both Rust and TypeScript
8. **Docs**: Update `doc/algorithms/README.md` and `doc/api/README.md`

## IDE Setup

### VS Code

Recommended extensions:
- `rust-analyzer` - Rust language support
- `Even Better TOML` - Cargo.toml support
- `ESLint` - TypeScript linting

`.vscode/settings.json`:
```json
{
  "rust-analyzer.cargo.features": "all",
  "editor.formatOnSave": true,
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  }
}
```

### IntelliJ / CLion

- Install Rust plugin
- Enable "Expand macros" for better wasm-bindgen support

## Common Issues

### WASM build fails

```bash
# Ensure wasm32 target is installed
rustup target add wasm32-unknown-unknown

# Update wasm-pack
cargo install wasm-pack --force

# Clear cache and rebuild
rm -rf ts/pkg
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg
```

### TypeScript can't find WASM module

```bash
# Ensure WASM is built first
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg

# Check pkg/ directory exists
ls ts/pkg/
# Should contain: rustful_wasm.js, rustful_wasm.d.ts, rustful_wasm_bg.wasm
```

### "getrandom" compilation error

This occurs when dependencies try to use OS entropy in WASM. Fixed by using:
```toml
# In Cargo.toml
nalgebra = { version = "0.32", default-features = false, features = ["std"] }
```

### Tests fail with "WASM not initialized"

```typescript
// Always init WASM before tests
beforeAll(async () => {
  await initWasm();
});
```

## Next Steps

- [Developer Guide](./developer-guide.md) - Code style, contribution guidelines
- [Architecture](../3-design/architecture.md) - How the library is structured
