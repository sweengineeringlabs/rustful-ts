# Setup Guide

How to set up the development environment for contributing to rustful-ts.

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Rust | 1.70+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| wasm-pack | latest | `cargo install wasm-pack` |
| maturin | latest | `pip install maturin` |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| Python | 3.8+ | [python.org](https://python.org) |
| Bun (optional) | latest | `curl -fsSL https://bun.sh/install \| bash` |

## Verify Installation

```bash
rustc --version      # rustc 1.70.0 or higher
wasm-pack --version  # wasm-pack 0.12.0 or higher
maturin --version    # maturin 1.0.0 or higher
node --version       # v18.0.0 or higher
python --version     # Python 3.8 or higher
bun --version        # (optional) 1.0.0 or higher
```

## Clone & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/rustful-ts.git
cd rustful-ts

# Build WASM (TypeScript)
wasm-pack build --target web --out-dir ts/pkg

# Install TypeScript dependencies
cd ts
npm install
cd ..

# Build Python package (development mode)
maturin develop --features python
```

## Build Commands

### Full Build (All Targets)

```bash
# TypeScript (WASM)
wasm-pack build --target web --out-dir ts/pkg && cd ts && npm run build

# Python
maturin build --features python --release
```

### TypeScript Only

```bash
# Check for errors (fast)
cargo check --features wasm

# Build WASM
wasm-pack build --target web --out-dir ts/pkg

# Build with release optimizations
wasm-pack build --target web --out-dir ts/pkg --release

# Build TypeScript
cd ts && npm run build
```

### Python Only

```bash
# Development mode (editable install)
maturin develop --features python

# Build wheel
maturin build --features python --release

# Build and install locally
maturin develop --features python --release
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
# 1. Edit files in src/
vim src/algorithms/arima.rs

# 2. Check for errors
cargo check

# 3. Run tests
cargo test

# 4. Rebuild WASM
wasm-pack build --target web --out-dir ts/pkg

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

1. **Rust implementation**: `src/algorithms/new_algo.rs`
2. **Add to module**: Update `src/algorithms/mod.rs`
3. **WASM bindings**: Add to `src/wasm_bindings.rs`
4. **Rebuild WASM**: `wasm-pack build --target web --out-dir ts/pkg`
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
# Update wasm-pack
cargo install wasm-pack --force

# Clear cache and rebuild
rm -rf ts/pkg
wasm-pack build --target web --out-dir ts/pkg
```

### TypeScript can't find WASM module

```bash
# Ensure WASM is built first
wasm-pack build --target web --out-dir ts/pkg

# Check pkg/ directory exists
ls ts/pkg/
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
