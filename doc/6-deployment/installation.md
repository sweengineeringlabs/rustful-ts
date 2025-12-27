# Installation Guide

How to install and use rustful-ts.

## npm Package (Recommended)

### Install

```bash
npm install rustful-ts
# or
pnpm add rustful-ts
# or
yarn add rustful-ts
```

### Usage

```typescript
import { initWasm, Arima, sharpeRatio } from 'rustful-ts';

// Initialize WASM (required once)
await initWasm();

// Use algorithms
const model = new Arima(1, 1, 1);
await model.fit([10, 12, 14, 16, 18, 20]);
const forecast = await model.predict(5);

// Financial metrics
const returns = [0.01, -0.02, 0.03, 0.01, -0.01];
const ratio = await sharpeRatio(returns, 0.02);
```

## Build from Source

### 1. Clone Repository

```bash
git clone https://github.com/sweengineeringlabs/rustful-ts.git
cd rustful-ts
```

### 2. Install Prerequisites

See [Prerequisites](prerequisites.md) for detailed instructions.

```bash
# Quick setup (Linux/macOS)
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

### 3. Build WASM

```bash
# For Node.js (benchmarks, CLI)
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg

# For browsers
wasm-pack build crates/rustful-wasm --target web --out-dir ../../ts/pkg

# For bundlers (Webpack, Vite)
wasm-pack build crates/rustful-wasm --target bundler --out-dir ../../ts/pkg
```

### 4. Install TypeScript Dependencies

```bash
cd ts
npm install
```

### 5. Verify Installation

```bash
# Run tests
npm test

# Run benchmarks
npx tsx benchmark/time-series-analytics-wasm.ts
```

## Using Individual Crates (Rust)

Add to your `Cargo.toml`:

```toml
[dependencies]
rustful-core = { git = "https://github.com/sweengineeringlabs/rustful-ts.git" }

# Or specific crates
rustful-financial = { git = "https://github.com/sweengineeringlabs/rustful-ts.git" }
rustful-anomaly = { git = "https://github.com/sweengineeringlabs/rustful-ts.git" }
```

### Rust Usage

```rust
use rustful_core::algorithm::{Arima, Predictor};

fn main() {
    let data = vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
    let mut model = Arima::new(1, 1, 1);
    model.fit(&data).unwrap();
    let forecast = model.predict(5).unwrap();
    println!("{:?}", forecast);
}
```

## Docker (Coming Soon)

```dockerfile
# Dockerfile for rustful-ts
FROM node:20-slim

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
CMD ["node", "your-app.js"]
```

## Verification

After installation, verify everything works:

```typescript
import { initWasm, isWasmReady, Arima } from 'rustful-ts';

async function verify() {
  await initWasm();
  console.log('WASM ready:', isWasmReady());  // true

  const model = new Arima(1, 0, 1);
  await model.fit([1, 2, 3, 4, 5]);
  const pred = await model.predict(3);
  console.log('Prediction:', pred);  // [6.x, 7.x, 8.x]
}

verify();
```

## Next Steps

- [API Reference](../api/README.md) - Complete API documentation
- [Algorithms](../algorithms/README.md) - Algorithm selection guide
- [Benchmarks](../../ts/benchmark/overview.md) - Performance benchmarks
