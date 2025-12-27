# WASM Build Guide

How to build and optimize WebAssembly packages for rustful-ts.

## Quick Build

```bash
# Node.js target (recommended for most uses)
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg
```

## Build Targets

| Target | Command | Use Case | Output |
|--------|---------|----------|--------|
| `nodejs` | `--target nodejs` | Node.js, CLI, benchmarks | CommonJS module |
| `web` | `--target web` | Direct browser use | ES module with init |
| `bundler` | `--target bundler` | Webpack, Vite, Rollup | ES module |
| `no-modules` | `--target no-modules` | Legacy browsers | Global variable |

### Node.js Target

```bash
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg
```

Output in `ts/pkg/`:
```
rustful_wasm.js          # CommonJS wrapper
rustful_wasm.d.ts        # TypeScript types
rustful_wasm_bg.wasm     # WASM binary
rustful_wasm_bg.wasm.d.ts
package.json
```

### Web Target

```bash
wasm-pack build crates/rustful-wasm --target web --out-dir ../../ts/pkg
```

Usage:
```html
<script type="module">
  import init, { Arima } from './pkg/rustful_wasm.js';

  async function run() {
    await init();
    const model = new Arima(1, 1, 1);
    // ...
  }
  run();
</script>
```

### Bundler Target

```bash
wasm-pack build crates/rustful-wasm --target bundler --out-dir ../../ts/pkg
```

Usage with Vite/Webpack:
```typescript
import { initWasm, Arima } from 'rustful-ts';
await initWasm();
```

## Build Modes

### Development Build (Default)

```bash
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg
```

- Faster compilation
- Debug symbols included
- Larger binary size

### Release Build (Optimized)

```bash
wasm-pack build crates/rustful-wasm --target nodejs --release --out-dir ../../ts/pkg
```

- Slower compilation
- Optimized for size and speed
- ~50% smaller binary

## Optimization

### wasm-opt

wasm-pack automatically runs `wasm-opt` for release builds. Ensure it's installed:

```bash
# Linux
sudo apt install binaryen

# macOS
brew install binaryen

# Verify
wasm-opt --version
```

### Manual Optimization

```bash
# Build without optimization
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg

# Manually optimize
wasm-opt -O3 ts/pkg/rustful_wasm_bg.wasm -o ts/pkg/rustful_wasm_bg.wasm
```

### Optimization Levels

| Flag | Description |
|------|-------------|
| `-O0` | No optimization |
| `-O1` | Light optimization |
| `-O2` | Standard optimization |
| `-O3` | Aggressive optimization |
| `-Os` | Optimize for size |
| `-Oz` | Aggressive size optimization |

## Binary Size

Typical sizes for rustful-wasm:

| Build | Size |
|-------|------|
| Debug | ~2.5 MB |
| Release | ~1.2 MB |
| Release + wasm-opt -Oz | ~800 KB |

### Size Reduction Tips

1. **Use release mode**: `--release`
2. **Enable LTO**: In `Cargo.toml`:
   ```toml
   [profile.release]
   lto = true
   ```
3. **Disable unused features**: Only include needed algorithms

## Common Issues

### "wasm-opt not found"

```bash
Error: failed to execute `wasm-opt`
```

Install binaryen:
```bash
sudo apt install binaryen  # Linux
brew install binaryen       # macOS
```

### "getrandom" WASM error

```
the wasm*-unknown-unknown targets are not supported by default
```

This occurs when dependencies require OS entropy. Fix in `Cargo.toml`:
```toml
nalgebra = { version = "0.32", default-features = false, features = ["std"] }
```

### Out of memory during build

```bash
# Increase Node.js memory limit
NODE_OPTIONS=--max-old-space-size=4096 wasm-pack build ...
```

### TypeScript types not generated

Ensure `#[wasm_bindgen]` annotations are correct in `crates/rustful-wasm/src/lib.rs`.

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install wasm-pack
  run: cargo install wasm-pack

- name: Build WASM
  run: wasm-pack build crates/rustful-wasm --target nodejs --release --out-dir ../../ts/pkg

- name: Test TypeScript
  run: cd ts && npm install && npm test
```

### Docker Build

```dockerfile
FROM rust:1.75-slim

RUN apt-get update && apt-get install -y binaryen
RUN rustup target add wasm32-unknown-unknown
RUN cargo install wasm-pack

WORKDIR /app
COPY . .
RUN wasm-pack build crates/rustful-wasm --target nodejs --release --out-dir ../../ts/pkg
```
