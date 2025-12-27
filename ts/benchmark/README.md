# Benchmark Package

Performance benchmarks comparing Pure TypeScript vs WASM-backed implementations.

## Files

| File | Description |
|------|-------------|
| `time-series-analytics-wasm.ts` | Benchmarks WASM-backed implementations |
| `time-series-analytics-pure.ts` | Benchmarks pure TypeScript implementations |

## Running Benchmarks

### WASM Version

```bash
# From repository root
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg

# Run benchmark
cd ts/benchmark
npx tsx time-series-analytics-wasm.ts
```

### Pure TypeScript Version

```bash
# Checkout the pure TS commit first
git checkout e73696b

# Run benchmark
cd ts/benchmark
npx tsx time-series-analytics-pure.ts
```

## Output

Both scripts output:
1. Console table with results
2. JSON for programmatic analysis

## Research Paper

See `doc/0-ideation/research/paper/wasm-vs-typescript-performance.md` for full analysis.
