# Performance Benchmark Suite

Performance benchmarks comparing Pure TypeScript vs WASM-backed implementations for time series analytics operations.

## Overview

This benchmark suite measures the performance difference between:
- **Pure TypeScript**: Native JavaScript/TypeScript implementations
- **WASM-backed**: Rust implementations compiled to WebAssembly

Key finding: WASM provides **up to 8.5x speedup** for compute-intensive operations on large datasets (10,000+ points), with a crossover point around 1,000 data points.

## Benchmark Scripts

| File | Description |
|------|-------------|
| `time-series-analytics-wasm.ts` | WASM benchmarks with synthetic data |
| `time-series-analytics-pure.ts` | Pure TypeScript benchmarks with synthetic data |
| `time-series-analytics-realdata.ts` | WASM benchmarks with real-world datasets |
| `time-series-analytics-realdata-pure.ts` | Pure TypeScript benchmarks with real-world datasets |

## Datasets

Real-world datasets are in the `datasets/` folder:

| Dataset | Source | Points | Characteristics |
|---------|--------|--------|-----------------|
| SPY Stock Prices | Yahoo Finance | 1,259 | Trends, volatility, fat tails |
| Melbourne Temperature | BoM | 3,650 | Strong seasonality |
| Sunspots | SIDC | 2,820 | ~11 year cycles |
| Airline Passengers | Box-Jenkins | 144 | Trend + seasonality |

See [datasets/README.md](datasets/README.md) for details.

## Running Benchmarks

### Prerequisites

```bash
# From repository root
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg
cd ts && npm install
```

### WASM with Synthetic Data

```bash
cd ts/benchmark
npx tsx time-series-analytics-wasm.ts
```

### WASM with Real-World Data

```bash
cd ts/benchmark
npx tsx time-series-analytics-realdata.ts
```

### Pure TypeScript (Historical Comparison)

```bash
# Checkout the pure TS commit
git checkout e73696b

# Run benchmark
cd ts/benchmark
npx tsx time-series-analytics-pure.ts

# Return to main branch
git checkout main
```

## Output Format

Both scripts output:
1. Console table with per-operation timings
2. JSON for programmatic analysis and comparison

## Research Paper

For comprehensive analysis, methodology, and findings, see the research paper:

**[Performance Analysis of WebAssembly vs Pure TypeScript for Time Series Analytics](../../../doc/0-ideation/research/paper/wasm-typescript-time-series-analytics-benchmark.md)**

The paper includes:
- Detailed performance data across 14 benchmarks
- Identification of the WASM/TypeScript crossover point
- Recommendations for hybrid architecture decisions
- Reproduction steps with commit references
