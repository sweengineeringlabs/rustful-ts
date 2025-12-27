# TypeScript Package

Guide for using and publishing the rustful-ts npm package.

## Package Structure

```
ts/
├── src/
│   ├── index.ts              # Main exports
│   ├── wasm-loader.ts        # WASM initialization
│   ├── core/                 # Core algorithm wrappers
│   ├── algorithms/           # Algorithm implementations
│   ├── financial/            # Financial analytics
│   ├── anomaly/              # Anomaly detection
│   ├── automl/               # AutoML features
│   └── pipeline/             # Pipeline builder
├── pkg/                      # WASM output (generated)
├── tests/                    # Test files
├── benchmark/                # Performance benchmarks
└── package.json
```

## Installation

```bash
npm install rustful-ts
```

## Usage

### Initialize WASM

WASM must be initialized before using any functions:

```typescript
import { initWasm, isWasmReady } from 'rustful-ts';

// Initialize once at app startup
await initWasm();

// Check status
console.log(isWasmReady());  // true
```

### Core Algorithms

```typescript
import { Arima, SimpleExponentialSmoothing, HoltWinters } from 'rustful-ts';

// ARIMA
const arima = new Arima(1, 1, 1);
await arima.fit([10, 12, 14, 16, 18, 20]);
const forecast = await arima.predict(5);

// Simple Exponential Smoothing
const ses = new SimpleExponentialSmoothing(0.3);
await ses.fit(data);
const prediction = await ses.predict(10);

// Holt-Winters
const hw = new HoltWinters(0.3, 0.1, 0.1, 12);
await hw.fit(seasonalData);
```

### Financial Risk Metrics

```typescript
import { sharpeRatio, sortinoRatio, maxDrawdown, varHistorical } from 'rustful-ts';

const returns = [0.01, -0.02, 0.03, 0.01, -0.01];

const sharpe = await sharpeRatio(returns, 0.02);
const sortino = await sortinoRatio(returns, 0.02);
const drawdown = await maxDrawdown(equityCurve);
const var95 = await varHistorical(returns, 0.95);
```

### Anomaly Detection

```typescript
import { ZScoreDetector, IQRDetector } from 'rustful-ts';

const detector = new ZScoreDetector(3.0);
await detector.fit(trainingData);
const anomalies = await detector.detect(newData);
```

### Ensemble Methods

```typescript
import { combinePredictions, EnsembleMethod } from 'rustful-ts';

const predictions = [model1Pred, model2Pred, model3Pred];
const combined = await combinePredictions(predictions, EnsembleMethod.Average);
```

## API Reference

### Types

```typescript
interface Predictor {
  fit(data: number[]): Promise<void>;
  predict(steps: number): Promise<number[]>;
}

interface AnomalyDetector {
  fit(data: number[]): Promise<void>;
  detect(data: number[]): Promise<AnomalyResult>;
}

interface AnomalyResult {
  isAnomaly: boolean[];
  scores: number[];
}

enum EnsembleMethod {
  Average = 'average',
  Median = 'median',
  WeightedAverage = 'weighted_average'
}
```

## Building from Source

### Prerequisites

- Node.js 18+
- Rust 1.75+
- wasm-pack 0.13+

### Build Steps

```bash
# Clone
git clone https://github.com/sweengineeringlabs/rustful-ts.git
cd rustful-ts

# Build WASM
wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg

# Install dependencies
cd ts
npm install

# Build TypeScript
npm run build

# Run tests
npm test
```

## Publishing to npm

### 1. Build Release

```bash
wasm-pack build crates/rustful-wasm --target nodejs --release --out-dir ../../ts/pkg
cd ts
npm run build
```

### 2. Update Version

```bash
npm version patch  # or minor, major
```

### 3. Publish

```bash
npm publish
```

## Bundle Size

| Component | Size |
|-----------|------|
| WASM binary | ~1.2 MB |
| TypeScript | ~50 KB |
| Total | ~1.3 MB |

### Tree Shaking

Import only what you need:

```typescript
// Good - only imports ARIMA
import { Arima } from 'rustful-ts/algorithms';

// Less optimal - imports everything
import { Arima, sharpeRatio, ZScoreDetector } from 'rustful-ts';
```

## Browser Support

| Browser | Support |
|---------|---------|
| Chrome 57+ | Full |
| Firefox 52+ | Full |
| Safari 11+ | Full |
| Edge 16+ | Full |
| Node.js 18+ | Full |

## Performance

See [Benchmark Suite](../../ts/benchmark/overview.md) for detailed performance analysis.

Key findings:
- WASM provides **3-8x speedup** for compute-intensive operations
- TypeScript faster for small arrays (<100 elements)
- Crossover point around 1,000 data points
