# rustful-ts

A TypeScript time series library with Rust-powered performance.

Write familiar TypeScript. Get 3-8x faster execution via WebAssembly.

```typescript
import { initWasm, Arima } from 'rustful-ts';

await initWasm();
const model = new Arima(1, 1, 1);
await model.fit(data);
const forecast = await model.predict(10);
```

## Why rustful-ts?

| Benefit | Description |
|---------|-------------|
| **TypeScript API** | Familiar async/await patterns, full type safety |
| **Rust Performance** | Algorithms compiled to WASM, 3-8x faster than pure JS |
| **No Native Dependencies** | Runs in Node.js and browsers without compilation |

## Features

- **Core Algorithms**: ARIMA, Holt-Winters, Exponential Smoothing, KNN
- **Pipeline Builder**: Composable forecasting with fluent API
- **Financial Analytics**: Portfolio management, backtesting, risk metrics
- **AutoML**: Automatic model selection and ensembles
- **Anomaly Detection**: Real-time monitoring with Z-score, IQR detectors

## Install

```bash
npm install rustful-ts
```

## Documentation

See [doc/overview.md](doc/overview.md) for complete documentation.

| Document | Description |
|----------|-------------|
| [Overview](doc/overview.md) | Main documentation hub |
| [Architecture](doc/3-design/architecture.md) | System design |
| [Developer Guide](doc/4-development/developer-guide.md) | Development setup |

## Performance

WASM-backed implementations provide **up to 8.5x speedup** for compute-intensive operations.

| Resource | Description |
|----------|-------------|
| [Benchmark Suite](ts/benchmark/overview.md) | Performance benchmarks and reproduction |
| [Research Paper](doc/0-ideation/research/paper/wasm-typescript-time-series-analytics-benchmark.md) | Academic analysis of WASM vs TypeScript performance |

## Project Structure

```
rustful-ts/
├── crates/                  # Rust workspace
│   ├── rustful-core/        # Core algorithms
│   ├── rustful-wasm/        # WASM bindings
│   ├── rustful-financial/   # Portfolio, backtesting
│   ├── rustful-automl/      # Model selection
│   ├── rustful-forecast/    # Pipeline infrastructure
│   ├── rustful-anomaly/     # Anomaly detection
│   ├── rustful-server/      # REST API
│   └── rustful-cli/         # CLI tool
└── ts/                      # TypeScript package
```

## Requirements

| Role | Requirements |
|------|--------------|
| Using | Node.js 18+ or Bun |
| Building | Rust 1.70+, wasm-pack |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
