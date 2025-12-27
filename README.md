# rustful-ts

High-performance time series prediction framework. Rust core with TypeScript bindings.

## Features

- **Core Algorithms**: ARIMA, Holt-Winters, Exponential Smoothing, KNN
- **Pipeline Builder**: Composable forecasting with fluent API
- **Financial Analytics**: Portfolio management, backtesting, risk metrics
- **AutoML**: Automatic model selection and ensembles
- **Anomaly Detection**: Real-time monitoring with Z-score, IQR detectors

## Quick Start

```bash
npm install rustful-ts
```

```typescript
import { initWasm, Arima, Pipeline } from 'rustful-ts';

await initWasm();

// Basic forecasting
const model = new Arima(1, 1, 1);
await model.fit([10, 12, 14, 16, 18, 20, 22, 24, 26, 28]);
const forecast = await model.predict(5);

// Pipeline API
const results = await Pipeline.create()
  .normalize()
  .difference(1)
  .withArima(1, 0, 1)
  .fitPredict(data, 10);
```

## Documentation

See [doc/overview.md](doc/overview.md) for complete documentation.

| Document | Description |
|----------|-------------|
| [Overview](doc/overview.md) | Main documentation hub |
| [Architecture](doc/3-design/architecture.md) | System design |
| [Developer Guide](doc/4-development/developer-guide.md) | Development setup |

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
