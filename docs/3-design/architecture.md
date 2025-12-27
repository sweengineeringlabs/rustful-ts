# Architecture

**Audience**: Architects, Technical Leadership, Security Teams

## WHAT: System Architecture

rustful-ts is a multi-crate Rust workspace providing time series prediction with TypeScript bindings via WebAssembly.

**Scope**:
- Rust workspace structure and crate relationships
- TypeScript package organization
- Interface layers (WASM, REST API, CLI)
- Data flow and integration points

**Out of Scope**:
- Algorithm implementation details (see [Algorithms](../algorithms/README.md))
- API usage (see [API Reference](../api/README.md))

## WHY: Design Rationale

### Problems Addressed

1. **Performance vs Accessibility**
   - Current impact: JavaScript time series libraries are slow for large datasets
   - Consequence: Users choose between performance (native) and accessibility (JS)

2. **Fragmented Ecosystem**
   - Current impact: Users need multiple libraries for forecasting, backtesting, anomaly detection
   - Consequence: Integration complexity, inconsistent APIs

3. **Deployment Flexibility**
   - Current impact: Most libraries target single environment (browser OR server)
   - Consequence: Code duplication, maintenance burden

### Benefits

- **Unified Performance**: Rust core provides native-like speed in JavaScript
- **Comprehensive Toolkit**: One framework for forecasting, financial analytics, anomaly detection
- **Isomorphic Deployment**: Same code runs in Node.js, Bun, and browsers

## HOW: Architecture Implementation

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATIONS                                    │
│  TypeScript: import { Arima, Pipeline } from 'rustful-ts';                  │
│  CLI:        rustful forecast -i data.csv -s 10 --model auto                │
│  REST:       POST /api/v1/forecast                                          │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│   TypeScript      │  │      CLI          │  │    REST API       │
│   ts/src/         │  │  rustful-cli      │  │  rustful-server   │
│                   │  │  (clap)           │  │  (axum)           │
│ ┌───────────────┐ │  └─────────┬─────────┘  └─────────┬─────────┘
│ │ core/         │ │            │                      │
│ │ pipeline/     │ │            │                      │
│ │ financial/    │ │            └──────────┬───────────┘
│ │ automl/       │ │                       │
│ │ anomaly/      │ │                       │
│ └───────┬───────┘ │                       │
└─────────┼─────────┘                       │
          │                                 │
          ▼                                 │
┌───────────────────┐                       │
│   rustful-wasm    │                       │
│   (wasm-bindgen)  │                       │
└─────────┬─────────┘                       │
          │                                 │
          └─────────────────┬───────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOMAIN LAYER                                        │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ rustful-forecast│  │rustful-financial│  │ rustful-anomaly │              │
│  │                 │  │                 │  │                 │              │
│  │ • Pipeline      │  │ • Portfolio     │  │ • ZScoreDetector│              │
│  │ • Decomposition │  │ • Backtester    │  │ • IQRDetector   │              │
│  │ • Seasonality   │  │ • Risk Metrics  │  │ • Monitor       │              │
│  │                 │  │ • Signals       │  │ • Alerting      │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                        │
│           └────────────────────┼────────────────────┘                        │
│                                │                                             │
│                                ▼                                             │
│                    ┌─────────────────────┐                                   │
│                    │   rustful-automl    │                                   │
│                    │                     │                                   │
│                    │ • AutoSelector      │                                   │
│                    │ • GridSearch        │                                   │
│                    │ • Ensembles         │                                   │
│                    └──────────┬──────────┘                                   │
│                               │                                              │
└───────────────────────────────┼──────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FOUNDATION LAYER                                     │
│                                                                              │
│                       ┌─────────────────────┐                                │
│                       │    rustful-core     │                                │
│                       │                     │                                │
│                       │  algorithms/        │                                │
│                       │  ├── Arima          │                                │
│                       │  ├── SES            │                                │
│                       │  ├── Holt           │                                │
│                       │  ├── HoltWinters    │                                │
│                       │  ├── SMA            │                                │
│                       │  ├── LinearRegression                                │
│                       │  └── TimeSeriesKNN  │                                │
│                       │                     │                                │
│                       │  utils/             │                                │
│                       │  ├── metrics        │                                │
│                       │  ├── preprocessing  │                                │
│                       │  └── validation     │                                │
│                       │                     │                                │
│                       │  data/              │                                │
│                       │  └── yahoo (fetch)  │                                │
│                       └─────────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Workspace Structure

| Crate | Purpose | Layer |
|-------|---------|-------|
| `rustful-core` | Core algorithms | Foundation |
| `rustful-forecast` | Pipeline infrastructure | Domain |
| `rustful-financial` | Portfolio, backtesting, risk | Domain |
| `rustful-automl` | Model selection, ensembles | Domain |
| `rustful-anomaly` | Detectors, monitoring | Domain |
| `rustful-wasm` | WASM bindings | Interface |
| `rustful-server` | REST API | Interface |
| `rustful-cli` | Command-line tool | Interface |

### Dependency Graph

```
                    rustful-core
                         │
      ┌──────────────────┼──────────────────┐
      │                  │                  │
      ▼                  ▼                  ▼
rustful-forecast  rustful-financial  rustful-anomaly
      │                  │                  │
      └──────────────────┼──────────────────┘
                         │
                   rustful-automl
                         │
           ┌─────────────┼─────────────────┐
           │             │                 │
           ▼             ▼                 ▼
     rustful-wasm  rustful-server   rustful-cli
```

### Interface Layers

#### TypeScript/WASM

```typescript
// User code
import { initWasm, Arima } from 'rustful-ts';

// WASM initialization (once)
await initWasm();

// Rust algorithm via WASM
const model = new Arima(1, 1, 1);
await model.fit(data);
const forecast = await model.predict(10);
```

**Data flow**:
1. TypeScript validates input
2. Converts to Float64Array
3. Calls WASM bindings
4. Rust processes data
5. Returns Float64Array to JavaScript

#### REST API

```
POST /api/v1/forecast
Content-Type: application/json

{
  "data": [1, 2, 3, 4, 5],
  "model": "arima",
  "params": {"p": 1, "d": 1, "q": 1},
  "steps": 5
}
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/forecast` | POST | Generate forecasts |
| `/api/v1/automl/select` | POST | Auto model selection |
| `/api/v1/financial/backtest` | POST | Run backtest |
| `/api/v1/anomaly/detect` | POST | Detect anomalies |

#### CLI

```bash
# Forecasting
rustful forecast -i data.csv -s 10 --model arima

# Anomaly detection
rustful detect -i metrics.csv --method zscore --threshold 3

# Backtesting
rustful backtest -i prices.csv --strategy sma_crossover

# Start server
rustful serve --port 8080
```

### Design Patterns

#### Predictor Interface (Strategy Pattern)

All algorithms implement a common interface:

```rust
pub trait Predictor {
    fn fit(&mut self, data: &[f64]) -> Result<()>;
    fn predict(&self, steps: usize) -> Result<Vec<f64>>;
    fn is_fitted(&self) -> bool;
}
```

**When to use**: Adding new prediction algorithms.

#### Pipeline Builder (Builder Pattern)

Composable preprocessing and forecasting:

```rust
pub trait PipelineStep {
    fn transform(&self, data: &[f64]) -> Result<Vec<f64>>;
    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>>;
}
```

**When to use**: Complex forecasting workflows.

#### Anomaly Detector (Strategy Pattern)

Interchangeable detection algorithms:

```rust
pub trait AnomalyDetector {
    fn fit(&mut self, data: &[f64]) -> Result<()>;
    fn detect(&self, data: &[f64]) -> Result<AnomalyResult>;
    fn score(&self, data: &[f64]) -> Result<Vec<f64>>;
}
```

**When to use**: Adding new detection methods.

### Best Practices

**DO**:
- Add new algorithms to `rustful-core` - keeps foundation stable
- Implement `Predictor` trait for algorithms - enables composition
- Use domain crates for specialized features - maintains separation

**DON'T**:
- Add algorithm logic to interface crates - violates layering
- Bypass WASM bindings in TypeScript - breaks consistency
- Create circular dependencies between crates - causes build issues

### Decision Matrix

| Scenario | Recommended Approach | Reasoning |
|----------|---------------------|-----------|
| New algorithm | Add to `rustful-core` | Foundation layer |
| New financial feature | Add to `rustful-financial` | Domain separation |
| TypeScript-only feature | Add to `ts/src/` | No Rust dependency |
| CLI command | Add to `rustful-cli` | Interface layer |

## Summary

rustful-ts uses a layered architecture with Rust at the core, domain modules for specialized functionality, and multiple interface layers for different deployment scenarios.

**Key Takeaways**:
1. Rust workspace provides modularity and performance
2. WASM enables high-performance TypeScript
3. Trait-based design enables extensibility

---

**Related Documentation**:
- [Developer Guide](../4-development/developer-guide.md) - Development setup
- [ADRs](adr/README.md) - Architecture decisions
- [Algorithms](../algorithms/README.md) - Algorithm details

**Last Updated**: 2024-12-27
**Version**: 0.2
