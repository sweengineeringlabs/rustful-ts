# rustful-ts Documentation

Main documentation hub for the rustful-ts time series prediction framework.

## Quick Navigation

| Section | Audience | Description |
|---------|----------|-------------|
| [Architecture](3-design/architecture.md) | Architects, Tech Leads | System design, module structure |
| [Developer Guide](4-development/developer-guide.md) | Developers | Setup, testing, contributing |
| [Algorithms](algorithms/README.md) | Developers | Algorithm theory and selection |
| [API Reference](api/README.md) | Developers | Complete API documentation |

## Modules

### Core Modules (Priority 1)

| Module | Purpose | Documentation |
|--------|---------|---------------|
| [rustful-core](../crates/rustful-core/doc/overview.md) | Core prediction algorithms | ARIMA, Holt-Winters, SES, KNN |
| [rustful-wasm](../crates/rustful-wasm/doc/overview.md) | WASM bindings | TypeScript/JavaScript interop |

### Domain Modules (Priority 2)

| Module | Purpose | Documentation |
|--------|---------|---------------|
| [rustful-forecast](../crates/rustful-forecast/doc/overview.md) | Pipeline infrastructure | Composable forecasting |
| [rustful-financial](../crates/rustful-financial/doc/overview.md) | Financial analytics | Portfolio, backtesting, risk |
| [rustful-automl](../crates/rustful-automl/doc/overview.md) | AutoML | Model selection, ensembles |
| [rustful-anomaly](../crates/rustful-anomaly/doc/overview.md) | Anomaly detection | Detectors, monitoring |

### Interface Modules (Priority 3)

| Module | Purpose | Documentation |
|--------|---------|---------------|
| [rustful-server](../crates/rustful-server/doc/overview.md) | REST API | HTTP endpoints |
| [rustful-cli](../crates/rustful-cli/doc/overview.md) | CLI tool | Command-line interface |

### TypeScript Package

| Module | Purpose | Documentation |
|--------|---------|---------------|
| [ts/core](../ts/src/core/) | Algorithm wrappers | TypeScript API |
| [ts/pipeline](../ts/src/pipeline/) | Pipeline builder | Fluent API |
| [ts/financial](../ts/src/financial/) | Financial module | Portfolio, risk |
| [ts/automl](../ts/src/automl/) | AutoML module | Model selection |
| [ts/anomaly](../ts/src/anomaly/) | Anomaly module | Detectors |

## Design Documentation

- [Architecture](3-design/architecture.md) - System design and module relationships
- [ADRs](3-design/adr/README.md) - Architecture Decision Records

## Development Documentation

- [Developer Guide](4-development/developer-guide.md) - Development hub
- [Setup Guide](4-development/setup-guide.md) - Environment setup

## Backlog

- [Framework Backlog](framework-backlog.md) - Cross-cutting work items

## Templates

- [Templates README](templates/README.md) - Documentation templates
- [Crate Overview Template](templates/crate-overview-template.md)
- [Framework Doc Template](templates/framework-doc-template.md)
