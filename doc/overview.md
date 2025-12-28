# rustful-ts Documentation

**rustful-ts** is a TypeScript time series library optimized with Rust and WebAssembly.

- **For TypeScript developers**: Familiar API with async/await, full type definitions
- **Rust under the hood**: Core algorithms compiled to WASM for 3-8x performance gains
- **Zero native dependencies**: Works in Node.js and browsers without compilation

## Quick Navigation

| Section | Audience | Description |
|---------|----------|-------------|
| [Architecture](3-design/architecture.md) | Architects, Tech Leads | System design, module structure |
| [Developer Guide](4-development/developer-guide.md) | Developers | Setup, testing, contributing |
| [Algorithms](algorithms/README.md) | Developers | Algorithm theory and selection |
| [API Reference](api/README.md) | Developers | Complete API documentation |

## Modules

### Core Layer

| Module | Purpose | Documentation |
|--------|---------|---------------|
| [algorithm](../crates/algorithm/doc/overview.md) | Core algorithms | ARIMA, Holt-Winters, SES, KNN |
| [data](../crates/data/doc/overview.md) | Data fetching | Yahoo Finance, validation |

### SEA Modules (Stratified Encapsulation Architecture)

| Module | Purpose | Documentation |
|--------|---------|---------------|
| [predictor](../crates/predictor/predictor-facade/doc/overview.md) | Prediction facade | Unified prediction API |
| [detector](../crates/detector/detector-facade/doc/overview.md) | Anomaly facade | Unified detection API |
| [pipeline](../crates/pipeline/pipeline-facade/doc/overview.md) | Pipeline facade | Data transformation |
| [signal](../crates/signal/signal-facade/doc/overview.md) | Signal facade | Trading signals |

### Domain Modules

| Module | Purpose | Documentation |
|--------|---------|---------------|
| [forecast](../crates/forecast/doc/overview.md) | Pipeline infrastructure | Composable forecasting |
| [financial](../crates/financial/doc/overview.md) | Financial analytics | Portfolio, backtesting, risk |
| [automl](../crates/automl/doc/overview.md) | AutoML | Model selection, ensembles |
| [anomaly](../crates/anomaly/doc/overview.md) | Anomaly detection | Detectors, monitoring |

### Interface Modules

| Module | Purpose | Documentation |
|--------|---------|---------------|
| [wasm](../crates/wasm/doc/overview.md) | WASM bindings | TypeScript/JavaScript interop |
| [server](../crates/server/doc/overview.md) | REST API | HTTP endpoints |
| [cli](../crates/cli/doc/overview.md) | CLI tool | Command-line interface |
| [tui](../crates/tui/doc/overview.md) | Terminal UI | Dashboard visualization |

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
- [Block Diagram](3-design/block-diagram.md) - Visual layer architecture
- [Workflow Diagrams](3-design/workflow-diagrams.md) - Data flow and process workflows
- [ADRs](3-design/adr/README.md) - Architecture Decision Records

## Development Documentation

- [Developer Guide](4-development/developer-guide.md) - Development hub
- [Setup Guide](4-development/guide/setup-guide.md) - Environment setup
- [Rust Optimization](4-development/guide/rust-optimization.md) - Performance optimization

## Research & Benchmarks

- [Benchmark Suite](../ts/benchmark/overview.md) - Performance benchmarks (TS vs WASM)
- [Research Paper](0-ideation/research/paper/wasm-typescript-time-series-analytics-benchmark.md) - WASM vs TypeScript performance analysis

## Backlog

- [Framework Backlog](framework-backlog.md) - Cross-cutting work items

## Templates

- [Templates README](templates/README.md) - Documentation templates
- [Crate Overview Template](templates/crate-overview-template.md)
- [Framework Doc Template](templates/framework-doc-template.md)

## Repository Governance

- [CHANGELOG](../CHANGELOG.md) - Version history
- [CODE_OF_CONDUCT](../CODE_OF_CONDUCT.md) - Community guidelines
- [CONTRIBUTING](../CONTRIBUTING.md) - Contribution process
- [SECURITY](../SECURITY.md) - Security policy
- [SUPPORT](../SUPPORT.md) - Getting help
