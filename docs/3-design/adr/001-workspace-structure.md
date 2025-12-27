# ADR 001: Cargo Workspace Structure

## Status

Accepted

## Context

rustful-ts needs to support multiple use cases (forecasting, financial analytics, anomaly detection) while maintaining a clean separation of concerns. The original implementation was a single crate with all functionality.

**Issues with single crate**:
- Users must compile all features even if they only need one
- No clear boundaries between components
- Difficult to maintain and test independently

## Decision

Adopt a Cargo workspace with 8 crates organized in layers:

**Foundation Layer**:
- `rustful-core` - Core algorithms (ARIMA, SES, Holt, etc.)

**Domain Layer**:
- `rustful-forecast` - Pipeline infrastructure
- `rustful-financial` - Portfolio, backtesting, risk
- `rustful-automl` - Model selection, ensembles
- `rustful-anomaly` - Detectors, monitoring

**Interface Layer**:
- `rustful-wasm` - WASM bindings
- `rustful-server` - REST API
- `rustful-cli` - Command-line tool

## Consequences

**Positive**:
- Clear separation of concerns
- Independent compilation and testing
- Users can depend on specific crates
- Easier to maintain

**Negative**:
- More complex project structure
- Need to manage inter-crate dependencies
- Potential for circular dependencies (mitigated by layer rules)

**Mitigations**:
- Strict layering: Interface -> Domain -> Foundation
- Workspace-level dependency management
- CI checks for dependency violations
